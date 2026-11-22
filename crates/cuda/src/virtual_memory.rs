use crate::{DevPtr, Device, Stream, check, memset_d8_async, sys};
use anyhow::{Result, ensure};
use std::{
    collections::BTreeMap,
    ptr::null_mut,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub struct PageAllocator {
    device: sys::CUdevice,
    granularity: usize,
    backing: Arc<Mutex<BackingPool<Allocation>>>,
}

impl PageAllocator {
    pub fn new(device: &Device) -> Result<Self> {
        use sys::{CUdevice_attribute::*, CUmemAllocationGranularity_flags::*};
        ensure!(
            device.attr(CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED)? != 0,
            "CUDA device {} does not support virtual memory management",
            device.ordinal
        );
        let mut granularity = 0;
        check(
            unsafe {
                sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &properties(device.dev),
                    CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "KV allocation granularity",
        )?;
        ensure!(granularity.is_power_of_two(), "invalid CUDA page size");
        Ok(Self {
            device: device.dev,
            granularity,
            backing: Arc::new(Mutex::new(BackingPool::default())),
        })
    }

    pub fn granularity(&self) -> usize {
        self.granularity
    }

    /// Enable physical backing reuse across this rank's ranges. Configure once
    /// before mapping pages. This cap includes both mapped and free backing.
    pub fn set_budget(&self, bytes: usize) -> Result<()> {
        self.backing.lock().unwrap().set_budget(bytes)
    }

    pub fn reserve(&self, bytes: usize) -> Result<VirtualRange> {
        let bytes = round_up(bytes.max(1), self.granularity)?;
        let mut base = 0;
        check(
            unsafe { sys::cuMemAddressReserve(&mut base, bytes, self.granularity, 0, 0) },
            "reserve KV virtual address range",
        )?;
        Ok(VirtualRange {
            allocator: self.clone(),
            base,
            bytes,
            mapped: 0,
            extents: Vec::new(),
        })
    }

    fn acquire(&self, bytes: usize) -> Result<Allocation> {
        self.backing.lock().unwrap().acquire(
            bytes,
            || Allocation::new(self.device, bytes),
            Allocation::release,
        )
    }

    fn recycle(&self, bytes: usize, backing: Allocation) {
        self.backing.lock().unwrap().recycle(bytes, backing);
    }
}

struct BackingPool<T> {
    limit: Option<usize>,
    allocated: usize,
    idle: usize,
    free: BTreeMap<usize, Vec<T>>,
}

impl<T> Default for BackingPool<T> {
    fn default() -> Self {
        Self {
            limit: None,
            allocated: 0,
            idle: 0,
            free: BTreeMap::new(),
        }
    }
}

impl<T> BackingPool<T> {
    fn set_budget(&mut self, bytes: usize) -> Result<()> {
        ensure!(bytes > 0, "KV backing budget must be positive");
        ensure!(
            self.limit.is_none() && self.allocated == 0,
            "configure the KV backing budget before mapping pages"
        );
        self.limit = Some(bytes);
        Ok(())
    }

    fn take_free(&mut self, bytes: usize) -> Option<T> {
        let group = self.free.get_mut(&bytes)?;
        let handle = group.pop().expect("free extent bucket is nonempty");
        if group.is_empty() {
            self.free.remove(&bytes);
        }
        self.idle -= bytes;
        Some(handle)
    }

    fn acquire(
        &mut self,
        bytes: usize,
        create: impl FnOnce() -> Result<T>,
        mut release: impl FnMut(T) -> Result<()>,
    ) -> Result<T> {
        if let Some(handle) = self.take_free(bytes) {
            return Ok(handle);
        }
        let cap = self.limit.unwrap_or(usize::MAX);
        let active = self.allocated - self.idle;
        ensure!(
            bytes <= cap - active,
            "KV backing budget exhausted ({active} active, {bytes} additional, {cap} cap)"
        );
        while self.allocated > cap - bytes {
            let (&size, _) = self
                .free
                .last_key_value()
                .expect("enough free backing to trim");
            let extent = self.take_free(size).expect("selected free extent");
            release(extent)?;
            self.allocated -= size;
        }
        let handle = create()?;
        self.allocated += bytes;
        Ok(handle)
    }

    fn recycle(&mut self, bytes: usize, handle: T) {
        if self.limit.is_some() {
            self.idle += bytes;
            self.free.entry(bytes).or_default().push(handle);
        } else {
            self.allocated -= bytes;
            drop(handle);
        }
    }
}

fn properties(device: sys::CUdevice) -> sys::CUmemAllocationProp {
    sys::CUmemAllocationProp {
        type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        location: location(device),
        win32HandleMetaData: null_mut(),
        allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
            compressionType: 0,
            gpuDirectRDMACapable: 0,
            usage: 0,
            reserved: [0; 4],
        },
    }
}

fn location(device: sys::CUdevice) -> sys::CUmemLocation {
    sys::CUmemLocation {
        type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
        id: device,
    }
}

pub fn round_up(bytes: usize, granularity: usize) -> Result<usize> {
    ensure!(granularity > 0, "KV page size must be positive");
    bytes
        .checked_next_multiple_of(granularity)
        .ok_or_else(|| anyhow::anyhow!("KV page byte count overflow"))
}

struct Allocation(Option<sys::CUmemGenericAllocationHandle>);

impl Allocation {
    fn new(device: sys::CUdevice, bytes: usize) -> Result<Self> {
        let mut handle = 0;
        check(
            unsafe { sys::cuMemCreate(&mut handle, bytes, &properties(device), 0) },
            "allocate KV backing pages",
        )?;
        Ok(Self(Some(handle)))
    }

    fn handle(&self) -> sys::CUmemGenericAllocationHandle {
        self.0.expect("live KV backing handle")
    }

    fn release(mut self) -> Result<()> {
        check(
            unsafe { sys::cuMemRelease(self.handle()) },
            "release idle KV backing",
        )?;
        self.0 = None;
        Ok(())
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if let Some(handle) = self.0 {
            unsafe { sys::cuMemRelease(handle) };
        }
    }
}

struct Extent {
    bytes: usize,
    backing: Allocation,
}

/// A stable address range with a prefix backed by device memory.
/// All copies and kernels touching it must finish before unmapping or dropping it.
pub struct VirtualRange {
    allocator: PageAllocator,
    base: DevPtr,
    bytes: usize,
    mapped: usize,
    extents: Vec<Extent>,
}

impl VirtualRange {
    pub fn address(&self) -> DevPtr {
        self.base
    }

    pub fn mapped_bytes(&self) -> usize {
        self.mapped
    }

    /// Map an additional extent and enqueue its initialization on the rank stream.
    pub fn grow(&mut self, through: usize, stream: &Stream) -> Result<()> {
        let target = round_up(through, self.allocator.granularity)?;
        ensure!(
            target <= self.bytes,
            "KV mapping exceeds virtual address range"
        );
        if target <= self.mapped {
            return Ok(());
        }
        let addr = self.base + self.mapped as u64;
        let bytes = target - self.mapped;
        let backing = self.allocator.acquire(bytes)?;
        let mapped = check(
            unsafe { sys::cuMemMap(addr, bytes, 0, backing.handle(), 0) },
            "map KV backing pages",
        );
        if let Err(error) = mapped {
            self.allocator.recycle(bytes, backing);
            return Err(error);
        }
        self.extents.push(Extent { bytes, backing });
        self.mapped = target;
        let access = sys::CUmemAccessDesc {
            location: location(self.allocator.device),
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        check(
            unsafe { sys::cuMemSetAccess(addr, bytes, &access, 1) },
            "set KV page access",
        )?;
        memset_d8_async(addr, 0, bytes, stream)
    }

    /// Undo growth to a previously observed extent boundary after GPU synchronization.
    pub fn restore(&mut self, target: usize) -> Result<()> {
        ensure!(target <= self.mapped, "KV restore exceeds mapped bytes");
        let mut boundary = self.mapped;
        for extent in self.extents.iter().rev() {
            if boundary <= target {
                break;
            }
            boundary -= extent.bytes;
        }
        ensure!(boundary == target, "KV restore would split a CUDA mapping");
        while self.mapped > target {
            let bytes = self.extents.last().expect("tracked CUDA extent").bytes;
            let start = self.mapped - bytes;
            check(
                unsafe { sys::cuMemUnmap(self.base + start as u64, bytes) },
                "unmap KV backing pages",
            )?;
            let extent = self.extents.pop().expect("tracked CUDA extent");
            self.mapped = start;
            self.allocator.recycle(extent.bytes, extent.backing);
        }
        Ok(())
    }
}

impl Drop for VirtualRange {
    fn drop(&mut self) {
        let _ = self.restore(0);
        unsafe { sys::cuMemAddressFree(self.base, self.bytes) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooled_reuse_and_eviction_share_the_physical_cap() -> Result<()> {
        let mut pool = BackingPool::<u32>::default();
        pool.set_budget(5)?;
        let release = |_| Ok(());
        let first = pool.acquire(2, || Ok(1), release)?;
        let second = pool.acquire(1, || Ok(2), release)?;
        pool.recycle(2, first);
        assert_eq!(pool.allocated, 3);
        assert_eq!(pool.idle, 2);

        let reused = pool.acquire(2, || panic!("allocate on an exact-size hit"), release)?;
        assert_eq!(reused, first);
        assert_eq!(pool.idle, 0);
        assert!(
            pool.acquire(3, || panic!("allocate over the cap"), release)
                .is_err()
        );

        pool.recycle(2, reused);
        let mut discarded = Vec::new();
        let larger = pool.acquire(
            3,
            || Ok(3),
            |handle| {
                discarded.push(handle);
                Ok(())
            },
        )?;
        assert_eq!(discarded, [first]);
        assert_eq!(pool.allocated, 4);
        pool.recycle(1, second);
        pool.recycle(3, larger);
        assert_eq!(pool.idle, pool.allocated);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device with virtual memory management"]
    fn stable_address_with_growing_and_reused_backing() -> Result<()> {
        crate::init()?;
        let device = Device::new(0)?;
        let allocator = PageAllocator::new(&device)?;
        let page = allocator.granularity();
        allocator.set_budget(2 * page)?;
        let stream = Stream::new()?;
        let mut range = allocator.reserve(2 * page)?;
        let mut other = allocator.reserve(2 * page)?;
        let address = range.address();

        range.grow(1, &stream)?;
        assert_eq!(range.mapped_bytes(), page);
        crate::memset_d8_async(address, 17, 1, &stream)?;
        range.grow(page + 1, &stream)?;
        assert_eq!(range.address(), address);
        assert_eq!(range.mapped_bytes(), 2 * page);
        let tail = address + page as u64;
        crate::memset_d8_async(tail, 23, 1, &stream)?;
        stream.synchronize()?;
        assert_eq!(crate::download::<u8>(address, 1)?, [17]);
        assert_eq!(crate::download::<u8>(tail, 1)?, [23]);
        assert!(other.grow(1, &stream).is_err());

        range.restore(page)?;
        other.grow(1, &stream)?;
        stream.synchronize()?;
        assert_eq!(crate::download::<u8>(other.address(), 1)?, [0]);
        other.restore(0)?;
        range.grow(page + 1, &stream)?;
        stream.synchronize()?;
        assert_eq!(crate::download::<u8>(address, 1)?, [17]);
        assert_eq!(crate::download::<u8>(tail, 1)?, [0]);
        range.restore(0)?;
        other.grow(2 * page, &stream)?;
        stream.synchronize()?;
        assert_eq!(crate::download::<u8>(other.address(), 1)?, [0]);
        other.restore(0)?;
        Ok(())
    }
}
