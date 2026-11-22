use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{
    DevPtr, Device, Stream,
    virtual_memory::{PageAllocator, VirtualRange, round_up},
};

const INDEX_TILE: usize = 128;

/// Shape of one contiguous cache region. Index loads use complete 128-row tiles.
#[derive(Clone, Copy, Debug)]
pub(super) struct Region {
    ratio: usize,
    tile_rows: usize,
    bytes_per_tile: usize,
    partial_tile_at_limit: bool,
    row_limit: usize,
    pack_window: bool,
}

impl Region {
    pub(super) fn rows(ratio: usize, bytes_per_row: usize) -> Self {
        Self {
            ratio,
            tile_rows: 1,
            bytes_per_tile: bytes_per_row,
            partial_tile_at_limit: true,
            row_limit: usize::MAX,
            pack_window: false,
        }
    }

    pub(super) fn history(ring_rows: usize, bytes_per_row: usize) -> Self {
        let mut region = Self::rows(1, bytes_per_row);
        if ring_rows != 0 {
            region.row_limit = ring_rows;
        }
        region
    }

    pub(super) fn short_window(ring_rows: usize, bytes_per_row: usize) -> Self {
        let mut region = Self::history(ring_rows, bytes_per_row);
        region.pack_window = ring_rows != 0;
        region
    }

    pub(super) fn index(ratio: usize, bytes_per_row: usize) -> Result<Self> {
        Ok(Self {
            ratio,
            tile_rows: INDEX_TILE,
            bytes_per_tile: product(bytes_per_row, INDEX_TILE)?,
            partial_tile_at_limit: true,
            row_limit: usize::MAX,
            pack_window: false,
        })
    }

    pub(super) fn scales(ratio: usize) -> Self {
        Self {
            ratio,
            tile_rows: INDEX_TILE,
            bytes_per_tile: 512,
            partial_tile_at_limit: false,
            row_limit: usize::MAX,
            pack_window: false,
        }
    }

    pub(super) fn capacity(self, max_seq: usize) -> Result<usize> {
        ensure!(self.ratio > 0, "cache compression ratio must be nonzero");
        let rows = (max_seq / self.ratio).min(self.row_limit);
        if self.partial_tile_at_limit {
            product(rows, self.bytes_per_tile / self.tile_rows)
        } else {
            product(rows.div_ceil(self.tile_rows), self.bytes_per_tile)
        }
    }

    fn live_bytes(self, tokens: usize, max_seq: usize) -> Result<usize> {
        let capacity = self.capacity(max_seq)?;
        let rows = (tokens / self.ratio).min(self.row_limit);
        let tiles = rows.div_ceil(self.tile_rows);
        Ok(product(tiles, self.bytes_per_tile)?.min(capacity))
    }
}

/// A bulk prefill writes every new key before any query reads its window.
/// Verification can also write past the committed end immediately before export.
/// Use dense indexing when the whole context fits in the retained row count.
pub(super) fn history_ring_rows(max_seq: usize, chunk: usize, window: usize, provisional: usize) -> usize {
    let attention = chunk.saturating_add(window.saturating_sub(1));
    let snapshot = deepseek_ultrafast_kv_transfer::SAVED_TAIL.saturating_add(provisional);
    let retained = attention.max(snapshot);
    if max_seq <= retained { 0 } else { retained }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct HistoryRows {
    pub(super) long: usize,
    pub(super) short: usize,
}

impl HistoryRows {
    pub(super) fn new(max_seq: usize, chunk: usize, window: usize, provisional: usize) -> Self {
        let short = window.saturating_add(provisional);
        Self {
            long: history_ring_rows(max_seq, chunk, window, provisional),
            short: if max_seq <= short { 0 } else { short },
        }
    }
}

pub(super) fn product(x: usize, y: usize) -> Result<usize> {
    x.checked_mul(y)
        .ok_or_else(|| anyhow::anyhow!("KV size overflow"))
}

/// Share reservation accounting and physical backing across slots on a rank.
/// Reservation credits remain independent of which prefixes are mapped.
#[derive(Clone)]
pub struct PagedKvLayout {
    allocator: PageAllocator,
    regions: Vec<Region>,
    bindings: Vec<RegionBinding>,
    max_seq: usize,
    owner: DevPtr,
    ring_rows: HistoryRows,
}

impl PagedKvLayout {
    pub(super) fn new(
        device: &Device,
        regions: Vec<Region>,
        max_seq: usize,
        owner: DevPtr,
        ring_rows: HistoryRows,
    ) -> Result<Self> {
        let allocator = PageAllocator::new(device)?;
        let (regions, bindings) = physical_regions(regions, max_seq, allocator.granularity())?;
        Ok(Self {
            allocator,
            regions,
            bindings,
            max_seq,
            owner,
            ring_rows,
        })
    }

    pub fn reservation_bytes(&self, tokens: usize) -> Result<usize> {
        reservation_bytes(
            &self.regions,
            self.allocator.granularity(),
            tokens,
            self.max_seq,
        )
    }

    pub fn granularity(&self) -> usize {
        self.allocator.granularity()
    }

    /// Set the rank's physical ceiling for mapped KV and retained free extents.
    pub fn set_backing_budget(&self, bytes: usize) -> Result<()> {
        self.allocator.set_budget(bytes)
    }

    pub(super) fn matches(&self, owner: DevPtr, max_seq: usize, ring_rows: HistoryRows) -> bool {
        self.owner == owner && self.max_seq == max_seq && self.ring_rows == ring_rows
    }

    pub(super) fn reserve(&self) -> Result<PagedKv> {
        let ranges = self
            .regions
            .iter()
            .map(|region| self.allocator.reserve(region.capacity(self.max_seq)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(PagedKv {
            layout: self.clone(),
            ranges,
            mapped_tokens: 0,
        })
    }
}

/// Each kernel tensor still has its own contiguous address interval. Small bounded
/// windows may occupy disjoint, aligned intervals in the same physical page.
#[derive(Clone, Copy, Debug)]
struct RegionBinding {
    range: usize,
    offset: usize,
}

fn physical_regions(
    logical: Vec<Region>,
    max_seq: usize,
    page: usize,
) -> Result<(Vec<Region>, Vec<RegionBinding>)> {
    let mut ranges = Vec::new();
    let mut bindings = Vec::with_capacity(logical.len());
    let mut open_page: Option<(usize, usize)> = None;
    for region in logical {
        let bytes = region.capacity(max_seq)?;
        if region.pack_window && bytes > 0 && bytes <= page / 2 {
            // Match CUDA allocation alignment even when a future row shape is smaller.
            let stride = round_up(bytes, 256)?;
            if stride <= page / 2 {
                let (range, offset) = match open_page {
                    Some((range, used)) if stride <= page - used => (range, used),
                    _ => {
                        let range = ranges.len();
                        ranges.push(Region::history(1, page));
                        (range, 0)
                    }
                };
                bindings.push(RegionBinding { range, offset });
                open_page = Some((range, offset + stride));
                continue;
            }
        }
        bindings.push(RegionBinding { range: ranges.len(), offset: 0 });
        ranges.push(region);
    }
    Ok((ranges, bindings))
}

fn reservation_bytes(
    regions: &[Region],
    granularity: usize,
    tokens: usize,
    max_seq: usize,
) -> Result<usize> {
    ensure!(tokens <= max_seq, "KV request exceeds context capacity");
    regions.iter().try_fold(0usize, |bytes, region| {
        let rounded = round_up(region.live_bytes(tokens, max_seq)?, granularity)?;
        bytes
            .checked_add(rounded)
            .ok_or_else(|| anyhow::anyhow!("KV reservation overflow"))
    })
}

pub(super) struct PagedKv {
    layout: PagedKvLayout,
    ranges: Vec<VirtualRange>,
    mapped_tokens: usize,
}

impl PagedKv {
    pub(super) fn addresses(&self) -> impl Iterator<Item = DevPtr> + '_ {
        self.layout.bindings.iter().map(|binding| {
            self.ranges[binding.range].address() + binding.offset as u64
        })
    }

    pub(super) fn mapped_tokens(&self) -> usize {
        self.mapped_tokens
    }

    pub(super) fn mapped_bytes(&self) -> usize {
        self.ranges.iter().map(VirtualRange::mapped_bytes).sum()
    }

    pub(super) fn ensure(&mut self, tokens: usize, stream: &Stream) -> Result<()> {
        ensure!(
            tokens <= self.layout.max_seq,
            "KV request exceeds context capacity"
        );
        if tokens <= self.mapped_tokens {
            return Ok(());
        }
        let mut grown = Vec::new();
        let outcome = self.grow(tokens, stream, &mut grown);
        if let Err(error) = outcome {
            // A mapped extent may already have a zero-fill queued on the stream.
            stream.synchronize()?;
            for (index, before) in grown.into_iter().rev() {
                self.ranges[index].restore(before)?;
            }
            return Err(error);
        }
        self.mapped_tokens = tokens;
        Ok(())
    }

    fn grow(
        &mut self,
        tokens: usize,
        stream: &Stream,
        grown: &mut Vec<(usize, usize)>,
    ) -> Result<()> {
        for (index, (region, range)) in self.layout.regions.iter().zip(&mut self.ranges).enumerate()
        {
            let bytes = region.live_bytes(tokens, self.layout.max_seq)?;
            let before = range.mapped_bytes();
            if bytes > before {
                grown.push((index, before));
                range.grow(bytes, stream)?;
            }
        }
        Ok(())
    }

    /// All GPU work referring to the current slot must be finished.
    pub(super) fn release(&mut self) -> Result<()> {
        self.mapped_tokens = 0;
        for range in &mut self.ranges {
            range.restore(0)?;
        }
        Ok(())
    }
}

impl Drop for PagedKv {
    fn drop(&mut self) {
        // An aborted rank command can leave work in flight. VirtualRange unmaps
        // its extents in Drop, so join device work before dropping the ranges.
        if self.ranges.iter().any(|range| range.mapped_bytes() != 0) {
            unsafe { deepseek_ultrafast_cuda::sys::cuCtxSynchronize() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_windows_share_driver_pages_without_overlapping_other_regions() {
        use std::collections::HashMap;

        let page = 2 << 20;
        let max_seq = 1_048_576;
        let row_bytes = 1024;
        let window = Region::short_window(144, row_bytes);
        let mut logical = vec![window; 23];
        logical.insert(7, Region::rows(4, 1024));
        let (ranges, bindings) = physical_regions(logical, max_seq, page).unwrap();
        assert_eq!(ranges.len(), 3);
        assert_eq!(bindings.len(), 24);
        let mut ends: HashMap<usize, usize> = HashMap::new();
        for (index, binding) in bindings.iter().enumerate() {
            if index == 7 {
                assert_eq!(binding.offset, 0);
                continue;
            }
            assert_ne!(binding.range, bindings[7].range);
            assert!(binding.offset.is_multiple_of(256));
            let end = binding.offset + window.capacity(max_seq).unwrap();
            assert!(end <= page);
            assert!(binding.offset >= ends.insert(binding.range, end).unwrap_or(0));
        }
        assert_eq!(ends.len(), 2);
        assert_eq!(reservation_bytes(&ranges, page, 0, max_seq).unwrap(), 0);
        for tokens in [1, 143, 144, 4096, max_seq] {
            let compressed = round_up(tokens / 4 * row_bytes, page).unwrap();
            assert_eq!(reservation_bytes(&ranges, page, tokens, max_seq).unwrap(), 2 * page + compressed);
        }

        // A small driver page keeps the original incremental window growth. The
        // same applies to the dense parity slot, even with a tiny context.
        for (geometry, rows, page) in [(window, max_seq, 65536), (Region::short_window(0, 1024), 100, page)] {
            let (ranges, bindings) = physical_regions(vec![geometry; 2], rows, page).unwrap();
            assert_eq!(ranges.len(), 2);
            assert_ne!(bindings[0].range, bindings[1].range);
            for tokens in [0, 1, rows] {
                let bytes = round_up(geometry.live_bytes(tokens, rows).unwrap(), page).unwrap();
                assert_eq!(reservation_bytes(&ranges, page, tokens, rows).unwrap(), 2 * bytes);
            }
        }
    }

    #[test]
    fn mapped_history_plateaus_and_dense_compressed_rows_keep_growing() {
        let max_seq = 1_048_576;
        let cap = history_ring_rows(max_seq, 1024, 128, 16);
        let history = Region::history(cap, 2048);
        let compressed = Region::rows(4, 1024);
        assert_eq!(history.live_bytes(100, max_seq).unwrap(), 100 * 2048);
        assert_eq!(history.capacity(max_seq).unwrap(), cap * 2048);
        assert_eq!(history.live_bytes(max_seq, max_seq).unwrap(), cap * 2048);
        let bytes = |n| reservation_bytes(&[history, compressed], 65536, n, max_seq).unwrap();
        assert_eq!(bytes(131072) - bytes(65536), 65536 / 4 * 1024);
        assert_eq!(history_ring_rows(4096, 1024, 128, 16), 0);
        assert_eq!(Region::history(0, 2048).capacity(4096).unwrap(), 4096 * 2048);
    }

    #[test]
    fn bulk_prefill_and_rejected_drafts_preserve_dense_history_at_wrap() {
        const WINDOW: usize = 128;
        const DRAFT_ROWS: usize = 9;
        const TAIL: usize = deepseek_ultrafast_kv_transfer::SAVED_TAIL;
        for chunk in [17, 1024, 8192] {
            let cap = history_ring_rows(1_048_576, chunk, WINDOW, 16);
            let mut ring = vec![usize::MAX; cap];
            let mut end = 0;
            while end < 2 * cap {
                let pos = end;
                end += chunk;
                for row in pos..end {
                    ring[row % cap] = row;
                }
                // The entire union of SWA reads survives a bulk QKV write.
                for row in pos.saturating_sub(WINDOW - 1)..end {
                    assert_eq!(ring[row % cap], row);
                }
            }
            // Place the accepted anchor and the uncommitted suffix across wrap.
            let anchor = end.next_multiple_of(cap) - 3;
            for row in end..anchor {
                ring[row % cap] = row;
            }
            for row in anchor..anchor + DRAFT_ROWS {
                ring[row % cap] = if row == anchor { row } else { usize::MAX };
            }
            let committed = anchor + 1;
            for row in committed - TAIL..committed {
                assert_eq!(ring[row % cap], row);
            }
            // A new branch can overwrite every rejected row and pool ratio-4 groups.
            for row in committed..committed + DRAFT_ROWS {
                ring[row % cap] = row;
                if (row + 1).is_multiple_of(4) {
                    for member in row - 3..=row {
                        assert_eq!(ring[member % cap], member);
                    }
                }
            }
        }
    }

    #[test]
    fn tensor_loads_map_complete_tiles_inside_the_descriptor() {
        let index = Region::index(4, 64).unwrap();
        let scales = Region::scales(4);
        assert_eq!(index.live_bytes(3, 516).unwrap(), 0);
        assert_eq!(index.live_bytes(4, 516).unwrap(), 128 * 64);
        assert_eq!(index.live_bytes(512, 516).unwrap(), 128 * 64);
        assert_eq!(index.live_bytes(516, 516).unwrap(), 129 * 64);
        assert_eq!(scales.live_bytes(512, 516).unwrap(), 512);
        assert_eq!(scales.live_bytes(516, 516).unwrap(), 1024);
    }

    #[test]
    fn reservation_rounds_regions_independently_and_grows_with_used_tokens() {
        let rows = Region::rows(1, 1024);
        let index = Region::index(4, 64).unwrap();
        let regions = [rows, index, Region::scales(4)];
        let page = 65536;
        assert_eq!(
            reservation_bytes(&regions, page, 1, 1_048_576).unwrap(),
            page
        );
        assert_eq!(
            reservation_bytes(&regions, page, 64, 1_048_576).unwrap(),
            3 * page
        );
        assert_eq!(
            reservation_bytes(&regions, page, 65, 1_048_576).unwrap(),
            4 * page
        );
        assert!(reservation_bytes(&regions, page, 1_048_577, 1_048_576).is_err());
        assert!(reservation_bytes(&[Region::rows(1, usize::MAX)], page, 2, 2).is_err());
    }
}
