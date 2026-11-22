use crate::model::{EncoderTail, Rank};
use anyhow::{Context, Result, ensure};
use deepseek_ultrafast_cuda::{self as cu, DevPtr};
use std::{ops::Range, sync::Arc};

pub use deepseek_ultrafast_kv_transfer::TAIL_READ;
use deepseek_ultrafast_kv_transfer::{Encoding, ImageSpan, SAVED_TAIL, Segment, Snapshot};

const STAGING_BYTES: usize = 8 << 20;
const MIN_TRANSFER_BYTES: usize = 64;
const MAX_CHUNKS_PER_STEP: usize = 16;

type KvLease = (DevPtr, u64);

#[derive(Clone, Copy, Default)]
struct TransferCursor {
    segment: usize,
    offset: usize,
}

impl TransferCursor {
    fn take(&mut self, segment: Segment, budget: usize) -> Option<(usize, usize)> {
        let unit = segment.encoding.group().map_or(1, |group| group * 2);
        let bytes = (segment.decoded_bytes - self.offset).min(budget) / unit * unit;
        if bytes == 0 {
            return None;
        }
        let start = self.offset;
        self.offset += bytes;
        if self.offset == segment.decoded_bytes {
            self.segment += 1;
            self.offset = 0;
        }
        Some((start, bytes))
    }

    fn remaining(self, sizes: impl Iterator<Item = usize>) -> usize {
        sizes.skip(self.segment).enumerate()
            .map(|(i, bytes)| if i == 0 { bytes - self.offset } else { bytes })
            .sum()
    }
}

fn fp4_regions(segment: Segment, group: usize, offset: usize, bytes: usize) -> [Range<usize>; 2] {
    let data = offset / 4;
    let scales = segment.decoded_bytes / 4 + offset / (2 * group);
    [data..data + bytes / 4, scales..scales + bytes / (2 * group)]
}

pub struct ExportSlot {
    image: cu::Buffer,
    staging: cu::PinnedHost,
    stream: cu::Stream,
    source_ready: cu::Event,
    ready: cu::Event,
    device: i32,
    fp4: bool,
    valid: cu::Buffer,
}

pub struct PendingExport {
    slot: ExportSlot,
    tokens: Vec<i32>,
    images: Vec<ImageSpan>,
    fingerprint: String,
    tail_start: usize,
    encoder_tail: [usize; 3],
    segments: Vec<Segment>,
}

struct ExportRange {
    source: DevPtr,
    dest: DevPtr,
    segment: Segment,
}

/// Rank-local capture of immutable sequence state. Retain the source slot until
/// the last step returns true or until the caller aborts the capture.
pub struct ExportCapture {
    pending: PendingExport,
    ranges: Vec<ExportRange>,
    source: KvLease,
    cursor: TransferCursor,
    in_flight: Option<TransferCursor>,
}

impl ExportSlot {
    pub fn new(rk: &Rank, budget: usize) -> Result<Self> {
        Self::with_format(rk, budget, true)
    }

    pub fn with_format(rk: &Rank, budget: usize, fp4: bool) -> Result<Self> {
        let bytes = image_bytes(rk, rk.max_seq, SAVED_TAIL, fp4)?.min(budget);
        ensure!(bytes >= 4, "KV capture budget is too small");
        Ok(Self {
            image: cu::Buffer::alloc(bytes)?,
            staging: cu::PinnedHost::alloc(bytes.min(STAGING_BYTES))?,
            stream: cu::Stream::new()?,
            source_ready: cu::Event::new()?,
            ready: cu::Event::new()?,
            device: rk.dev.ordinal,
            fp4,
            valid: cu::Buffer::alloc(4)?,
        })
    }

    pub fn fits(&self, rk: &Rank, n: usize, tail: usize) -> bool {
        capture_tail_start(n, tail, rk.raw_tail_floor())
            .and_then(|start| {
                rk.check_snapshot_coverage(n, start)?;
                image_bytes(rk, n, n - start, self.fp4)
            })
            .is_ok_and(|bytes| bytes <= self.image.bytes)
    }

    pub fn capture(
        self,
        rk: &Rank,
        tokens: &[i32],
        images: &[ImageSpan],
        fingerprint: &str,
        tail: usize,
    ) -> Result<PendingExport> {
        let capture = self.begin_capture(rk, tokens, images, fingerprint, tail)?;
        let result = (|| -> Result<()> {
            let pending = &capture.pending;
            cu::memset_d32_async(pending.slot.valid.ptr, 1, 1, &rk.stream)?;
            for range in &capture.ranges {
                range.queue(rk, &rk.stream, 0, range.segment.decoded_bytes, pending.slot.valid.ptr)?;
            }
            pending.slot.ready.record(&rk.stream)
        })();
        if result.is_err() {
            rk.stream.synchronize()?;
        }
        result?;
        Ok(capture.pending)
    }

    pub fn begin_capture(
        self,
        rk: &Rank,
        tokens: &[i32],
        images: &[ImageSpan],
        fingerprint: &str,
        tail: usize,
    ) -> Result<ExportCapture> {
        ensure!(
            rk.cmp_hist != 0,
            "KV capture requires a bound sequence slot"
        );
        ensure!(
            (TAIL_READ + 1..=SAVED_TAIL).contains(&tail),
            "invalid snapshot rollback window"
        );
        ensure!(
            !tokens.is_empty() && tokens.len() <= rk.max_seq && self.fits(rk, tokens.len(), tail),
            "invalid asynchronous snapshot size"
        );
        ensure!(
            images.iter().all(|im| im.end <= tokens.len()),
            "image spans outside the snapshot"
        );
        rk.check_kv_mapping(tokens.len())?;
        let tail_start = capture_tail_start(tokens.len(), tail, rk.raw_tail_floor())?;
        rk.check_snapshot_coverage(tokens.len(), tail_start)?;
        let segs = encoded_segments(rk, tokens.len(), tail_start, false, self.fp4);
        let tail = rk.encoder_tail_state();
        let ranges = export_ranges(&segs, self.image.ptr)?;
        let pending = PendingExport {
            slot: self,
            tokens: tokens.to_vec(),
            images: images.to_vec(),
            fingerprint: fingerprint.into(),
            tail_start,
            encoder_tail: [tail.start, tail.len, tail.end],
            segments: segs.iter().map(|s| s.1).collect(),
        };
        Ok(ExportCapture {
            pending,
            ranges,
            source: rk.kv_lease(),
            cursor: TransferCursor::default(),
            in_flight: None,
        })
    }
}

impl ExportCapture {
    /// Decoded bytes whose copy completion has not yet been observed.
    pub fn remaining_bytes(&self) -> usize {
        self.cursor.remaining(self.ranges.iter().map(|range| range.segment.decoded_bytes))
    }

    /// Copy at most `decoded_byte_budget` source bytes and wait for this quantum.
    /// Resume with the same rank and bound sequence slot.
    pub fn step(&mut self, rk: &Rank, decoded_byte_budget: usize) -> Result<bool> {
        self.validate_step(rk, decoded_byte_budget)?;
        ensure!(self.in_flight.is_none(), "KV capture has an asynchronous copy in flight");
        if self.is_complete() {
            return Ok(true);
        }
        let result = (|| -> Result<TransferCursor> {
            let cursor = self.queue_quantum(rk, &rk.stream, decoded_byte_budget)?;
            self.pending.slot.ready.record(&rk.stream)?;
            self.pending.slot.ready.synchronize()?;
            Ok(cursor)
        })();
        if result.is_err() {
            let _ = rk.stream.synchronize();
        }
        self.cursor = result?;
        Ok(self.is_complete())
    }

    /// Poll the previous quantum and queue at most one new quantum. Keep the
    /// source slot mapped and immutable until this returns true or abort drains.
    pub fn step_async(&mut self, rk: &Rank, decoded_byte_budget: usize) -> Result<bool> {
        self.validate_step(rk, decoded_byte_budget)?;
        let slot = &self.pending.slot;
        if let Some(cursor) = self.in_flight {
            if !slot.ready.is_ready()? {
                return Ok(false);
            }
            self.cursor = cursor;
            self.in_flight = None;
        }
        if self.is_complete() {
            return Ok(true);
        }
        let result = (|| -> Result<TransferCursor> {
            if self.cursor.segment == 0 && self.cursor.offset == 0 {
                slot.source_ready.record(&rk.stream)?;
                slot.stream.wait(&slot.source_ready)?;
            }
            let cursor = self.queue_quantum(rk, &slot.stream, decoded_byte_budget)?;
            slot.ready.record(&slot.stream)?;
            Ok(cursor)
        })();
        if result.is_err() {
            let _ = slot.stream.synchronize();
        }
        self.in_flight = Some(result?);
        Ok(false)
    }

    fn validate_step(&self, rk: &Rank, decoded_byte_budget: usize) -> Result<()> {
        ensure!(
            decoded_byte_budget >= MIN_TRANSFER_BYTES,
            "KV copy budget must be at least {MIN_TRANSFER_BYTES} bytes"
        );
        ensure!(
            rk.dev.ordinal == self.pending.slot.device && rk.kv_lease() == self.source,
            "KV capture resumed on a different slot"
        );
        Ok(())
    }

    fn queue_quantum(
        &self,
        rk: &Rank,
        stream: &cu::Stream,
        decoded_byte_budget: usize,
    ) -> Result<TransferCursor> {
        let mut cursor = self.cursor;
        let valid = self.pending.slot.valid.ptr;
        if cursor.segment == 0 && cursor.offset == 0 {
            cu::memset_d32_async(valid, 1, 1, stream)?;
        }
        let mut remaining = decoded_byte_budget;
        for _ in 0..MAX_CHUNKS_PER_STEP {
            let Some(range) = self.ranges.get(cursor.segment) else {
                break;
            };
            let Some((offset, bytes)) = cursor.take(range.segment, remaining) else {
                break;
            };
            range.queue(rk, stream, offset, bytes, valid)?;
            remaining -= bytes;
        }
        Ok(cursor)
    }

    pub fn is_complete(&self) -> bool {
        self.cursor.segment == self.ranges.len()
    }

    pub fn finish(self) -> Result<PendingExport> {
        ensure!(self.is_complete(), "KV capture is incomplete");
        Ok(self.pending)
    }

    pub fn abort(self) -> Result<ExportSlot> {
        let slot = self.pending.slot;
        if self.in_flight.is_some() {
            let _device = cu::Device::new(slot.device)?;
            slot.ready.synchronize()?;
        }
        Ok(slot)
    }
}

impl ExportRange {
    fn queue(&self, rk: &Rank, stream: &cu::Stream, offset: usize, bytes: usize, valid: DevPtr) -> Result<()> {
        let source = self.source + offset as u64;
        if let Some(group) = self.segment.encoding.group() {
            let elements = bytes / 2;
            if bytes == self.segment.decoded_bytes {
                return crate::fp4::pack(
                    &rk.k,
                    stream,
                    self.segment.encoding,
                    source,
                    self.dest,
                    valid,
                    elements,
                );
            }
            let [data, scales] = fp4_regions(self.segment, group, offset, bytes);
            let mut args = cu::Args::new();
            args.ptr(source)
                .ptr(self.dest + data.start as u64)
                .ptr(self.dest + scales.start as u64)
                .ptr(valid)
                .push(elements);
            cu::launch(
                rk.k.get(&format!("kv_fp4_pack{group}"))?,
                &cu::Launch::new(u32::try_from(elements.div_ceil(256))?, 256),
                stream,
                &mut args,
            )
        } else {
            cu::memcpy_dtod_async(self.dest + offset as u64, source, bytes, stream)
        }
    }
}

impl PendingExport {
    pub fn finish(self) -> Result<(Snapshot, ExportSlot)> {
        let (snapshot, slot) = self.finish_reusable()?;
        Ok((snapshot?, slot))
    }

    /// A CUDA failure poisons this staging slot. Payload errors leave the slot
    /// drained and reusable, including invalid FP4 values and allocation failure.
    pub fn finish_reusable(mut self) -> Result<(Result<Snapshot>, ExportSlot)> {
        let _device = cu::Device::new(self.slot.device)?;
        self.slot.stream.wait(&self.slot.ready)?;
        let status = &mut self.slot.staging.as_mut_slice::<u32>()[..1];
        cu::memcpy_dtoh_async(status, self.slot.valid.ptr, &self.slot.stream)?;
        self.slot.stream.synchronize()?;
        let payload = (|| -> Result<(usize, Vec<u8>)> {
            ensure!(status[0] == 1, "invalid FP4 capture: nonfinite or out-of-range KV");
            let bytes = self.segments.iter().try_fold(0usize, |n, segment| {
                n.checked_add(segment.stored_bytes()?)
                    .ok_or_else(|| anyhow::anyhow!("KV export size overflow"))
            })?;
            let mut data = Vec::new();
            data.try_reserve_exact(bytes)?;
            Ok((bytes, data))
        })();
        let (bytes, mut data) = match payload {
            Ok(payload) => payload,
            Err(error) => return Ok((Err(error), self.slot)),
        };
        for offset in (0..bytes).step_by(self.slot.staging.bytes) {
            let n = (bytes - offset).min(self.slot.staging.bytes);
            let buf = &mut self.slot.staging.as_mut_slice::<u8>()[..n];
            cu::memcpy_dtoh_async(buf, self.slot.image.ptr + offset as u64, &self.slot.stream)?;
            self.slot.stream.synchronize()?;
            data.extend_from_slice(buf);
        }
        let snapshot = Snapshot::encoded(
            self.fingerprint,
            self.tokens,
            self.images,
            self.tail_start,
            self.encoder_tail,
            self.segments,
            data.into(),
        );
        let result = snapshot.validate_encoding().map(|()| snapshot);
        Ok((result, self.slot))
    }

    /// A completed capture may be discarded without copying the image to host.
    pub fn abort(self) -> Result<ExportSlot> {
        let _device = cu::Device::new(self.slot.device)?;
        self.slot.ready.synchronize()?;
        Ok(self.slot)
    }
}

impl Drop for ExportSlot {
    fn drop(&mut self) {
        if let Ok(device) = cu::Device::new(self.device) {
            let _ = device.bind();
            let _ = self.ready.synchronize();
            let _ = self.stream.synchronize();
        }
    }
}

pub fn validate_snapshot(rk: &Rank, snapshot: &Snapshot) -> Result<()> {
    let m = &snapshot.manifest;
    m.validate_geometry(rk.max_seq, rk.cfg.vocab_size, rk.cfg.window_size)?;
    ensure!(
        m.version == deepseek_ultrafast_kv_transfer::VERSION,
        "unsupported KV wire version"
    );
    let segs = encoded_segments(rk, m.tokens.len(), m.tail_start, false, true);
    ensure!(
        segs.len() == m.segments.len()
            && segs.iter().zip(&m.segments).all(|((_, expected), actual)| {
                expected.decoded_bytes == actual.decoded_bytes
                    && (actual.encoding == Encoding::Raw || actual.encoding == expected.encoding)
            }),
        "KV segment layout mismatch"
    );
    ensure!(m.bytes == snapshot.data.len(), "KV payload length mismatch");
    snapshot.validate_encoding()
}

pub struct RestoreSlot {
    host: cu::PinnedHost,
    device: cu::Buffer,
    stream: cu::Stream,
    destination_ready: cu::Event,
    ready: cu::Event,
    in_flight: Option<(i32, KvLease)>,
}

/// Retains the validated wire image while its prefix is copied to one rank's slot.
pub struct PrefixRestore {
    snapshot: Arc<Snapshot>,
    plan: PrefixPlan,
    pending_cursor: Option<TransferCursor>,
}

/// The uploaded wire image cannot be planned against this checkpoint and prefix.
#[derive(Debug)]
pub struct IncompatiblePrefix;

impl std::fmt::Display for IncompatiblePrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("incompatible KV snapshot prefix")
    }
}

impl PrefixRestore {
    pub fn remaining_bytes(&self) -> usize {
        self.plan.cursor.remaining(self.plan.ranges.iter().map(|range| range.2))
    }
}

struct PrefixPlan {
    ranges: Vec<(DevPtr, Segment, usize, usize)>,
    cursor: TransferCursor,
    tokens: usize,
    destination: KvLease,
    device: i32,
    tail: EncoderTail,
    prefix: usize,
    raw_tail_floor: usize,
}

impl PrefixPlan {
    fn new(rk: &Rank, snapshot: &Snapshot, prefix: usize) -> Result<Self> {
        ensure!(
            rk.cmp_hist != 0,
            "KV restore requires a bound sequence slot"
        );
        Self::from_wire(rk, snapshot, prefix).context(IncompatiblePrefix)
    }

    fn from_wire(rk: &Rank, snapshot: &Snapshot, prefix: usize) -> Result<Self> {
        validate_snapshot(rk, snapshot)?;
        let m = &snapshot.manifest;
        ensure!(
            prefix > 0 && m.covers_prefix(prefix),
            "KV truncation outside saved coverage"
        );
        let [start, len, end] = m.encoder_tail;
        let tail = EncoderTail { start, len, end }.truncate(prefix);
        let full = encoded_layout(rk, m.tokens.len(), m.tail_start, false, true);
        let mut clipped = encoded_layout(rk, prefix, m.tail_start, false, true);
        if tail.len == 0 {
            for (_, segment) in clipped.iter_mut().rev().take(2) {
                segment.decoded_bytes = 0;
            }
        }
        Ok(Self {
            ranges: restore_plan(&full, &clipped, &m.segments)?,
            cursor: TransferCursor::default(),
            tokens: prefix,
            destination: rk.kv_lease(),
            device: rk.dev.ordinal,
            tail,
            prefix,
            raw_tail_floor: restored_tail_floor(&clipped, prefix, m.tail_start),
        })
    }
}

impl RestoreSlot {
    pub fn new() -> Result<Self> {
        Ok(Self {
            host: cu::PinnedHost::alloc(STAGING_BYTES)?,
            device: cu::Buffer::alloc(STAGING_BYTES)?,
            stream: cu::Stream::new()?,
            destination_ready: cu::Event::new()?,
            ready: cu::Event::new()?,
            in_flight: None,
        })
    }

    pub fn import_prefix(&mut self, rk: &Rank, snapshot: &Snapshot, prefix: usize) -> Result<()> {
        let mut plan = PrefixPlan::new(rk, snapshot, prefix)?;
        while !self.copy_step(rk, snapshot, &mut plan, STAGING_BYTES)? {}
        Ok(())
    }

    /// Plan and validate an import without writing or requiring mapped GPU pages.
    pub fn begin_prefix(
        &self,
        rk: &Rank,
        snapshot: Arc<Snapshot>,
        prefix: usize,
    ) -> Result<PrefixRestore> {
        Ok(PrefixRestore {
            plan: PrefixPlan::new(rk, &snapshot, prefix)?,
            snapshot,
            pending_cursor: None,
        })
    }

    /// Restore a bounded number of decoded bytes and wait for this quantum. The
    /// committed encoder tail changes only when all ranges have been restored.
    pub fn step_prefix(
        &mut self,
        rk: &Rank,
        work: &mut PrefixRestore,
        decoded_byte_budget: usize,
    ) -> Result<bool> {
        ensure!(work.pending_cursor.is_none(), "KV restore has an asynchronous copy in flight");
        self.copy_step(rk, &work.snapshot, &mut work.plan, decoded_byte_budget)
    }

    /// Poll completed staging and queue at most one new quantum. Other prefixes
    /// may advance between calls. This rank's destination becomes readable only
    /// after the final DMA has finished and the encoder tail is installed.
    pub fn step_prefix_async(
        &mut self,
        rk: &Rank,
        work: &mut PrefixRestore,
        decoded_byte_budget: usize,
    ) -> Result<bool> {
        let plan = &mut work.plan;
        plan.validate_step(rk, decoded_byte_budget)?;
        let destination = (plan.device, plan.destination);
        if let Some(owner) = self.in_flight {
            if self.ready.is_ready()? {
                self.in_flight = None;
            } else if owner == destination {
                return Ok(false);
            }
        }
        if let Some(cursor) = work.pending_cursor.take() {
            plan.cursor = cursor;
        }
        if plan.finish_if_complete(rk) {
            return Ok(true);
        }
        if self.in_flight.is_none() {
            let cursor = self.submit(rk, &work.snapshot, plan, decoded_byte_budget, true)?;
            self.in_flight = Some(destination);
            work.pending_cursor = Some(cursor);
        }
        Ok(false)
    }

    /// Before releasing a cancelled destination, drain only a quantum that
    /// still writes into that destination. A different restore may continue.
    pub fn abort_prefix(&mut self, work: &PrefixRestore) -> Result<()> {
        if self.in_flight == Some((work.plan.device, work.plan.destination)) {
            self.ready.synchronize()?;
            self.in_flight = None;
        }
        Ok(())
    }

    fn copy_step(
        &mut self,
        rk: &Rank,
        snapshot: &Snapshot,
        plan: &mut PrefixPlan,
        budget: usize,
    ) -> Result<bool> {
        plan.validate_step(rk, budget)?;
        ensure!(self.in_flight.is_none(), "KV restore staging has an asynchronous copy in flight");
        plan.cursor = self.submit(rk, snapshot, plan, budget, false)?;
        Ok(plan.finish_if_complete(rk))
    }

    fn submit(
        &mut self,
        rk: &Rank,
        snapshot: &Snapshot,
        plan: &PrefixPlan,
        budget: usize,
        asynchronous: bool,
    ) -> Result<TransferCursor> {
        let stream = if asynchronous { &self.stream } else { &rk.stream };
        let mut cursor = plan.cursor;
        let mut remaining = budget.min(self.host.bytes);
        let mut scratch = self.device.ptr;
        let mut staging = self.host.as_mut_slice::<u8>();
        let result = (|| -> Result<TransferCursor> {
            if cursor.segment == 0 && cursor.offset == 0 {
                if asynchronous {
                    self.destination_ready.record(&rk.stream)?;
                    self.stream.wait(&self.destination_ready)?;
                }
                // Cancellation after the first copy leaves a partial destination.
                // Only the final completion can publish the restored intervals.
                rk.invalidate_restoring_slot();
            }
            for _ in 0..MAX_CHUNKS_PER_STEP {
                let Some(&(ptr, wire, clipped_bytes, image_offset)) =
                    plan.ranges.get(cursor.segment)
                else {
                    break;
                };
                let clipped = Segment {
                    decoded_bytes: clipped_bytes,
                    ..wire
                };
                let Some((offset, decoded_bytes)) = cursor.take(clipped, remaining) else {
                    break;
                };
                let source = &snapshot.data[image_offset..image_offset + wire.stored_bytes()?];
                let dest = ptr + offset as u64;
                let stored = Segment {
                    decoded_bytes,
                    ..wire
                }
                .stored_bytes()?;
                let (host, rest) = std::mem::take(&mut staging).split_at_mut(stored);
                staging = rest;
                let upload = if let Some(group) = wire.encoding.group() {
                    let [data, scales] = fp4_regions(wire, group, offset, decoded_bytes);
                    let data_bytes = data.len();
                    host[..data_bytes].copy_from_slice(&source[data]);
                    host[data_bytes..].copy_from_slice(&source[scales]);
                    scratch
                } else {
                    host.copy_from_slice(&source[offset..offset + decoded_bytes]);
                    dest
                };
                cu::memcpy_htod_async(upload, host, stream)?;
                if wire.encoding != Encoding::Raw {
                    crate::fp4::unpack(
                        &rk.k,
                        stream,
                        wire.encoding,
                        scratch,
                        dest,
                        decoded_bytes / 2,
                    )?;
                    scratch += stored as u64;
                }
                remaining -= decoded_bytes;
            }
            self.ready.record(stream)?;
            if !asynchronous {
                self.ready.synchronize()?;
            }
            Ok(cursor)
        })();
        if result.is_err() {
            let _ = stream.synchronize();
        }
        result
    }
}

impl Drop for RestoreSlot {
    fn drop(&mut self) {
        if let Some((ordinal, _)) = self.in_flight
            && let Ok(device) = cu::Device::new(ordinal)
        {
            let _ = self.ready.synchronize();
            let _ = self.stream.synchronize();
            drop(device);
        }
    }
}

impl PrefixPlan {
    fn validate_step(&self, rk: &Rank, budget: usize) -> Result<()> {
        ensure!(
            budget >= MIN_TRANSFER_BYTES,
            "KV copy budget must be at least {MIN_TRANSFER_BYTES} bytes"
        );
        ensure!(
            rk.dev.ordinal == self.device && rk.kv_lease() == self.destination,
            "KV restore resumed on a different slot"
        );
        rk.check_kv_mapping(self.tokens)
    }

    fn finish_if_complete(&self, rk: &Rank) -> bool {
        let done = self.cursor.segment == self.ranges.len();
        if done {
            rk.restore_encoder_tail_state(self.tail);
            rk.restore_window_coverage(self.prefix, self.raw_tail_floor);
        }
        done
    }
}

fn export_ranges(segments: &[(Storage, Segment)], image: DevPtr) -> Result<Vec<ExportRange>> {
    let mut ranges = Vec::with_capacity(segments.len());
    let mut offset = 0usize;
    for &(storage, segment) in segments {
        for (source, bytes) in storage.spans(0, segment.decoded_bytes)? {
            if bytes == 0 {
                continue;
            }
            let part = Segment {
                decoded_bytes: bytes,
                ..segment
            };
            ranges.push(ExportRange {
                source,
                dest: image + offset as u64,
                segment: part,
            });
            offset = offset
                .checked_add(part.stored_bytes()?)
                .ok_or_else(|| anyhow::anyhow!("KV export offset overflow"))?;
        }
    }
    Ok(ranges)
}

fn restore_plan(
    full: &[(Storage, Segment)],
    clipped: &[(Storage, Segment)],
    wire: &[Segment],
) -> Result<Vec<(DevPtr, Segment, usize, usize)>> {
    ensure!(full.len() == clipped.len(), "KV truncation layout mismatch");
    let mut plan = Vec::with_capacity(wire.len());
    let mut segments = wire.iter();
    let mut offset = 0usize;
    for (&(ptr, source), &(dst, target)) in full.iter().zip(clipped) {
        ensure!(
            ptr == dst && target.decoded_bytes <= source.decoded_bytes,
            "invalid KV truncation range"
        );
        if source.decoded_bytes == 0 {
            continue;
        }
        let &segment = segments
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing KV segment"))?;
        ensure!(
            segment.decoded_bytes == source.decoded_bytes,
            "KV source layout mismatch"
        );
        if let Storage::Ring { rows, row_bytes, .. } = dst {
            let capacity = rows * row_bytes;
            ensure!(segment.encoding == Encoding::Raw, "encoded KV ring segment");
            // Older senders may retain a larger rollback window. Copy the most
            // recent resident suffix of the requested prefix, in logical order.
            let skip = target.decoded_bytes.saturating_sub(capacity);
            let mut wire_offset = offset
                .checked_add(skip)
                .ok_or_else(|| anyhow::anyhow!("KV import offset overflow"))?;
            for (ptr, bytes) in dst.spans(skip, target.decoded_bytes - skip)? {
                if bytes != 0 {
                    plan.push((ptr, Segment::raw(bytes), bytes, wire_offset));
                    wire_offset = wire_offset
                        .checked_add(bytes)
                        .ok_or_else(|| anyhow::anyhow!("KV import offset overflow"))?;
                }
            }
        } else if let Storage::Linear(ptr) = dst {
            Segment {
                decoded_bytes: target.decoded_bytes,
                ..segment
            }
            .stored_bytes()?;
            if target.decoded_bytes != 0 {
                plan.push((ptr, segment, target.decoded_bytes, offset));
            }
        }
        offset = offset
            .checked_add(segment.stored_bytes()?)
            .ok_or_else(|| anyhow::anyhow!("KV import offset overflow"))?;
    }
    ensure!(segments.next().is_none(), "trailing KV segments");
    Ok(plan)
}

/// The wire omits decoder and draft windows. Only a raw ring actually copied
/// during restore can reduce the advertised encoder/history coverage.
fn restored_tail_floor(layout: &[(Storage, Segment)], prefix: usize, wire_floor: usize) -> usize {
    layout.iter().fold(wire_floor, |floor, (storage, segment)| match storage {
        Storage::Ring { rows, .. } if segment.decoded_bytes > 0 =>
            floor.max(prefix.saturating_sub(*rows)),
        _ => floor,
    })
}

fn capture_tail_start(n: usize, tail: usize, resident_floor: usize) -> Result<usize> {
    let start = n.saturating_sub(tail).max(resident_floor);
    ensure!(
        start <= n && (start == 0 || n - start >= TAIL_READ),
        "insufficient resident KV rollback history"
    );
    Ok(start)
}

pub fn image_bytes(rk: &Rank, n: usize, tail: usize, fp4: bool) -> Result<usize> {
    ensure!(n <= rk.max_seq, "invalid snapshot length");
    encoded_segments(rk, n, n.saturating_sub(tail), false, fp4)
        .iter()
        .try_fold(0usize, |bytes, (_, s)| {
            bytes
                .checked_add(s.stored_bytes()?)
                .ok_or_else(|| anyhow::anyhow!("KV image size overflow"))
        })
}

/// Raw device spans in logical snapshot order. The requested interval must fit
/// the resident window. Use `try_segments` for a caller-supplied interval.
pub fn segments(rk: &Rank, n: usize, t0: usize) -> Vec<(DevPtr, usize)> {
    try_segments(rk, n, t0).expect("invalid resident KV interval")
}

pub fn try_segments(rk: &Rank, n: usize, t0: usize) -> Result<Vec<(DevPtr, usize)>> {
    rk.check_kv_mapping(n)?;
    ensure!(t0 <= n, "invalid resident KV interval");
    rk.check_all_window_coverage(n, t0)?;
    let mut ranges = Vec::new();
    for (storage, segment) in encoded_segments(rk, n, t0, true, false) {
        ranges.extend(
            storage
                .spans(0, segment.decoded_bytes)?
                .into_iter()
                .filter(|&(_, bytes)| bytes != 0),
        );
    }
    Ok(ranges)
}

/// One logical wire segment can cross the end of a physical ring. Only raw
/// windows and raw compressor histories use this representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Storage {
    Linear(DevPtr),
    Ring {
        base: DevPtr,
        start: usize,
        rows: usize,
        row_bytes: usize,
    },
}

impl Storage {
    fn rows(base: DevPtr, first: usize, row_bytes: usize, ring_rows: usize) -> Self {
        if ring_rows == 0 {
            Self::Linear(base + (first * row_bytes) as u64)
        } else {
            Self::Ring {
                base,
                start: (first % ring_rows) * row_bytes,
                rows: ring_rows,
                row_bytes,
            }
        }
    }

    fn spans(self, offset: usize, bytes: usize) -> Result<[(DevPtr, usize); 2]> {
        match self {
            Self::Linear(base) => Ok([(base + offset as u64, bytes), (0, 0)]),
            Self::Ring {
                base,
                start,
                rows,
                row_bytes,
            } => {
                let capacity = rows * row_bytes;
                ensure!(bytes <= capacity, "KV interval exceeds the resident ring");
                let pos = (start + offset % capacity) % capacity;
                let first = bytes.min(capacity - pos);
                Ok([(base + pos as u64, first), (base, bytes - first)])
            }
        }
    }
}

fn encoded_segments(
    rk: &Rank,
    n: usize,
    t0: usize,
    include_decoder_windows: bool,
    fp4: bool,
) -> Vec<(Storage, Segment)> {
    encoded_layout(rk, n, t0, include_decoder_windows, fp4)
        .into_iter()
        .filter(|(_, s)| s.decoded_bytes > 0)
        .collect()
}

fn encoded_layout(
    rk: &Rank,
    n: usize,
    t0: usize,
    include_decoder_windows: bool,
    fp4: bool,
) -> Vec<(Storage, Segment)> {
    let c = &rk.cfg;
    let dec0 = *c.kv_source_layers.iter().max().unwrap();
    let (hd, ihd) = (c.head_dim, c.index_head_dim);
    let mut v = Vec::new();
    let mut add = |storage, len: usize, encoding: Encoding| {
        v.push((
            storage,
            Segment {
                decoded_bytes: len,
                encoding: if fp4 { encoding } else { Encoding::Raw },
            },
        ));
    };
    for (l, lc) in rk.caches.iter().chain(rk.mtp_caches.iter()).enumerate() {
        if include_decoder_windows || l < dec0 {
            add(
                Storage::rows(lc.win, t0, hd * 2, lc.ring_rows),
                (n - t0) * hd * 2,
                Encoding::Raw,
            );
        }
        let ratio = if l < c.n_layers {
            c.compress_ratios[l]
        } else {
            0
        };
        if ratio == 0 {
            continue;
        }
        let rows = n / ratio;
        if let Some(p) = lc.comp {
            add(Storage::Linear(p), rows * hd * 2, Encoding::Fp4E4m3);
        }
        if let Some(p) = lc.idx_k {
            add(Storage::Linear(p), rows * ihd * 2, Encoding::Fp4E8m0);
        }
        if let Some((p, s)) = lc.idx_k4 {
            add(Storage::Linear(p), rows * 64, Encoding::Raw);
            add(Storage::Linear(s), rows.div_ceil(128) * 512, Encoding::Raw);
        }
        for h in [lc.kv_hist, lc.sc_hist].into_iter().flatten() {
            add(
                Storage::rows(h, t0, hd * 4, lc.hist_rows),
                (n - t0) * hd * 4,
                Encoding::Raw,
            );
        }
    }
    add(Storage::Linear(rk.cmp_hist), n * 4, Encoding::Raw);
    v.extend(
        rk.encoder_tail_segments()
            .into_iter()
            .map(|(ptr, n)| (Storage::Linear(ptr), Segment::raw(n))),
    );
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_cursor_respects_byte_budgets_and_whole_fp4_groups() {
        let mut cursor = TransferCursor::default();
        assert_eq!(cursor.take(Segment::raw(77), 64), Some((0, 64)));
        assert_eq!(cursor.take(Segment::raw(77), 64), Some((64, 13)));
        assert_eq!(cursor.segment, 1);

        let segment = Segment {
            decoded_bytes: 192,
            encoding: Encoding::Fp4E8m0,
        };
        assert_eq!(cursor.take(segment, 64 - 13), None);
        assert_eq!(cursor.take(segment, 96), Some((0, 64)));
        assert_eq!(cursor.take(segment, 128), Some((64, 128)));
        assert_eq!(cursor.segment, 2);
        assert_eq!(cursor.offset, 0);
    }

    #[test]
    fn pending_quantum_counts_as_remaining_until_committed() {
        let mut committed = TransferCursor::default();
        let mut scheduled = committed;
        let raw = Segment::raw(77);
        let fp4 = Segment {
            decoded_bytes: 192,
            encoding: Encoding::Fp4E8m0,
        };
        let sizes = || [raw.decoded_bytes, fp4.decoded_bytes].into_iter();

        assert_eq!(scheduled.take(raw, 64), Some((0, 64)));
        assert_eq!(committed.remaining(sizes()), 269);
        assert_eq!(scheduled.remaining(sizes()), 205);

        committed = scheduled;
        assert_eq!(scheduled.take(raw, 77), Some((64, 13)));
        assert_eq!(scheduled.take(fp4, 64), Some((0, 64)));
        assert_eq!(committed.remaining(sizes()), 205);
        assert_eq!(scheduled.remaining(sizes()), 128);
        assert_eq!(scheduled.take(fp4, 128), Some((64, 128)));
        assert_eq!(scheduled.remaining(sizes()), 0);
    }

    #[test]
    fn split_fp4_transfer_keeps_scales_in_the_wire_scale_plane() {
        for encoding in [Encoding::Fp4E4m3, Encoding::Fp4E8m0] {
            let group = encoding.group().unwrap();
            let bytes = 8 * group;
            let segment = Segment {
                decoded_bytes: bytes,
                encoding,
            };
            let first = fp4_regions(segment, group, 0, bytes / 2);
            let last = fp4_regions(segment, group, bytes / 2, bytes / 2);
            assert_eq!(first, [0..group, 2 * group..2 * group + 2]);
            assert_eq!(last, [group..2 * group, 2 * group + 2..2 * group + 4]);
            assert_eq!(last[1].end, segment.stored_bytes().unwrap());
        }
    }

    #[test]
    fn restore_floor_uses_only_present_raw_rings_and_their_own_row_capacity() {
        let layout = [
            (Storage::rows(1000, 100, 4, 16), Segment::raw(100 * 4)),
            (Storage::rows(2000, 100, 8, 12), Segment::raw(100 * 8)),
            (Storage::rows(3000, 100, 4, 3), Segment::raw(0)),
            (Storage::Linear(4000), Segment::raw(64)),
        ];
        assert_eq!(restored_tail_floor(&layout, 200, 100), 188);
        assert_eq!(restored_tail_floor(&layout, 200, 194), 194);
        assert_eq!(restored_tail_floor(&layout[2..], 200, 100), 100);
    }

    #[test]
    fn restored_snapshot_advertises_only_the_available_rollback_tail() {
        let n = 2000;
        assert_eq!(capture_tail_start(n, 512, 0).unwrap(), 1488);
        assert_eq!(capture_tail_start(n, 512, 1600).unwrap(), 1600);
        assert_eq!(
            capture_tail_start(n, 512, n - TAIL_READ).unwrap(),
            n - TAIL_READ
        );
        assert!(capture_tail_start(n, 512, n - TAIL_READ + 1).is_err());
        assert!(capture_tail_start(n, 512, n + 1).is_err());
        assert_eq!(capture_tail_start(16, 512, 0).unwrap(), 0);
    }

    #[test]
    fn raw_ring_wrap_keeps_wire_order_and_restores_clipped_prefixes() {
        // Logical token 14 starts in row 6 of an eight-row device ring.
        let ring = Storage::rows(1000, 14, 4, 8);
        let dense = Storage::rows(2000, 14, 4, 0);
        assert_eq!(ring.spans(0, 24).unwrap(), [(1024, 8), (1000, 16)]);
        assert_eq!(dense.spans(0, 24).unwrap(), [(2056, 24), (0, 0)]);
        assert!(ring.spans(0, 36).is_err());
        let compressed = Segment {
            decoded_bytes: 64,
            encoding: Encoding::Fp4E8m0,
        };
        let exported = export_ranges(
            &[
                (dense, compressed),
                (ring, Segment::raw(24)),
                (dense, Segment::raw(4)),
            ],
            4000,
        )
        .unwrap();
        let ranges: Vec<_> = exported
            .iter()
            .map(|r| (r.source, r.dest, r.segment))
            .collect();
        assert_eq!(
            ranges,
            [
                (2056, 4000, compressed),
                (1024, 4017, Segment::raw(8)),
                (1000, 4025, Segment::raw(16)),
                (2056, 4041, Segment::raw(4)),
            ]
        );
        let full = [(ring, Segment::raw(56)), (dense, Segment::raw(12))];
        let wire = [Segment::raw(56), Segment::raw(12)];
        for keep in 0..=14 {
            let clipped = [(ring, Segment::raw(keep * 4)), (dense, Segment::raw(4))];
            let plan = restore_plan(&full, &clipped, &wire).unwrap();
            assert_eq!(plan.last(), Some(&(2056, wire[1], 4, 56)));
            let mut row_bytes = Vec::new();
            for &(dest, part, bytes, offset) in &plan[..plan.len() - 1] {
                assert_eq!(part, Segment::raw(bytes));
                assert!(dest >= 1000 && dest + bytes as u64 <= 1032);
                row_bytes.extend((0..bytes).map(|i| (offset + i, dest + i as u64)));
            }
            let expected: Vec<_> = (keep.saturating_sub(8) * 4..keep * 4)
                .map(|i| (i, 1000 + ((14 * 4 + i) % 32) as u64))
                .collect();
            assert_eq!(row_bytes, expected);
        }
    }

    #[test]
    fn truncation_plan_preserves_source_offsets_when_groups_disappear() {
        for encoding in [Encoding::Raw, Encoding::Fp4E4m3, Encoding::Fp4E8m0] {
            let segment = Segment {
                decoded_bytes: 256,
                encoding,
            };
            let full = [
                (Storage::Linear(100), Segment::raw(10)),
                (Storage::Linear(200), segment),
                (Storage::Linear(300), Segment::raw(0)),
                (Storage::Linear(400), segment),
            ];
            let wire = [full[0].1, segment, segment];
            for kept in [0, 64, 128, 256] {
                let clipped = [
                    (Storage::Linear(100), Segment::raw(1)),
                    (
                        Storage::Linear(200),
                        Segment {
                            decoded_bytes: kept,
                            encoding,
                        },
                    ),
                    (Storage::Linear(300), Segment::raw(0)),
                    (
                        Storage::Linear(400),
                        Segment {
                            decoded_bytes: 64,
                            encoding,
                        },
                    ),
                ];
                let plan = restore_plan(&full, &clipped, &wire).unwrap();
                assert_eq!(plan[0], (100, Segment::raw(10), 1, 0));
                assert_eq!(
                    *plan.last().unwrap(),
                    (400, segment, 64, 10 + segment.stored_bytes().unwrap())
                );
                assert_eq!(plan.len(), if kept == 0 { 2 } else { 3 });
                if kept != 0 {
                    assert_eq!(plan[1], (200, segment, kept, 10));
                }
            }
        }
        let full = [(Storage::Linear(1), Segment::raw(64))];
        assert!(
            restore_plan(
                &full,
                &[(Storage::Linear(2), Segment::raw(32))],
                &[full[0].1]
            )
            .is_err()
        );
        assert!(
            restore_plan(
                &full,
                &[(Storage::Linear(1), Segment::raw(65))],
                &[full[0].1]
            )
            .is_err()
        );
        assert!(restore_plan(&full, &full, &[]).is_err());
        assert!(restore_plan(&full, &full, &[Segment::raw(1)]).is_err());
    }
}
