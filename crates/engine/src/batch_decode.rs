use super::{
    AttnFParams, Bufs, IdxParams, LayerCache, MAX_M, Meta, Q_FP8, Q_NONE, Rank, TailBufs,
    TopkParams,
};
use crate::ops::{self, GemvOpts};
use crate::weights::Arena;
use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{Args, DevPtr, Launch};
use deepseek_ultrafast_model::config::Config;

#[path = "paged_kv.rs"]
mod paged_kv;
pub use paged_kv::PagedKvLayout;
use paged_kv::{HistoryRows, PagedKv, Region, product};

pub const MAX_PACKED_DECODE_WIDTH: usize = 128;
pub const MAX_PACKED_DECODE_CONTEXT: usize = 1_048_576;

#[derive(Clone, Copy, Debug)]
pub struct DecodeToken {
    pub position: usize,
    pub token: i32,
}

impl DecodeToken {
    pub(super) fn meta(self) -> Meta {
        let mut meta = Meta {
            pos: self.position as i32,
            m: 1,
            ..Meta::default()
        };
        meta.tokens[0] = self.token;
        meta
    }
}

#[derive(Clone, Copy)]
pub(super) struct PackedSegment {
    pub row: usize,
    pub position: usize,
    pub rows: usize,
    pub candidates: Option<(DevPtr, DevPtr)>,
}

/// Device ABI in packed_rows.cuh. Row-local metadata contains the live proposal tokens.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct DeviceRow {
    meta: DevPtr,
    history: DevPtr,
    sampling: DevPtr,
    layers: DevPtr,
    candidate_mask: DevPtr,
    candidate_ids: DevPtr,
    local: i32,
    position: i32,
    anchor: i32,
    _pad: i32,
}
const _: () = assert!(size_of::<DeviceRow>() == 64);
const ROWS_PER_UPLOAD: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceLayer {
    window: DevPtr,
    compressed: DevPtr,
    keys: DevPtr,
    keys_fp4: DevPtr,
    key_scales: DevPtr,
    kv_history: DevPtr,
    score_history: DevPtr,
    ring_rows: i32,
    history_rows: i32,
}

struct SlotState {
    caches: Vec<LayerCache>,
    mtp_caches: Vec<LayerCache>,
    token_history: DevPtr,
    tail: TailBufs,
    short_ready: bool,
    sampling: DevPtr,
    meta: DevPtr,
    candidates: DevPtr,
    candidate_ids: DevPtr,
}

/// Per-sequence storage. Kernel addresses stay stable as the backing pages grow.
pub struct DecodeSlot {
    owner: DevPtr,
    layer_table: DevPtr,
    paged: PagedKv,
    state: SlotState,
    fixed: Arena,
}

impl DecodeSlot {
    pub fn fixed_bytes(&self) -> usize {
        self.fixed.bytes
    }

    pub fn mapped_bytes(&self) -> usize {
        self.paged.mapped_bytes()
    }

    pub fn mapped_tokens(&self) -> usize {
        self.paged.mapped_tokens()
    }

    pub(super) fn check_draft_window(&self, position: usize, window: usize) -> Result<()> {
        for c in &self.state.mtp_caches {
            c.check_window(position.saturating_sub(window), position)?;
        }
        Ok(())
    }

    pub(super) fn commit_verification(&self, position: usize, end: usize) {
        for c in &self.state.caches {
            c.truncate(end);
        }
        for c in &self.state.mtp_caches {
            c.record_window(position, end);
        }
    }

    pub fn raw_tail_floor(&self) -> usize {
        super::snapshot_floor(&self.state.caches)
    }
}

pub(super) fn allocate_layer_caches(
    config: &Config,
    max_seq: usize,
    mut allocate: impl FnMut(usize) -> Result<DevPtr>,
) -> Result<(Vec<LayerCache>, Vec<LayerCache>)> {
    layer_cache_regions(config, HistoryRows::default(), |region| {
        allocate(region.capacity(max_seq)?)
    })
}

fn layer_cache_regions(
    config: &Config,
    ring_rows: HistoryRows,
    mut allocate: impl FnMut(Region) -> Result<DevPtr>,
) -> Result<(Vec<LayerCache>, Vec<LayerCache>)> {
    let mut caches = Vec::with_capacity(config.n_layers + config.n_mtp_layers);
    let head_dim = config.head_dim;
    let decoder_start = config
        .kv_source_layers
        .iter()
        .copied()
        .max()
        .ok_or_else(|| anyhow::anyhow!("missing KV source layers"))?;
    for layer in 0..config.n_layers + config.n_mtp_layers {
        let ratio = config.compress_ratios[layer];
        let source = config.is_kv_source(layer);
        let on_wire = layer < decoder_start;
        let row_bytes = product(head_dim, 2)?;
        let win = allocate(if on_wire {
            Region::history(ring_rows.long, row_bytes)
        } else {
            Region::short_window(ring_rows.short, row_bytes)
        })?;
        let (comp, idx_k, idx_k4) = if source {
            let comp = allocate(Region::rows(ratio, product(head_dim, 2)?))?;
            let index = allocate(Region::index(ratio, product(config.index_head_dim, 2)?)?)?;
            let values = allocate(Region::index(ratio, 64)?)?;
            let scales = allocate(Region::scales(ratio))?;
            (Some(comp), Some(index), Some((values, scales)))
        } else {
            (None, None, None)
        };
        let (kv_hist, sc_hist) = if source && ratio > 1 {
            (
                Some(allocate(Region::history(
                    ring_rows.long,
                    product(head_dim, 4)?,
                ))?),
                Some(allocate(Region::history(
                    ring_rows.long,
                    product(head_dim, 4)?,
                ))?),
            )
        } else {
            (None, None)
        };
        caches.push(LayerCache {
            win,
            ring_rows: if on_wire {
                ring_rows.long
            } else {
                ring_rows.short
            },
            hist_rows: ring_rows.long,
            on_wire,
            compress_ratio: ratio,
            compact_live: Default::default(),
            window_live: Default::default(),
            history_live: Default::default(),
            comp,
            idx_k,
            idx_k4,
            kv_hist,
            sc_hist,
        });
    }
    let mtp_caches = caches.split_off(config.n_layers);
    Ok((caches, mtp_caches))
}

fn slot_state(
    config: &Config,
    max_seq: usize,
    ring_rows: HistoryRows,
    mut bind: impl FnMut(Region) -> Result<DevPtr>,
    mut allocate: impl FnMut(usize) -> Result<DevPtr>,
) -> Result<SlotState> {
    let (caches, mtp_caches) = layer_cache_regions(config, ring_rows, &mut bind)?;
    Ok(SlotState {
        caches,
        mtp_caches,
        token_history: bind(Region::rows(1, 4))?,
        tail: TailBufs {
            h: allocate(product(product(config.window_size, config.dim)?, 8)?)?,
            pre: allocate(product(config.window_size, 16)?)?,
            state: Default::default(),
            raw_floor: Default::default(),
            mapped_tokens: 0,
            generation: 0,
        },
        short_ready: false,
        sampling: allocate(64)?,
        meta: allocate(std::mem::size_of::<Meta>())?,
        candidates: allocate(max_seq / 8 + 8)?,
        candidate_ids: allocate(product(config.candidate_topk_blocks, 4)?)?,
    })
}

fn swap_slot(rank: &mut Rank, state: &mut SlotState) {
    std::mem::swap(&mut rank.caches, &mut state.caches);
    std::mem::swap(&mut rank.mtp_caches, &mut state.mtp_caches);
    std::mem::swap(&mut rank.cmp_hist, &mut state.token_history);
    std::mem::swap(&mut rank.tail, &mut state.tail);
    std::mem::swap(&mut rank.short_ready, &mut state.short_ready);
    std::mem::swap(&mut rank.samp.0, &mut state.sampling);
}

struct BoundSlot<'a> {
    rank: &'a mut Rank,
    state: &'a mut SlotState,
}

impl<'a> BoundSlot<'a> {
    fn new(rank: &'a mut Rank, state: &'a mut SlotState) -> Self {
        swap_slot(rank, state);
        Self { rank, state }
    }
}

impl Drop for BoundSlot<'_> {
    fn drop(&mut self) {
        swap_slot(self.rank, self.state);
    }
}

impl Rank {
    pub fn packed_decode_capacity(&self) -> usize {
        if self.packed_rows == 0 {
            0
        } else {
            MAX_PACKED_DECODE_WIDTH.min(self.pmax)
        }
    }

    /// Select the numerical execution shape before admitting requests on this replica.
    pub fn set_packed_target_rows(&mut self, rows: usize) -> Result<()> {
        ensure!(
            (1..=self.packed_decode_capacity()).contains(&rows),
            "invalid packed target width {rows}"
        );
        ensure!(
            self.packed_graphs.is_empty() || self.packed_target_rows == rows,
            "cannot change an initialized packed execution shape"
        );
        self.packed_target_rows = rows;
        Ok(())
    }

    pub fn packed_target_graph_ready(&self, speculative: bool) -> bool {
        self.packed_graphs
            .get(&(self.packed_target_rows, speculative))
            .is_some_and(Option::is_some)
    }

    pub(super) fn packed_execution_rows(&self, live: usize) -> usize {
        if self.packed_target_rows > 8 {
            self.packed_target_rows
        } else {
            live
        }
    }

    fn history_rows(&self) -> HistoryRows {
        HistoryRows::new(self.max_seq, self.pmax, self.cfg.window_size, MAX_M)
    }

    fn paged_layout(&self, ring_rows: HistoryRows) -> Result<PagedKvLayout> {
        ensure!(
            self.max_seq <= MAX_PACKED_DECODE_CONTEXT,
            "packed decode supports at most {MAX_PACKED_DECODE_CONTEXT} context tokens"
        );
        let mut regions = Vec::new();
        slot_state(
            &self.cfg,
            self.max_seq,
            ring_rows,
            |region| {
                regions.push(region);
                Ok(0)
            },
            |_| Ok(0),
        )?;
        PagedKvLayout::new(&self.dev, regions, self.max_seq, self.comm.recv, ring_rows)
    }

    pub fn decode_paged_layout(&self) -> Result<PagedKvLayout> {
        self.paged_layout(self.history_rows())
    }

    pub fn decode_slot_reservation_bytes(&self, tokens: usize) -> Result<usize> {
        self.decode_paged_layout()?.reservation_bytes(tokens)
    }

    pub fn new_decode_slot(&self) -> Result<DecodeSlot> {
        self.new_decode_slot_in(&self.decode_paged_layout()?)
    }

    /// Share the rank layout and its backing pool across serving slots.
    pub fn new_decode_slot_in(&self, layout: &PagedKvLayout) -> Result<DecodeSlot> {
        let ring_rows = self.history_rows();
        ensure!(
            layout.matches(self.comm.recv, self.max_seq, ring_rows),
            "KV layout belongs to another rank or cache shape"
        );
        self.allocate_decode_slot(layout, ring_rows)
    }

    /// Isolated parity oracle for the batch probe. Uses absolute row addresses.
    pub fn new_dense_reference_slot(&self) -> Result<DecodeSlot> {
        self.allocate_decode_slot(
            &self.paged_layout(HistoryRows::default())?,
            HistoryRows::default(),
        )
    }

    fn allocate_decode_slot(
        &self,
        layout: &PagedKvLayout,
        ring_rows: HistoryRows,
    ) -> Result<DecodeSlot> {
        let paged = layout.reserve()?;
        let mut fixed = Arena::default();
        let mut state = {
            let mut addresses = paged.addresses();
            let state = slot_state(
                &self.cfg,
                self.max_seq,
                ring_rows,
                |_| {
                    addresses
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("missing KV region"))
                },
                |bytes| fixed.zeroed(bytes.max(1)),
            )?;
            ensure!(addresses.next().is_none(), "unused KV region");
            state
        };
        state.tail.renew_lease();
        let layers: Vec<_> = state
            .caches
            .iter()
            .enumerate()
            .map(|(l, cache)| {
                let src = self.cfg.kv_source_of(l).map(|src| &state.caches[src]);
                let (keys_fp4, key_scales) = src.and_then(|s| s.idx_k4).unwrap_or_default();
                DeviceLayer {
                    window: cache.win,
                    compressed: src.and_then(|s| s.comp).unwrap_or(0),
                    keys: src.and_then(|s| s.idx_k).unwrap_or(0),
                    keys_fp4,
                    key_scales,
                    kv_history: cache.kv_hist.unwrap_or(0),
                    score_history: cache.sc_hist.unwrap_or(0),
                    ring_rows: cache.ring_rows as i32,
                    history_rows: cache.hist_rows as i32,
                }
            })
            .collect();
        let layer_table = fixed.put(&layers)?;
        Ok(DecodeSlot {
            layer_table,
            owner: self.comm.recv,
            paged,
            state,
            fixed,
        })
    }

    pub(super) fn validate_decode_slot(&self, slot: &DecodeSlot) -> Result<()> {
        ensure!(
            slot.owner == self.comm.recv,
            "packed decode slot belongs to a different rank"
        );
        Ok(())
    }

    /// Ensure all cache regions cover the requested exclusive logical token bound.
    pub fn ensure_decode_slot(&self, slot: &mut DecodeSlot, through: usize) -> Result<()> {
        ensure!(
            slot.owner == self.comm.recv,
            "packed decode slot belongs to a different rank"
        );
        slot.paged.ensure(through, &self.stream)?;
        slot.state.tail.mapped_tokens = slot.paged.mapped_tokens();
        Ok(())
    }

    /// Fence the rank's compute streams while independent transfer workers continue.
    pub fn synchronize_packed_compute(&self) -> Result<()> {
        for stream in [&self.stream, &self.side, &self.par, &self.ixs, &self.eng] {
            stream.synchronize()?;
        }
        Ok(())
    }

    pub fn release_decode_slot(&self, slot: &mut DecodeSlot) -> Result<()> {
        ensure!(
            slot.owner == self.comm.recv,
            "packed decode slot belongs to a different rank"
        );
        self.dev.synchronize()?;
        self.release_idle_decode_slot(slot)
    }

    /// The caller has joined every stream and transfer retaining this idle slot.
    pub fn release_idle_decode_slot(&self, slot: &mut DecodeSlot) -> Result<()> {
        self.validate_decode_slot(slot)?;
        slot.paged.release()?;
        slot.state.tail.state.set(Default::default());
        slot.state.tail.raw_floor.set(0);
        for cache in slot.state.caches.iter().chain(&slot.state.mtp_caches) {
            cache.truncate(0);
        }
        slot.state.tail.mapped_tokens = 0;
        slot.state.tail.renew_lease();
        slot.state.short_ready = false;
        Ok(())
    }

    /// Run eager prefill, restore, or capture for a slot using the rank's shared scratch.
    /// GPU tasks must be ordered on the rank's streams. The slot must outlive outstanding work.
    pub fn with_decode_slot<T>(
        &mut self,
        slot: &mut DecodeSlot,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        ensure!(
            slot.owner == self.comm.recv,
            "packed decode slot belongs to a different rank"
        );
        let bound = BoundSlot::new(self, &mut slot.state);
        f(bound.rank)
    }

    /// Run a target-model step with one anchor per sequence. All TP ranks must see the same
    /// ordered lanes and tokens. Returns the next token for each lane on every rank.
    pub fn decode_batch(
        &mut self,
        lanes: &mut [(&mut DecodeSlot, DecodeToken)],
    ) -> Result<Vec<i32>> {
        let width = lanes.len();
        ensure!(
            (1..=self.packed_decode_capacity()).contains(&width),
            "packed decode width {width} is out of range"
        );
        ensure!(
            !self.image_prefill && !self.spec_prep_par.get() && !self.stream.is_capturing()?,
            "packed decode requires an eager autoregressive step"
        );
        for (slot, token) in lanes.iter() {
            ensure!(
                slot.owner == self.comm.recv,
                "packed decode slot belongs to a different rank"
            );
            ensure!(
                token.position < self.max_seq,
                "packed decode position exceeds context capacity"
            );
            ensure!(
                token.position < slot.paged.mapped_tokens(),
                "packed decode slot has insufficient mapped KV"
            );
            ensure!(
                (0..self.cfg.vocab_size as i32).contains(&token.token),
                "packed decode token exceeds vocabulary"
            );
        }

        let saved = (self.b, self.meta, self.cur_m, self.host_pos_end, self.hcur);
        let result = self.decode_batch_inner(lanes);
        (self.b, self.meta, self.cur_m, self.host_pos_end, self.hcur) = saved;
        result
    }

    fn decode_batch_inner(
        &mut self,
        lanes: &mut [(&mut DecodeSlot, DecodeToken)],
    ) -> Result<Vec<i32>> {
        let segments: Vec<_> = lanes
            .iter()
            .enumerate()
            .map(|(row, (_, token))| PackedSegment {
                row,
                position: token.position,
                rows: 1,
                candidates: None,
            })
            .collect();
        let width = lanes.len();
        self.forward_packed_rows(lanes, &segments, width, false)?;
        // Persistent-head records only read temperature, fixed to one by set_sampling.
        // Keep the seed, step and final draw bound to the owning sequence.
        let quads = self.w.head.n / 64;
        let records = ((self.tp == 4 || self.tp == 8) && quads <= 512).then_some(self.samp_rec);
        self.packed_lm_head(
            self.packed_execution_rows(width),
            records.map(|_| lanes[0].0.state.sampling),
        )?;
        // Each rank owns one peer mailbox and epoch counter. One CTA walks the live
        // descriptors in identical order across ranks and stores one draw per slot.
        self.prep_gather_pending.set(false);
        self.launch(
            "samp_packed_ar",
            Launch::new(1, 1024),
            Args::new()
                .ptr(self.packed_rows)
                .i32(width as i32)
                .push(self.head_sampling_io(records.is_some())),
        )?;
        self.head_rows = width;
        self.read_next(width)
    }

    pub(super) fn forward_packed_rows(
        &mut self,
        lanes: &mut [(&mut DecodeSlot, DecodeToken)],
        segments: &[PackedSegment],
        width: usize,
        draft_hidden: bool,
    ) -> Result<()> {
        ensure!(segments.len() == lanes.len() && (1..=self.packed_target_rows).contains(&width));
        for ((slot, token), segment) in lanes.iter().zip(segments) {
            let end = segment
                .position
                .checked_add(segment.rows)
                .ok_or_else(|| anyhow::anyhow!("packed position overflow"))?;
            ensure!(
                segment.position == token.position && end <= slot.mapped_tokens(),
                "unmapped packed target rows"
            );
            let first_read = segment.position.saturating_sub(self.cfg.window_size - 1);
            for cache in &slot.state.caches {
                cache.check_window(first_read, segment.position)?;
                cache.check_history(segment.position, end)?;
            }
        }
        if let Err(error) = self.run_packed_rows(lanes, segments, width, draft_hidden) {
            // Some layers can already have overwritten rows on a failed step.
            for (slot, _) in lanes.iter() {
                for c in &slot.state.caches {
                    c.truncate(0);
                }
            }
            return Err(error);
        }
        for ((slot, _), segment) in lanes.iter().zip(segments) {
            let end = segment.position + segment.rows;
            for c in &slot.state.caches {
                c.record_window(segment.position, end);
                c.record_history(segment.position, end);
            }
        }
        Ok(())
    }

    fn run_packed_rows(
        &mut self,
        lanes: &mut [(&mut DecodeSlot, DecodeToken)],
        segments: &[PackedSegment],
        width: usize,
        draft_hidden: bool,
    ) -> Result<()> {
        let execution_rows = self.packed_execution_rows(width);
        self.set_packed_meta(execution_rows)?;
        let mut upload = [DeviceRow::default(); ROWS_PER_UPLOAD];
        let mut row = 0usize;
        for ((slot, token), segment) in lanes.iter().zip(segments) {
            ensure!(
                segment.row == row
                    && segment.rows > 0
                    && segment.rows <= MAX_M
                    && segment.rows <= width - row
                    && segment.position == token.position,
                "invalid packed target row range"
            );
            ensure!(
                slot.owner == self.comm.recv,
                "packed slot belongs to another rank"
            );
            let (mask, ids) = segment
                .candidates
                .unwrap_or((slot.state.candidates, slot.state.candidate_ids));
            for local in 0..segment.rows {
                upload[row % ROWS_PER_UPLOAD] = DeviceRow {
                    meta: slot.state.meta,
                    history: slot.state.token_history,
                    sampling: slot.state.sampling,
                    candidate_mask: mask + (local * (self.max_seq / 8 + 8)) as u64,
                    candidate_ids: ids + (local * self.cfg.candidate_topk_blocks * 4) as u64,
                    layers: slot.layer_table,
                    local: local as i32,
                    position: (segment.position + local) as i32,
                    anchor: token.token,
                    _pad: 0,
                };
                row += 1;
                if row.is_multiple_of(ROWS_PER_UPLOAD) || row == width {
                    let base = (row - 1) / ROWS_PER_UPLOAD * ROWS_PER_UPLOAD;
                    self.launch(
                        "install_packed_rows",
                        Launch::new(1, 32),
                        Args::new()
                            .ptr(self.packed_rows)
                            .push(upload)
                            .i32(base as i32)
                            .i32((row - base) as i32)
                            .i32(i32::from(!draft_hidden)),
                    )?;
                }
            }
        }
        ensure!(row == width, "incomplete packed target row range");
        self.hcur = 0;
        self.eng_pending.set(0);
        self.eng_joined.set(0);
        if width < execution_rows {
            let stride = size_of::<DeviceRow>();
            deepseek_ultrafast_cuda::memset_d8_async(
                self.packed_rows + (width * stride) as u64,
                0,
                (execution_rows - width) * stride,
                &self.stream,
            )?;
        }
        if execution_rows > 8 {
            return self.wide_backbone(execution_rows, draft_hidden);
        }

        for ((slot, _), &segment) in lanes.iter_mut().zip(segments) {
            self.with_packed_segment(segment, slot, |rank| {
                let hc = rank.w.layers[0].hc_attn.0;
                if !rank.embed_prep(segment.rows, hc, rank.cmp_hist)? {
                    rank.embed_meta(segment.rows, rank.cmp_hist)?;
                    rank.hc_prep(segment.rows, rank.epi(hc, true))?;
                }
                Ok(())
            })?;
        }

        for layer in 0..self.cfg.n_layers {
            // The SGL chain can pre-quantize the next layer's packed inputs. The
            // attention kernel below instead reads one row of the raw inputs.
            self.qkv_img.set((0, 0));
            for ((slot, _), &segment) in lanes.iter_mut().zip(segments) {
                self.with_packed_segment(segment, slot, |rank| {
                    rank.attention_layer(layer, segment.rows, draft_hidden)?;
                    if rank.cand_pending.replace(false) {
                        rank.par_join()?;
                    }
                    rank.hc_join()
                })?;
            }
            self.attention_output(layer, width)?;
            self.ffn_layer(layer, width)?;
        }
        Ok(())
    }

    fn forward_wide_rows(&mut self, rows: usize, draft_hidden: bool) -> Result<()> {
        ensure!(
            self.pm.is_some(),
            "wide packed decode needs grouped MoE workspaces"
        );
        let c = self.cfg.clone();
        let heads = c.n_heads / self.tp;
        let hd = c.head_dim;
        ensure!(
            hd == 512
                && c.rope_head_dim == 64
                && c.window_size == 128
                && 128 + c.index_topk <= 16 * 48,
            "unsupported packed attention dimensions"
        );
        ensure!(
            c.candidate_block_size > 0
                && self.max_seq.div_ceil(c.candidate_block_size) <= self.max_seq / 8 + 8,
            "packed candidate mask exceeds its allocated row stride"
        );
        self.launch(
            "embed_packed",
            Launch::new(rows as u32, 256),
            Args::new()
                .ptr(self.packed_rows)
                .ptr(self.w.embed)
                .ptr(self.hbuf())
                .i32(c.dim as i32)
                .i32(c.hc_mult as i32)
                .ptr(self.engram.token_map),
        )?;
        self.hc_prep(rows, self.epi(self.w.layers[0].hc_attn.0, true))?;
        for layer in 0..c.n_layers {
            self.cur_layer.set(layer);
            self.qkv_img.set((0, 0));
            let ly = &self.w.layers[layer];
            if let Some(eg) = &ly.engram {
                self.engram_pre(
                    layer,
                    rows,
                    (self.b.eg, self.b.ex, self.b.ekv_part, self.b.ekv),
                    true,
                )?;
                self.launch(
                    "engram_apply_packed",
                    Launch::new(4, 256).grid3((4, rows as u32, 1)),
                    Args::new()
                        .ptr(self.packed_rows)
                        .ptr(0)
                        .ptr(self.hbuf())
                        .ptr(self.b.ekv)
                        .ptr(eg.qk)
                        .i32(c.dim as i32)
                        .f32(c.norm_eps),
                )?;
                self.hc_prep(rows, self.epi(ly.hc_attn.0, false))?;
            }
            if draft_hidden
                && let Some(t) = c.dspark_target_layer_ids.iter().position(|&l| l == layer)
            {
                let gx = c.dim.div_ceil(256) as u32;
                self.launch(
                    "hc_mean",
                    Launch::new(gx, 256).grid3((gx, rows as u32, 1)),
                    Args::new()
                        .ptr(self.meta)
                        .ptr(self.hbuf())
                        .ptr(self.b.main_hidden + (t * c.dim * 2) as u64)
                        .i32(c.dim as i32)
                        .i32((3 * c.dim) as i32),
                )?;
            }

            self.hc_fork_dots(
                ly.hc_attn,
                (layer != 0 && ly.engram.is_none()) || self.prep_side_dots(),
                "attn",
            )?;
            self.gemv(
                &ly.wqkv,
                self.b.x_raw,
                c.dim,
                rows,
                self.b.q_lat,
                c.q_lora_rank,
                GemvOpts {
                    split: Some((c.q_lora_rank, self.b.kv, hd)),
                    ..self.xin(ly.attn_norm)
                },
            )?;
            self.gemv(
                &ly.wq_b,
                self.b.q_lat,
                c.q_lora_rank,
                rows,
                self.b.q,
                heads * hd,
                GemvOpts {
                    prenorm: Some((ly.q_norm, c.norm_eps)),
                    ..Default::default()
                },
            )?;
            let ratio = c.compress_ratios[layer];
            let table = if ratio > 0 {
                self.rope_comp
            } else {
                self.rope_base
            };
            let q = self.rope_params(self.b.q, hd, hd, heads, table, false, Q_NONE, 0, 0, 0);
            let kv = self.rope_params(self.b.kv, hd, hd, 1, table, false, Q_FP8, 0, 0, ly.kv_norm);
            let gx = (heads + 1) as u32;
            self.launch(
                "rope_packed_qkv",
                Launch::new(gx, 64).grid3((gx, rows as u32, 1)),
                Args::new()
                    .ptr(self.packed_rows)
                    .i32(layer as i32)
                    .push(q)
                    .push(kv),
            )?;

            if ly.indexer.is_some() {
                self.packed_index_layer(layer, rows)?;
            }

            self.launch_packed_attention(layer, rows)?;
            self.hc_late_fork()?;
            self.attention_output(layer, rows)?;
            let ly = &self.w.layers[layer];
            self.hc_fork_dots(ly.hc_ffn, true, "ffn")?;
            let ly = ly as *const _;
            self.ffn_prefill(ly, rows, c.n_activated_experts)?;
        }
        Ok(())
    }

    fn packed_index_layer(&self, layer: usize, rows: usize) -> Result<()> {
        let c = &self.cfg;
        let ly = &self.w.layers[layer];
        let ix = ly.indexer.as_ref().expect("indexed packed attention");
        let (hd, ihd, inh) = (c.head_dim, c.index_head_dim, c.index_n_heads);
        let ratio = c.compress_ratios[layer];
        ensure!(
            ratio > 0 && ihd == 128 && inh == 32,
            "unsupported packed index shape"
        );
        let table = self.rope_comp;
        if let Some(cp) = &ly.compressor {
            let x = self.b.x_raw;
            if ratio > 1 {
                ensure!(ratio <= 4 && hd <= 512);
                self.gemv(
                    &cp.wkv,
                    x,
                    c.dim,
                    rows,
                    self.b.ckv,
                    hd,
                    GemvOpts {
                        out_f32: true,
                        ..self.xin(ly.attn_norm)
                    },
                )?;
                self.gemv(
                    cp.wgate.as_ref().unwrap(),
                    x,
                    c.dim,
                    rows,
                    self.b.csc,
                    hd,
                    GemvOpts {
                        out_f32: true,
                        ..self.xin(ly.attn_norm)
                    },
                )?;
                self.launch(
                    "packed_store_hist",
                    Launch::new(rows as u32, 128),
                    Args::new()
                        .ptr(self.packed_rows)
                        .i32(layer as i32)
                        .ptr(self.b.ckv)
                        .ptr(self.b.csc)
                        .i32(hd as i32),
                )?;
                self.launch(
                    "packed_compress_pool",
                    Launch::new(rows as u32, 128),
                    Args::new()
                        .ptr(self.packed_rows)
                        .i32(layer as i32)
                        .i32(ratio as i32)
                        .ptr(cp.norm)
                        .f32(c.norm_eps)
                        .ptr(self.b.latent)
                        .i32(hd as i32),
                )?;
            } else {
                self.gemv(
                    &cp.wkv,
                    x,
                    c.dim,
                    rows,
                    self.b.latent,
                    hd,
                    self.xin(ly.attn_norm),
                )?;
            }
            let norm = (ratio == 1).then_some((cp.norm, c.norm_eps));
            self.gemv(
                ix.wk.as_ref().unwrap(),
                self.b.latent,
                hd,
                rows,
                self.b.ik,
                ihd,
                GemvOpts {
                    prenorm: norm,
                    ..Default::default()
                },
            )?;
            let key = self.rope_params(
                self.b.ik,
                ihd,
                ihd,
                1,
                table,
                false,
                super::Q_FP4_E8M0,
                ratio,
                0,
                ix.k_norm,
            );
            let comp = self.rope_params(
                self.b.latent,
                hd,
                hd,
                1,
                table,
                false,
                super::Q_FP4_E4M3_16,
                ratio,
                0,
                norm.map_or(0, |p| p.0),
            );
            self.launch(
                "rope_packed_compressed",
                Launch::new(2, 64).grid3((2, rows as u32, 1)),
                Args::new()
                    .ptr(self.packed_rows)
                    .i32(layer as i32)
                    .push(key)
                    .push(comp),
            )?;
        }

        self.gemv(
            &ix.wq_b,
            self.b.q_lat,
            c.q_lora_rank,
            rows,
            self.b.iq,
            inh * ihd,
            GemvOpts {
                prenorm: Some((ly.q_norm, c.norm_eps)),
                ..Default::default()
            },
        )?;
        let iq = self.rope_params(
            self.b.iq,
            ihd,
            ihd,
            inh,
            table,
            false,
            super::Q_FP4_E8M0,
            0,
            0,
            0,
        );
        self.launch(
            "rope_packed_index_q",
            Launch::new(inh as u32, 64).grid3((inh as u32, rows as u32, 1)),
            Args::new().ptr(self.packed_rows).push(iq),
        )?;
        self.idx_weights(ly, rows)?;

        // Score workspaces are bounded separately from activation rows. Each tile reads
        // per-slot logical positions, and only scans the compressed keys reached by that row.
        let workspace_rows = Self::score_rows(self.pmax, self.tp);
        let mask_stride = self.max_seq / 8 + 8;
        let block = c.candidate_block_size;
        for base in (0..rows).step_by(workspace_rows) {
            let n = (rows - base).min(workspace_rows);
            let desc = self.packed_rows + (base * size_of::<DeviceRow>()) as u64;
            let ip = IdxParams {
                q: self.b.iq + (base * inh * ihd * 2) as u64,
                w: self.b.iw + (base * 64 * 2) as u64,
                ldw: 64,
                kc: 0,
                scores: self.b.scores,
                ld: self.max_seq as i32,
                ratio: ratio as i32,
                tmax: (self.max_seq / ratio) as i32,
                cand: 0,
                ld_cand: mask_stride as i32,
                cand_block: block as i32,
                bf16_ref: 0,
            };
            let gx = ((2 * self.sms) / n as u32).max(1);
            self.launch(
                "packed_idx_score",
                Launch::new(gx, 128).grid3((gx, n as u32, 1)),
                Args::new()
                    .ptr(desc)
                    .i32(layer as i32)
                    .push(ip)
                    .i32(i32::from(layer > c.candidate_source_layer)),
            )?;
            let mut tk = TopkParams {
                x: self.b.scores,
                ld: self.max_seq as i32,
                ratio: ratio as i32,
                block: 1,
                k: c.index_topk as i32,
                out: self.b.topk + (base * c.index_topk * 4) as u64,
                mask: 0,
                ld_mask: 0,
                drop_neg_inf: 0,
            };
            self.launch(
                "packed_topk",
                Launch::new(n as u32, 1024),
                Args::new().ptr(desc).push(tk),
            )?;
            if layer == c.candidate_source_layer {
                self.launch(
                    "packed_block_max",
                    Launch::new(64, 256).grid3((64, n as u32, 1)),
                    Args::new()
                        .ptr(desc)
                        .ptr(self.b.scores)
                        .i32(self.max_seq as i32)
                        .i32(ratio as i32)
                        .i32(block as i32)
                        .ptr(self.b.bscores)
                        .i32(mask_stride as i32),
                )?;
                tk = TopkParams {
                    x: self.b.bscores,
                    ld: mask_stride as i32,
                    block: block as i32,
                    k: c.candidate_topk_blocks as i32,
                    ld_mask: mask_stride as i32,
                    drop_neg_inf: 1,
                    ..tk
                };
                self.launch(
                    "packed_topk",
                    Launch::new(n as u32, 1024),
                    Args::new().ptr(desc).push(tk),
                )?;
            }
        }
        Ok(())
    }

    fn launch_packed_attention(&self, layer: usize, rows: usize) -> Result<()> {
        let c = &self.cfg;
        let compressed = c.compress_ratios[layer] > 0;
        let (sp, tile) = if compressed { (16, 48) } else { (8, 16) };
        let table = if compressed {
            self.rope_comp
        } else {
            self.rope_base
        };
        let heads = c.n_heads / self.tp;
        let p = AttnFParams {
            q: self.b.q,
            win: 0,
            comp: 0,
            topk: if compressed { self.b.topk } else { 0 },
            ktop: if compressed { c.index_topk as i32 } else { 0 },
            scale: (c.head_dim as f32).powf(-0.5),
            sink: self.w.layers[layer].attn_sink,
            out: self.b.o,
            cos: table.0,
            sin: table.1,
            r: tile as i32,
            fixed_window: 0,
            nh: heads as i32,
            pre: 1 | if compressed && self.w.layers[layer].indexer.is_none() {
                2
            } else {
                0
            },
            win_floor: self.win_floor.get(),
            ring_rows: 0,
        };
        let smem = ops::attn_smem_for("attn_packed", tile, sp, heads.min(16));
        self.k.get("attn_packed")?.set_max_dynamic_smem(smem)?;
        let mut cfg = Launch::new(sp as u32, 128)
            .grid3((sp as u32, rows as u32, heads.div_ceil(16) as u32))
            .smem(smem);
        cfg.cluster = Some((sp as u32, 1, 1));
        self.launch(
            "attn_packed",
            cfg,
            Args::new().ptr(self.packed_rows).i32(layer as i32).push(p),
        )
    }

    /// Aggregate metadata and bounded slices for row-only projection, MoE and TP kernels.
    /// Positional kernels always use the slot metadata or the per-row descriptors.
    pub(super) fn set_packed_meta(&mut self, rows: usize) -> Result<()> {
        ensure!(
            (1..=self.packed_decode_capacity()).contains(&rows),
            "invalid packed row count"
        );
        self.meta = self.metas;
        self.cur_m = rows;
        self.host_pos_end = rows;
        self.launch(
            "set_packed_meta",
            Launch::new(1, 32),
            Args::new().ptr(self.meta).i32(rows as i32),
        )
    }

    fn wide_backbone(&mut self, rows: usize, draft_hidden: bool) -> Result<()> {
        let key = (rows, draft_hidden);
        if !self.packed_graphs_enabled || !matches!(rows, 32 | 64 | 128) {
            return self.forward_wide_rows(rows, draft_hidden);
        }
        match self.packed_graphs.get(&key) {
            Some(Some(graph)) => return graph.launch(&self.stream),
            None => {
                // Lazy tensor maps, kernel attributes and stream state must be initialized
                // outside capture. The first request at this shape executes exactly once.
                self.forward_wide_rows(rows, draft_hidden)?;
                self.packed_graphs.insert(key, None);
                return Ok(());
            }
            Some(None) => {}
        }
        // The per-call table is populated before this graph. Every recorded pointer
        // belongs to the rank. CUDA kernels dereference the current slot descriptors.
        self.stream.begin_capture()?;
        let captured = self.forward_wide_rows(rows, draft_hidden);
        let graph = self.stream.end_capture();
        captured?;
        let exec = graph?.instantiate()?;
        exec.launch(&self.stream)?;
        self.packed_graphs.insert(key, Some(exec));
        Ok(())
    }

    pub(super) fn packed_lm_head(&self, width: usize, sampling: Option<DevPtr>) -> Result<()> {
        ensure!((1..=self.packed_decode_capacity()).contains(&width));
        let vpad = self.w.head.n;
        let vloc = self.cfg.vocab_size / self.tp;
        let weight = self
            .w
            .head_fp8
            .as_ref()
            .filter(|_| ops::verify_head_fp8())
            .unwrap_or(&self.w.head);
        // The persistent head and its optional top-probability records use at most eight
        // rows per CTA. Each tile advances all row-strided inputs and outputs together.
        let ns = self.cfg.dim / 256;
        for row in (0..width).step_by(8) {
            let offset = row as u64;
            self.gemv(
                weight,
                self.b.x_raw + offset * (self.cfg.dim * 2) as u64,
                self.cfg.dim,
                (width - row).min(8),
                self.b.logits + offset * (vpad * 4) as u64,
                vpad,
                GemvOpts {
                    out_f32: true,
                    persist: true,
                    prenorm: Some((self.w.norm, self.cfg.norm_eps)),
                    norm_part: Some((self.b.part + offset * (ns * 26 * 4) as u64, ns)),
                    samp_rec: sampling.map(|sp| {
                        (
                            sp,
                            self.samp_rec + offset * (vpad / 64 * 20 * 4) as u64,
                            vloc,
                            self.rank * vloc,
                        )
                    }),
                    ..Default::default()
                },
            )?;
        }
        Ok(())
    }

    pub(super) fn with_packed_segment<T>(
        &mut self,
        segment: PackedSegment,
        slot: &mut DecodeSlot,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let packed = (self.b, self.meta, self.cur_m, self.host_pos_end);
        let view = self.row_view(segment.row);
        let (cand, cand_ids) = segment
            .candidates
            .unwrap_or((slot.state.candidates, slot.state.candidate_ids));
        // Only activations carried by packed operations use a row offset. Temporary query,
        // indexer and attention workspaces remain in their original aligned buffers.
        self.b = Bufs {
            h: view.h,
            pre_a: view.pre_a,
            post_a: view.post_a,
            comb_a: view.comb_a,
            o: view.o,
            topk: self.b.topk + (segment.row * self.cfg.index_topk * 4) as u64,
            x_raw: view.x_raw,
            part: view.part,
            next: view.next,
            main_hidden: view.main_hidden,
            cand,
            cand_ids,
            ..self.b
        };
        self.meta = slot.state.meta;
        self.cur_m = segment.rows;
        self.host_pos_end = segment.position + segment.rows;
        let result = self.with_decode_slot(slot, f);
        (self.b, self.meta, self.cur_m, self.host_pos_end) = packed;
        result
    }
}

impl DecodeSlot {
    pub(super) fn packed_addresses(&self) -> (DevPtr, DevPtr) {
        (self.state.meta, self.state.sampling)
    }
}
