use super::batch_decode::PackedSegment;
use super::{DecodeSlot, DecodeToken, Meta, Rank};
use crate::weights::Arena;
use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{self as cu, Args, DevPtr, Launch, PinnedHost};

const MAX_VERIFIER_ROWS_PER_LANE: usize = 8;

#[derive(Debug, PartialEq, Eq)]
pub struct SpecDecoded {
    /// Accepted proposals followed by the target sample. Excludes the input anchor.
    pub tokens: Vec<i32>,
    pub accepted_drafts: usize,
}

#[derive(Clone, Copy)]
struct Proposal {
    tokens: DevPtr,
    logits: DevPtr,
    stats: DevPtr,
    verdict: DevPtr,
    candidates: (DevPtr, DevPtr),
}

pub(super) struct Workspace {
    width: usize,
    lanes: Vec<Proposal>,
    descriptors: DevPtr,
    results: DevPtr,
    descriptors_host: PinnedHost,
    results_host: PinnedHost,
    _arena: Arena,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DraftLane {
    meta: DevPtr,
    proposal: DevPtr,
    verdict: DevPtr,
    input: Meta,
}
const _: () = assert!(size_of::<DraftLane>() == 96);

impl Workspace {
    fn new(rank: &Rank, drafts: usize, capacity: usize) -> Result<Self> {
        let width = drafts + 1;
        let mut arena = Arena::default();
        let mut lanes = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            lanes.push(Proposal {
                tokens: arena.zeroed(64)?,
                // The clustered verifier forms the last target row's proposal
                // pointer too, even though that row never uses draft probabilities.
                logits: arena.zeroed(width * rank.w.head.n * size_of::<f32>())?,
                stats: arena.zeroed(drafts * 1024 * size_of::<[f32; 2]>())?,
                verdict: arena.zeroed(64)?,
                candidates: (
                    arena.zeroed(width * (rank.max_seq / 8 + 8))?,
                    arena.zeroed(
                        (width * rank.cfg.candidate_topk_blocks * size_of::<i32>()).max(1),
                    )?,
                ),
            });
        }
        let descriptor_bytes = capacity * size_of::<DraftLane>();
        let result_bytes = capacity * (width + 2) * size_of::<i32>();
        let descriptors_host = PinnedHost::alloc(descriptor_bytes)?;
        let results_host = PinnedHost::alloc(result_bytes)?;
        // A typed slice is used for each buffer. Initialize the allocations
        // before creating references, including the inactive capacity.
        unsafe {
            descriptors_host.ptr.write_bytes(0, descriptor_bytes);
            results_host.ptr.write_bytes(0, result_bytes);
        }
        Ok(Self {
            width,
            lanes,
            descriptors: arena.alloc(descriptor_bytes)?,
            results: arena.alloc(result_bytes)?,
            descriptors_host,
            results_host,
            _arena: arena,
        })
    }
}

impl Rank {
    pub fn validate_packed_spec(&self, drafts: usize) -> Result<()> {
        let max_drafts = self
            .cfg
            .dspark_block_size
            .min(MAX_VERIFIER_ROWS_PER_LANE - 1);
        ensure!(
            (1..=max_drafts).contains(&drafts),
            "packed DSpark requires 1..={max_drafts} drafts for this checkpoint and profile"
        );
        ensure!(
            !self.w.mtp.is_empty(),
            "packed DSpark requires draft-model weights"
        );
        ensure!(
            drafts < self.packed_decode_capacity(),
            "packed DSpark requires space for its verifier rows"
        );
        self.moe_rows_ok(drafts + 1)
    }

    /// Share MTP projections, FFN, and target verification across the batch.
    /// Attention, RNG and committed cache lengths remain separate per lane.
    /// All ranks must receive the same ordered lanes. A lane's cache must cover the
    /// exclusive position `anchor.position + drafts + 1` for provisional writes.
    /// The committed input length is `anchor.position + accepted_drafts + 1`.
    pub fn spec_decode_batch(
        &mut self,
        lanes: &mut [(&mut DecodeSlot, DecodeToken)],
        drafts: usize,
    ) -> Result<Vec<SpecDecoded>> {
        self.validate_packed_spec(drafts)?;
        let width = drafts + 1;
        let max_lanes = self.packed_target_rows / width;
        ensure!(
            (1..=max_lanes).contains(&lanes.len()),
            "packed DSpark supports up to {max_lanes} lanes with {drafts} drafts"
        );
        ensure!(
            !self.image_prefill && !self.spec_prep_par.get() && !self.stream.is_capturing()?,
            "packed DSpark requires eager decode"
        );
        let execution_rows = self.packed_execution_rows(lanes.len() * width);
        let draft_rows = execution_rows / width * drafts;
        ensure!(
            draft_rows <= MAX_VERIFIER_ROWS_PER_LANE || self.pm.is_some(),
            "wide packed DSpark requires grouped MoE scratch"
        );
        for (slot, token) in &*lanes {
            self.validate_decode_slot(slot)?;
            slot.check_draft_window(token.position, self.cfg.window_size)?;
            ensure!(
                token
                    .position
                    .checked_add(width)
                    .is_some_and(|end| end <= self.max_seq && end <= slot.mapped_tokens()),
                "packed DSpark verifier rows must fit the mapped KV cache"
            );
            ensure!(
                (0..self.cfg.vocab_size as i32).contains(&token.token),
                "packed DSpark anchor exceeds vocabulary"
            );
        }
        if self
            .packed_spec
            .as_ref()
            .is_none_or(|workspace| workspace.width != width || workspace.lanes.len() < lanes.len())
        {
            let capacity = lanes.len().next_power_of_two().min(max_lanes);
            self.packed_spec = Some(Workspace::new(self, drafts, capacity)?);
        }
        let mut workspace = self
            .packed_spec
            .take()
            .expect("packed draft workspace initialized");
        let saved = (
            self.b,
            self.meta,
            self.cur_m,
            self.host_pos_end,
            self.hcur,
            self.samp,
            self.samp_rec,
            self.blk,
            self.mk_grid.get(),
        );
        let result = self.spec_decode_inner(lanes, &mut workspace);
        // A failed launch can leave earlier DMA enqueued against the pinned
        // workspace. Drain those streams before returning it to a caller.
        let result = match result {
            Ok(outputs) => Ok(outputs),
            Err(error) => self.dev.synchronize().and(Err(error)),
        };
        let saved_grid;
        (
            self.b,
            self.meta,
            self.cur_m,
            self.host_pos_end,
            self.hcur,
            self.samp,
            self.samp_rec,
            self.blk,
            saved_grid,
        ) = saved;
        self.mk_grid.set(saved_grid);
        self.packed_spec = Some(workspace);
        result
    }

    fn spec_decode_inner(
        &mut self,
        lanes: &mut [(&mut DecodeSlot, DecodeToken)],
        workspace: &mut Workspace,
    ) -> Result<Vec<SpecDecoded>> {
        let width = workspace.width;
        let rows = lanes.len() * width;
        let execution_rows = self.packed_execution_rows(rows);
        self.blk = width - 1;
        let draft_rows = execution_rows / width * self.blk;
        self.block_idx_for(self.blk)?;
        let descriptors =
            &mut workspace.descriptors_host.as_mut_slice::<DraftLane>()[..lanes.len()];
        let mut segments = Vec::with_capacity(lanes.len());
        for (lane, ((slot, token), &proposal)) in lanes.iter().zip(&workspace.lanes).enumerate() {
            let mut input = token.meta();
            input.m = self.blk as i32;
            input.tokens[1..self.blk].fill(self.cfg.dspark_noise_token_id as i32);
            descriptors[lane] = DraftLane {
                meta: slot.packed_addresses().0,
                proposal: proposal.tokens,
                verdict: proposal.verdict,
                input,
            };
            segments.push(PackedSegment {
                row: lane * width,
                position: token.position,
                rows: width,
                candidates: Some(proposal.candidates),
            });
        }
        let grid = Launch::new(lanes.len() as u32, 32);
        cu::memcpy_htod_async(workspace.descriptors, descriptors, &self.stream)?;
        self.launch(
            "packed_spec_draft_meta",
            grid,
            Args::new().ptr(workspace.descriptors),
        )?;
        self.draft_batch(lanes, &workspace.lanes, draft_rows)?;
        self.launch(
            "packed_spec_verify_meta",
            Launch::new(lanes.len() as u32, 32),
            Args::new().ptr(workspace.descriptors).i32(width as i32),
        )?;

        self.forward_packed_rows(lanes, &segments, rows, true)?;
        // Head records only read the temperature. All lanes use temperature one
        // and verify with their own seed and counter after the shared projection.
        self.packed_lm_head(execution_rows, Some(lanes[0].0.packed_addresses().1))?;
        let logits = self.b.logits;
        let records = self.samp_rec;
        let logit_stride = (self.w.head.n * size_of::<f32>()) as u64;
        let record_stride = (self.w.head.n / 64 * 20 * size_of::<f32>()) as u64;
        for (((slot, _), &segment), &proposal) in
            lanes.iter_mut().zip(&segments).zip(&workspace.lanes)
        {
            self.with_packed_segment(segment, slot, |rank| {
                rank.b.logits = logits + segment.row as u64 * logit_stride;
                rank.samp_rec = records + segment.row as u64 * record_stride;
                rank.b.drafts = proposal.tokens;
                (rank.samp.3, rank.samp.4, rank.samp.5) =
                    (proposal.logits, proposal.stats, proposal.verdict);
                rank.spec_prep_par.set(true);
                let sampled = rank.pick_head(width, true);
                rank.spec_prep_par.set(false);
                sampled
            })?;
        }
        self.launch(
            "packed_spec_result",
            Launch::new(lanes.len() as u32, 32),
            Args::new()
                .ptr(workspace.descriptors)
                .i32(width as i32)
                .i32(self.cfg.vocab_size as i32)
                .ptr(workspace.results),
        )?;
        let results =
            &mut workspace.results_host.as_mut_slice::<i32>()[..lanes.len() * (width + 2)];
        cu::memcpy_dtoh_async(results, workspace.results, &self.stream)?;
        // Keep projection and TP gather on the same shape as the verifier.
        // Only live lane Meta values are trimmed and passed to the append.
        self.clear_spec_padding(
            self.b.main_hidden,
            3 * self.cfg.dim * size_of::<u16>(),
            rows,
            execution_rows,
        )?;
        self.prep_proj(execution_rows)?;
        self.prep_gather(execution_rows)?;
        let main_x = self.b.main_x;
        let main_stride = (self.cfg.dim * size_of::<u16>()) as u64;
        for ((slot, _), &segment) in lanes.iter_mut().zip(&segments) {
            self.with_packed_segment(segment, slot, |rank| {
                rank.b.main_x = main_x + segment.row as u64 * main_stride;
                rank.prep_kv(width)
            })?;
        }
        self.stream.synchronize()?;
        self.head_rows = rows;
        results
            .chunks_exact(width + 2)
            .zip(lanes.iter())
            .map(|(lane, (slot, anchor))| {
                ensure!(
                    lane[2] == anchor.token,
                    "packed DSpark proposal has the wrong anchor"
                );
                let output = verified_output(&lane[2..], lane[0], lane[1], self.cfg.vocab_size)?;
                slot.commit_verification(
                    anchor.position,
                    anchor.position + output.accepted_drafts + 1,
                );
                Ok(output)
            })
            .collect()
    }

    fn draft_batch(
        &mut self,
        lanes: &mut [(&mut DecodeSlot, DecodeToken)],
        proposals: &[Proposal],
        execution_rows: usize,
    ) -> Result<()> {
        let drafts = self.blk;
        let rows = lanes.len() * drafts;
        let segments: Vec<_> = lanes
            .iter()
            .enumerate()
            .map(|(lane, (_, token))| PackedSegment {
                row: lane * drafts,
                position: token.position,
                rows: drafts,
                candidates: None,
            })
            .collect();
        self.set_packed_meta(execution_rows)?;
        self.hcur = 0;
        let dim = self.cfg.dim;
        let hidden = 4 * dim * size_of::<u16>();
        let mix = 4 * size_of::<f32>();
        let attention = (self.cfg.n_heads / self.tp) * self.cfg.head_dim * size_of::<u16>();
        // Per-lane embedding, attention and Markov only touch live rows. The
        // shared draft projection, MoE and head always use a fixed shape.
        for (buffer, stride) in [
            (self.b.h[0], hidden),
            (self.b.h[1], hidden),
            (self.b.pre_a, mix),
            (self.b.post_a, mix),
            (self.b.comb_a, 4 * mix),
            (self.b.o, attention),
            (self.b.x_raw, dim * size_of::<u16>()),
            (self.b.part, (dim / 256) * 26 * size_of::<f32>()),
        ] {
            self.clear_spec_padding(buffer, stride, rows, execution_rows)?;
        }
        let hc = self.w.mtp[0].hc_attn.0;
        for ((slot, _), &segment) in lanes.iter_mut().zip(&segments) {
            self.with_packed_segment(segment, slot, |rank| {
                if !rank.embed_prep(drafts, hc, 0)? {
                    rank.embed_meta(drafts, 0)?;
                    rank.hc_prep(drafts, rank.epi(hc, true))?;
                }
                Ok(())
            })?;
        }
        for layer in 0..self.w.mtp.len() {
            self.qkv_img.set((0, 0));
            for ((slot, _), &segment) in lanes.iter_mut().zip(&segments) {
                self.with_packed_segment(segment, slot, |rank| {
                    rank.mtp_attention(layer, drafts)?;
                    rank.hc_join()
                })?;
            }
            self.mtp_attention_output(layer, execution_rows)?;
            self.mtp_ffn(layer, execution_rows)?;
        }
        self.dspark_draft_head(execution_rows)?;
        let logits = self.b.logits;
        let logit_stride = (self.w.head.n * size_of::<f32>()) as u64;
        // Markov's per-lane grid must run co-resident. Serialize these
        // launches on the GPU while keeping the expensive MTP matrices shared.
        for (((slot, _), &segment), &proposal) in lanes.iter_mut().zip(&segments).zip(proposals) {
            self.with_packed_segment(segment, slot, |rank| {
                rank.b.logits = logits + segment.row as u64 * logit_stride;
                rank.b.drafts = proposal.tokens;
                let saved = (rank.samp.3, rank.samp.4);
                (rank.samp.3, rank.samp.4) = (proposal.logits, proposal.stats);
                let result = rank.dspark_markov(false);
                (rank.samp.3, rank.samp.4) = saved;
                result
            })?;
        }
        Ok(())
    }

    fn clear_spec_padding(
        &self,
        buffer: DevPtr,
        stride: usize,
        live_rows: usize,
        execution_rows: usize,
    ) -> Result<()> {
        if live_rows < execution_rows {
            cu::memset_d8_async(
                buffer + (live_rows * stride) as u64,
                0,
                (execution_rows - live_rows) * stride,
                &self.stream,
            )?;
        }
        Ok(())
    }
}

fn verified_output(
    proposed: &[i32],
    accepted: i32,
    sample: i32,
    vocab: usize,
) -> Result<SpecDecoded> {
    let accepted_drafts = usize::try_from(accepted)?;
    ensure!(
        accepted_drafts < proposed.len(),
        "invalid packed DSpark acceptance count"
    );
    let mut tokens = proposed[1..1 + accepted_drafts].to_vec();
    tokens.push(sample);
    ensure!(
        tokens
            .iter()
            .all(|&token| (0..vocab as i32).contains(&token)),
        "invalid token in packed DSpark result"
    );
    Ok(SpecDecoded {
        tokens,
        accepted_drafts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_emits_accepted_prefix_and_fresh_sample() {
        let draft = [10, 11, 12, 13];
        assert_eq!(verified_output(&draft, 0, 40, 100).unwrap().tokens, [40]);
        assert_eq!(
            verified_output(&draft, 1, 40, 100).unwrap().tokens,
            [11, 40]
        );
        assert_eq!(
            verified_output(&draft, 3, 40, 100).unwrap().tokens,
            [11, 12, 13, 40]
        );
        assert!(verified_output(&draft, 4, 40, 100).is_err());
        assert!(verified_output(&draft, -1, 40, 100).is_err());
        assert!(verified_output(&draft, 2, 100, 100).is_err());
    }
}
