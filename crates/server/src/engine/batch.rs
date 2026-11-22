use super::{
    AdmissionQueue, Capture, Event, Fence, Finish, Job, Prepared, Shared, Worker, align_job,
};
use crate::publisher::CaptureProgress;
use anyhow::{Result, ensure};
use deepseek_ultrafast_engine::{
    kvstore,
    model::{
        DecodeSlot, DecodeToken, MAX_PACKED_DECODE_CONTEXT, MAX_PACKED_DECODE_WIDTH, PMAX,
        PagedKvLayout, PrefillProgress, Rank,
    },
    vision::{InvalidImageStaging, StagedImages},
};
use deepseek_ultrafast_kv_transfer::Candidate;
use deepseek_ultrafast_logging::{PackedPlanBudget, PackedPlanStep};
use deepseek_ultrafast_server::{
    generation::SchedulingWork,
    request_log::Decode,
    scheduler::{
        Cancelled, Handle, Limits, Outcome, Prefill, Request, Scheduler, StepPlan, StepResult,
    },
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Mutex, Weak, atomic::Ordering},
    time::Instant,
};

#[path = "hot_prefix.rs"]
mod hot_prefix;
use hot_prefix::{HotPrefix, HotPrefixes};

const REPLAY_ROWS: usize = 128;
const MAX_RESIDENTS: usize = 128;
const READY_LOOKAHEAD: usize = 4;
const DEFAULT_COPY_MIB: usize = 2;
const MAX_COPY_MIB: usize = 8;
const RUNTIME_HBM_HEADROOM: usize = 2 << 30;
const NS_PER_MS: u128 = 1_000_000;
const DECODE_OBSERVATIONS: usize = 32;

#[derive(Clone, Copy)]
pub struct Config {
    pub max_num_seqs: usize,
    pub max_num_decode_seqs: Option<usize>,
    pub max_num_active_seqs: Option<usize>,
    pub max_target_rows: usize,
    pub kv_bytes: Option<usize>,
    pub hot_kv_bytes: usize,
    pub kv_copy_mib: usize,
    pub drafts: usize,
    pub busy_prefill_ms: u64,
    pub busy_prefill_interval: u64,
    pub max_prefill_wait_ms: u64,
    pub decode_max_gap_ms: Option<u64>,
    pub decode_min_tokens_per_sec: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_num_seqs: 1,
            max_num_decode_seqs: None,
            max_num_active_seqs: None,
            max_target_rows: PMAX,
            kv_bytes: None,
            hot_kv_bytes: 0,
            kv_copy_mib: DEFAULT_COPY_MIB,
            drafts: 0,
            busy_prefill_ms: 4,
            busy_prefill_interval: 1,
            max_prefill_wait_ms: 50,
            decode_max_gap_ms: None,
            decode_min_tokens_per_sec: None,
        }
    }
}

impl Config {
    pub fn enabled(self) -> bool {
        self.max_num_seqs > 1
    }

    pub(crate) fn resident_candidate_leases(self) -> usize {
        if self.enabled() {
            self.max_num_seqs + READY_LOOKAHEAD
        } else {
            0
        }
    }

    fn decode_width(self) -> usize {
        self.max_num_decode_seqs.unwrap_or(
            self.max_num_seqs
                .min(8 / (1 + self.drafts)),
        )
    }

    fn active_decoder_limit(self) -> usize {
        self.max_num_active_seqs.unwrap_or(self.max_num_seqs)
    }

    pub(crate) fn target_decode_rows(self) -> usize {
        self.decode_width() * (1 + self.drafts)
    }

    fn padded_decode_rows(self) -> Option<usize> {
        let rows = self.target_decode_rows();
        (rows > 8).then_some(rows)
    }

    fn lifecycle_rows(self) -> usize {
        // Begin-prefill reconstruction can run on the same tick as the final
        // retirement anchor, which also uses the fixed backbone shape.
        (REPLAY_ROWS - 1) + self.padded_decode_rows().unwrap_or(1)
    }

    pub fn validate(self, max_seq: usize, chunk: usize) -> Result<()> {
        ensure!(
            (1..=MAX_RESIDENTS).contains(&self.max_num_seqs),
            "--max-num-seqs must be between 1 and {MAX_RESIDENTS}"
        );
        ensure!(
            self.drafts < 8 && (self.drafts == 0 || self.enabled()),
            "--packed-drafts requires packed decode and at most seven drafts"
        );
        ensure!(
            (1..=self.max_num_seqs).contains(&self.active_decoder_limit()),
            "--max-num-active-seqs must be positive and fit the resident limit"
        );
        ensure!(
            self.busy_prefill_interval > 0
                && self.max_prefill_wait_ms > 0
                && self.decode_max_gap_ms != Some(0)
                && self.decode_min_tokens_per_sec != Some(0),
            "prefill cadence and decode targets must be positive"
        );
        ensure!(
            self.hot_kv_bytes == 0
                || (self.enabled() && self.kv_bytes.is_some_and(|bytes| self.hot_kv_bytes <= bytes)),
            "--hot-kv-cache-memory-bytes requires packed decode and cannot exceed the KV budget"
        );
        ensure!(
            (DEFAULT_COPY_MIB..=MAX_COPY_MIB).contains(&self.kv_copy_mib),
            "--kv-copy-mib must be between {DEFAULT_COPY_MIB} and {MAX_COPY_MIB}"
        );
        ensure!(
            (1..=self
                .max_num_seqs
                .min(MAX_PACKED_DECODE_WIDTH / (1 + self.drafts)))
                .contains(&self.decode_width()),
            "--max-num-decode-seqs must fit the resident limit and {MAX_PACKED_DECODE_WIDTH} packed target rows"
        );
        if self.enabled() {
            ensure!(
                (REPLAY_ROWS..=MAX_PACKED_DECODE_CONTEXT).contains(&max_seq),
                "packed decode requires --max-seq between {REPLAY_ROWS} and {MAX_PACKED_DECODE_CONTEXT}"
            );
            ensure!(
                (1..=PMAX).contains(&chunk),
                "packed prefill requires --chunk between 1 and {PMAX}"
            );
            ensure!(
                (self.lifecycle_rows() + REPLAY_ROWS + self.target_decode_rows()..=PMAX)
                    .contains(&self.max_target_rows),
                "--max-num-batched-tokens must fit one decode launch, admission, replay, and retirement, and be at most {PMAX}"
            );
            ensure!(
                self.kv_bytes.is_some_and(|bytes| bytes > 0),
                "packed decode requires a positive --kv-cache-memory-bytes per rank"
            );
            ensure!(
                self.busy_prefill_ms > 0,
                "--busy-prefill-ms must be positive"
            );
        }
        Ok(())
    }

    fn limits(self, chunk: usize, width: usize, capacity: usize) -> Limits {
        Limits {
            max_residents: self.max_num_seqs,
            max_decode_seqs: width,
            max_active_decoders: self.active_decoder_limit(),
            decode_rows: 1 + self.drafts,
            padded_decode_rows: self.padded_decode_rows(),
            max_target_rows: self.max_target_rows - self.lifecycle_rows(),
            max_replay_rows: REPLAY_ROWS,
            busy_prefill_rows: chunk.min(256),
            busy_prefill_interval: self.busy_prefill_interval,
            max_prefill_wait_ns: u128::from(self.max_prefill_wait_ms) * NS_PER_MS,
            idle_prefill_rows: chunk,
            reservation_capacity: capacity,
            max_bypass: 4,
        }
    }
}

pub(super) struct Sessions {
    live: Arc<Mutex<HashSet<String>>>,
    queue: Weak<AdmissionQueue>,
}

impl Sessions {
    pub(super) fn new(queue: &Arc<AdmissionQueue>) -> Self {
        Self {
            live: Arc::new(Mutex::new(HashSet::new())),
            queue: Arc::downgrade(queue),
        }
    }

    pub(super) fn claim(&self, session: &str) -> Option<SessionLease> {
        let mut live = self.live.lock().unwrap();
        if live.contains(session) {
            return None;
        }
        live.insert(session.to_owned());
        Some(SessionLease {
            session: session.to_owned(),
            live: self.live.clone(),
            queue: self.queue.clone(),
        })
    }
}

pub(super) struct SessionLease {
    session: String,
    live: Arc<Mutex<HashSet<String>>>,
    queue: Weak<AdmissionQueue>,
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        self.live.lock().unwrap().remove(&self.session);
        if let Some(queue) = self.queue.upgrade() {
            queue.wake();
        }
    }
}

struct Entry {
    job: Arc<Job>,
    prep: Prepared,
    reserved_tokens: usize,
    reserved_bytes: usize,
    emitted: Vec<i32>,
    handle: Option<Handle>,
    cached: usize,
    decode_start: Option<Instant>,
    iterations: usize,
    accepted_drafts: usize,
    forwarded: usize,
    snapshot_tokens: Option<Arc<Vec<i32>>>,
    estimated_export_bytes: Option<usize>,
}

struct Coordinator {
    scheduler: Scheduler,
    entries: HashMap<u64, Entry>,
    ids: Vec<Option<u64>>,
    request_id: u64,
    layout: PagedKvLayout,
    max_seq: usize,
    pool_bytes: usize,
    hot: HotPrefixes,
    hot_budget: usize,
    prefill_ns_per_row: u128,
    prefill_budget_ns: u128,
    replay_ns: u128,
    restore_timing: CopyTiming,
    retirement_timing: CopyTiming,
    clock: Instant,
    max_decode_width: usize,
    max_decode_gap_ns: Option<u128>,
    min_decode_tokens_per_sec: Option<u128>,
    decode_samples: VecDeque<DecodeSample>,
}

struct DecodeSample {
    plan_id: u64,
    wall_ns: u128,
    prefill_ns: u128,
    useful_per_lane: usize,
    width: usize,
}

struct Execution {
    results: Vec<StepResult>,
    decode_useful_per_lane: Option<usize>,
    decode_ms: Option<f64>,
    decode_forward_ms: Option<f64>,
    decode_tokens: usize,
    prefill_ns: u128,
    restore_ms: Option<f64>,
    restore_bytes: Option<usize>,
    retirement: Option<RetirementTiming>,
    export_bytes: Option<usize>,
}

impl Execution {
    fn trace(&self, step: &Step, wall_ns: u128) {
        let Some(budget) = step.budget_trace else {
            return;
        };
        if let Some(elapsed_ms) = self.decode_ms {
            let mut lanes: Vec<_> = step
                .decode
                .iter()
                .map(|(handle, _)| [handle.slot as u64, handle.generation])
                .collect();
            lanes.sort_unstable_by_key(|lane| lane[0]);
            deepseek_ultrafast_logging::emit(deepseek_ultrafast_logging::Event::PackedDecodeStep {
                plan_id: step.id,
                lanes,
                scheduled_target_rows: step.target_rows,
                elapsed_ms,
            });
        }
        let (prefill_encode_rows, prefill_replay_rows) = match &step.prefill {
            Some((Prefill::Encode { start, end, .. }, _)) => (end - start, 0),
            Some((Prefill::Replay { rows, .. }, _)) => (0, *rows),
            None => (0, 0),
        };
        let retirement = self.retirement.as_ref();
        deepseek_ultrafast_logging::emit(deepseek_ultrafast_logging::Event::PackedPlanStep(
            PackedPlanStep {
                plan_id: step.id,
                wall_ms: trace_ms(wall_ns),
                decode_ms: self.decode_ms,
                decode_forward_ms: self.decode_forward_ms,
                decode_lanes: step.decode.len(),
                decode_tokens: self.decode_tokens,
                prefill_encode_rows,
                prefill_replay_rows,
                prefill_ms: step.prefill.as_ref().map(|_| trace_ms(self.prefill_ns)),
                restore_ms: self.restore_ms,
                restore_bytes: self.restore_bytes,
                retirement_ms: retirement.and_then(|timing| timing.total_ms),
                retirement_anchor_ms: retirement.and_then(|timing| timing.anchor_ms),
                retirement_export_ms: retirement.and_then(|timing| timing.export_ms),
                export_bytes: self.export_bytes,
                budget,
            },
        ));
    }
}

#[derive(Default)]
struct RetirementTiming {
    total_ms: Option<f64>,
    anchor_ms: Option<f64>,
    export_ms: Option<f64>,
}

impl RetirementTiming {
    fn record_export(&mut self, elapsed_ms: f64) {
        *self.export_ms.get_or_insert(0.0) += elapsed_ms;
    }
}

#[derive(Default)]
struct CopyTiming {
    ns_per_kib: u128,
    pending_ns: u128,
}

impl CopyTiming {
    fn observe(&mut self, bytes: Option<usize>, wall_ns: u128) {
        if let Some(bytes) = bytes {
            self.pending_ns += wall_ns;
            if bytes > 0 {
                let observed = (self.pending_ns * 1024 / bytes as u128).max(1);
                self.ns_per_kib = if self.ns_per_kib == 0 {
                    observed
                } else {
                    (self.ns_per_kib * 3 + observed) / 4
                };
                self.pending_ns = 0;
            }
        } else {
            self.pending_ns = 0;
        }
    }

    fn estimate_ms(&self, bytes: usize) -> Option<u64> {
        if bytes == 0 {
            Some(0)
        } else if self.ns_per_kib > 0 {
            Some(ns_to_ms(bytes as u128 * self.ns_per_kib / 1024))
        } else {
            None
        }
    }
}

struct Admission {
    handle: Handle,
    job: Arc<Job>,
    candidate: Option<Candidate>,
    local_prefix: usize,
}

struct Retirement {
    handle: Handle,
    job: Arc<Job>,
    // Includes the prompt and precisely the output accepted by its event channel.
    tokens: Arc<Vec<i32>>,
    forwarded: usize,
    retain_hot: bool,
}

pub(super) struct Step {
    id: u64,
    target_rows: usize,
    admission: Option<Admission>,
    decode: Vec<(Handle, DecodeToken)>,
    restore: Option<(Handle, Arc<Job>)>,
    prefill: Option<(Prefill, Arc<Job>)>,
    retire: Option<Retirement>,
    evict_hot: Vec<usize>,
    budget_trace: Option<PackedPlanBudget>,
}

impl Coordinator {
    fn new(limits: Limits, layout: PagedKvLayout, max_seq: usize, config: Config) -> Result<Self> {
        Ok(Self {
            scheduler: Scheduler::new(limits)?,
            entries: HashMap::new(),
            ids: vec![None; limits.max_residents],
            request_id: 0,
            layout,
            max_seq,
            pool_bytes: limits.reservation_capacity,
            hot: HotPrefixes::new(limits.max_residents),
            hot_budget: config.hot_kv_bytes,
            prefill_ns_per_row: 0,
            prefill_budget_ns: u128::from(config.busy_prefill_ms) * NS_PER_MS,
            replay_ns: 0,
            restore_timing: CopyTiming::default(),
            retirement_timing: CopyTiming::default(),
            clock: Instant::now(),
            max_decode_width: limits.max_decode_seqs,
            max_decode_gap_ns: config.decode_max_gap_ms.map(|ms| u128::from(ms) * NS_PER_MS),
            min_decode_tokens_per_sec: config.decode_min_tokens_per_sec.map(u128::from),
            decode_samples: VecDeque::with_capacity(DECODE_OBSERVATIONS),
        })
    }

    fn enqueue(&mut self, job: Job, prep: Prepared) -> Result<()> {
        self.request_id = self
            .request_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("request ID overflow"))?;
        let id = self.request_id;
        let prompt_tokens = prep
            .candidate
            .as_ref()
            .and_then(|c| c.aligned.as_ref())
            .map(|aligned| aligned.tokens.len())
            .filter(|&n| {
                deepseek_ultrafast_server::generation::max_new_limit(n, self.max_seq) >= job.max_new
            })
            .unwrap_or(0)
            .max(job.tokens.len());
        let through = prompt_tokens + job.max_new + 4 * deepseek_ultrafast_model::limits::MAX_M;
        let reserved_bytes = self.layout.reservation_bytes(through)?;
        self.scheduler.enqueue(Request {
            id,
            prompt_tokens,
            reservation: reserved_bytes,
        })?;
        self.entries.insert(
            id,
            Entry {
                job: Arc::new(job),
                prep,
                reserved_tokens: through,
                reserved_bytes,
                emitted: Vec::new(),
                handle: None,
                cached: 0,
                decode_start: None,
                iterations: 0,
                accepted_drafts: 0,
                forwarded: 0,
                snapshot_tokens: None,
                estimated_export_bytes: None,
            },
        );
        Ok(())
    }

    fn cancel(&mut self) {
        self.entries.retain(|&id, entry| {
            if !entry.job.cancel.load(Ordering::Relaxed) {
                return true;
            }
            match self.scheduler.cancel(id) {
                Some(Cancelled::Ready) => {
                    entry.job.done(Finish::Cancelled, 0);
                    false
                }
                Some(Cancelled::Resident(_)) => {
                    entry.finish(Finish::Cancelled);
                    true
                }
                None => true,
            }
        });
    }

    fn entry(&self, handle: Handle) -> &Entry {
        let entry = &self.entries[&self.ids[handle.slot].expect("resident slot has a request")];
        assert_eq!(entry.handle, Some(handle));
        entry
    }

    fn plan(
        &mut self,
        queue: &AdmissionQueue,
        slots: &[ModelSlot],
        trace: bool,
    ) -> Result<Option<Step>> {
        self.cancel();
        if self.scheduler.is_empty() {
            let Some((job, prep)) = queue.pop() else {
                return Ok(None);
            };
            self.enqueue(job, prep)?;
        } else if self.scheduler.resident_len() < self.ids.len() {
            while self.scheduler.ready_len() < self.ids.len().min(READY_LOOKAHEAD) {
                let Some((job, prep)) = queue.try_pop() else {
                    break;
                };
                self.enqueue(job, prep)?;
            }
        }

        let (budget, budget_trace) = self.update_compute_budget(trace);

        let plan = self
            .scheduler
            .next_plan_with_slot(
                self.clock.elapsed().as_nanos(),
                budget > 0 && self.replay_ns <= budget,
                |id, slot| match self.hot.get(slot) {
                    None => 1,
                    Some(prefix) => {
                        let entry = &self.entries[&id];
                        if Self::local_prefix(prefix, entry) > 0 {
                            0
                        } else {
                            prefix.retired_at.saturating_add(2)
                        }
                    }
                },
            )?
            .ok_or_else(|| anyhow::anyhow!("resident work has no runnable plan"))?;
        let StepPlan {
            id,
            admission,
            decode,
            restore,
            prefill,
            retire,
            target_rows,
        } = plan;
        let mut evict_hot = Vec::new();
        let admission = admission.map(|admit| {
            let entry = self
                .entries
                .get_mut(&admit.request_id)
                .expect("admitted request exists");
            self.ids[admit.handle.slot] = Some(admit.request_id);
            entry.handle = Some(admit.handle);
            let local_prefix = self.hot.remove(admit.handle.slot).map_or(0, |prefix| {
                let reused = Self::local_prefix(&prefix, entry);
                if reused == 0 {
                    evict_hot.push(admit.handle.slot);
                }
                reused
            });
            self.hot.discard_session(&entry.job.session, &mut evict_hot);
            Admission {
                handle: admit.handle,
                job: entry.job.clone(),
                candidate: entry.prep.candidate.take(),
                local_prefix,
            }
        });
        let reserved = self.scheduler.reserved();
        self.hot.evict_to(self.hot_budget.min(self.pool_bytes - reserved), &mut evict_hot);
        let decode = decode
            .into_iter()
            .map(|handle| {
                let entry = self.entry(handle);
                (
                    handle,
                    DecodeToken {
                        position: entry.job.tokens.len() + entry.emitted.len() - 1,
                        token: *entry.emitted.last().expect("decode has an anchor"),
                    },
                )
            })
            .collect();
        let restore = restore.map(|handle| (handle, self.entry(handle).job.clone()));
        let prefill = prefill.map(|work| (work, self.entry(prefill_handle(work)).job.clone()));
        let retire = if let Some(handle) = retire {
            let entry = self.entry_mut(handle)?;
            let tokens = entry
                .snapshot_tokens
                .get_or_insert_with(|| {
                    let mut tokens = Vec::new();
                    if !entry.emitted.is_empty() {
                        tokens.reserve(entry.job.tokens.len() + entry.emitted.len());
                        tokens.extend_from_slice(&entry.job.tokens);
                        tokens.extend_from_slice(&entry.emitted);
                    }
                    Arc::new(tokens)
                })
                .clone();
            let reserved_bytes = entry.reserved_bytes;
            let mut retirement = Retirement {
                handle,
                job: entry.job.clone(),
                tokens,
                forwarded: entry.forwarded,
                retain_hot: false,
            };
            if self.hot_budget > 0 && !retirement.tokens.is_empty() {
                let bound = slots[handle.slot].storage.mapped_tokens().max(retirement.tokens.len());
                let bytes = self.layout.reservation_bytes(bound)?;
                let hot_after = self.hot_budget.min(self.pool_bytes - (reserved - reserved_bytes));
                if bytes <= hot_after {
                    self.hot.evict_to(hot_after - bytes, &mut evict_hot);
                    retirement.retain_hot = true;
                }
            }
            Some(retirement)
        } else {
            None
        };
        Ok(Some(Step {
            id,
            target_rows,
            admission,
            decode,
            restore,
            prefill,
            retire,
            evict_hot,
            budget_trace,
        }))
    }

    fn entry_mut(&mut self, handle: Handle) -> Result<&mut Entry> {
        let id = self.ids[handle.slot].expect("completed slot has a request");
        let entry = self.entries.get_mut(&id).expect("completed request exists");
        ensure!(
            entry.handle == Some(handle),
            "stale packed worker slot result"
        );
        Ok(entry)
    }

    fn local_prefix(prefix: &HotPrefix, entry: &Entry) -> usize {
        prefix.reusable(&entry.job.session, entry.job.prompt(), entry.reserved_tokens)
    }

    fn complete(&mut self, step: &Step, results: &[StepResult], slots: &[ModelSlot]) -> Result<()> {
        self.scheduler.complete(step.id, results)?;
        for retired in results
            .iter()
            .filter(|result| result.outcome == Outcome::Retired)
        {
            let retirement = step.retire.as_ref().expect("retirement has a scheduled slot");
            if retirement.retain_hot {
                let storage = &slots[retired.handle.slot].storage;
                self.hot.insert(retired.handle.slot, HotPrefix {
                    session: retirement.job.session.clone(),
                    tokens: retirement.tokens.clone(),
                    images: retirement.job.spans().to_vec(),
                    mapped_tokens: storage.mapped_tokens(),
                    raw_tail_floor: storage.raw_tail_floor(),
                    bytes: storage.mapped_bytes(),
                    retired_at: step.id,
                });
            }
            let id = self.ids[retired.handle.slot]
                .take()
                .expect("retired slot has a request");
            self.entries.remove(&id);
        }
        Ok(())
    }

    fn observe_prefill(&mut self, elapsed_ns: u128, rows: usize) {
        let observed = (elapsed_ns / rows as u128).max(1);
        self.prefill_ns_per_row = if self.prefill_ns_per_row == 0 {
            observed
        } else {
            (self.prefill_ns_per_row * 3 + observed) / 4
        };
    }

    fn observe_step(&mut self, step: &Step, execution: &Execution, wall_ns: u128) {
        if let Some((prefill, _)) = &step.prefill {
            match prefill {
                Prefill::Encode { start, end, .. } => {
                    self.observe_prefill(execution.prefill_ns, end - start);
                }
                Prefill::Replay { .. } => {
                    self.replay_ns = if self.replay_ns == 0 {
                        execution.prefill_ns
                    } else {
                        (self.replay_ns * 3 + execution.prefill_ns) / 4
                    };
                }
            }
        }
        if let Some(useful_per_lane) = execution.decode_useful_per_lane {
            if self.decode_samples.len() == DECODE_OBSERVATIONS {
                self.decode_samples.pop_front();
            }
            self.decode_samples.push_back(DecodeSample {
                plan_id: step.id,
                wall_ns: wall_ns.max(1),
                prefill_ns: execution.prefill_ns,
                useful_per_lane,
                width: step.decode.len(),
            });
        }
        self.restore_timing.observe(execution.restore_bytes, wall_ns);
        self.retirement_timing.observe(execution.export_bytes, wall_ns);
    }

    fn update_compute_budget(&mut self, trace: bool) -> (u128, Option<PackedPlanBudget>) {
        let mut budget = self.prefill_budget_ns;
        let mut peaks = None;
        let window_ns = self
            .min_decode_tokens_per_sec
            .and_then(|rate| {
                self.decode_samples
                    .iter()
                    .map(|sample| sample.useful_per_lane as u128 * 1_000 * NS_PER_MS / rate)
                    .min()
            })
            .into_iter()
            .chain(self.max_decode_gap_ns)
            .min();
        if let Some(window_ns) = window_ns
            && let Some(peak) = self.decode_samples.iter().max_by_key(|sample| sample.wall_ns)
        {
            // Capacity is an admission target from recent full-step durations.
            // Existing decoders drain naturally when this target moves lower.
            let max_active = if peak.wall_ns > window_ns {
                peak.width.saturating_sub(1).max(1)
            } else {
                (window_ns / peak.wall_ns)
                    .min(MAX_RESIDENTS as u128) as usize
                    * self.max_decode_width
            };
            self.scheduler.set_active_decoder_limit(max_active);
            let rounds = self
                .scheduler
                .active_decoder_slots()
                .count()
                .div_ceil(self.max_decode_width)
                .max(1);
            let without_prefill =
                |sample: &DecodeSample| sample.wall_ns.saturating_sub(sample.prefill_ns);
            let non_prefill_peak = self
                .decode_samples
                .iter()
                .max_by_key(|sample| without_prefill(sample))
                .expect("decode sample window has a peak");
            budget = budget.min(
                (window_ns / rounds as u128).saturating_sub(without_prefill(non_prefill_peak)),
            );
            peaks = trace.then_some((peak, non_prefill_peak));
        }
        let rows = if budget == 0 {
            0
        } else if self.prefill_ns_per_row > 0 {
            (budget / self.prefill_ns_per_row).min(usize::MAX as u128) as usize
        } else {
            256
        };
        self.scheduler.set_busy_prefill_rows(rows);
        let trace = trace.then(|| PackedPlanBudget {
            busy_prefill_budget_ms: trace_ms(budget),
            busy_prefill_rows: self.scheduler.busy_prefill_rows(),
            active_decoders: self.scheduler.active_decoder_slots().count(),
            active_decoder_limit: self.scheduler.active_decoder_limit(),
            prefill_backlog_requests: self.scheduler.prefill_backlog().1,
            decode_samples: self.decode_samples.len(),
            decode_window_ms: window_ns.map(trace_ms),
            peak_plan_id: peaks.map(|(peak, _)| peak.plan_id),
            peak_wall_ms: peaks.map(|(peak, _)| trace_ms(peak.wall_ns)),
            peak_decode_lanes: peaks.map(|(peak, _)| peak.width),
            non_prefill_peak_plan_id: peaks.map(|(_, peak)| peak.plan_id),
            non_prefill_peak_ms: peaks
                .map(|(_, peak)| trace_ms(peak.wall_ns.saturating_sub(peak.prefill_ns))),
        });
        (budget, trace)
    }

    fn scheduling_work_ms(&self) -> (Option<u64>, Option<u64>) {
        let (prefill_rows, prefill_requests) = self.scheduler.prefill_backlog();
        let prefill = if prefill_requests == 0 {
            Some(0)
        } else if (prefill_rows == 0 || self.prefill_ns_per_row > 0) && self.replay_ns > 0 {
            Some(ns_to_ms(
                (prefill_rows as u128 * self.prefill_ns_per_row)
                    + prefill_requests as u128 * self.replay_ns,
            ))
        } else {
            None
        };
        let active = self.scheduler.active_decoder_slots();
        let decode = if active.clone().count() < self.scheduler.active_decoder_limit() {
            Some(0)
        } else if let Some(slowest) = self.decode_samples.iter().max_by_key(|sample| sample.wall_ns) {
            let waves = active.clone().count().div_ceil(self.max_decode_width);
            active
                .map(|handle| {
                    let entry = self.entry(handle);
                    entry.job.max_new.saturating_sub(entry.emitted.len())
                })
                .min()
                .map(|remaining| {
                    let ticks = remaining.div_ceil(slowest.useful_per_lane.max(1)) as u128 * waves as u128;
                    ns_to_ms(ticks * slowest.wall_ns)
                })
        } else {
            None
        };
        (prefill, decode)
    }

    fn retirement_backlog(&mut self, slots: &[ModelSlot], rk: &Rank) -> usize {
        self.scheduler
            .retiring_slots()
            .map(|handle| {
                let slot = &slots[handle.slot];
                if let Some(export) = &slot.export {
                    return export.remaining_bytes();
                }
                let id = self.ids[handle.slot].expect("retiring slot has a request");
                let entry = self.entries.get_mut(&id).expect("retiring request exists");
                if slot.retirement_started || slot.fence.is_none() || entry.emitted.is_empty() {
                    return 0;
                }
                *entry.estimated_export_bytes.get_or_insert_with(|| {
                    let tokens = entry.job.tokens.len() + entry.emitted.len();
                    let tail = Capture::Complete(entry.emitted.len()).tail();
                    // Use the full requested tail until the export fixes its source ranges.
                    kvstore::image_bytes(rk, tokens, tail, false).unwrap_or_default()
                })
            })
            .sum()
    }
}

fn ns_to_ms(ns: u128) -> u64 {
    ns.div_ceil(NS_PER_MS).min(u64::MAX as u128) as u64
}

fn trace_ms(ns: u128) -> f64 {
    ns as f64 / NS_PER_MS as f64
}

fn elapsed_ms(started: Instant) -> f64 {
    deepseek_ultrafast_logging::ms(started.elapsed())
}

impl Entry {
    fn finish(&self, finish: Finish) {
        if let Some(start) = self.decode_start {
            self.job.trace.decoded(Decode {
                elapsed: start.elapsed(),
                iterations: self.iterations,
                accepted_drafts: self.accepted_drafts,
            });
        }
        self.job.done(finish, self.emitted.len());
    }

    fn sample(
        &mut self,
        tokens: &[i32],
        first: bool,
        forwarded: usize,
        accepted: usize,
    ) -> Outcome {
        self.forwarded = forwarded;
        if first {
            self.job.trace.prefilled(self.cached);
            self.decode_start = Some(Instant::now());
        } else {
            self.iterations += 1;
            self.accepted_drafts += accepted;
        }
        let job = &self.job;
        if job.cancel.load(Ordering::Relaxed)
            || (first
                && job
                    .events
                    .send(Event::Started {
                        prompt_tokens: job.tokens.len(),
                        cached_tokens: self.cached,
                    })
                    .is_err())
        {
            self.finish(Finish::Cancelled);
            return Outcome::Finished;
        }
        let room = job.max_new - self.emitted.len();
        let within_limit = &tokens[..tokens.len().min(room)];
        let (output, reason) = if let Some(index) = within_limit
            .iter()
            .position(|token| job.stop_ids.contains(token))
        {
            (&within_limit[..=index], Some(Finish::Stop))
        } else {
            (
                within_limit,
                (within_limit.len() == room).then_some(Finish::Length),
            )
        };
        if job.events.send(Event::Tokens(output.to_vec())).is_err() {
            self.finish(Finish::Cancelled);
            return Outcome::Finished;
        }
        self.emitted.extend_from_slice(output);
        match reason {
            Some(reason) => {
                self.finish(reason);
                Outcome::Finished
            }
            None => Outcome::Advanced,
        }
    }
}

fn prefill_handle(work: Prefill) -> Handle {
    match work {
        Prefill::Encode { handle, .. } | Prefill::Replay { handle, .. } => handle,
    }
}

struct ModelSlot {
    storage: DecodeSlot,
    restore: Option<kvstore::PrefixRestore>,
    cached: usize,
    progress: Option<PrefillProgress>,
    images: Option<StagedImages>,
    fence: Option<Fence>,
    export: Option<CaptureProgress>,
    retirement_started: bool,
}

pub(super) struct Runner {
    slots: Vec<ModelSlot>,
    coordinator: Option<Coordinator>,
    trace_steps: bool,
    drafts: usize,
    copy_bytes: usize,
}

impl Runner {
    fn memory_pool(
        rk: &Rank,
        shared: &Shared,
        layout: &PagedKvLayout,
        configured: usize,
    ) -> Result<usize> {
        let (free, _) = rk.dev.mem_info()?;
        let budget = free.saturating_sub(RUNTIME_HBM_HEADROOM).min(configured);
        shared.rank_memory.lock().unwrap()[rk.rank] = (budget, layout.granularity());
        shared.barrier.wait();
        let pool = {
            let ranks = shared.rank_memory.lock().unwrap();
            ensure!(
                ranks.iter().all(|&(_, page)| page == layout.granularity()),
                "packed TP ranks have different KV page granularity"
            );
            ranks.iter().map(|&(free, _)| free).min().expect("at least one rank")
        };
        shared.barrier.wait();
        Ok(pool)
    }

    pub(super) fn new(
        rk: &mut Rank,
        shared: &Shared,
        config: Config,
        chunk: usize,
    ) -> Result<Self> {
        rk.set_packed_target_rows(config.target_decode_rows())?;
        let layout = rk.decode_paged_layout()?;
        if config.drafts != 0 {
            rk.validate_packed_spec(config.drafts)?;
        }
        let required = layout.reservation_bytes(rk.max_seq)?;
        let mut slots = Vec::with_capacity(config.max_num_seqs);
        for _ in 0..config.max_num_seqs {
            slots.push(ModelSlot {
                storage: rk.new_decode_slot_in(&layout)?,
                restore: None,
                cached: 0,
                progress: None,
                images: None,
                fence: None,
                export: None,
                retirement_started: false,
            });
        }
        let fixed = slots
            .iter()
            .map(|slot| slot.storage.fixed_bytes())
            .sum::<usize>();
        let configured = config.kv_bytes.expect("validated KV budget").saturating_sub(fixed);
        let pool = Self::memory_pool(rk, shared, &layout, configured)?;
        ensure!(
            required <= pool,
            "the maximum context needs {required} mapped KV bytes per rank, but the effective pool has {pool}"
        );
        let rows_for_warm = 2 + config.drafts;
        let warm_slot = layout.reservation_bytes(rows_for_warm)?;
        let width = config
            .decode_width()
            .min(config.active_decoder_limit())
            .min(pool / warm_slot);
        ensure!(width > 0, "KV budget leaves no room for packed warmup");
        let mut warm_anchors = Vec::with_capacity(slots.len());
        for slot in slots.iter_mut().take(width) {
            rk.ensure_decode_slot(&mut slot.storage, rows_for_warm)?;
            let token = rk.with_decode_slot(&mut slot.storage, |rk| {
                rk.set_sampling(0)?;
                rk.prefill(&[0], 0, 1)
            })?;
            warm_anchors.push(DecodeToken { position: 1, token });
        }
        // Buffers, tensor maps, draft workspaces and graph executables must be live
        // before the final free-HBM sample. The second identical shape captures
        // the warmed backbone. Retirement uses AR even on a DSpark replica.
        for _ in 0..2 {
            let mut lanes: Vec<_> = slots
                .iter_mut()
                .zip(&warm_anchors)
                .map(|(slot, &token)| (&mut slot.storage, token))
                .collect();
            if config.drafts == 0 {
                rk.decode_batch(&mut lanes)?;
            } else {
                rk.spec_decode_batch(&mut lanes, config.drafts)?;
                rk.decode_batch(&mut lanes[..1])?;
            }
        }
        for slot in slots.iter_mut().take(width) {
            rk.release_decode_slot(&mut slot.storage)?;
        }
        let pool = Self::memory_pool(rk, shared, &layout, configured)?;
        ensure!(
            required <= pool,
            "after packed warmup the maximum context needs {required} KV bytes per rank, but the effective pool has {pool}"
        );
        layout.set_backing_budget(pool)?;
        let width = width.min(pool / warm_slot);
        let limits = config.limits(chunk, width, pool);
        let coordinator = if rk.rank == 0 {
            shared.load.set_kv_usage(pool, 0, 0);
            shared.load.set_scheduling_work(SchedulingWork {
                prefill_ms: Some(0),
                decode_ms: Some(0),
                restore_ms: Some(0),
                retirement_bytes: Some(0),
                retirement_ms: Some(0),
            });
            Some(Coordinator::new(
                limits,
                layout,
                rk.max_seq,
                config,
            )?)
        } else {
            None
        };
        if coordinator.is_some() {
            deepseek_ultrafast_logging::emit(
                deepseek_ultrafast_logging::Event::PackedDecodeReady {
                    max_residents: config.max_num_seqs,
                    max_decode_width: width,
                    slot_bytes_per_rank: required,
                    kv_bytes_per_rank: pool,
                },
            );
        }
        let trace_steps = coordinator.is_some()
            && std::env::var_os("DSUF_TRACE_BATCH_STEPS").is_some_and(|v| v == "1");
        Ok(Self {
            slots,
            coordinator,
            trace_steps,
            drafts: config.drafts,
            copy_bytes: config.kv_copy_mib << 20,
        })
    }

    pub(super) fn serve(&mut self, w: &mut Worker<'_>, queue: &AdmissionQueue) -> Result<()> {
        let outcome = self.serve_loop(w, queue);
        if outcome.is_err() {
            // Error teardown keeps pending copy sources and destinations alive until DMA drains.
            let _ = w.rk.dev.synchronize();
        }
        outcome
    }

    fn serve_loop(&mut self, w: &mut Worker<'_>, queue: &AdmissionQueue) -> Result<()> {
        loop {
            let started = self.coordinator.is_some().then(Instant::now);
            if let Some(coordinator) = &mut self.coordinator {
                let next = coordinator
                    .plan(queue, &self.slots, self.trace_steps)?
                    .map(Arc::new);
                *w.shared.step.lock().unwrap() = next;
            }
            w.shared.barrier.wait();
            let next = w.shared.step.lock().unwrap().clone();
            let Some(step) = next else { return Ok(()) };
            let execution = self.execute(w, &step)?;
            // All rank work and reads of the plan must finish before rank 0 can recycle a slot.
            w.shared.barrier.wait();
            if let Some(coordinator) = &mut self.coordinator {
                let wall_ns = started
                    .expect("rank zero times the plan")
                    .elapsed()
                    .as_nanos();
                coordinator.observe_step(&step, &execution, wall_ns);
                coordinator.complete(&step, &execution.results, &self.slots)?;
                // The controller samples the plan before diagnostic log I/O.
                execution.trace(&step, wall_ns);
                if step.admission.is_some() || step.retire.is_some() || step.id % 16 == 0 {
                    let reserved = coordinator.scheduler.reserved();
                    let mapped = self
                        .slots
                        .iter()
                        .map(|slot| slot.storage.mapped_bytes())
                        .sum();
                    w.shared
                        .load
                        .set_kv_usage(coordinator.pool_bytes - reserved, reserved, mapped);
                }
                let (prefill, decode) = coordinator.scheduling_work_ms();
                let restore_bytes: usize = self
                    .slots
                    .iter()
                    .filter_map(|slot| slot.restore.as_ref())
                    .map(kvstore::PrefixRestore::remaining_bytes)
                    .sum();
                let retirement_bytes = coordinator.retirement_backlog(&self.slots, &w.rk);
                w.shared.load.set_scheduling_work(SchedulingWork {
                    prefill_ms: prefill,
                    decode_ms: decode,
                    restore_ms: coordinator.restore_timing.estimate_ms(restore_bytes),
                    retirement_bytes: Some(retirement_bytes),
                    retirement_ms: coordinator.retirement_timing.estimate_ms(retirement_bytes),
                });
                if step.admission.is_none()
                    && step.decode.is_empty()
                    && step.prefill.is_none()
                    && execution.results.iter().all(|result| result.outcome == Outcome::Advanced)
                {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            }
        }
    }

    fn execute(&mut self, w: &mut Worker<'_>, step: &Step) -> Result<Execution> {
        for &slot in &step.evict_hot {
            let idle = &mut self.slots[slot];
            ensure!(idle.restore.is_none() && idle.export.is_none(), "idle KV slot still owns a transfer");
            w.rk.release_idle_decode_slot(&mut idle.storage)?;
        }
        let mut completed = Vec::new();
        let mut decode_useful_per_lane = None;
        let mut decode_ms = None;
        let mut decode_forward_ms = None;
        let mut decode_tokens = 0;
        let mut prefill_ns = 0;
        let mut restore_ms = None;
        let mut restore_bytes = None;
        let mut retirement = None;
        let mut export_bytes = None;
        if !step.decode.is_empty() {
            let (handles, mut lanes): (Vec<_>, Vec<_>) = self
                .slots
                .iter_mut()
                .enumerate()
                .filter_map(|(index, slot)| {
                    step.decode
                        .iter()
                        .find(|(handle, _)| handle.slot == index)
                        .map(|&(handle, token)| {
                            ((handle, token.position), (&mut slot.storage, token))
                        })
                })
                .unzip();
            let started = self.trace_steps.then(Instant::now);
            for (slot, token) in &mut lanes {
                w.rk.ensure_decode_slot(slot, token.position + 1 + self.drafts)?;
            }
            let forward_started = self.trace_steps.then(Instant::now);
            let tokens: Vec<(Vec<i32>, usize)> = if self.drafts == 0 {
                w.rk.decode_batch(&mut lanes)?
                    .into_iter()
                    .map(|token| (vec![token], 0))
                    .collect()
            } else {
                w.rk.spec_decode_batch(&mut lanes, self.drafts)?
                    .into_iter()
                    .map(|result| (result.tokens, result.accepted_drafts))
                    .collect()
            };
            decode_forward_ms = forward_started.map(elapsed_ms);
            decode_useful_per_lane = tokens.iter().map(|(tokens, _)| tokens.len()).min();
            ensure!(
                tokens.len() == handles.len() && handles.len() == step.decode.len(),
                "packed result size mismatch"
            );
            let vocab_size = w.rk.cfg.vocab_size;
            ensure!(
                tokens.iter().all(|(tokens, accepted)| {
                    *accepted <= self.drafts
                        && tokens.len() == *accepted + 1
                        && tokens.iter().all(|&token| (token as usize) < vocab_size)
                }),
                "packed decode returned an invalid causal prefix"
            );
            if let Some(started) = started {
                decode_ms = Some(elapsed_ms(started));
                decode_tokens = tokens.iter().map(|(tokens, _)| tokens.len()).sum();
            }
            if let Some(coordinator) = &mut self.coordinator {
                // Stream ready decode output before a possibly slower restore or prefill.
                for ((handle, anchor), (tokens, accepted)) in handles.into_iter().zip(tokens) {
                    let outcome = coordinator.entry_mut(handle)?.sample(
                        &tokens,
                        false,
                        anchor + accepted + 1,
                        accepted,
                    );
                    completed.push(StepResult { handle, outcome });
                }
            }
        }
        if let Some(admit) = &step.admission {
            let slot = &mut self.slots[admit.handle.slot];
            let (job, cached) = slot.admit(w, admit)?;
            if let Some(coordinator) = &mut self.coordinator {
                let entry = coordinator.entry_mut(admit.handle)?;
                entry.cached = cached;
                entry.job = job;
                entry.job.mark_active();
                entry.job.trace.fetched();
                entry.job.trace.admitted();
                completed.push(StepResult {
                    handle: admit.handle,
                    outcome: Outcome::RestorePending {
                        cached_tokens: cached,
                        prompt_tokens: entry.job.tokens.len(),
                    },
                });
            }
        }
        if let Some((handle, job)) = &step.restore {
            let slot = &mut self.slots[handle.slot];
            let before = slot.restore.as_ref().map(kvstore::PrefixRestore::remaining_bytes);
            let started = self.trace_steps.then(Instant::now);
            let progress = slot.restore_step(w, job, self.copy_bytes)?;
            restore_ms = started.map(elapsed_ms);
            let after = slot.restore.as_ref().map_or(0, kvstore::PrefixRestore::remaining_bytes);
            restore_bytes = before.map(|bytes| bytes.saturating_sub(after));
            completed.push(StepResult {
                handle: *handle,
                outcome: match progress {
                    RestoreStep::Pending => Outcome::Advanced,
                    RestoreStep::Ready => Outcome::Restored,
                    RestoreStep::InvalidImage => {
                        if let Some(coordinator) = &mut self.coordinator {
                            coordinator.entry_mut(*handle)?.finish(Finish::InvalidImage);
                        }
                        Outcome::Finished
                    }
                },
            });
        }
        if let Some((work, job)) = &step.prefill {
            let handle = prefill_handle(*work);
            let start_time = Instant::now();
            let token = self.slots[handle.slot].prefill(w, *work, job)?;
            prefill_ns = start_time.elapsed().as_nanos();
            ensure!(
                token.is_none_or(|token| (token as usize) < w.rk.cfg.vocab_size),
                "packed prefill returned an invalid token"
            );
            if let Some(coordinator) = &mut self.coordinator {
                let outcome = match token {
                    Some(token) => {
                        coordinator
                            .entry_mut(handle)?
                            .sample(&[token], true, job.tokens.len(), 0)
                    }
                    None => Outcome::Advanced,
                };
                completed.push(StepResult { handle, outcome });
            }
        }
        if let Some(retire) = &step.retire {
            let slot = &mut self.slots[retire.handle.slot];
            let before = slot.export.as_ref().map(CaptureProgress::remaining_bytes);
            let started = self.trace_steps.then(Instant::now);
            retirement = self.trace_steps.then(RetirementTiming::default);
            let released = slot.retire(w, retire, self.copy_bytes, &mut retirement)?;
            if let Some(timing) = retirement.as_mut() {
                timing.total_ms = started.map(elapsed_ms);
            }
            let after = slot.export.as_ref().map(CaptureProgress::remaining_bytes);
            export_bytes = match (before, after) {
                (Some(before), _) => Some(before.saturating_sub(after.unwrap_or(0))),
                (None, Some(_)) => Some(0),
                (None, None) => None,
            };
            completed.push(StepResult {
                handle: retire.handle,
                outcome: if released {
                    Outcome::Retired
                } else {
                    Outcome::Advanced
                },
            });
        }
        Ok(Execution {
            results: completed,
            decode_useful_per_lane,
            decode_ms,
            decode_forward_ms,
            decode_tokens,
            prefill_ns,
            restore_ms,
            restore_bytes,
            retirement,
            export_bytes,
        })
    }
}

enum RestoreStep {
    Pending,
    Ready,
    InvalidImage,
}

fn all_ranks_accept(w: &Worker<'_>, accepted: bool) -> bool {
    if w.r == 0 {
        w.shared.restore_ok.store(true, Ordering::Release);
    }
    w.shared.barrier.wait();
    if !accepted {
        w.shared.restore_ok.store(false, Ordering::Release);
    }
    w.shared.barrier.wait();
    let accepted = w.shared.restore_ok.load(Ordering::Acquire);
    // A subsequent restore/reduction may reuse the same flag.
    w.shared.barrier.wait();
    accepted
}

impl ModelSlot {
    fn admit(&mut self, w: &mut Worker<'_>, admit: &Admission) -> Result<(Arc<Job>, usize)> {
        let candidate = admit.candidate.as_ref();
        if admit.local_prefix > 0 {
            let job = &admit.job;
            if let (Some(publisher), Some(candidate)) = (&mut w.publisher, candidate) {
                self.fence = Some(publisher.fence(
                    &job.session,
                    Some(candidate.revision),
                    candidate.is_local(),
                ));
            }
            self.cached = admit.local_prefix;
            return Ok((job.clone(), self.cached));
        }
        let snapshot = candidate.and_then(|candidate| candidate.snapshot.as_ref());
        if w.r == 0 {
            let job = if let (Some(snapshot), Some(candidate)) = (snapshot, candidate)
                && candidate.aligned.is_some()
            {
                let mut aligned = (*admit.job).clone();
                align_job(&mut aligned, candidate.clone(), w.shared.max_seq);
                if snapshot.manifest.reusable_prefix(aligned.prompt()) > 0 {
                    Arc::new(aligned)
                } else {
                    admit.job.clone()
                }
            } else {
                admit.job.clone()
            };
            *w.shared.aligned.lock().unwrap() = Some(job);
        }
        w.shared.barrier.wait();
        let mut job = w
            .shared
            .aligned
            .lock()
            .unwrap()
            .clone()
            .expect("admission chose a prompt");
        w.shared.barrier.wait();
        if w.r == 0 {
            w.shared.aligned.lock().unwrap().take();
        }

        let mut cached = 0;
        if let Some(snapshot) = snapshot {
            let prefix = snapshot.manifest.reusable_prefix(job.prompt());
            if prefix > 0 {
                // Planning is host-only. No rank maps pages or launches restore
                // DMA until every rank has accepted the source layout.
                let plan = w.rk.with_decode_slot(&mut self.storage, |rk| {
                    w.restore.begin_prefix(rk, snapshot.clone(), prefix)
                });
                let plan = match plan {
                    Ok(plan) => Some(plan),
                    Err(error) if error.is::<kvstore::IncompatiblePrefix>() => {
                        deepseek_ultrafast_logging::diagnostic!(
                            "rank {} rejected a KV snapshot prefix: {error:#}",
                            w.r
                        );
                        None
                    }
                    Err(error) => return Err(error),
                };
                if all_ranks_accept(w, plan.is_some()) {
                    w.rk.ensure_decode_slot(&mut self.storage, prefix)?;
                    self.restore = plan;
                    cached = prefix;
                } else {
                    job = admit.job.clone();
                }
            }
        }
        self.cached = cached;
        if let (Some(publisher), Some(candidate)) = (&mut w.publisher, candidate) {
            self.fence = Some(publisher.fence(
                &job.session,
                Some(candidate.revision),
                cached > 0 && candidate.is_local(),
            ));
        }
        Ok((job, cached))
    }

    fn restore_step(
        &mut self,
        w: &mut Worker<'_>,
        job: &Job,
        copy_bytes: usize,
    ) -> Result<RestoreStep> {
        if let Some(restore) = &mut self.restore {
            if w.r == 0 {
                w.shared.restore_ok.store(true, Ordering::Release);
            }
            w.shared.barrier.wait();
            let done = w.rk.with_decode_slot(&mut self.storage, |rk| {
                w.restore.step_prefix_async(rk, restore, copy_bytes)
            })?;
            if !done {
                w.shared.restore_ok.store(false, Ordering::Release);
            }
            w.shared.barrier.wait();
            if w.shared.restore_ok.load(Ordering::Acquire) {
                self.restore = None;
            }
            return Ok(RestoreStep::Pending);
        }
        if self.images.is_none() {
            let images = job
                .images
                .as_ref()
                .map_or(&[][..], |images| images.items.as_slice());
            let staged = match w.rk.begin_image_staging(images, &job.tokens, self.cached) {
                Ok(staged) => Some(staged),
                Err(error) if error.is::<InvalidImageStaging>() => {
                    if w.r == 0 {
                        deepseek_ultrafast_logging::diagnostic!("request image staging: {error:#}");
                    }
                    None
                }
                Err(error) => return Err(error),
            };
            if !images.is_empty() && !all_ranks_accept(w, staged.is_some()) {
                return Ok(RestoreStep::InvalidImage);
            }
            self.images = staged;
        }
        let images = self.images.as_mut().expect("image stage initialized");
        let staged = w.rk.stage_next_image(images)?;
        if !staged {
            w.rk.synchronize_packed_compute()?;
            return Ok(RestoreStep::Pending);
        }
        w.rk.with_decode_slot(&mut self.storage, |rk| {
            rk.with_staged_images(images, |rk| {
                rk.set_sampling(job.seed)?;
                self.progress = Some(rk.begin_prefill(&job.tokens, self.cached)?);
                rk.synchronize_packed_compute()
            })
        })?;
        Ok(RestoreStep::Ready)
    }

    fn prefill(&mut self, w: &mut Worker<'_>, work: Prefill, job: &Job) -> Result<Option<i32>> {
        let through = match work {
            Prefill::Encode { end, .. } => end,
            Prefill::Replay { .. } => job.tokens.len(),
        };
        w.rk.ensure_decode_slot(&mut self.storage, through)?;
        let progress = &mut self.progress;
        let images = self
            .images
            .as_mut()
            .expect("prefill has staged image spans");
        let token = w.rk.with_decode_slot(&mut self.storage, |rk| {
            rk.with_staged_images(images, |rk| {
                let token = match work {
                    Prefill::Encode { start, end, .. } => {
                        let rows = end - start;
                        let n = rk.prefill_chunk(
                            &job.tokens,
                            progress.as_mut().expect("encoder has progress"),
                            rows,
                        )?;
                        ensure!(n == rows, "packed encoder cursor mismatch");
                        None
                    }
                    Prefill::Replay { .. } => {
                        // Decode will mutate these slot ranges. Export the immutable
                        // completion under the retirement lease instead.
                        Some(rk.finish_prefill(
                            &job.tokens,
                            progress.take().expect("replay has progress"),
                        )?)
                    }
                };
                rk.synchronize_packed_compute()?;
                Ok(token)
            })
        })?;
        if token.is_some() {
            self.images = None;
        }
        Ok(token)
    }

    fn retire(
        &mut self,
        w: &mut Worker<'_>,
        retire: &Retirement,
        copy_bytes: usize,
        timing: &mut Option<RetirementTiming>,
    ) -> Result<bool> {
        if !self.retirement_started {
            self.retirement_started = true;
            if let Some(restore) = self.restore.take() {
                w.restore.abort_prefix(&restore)?;
            }
            if let Some(&token) = retire.tokens.last() {
                if retire.forwarded < retire.tokens.len() {
                    ensure!(
                        retire.forwarded + 1 == retire.tokens.len(),
                        "retirement is missing more than its final anchor"
                    );
                    let input = DecodeToken {
                        position: retire.tokens.len() - 1,
                        token,
                    };
                    w.rk.ensure_decode_slot(&mut self.storage, retire.tokens.len())?;
                    let started = timing.is_some().then(Instant::now);
                    w.rk.decode_batch(&mut [(&mut self.storage, input)])?;
                    if let (Some(started), Some(timing)) = (started, timing.as_mut()) {
                        timing.anchor_ms = Some(elapsed_ms(started));
                    }
                }
                if let (Some(publisher), Some(fence)) = (&w.publisher, &self.fence) {
                    let n = retire.tokens.len() - retire.job.tokens.len();
                    let started = timing.is_some().then(Instant::now);
                    self.export = w.rk.with_decode_slot(&mut self.storage, |rk| {
                        publisher.begin_capture(
                            rk,
                            &retire.job.session,
                            retire.job.trace.session_label(),
                            &retire.tokens,
                            retire.job.spans(),
                            fence.clone(),
                            Capture::Complete(n),
                        )
                    })?;
                    if let (Some(started), Some(timing)) = (started, timing.as_mut()) {
                        timing.record_export(elapsed_ms(started));
                    }
                }
            }
        }
        if w.r == 0 {
            let done = if let Some(capture) = &mut self.export {
                let publisher = w.publisher.as_ref().expect("rank zero owns the publisher");
                let started = timing.is_some().then(Instant::now);
                let done = w.rk.with_decode_slot(&mut self.storage, |rk| {
                    publisher.step_capture_async(rk, capture, copy_bytes)
                })?;
                if done {
                    publisher.finish_capture(self.export.take().expect("finished capture"))?;
                }
                if let (Some(started), Some(timing)) = (started, timing.as_mut()) {
                    timing.record_export(elapsed_ms(started));
                }
                done
            } else {
                true
            };
            w.shared.retired_slot.store(done, Ordering::Release);
        }
        w.shared.barrier.wait();
        let done = w.shared.retired_slot.load(Ordering::Acquire);
        if done {
            w.rk.synchronize_packed_compute()?;
            self.progress = None;
            self.images = None;
            self.fence = None;
            self.cached = 0;
            self.retirement_started = false;
            if !retire.retain_hot {
                w.rk.release_idle_decode_slot(&mut self.storage)?;
            }
        }
        Ok(done)
    }
}
