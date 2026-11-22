use std::{collections::VecDeque, fmt};

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_residents: usize,
    pub max_decode_seqs: usize,
    pub max_active_decoders: usize,
    pub decode_rows: usize,
    pub padded_decode_rows: Option<usize>,
    pub max_target_rows: usize,
    pub max_replay_rows: usize,
    pub busy_prefill_rows: usize,
    pub busy_prefill_interval: u64,
    pub max_prefill_wait_ns: u128,
    pub idle_prefill_rows: usize,
    pub reservation_capacity: usize,
    pub max_bypass: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Request {
    pub id: u64,
    pub prompt_tokens: usize,
    pub reservation: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Handle {
    pub slot: usize,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Admission {
    pub request_id: u64,
    pub handle: Handle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Prefill {
    Encode {
        handle: Handle,
        start: usize,
        end: usize,
    },
    Replay {
        handle: Handle,
        rows: usize,
    },
}

#[derive(Clone, Debug)]
pub struct StepPlan {
    pub id: u64,
    pub admission: Option<Admission>,
    pub decode: Vec<Handle>,
    pub restore: Option<Handle>,
    pub prefill: Option<Prefill>,
    pub retire: Option<Handle>,
    pub target_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepResult {
    pub handle: Handle,
    pub outcome: Outcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Admitted {
        cached_tokens: usize,
    },
    RestorePending {
        cached_tokens: usize,
        prompt_tokens: usize,
    },
    Restored,
    Advanced,
    Finished,
    Retired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cancelled {
    Ready,
    Resident(Handle),
}

#[derive(Debug, Eq, PartialEq)]
pub enum ScheduleError {
    InvalidLimits,
    InvalidRequest(u64),
    DuplicateRequest(u64),
    PlanInFlight(u64),
    StalePlan(u64),
    InvalidResults(u64),
    IdExhausted,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => write!(f, "invalid scheduler limits"),
            Self::InvalidRequest(id) => write!(f, "request {id} cannot fit scheduler limits"),
            Self::DuplicateRequest(id) => write!(f, "duplicate request ID {id}"),
            Self::PlanInFlight(id) => write!(f, "step plan {id} is still in flight"),
            Self::StalePlan(id) => write!(f, "step plan {id} is no longer in flight"),
            Self::InvalidResults(id) => write!(f, "invalid results for step plan {id}"),
            Self::IdExhausted => write!(f, "scheduler ID space exhausted"),
        }
    }
}

impl std::error::Error for ScheduleError {}

#[derive(Clone, Copy)]
enum Phase {
    Admitting,
    Restoring { cached_tokens: usize },
    Prefill { cursor: usize },
    Decode,
    Retiring,
}

struct Waiting {
    request: Request,
    bypassed: usize,
}

struct Resident {
    request: Request,
    phase: Phase,
    last_served: u64,
    prefill_served_ns: u128,
    cancelled: bool,
}

#[derive(Default)]
struct Slot {
    generation: u64,
    resident: Option<Resident>,
}

impl Slot {
    fn handle(&self, slot: usize) -> Handle {
        Handle {
            slot,
            generation: self.generation,
        }
    }
}

#[derive(Clone, Copy)]
enum Work {
    Admit,
    Decode,
    Restore,
    Encode { end: usize },
    Replay,
    Retire,
}

impl StepPlan {
    fn work(&self) -> impl Iterator<Item = (Handle, Work)> + '_ {
        self.admission
            .iter()
            .map(|entry| (entry.handle, Work::Admit))
            .chain(self.decode.iter().map(|&handle| (handle, Work::Decode)))
            .chain(self.restore.iter().map(|&handle| (handle, Work::Restore)))
            .chain(self.prefill.iter().map(|prefill| match *prefill {
                Prefill::Encode { handle, end, .. } => (handle, Work::Encode { end }),
                Prefill::Replay { handle, .. } => (handle, Work::Replay),
            }))
            .chain(self.retire.iter().map(|&handle| (handle, Work::Retire)))
    }
}

/// Owns ready requests and GPU reservations. The caller owns request payloads,
/// I/O and compute. It reports retirement only after the slot's last GPU reader.
pub struct Scheduler {
    limits: Limits,
    ready: VecDeque<Waiting>,
    slots: Vec<Slot>,
    reserved: usize,
    plan_id: u64,
    in_flight: Option<StepPlan>,
    busy_prefill_rows: usize,
    active_decoder_limit: usize,
    last_busy_prefill: Option<u64>,
    now_ns: u128,
}

impl Scheduler {
    pub fn new(limits: Limits) -> Result<Self, ScheduleError> {
        let packed_rows = limits
            .max_decode_seqs
            .checked_mul(limits.decode_rows)
            .and_then(|live| match limits.padded_decode_rows {
                Some(padded) if padded < live => None,
                Some(padded) => Some(padded),
                None => Some(live),
            })
            .and_then(|rows| rows.checked_add(limits.max_replay_rows));
        if limits.max_decode_seqs == 0
            || limits.max_decode_seqs > limits.max_residents
            || limits.max_active_decoders == 0
            || limits.max_active_decoders > limits.max_residents
            || limits.decode_rows == 0
            || limits.max_replay_rows == 0
            || limits.busy_prefill_rows == 0
            || limits.busy_prefill_interval == 0
            || limits.max_prefill_wait_ns == 0
            || limits.idle_prefill_rows == 0
            || limits.reservation_capacity == 0
            || !packed_rows.is_some_and(|rows| rows <= limits.max_target_rows)
        {
            return Err(ScheduleError::InvalidLimits);
        }
        Ok(Self {
            limits,
            ready: VecDeque::new(),
            slots: (0..limits.max_residents).map(|_| Slot::default()).collect(),
            reserved: 0,
            plan_id: 0,
            in_flight: None,
            busy_prefill_rows: limits.busy_prefill_rows,
            active_decoder_limit: limits.max_active_decoders,
            last_busy_prefill: None,
            now_ns: 0,
        })
    }

    pub fn enqueue(&mut self, request: Request) -> Result<(), ScheduleError> {
        if request.prompt_tokens == 0
            || request.reservation == 0
            || request.reservation > self.limits.reservation_capacity
        {
            return Err(ScheduleError::InvalidRequest(request.id));
        }
        if self
            .ready
            .iter()
            .any(|entry| entry.request.id == request.id)
            || self.slots.iter().any(|slot| {
                slot.resident
                    .as_ref()
                    .is_some_and(|entry| entry.request.id == request.id)
            })
        {
            return Err(ScheduleError::DuplicateRequest(request.id));
        }
        self.ready.push_back(Waiting {
            request,
            bypassed: 0,
        });
        Ok(())
    }

    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    pub fn resident_len(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.resident.is_some())
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.ready_len() == 0 && self.resident_len() == 0
    }

    pub fn reserved(&self) -> usize {
        self.reserved
    }

    pub fn set_busy_prefill_rows(&mut self, rows: usize) {
        self.busy_prefill_rows = rows.min(self.limits.busy_prefill_rows);
    }

    pub fn busy_prefill_rows(&self) -> usize {
        self.busy_prefill_rows
    }

    pub fn set_active_decoder_limit(&mut self, maximum: usize) {
        self.active_decoder_limit = maximum.clamp(1, self.limits.max_active_decoders);
    }

    pub fn active_decoder_limit(&self) -> usize {
        self.active_decoder_limit
    }

    pub fn active_decoder_slots(&self) -> impl Iterator<Item = Handle> + Clone + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.resident
                .as_ref()
                .filter(|entry| !entry.cancelled && matches!(entry.phase, Phase::Decode))
                .map(|_| slot.handle(index))
        })
    }

    pub fn retiring_slots(&self) -> impl Iterator<Item = Handle> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.resident
                .as_ref()
                .filter(|entry| matches!(entry.phase, Phase::Retiring))
                .map(|_| slot.handle(index))
        })
    }

    pub fn prefill_backlog(&self) -> (usize, usize) {
        self.slots
            .iter()
            .filter_map(|slot| {
                let entry = slot.resident.as_ref()?;
                if entry.cancelled {
                    return None;
                }
                let cursor = match entry.phase {
                    Phase::Prefill { cursor }
                    | Phase::Restoring {
                        cached_tokens: cursor,
                    } => cursor,
                    _ => return None,
                };
                Some(entry.request.prompt_tokens - cursor)
            })
            .fold((0, 0), |(rows, requests), remaining| {
                (rows.saturating_add(remaining), requests + 1)
            })
    }

    pub fn cancel(&mut self, id: u64) -> Option<Cancelled> {
        if let Some(index) = self.ready.iter().position(|entry| entry.request.id == id) {
            self.ready.remove(index);
            return Some(Cancelled::Ready);
        }
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if let Some(entry) = &mut slot.resident
                && entry.request.id == id
                && !entry.cancelled
                && !matches!(entry.phase, Phase::Retiring)
            {
                entry.cancelled = true;
                return Some(Cancelled::Resident(slot.handle(index)));
            }
        }
        None
    }

    fn admit(
        &mut self,
        plan_id: u64,
        slot_priority: impl FnMut(u64, usize) -> u64,
    ) -> Result<Option<Admission>, ScheduleError> {
        if self.slots.iter().all(|slot| slot.resident.is_some()) {
            return Ok(None);
        }
        let available = self.limits.reservation_capacity - self.reserved;
        let mut eligible = None;
        for (waiting_index, entry) in self.ready.iter().enumerate() {
            if entry.request.reservation <= available {
                eligible = Some(waiting_index);
                break;
            }
            if entry.bypassed >= self.limits.max_bypass {
                break;
            }
        }
        let Some(ready_index) = eligible else {
            return Ok(None);
        };
        let request_id = self.ready[ready_index].request.id;
        let mut slot_priority = slot_priority;
        let index = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.resident.is_none())
            .min_by_key(|(index, _)| (slot_priority(request_id, *index), *index))
            .expect("free slot for eligible request")
            .0;
        let slot = &mut self.slots[index];
        let generation = slot
            .generation
            .checked_add(1)
            .ok_or(ScheduleError::IdExhausted)?;
        for entry in self.ready.iter_mut().take(ready_index) {
            entry.bypassed += 1;
        }
        let request = self
            .ready
            .remove(ready_index)
            .expect("eligible request in queue")
            .request;
        self.reserved += request.reservation;
        slot.generation = generation;
        slot.resident = Some(Resident {
            request,
            phase: Phase::Admitting,
            last_served: plan_id,
            prefill_served_ns: self.now_ns,
            cancelled: false,
        });
        Ok(Some(Admission {
            request_id: request.id,
            handle: slot.handle(index),
        }))
    }

    pub fn next_plan(
        &mut self,
        now_ns: u128,
        busy_prefill_replay: bool,
    ) -> Result<Option<StepPlan>, ScheduleError> {
        self.next_plan_with_slot(now_ns, busy_prefill_replay, |_, _| 0)
    }

    pub fn next_plan_with_slot(
        &mut self,
        now_ns: u128,
        busy_prefill_replay: bool,
        slot_priority: impl FnMut(u64, usize) -> u64,
    ) -> Result<Option<StepPlan>, ScheduleError> {
        if let Some(plan) = &self.in_flight {
            return Err(ScheduleError::PlanInFlight(plan.id));
        }
        let id = self
            .plan_id
            .checked_add(1)
            .ok_or(ScheduleError::IdExhausted)?;
        self.now_ns = self.now_ns.max(now_ns);
        for entry in self
            .slots
            .iter_mut()
            .filter_map(|slot| slot.resident.as_mut())
        {
            if entry.cancelled {
                entry.phase = Phase::Retiring;
            }
        }
        let admission = self.admit(id, slot_priority)?;
        let active_decoders = self.active_decoder_slots().count();
        let can_replay = active_decoders < self.active_decoder_limit;
        let idle = active_decoders == 0;
        let interval = self
            .last_busy_prefill
            .is_none_or(|last| id - last >= self.limits.busy_prefill_interval);
        let mut ordered: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                slot.resident
                    .as_ref()
                    .map(|entry| (entry.last_served, index, entry))
            })
            .collect();
        ordered.sort_unstable_by_key(|(age, index, _)| (*age, *index));
        let mut decode = Vec::new();
        let mut next_prefill = None;
        let mut restore = None;
        let mut retire = None;
        for (_, index, entry) in ordered {
            let handle = self.slots[index].handle(index);
            match entry.phase {
                Phase::Decode if decode.len() < self.limits.max_decode_seqs => decode.push(handle),
                Phase::Restoring { .. } if restore.is_none() => restore = Some(handle),
                Phase::Prefill { cursor } if next_prefill.is_none() => {
                    let prompt = entry.request.prompt_tokens;
                    let aged =
                        self.now_ns - entry.prefill_served_ns >= self.limits.max_prefill_wait_ns;
                    let can_encode = cursor < prompt && (idle || self.busy_prefill_rows > 0);
                    if (idle || aged || interval)
                        && (can_encode
                            || (cursor < prompt && aged)
                            || (cursor == prompt
                                && can_replay
                                && (idle || aged || busy_prefill_replay)))
                    {
                        next_prefill = Some((handle, cursor, prompt, aged));
                    }
                }
                Phase::Retiring if retire.is_none() => retire = Some(handle),
                _ => {}
            }
        }
        let mut target_rows = if decode.is_empty() {
            0
        } else {
            self.limits.padded_decode_rows.unwrap_or(decode.len() * self.limits.decode_rows)
        };
        let mut prefill = None;
        if let Some((handle, cursor, prompt, aged)) = next_prefill {
            if cursor == prompt {
                let rows = prompt.min(self.limits.max_replay_rows);
                prefill = Some(Prefill::Replay { handle, rows });
                target_rows += rows;
            } else {
                let quantum = if idle {
                    self.limits.idle_prefill_rows
                } else if aged {
                    self.busy_prefill_rows.max(1)
                } else {
                    self.busy_prefill_rows
                };
                let rows = quantum
                    .min(self.limits.max_target_rows - target_rows)
                    .min(prompt - cursor);
                if rows > 0 {
                    prefill = Some(Prefill::Encode {
                        handle,
                        start: cursor,
                        end: cursor + rows,
                    });
                    target_rows += rows;
                }
            }
        }
        let plan = StepPlan {
            id,
            admission,
            decode,
            restore,
            prefill,
            retire,
            target_rows,
        };
        if plan.work().next().is_none() {
            return Ok(None);
        }
        if !plan.decode.is_empty() && plan.prefill.is_some() {
            self.last_busy_prefill = Some(id);
        }
        self.plan_id = id;
        self.in_flight = Some(plan.clone());
        Ok(Some(plan))
    }

    /// Results can arrive in any order. Validate the whole plan before changing
    /// cursors or freeing reservations, so a rejected result leaves it retryable.
    pub fn complete(&mut self, plan_id: u64, results: &[StepResult]) -> Result<(), ScheduleError> {
        let plan = self
            .in_flight
            .as_ref()
            .filter(|plan| plan.id == plan_id)
            .ok_or(ScheduleError::StalePlan(plan_id))?;
        if results.len() != plan.work().count() {
            return Err(ScheduleError::InvalidResults(plan_id));
        }
        let mut outcomes = vec![None; self.slots.len()];
        for result in results {
            let Some(slot) = self.slots.get(result.handle.slot) else {
                return Err(ScheduleError::InvalidResults(plan_id));
            };
            if slot.generation != result.handle.generation
                || outcomes[result.handle.slot]
                    .replace(result.outcome)
                    .is_some()
            {
                return Err(ScheduleError::InvalidResults(plan_id));
            }
        }
        for (handle, work) in plan.work() {
            let Some(outcome) = outcomes[handle.slot] else {
                return Err(ScheduleError::InvalidResults(plan_id));
            };
            let entry = self.slots[handle.slot]
                .resident
                .as_ref()
                .expect("in-flight slot has a resident");
            let valid = match (work, outcome) {
                (Work::Admit, Outcome::Admitted { cached_tokens }) => {
                    cached_tokens < entry.request.prompt_tokens
                }
                (
                    Work::Admit,
                    Outcome::RestorePending {
                        cached_tokens,
                        prompt_tokens,
                    },
                ) => cached_tokens < prompt_tokens && prompt_tokens <= entry.request.prompt_tokens,
                (Work::Restore, Outcome::Advanced | Outcome::Restored | Outcome::Finished) => true,
                (Work::Retire, Outcome::Advanced | Outcome::Retired) => true,
                (
                    Work::Encode { .. } | Work::Replay | Work::Decode,
                    Outcome::Advanced | Outcome::Finished,
                ) => true,
                _ => false,
            };
            if !valid {
                return Err(ScheduleError::InvalidResults(plan_id));
            }
        }
        let plan = self.in_flight.take().expect("validated step plan");
        for (handle, work) in plan.work() {
            let outcome = outcomes[handle.slot].expect("validated result");
            let slot = &mut self.slots[handle.slot];
            if outcome == Outcome::Retired {
                let entry = slot.resident.take().expect("retiring slot has a resident");
                self.reserved -= entry.request.reservation;
                continue;
            }
            let entry = slot
                .resident
                .as_mut()
                .expect("in-flight slot has a resident");
            entry.last_served = plan.id;
            if matches!(work, Work::Admit | Work::Encode { .. } | Work::Restore) {
                entry.prefill_served_ns = self.now_ns;
            }
            entry.phase = match (work, outcome) {
                _ if entry.cancelled => Phase::Retiring,
                (Work::Admit, Outcome::Admitted { cached_tokens }) => Phase::Prefill {
                    cursor: cached_tokens,
                },
                (
                    Work::Admit,
                    Outcome::RestorePending {
                        cached_tokens,
                        prompt_tokens,
                    },
                ) => {
                    entry.request.prompt_tokens = prompt_tokens;
                    Phase::Restoring { cached_tokens }
                }
                (Work::Restore, Outcome::Restored) => {
                    let Phase::Restoring { cached_tokens } = entry.phase else {
                        unreachable!("restoring slot")
                    };
                    Phase::Prefill {
                        cursor: cached_tokens,
                    }
                }
                (Work::Restore | Work::Retire, Outcome::Advanced) => entry.phase,
                (Work::Encode { end }, Outcome::Advanced) => Phase::Prefill { cursor: end },
                (Work::Replay | Work::Decode, Outcome::Advanced) => Phase::Decode,
                (_, Outcome::Finished) => Phase::Retiring,
                _ => unreachable!("validated result"),
            };
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
