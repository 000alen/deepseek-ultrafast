use deepseek_ultrafast_kv_transfer::PeerCache;
use deepseek_ultrafast_model::limits::{MAX_M, OUT_CAP};
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::mpsc;

pub const DEFAULT_MAX_JOBS: usize = 64;
pub const DEFAULT_MAX_JOB_BYTES: usize = 256 << 20;
const MAX_BUFFERED_EVENTS: usize = 64;

#[derive(Clone)]
pub struct JobIngress(Arc<Mutex<IngressState>>);

struct IngressState {
    sender: Option<mpsc::Sender<Job>>,
    max_jobs: usize,
    max_bytes: usize,
    jobs: usize,
    bytes: usize,
    epoch: String,
    active: usize,
    requests: HashMap<usize, Weak<AtomicBool>>,
    kv_free_bytes: Option<usize>,
    kv_reserved_bytes: Option<usize>,
    kv_mapped_bytes: Option<usize>,
    kv_peer: Option<Weak<PeerCache>>,
    work: SchedulingWork,
    standard_pool_max_output: Option<usize>,
}

#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct SchedulingWork {
    pub prefill_ms: Option<u64>,
    pub decode_ms: Option<u64>,
    pub restore_ms: Option<u64>,
    pub retirement_bytes: Option<usize>,
    pub retirement_ms: Option<u64>,
}

impl SchedulingWork {
    fn unknown(&self) -> bool {
        self.prefill_ms.is_none()
            && self.decode_ms.is_none()
            && self.restore_ms.is_none()
            && self.retirement_bytes.is_none()
            && self.retirement_ms.is_none()
    }
}

#[derive(serde::Serialize)]
pub(crate) struct AdmissionStatus {
    epoch: String,
    queued: usize,
    active: usize,
    available: usize,
    kv_free_bytes: Option<usize>,
    kv_reserved_bytes: Option<usize>,
    kv_mapped_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_data_plane_failures: Option<u64>,
    #[serde(skip_serializing_if = "SchedulingWork::unknown")]
    work: SchedulingWork,
}

#[derive(Clone)]
pub struct LoadReporter(Weak<Mutex<IngressState>>);

impl LoadReporter {
    pub fn set_kv_usage(&self, free: usize, reserved: usize, mapped: usize) {
        if let Some(state) = self.0.upgrade() {
            let mut state = state.lock().unwrap();
            state.kv_free_bytes = Some(free);
            state.kv_reserved_bytes = Some(reserved);
            state.kv_mapped_bytes = Some(mapped);
        }
    }

    pub fn set_scheduling_work(&self, work: SchedulingWork) {
        if let Some(state) = self.0.upgrade() {
            state.lock().unwrap().work = work;
        }
    }

    pub fn observe_kv_peer(&self, peer: &Arc<PeerCache>) {
        if let Some(state) = self.0.upgrade() {
            state.lock().unwrap().kv_peer = Some(Arc::downgrade(peer));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubmitError {
    Full,
    Closed,
}

impl JobIngress {
    pub fn channel(max_jobs: usize, max_bytes: usize) -> (Self, mpsc::Receiver<Job>) {
        assert!(max_bytes > 0, "job byte limit must be positive");
        let (sender, receiver) = mpsc::channel(max_jobs);
        let state = IngressState {
            sender: Some(sender),
            max_jobs,
            max_bytes,
            jobs: 0,
            bytes: 0,
            epoch: format!("{:032x}", rand::random::<u128>()),
            active: 0,
            requests: HashMap::with_capacity(max_jobs),
            kv_free_bytes: None,
            kv_reserved_bytes: None,
            kv_mapped_bytes: None,
            kv_peer: None,
            work: SchedulingWork::default(),
            standard_pool_max_output: None,
        };
        (Self(Arc::new(Mutex::new(state))), receiver)
    }

    pub fn close(&self) {
        self.0.lock().unwrap().sender.take();
    }

    pub fn cancel_outstanding(&self) {
        let mut state = self.0.lock().unwrap();
        state.sender.take();
        for flag in state.requests.values().filter_map(Weak::upgrade) {
            flag.store(true, Ordering::Release);
        }
    }

    pub fn load_reporter(&self) -> LoadReporter {
        LoadReporter(Arc::downgrade(&self.0))
    }

    pub fn with_standard_pool(self, max_output: usize) -> Self {
        self.0.lock().unwrap().standard_pool_max_output = Some(max_output);
        self
    }

    pub(crate) fn standard_pool_max_output(&self) -> Option<usize> {
        self.0.lock().unwrap().standard_pool_max_output
    }

    pub(crate) fn status(&self) -> AdmissionStatus {
        let state = self.0.lock().unwrap();
        AdmissionStatus {
            epoch: state.epoch.clone(),
            queued: state.jobs - state.active,
            active: state.active,
            available: state
                .sender
                .as_ref()
                .filter(|sender| !sender.is_closed())
                .map_or(0, |_| state.max_jobs - state.jobs),
            kv_free_bytes: state.kv_free_bytes,
            kv_reserved_bytes: state.kv_reserved_bytes,
            kv_mapped_bytes: state.kv_mapped_bytes,
            kv_data_plane_failures: state
                .kv_peer
                .as_ref()
                .and_then(Weak::upgrade)
                .map(|peer| peer.data_plane_failures()),
            work: state.work,
        }
    }

    pub(crate) fn submit(&self, mut job: Job) -> Result<(), SubmitError> {
        let bytes = job.retained_bytes();
        let mut state = self.0.lock().unwrap();
        let sender = state
            .sender
            .as_ref()
            .filter(|sender| !sender.is_closed())
            .ok_or(SubmitError::Closed)?;
        if state.jobs >= state.max_jobs || bytes > state.max_bytes - state.bytes {
            return Err(SubmitError::Full);
        }
        let permit = sender
            .clone()
            .try_reserve_owned()
            .map_err(|err| match err {
                mpsc::error::TrySendError::Full(_) => SubmitError::Full,
                mpsc::error::TrySendError::Closed(_) => SubmitError::Closed,
            })?;
        state.jobs += 1;
        state.bytes += bytes;
        let id = Arc::as_ptr(&job.cancel) as usize;
        state.requests.insert(id, Arc::downgrade(&job.cancel));
        job.admission = Some(Arc::new(Admission {
            ingress: Arc::downgrade(&self.0),
            bytes: AtomicUsize::new(bytes),
            id,
            active: AtomicBool::new(false),
        }));
        drop(state);
        permit.send(job);
        Ok(())
    }
}

pub(crate) struct Admission {
    ingress: Weak<Mutex<IngressState>>,
    bytes: AtomicUsize,
    id: usize,
    active: AtomicBool,
}

impl Admission {
    fn try_grow(&self, bytes: usize) -> bool {
        let Some(ingress) = self.ingress.upgrade() else {
            return true;
        };
        let mut state = ingress.lock().unwrap();
        if bytes > state.max_bytes - state.bytes {
            return false;
        }
        state.bytes += bytes;
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        true
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        if let Some(ingress) = self.ingress.upgrade() {
            let mut state = ingress.lock().unwrap();
            state.jobs -= 1;
            state.bytes -= *self.bytes.get_mut();
            state.active -= usize::from(*self.active.get_mut());
            state.requests.remove(&self.id);
        }
    }
}

#[derive(Clone)]
pub struct Job {
    pub trace: crate::request_log::Trace,
    pub session: String,
    pub images: Option<Arc<crate::image_identity::Images>>,
    pub tokens: Arc<Vec<i32>>,
    pub max_new: usize,
    pub seed: u64,
    pub stop_ids: Arc<Vec<i32>>,
    pub cancel: Arc<AtomicBool>,
    pub events: EventSender,
    pub(crate) admission: Option<Arc<Admission>>,
}
impl Job {
    pub fn mark_active(&self) {
        if let Some(admission) = &self.admission
            && let Some(ingress) = admission.ingress.upgrade()
        {
            let mut state = ingress.lock().unwrap();
            if !admission.active.swap(true, Ordering::Relaxed) {
                state.active += 1;
            }
        }
    }

    pub fn prompt(&self) -> deepseek_ultrafast_kv_transfer::Prompt<'_> {
        deepseek_ultrafast_kv_transfer::Prompt::new(&self.tokens, self.spans())
    }
    pub fn spans(&self) -> &[deepseek_ultrafast_kv_transfer::ImageSpan] {
        self.images.as_ref().map_or(&[], |v| &v.spans)
    }
    pub fn done(&self, finish: Finish, completion_tokens: usize) {
        let _ = self.events.send(Event::Done {
            finish,
            completion_tokens,
        });
    }
    fn retained_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>() + self.session.capacity();
        bytes = bytes.saturating_add(self.tokens.capacity().saturating_mul(size_of::<i32>()));
        bytes = bytes.saturating_add(self.stop_ids.capacity().saturating_mul(size_of::<i32>()));
        if let Some(images) = &self.images {
            bytes = bytes.saturating_add(
                images
                    .items
                    .capacity()
                    .saturating_mul(size_of::<deepseek_ultrafast_model::vision::Image>()),
            );
            bytes = bytes.saturating_add(
                images
                    .spans
                    .capacity()
                    .saturating_mul(size_of::<deepseek_ultrafast_kv_transfer::ImageSpan>()),
            );
            for image in &images.items {
                bytes = bytes.saturating_add(size_of_val(image.patches.as_ref()));
            }
        }
        bytes
    }

    /// Reserve any increase before replacing the prompt during cache alignment.
    pub fn reserve_prompt_growth(&self, new_capacity: usize) -> bool {
        let bytes = new_capacity
            .saturating_sub(self.tokens.capacity())
            .saturating_mul(size_of::<i32>());
        bytes == 0
            || self
                .admission
                .as_ref()
                .is_none_or(|admission| admission.try_grow(bytes))
    }
}
impl crate::request_queue::Request for Job {
    fn cancelled(&self) -> bool {
        self.cancel.load(std::sync::atomic::Ordering::Relaxed)
    }
    fn discard(self) {
        self.done(Finish::Cancelled, 0);
    }
}
#[derive(Debug)]
pub enum Event {
    Started {
        prompt_tokens: usize,
        cached_tokens: usize,
    },
    Tokens(Vec<i32>),
    Done {
        finish: Finish,
        completion_tokens: usize,
    },
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
    Backpressure,
    InvalidImage,
}

#[derive(Clone)]
pub struct EventSender(Arc<EventChannel>);

struct EventChannel {
    sender: mpsc::Sender<Event>,
    terminal: Mutex<Option<mpsc::OwnedPermit<Event>>>,
    cancel: Arc<AtomicBool>,
    trace: crate::request_log::Trace,
}

#[derive(Debug)]
pub struct OutputUnavailable;

impl EventSender {
    pub(crate) fn channel(
        cancel: Arc<AtomicBool>,
        trace: crate::request_log::Trace,
    ) -> (Self, mpsc::Receiver<Event>) {
        Self::with_capacity(MAX_BUFFERED_EVENTS, cancel, trace)
    }

    fn with_capacity(
        capacity: usize,
        cancel: Arc<AtomicBool>,
        trace: crate::request_log::Trace,
    ) -> (Self, mpsc::Receiver<Event>) {
        let (sender, receiver) = mpsc::channel(capacity + 1);
        let terminal = sender
            .clone()
            .try_reserve_owned()
            .ok()
            .expect("terminal slot in an empty channel");
        let channel = EventChannel {
            sender,
            terminal: Mutex::new(Some(terminal)),
            cancel,
            trace,
        };
        (Self(Arc::new(channel)), receiver)
    }

    pub fn send(&self, event: Event) -> Result<(), OutputUnavailable> {
        let mut terminal = self.0.terminal.lock().unwrap();
        if terminal.is_none() {
            return Err(OutputUnavailable);
        }
        if let Event::Done {
            finish,
            completion_tokens,
        } = event
        {
            self.finish(terminal.take().unwrap(), finish, completion_tokens);
            return Ok(());
        }
        match self.0.sender.try_send(event) {
            Ok(()) => Ok(()),
            Err(err) => {
                self.0.cancel.store(true, Ordering::Relaxed);
                if matches!(err, mpsc::error::TrySendError::Full(_)) {
                    self.finish(terminal.take().unwrap(), Finish::Backpressure, 0);
                }
                Err(OutputUnavailable)
            }
        }
    }

    fn finish(&self, terminal: mpsc::OwnedPermit<Event>, finish: Finish, completion_tokens: usize) {
        self.0.trace.engine_finished(match finish {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::Cancelled => "cancelled",
            Finish::Backpressure | Finish::InvalidImage => "error",
        });
        match finish {
            Finish::Backpressure => self.0.trace.error("output_backpressure"),
            Finish::InvalidImage => self.0.trace.error("invalid_image"),
            _ => {}
        }
        terminal.send(Event::Done {
            finish,
            completion_tokens,
        });
    }
}

pub fn max_new_limit(n: usize, max_seq: usize) -> usize {
    max_seq
        .saturating_sub(n + 4 * MAX_M)
        .min(OUT_CAP - 4 * MAX_M)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_retirement_bytes_before_export_timing_is_known() {
        let (ingress, _requests) = JobIngress::channel(1, 1 << 20);
        ingress.load_reporter().set_scheduling_work(SchedulingWork {
            restore_ms: Some(2),
            retirement_bytes: Some(8192),
            ..SchedulingWork::default()
        });
        let status = serde_json::to_value(ingress.status()).unwrap();
        assert_eq!(status["work"]["restore_ms"], 2);
        assert_eq!(status["work"]["retirement_bytes"], 8192);
        assert!(status["work"]["retirement_ms"].is_null());
        assert!(status.get("kv_data_plane_failures").is_none());
    }

    fn job(tokens: Vec<i32>) -> (Job, mpsc::Receiver<Event>) {
        let trace = crate::request_log::Trace::default();
        let cancel = Arc::new(AtomicBool::new(false));
        let (events, receiver) = EventSender::with_capacity(1, cancel.clone(), trace.clone());
        (
            Job {
                trace,
                session: "session".into(),
                images: None,
                tokens: Arc::new(tokens),
                max_new: 8,
                seed: 0,
                stop_ids: Arc::new(vec![3]),
                cancel,
                events,
                admission: None,
            },
            receiver,
        )
    }

    #[test]
    fn outstanding_limit_counts_jobs_until_the_last_rank_releases_them() {
        let (ingress, mut requests) = JobIngress::channel(1, 1 << 20);
        ingress.submit(job(vec![1]).0).unwrap();
        let first = requests.try_recv().unwrap();
        let rank_copy = first.clone();
        first.done(Finish::Stop, 1);
        drop(first);
        assert_eq!(ingress.submit(job(vec![2]).0), Err(SubmitError::Full));
        drop(rank_copy);
        ingress.submit(job(vec![3]).0).unwrap();
        let active = requests.try_recv().unwrap();
        drop(requests);
        assert_eq!(ingress.submit(job(vec![4]).0), Err(SubmitError::Closed));
        drop(active);
    }

    #[test]
    fn byte_limit_includes_prompt_growth_and_releases_on_retirement() {
        let (first, _) = job(vec![1]);
        let bytes = first.retained_bytes();
        let (ingress, mut requests) = JobIngress::channel(2, bytes + size_of::<i32>());
        ingress.submit(first).unwrap();
        let mut active = requests.try_recv().unwrap();
        assert!(!active.reserve_prompt_growth(3));
        assert!(active.reserve_prompt_growth(2));
        active.tokens = Arc::new(vec![1, 2]);
        assert!(active.reserve_prompt_growth(2));
        assert_eq!(ingress.submit(job(vec![2]).0), Err(SubmitError::Full));
        drop(active);
        ingress.submit(job(vec![2]).0).unwrap();
        ingress.close();
        assert_eq!(ingress.submit(job(vec![3]).0), Err(SubmitError::Closed));
        assert!(requests.try_recv().is_ok());
        assert!(matches!(
            requests.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn image_job_keeps_its_spans_and_counts_against_the_byte_limit() {
        use deepseek_ultrafast_model::vision::{IMAGE_TOKEN, Image};

        let image = Image {
            start: 1,
            height: 3,
            width: 3,
            digest: [42; 32],
            patches: vec![0; 9 * 588].into(),
        };
        let mut tokens = vec![1];
        tokens.extend(std::iter::repeat_n(IMAGE_TOKEN, image.len()));
        let (mut job, _) = job(tokens);
        let permit = Arc::new(tokio::sync::Semaphore::new(1))
            .try_acquire_owned()
            .unwrap();
        let end = image.end();
        job.images = Some(Arc::new(crate::image_identity::Images::new(
            vec![image],
            permit,
        )));
        let bytes = job.retained_bytes();
        let (small, _small_receiver) = JobIngress::channel(1, bytes - 1);
        assert_eq!(small.submit(job.clone()), Err(SubmitError::Full));
        let (ingress, mut receiver) = JobIngress::channel(1, bytes);
        ingress.submit(job).unwrap();
        let accepted = receiver.try_recv().unwrap();
        let span = &accepted.spans()[0];
        assert_eq!((span.start, span.end, span.digest), (1, end, [42; 32]));
    }

    #[test]
    fn full_output_buffer_delivers_only_one_terminal_failure() {
        let (job, mut receiver) = job(vec![1]);
        assert!(job.events.send(Event::Tokens(vec![8])).is_ok());
        assert!(job.events.send(Event::Tokens(vec![9])).is_err());
        assert!(job.cancel.load(Ordering::Relaxed));
        job.done(Finish::Stop, 2);
        assert!(matches!(receiver.try_recv(), Ok(Event::Tokens(ids)) if ids == [8]));
        assert!(matches!(
            receiver.try_recv(),
            Ok(Event::Done {
                finish: Finish::Backpressure,
                ..
            })
        ));
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn full_output_buffer_can_finish_normally() {
        let (job, mut receiver) = job(vec![1]);
        assert!(job.events.send(Event::Tokens(vec![9])).is_ok());
        job.done(Finish::Length, 1);
        assert!(matches!(receiver.try_recv(), Ok(Event::Tokens(ids)) if ids == [9]));
        assert!(matches!(
            receiver.try_recv(),
            Ok(Event::Done {
                finish: Finish::Length,
                completion_tokens: 1
            })
        ));
        assert!(!job.cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn output_budget_reserves_speculative_headroom() {
        let context = 1048576;
        let headroom = 4 * MAX_M;
        assert_eq!(max_new_limit(1, context), 256000);
        let full_budget_prompt = context - 256000 - headroom;
        assert_eq!(max_new_limit(full_budget_prompt, context), 256000);
        assert_eq!(max_new_limit(full_budget_prompt + 1, context), 255999);
        assert_eq!(max_new_limit(4096, 8192), 4096 - headroom);
        assert_eq!(max_new_limit(context - headroom - 1, context), 1);
        assert_eq!(max_new_limit(context - headroom, context), 0);
        assert_eq!(max_new_limit(context, context), 0);
        assert_eq!(max_new_limit(context + 1, context), 0);
    }
}
