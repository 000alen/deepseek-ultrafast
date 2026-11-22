use anyhow::{Result, ensure};
use deepseek_ultrafast_engine::kvstore::{ExportCapture, ExportSlot, PendingExport};
use deepseek_ultrafast_engine::model::Rank;
use deepseek_ultrafast_kv_transfer::{ImageSpan, PeerCache, Revision, Snapshot};
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, Weak, mpsc};
use std::time::Duration;

pub type Fence = Arc<Mutex<Option<Revision>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capture {
    Prefill,
    Complete(usize),
}
const PROMPT_REWIND: usize = 256;

impl Capture {
    fn input_tokens(self, total: usize) -> Result<usize> {
        let generated = match self {
            Self::Prefill => 0,
            Self::Complete(n) => n,
        };
        total
            .checked_sub(generated)
            .filter(|&n| n > 0)
            .ok_or_else(|| anyhow::anyhow!("invalid capture input boundary"))
    }
    pub(super) fn tail(self) -> usize {
        let generated = match self {
            Self::Prefill => 0,
            Self::Complete(n) => n,
        };
        generated
            .saturating_add(deepseek_ultrafast_kv_transfer::TAIL_READ + PROMPT_REWIND)
            .min(deepseek_ultrafast_kv_transfer::SAVED_TAIL)
    }
}

fn take_slot<T>(slots: &mut Vec<T>, phase: Capture) -> Option<T> {
    if phase == Capture::Prefill && slots.len() < 2 {
        None
    } else {
        slots.pop()
    }
}

struct FenceRef {
    origin: Option<Revision>,
    fence: Weak<Mutex<Option<Revision>>>,
}
#[derive(Default)]
struct Fences(HashMap<String, FenceRef>);
impl Fences {
    fn begin(&mut self, session: &str, observed: Option<Revision>, local: bool) -> Fence {
        self.0.retain(|_, entry| entry.fence.strong_count() > 0);
        if let Some(observed) = observed
            && let Some(fence) = self
                .0
                .get(session)
                .filter(|entry| {
                    // An early follow-up can still see the exact revision that
                    // the earlier turn is publishing from, before it sees a local owner.
                    entry.origin == Some(observed)
                        || (local && entry.origin.is_some_and(|r| r.epoch == observed.epoch))
                })
                .and_then(|entry| entry.fence.upgrade())
        {
            let current = fence
                .lock()
                .unwrap()
                .is_some_and(|r| r.epoch == observed.epoch && r.sequence >= observed.sequence);
            if current {
                return fence;
            }
        }
        let fence = Arc::new(Mutex::new(observed));
        self.0.insert(
            session.into(),
            FenceRef {
                origin: observed,
                fence: Arc::downgrade(&fence),
            },
        );
        fence
    }
}

#[derive(Default)]
pub struct Pending {
    sessions: Mutex<HashMap<String, usize>>,
    done: Condvar,
}
impl Pending {
    fn add(&self, session: &str) {
        *self
            .sessions
            .lock()
            .unwrap()
            .entry(session.into())
            .or_default() += 1;
    }
    fn finish(&self, session: &str) {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(n) = sessions.get_mut(session) {
            *n -= 1;
            if *n == 0 {
                sessions.remove(session);
            }
        }
        self.done.notify_all();
    }
    pub fn wait(&self, session: &str, timeout: Duration) -> bool {
        let sessions = self.sessions.lock().unwrap();
        let (sessions, _) = self
            .done
            .wait_timeout_while(sessions, timeout, |s| s.contains_key(session))
            .unwrap();
        !sessions.contains_key(session)
    }
}

struct Publication {
    session: String,
    session_id: Option<Arc<str>>,
    revision: Fence,
    input_tokens: usize,
}

struct Work<Image> {
    publication: Publication,
    image: Image,
}

enum CaptureFailure {
    Snapshot(anyhow::Error),
    Device(anyhow::Error),
}

fn drain<Image>(
    recv: mpsc::Receiver<Work<Image>>,
    pending: &Pending,
    finish: impl Fn(Image) -> std::result::Result<Arc<Snapshot>, CaptureFailure>,
    mut publish: impl FnMut(&str, Revision, Arc<Snapshot>, usize, Option<Arc<str>>) -> Result<Revision>,
) -> Result<()> {
    while let Ok(Work { publication, image }) = recv.recv() {
        let result = match finish(image) {
            Ok(snapshot) => {
                // This worker is the only writer to a fence. Admission may read
                // its current revision while a directory claim is in flight.
                let expected = *publication.revision.lock().unwrap();
                if let Some(expected) = expected {
                    let revision = match publish(
                        &publication.session,
                        expected,
                        snapshot,
                        publication.input_tokens,
                        publication.session_id,
                    ) {
                        Ok(next) => Some(next),
                        Err(error) => {
                            // A failed announcement may have advanced Redis ownership.
                            // The next request must look up a fresh revision.
                            deepseek_ultrafast_logging::kv_failure("publication");
                            deepseek_ultrafast_logging::diagnostic!(
                                "KV background publication fenced: {error:#}"
                            );
                            None
                        }
                    };
                    *publication.revision.lock().unwrap() = revision;
                }
                Ok(())
            }
            Err(CaptureFailure::Snapshot(error)) => {
                deepseek_ultrafast_logging::kv_failure("export_payload");
                deepseek_ultrafast_logging::diagnostic!(
                    "KV background capture discarded: {error:#}"
                );
                Ok(())
            }
            Err(CaptureFailure::Device(error)) => Err(error),
        };
        pending.finish(&publication.session);
        result?;
    }
    Ok(())
}

/// Holds a capture slot between scheduler turns. Its GPU source must remain
/// immutable and leased until the capture completes or this value is dropped.
pub struct CaptureProgress {
    image: Option<ExportCapture>,
    publication: Option<Publication>,
    slots: Arc<Mutex<Vec<ExportSlot>>>,
}

impl CaptureProgress {
    pub fn remaining_bytes(&self) -> usize {
        self.image.as_ref().map_or(0, ExportCapture::remaining_bytes)
    }
}

impl Drop for CaptureProgress {
    fn drop(&mut self) {
        if let Some(image) = self.image.take() {
            match image.abort() {
                Ok(slot) => self.slots.lock().unwrap().push(slot),
                Err(error) => deepseek_ultrafast_logging::diagnostic!(
                    "KV capture cancellation could not drain staging: {error:#}"
                ),
            }
        }
    }
}

pub struct Publisher {
    slots: Arc<Mutex<Vec<ExportSlot>>>,
    send: Option<mpsc::SyncSender<Work<PendingExport>>>,
    worker: Option<std::thread::JoinHandle<()>>,
    fingerprint: String,
    fences: Fences,
    pending: Arc<Pending>,
}
impl Publisher {
    pub fn new(rank: &Rank, peer: Arc<PeerCache>, runtime: tokio::runtime::Handle) -> Result<Self> {
        let slots = Arc::new(Mutex::new(vec![
            ExportSlot::new(rank, peer.max_bytes())?,
            ExportSlot::new(rank, peer.max_bytes())?,
        ]));
        let fingerprint = peer.fingerprint().to_string();
        let (send, recv) = mpsc::sync_channel::<Work<PendingExport>>(2);
        let pending = Arc::new(Pending::default());
        let drained = pending.clone();
        let returned = slots.clone();
        let worker = std::thread::Builder::new()
            .name("kv-publisher".into())
            .spawn(move || {
                if let Err(e) = drain(
                    recv,
                    &drained,
                    |image| {
                        let (snapshot, slot) =
                            image.finish_reusable().map_err(CaptureFailure::Device)?;
                        returned.lock().unwrap().push(slot);
                        snapshot.map(Arc::new).map_err(CaptureFailure::Snapshot)
                    },
                    |session, revision, snapshot, input_tokens, session_id| {
                        runtime.block_on(peer.publish_with_input_named(
                            session,
                            revision,
                            snapshot,
                            input_tokens,
                            session_id,
                        ))
                    },
                ) {
                    deepseek_ultrafast_logging::termination(&format!("kv publisher: {e:#}"));
                    deepseek_ultrafast_logging::flush_fatal();
                    std::process::exit(1);
                }
            })?;
        Ok(Self {
            slots,
            send: Some(send),
            worker: Some(worker),
            fingerprint,
            fences: Fences::default(),
            pending,
        })
    }
    pub fn pending(&self) -> Arc<Pending> {
        self.pending.clone()
    }
    pub fn fence(&mut self, session: &str, observed: Option<Revision>, local: bool) -> Fence {
        self.fences.begin(session, observed, local)
    }
    pub fn capture(
        &self,
        rank: &Rank,
        session: &str,
        session_id: Option<Arc<str>>,
        tokens: &[i32],
        images: &[ImageSpan],
        revision: Fence,
        phase: Capture,
    ) -> Result<()> {
        let input_tokens = phase.input_tokens(tokens.len())?;
        let Some(slot) = self.slot_for_capture(rank, tokens.len(), phase) else {
            return Ok(());
        };
        let image = slot.capture(rank, tokens, images, &self.fingerprint, phase.tail())?;
        self.submit(
            image,
            Publication {
                session: session.into(),
                session_id,
                revision,
                input_tokens,
            },
        )
    }

    pub fn begin_capture(
        &self,
        rank: &Rank,
        session: &str,
        session_id: Option<Arc<str>>,
        tokens: &[i32],
        images: &[ImageSpan],
        revision: Fence,
        phase: Capture,
    ) -> Result<Option<CaptureProgress>> {
        let input_tokens = phase.input_tokens(tokens.len())?;
        let Some(slot) = self.slot_for_capture(rank, tokens.len(), phase) else {
            return Ok(None);
        };
        let image = slot.begin_capture(rank, tokens, images, &self.fingerprint, phase.tail())?;
        Ok(Some(CaptureProgress {
            image: Some(image),
            publication: Some(Publication {
                session: session.into(),
                session_id,
                revision,
                input_tokens,
            }),
            slots: self.slots.clone(),
        }))
    }

    pub fn step_capture_async(
        &self,
        rank: &Rank,
        capture: &mut CaptureProgress,
        decoded_byte_budget: usize,
    ) -> Result<bool> {
        ensure!(
            Arc::ptr_eq(&capture.slots, &self.slots),
            "KV capture belongs to a different publisher"
        );
        capture
            .image
            .as_mut()
            .expect("capture has an image")
            .step_async(rank, decoded_byte_budget)
    }

    pub fn finish_capture(&self, mut capture: CaptureProgress) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&capture.slots, &self.slots),
            "KV capture belongs to a different publisher"
        );
        ensure!(
            capture
                .image
                .as_ref()
                .is_some_and(ExportCapture::is_complete),
            "KV capture is incomplete"
        );
        let image = capture
            .image
            .take()
            .expect("capture has an image")
            .finish()?;
        let publication = capture
            .publication
            .take()
            .expect("capture has publication metadata");
        self.submit(image, publication)
    }

    fn slot_for_capture(&self, rank: &Rank, n: usize, phase: Capture) -> Option<ExportSlot> {
        let slot = take_slot(&mut self.slots.lock().unwrap(), phase)?;
        if slot.fits(rank, n, phase.tail()) {
            return Some(slot);
        }
        self.slots.lock().unwrap().push(slot);
        None
    }

    fn submit(&self, image: PendingExport, publication: Publication) -> Result<()> {
        let session = publication.session.clone();
        self.pending.add(&session);
        if let Err(error) = self
            .send
            .as_ref()
            .unwrap()
            .try_send(Work { publication, image })
        {
            self.pending.finish(&session);
            deepseek_ultrafast_logging::kv_failure("publication_queue");
            deepseek_ultrafast_logging::diagnostic!("KV publication queue unavailable: {error}");
            let work = match error {
                mpsc::TrySendError::Full(work) | mpsc::TrySendError::Disconnected(work) => work,
            };
            let slot = work.image.abort()?;
            self.slots.lock().unwrap().push(slot);
        }
        Ok(())
    }
}
impl Drop for Publisher {
    fn drop(&mut self) {
        self.send.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod pending_tests {
    use super::*;

    fn revision(sequence: u64) -> Revision {
        Revision {
            epoch: "00000000-0000-0000-0000-000000000001".parse().unwrap(),
            sequence,
        }
    }

    fn publication(session: &str, fence: &Fence) -> Publication {
        Publication {
            session: session.into(),
            session_id: None,
            revision: fence.clone(),
            input_tokens: 1,
        }
    }

    #[test]
    fn early_follow_up_reuses_the_unannounced_origin() {
        let pending = Pending::default();
        let mut fences = Fences::default();
        let start = revision(0);
        let first = fences.begin("branch", Some(start), false);
        let snapshot = Arc::new(Snapshot::new(
            "test".into(),
            vec![1],
            0,
            [0, 1, 1],
            vec![1],
            vec![0].into(),
        ));
        let (send, recv) = mpsc::channel();
        pending.add("branch");
        assert!(
            send.send(Work {
                publication: publication("branch", &first),
                image: snapshot.clone(),
            })
            .is_ok()
        );

        let mut send = Some(send);
        let mut attempts = Vec::new();
        drain(recv, &pending, Ok, |session, expected, _, _, _| {
            if expected == start {
                let follow_up = fences.begin(session, Some(start), false);
                assert!(Arc::ptr_eq(&first, &follow_up));
                pending.add(session);
                assert!(
                    send.take()
                        .unwrap()
                        .send(Work {
                            publication: publication(session, &follow_up),
                            image: snapshot.clone(),
                        })
                        .is_ok()
                );
            }
            attempts.push(expected);
            Ok(revision(expected.sequence + 1))
        })
        .unwrap();
        assert_eq!(attempts, [start, revision(1)]);
        assert_eq!(*first.lock().unwrap(), Some(revision(2)));
        assert!(pending.wait("branch", Duration::ZERO));
    }

    #[test]
    fn an_invalid_or_new_remote_origin_starts_a_separate_fence() {
        let mut fences = Fences::default();
        let first = fences.begin("branch", Some(revision(0)), false);
        *first.lock().unwrap() = None;
        let retried = fences.begin("branch", Some(revision(0)), false);
        assert!(!Arc::ptr_eq(&first, &retried));
        let remote = fences.begin("branch", Some(revision(1)), false);
        assert!(!Arc::ptr_eq(&retried, &remote));
    }

    #[test]
    fn a_bad_snapshot_or_failed_announcement_does_not_stop_next_publication() {
        let pending = Pending::default();
        let alpha = Arc::new(Mutex::new(Some(revision(1))));
        let beta = Arc::new(Mutex::new(Some(revision(1))));
        let snapshot = Arc::new(Snapshot::new(
            "test".into(),
            vec![1],
            0,
            [0, 1, 1],
            vec![1],
            vec![0].into(),
        ));
        let (send, recv) = mpsc::sync_channel(4);
        for (publication, image) in [
            (
                publication("alpha", &alpha),
                Err(CaptureFailure::Snapshot(anyhow::anyhow!(
                    "invalid FP4 payload"
                ))),
            ),
            (publication("beta", &beta), Ok(snapshot.clone())),
            (publication("beta", &beta), Ok(snapshot.clone())),
            (publication("alpha", &alpha), Ok(snapshot)),
        ] {
            pending.add(&publication.session);
            assert!(send.try_send(Work { publication, image }).is_ok());
        }
        drop(send);

        let announcements = std::cell::RefCell::new(Vec::new());
        drain(
            recv,
            &pending,
            |image| image,
            |session, expected, _, _, _| {
                announcements.borrow_mut().push(session.to_string());
                assert_eq!(expected, revision(1));
                if session == "beta" {
                    return Err(deepseek_ultrafast_kv_transfer::RdmaRequired(
                        "registration failed".into(),
                    )
                    .into());
                }
                Ok(revision(2))
            },
        )
        .unwrap();

        assert_eq!(announcements.into_inner(), vec!["beta", "alpha"]);
        assert_eq!(*alpha.lock().unwrap(), Some(revision(2)));
        assert_eq!(*beta.lock().unwrap(), None);
        assert!(pending.wait("alpha", Duration::ZERO));
        assert!(pending.wait("beta", Duration::ZERO));
    }

    #[test]
    fn device_capture_failure_is_fatal_and_releases_its_waiter() {
        let pending = Pending::default();
        let fence = Arc::new(Mutex::new(Some(revision(1))));
        let (send, recv) = mpsc::sync_channel(1);
        pending.add("alpha");
        assert!(
            send.try_send(Work {
                publication: publication("alpha", &fence),
                image: ()
            })
            .is_ok()
        );
        drop(send);

        let error = drain(
            recv,
            &pending,
            |()| Err(CaptureFailure::Device(anyhow::anyhow!("CUDA device fault"))),
            |_, _, _, _, _| panic!("failed capture cannot be published"),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "CUDA device fault");
        assert!(pending.wait("alpha", Duration::ZERO));
    }

    #[test]
    fn waits_for_its_own_session_only() {
        let pending = Arc::new(Pending::default());
        assert!(pending.wait("a", Duration::ZERO));
        pending.add("a");
        pending.add("a");
        assert!(
            pending.wait("b", Duration::ZERO),
            "another session's publication"
        );
        assert!(
            !pending.wait("a", Duration::from_millis(20)),
            "in flight: the wait times out"
        );
        let p = pending.clone();
        let t = std::thread::spawn(move || {
            p.finish("a");
            std::thread::sleep(Duration::from_millis(20));
            p.finish("a");
        });
        assert!(
            pending.wait("a", Duration::from_secs(5)),
            "both publications finished"
        );
        t.join().unwrap();
    }
}
