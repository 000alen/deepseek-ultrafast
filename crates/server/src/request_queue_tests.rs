use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct Job {
    id: usize,
    cancel: Arc<AtomicBool>,
    discarded: Arc<Mutex<Vec<usize>>>,
}
impl Request for Job {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
    fn discard(self) {
        self.discarded.lock().unwrap().push(self.id);
    }
}
fn job(id: usize) -> Job {
    Job {
        id,
        cancel: Default::default(),
        discarded: Default::default(),
    }
}
fn ready(q: &Queue<Job, usize>, id: usize) {
    q.push(job(id));
    let j = q.prepare().unwrap();
    assert_eq!(j.id, id);
    q.prepared(j, id);
}
fn next(q: &Queue<Job, usize>) -> usize {
    let (j, payload) = q.pop().unwrap();
    assert_eq!(j.id, payload);
    j.id
}

#[test]
fn fifo_admission_does_not_preempt_active() {
    let q = Queue::default();
    ready(&q, 0);
    let active = q.pop().unwrap();
    ready(&q, 1);
    ready(&q, 2);
    ready(&q, 3);
    assert_eq!(active.0.id, 0);
    drop(active);
    assert_eq!([next(&q), next(&q), next(&q)], [1, 2, 3]);
    q.close();
    assert!(q.pop().is_none());
    assert!(q.prepare().is_none());
}

#[test]
fn cancellation_prunes_pending_ready_and_completed_transfers_once() {
    let q: Queue<Job, usize> = Queue::default();
    let j = job(1);
    let cancelled = j.cancel.clone();
    let discarded = j.discarded.clone();
    q.push(j);
    cancelled.store(true, Ordering::Relaxed);
    let j = job(2);
    let cancel_ready = j.cancel.clone();
    let discarded_ready = j.discarded.clone();
    q.push(j);
    let j = q.prepare().unwrap();
    assert_eq!(j.id, 2);
    q.prepared(j, 2);
    cancel_ready.store(true, Ordering::Relaxed);
    let j = job(3);
    let cancel_fetching = j.cancel.clone();
    let discarded_fetching = j.discarded.clone();
    q.push(j);
    let j = q.prepare().unwrap();
    assert_eq!(j.id, 3);
    cancel_fetching.store(true, Ordering::Relaxed);
    q.prepared(j, 3);
    q.close();
    assert!(q.pop().is_none());
    assert!(q.prepare().is_none());
    assert_eq!(*discarded.lock().unwrap(), [1]);
    assert_eq!(*discarded_ready.lock().unwrap(), [2]);
    assert_eq!(*discarded_fetching.lock().unwrap(), [3]);
}

#[test]
fn preparation_skips_blocked_sessions_and_polling_stays_nonblocking() {
    let q = Queue::default();
    q.push(job(1));
    q.push(job(2));
    assert!(q.try_pop().is_none());

    let unrelated = q.prepare_if(|job| job.id != 1).unwrap();
    assert_eq!(unrelated.id, 2);
    q.prepared(unrelated, 2);
    assert_eq!(q.try_pop().unwrap().0.id, 2);
    assert!(q.try_pop().is_none());

    let formerly_blocked = q.prepare().unwrap();
    assert_eq!(formerly_blocked.id, 1);
    q.prepared(formerly_blocked, 1);
    q.close();
    assert_eq!(next(&q), 1);
    assert!(q.pop().is_none());
}

#[test]
fn unrelated_preparation_can_finish_while_an_earlier_job_is_in_flight() {
    let q = Queue::default();
    q.push(job(1));
    q.push(job(2));
    let slow = q.prepare().unwrap();
    q.close();
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel();
        let worker_queue = &q;
        scope.spawn(move || {
            if let Some(fast) = worker_queue.prepare() {
                let id = fast.id;
                worker_queue.prepared(fast, id);
                tx.send(id).unwrap();
            }
        });
        // Release the first job before checking the assertion so a regression
        // to serialized preparation cannot strand the scoped worker.
        let fast = rx.recv_timeout(std::time::Duration::from_secs(1));
        q.prepared(slow, 1);
        assert_eq!(fast.unwrap(), 2);
    });
    assert_eq!([next(&q), next(&q)], [2, 1]);
    assert!(q.pop().is_none());
}

#[test]
fn in_flight_and_ready_jobs_share_the_preparation_budget() {
    let q = Queue::default();
    for id in 1..MAX_PREPARED_REQUESTS {
        ready(&q, id);
    }
    q.push(job(0));
    let slow = q.prepare().unwrap();
    q.push(job(MAX_PREPARED_REQUESTS));
    q.close();
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel();
        let worker_queue = &q;
        scope.spawn(move || {
            let extra = worker_queue.prepare().unwrap();
            let id = extra.id;
            worker_queue.prepared(extra, id);
            tx.send(id).unwrap();
        });
        let blocked = rx.recv_timeout(std::time::Duration::from_millis(50));
        assert_eq!(next(&q), 1);
        let extra = rx.recv_timeout(std::time::Duration::from_secs(1));
        q.prepared(slow, 0);
        assert_eq!(blocked, Err(std::sync::mpsc::RecvTimeoutError::Timeout));
        assert_eq!(extra.unwrap(), MAX_PREPARED_REQUESTS);
    });
}
