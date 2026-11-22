use super::*;

fn limits() -> Limits {
    Limits {
        max_residents: 3,
        max_decode_seqs: 2,
        max_active_decoders: 3,
        decode_rows: 3,
        padded_decode_rows: None,
        max_target_rows: 10,
        max_replay_rows: 4,
        busy_prefill_rows: 2,
        busy_prefill_interval: 1,
        max_prefill_wait_ns: 50_000_000,
        idle_prefill_rows: 8,
        reservation_capacity: 64,
        max_bypass: 1,
    }
}

fn request(id: u64, prompt_tokens: usize, reservation: usize) -> Request {
    Request {
        id,
        prompt_tokens,
        reservation,
    }
}

fn result(handle: Handle, outcome: Outcome) -> StepResult {
    StepResult { handle, outcome }
}

fn done(plan: &StepPlan) -> Vec<StepResult> {
    plan.work()
        .map(|(handle, work)| {
            result(
                handle,
                match work {
                    Work::Admit => Outcome::Admitted { cached_tokens: 0 },
                    Work::Retire => Outcome::Retired,
                    _ => Outcome::Advanced,
                },
            )
        })
        .collect()
}

fn run(scheduler: &mut Scheduler) -> StepPlan {
    run_at(scheduler, 0, true)
}

fn run_at(scheduler: &mut Scheduler, now_ns: u128, busy_replay: bool) -> StepPlan {
    let plan = scheduler.next_plan(now_ns, busy_replay).unwrap().unwrap();
    let results = done(&plan);
    scheduler.complete(plan.id, &results).unwrap();
    plan
}

#[test]
fn active_decoder_limit_holds_replay_but_allows_other_prefills() {
    let mut scheduler = Scheduler::new(Limits {
        max_active_decoders: 1,
        ..limits()
    })
    .unwrap();
    let a = start_decoder(&mut scheduler, 1, 10);
    scheduler.enqueue(request(2, 1, 10)).unwrap();
    scheduler.enqueue(request(3, 1, 10)).unwrap();
    let b = run(&mut scheduler).admission.unwrap().handle;
    assert!(matches!(
        run(&mut scheduler).prefill,
        Some(Prefill::Encode { handle, .. }) if handle == b
    ));
    let plan = run(&mut scheduler);
    assert_eq!(plan.decode, [a]);
    assert!(matches!(plan.prefill, Some(Prefill::Encode { .. })));
    assert_eq!(scheduler.prefill_backlog(), (0, 2));
    assert_eq!(run(&mut scheduler).prefill, None);
    scheduler.cancel(1);
    let plan = run(&mut scheduler);
    assert_eq!(plan.retire, Some(a));
    assert_eq!(plan.prefill, Some(Prefill::Replay { handle: b, rows: 1 }));
    assert_eq!(scheduler.active_decoder_slots().collect::<Vec<_>>(), [b]);
}

#[test]
fn lowering_active_limit_does_not_suspend_running_decoders() {
    let mut scheduler = Scheduler::new(limits()).unwrap();
    let a = start_decoder(&mut scheduler, 1, 10);
    let b = start_decoder(&mut scheduler, 2, 10);
    scheduler.set_active_decoder_limit(1);
    assert_eq!(run(&mut scheduler).decode, [a, b]);
    assert_eq!(scheduler.active_decoder_limit(), 1);
}

#[test]
fn a_rejected_restore_retires_and_releases_capacity_while_others_decode() {
    let mut scheduler = Scheduler::new(limits()).unwrap();
    let decoding = start_decoder(&mut scheduler, 1, 10);
    scheduler.enqueue(request(2, 4, 12)).unwrap();
    let plan = scheduler.next_plan(0, true).unwrap().unwrap();
    let rejected = plan.admission.unwrap().handle;
    scheduler
        .complete(
            plan.id,
            &[
                result(decoding, Outcome::Advanced),
                result(
                    rejected,
                    Outcome::RestorePending {
                        cached_tokens: 0,
                        prompt_tokens: 4,
                    },
                ),
            ],
        )
        .unwrap();
    let plan = scheduler.next_plan(0, true).unwrap().unwrap();
    assert_eq!(plan.restore, Some(rejected));
    scheduler
        .complete(
            plan.id,
            &[
                result(decoding, Outcome::Advanced),
                result(rejected, Outcome::Finished),
            ],
        )
        .unwrap();
    assert_eq!(scheduler.reserved(), 22);
    assert_eq!(scheduler.retiring_slots().collect::<Vec<_>>(), [rejected]);
    let plan = run(&mut scheduler);
    assert_eq!(plan.decode, [decoding]);
    assert_eq!(plan.retire, Some(rejected));
    assert_eq!(scheduler.reserved(), 10);
    assert!(scheduler.retiring_slots().next().is_none());
}

#[test]
fn busy_prefill_observes_cadence_but_aged_work_makes_progress() {
    let mut scheduler = Scheduler::new(Limits {
        busy_prefill_interval: 3,
        max_prefill_wait_ns: 100,
        ..limits()
    })
    .unwrap();
    let a = start_decoder(&mut scheduler, 1, 10);
    scheduler.enqueue(request(2, 4, 10)).unwrap();
    let b = run(&mut scheduler).admission.unwrap().handle;
    assert_eq!(
        run(&mut scheduler).prefill,
        Some(Prefill::Encode {
            handle: b,
            start: 0,
            end: 2
        })
    );
    assert_eq!(run_at(&mut scheduler, 30, false).prefill, None);
    assert_eq!(run_at(&mut scheduler, 90, false).prefill, None);
    scheduler.set_busy_prefill_rows(0);
    let aged = run_at(&mut scheduler, 100, false);
    assert_eq!(aged.decode, [a]);
    assert_eq!(
        aged.prefill,
        Some(Prefill::Encode {
            handle: b,
            start: 2,
            end: 3
        })
    );
    assert_eq!(
        run_at(&mut scheduler, 200, false).prefill,
        Some(Prefill::Encode {
            handle: b,
            start: 3,
            end: 4
        })
    );
    assert_eq!(run_at(&mut scheduler, 299, false).prefill, None);
    assert_eq!(
        run_at(&mut scheduler, 300, false).prefill,
        Some(Prefill::Replay { handle: b, rows: 4 })
    );
}

#[test]
fn preferred_idle_slot_preserves_queue_order() {
    let mut scheduler = Scheduler::new(limits()).unwrap();
    scheduler.enqueue(request(1, 1, 10)).unwrap();
    scheduler.enqueue(request(2, 1, 10)).unwrap();
    let plan = scheduler
        .next_plan_with_slot(0, true, |id, slot| {
            assert_eq!(id, 1);
            u64::from(slot != 2)
        })
        .unwrap()
        .unwrap();
    let admitted = plan.admission.unwrap();
    assert_eq!(admitted.request_id, 1);
    assert_eq!(admitted.handle.slot, 2);
    scheduler.complete(plan.id, &done(&plan)).unwrap();
    let fallback = run(&mut scheduler).admission.unwrap();
    assert_eq!(fallback.request_id, 2);
    assert_eq!(fallback.handle.slot, 0);
}

fn start_decoder(scheduler: &mut Scheduler, id: u64, reservation: usize) -> Handle {
    scheduler.enqueue(request(id, 1, reservation)).unwrap();
    let handle = run(scheduler).admission.unwrap().handle;
    assert_eq!(
        run(scheduler).prefill,
        Some(Prefill::Encode {
            handle,
            start: 0,
            end: 1
        })
    );
    assert_eq!(
        run(scheduler).prefill,
        Some(Prefill::Replay { handle, rows: 1 })
    );
    handle
}

#[test]
fn restores_and_exports_yield_while_another_request_decodes() {
    let mut scheduler = Scheduler::new(limits()).unwrap();
    let a = start_decoder(&mut scheduler, 1, 10);
    scheduler.enqueue(request(2, 7, 15)).unwrap();
    let step = scheduler.next_plan(0, true).unwrap().unwrap();
    let b = step.admission.unwrap().handle;
    scheduler
        .complete(
            step.id,
            &[
                result(a, Outcome::Advanced),
                result(
                    b,
                    Outcome::RestorePending {
                        cached_tokens: 2,
                        prompt_tokens: 6,
                    },
                ),
            ],
        )
        .unwrap();
    for _ in 0..3 {
        let step = run(&mut scheduler);
        assert_eq!(step.decode, [a]);
        assert_eq!(step.restore, Some(b));
        assert!(step.prefill.is_none());
    }
    let step = scheduler.next_plan(0, true).unwrap().unwrap();
    scheduler
        .complete(
            step.id,
            &[result(a, Outcome::Advanced), result(b, Outcome::Restored)],
        )
        .unwrap();
    scheduler.set_busy_prefill_rows(1);
    assert_eq!(
        run(&mut scheduler).prefill,
        Some(Prefill::Encode {
            handle: b,
            start: 2,
            end: 3
        })
    );
    scheduler.cancel(2);
    let step = scheduler.next_plan(0, true).unwrap().unwrap();
    assert_eq!(step.retire, Some(b));
    scheduler
        .complete(
            step.id,
            &[result(a, Outcome::Advanced), result(b, Outcome::Advanced)],
        )
        .unwrap();
    assert_eq!(scheduler.reserved(), 25);
    assert_eq!(run(&mut scheduler).retire, Some(b));
    assert_eq!(scheduler.reserved(), 10);
}

#[test]
fn late_arrival_prefills_then_decodes_with_the_running_request() {
    let mut scheduler = Scheduler::new(Limits {
        max_residents: 2,
        max_active_decoders: 2,
        ..limits()
    })
    .unwrap();
    let a = start_decoder(&mut scheduler, 1, 10);

    scheduler.enqueue(request(2, 7, 10)).unwrap();
    let plan = scheduler.next_plan(0, true).unwrap().unwrap();
    assert_eq!(plan.decode, [a]);
    let b = plan.admission.unwrap().handle;
    scheduler
        .complete(
            plan.id,
            &[
                result(a, Outcome::Advanced),
                result(b, Outcome::Admitted { cached_tokens: 3 }),
            ],
        )
        .unwrap();
    for (start, end) in [(3, 5), (5, 7)] {
        let step = run(&mut scheduler);
        assert_eq!(step.decode, [a]);
        assert_eq!(
            step.prefill,
            Some(Prefill::Encode {
                handle: b,
                start,
                end
            })
        );
        assert_eq!(step.target_rows, limits().decode_rows + end - start);
    }
    let replay = run(&mut scheduler);
    assert_eq!(replay.decode, [a]);
    assert_eq!(replay.prefill, Some(Prefill::Replay { handle: b, rows: 4 }));

    let packed = scheduler.next_plan(0, true).unwrap().unwrap();
    assert_eq!(packed.decode, [a, b]);
    assert_eq!(
        scheduler.complete(
            packed.id,
            &[result(b, Outcome::Advanced), result(b, Outcome::Advanced)]
        ),
        Err(ScheduleError::InvalidResults(packed.id))
    );
    scheduler
        .complete(
            packed.id,
            &[result(b, Outcome::Advanced), result(a, Outcome::Finished)],
        )
        .unwrap();
    assert_eq!(scheduler.cancel(1), None);
    scheduler.enqueue(request(3, 3, 10)).unwrap();
    let retirement = scheduler.next_plan(0, true).unwrap().unwrap();
    assert_eq!(retirement.decode, [b]);
    assert_eq!(retirement.retire, Some(a));
    assert_eq!(retirement.admission, None);
    assert_eq!(scheduler.reserved(), 20);
    scheduler
        .complete(retirement.id, &done(&retirement))
        .unwrap();
    assert_eq!(scheduler.reserved(), 10);

    let replacement = run(&mut scheduler);
    assert_eq!(replacement.decode, [b]);
    let c = replacement.admission.unwrap().handle;
    assert_eq!(c.slot, a.slot);
    assert_ne!(c.generation, a.generation);
}

#[test]
fn padded_decode_consumes_the_execution_width_before_prefill() {
    let mut scheduler = Scheduler::new(Limits {
        padded_decode_rows: Some(8),
        max_target_rows: 12,
        busy_prefill_rows: 8,
        ..limits()
    })
    .unwrap();
    let decoder = start_decoder(&mut scheduler, 1, 10);

    scheduler.enqueue(request(2, 12, 10)).unwrap();
    let admission = run(&mut scheduler).admission.unwrap().handle;
    let step = run(&mut scheduler);
    assert_eq!(step.decode, [decoder]);
    assert_eq!(
        step.prefill,
        Some(Prefill::Encode {
            handle: admission,
            start: 0,
            end: 4,
        })
    );
    assert_eq!(step.target_rows, 12);
}

#[test]
fn prefill_and_decode_rotate_under_a_packed_width_cap() {
    let mut scheduler = Scheduler::new(Limits {
        max_decode_seqs: 1,
        ..limits()
    })
    .unwrap();
    let a = start_decoder(&mut scheduler, 1, 10);
    scheduler.enqueue(request(2, 6, 10)).unwrap();
    scheduler.enqueue(request(3, 6, 10)).unwrap();
    let b = run(&mut scheduler).admission.unwrap().handle;
    let c = run(&mut scheduler).admission.unwrap().handle;
    let mut first = Vec::new();
    for _ in 0..4 {
        let plan = run(&mut scheduler);
        assert_eq!(plan.decode, [a]);
        let Some(Prefill::Encode { handle, start, end }) = plan.prefill else {
            panic!("expected prefill")
        };
        assert_eq!(end - start, 2);
        first.push(handle);
    }
    assert_eq!(first, [b, c, b, c]);
    for _ in 0..3 {
        run(&mut scheduler);
    }
    let served: Vec<_> = (0..3).map(|_| run(&mut scheduler).decode[0]).collect();
    assert_eq!(served, [a, b, c]);
}

#[test]
fn invalid_or_stale_results_cannot_change_a_reused_slot() {
    let mut scheduler = Scheduler::new(Limits {
        max_residents: 1,
        max_decode_seqs: 1,
        max_active_decoders: 1,
        ..limits()
    })
    .unwrap();
    let a = start_decoder(&mut scheduler, 1, 20);
    let decoding = scheduler.next_plan(0, true).unwrap().unwrap();
    assert_eq!(scheduler.cancel(1), Some(Cancelled::Resident(a)));
    assert_eq!(scheduler.cancel(1), None);
    assert_eq!(
        scheduler.next_plan(0, true).unwrap_err(),
        ScheduleError::PlanInFlight(decoding.id)
    );
    scheduler.complete(decoding.id, &done(&decoding)).unwrap();
    assert_eq!(scheduler.reserved(), 20);
    let retirement = run(&mut scheduler);
    assert_eq!(retirement.retire, Some(a));
    assert_eq!(scheduler.reserved(), 0);

    scheduler.enqueue(request(2, 2, 9)).unwrap();
    let current = scheduler.next_plan(0, true).unwrap().unwrap();
    let b = current.admission.unwrap().handle;
    assert_eq!(b.slot, a.slot);
    assert_eq!(
        scheduler.complete(retirement.id, &done(&retirement)),
        Err(ScheduleError::StalePlan(retirement.id))
    );
    assert_eq!(
        scheduler.complete(
            current.id,
            &[result(a, Outcome::Admitted { cached_tokens: 0 })]
        ),
        Err(ScheduleError::InvalidResults(current.id))
    );
    assert_eq!(
        scheduler.complete(
            current.id,
            &[result(b, Outcome::Admitted { cached_tokens: 2 })]
        ),
        Err(ScheduleError::InvalidResults(current.id))
    );
    assert_eq!(scheduler.reserved(), 9);
    scheduler.complete(current.id, &done(&current)).unwrap();
    assert_eq!(
        run(&mut scheduler).prefill,
        Some(Prefill::Encode {
            handle: b,
            start: 0,
            end: 2
        })
    );
}

#[test]
fn one_bypass_preserves_progress_for_a_request_using_the_whole_pool() {
    let mut scheduler = Scheduler::new(Limits {
        reservation_capacity: 10,
        ..limits()
    })
    .unwrap();
    let a = start_decoder(&mut scheduler, 1, 6);
    scheduler.enqueue(request(2, 2, 10)).unwrap();
    scheduler.enqueue(request(3, 2, 4)).unwrap();
    scheduler.enqueue(request(4, 2, 4)).unwrap();
    let admitted = run(&mut scheduler).admission.unwrap();
    assert_eq!(admitted.request_id, 3);
    assert_eq!(scheduler.reserved(), 10);
    assert_eq!(
        scheduler.cancel(3),
        Some(Cancelled::Resident(admitted.handle))
    );
    assert_eq!(run(&mut scheduler).retire, Some(admitted.handle));
    assert_eq!(scheduler.reserved(), 6);
    assert!(run(&mut scheduler).admission.is_none());
    assert_eq!(scheduler.cancel(1), Some(Cancelled::Resident(a)));
    assert_eq!(run(&mut scheduler).retire, Some(a));
    assert_eq!(run(&mut scheduler).admission.unwrap().request_id, 2);
    assert_eq!(scheduler.reserved(), 10);
    assert_eq!(scheduler.cancel(4), Some(Cancelled::Ready));
    assert_eq!(scheduler.cancel(4), None);
    assert_eq!(scheduler.ready_len(), 0);
    assert_eq!(scheduler.resident_len(), 1);
}

#[test]
fn rejects_invalid_budgets_and_caps_idle_prefill_rows() {
    let invalid = Limits {
        max_target_rows: 9,
        ..limits()
    };
    assert!(matches!(
        Scheduler::new(invalid),
        Err(ScheduleError::InvalidLimits)
    ));
    let mut scheduler = Scheduler::new(Limits {
        idle_prefill_rows: 100,
        ..limits()
    })
    .unwrap();
    assert!(scheduler.is_empty());
    assert!(scheduler.next_plan(0, true).unwrap().is_none());
    assert_eq!(
        scheduler.enqueue(request(1, 10, 65)),
        Err(ScheduleError::InvalidRequest(1))
    );
    scheduler.enqueue(request(1, 20, 30)).unwrap();
    assert_eq!(
        scheduler.enqueue(request(1, 20, 30)),
        Err(ScheduleError::DuplicateRequest(1))
    );
    let handle = run(&mut scheduler).admission.unwrap().handle;
    let plan = run(&mut scheduler);
    assert_eq!(plan.target_rows, 10);
    assert_eq!(
        plan.prefill,
        Some(Prefill::Encode {
            handle,
            start: 0,
            end: 10
        })
    );
}
