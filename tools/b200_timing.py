#!/usr/bin/env python3
"""Summarize opt-in packed plan traces from a B200 scheduler test."""

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path


MAX_PLANS = 200_000
PEAK_HISTORY = 32
STAGES = {
    "decode": "decode_ms",
    "restore": "restore_ms",
    "retirement": "retirement_ms",
    "retirement_anchor": "retirement_anchor_ms",
    "retirement_export": "retirement_export_ms",
}
MEASUREMENTS = (
    "decode_ms", "decode_forward_ms", "decode_tokens", "restore_ms", "restore_bytes",
    "retirement_ms", "retirement_anchor_ms", "retirement_export_ms", "export_bytes",
    "busy_prefill_budget_ms", "busy_prefill_rows", "active_decoders", "active_decoder_limit",
    "prefill_backlog_requests", "decode_window_ms", "peak_wall_ms", "non_prefill_peak_ms",
)
LIMITER_COUNTS = (
    "decode_plans_with_prefill_backlog", "backlogged_decode_plans_with_budget",
    "decode_plans_with_window", "decode_plans_over_window", "decode_plans_over_window_with_retirement_anchor",
    "zero_busy_budget_plans", "zero_busy_row_cap_plans", "restricted_plans",
    "restricted_plans_with_one_encoder_row", "restricted_stretches",
    "active_limit_decreases", "active_limit_increases",
    "restricted_peak_with_retirement_anchor", "restricted_peak_without_retirement_anchor",
    "restricted_peak_not_observed", "active_limit_decrease_peak_with_retirement_anchor",
    "active_limit_decrease_peak_without_retirement_anchor", "active_limit_decrease_peak_not_observed",
)


def nonnegative(value):
    return ((type(value) is int and value >= 0)
            or (type(value) is float and math.isfinite(value) and value >= 0))


def log_event(line):
    # Docker log exports may wrap the event or prepend their own timestamp.
    stamp, _, payload = line.partition(" ")
    if stamp[:4].isdigit() and "T" in stamp:
        line = payload
    try:
        item = json.loads(line)
        if isinstance(item, dict) and isinstance(item.get("log"), str):
            item = json.loads(item["log"])
    except ValueError:
        return None
    return item if isinstance(item, dict) else None


def is_plan(event):
    return (type(event.get("plan_id")) is int and event["plan_id"] > 0
            and type(event.get("decode_lanes")) is int and event["decode_lanes"] >= 0
            and nonnegative(event.get("wall_ms")))


def update_stages(stages, event):
    for stage, name in STAGES.items():
        if nonnegative(event.get(name)):
            stages[stage] += 1
    for kind in ("encode", "replay"):
        rows = event.get(f"prefill_{kind}_rows", 0)
        if nonnegative(rows) and rows > 0:
            stages[f"prefill_{kind}"] += 1


def add_measurements(samples, event):
    wall_key = "wall_ms" if event["decode_lanes"] else "wall_ms_including_idle_wait"
    samples[wall_key].append(event["wall_ms"])
    for name in MEASUREMENTS:
        value = event.get(name)
        if nonnegative(value):
            samples[name].append(value)
    for kind in ("encode", "replay"):
        rows = event.get(f"prefill_{kind}_rows", 0)
        if nonnegative(rows) and rows > 0:
            samples[f"prefill_{kind}_rows"].append(rows)
            if nonnegative(event.get("prefill_ms")):
                samples[f"prefill_{kind}_ms"].append(event["prefill_ms"])


def quantiles(values):
    values.sort()
    n = len(values)
    return {"n": n, **{f"p{p}": round(values[(n * p + 99) // 100 - 1], 4)
                        for p in (50, 95, 99)}}


@dataclass
class Source:
    index: int
    plans: int = 0
    invalid_plans: int = 0
    processes: int = 0
    plan_gaps: int = 0
    last_plan: int | None = None
    previous_limit: int | None = None
    peaks: OrderedDict = field(default_factory=OrderedDict)
    limiter: Counter = field(default_factory=Counter)
    streak: int = 0
    streak_ms: float = 0
    longest_streak: int = 0
    longest_streak_ms: float = 0

    def end_process(self):
        self.last_plan = None
        self.previous_limit = None
        self.peaks.clear()
        self.end_streak()

    def end_streak(self):
        self.streak = 0
        self.streak_ms = 0

    def attribute(self, event, label, peak_key):
        peak = event.get(peak_key)
        if type(peak) is not int:
            return
        if peak not in self.peaks:
            self.limiter[f"{label}_peak_not_observed"] += 1
        elif self.peaks[peak]:
            self.limiter[f"{label}_peak_with_retirement_anchor"] += 1
        else:
            self.limiter[f"{label}_peak_without_retirement_anchor"] += 1

    def add(self, event):
        plan = event["plan_id"]
        if self.last_plan is not None and plan <= self.last_plan:
            self.end_process()
        if self.last_plan is None:
            self.processes += 1
        elif plan != self.last_plan + 1:
            self.plan_gaps += 1
            self.previous_limit = None
            self.end_streak()
        self.last_plan = plan
        self.plans += 1

        limit = event.get("active_decoder_limit")
        if type(limit) is int and limit >= 0:
            if self.previous_limit is not None and limit < self.previous_limit:
                self.limiter["active_limit_decreases"] += 1
                self.attribute(event, "active_limit_decrease", "peak_plan_id")
            elif self.previous_limit is not None and limit > self.previous_limit:
                self.limiter["active_limit_increases"] += 1
            self.previous_limit = limit
        else:
            self.previous_limit = None

        backlog = event.get("prefill_backlog_requests")
        budget = event.get("busy_prefill_budget_ms")
        rows = event.get("busy_prefill_rows")
        zero_budget = nonnegative(budget) and budget == 0
        zero_rows = nonnegative(rows) and rows == 0
        busy = event["decode_lanes"] > 0 and nonnegative(backlog) and backlog > 0
        if busy:
            self.limiter["decode_plans_with_prefill_backlog"] += 1
            if nonnegative(budget) and nonnegative(rows):
                self.limiter["backlogged_decode_plans_with_budget"] += 1
            if zero_budget:
                self.limiter["zero_busy_budget_plans"] += 1
            if zero_rows:
                self.limiter["zero_busy_row_cap_plans"] += 1
        if busy and (zero_budget or zero_rows):
            self.limiter["restricted_plans"] += 1
            if event.get("prefill_encode_rows") == 1:
                self.limiter["restricted_plans_with_one_encoder_row"] += 1
            self.attribute(event, "restricted", "non_prefill_peak_plan_id")
            if self.streak == 0:
                self.limiter["restricted_stretches"] += 1
            self.streak += 1
            self.streak_ms += event["wall_ms"]
            self.longest_streak = max(self.longest_streak, self.streak)
            self.longest_streak_ms = max(self.longest_streak_ms, self.streak_ms)
        else:
            self.end_streak()

        if event["decode_lanes"]:
            has_anchor = nonnegative(event.get("retirement_anchor_ms"))
            window = event.get("decode_window_ms")
            if nonnegative(window):
                self.limiter["decode_plans_with_window"] += 1
                if event["wall_ms"] > window:
                    self.limiter["decode_plans_over_window"] += 1
                    if has_anchor:
                        self.limiter["decode_plans_over_window_with_retirement_anchor"] += 1
            self.peaks[plan] = has_anchor
            if len(self.peaks) > PEAK_HISTORY:
                self.peaks.popitem(last=False)

    def report(self):
        return {"source_index": self.index, "plans": self.plans, "invalid_plans": self.invalid_plans,
                "processes": self.processes, "plan_gaps": self.plan_gaps,
                "limiter": {**self.limiter,
                            "longest_consecutive_restricted_plans": self.longest_streak,
                            "longest_consecutive_restricted_wall_ms": round(self.longest_streak_ms, 4)}}


def log_files(paths):
    files = dict()
    for path in paths:
        if path.is_dir():
            candidates = sorted(child for child in path.iterdir()
                                if child.is_file() and child.suffix in (".log", ".jsonl", ".ndjson"))
        elif path.is_file():
            candidates = [path]
        else:
            raise ValueError("each log path must be a file or directory")
        for candidate in candidates:
            files[candidate.resolve()] = None
    return files


def summarize(paths, max_plans=MAX_PLANS):
    if max_plans <= 0:
        raise ValueError("max_plans must be positive")
    by_lanes = defaultdict(lambda: defaultdict(list))
    lane_counts = Counter()
    stages = Counter()
    limiter = Counter()
    sources = []
    plans = invalid_plans = 0
    limited = False
    for index, path in enumerate(log_files(paths), 1):
        source = Source(index)
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                event = log_event(line)
                if event is None:
                    continue
                if event.get("event") == "packed_decode_ready":
                    source.end_process()
                if event.get("event") != "packed_plan_step":
                    continue
                if not is_plan(event):
                    source.invalid_plans += 1
                    invalid_plans += 1
                    continue
                if plans >= max_plans:
                    limited = True
                    break
                source.add(event)
                plans += 1
                lanes = event["decode_lanes"]
                lane_counts[lanes] += 1
                update_stages(stages, event)
                add_measurements(by_lanes[lanes], event)
        sources.append(source.report())
        limiter.update(source.limiter)
        if limited:
            break
    if limited:
        status = "limited"
    elif plans:
        status = "ok"
    else:
        status = "no_traces"
    return {
        "schema_version": 1, "status": status, "plans": plans, "max_plans": max_plans,
        "invalid_plans": invalid_plans, "stage_plans": dict(stages), "sources": sources,
        "limiter": {
            **{name: limiter[name] for name in LIMITER_COUNTS},
            **{name: max((source["limiter"][name] for source in sources), default=0) for name in (
                "longest_consecutive_restricted_plans", "longest_consecutive_restricted_wall_ms")},
        },
        "by_decode_lanes": {str(lanes): {"plans": lane_counts[lanes],
                                        "metrics": {name: quantiles(values)
                                                    for name, values in by_lanes[lanes].items()}}
                            for lanes in sorted(lane_counts)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path,
                        help="log files or directories of .log, .jsonl and .ndjson files")
    parser.add_argument("--max-plans", type=int, default=MAX_PLANS,
                        help="stop after this many valid plan events across the input")
    args = parser.parse_args()
    try:
        report = summarize(args.logs, args.max_plans)
    except OSError:
        parser.error("unable to read a log path")
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
