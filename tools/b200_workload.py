#!/usr/bin/env python3
"""Prepare immutable replay inputs and deterministic parity prompts for B200 tests."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

from bench_coding import load_sessions, nonnegative_seconds, percentiles, task_id


MAX_CONTEXT = 131072
MAX_OUTPUT = 8192
LOOKAHEAD = 64
PARITY_DECODE = 64
MODEL = "deepseek-v4.1-flash"
SMOKE_SOURCE = "synthetic_smoke"


def bounded_int(minimum, maximum):
    def parse(value):
        number = int(value)
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(f"expected an integer in [{minimum}, {maximum}]")
        return number
    return parse


def validate_standard_trace(sessions, max_seq):
    caps, prompt_counts, gaps, arrivals = [], [], [], []
    for session_index, session in enumerate(sessions, 1):
        previous_arrival = 0
        for turn_index, turn in enumerate(session["turns"], 1):
            where = f"session {session_index}, turn {turn_index}"
            if "gap_s" not in turn or "at_s" not in turn:
                raise ValueError(f"{where} needs both gap_s and at_s from the captured trace")
            gap = nonnegative_seconds(turn["gap_s"])
            arrival = nonnegative_seconds(turn["at_s"])
            if arrival < previous_arrival:
                raise ValueError(f"{where} arrives before the previous turn in its branch")
            previous_arrival = arrival
            gaps.append(gap)
            arrivals.append(arrival)
            body = turn["body"]
            messages = body.get("messages")
            if (not isinstance(messages, list) or not messages
                    or any(not isinstance(message, dict) for message in messages)):
                raise ValueError(f"{where} needs full-history Chat Completions messages")
            for message in messages:
                if isinstance(parts := message.get("content"), list) and any(
                    not isinstance(part, dict) or part.get("type") not in ("text", "refusal") for part in parts
                ):
                    raise ValueError(f"{where} uses content outside the text-only backend comparison")
            limit = body.get("max_completion_tokens")
            if limit is None:
                limit = body.get("max_tokens")
            if type(limit) is not int or not 1 <= limit <= MAX_OUTPUT or limit + LOOKAHEAD >= max_seq:
                raise ValueError(f"{where} needs an explicit positive output cap <= {MAX_OUTPUT} that fits --max-seq")
            caps.append(limit)
            if "prompt_tokens" in turn:
                count = turn["prompt_tokens"]
                if type(count) is not int or count < 1:
                    raise ValueError(f"{where} has an invalid prompt_tokens count")
                if count + limit + LOOKAHEAD > max_seq:
                    raise ValueError(f"{where} prompt_tokens + output cap + {LOOKAHEAD} exceed --max-seq")
                prompt_counts.append(count)
    per_task = Counter(task_id(session) for session in sessions)
    arrivals.sort()
    return {
        "tasks": len(per_task),
        "branches": len(sessions),
        "tasks_with_multiple_branches": sum(count > 1 for count in per_task.values()),
        "model_calls": len(caps),
        "explicit_output_cap_tokens": percentiles(caps),
        "trace_prompt_tokens": percentiles(prompt_counts),
        "calls_without_trace_prompt_token_count": len(caps) - len(prompt_counts),
        "context_limit_tokens": max_seq,
        "gap_s": percentiles(gaps),
        "arrival_span_s": arrivals[-1] - arrivals[0],
        "peak_offered_calls_per_s": {f"over_{window}s": peak_rate(arrivals, window) for window in (1, 10, 60)},
    }


def peak_rate(ordered, window):
    first = peak = 0
    for index, time in enumerate(ordered):
        while time - ordered[first] >= window:
            first += 1
        peak = max(peak, index + 1 - first)
    return round(peak / window, 3)


def kickoff_trace(sessions, count):
    roots = {}
    for session in sessions:
        first = session["turns"][0]
        if first.get("depends_on"):
            continue
        task = task_id(session)
        if task not in roots or first["at_s"] < roots[task]["turns"][0]["at_s"]:
            roots[task] = session
    ordered = sorted(roots.values(), key=lambda session: session["turns"][0]["at_s"])
    result = []
    for session in ordered[:count]:
        first = session["turns"][0]
        turn = {**first, "at_s": 0, "gap_s": 0, "cohort": "kickoff"}
        turn.pop("depends_on", None)
        result.append({**session, "turns": [turn]})
    if not result:
        raise ValueError("no independent first turns are available for the kickoff burst")
    return result


def synthetic_trace(tasks):
    tool = {"type": "function", "function": {
        "name": "read_file", "description": "Reads a starter repo file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }}

    def body(messages):
        return {"model": MODEL, "messages": messages, "tools": [tool], "max_completion_tokens": 256,
                "thinking": {"type": "enabled"}}

    sessions = []
    for index in range(tasks):
        task = f"synthetic-{index:04d}"
        main, child = f"{task}-main", f"{task}-review"
        first, review = f"{main}-read", f"{child}-inspect"
        start = round(index * .025, 3)
        messages = [{"role": "system", "content": "You are reviewing a small Python starter repository."},
                    {"role": "user", "content": f"Task {index}: implement inclusive ranges, with an example and tests."}]
        call = {"id": "call_range", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"src/ranges.py"}'}}
        history = [*messages, {"role": "assistant", "content": None, "tool_calls": [call]}, {
            "role": "tool", "tool_call_id": "call_range", "content": "def inclusive(first, last):\n    return range(first, last)\n",
        }]
        branch = index % 10 == 0
        followup = {"id": f"{main}-answer", "gap_s": 1.25,
                    "at_s": round(start + (6.5 if branch else 3.75), 3), "body": body(history)}
        if branch:
            followup["depends_on"] = [review]
        sessions.append({"id": main, "task_id": task, "source": SMOKE_SOURCE, "turns": [
            {"id": first, "gap_s": start, "at_s": start, "cohort": "cold", "body": body(messages)}, followup,
        ]})
        if branch:
            brief = [{"role": "user", "content": f"Task {index}: review tests for inclusive Python ranges."}]
            sessions.append({"id": child, "task_id": task, "source": SMOKE_SOURCE, "turns": [
                {"id": review, "depends_on": [first], "gap_s": .75, "at_s": round(start + 2.5, 3), "body": body(brief)},
            ]})
    return sessions


def json_file(path, value):
    with path.open("w") as out:
        json.dump(value, out, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        out.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_replay(args):
    captured = args.command == "prepare"
    source = "captured" if captured else SMOKE_SOURCE
    requested_kickoff = args.kickoff_sessions if captured else args.tasks
    args.output.mkdir(parents=True)
    try:
        mixed = args.output / "mixed.json"
        if captured:
            shutil.copyfile(args.trace, mixed)
        else:
            json_file(mixed, {"source": source, "sessions": synthetic_trace(args.tasks)})
        sessions = load_sessions(mixed, "open", None)
        if captured and any(session.get("source") == SMOKE_SOURCE for session in sessions):
            raise ValueError("synthetic smoke fixtures cannot be staged as captured traces")
        report = validate_standard_trace(sessions, args.max_seq)
        kickoff = kickoff_trace(sessions, requested_kickoff)
        report.update(source=source, kickoff_requests=len(kickoff), requested_kickoff_requests=requested_kickoff,
                      context_validation="Checks supplied prompt token counts. The native renderer is authoritative.")
        kickoff_path = args.output / "kickoff.json"
        json_file(kickoff_path, {"source": source, "sessions": kickoff})
        load_sessions(kickoff_path, "open", None)
        report.update(mixed_sha256=sha256(mixed), kickoff_sha256=sha256(kickoff_path))
        json_file(args.output / "workload.json", report)
    except BaseException:
        shutil.rmtree(args.output)
        raise
    print(json.dumps(report, indent=2))


PARITY_LENGTHS = (
    127, 128, 129, 8191, 8192, 8318, 8319, 8320, 511, 512, 513, 1023,
    1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8193, 8321, 9000,
    10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536, MAX_CONTEXT - PARITY_DECODE,
)


def parity_prompts(tokenizer, max_seq):
    snippets = (
        "def validate_range(first: int, last: int) -> None:\n    assert first <= last\n",
        "func readRecord(code int) (string, error) {\n    return lookup(code)\n}\n",
        "const results = await Promise.all(states.map(async (state) => state.load()));\n",
        "SELECT account_id, SUM(total) FROM invoices GROUP BY account_id ORDER BY 2 DESC;\n",
        "fn matches_stop(&self, next: &str) -> bool { self.values.iter().any(|s| next.starts_with(s)) }\n",
        "class Worker {\n  async run(job) {\n    return await job.finish();\n  }\n}\n",
        "# Given the following logs, explain the timeout and write a regression test.\n",
        "memcpy(buffer + offset, source, length);\nif (end < offset) { return EINVAL; }\n",
    )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids
    prompts = []
    for index, length in enumerate(PARITY_LENGTHS):
        length = min(length, max_seq - PARITY_DECODE)
        head = encode(f"Code review fixture {index}. Inspect the following source and return a patch:\n")
        unit = encode(snippets[index % len(snippets)])
        if not head or not unit:
            raise ValueError("checkpoint tokenizer produced an empty parity segment")
        prompt = head + unit * math.ceil(max(0, length - len(head)) / len(unit))
        prompts.append(prompt[:length])
    return prompts


def prepare_parity(args):
    path = args.checkpoint / "tokenizer.json"
    if not path.is_file():
        raise ValueError("checkpoint directory must contain tokenizer.json")
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ValueError("parity generation needs Python tokenizers (python3 -m pip install tokenizers)") from error
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as error:
        raise ValueError("unable to load checkpoint tokenizer.json") from error
    prompts = parity_prompts(tokenizer, args.max_seq)
    args.output.mkdir(parents=True)
    try:
        for width in (2, 8, 32):
            json_file(args.output / f"parity-{width}.json", prompts[:width])
        json_file(args.output / "parity.json", {"source": "synthetic_parity", "tokenizer_sha256": sha256(path),
                  "max_seq": args.max_seq, "decode_tokens": PARITY_DECODE, "prompt_lengths": [len(p) for p in prompts]})
    except BaseException:
        shutil.rmtree(args.output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="stage a sanitized captured Chat Completions manifest")
    prepare.add_argument("trace", type=Path)
    prepare.add_argument("--kickoff-tasks", "--kickoff-sessions", dest="kickoff_sessions",
                         type=bounded_int(1, 10000), default=500)
    smoke = commands.add_parser("smoke", help="stage a deterministic synthetic smoke fixture")
    smoke.add_argument("--tasks", "--sessions", type=bounded_int(1, 10000), default=500)
    parity = commands.add_parser("parity", help="tokenize isolated-versus-packed probe inputs")
    parity.add_argument("--checkpoint", type=Path, required=True)
    for command in (prepare, smoke, parity):
        command.add_argument("--output", type=Path, required=True, help="new output directory")
        command.add_argument("--max-seq", type=bounded_int(128, MAX_CONTEXT), default=MAX_CONTEXT)
    args = parser.parse_args()
    try:
        if args.command == "parity":
            prepare_parity(args)
        else:
            prepare_replay(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
