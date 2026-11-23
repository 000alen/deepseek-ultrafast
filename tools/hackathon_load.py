#!/usr/bin/env python3
"""Replay OpenAI chat requests at a fixed offered rate and summarize SSE timings."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import http.client
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import uuid


@dataclass
class Result:
    status: str
    dispatch_delay: float
    elapsed: float
    first_token: float | None = None
    first_reasoning: float | None = None
    first_answer: float | None = None
    last_token: float | None = None
    finish: str | None = None
    usage: dict = field(default_factory=dict)
    budget: int | None = None


def load_workload(path, model):
    requests = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            body = json.loads(line)
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
                raise ValueError("expected an object with a nonempty messages array")
            options = body.get("stream_options")
            if options is None:
                options = {}
            if not isinstance(options, dict):
                raise ValueError("stream_options must be an object")
            session = body.pop("_session_id", None)
            if session is not None and (
                not isinstance(session, str) or not session or not session.isascii()
                or not session.isprintable() or session != session.strip()
            ):
                raise ValueError("_session_id must be a nonblank printable ASCII string")
            body.update(model=model, stream=True, stream_options={**options, "include_usage": True})
            budget = body.get("max_completion_tokens", body.get("max_tokens"))
            requests.append((json.dumps(body, allow_nan=False).encode(), session, budget))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not requests:
        raise ValueError(f"{path}: workload is empty")
    return requests


def sse_events(response, sock, deadline):
    data = []
    size = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        sock.settimeout(remaining)
        line = response.readline(1024 * 1024 + 1)
        if not line:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError
        size += len(line)
        if size > 1024 * 1024:
            raise ValueError("SSE event exceeds 1 MiB")
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                yield b"\n".join(data)
            data.clear()
            size = 0
            continue
        name, _, value = line.partition(b":")
        if name == b"data":
            data.append(value.removeprefix(b" "))


def replay(target, scheduled, payload, session, budget, key, timeout):
    start = time.monotonic()
    result = Result("incomplete_stream", start - scheduled, 0, budget=budget)
    connection_type = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(target.hostname, target.port, timeout=timeout)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "X-Session-ID": session}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        connection.request("POST", target.path, body=payload, headers=headers)
        sock = connection.sock
        remaining = start + timeout - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        sock.settimeout(remaining)
        with connection.getresponse() as response:
            if response.status != 200:
                result.status = f"http_{response.status}"
            elif response.headers.get_content_type() != "text/event-stream":
                result.status = "invalid_content_type"
            else:
                for event in sse_events(response, sock, start + timeout):
                    if event == b"[DONE]":
                        if result.finish is not None:
                            result.status = "ok"
                        break
                    chunk = json.loads(event)
                    if chunk.get("error") is not None:
                        result.status = "sse_error"
                        break
                    usage = chunk.get("usage")
                    if usage is not None:
                        if not isinstance(usage, dict):
                            raise ValueError("usage must be an object")
                        result.usage = usage
                    now = time.monotonic() - start
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        answer = delta.get("content") or delta.get("refusal") or any(
                            call.get("id") or any((call.get("function") or {}).values())
                            for call in delta.get("tool_calls") or []
                        )
                        if reasoning and result.first_reasoning is None:
                            result.first_reasoning = now
                        if answer and result.first_answer is None:
                            result.first_answer = now
                        if reasoning or answer:
                            if result.first_token is None:
                                result.first_token = now
                            result.last_token = now
                        finish = choice.get("finish_reason")
                        if finish is not None:
                            if not isinstance(finish, str):
                                raise ValueError("finish_reason must be a string")
                            result.finish = finish
    except TimeoutError:
        result.status = "timeout"
    except (http.client.IncompleteRead, http.client.RemoteDisconnected):
        result.status = "incomplete_stream"
    except (OSError, http.client.HTTPException):
        result.status = "transport_error"
    except (AttributeError, KeyError, TypeError, ValueError):
        result.status = "invalid_stream"
    finally:
        connection.close()
        result.elapsed = time.monotonic() - start
    return result


def percentiles(values, scale=1000):
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return {"n": 0, "p50": None, "p95": None}
    return {
        "n": len(ordered),
        **{f"p{p}": round(scale * ordered[math.ceil(p * len(ordered) / 100) - 1], 3) for p in (50, 95)},
    }


def token_total(rows, *keys):
    counts = []
    for row in rows:
        data = row.usage
        for key in keys:
            data = data.get(key) if isinstance(data, dict) else None
        if type(data) is int and data >= 0:
            counts.append(data)
    return {"requests_reporting": len(counts), "tokens": sum(counts) if counts else None}


def summarize(results, elapsed, count, rate, burst):
    ok = [row for row in results if row.status == "ok"]
    completion = token_total(ok, "completion_tokens")
    outcomes = Counter(row.status for row in results)
    outcomes["client_admission_drop"] = count - len(results)
    return {
        "requests_offered": count,
        "offered_requests_per_second_after_burst": rate,
        "initial_burst": burst,
        "elapsed_s": round(elapsed, 3),
        "outcomes": dict(outcomes),
        "requests_with_answer_or_tool": sum(row.first_answer is not None for row in ok),
        "finish_reasons": dict(Counter(row.finish for row in ok)),
        "successful_requests_per_second": round(len(ok) / elapsed, 3),
        "output_tokens_per_second": round(completion["tokens"] / elapsed, 3) if completion["tokens"] is not None else None,
        "prompt_usage": token_total(ok, "prompt_tokens"),
        "cached_prompt_usage": token_total(ok, "prompt_tokens_details", "cached_tokens"),
        "completion_usage": completion,
        "reasoning_usage": token_total(ok, "completion_tokens_details", "reasoning_tokens"),
        "requested_output_budget_tokens": percentiles(
            (row.budget for row in results if type(row.budget) is int), scale=1
        ),
        "dispatch_delay_ms": percentiles(row.dispatch_delay for row in results),
        "ttft_ms": percentiles(row.first_token for row in ok),
        "first_reasoning_ms": percentiles(row.first_reasoning for row in ok),
        "first_answer_or_tool_ms": percentiles(row.first_answer for row in ok),
        "end_to_end_ms": percentiles(row.elapsed for row in ok),
        "per_request_tpot_ms": percentiles(
            (row.last_token - row.first_token) / (row.usage["completion_tokens"] - 1)
            for row in ok
            if row.last_token is not None and type(row.usage.get("completion_tokens")) is int
            and row.usage["completion_tokens"] > 1
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="HTTP(S) base URL, optionally ending in /v1")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--input", type=Path, required=True, help="JSONL of chat request bodies, cycled if needed")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--rate", type=float, default=5, help="scheduled requests per second, 0 for a full burst")
    parser.add_argument("--burst", type=int, default=1, help="requests scheduled immediately")
    parser.add_argument("--concurrency", type=int, default=128, help="drop new arrivals at this client-side limit")
    parser.add_argument("--timeout", type=float, default=120, help="request deadline in seconds")
    parser.add_argument("--key-env", default="OPENAI_API_KEY", help="environment variable containing optional bearer token")
    args = parser.parse_args()
    if args.requests < 1 or not 1 <= args.burst <= args.requests or args.concurrency < 1:
        parser.error("requests and concurrency must be positive and burst must be between 1 and requests")
    if not math.isfinite(args.rate) or args.rate < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("rate must be finite and nonnegative and timeout must be finite and positive")
    try:
        target = urlsplit(args.base_url)
        if (
            target.scheme not in ("https", "http") or not target.hostname
            or target.username is not None or target.query or target.fragment or target.port == 0
        ):
            raise ValueError("base URL must have an HTTP(S) origin and no credentials, query, or fragment")
        path = target.path.rstrip("/")
        target = target._replace(path=path + ("/chat/completions" if path.endswith("/v1") else "/v1/chat/completions"))
        workload = load_workload(args.input, args.model)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    burst = args.burst if args.rate else args.requests
    slots = threading.BoundedSemaphore(args.concurrency)
    futures = []
    run_id = uuid.uuid4().hex
    key = os.environ.get(args.key_env)
    with ThreadPoolExecutor(max_workers=min(args.concurrency, args.requests)) as pool:
        start = time.monotonic()
        for index in range(args.requests):
            scheduled = start if index < burst else start + (index - burst + 1) / args.rate
            delay = scheduled - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if not slots.acquire(blocking=False):
                continue
            payload, session, budget = workload[index % len(workload)]
            future = pool.submit(replay, target, scheduled, payload, session or f"{run_id}-{index}", budget, key, args.timeout)
            future.add_done_callback(lambda _: slots.release())
            futures.append(future)
        results = [future.result() for future in futures]
    elapsed = time.monotonic() - start
    print(json.dumps(summarize(results, elapsed, args.requests, args.rate, burst), indent=2))
    return 0 if len(results) == args.requests and all(row.status == "ok" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
