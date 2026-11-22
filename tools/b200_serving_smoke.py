#!/usr/bin/env python3
"""Check cancellation and prefix reuse against native Chat Completions SSE."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
from threading import Event
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from bench_coding import NoRedirect, endpoint_identity, nonnegative_seconds, sse_data, usage_count


def unix_ms():
    return time.time_ns() // 1_000_000


def user(text):
    return {"role": "user", "content": text}


def body(model, messages, max_tokens):
    return {
        "model": model,
        "messages": messages,
        "seed": 42,
        "thinking": {"type": "disabled"},
        "max_completion_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@dataclass
class Stream:
    session_id: str
    request_id: str = field(default_factory=lambda: uuid4().hex)
    first_or_done: Event = field(default_factory=Event)
    started_unix_ms: int | None = None
    first_token_unix_ms: int | None = None
    cancel_unix_ms: int | None = None
    finished_unix_ms: int | None = None
    completion_id: str | None = None
    status: int | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    parts: list[str] = field(default_factory=list, repr=False)

    def text(self):
        return "".join(self.parts)

    def report(self):
        data = {key: value for key, value in vars(self).items()
                if key not in ("parts", "first_or_done")}
        text = self.text()
        data["output_sha256"] = hashlib.sha256(text.encode()).hexdigest() if text else None
        data["content_chunks"] = len(self.parts)
        return data


def read_stream(url, request_body, result, api_key, timeout, cancel=None, on_first=None):
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Session-ID": result.session_id,
        "X-Request-ID": result.request_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    result.started_unix_ms = unix_ms()
    deadline = time.monotonic() + timeout
    saw_done = False
    try:
        opener = build_opener(ProxyHandler({}), NoRedirect())
        req = Request(url, data=json.dumps(request_body).encode(), headers=headers, method="POST")
        with opener.open(req, timeout=timeout) as response:
            result.status = response.status
            if response.headers.get_content_type() != "text/event-stream":
                result.error = "expected_sse"
                return
            if response.headers.get("X-Session-ID") != result.session_id:
                result.error = "session_id_mismatch"
                return
            for data in sse_data(response):
                if time.monotonic() > deadline:
                    result.error = "deadline"
                    break
                if data == b"[DONE]":
                    saw_done = True
                    break
                payload = json.loads(data)
                if payload.get("error"):
                    result.error = "stream_error"
                    break
                if result.completion_id is None:
                    result.completion_id = payload.get("id")
                for choice in payload.get("choices") or ():
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        result.finish_reason = reason
                    content = (choice.get("delta") or {}).get("content")
                    if isinstance(content, str) and content:
                        result.parts.append(content)
                        if result.first_token_unix_ms is None:
                            result.first_token_unix_ms = unix_ms()
                            result.first_or_done.set()
                            if on_first is not None:
                                on_first()
                if result.finish_reason not in (None, "stop", "length"):
                    result.error = "unexpected_finish_reason"
                    break
                if (cancel is not None and cancel.is_set() and result.parts
                        and result.finish_reason is None):
                    result.cancel_unix_ms = unix_ms()
                    break
                if usage := payload.get("usage"):
                    result.prompt_tokens = usage_count(usage, "prompt_tokens")
                    result.completion_tokens = usage_count(usage, "completion_tokens")
                    result.cached_tokens = usage_count(usage.get("prompt_tokens_details") or {},
                                                       "cached_tokens")
        if result.error is None and result.cancel_unix_ms is None:
            if not (saw_done and result.finish_reason and result.parts):
                result.error = "incomplete_stream"
            elif (result.prompt_tokens is None or result.completion_tokens is None
                  or result.cached_tokens is None):
                result.error = "missing_usage"
            elif result.cached_tokens > result.prompt_tokens:
                result.error = "invalid_cache_usage"
    except HTTPError as error:
        result.status = error.code
        result.error = f"http_{error.code}"
        error.close()
    except (TimeoutError, URLError, OSError, HTTPException) as error:
        cause = error.reason if isinstance(error, URLError) else error
        result.error = "timeout" if isinstance(cause, TimeoutError) else "transport_error"
    except (ValueError, AttributeError, TypeError):
        result.error = "invalid_response"
    finally:
        result.finished_unix_ms = unix_ms()
        result.first_or_done.set()


def overlap(args, namespace):
    url = args.overlap_url or args.url
    a = Stream(f"{namespace}-overlap-a")
    b = Stream(f"{namespace}-overlap-b")
    c = Stream(f"{namespace}-overlap-c")
    cancel = Event()
    long_reply = user("Output consecutive numbered lines from 1 through 10000. "
                      "After each number write: the example compiler has three passes. "
                      "Begin with line 1 now, and continue the pattern without abbreviation.")
    ongoing = body(args.model, [long_reply], 8192)
    companion = body(args.model, [long_reply], 256)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a_job = pool.submit(read_stream, url, ongoing, a, args.api_key, args.timeout_s, cancel)
        if a.first_or_done.wait(args.timeout_s) and a.first_token_unix_ms is not None:
            b_job = pool.submit(read_stream, url, companion, b, args.api_key, args.timeout_s,
                                on_first=cancel.set)
            b_job.result()
        cancel.set()
        a_job.result()
    read_stream(url, body(args.model, [user("Reply briefly with the name of your language.")], 32),
                c, args.api_key, args.timeout_s)
    checks = {
        "started_second_after_first_token": (
            a.first_token_unix_ms is not None and b.started_unix_ms is not None
            and a.first_token_unix_ms <= b.started_unix_ms
        ),
        "cancelled_first_after_second_started": (
            a.error is None and a.cancel_unix_ms is not None and b.first_token_unix_ms is not None
            and b.first_token_unix_ms <= a.cancel_unix_ms
        ),
        "second_completed_across_cancellation": (
            b.error is None and b.finished_unix_ms is not None and a.finished_unix_ms is not None
            and a.cancel_unix_ms is not None and a.finished_unix_ms <= b.finished_unix_ms
        ),
        "new_request_completed": c.error is None,
    }
    return {"endpoint": endpoint_identity(url), "checks": checks,
            "requests": {name: stream.report() for name, stream in (("a", a), ("b", b), ("c", c))}}


def continuation(args, namespace):
    branch = f"{namespace}-continuation"
    first = Stream(branch)
    warm = Stream(branch)
    cold = Stream(f"{namespace}-continuation-cold")
    continuation_url = args.continuation_url or args.url
    # A long, ordinary text prefix makes a warm hit distinguishable from a template-only match.
    source = "".join(f"def check_case_{i}(value):\n    return value + {i}\n" for i in range(96))
    messages = [user(f"Here is the fixture source:\n{source}\n"
                     "Describe this fixture in two plain-text sentences.")]
    read_stream(args.url, body(args.model, messages, 96), first, args.api_key, args.timeout_s)
    if first.error is None:
        if args.publication_wait_s:
            time.sleep(args.publication_wait_s)
        follow_up = body(args.model, [*messages, {"role": "assistant", "content": first.text()},
                                    user("Summarize what the preceding explanation says in one sentence.")],
                         96)
        read_stream(continuation_url, follow_up, warm, args.api_key, args.timeout_s)
        read_stream(continuation_url, follow_up, cold, args.api_key, args.timeout_s)
    checks = {
        "first_completed_cold": first.error is None and first.cached_tokens == 0,
        "same_branch_reused_initial_prompt": (
            warm.error is None and warm.cached_tokens is not None and first.prompt_tokens is not None
            and warm.cached_tokens >= first.prompt_tokens > 0
        ),
        "independent_branch_completed_cold": cold.error is None and cold.cached_tokens == 0,
    }
    requests = {name: stream.report()
                for name, stream in (("first", first), ("warm", warm), ("cold", cold))}
    warm_hash = requests["warm"]["output_sha256"]
    cold_hash = requests["cold"]["output_sha256"]
    return {
        "initial_endpoint": endpoint_identity(args.url),
        "continuation_endpoint": endpoint_identity(continuation_url),
        "publication_wait_s": args.publication_wait_s,
        "checks": checks,
        "warm_and_cold_text_match": warm_hash == cold_hash if warm_hash and cold_hash else None,
        "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--overlap-url", help="direct native endpoint to force the same TP replica")
    parser.add_argument("--continuation-url", help="direct peer endpoint for a forced migration")
    parser.add_argument("--publication-wait-s", type=float, default=0)
    parser.add_argument("--timeout-s", type=float, default=45)
    parser.add_argument("--scenario", choices=("all", "overlap", "continuation"), default="all")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--report", type=Path, help="write timing, statuses and digests to this JSON file")
    args = parser.parse_args()
    try:
        for url in (args.url, args.overlap_url, args.continuation_url):
            if url is not None:
                endpoint = endpoint_identity(url)
                parts = urlsplit(url)
                if (parts.query or parts.fragment
                        or not endpoint["path"].endswith("/chat/completions")):
                    raise ValueError("each URL must address Chat Completions without a query or fragment")
        nonnegative_seconds(args.publication_wait_s)
        if not nonnegative_seconds(args.timeout_s):
            raise ValueError("timeout must be positive")
    except ValueError as error:
        parser.error(str(error))
    args.api_key = os.environ.get(args.api_key_env)
    namespace = "b200-smoke-" + uuid4().hex
    report = {"schema_version": 1, "namespace": namespace, "seed": 42, "scenarios": {}}
    for name, fn in (("overlap", overlap), ("continuation", continuation)):
        if args.scenario in ("all", name):
            report["scenarios"][name] = fn(args, namespace)
    report["passed"] = all(all(result["checks"].values()) for result in report["scenarios"].values())
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
