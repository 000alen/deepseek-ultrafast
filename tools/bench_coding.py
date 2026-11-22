#!/usr/bin/env python3
"""Replay a fixed multi-turn Chat Completions workload and report SSE timing.

The manifest contains a sessions array. Each session has a unique id and turns
containing a complete request body, an optional closed-loop gap_s, and an at_s
offset for open-loop replay. Turn bodies include the full conversation history.
Sessions can share a task_id while retaining distinct cache identities. Closed
replay observes optional turn IDs and depends_on edges as well as branch order.
Closed-loop replenishment replaces completed tasks with fresh instances of the
same trace corpus until a configured deadline, then drains the active tasks.
An optional turn cohort describes the intended mix. Qualification determines
warm and cold turns from the server's reported prompt and cached token counts.
"""

import argparse
from array import array
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from graphlib import CycleError, TopologicalSorter
import heapq
from http.client import HTTPException
from itertools import count, cycle, pairwise
import json
import math
import os
from pathlib import Path
import re
import sys
from threading import Timer
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

WARM_MAX_NEW = 8192
COLD_MAX_INPUT = 32768
WARM_TTFT_MS = 2000
COLD_TTFT_MS = 5000
GAP_MS = 250
MIN_DECODE_TOKENS = 128
MIN_DECODE_TOKENS_PER_S = 100
MIN_SUCCESS_RATE = 0.995
PRODUCTION_CACHE_SAMPLES = 20
MAX_PROBE_BYTES = 1 << 20


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def nonnegative_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timestamps and gaps must be nonnegative numbers")
    if value < 0 or not math.isfinite(value):
        raise ValueError("timestamps and gaps must be finite and nonnegative")
    return float(value)


def endpoint_identity(url):
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username is not None:
            raise ValueError
        host = parts.hostname
        host = f"[{host}]" if ":" in host else host
        port = parts.port
        if port is not None and port != {"http": 80, "https": 443}[parts.scheme]:
            host = f"{host}:{port}"
        return {"origin": f"{parts.scheme}://{host}", "path": parts.path or "/"}
    except (TypeError, ValueError):
        raise ValueError("endpoints must be HTTP or HTTPS without embedded credentials") from None


def inventory_identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.:-]{1,253}", value) is not None


def load_inventory(path, endpoint):
    if path is None:
        return None
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except ValueError:
        raise ValueError("inventory must be JSON") from None
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or not isinstance(value.get("endpoint"), str)
            or endpoint_identity(value["endpoint"]) != endpoint):
        raise ValueError("inventory must use schema_version 1 and match the benchmark endpoint")
    rows = value.get("replicas")
    if not isinstance(rows, list) or not rows:
        raise ValueError("inventory must contain a nonempty replicas array")
    replicas = []
    seen_ids = set()
    seen_gpus = set()
    startup_sources = {
        "native": ("packed_decode_ready", "model_loading_end"),
        "vllm": ("vllm_engine_ready", "kubernetes_ready"),
        "dynamo": ("dynamo_ready", "vllm_engine_ready", "kubernetes_ready"),
    }
    for index, row in enumerate(rows):
        where = f"inventory replica {index + 1}"
        if not isinstance(row, dict) or not all(inventory_identifier(row.get(k)) for k in ("id", "host", "epoch")):
            raise ValueError(f"{where} needs valid id, host and epoch identifiers")
        if row["id"] in seen_ids:
            raise ValueError("inventory repeats a replica ID")
        seen_ids.add(row["id"])
        gpu_uuids = row.get("gpu_uuids")
        if not isinstance(gpu_uuids, list) or not gpu_uuids or any(
            not isinstance(gpu, str)
            or not re.fullmatch(r"GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}", gpu)
            for gpu in gpu_uuids
        ):
            raise ValueError(f"{where} needs physical GPU UUIDs")
        gpu_keys = {gpu.lower() for gpu in gpu_uuids}
        if len(gpu_keys) != len(gpu_uuids) or seen_gpus.intersection(gpu_keys):
            raise ValueError("inventory repeats a physical GPU UUID")
        seen_gpus.update(gpu_keys)
        topology = row.get("topology")
        if (not isinstance(topology, dict) or topology.get("backend") not in ["native", "vllm", "dynamo"]
                or topology.get("stage") not in ("combined", "prefill", "decode")
                or not all(type(topology.get(k)) is int and topology[k] > 0 for k in ("tp", "dp", "ep"))
                or topology["tp"] * topology["dp"] != len(gpu_uuids)
                or len(gpu_uuids) % topology["ep"] != 0):
            raise ValueError(f"{where} needs backend, stage, and compatible tp/dp/ep topology")
        startup = row.get("startup")
        if (not isinstance(startup, dict) or startup.get("source") not in startup_sources[topology["backend"]]
                or startup.get("ready") is not True or not isinstance(startup.get("image_digest"), str)
                or not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", startup["image_digest"])
                or not isinstance(startup.get("checkpoint_revision"), str)
                or not re.fullmatch(r"(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})", startup["checkpoint_revision"])):
            raise ValueError(f"{where} needs ready startup evidence, an image digest and a pinned checkpoint revision")
        width = row.get("decode_width")
        if ((width is not None and (type(width) is not int or width <= 0))
                or (startup["source"] == "packed_decode_ready" and topology["stage"] != "prefill" and width is None)
                or (topology["stage"] == "prefill" and width is not None)):
            raise ValueError(f"{where} has invalid decode_width for its startup and stage")
        startup_fields = ("max_residents", "max_decode_width", "slot_bytes_per_rank", "kv_bytes_per_rank")
        for name in startup_fields:
            if name in startup and (type(startup[name]) is not int or startup[name] < 0):
                raise ValueError(f"{where} has invalid startup metrics")
        if "max_decode_width" in startup and width != startup["max_decode_width"]:
            raise ValueError(f"{where} startup width disagrees with decode_width")
        replica = {
            "id": row["id"], "host": row["host"], "epoch": row["epoch"],
            "gpu_uuids": gpu_uuids, "decode_width": width,
            "topology": {k: topology[k] for k in ("backend", "stage", "tp", "dp", "ep")},
            "startup": {k: startup[k] for k in ("source", "ready", "image_digest", "checkpoint_revision", *startup_fields)
                        if k in startup},
        }
        for name, paths in (("admission_url", ("/internal/admission",)), ("readiness_url", ("/health", "/readyz"))):
            if (url := row.get(name)) is None:
                continue
            if not isinstance(url, str):
                raise ValueError(f"{where} has invalid probe URL")
            probe = endpoint_identity(url)
            if urlsplit(url).query or urlsplit(url).fragment or not probe["path"].endswith(paths):
                raise ValueError(f"{where} has invalid probe URL")
            replica[name] = probe["origin"] + probe["path"]
        replicas.append(replica)
    return {
        "schema_version": 1, "endpoint": endpoint, "sha256": hashlib.sha256(raw).hexdigest(),
        "gpu_count": len(seen_gpus), "replicas": replicas,
    }


def probe(args, url, *, token=None, expect_json=True):
    headers = {"Accept": "application/json"} if expect_json else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        request = Request(url, headers=headers)
        with args.opener.open(request, timeout=min(args.timeout, 3)) as response:
            if not expect_json:
                return "observed", None
            data = response.read(MAX_PROBE_BYTES + 1)
            if len(data) <= MAX_PROBE_BYTES and isinstance(value := json.loads(data), dict):
                return "observed", value
            return "invalid_response", None
    except HTTPError as error:
        error.close()
        return f"http_{error.code}", None
    except (OSError, URLError, HTTPException):
        return "unreachable", None
    except (ValueError, TypeError):
        return "invalid_response", None


def model_evidence(args):
    prefix, separator, _ = args.endpoint["path"].rpartition("/v1/")
    path = f"{prefix}/v1/models" if separator else "/v1/models"
    status, value = probe(args, args.endpoint["origin"] + path, token=args.api_key)
    ids = set()
    if value is not None:
        models = value.get("data")
        if isinstance(models, list):
            ids = {m["id"] for m in models if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]}
        if not ids:
            status = "invalid_response"
    return {"status": status, "path": path, "ids": sorted(ids)}


def replica_probe(args, replica):
    report = {}
    if (url := replica.get("readiness_url")) is not None:
        status, _ = probe(args, url, expect_json=False)
        report["readiness"] = {"endpoint": endpoint_identity(url), "status": status}
    if (url := replica.get("admission_url")) is not None:
        status, value = probe(args, url)
        item = {"endpoint": endpoint_identity(url), "status": status}
        if value is not None:
            epoch = value.get("epoch")
            if inventory_identifier(epoch):
                item.update(epoch=epoch, status="matched" if epoch == replica["epoch"] else "mismatch")
            else:
                item["status"] = "invalid_response"
        report["admission_epoch"] = item
    return report


def replica_evidence(args, before):
    if args.inventory is None:
        return {"status": "missing"}
    inventory = args.inventory
    rows = []
    for replica in inventory["replicas"]:
        observations = {"before": before[replica["id"]], "after": replica_probe(args, replica)}
        provenance = ["pinned_operator_inventory"]
        if any(sample.get("readiness", {}).get("status") == "observed" for sample in observations.values()):
            provenance.append("live_readiness_observed")
        if any(sample.get("admission_epoch", {}).get("status") == "matched" for sample in observations.values()):
            provenance.append("live_admission_epoch_matched")
        rows.append({**{k: v for k, v in replica.items() if not k.endswith("_url")},
                     "provenance": provenance, "live_observations": observations})
    return {**inventory, "status": "pinned", "replicas": rows}


def task_id(session):
    return session.get("task_id", session["id"])


def manifest_id(value):
    return isinstance(value, str) and bool(value) and all(33 <= ord(char) <= 126 for char in value)


def turn_dependencies(sessions):
    explicit_ids = {}
    for i, session in enumerate(sessions):
        for j, turn in enumerate(session["turns"]):
            if "id" in turn:
                identifier = turn["id"]
                if not manifest_id(identifier) or identifier in explicit_ids:
                    raise ValueError("turn IDs must be distinct, nonempty printable ASCII strings")
                explicit_ids[identifier] = (i, j)

    dependencies = {}
    for i, session in enumerate(sessions):
        for j, turn in enumerate(session["turns"]):
            declared = turn.get("depends_on", [])
            if not isinstance(declared, list) or not all(manifest_id(dep) for dep in declared):
                raise ValueError("turn depends_on must be an array of turn IDs")
            before = {(i, j - 1)} if j else set()
            for identifier in declared:
                parent = explicit_ids.get(identifier)
                if parent is None or task_id(sessions[parent[0]]) != task_id(session):
                    raise ValueError("turn dependencies must reference known turns in the same task")
                before.add(parent)
            dependencies[i, j] = before
    return dependencies


def load_sessions(path, mode, limit, *, task_limit=None):
    root = json.loads(path.read_text())
    sessions = root.get("sessions") if isinstance(root, dict) else None
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("the manifest must contain a nonempty sessions array")
    if limit is not None and task_limit is not None:
        raise ValueError("select sessions or tasks, not both")
    if limit is not None:
        if len(sessions) < limit:
            raise ValueError(f"requested {limit} sessions and the manifest has {len(sessions)}")
        sessions = sessions[:limit]

    seen = set()
    for session in sessions:
        if not isinstance(session, dict):
            raise ValueError("each session must be an object")
        session_id = session.get("id")
        if not isinstance(session_id, str) or not session_id or session_id in seen:
            raise ValueError("session IDs must be nonempty, distinct strings")
        if any(ord(char) < 33 or ord(char) > 126 for char in session_id):
            raise ValueError("session IDs must use printable ASCII without spaces")
        if not manifest_id(task_id(session)):
            raise ValueError("task IDs must be nonempty printable ASCII strings")
        seen.add(session_id)
        turns = session.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"session {session_id} must contain a nonempty turns array")
        for turn in turns:
            if not isinstance(turn, dict) or not isinstance(turn.get("body"), dict):
                raise ValueError("each turn must have a Chat Completions body object")
            options = turn["body"].get("stream_options")
            if options is not None and not isinstance(options, dict):
                raise ValueError("stream_options must be an object or null")
            if turn["body"].get("n", 1) != 1:
                raise ValueError("each stream must request one choice")
            cohort = turn.get("cohort", "unclassified")
            if not isinstance(cohort, str) or not cohort:
                raise ValueError("turn cohort must be a nonempty string")
    if task_limit is not None:
        task_ids = list(dict.fromkeys(task_id(session) for session in sessions))
        if len(task_ids) < task_limit:
            raise ValueError(f"requested {task_limit} tasks and the manifest has {len(task_ids)}")
        selected = set(task_ids[:task_limit])
        sessions = [session for session in sessions if task_id(session) in selected]

    for session in sessions:
        for turn in session["turns"]:
            nonnegative_seconds(turn.get("at_s") if mode == "open" else turn.get("gap_s", 0))
    if any("id" in turn or "depends_on" in turn for session in sessions for turn in session["turns"]):
        try:
            for _ in TopologicalSorter(turn_dependencies(sessions)).static_order():
                pass
        except CycleError:
            raise ValueError("turn dependencies contain a cycle") from None
    return sessions


@dataclass
class Turn:
    session: str
    cohort: str
    offered: float
    started: float
    finished: float = 0.0
    chunks: array = field(default_factory=lambda: array("d"))
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    error: str | None = None
    task: str | None = None

    @property
    def cache_cohort(self):
        if self.prompt_tokens is None or self.cached_tokens is None:
            return "unknown"
        if self.cached_tokens == 0:
            return "cold" if self.prompt_tokens <= COLD_MAX_INPUT else "cold_large"
        return "warm" if self.prompt_tokens - self.cached_tokens <= WARM_MAX_NEW else "warm_large"

    @property
    def ttft_ms(self):
        return (self.chunks[0] - self.offered) * 1000 if self.chunks else None

    @property
    def worst_gap_ms(self):
        return max(((b - a) * 1000 for a, b in pairwise(self.chunks)), default=None)

    @property
    def decode_tokens_per_s(self):
        if (self.output_tokens is None or self.output_tokens < MIN_DECODE_TOKENS
                or len(self.chunks) < 2 or self.chunks[-1] <= self.chunks[0]):
            return None
        # The compatible endpoints report total output tokens at end-of-stream.
        # Estimate decode throughput with a one-token first chunk.
        return (self.output_tokens - 1) / (self.chunks[-1] - self.chunks[0])


def request_body(turn, model, request_format, cache_namespace=None):
    body = dict(turn["body"])
    if request_format == "vllm":
        template_kwargs = body.get("chat_template_kwargs")
        if template_kwargs is not None and not isinstance(template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object or null")
        kwargs = dict(template_kwargs or {})
        thinking = body.pop("thinking", None)
        if thinking is not None:
            if isinstance(thinking, bool):
                enabled = thinking
            elif isinstance(thinking, dict) and thinking.get("type") in ("enabled", "disabled"):
                enabled = thinking["type"] == "enabled"
            else:
                raise ValueError("vLLM replay requires thinking to be enabled, disabled, or a boolean")
            kwargs.update(enable_thinking=enabled, thinking_mode="thinking" if enabled else "chat")
        if type(effort := body.get("reasoning_effort")) is int:
            if not 1 <= effort <= 100:
                raise ValueError("numeric reasoning_effort must be in [1, 100]")
            kwargs["reasoning_effort"] = body.pop("reasoning_effort")
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        if cache_namespace is not None:
            existing = body.get("cache_salt")
            if existing is not None and not isinstance(existing, str):
                raise ValueError("cache_salt must be a string")
            body["cache_salt"] = hashlib.sha256(
                f"{cache_namespace}:{existing or ''}".encode()).hexdigest()
    body["stream"] = True
    body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
    if model:
        body["model"] = model
    return body


def sse_data(response):
    data = []
    for line in response:
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                yield b"\n".join(data)
                data.clear()
        elif line.startswith(b"data:"):
            data.append(line[5:].removeprefix(b" "))
    if data:
        yield b"\n".join(data)


def token_bearing(choices):
    for choice in choices:
        delta = choice.get("delta") or {}
        if any(delta.get(key) for key in ("content", "reasoning_content", "reasoning", "refusal")):
            return True
        function = delta.get("function_call") or {}
        if function.get("name") or function.get("arguments"):
            return True
        for call in delta.get("tool_calls") or ():
            function = call.get("function") or {}
            if function.get("name") or function.get("arguments"):
                return True
    return False


def usage_count(usage, key):
    n = usage.get(key)
    return n if type(n) is int and n >= 0 else None


def run_turn(args, session, turn, offered, task=None):
    result = Turn(session, turn.get("cohort", "unclassified"), offered, time.monotonic(), task=task)
    cache_namespace = session if args.replenish_s is not None else args.cache_namespace
    body = request_body(turn, args.model, args.request_format, cache_namespace)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream",
               args.session_header: args.session_prefix + session}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    saw_done = saw_finish = False
    try:
        request = Request(args.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with args.opener.open(request, timeout=args.timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                result.error = "expected_sse"
                return result
            for data in sse_data(response):
                if data == b"[DONE]":
                    saw_done = True
                    break
                payload = json.loads(data)
                if payload.get("error"):
                    result.error = "stream_error"
                    break
                choices = payload.get("choices") or []
                if token_bearing(choices):
                    result.chunks.append(time.monotonic())
                if any(choice.get("finish_reason") in ("error", "abort", "failed", "cancelled", "canceled")
                       for choice in choices):
                    result.error = "stream_error"
                    break
                saw_finish |= any(choice.get("finish_reason") is not None for choice in choices)
                usage = payload.get("usage")
                if usage:
                    if (count := usage_count(usage, "completion_tokens")) is not None:
                        result.output_tokens = count
                    if (count := usage_count(usage, "prompt_tokens")) is not None:
                        result.prompt_tokens = count
                    details = usage.get("prompt_tokens_details") or {}
                    if (count := usage_count(details, "cached_tokens")) is not None:
                        result.cached_tokens = count
        if result.error is None and not (saw_done and saw_finish):
            result.error = "incomplete_stream"
        if result.error is None and result.output_tokens is None:
            result.error = "missing_usage"
        if (result.error is None and result.cached_tokens is not None
                and result.prompt_tokens is not None and result.cached_tokens > result.prompt_tokens):
            result.error = "invalid_cache_usage"
    except HTTPError as err:
        result.error = f"http_{err.code}"
        err.close()
    except (TimeoutError, URLError, OSError, HTTPException) as err:
        cause = err.reason if isinstance(err, URLError) else err
        result.error = "timeout" if isinstance(cause, TimeoutError) else "transport_error"
    except (ValueError, AttributeError, TypeError):
        result.error = "invalid_response"
    finally:
        result.finished = time.monotonic()
    return result


def sleep_until(when):
    remaining = when - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def run_session(args, session, epoch):
    results = []
    next_offer = epoch
    for turn in session["turns"]:
        next_offer += turn.get("gap_s", 0)
        sleep_until(next_offer)
        result = run_turn(args, session["id"], turn, next_offer, task_id(session))
        results.append(result)
        next_offer = result.finished
    return results


@dataclass
class ReplayTask:
    sessions: list
    dependencies: dict
    graph: TopologicalSorter
    start: float
    root_gap: float
    namespace: str | None
    completions: dict = field(default_factory=dict)


def run_dependent_turns(args, sessions, executor, workers, epoch):
    grouped = defaultdict(list)
    for session in sessions:
        grouped[task_id(session)].append(session)
    templates = [(branches, turn_dependencies(branches)) for branches in grouped.values()]
    rotate = cycle(templates)
    replenish_s = getattr(args, "replenish_s", None)
    deadline = None if replenish_s is None else epoch + replenish_s
    slots = len(templates) if deadline is None else args.target_active
    ordinals = count()
    active = {}
    ready = []
    in_flight = {}
    results = []

    def schedule_ready(ordinal, task):
        for i, j in task.graph.get_ready():
            gap = task.sessions[i]["turns"][j].get("gap_s", 0)
            predecessors = task.dependencies[i, j]
            if predecessors:
                offered = max(task.completions[parent] for parent in predecessors) + gap
            else:
                offered = task.start + (gap - task.root_gap)
            heapq.heappush(ready, (offered, ordinal, i, j))

    def admit(start, replacement=False):
        branches, dependencies = next(rotate)
        graph = TopologicalSorter(dependencies)
        graph.prepare()
        # Replacements use the same relative root delays, with the earliest root
        # starting immediately when a slot is free.
        root_gap = 0
        if replacement:
            root_gap = min(branches[i]["turns"][j].get("gap_s", 0)
                           for (i, j), before in dependencies.items() if not before)
        ordinal = next(ordinals)
        task = ReplayTask(branches, dependencies, graph, start, root_gap,
                          uuid4().hex if deadline is not None else None)
        active[ordinal] = task
        schedule_ready(ordinal, task)

    for _ in range(slots):
        admit(epoch)
    while active:
        while ready and ready[0][0] <= time.monotonic() and len(in_flight) < workers:
            offered, ordinal, i, j = heapq.heappop(ready)
            task = active[ordinal]
            session = task.sessions[i]
            branch_id = session["id"] if task.namespace is None else f"{task.namespace}:{i}"
            future = executor.submit(run_turn, args, branch_id, session["turns"][j],
                                     offered, task.namespace or task_id(session))
            in_flight[future] = (ordinal, (i, j))

        if not in_flight:
            sleep_until(ready[0][0])
            continue
        timeout = max(0, ready[0][0] - time.monotonic()) if ready and len(in_flight) < workers else None
        done, _ = wait(in_flight, timeout=timeout, return_when=FIRST_COMPLETED)
        for future in done:
            ordinal, key = in_flight.pop(future)
            task = active[ordinal]
            result = future.result()
            results.append(result)
            task.completions[key] = result.finished
            task.graph.done(key)
            if task.graph.is_active():
                schedule_ready(ordinal, task)
            else:
                del active[ordinal]
                if deadline is not None and time.monotonic() < deadline:
                    admit(max(task.completions.values()), replacement=True)
    return results


def replay(args, sessions):
    workers = args.workers or len(sessions)
    offers = (sorted((turn["at_s"], i, n) for i, session in enumerate(sessions)
                     for n, turn in enumerate(session["turns"])) if args.mode == "open" else None)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        epoch = time.monotonic()
        event = {}
        timer = None
        if args.event_at_s is not None:
            def mark_event():
                event.update(label=args.event_label, scheduled_s=args.event_at_s,
                             marker_s=time.monotonic() - epoch)
                print(json.dumps({"benchmark_event": event}), file=sys.stderr, flush=True)

            timer = Timer(args.event_at_s, mark_event)
            timer.start()
        try:
            if offers is None:
                if getattr(args, "replenish_s", None) is not None or any(
                        turn.get("depends_on") for session in sessions for turn in session["turns"]):
                    results = run_dependent_turns(args, sessions, executor, workers, epoch)
                else:
                    futures = [executor.submit(run_session, args, session, epoch) for session in sessions]
                    results = [turn for future in futures for turn in future.result()]
            else:
                futures = []
                for offset, i, n in offers:
                    offered = epoch + offset / args.arrival_scale
                    sleep_until(offered)
                    session = sessions[i]
                    futures.append(executor.submit(run_turn, args, session["id"], session["turns"][n],
                                                   offered, task_id(session)))
                results = [future.result() for future in futures]
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
        return epoch, results, event


def percentiles(values):
    values = sorted(values)
    if not values:
        return {}
    return {f"p{p}": round(values[math.ceil(p * len(values) / 100) - 1], 3)
            for p in (10, 50, 95, 99)}


def gap_percentiles(results):
    # Millisecond bins avoid materializing every gap in a multi-hour soak.
    counts = Counter(math.ceil((b - a) * 1000) for r in results for a, b in pairwise(r.chunks))
    total = sum(counts.values())
    if not total:
        return {}
    quantiles = iter((10, 50, 95, 99))
    quantile = next(quantiles)
    cumulative = 0
    result = {}
    for duration, count in sorted(counts.items()):
        cumulative += count
        while quantile is not None and 100 * cumulative >= quantile * total:
            result[f"p{quantile}"] = duration
            quantile = next(quantiles, None)
    return result


def progress_intervals(chunks, window):
    if len(chunks) < 2:
        return
    start = chunks[0]
    for earlier, later in pairwise(chunks):
        if later - earlier > window:
            yield start, earlier + window
            start = later
    if chunks[-1] > start:
        yield start, chunks[-1]


def concurrency(intervals, target):
    events = defaultdict(int)
    for start, end in intervals:
        if start < end:
            events[start] += 1
            events[end] -= 1
    count = peak = 0
    at_target = None
    longest = 0.0
    for when, change in sorted(events.items()):
        count += change
        peak = max(peak, count)
        if count >= target and at_target is None:
            at_target = when
        elif count < target and at_target is not None:
            longest = max(longest, when - at_target)
            at_target = None
    return {"peak": peak, "longest_at_target_s": round(longest, 3)}


def turn_metrics(results):
    successful = [r for r in results if r.error is None]
    with_prompt = [r for r in successful if r.prompt_tokens is not None]
    with_cache = [r for r in with_prompt if r.cached_tokens is not None]
    long_outputs = sum(r.output_tokens >= MIN_DECODE_TOKENS for r in successful)
    decode_rates = [rate for r in successful if (rate := r.decode_tokens_per_s) is not None]
    worst_gaps = [gap for r in successful if (gap := r.worst_gap_ms) is not None]
    return {
        "turns_offered": len(results),
        "turns_completed": len(successful),
        "success_rate": len(successful) / len(results) if results else 0,
        "errors": dict(Counter(r.error for r in results if r.error)),
        "successful_output_tokens": sum(r.output_tokens for r in successful),
        "reported_prompt_tokens": sum(r.prompt_tokens for r in with_prompt),
        "reported_cached_tokens": sum(r.cached_tokens for r in with_cache),
        "reported_uncached_input_tokens": sum(r.prompt_tokens - r.cached_tokens for r in with_cache),
        "turns_with_cache_usage": len(with_cache),
        "prompt_tokens_per_turn": percentiles(r.prompt_tokens for r in with_prompt),
        "uncached_input_tokens_per_turn": percentiles(r.prompt_tokens - r.cached_tokens for r in with_cache),
        "output_tokens_per_turn": percentiles(r.output_tokens for r in successful),
        "dispatch_delay_ms": percentiles((r.started - r.offered) * 1000 for r in results),
        "ttft_ms": percentiles(r.ttft_ms for r in successful if r.chunks),
        "successful_turns_with_token_chunks": sum(bool(r.chunks) for r in successful),
        "inter_chunk_gap_ms": gap_percentiles(successful),
        "per_turn_max_inter_chunk_gap_ms": percentiles(worst_gaps),
        "turns_with_inter_chunk_stalls": sum(gap > GAP_MS for gap in worst_gaps),
        "turns_with_128_plus_output_tokens": long_outputs,
        "turns_with_unobserved_128_plus_decode_rate": long_outputs - len(decode_rates),
        "decode_tokens_per_s_128_plus": percentiles(decode_rates),
    }


def joint_slo_failures(result):
    if result.error is not None:
        return ("request_failed",)
    kind = result.cache_cohort
    if kind == "unknown":
        return ("unreported_cache_usage",)
    if kind.endswith("_large"):
        return ("outside_standard_prompt_envelope",)
    failures = []
    ttft_limit = WARM_TTFT_MS if kind == "warm" else COLD_TTFT_MS
    if result.ttft_ms is None or result.ttft_ms > ttft_limit:
        failures.append("ttft")
    if (gap := result.worst_gap_ms) is not None and gap > GAP_MS:
        failures.append("stream_stall")
    if result.output_tokens >= MIN_DECODE_TOKENS:
        rate = result.decode_tokens_per_s
        if rate is None or rate < MIN_DECODE_TOKENS_PER_S:
            failures.append("decode_rate")
    return failures


def clipped(intervals, start, end):
    return ((max(a, start), min(b, end)) for a, b in intervals)


def throughput(units, elapsed):
    return round(units / elapsed, 3) if elapsed > 0 else 0


def task_metrics(results, start, end, target):
    grouped = defaultdict(list)
    for result in results:
        grouped[result.task or result.session].append(result)

    intervals = []
    offered = unfinished = failed = 0
    completed_durations = []
    failures = Counter()
    for turns in grouped.values():
        first = min(turn.offered for turn in turns)
        last = max(turn.finished for turn in turns)
        intervals.append((first, last))
        if not start <= first < end:
            continue
        offered += 1
        if last > end:
            unfinished += 1
            continue
        if errors := {turn.error for turn in turns if turn.error is not None}:
            failed += 1
            failures.update(errors)
        else:
            completed_durations.append((last - first) * 1000)

    completed = len(completed_durations)
    return concurrency(clipped(intervals, start, end), target), {
        "offered": offered,
        "completed": completed,
        "failed": failed,
        "unfinished_at_window_end": unfinished,
        "success_rate": completed / offered if offered else 0,
        "completed_per_s": throughput(completed, end - start),
        "completion_ms": percentiles(completed_durations),
        "failed_by_error": dict(failures),
    }


def summarize(results, args, start, end):
    offered = [r for r in results if start <= r.offered < end]
    completed_in_window = [r for r in offered if r.finished <= end]
    metrics = turn_metrics(offered)
    elapsed = max(0., end - start)
    sessions = {}
    for result in results:
        first, last = sessions.get(result.session, (result.offered, result.finished))
        sessions[result.session] = (min(first, result.offered), max(last, result.finished))
    cohorts = defaultdict(list)
    measured = defaultdict(list)
    planned_vs_observed = defaultdict(Counter)
    joint = []
    joint_reasons = Counter()
    for result in offered:
        cohorts[result.cohort].append(result)
        actual = result.cache_cohort if result.error is None else "failed"
        measured[actual].append(result)
        planned_vs_observed[result.cohort][actual] += 1
        reasons = joint_slo_failures(result)
        if not reasons and result.finished <= end:
            joint.append(result)
        joint_reasons.update(reasons)
    generation_spans = ((r.chunks[0], r.chunks[-1]) for r in results if len(r.chunks) > 1)
    progress = (span for r in results for span in progress_intervals(r.chunks, args.progress_window_ms / 1000))
    successful_progress = (span for r in results if r.error is None
                           for span in progress_intervals(r.chunks, args.progress_window_ms / 1000))
    output = sum(r.output_tokens for r in completed_in_window if r.error is None)
    uncached = sum(r.prompt_tokens - r.cached_tokens for r in completed_in_window
                   if r.error is None and r.prompt_tokens is not None and r.cached_tokens is not None)
    joint_output = sum(r.output_tokens for r in joint)
    live_tasks, tasks = task_metrics(results, start, end, args.target_active)
    report = {
        **metrics,
        "mode": args.mode,
        "arrival_scale": args.arrival_scale,
        "wall_s": round(elapsed, 3),
        "offered_turns_per_s": throughput(len(offered), elapsed),
        "successful_output_tokens_per_s": throughput(output, elapsed),
        "measured_uncached_input_tokens_per_s": throughput(uncached, elapsed),
        "turns_finished_after_window": len(offered) - len(completed_in_window),
        "target_active": args.target_active,
        "progress_window_ms": args.progress_window_ms,
        "live_sessions": concurrency(clipped(sessions.values(), start, end), args.target_active),
        "live_tasks": live_tasks,
        "tasks": tasks,
        "http_inflight": concurrency(clipped(((r.started, r.finished) for r in results), start, end), args.target_active),
        "first_to_last_token_chunk_overlap": concurrency(clipped(generation_spans, start, end), args.target_active),
        "recent_token_progress": concurrency(clipped(progress, start, end), args.target_active),
        "successful_recent_token_progress": concurrency(clipped(successful_progress, start, end), args.target_active),
        "cohorts": {name: metrics if len(turns) == len(offered) else turn_metrics(turns)
                    for name, turns in cohorts.items()},
        "measured_cache_cohorts": {name: metrics if len(turns) == len(offered) else turn_metrics(turns)
                                   for name, turns in measured.items()},
        "planned_vs_observed_cache_cohorts": planned_vs_observed,
        "joint_slo_goodput": {
            "turns_completed_within_all_slos": len(joint),
            "turns_per_s": throughput(len(joint), elapsed),
            "output_tokens_per_s": throughput(joint_output, elapsed),
            "turns_by_failed_criterion": dict(joint_reasons),
        },
    }
    if args.gpus is not None:
        report["gpus"] = args.gpus
        report["joint_slo_goodput"].update(
            output_tokens_per_gpu_s=throughput(joint_output, elapsed * args.gpus),
            turns_per_gpu_hour=throughput(3600 * len(joint), elapsed * args.gpus),
        )
    return report


def qualification(report, workload, milestone_s=None, min_turns_per_s=None, min_cache_samples=None,
                  *, event_replay=False, post_event_s=None):
    cohorts = report["measured_cache_cohorts"]
    warm = cohorts.get("warm", {}).get("ttft_ms", {}).get("p95")
    cold = cohorts.get("cold", {}).get("ttft_ms", {}).get("p95")
    gap = report["inter_chunk_gap_ms"].get("p99")
    rate = report["decode_tokens_per_s_128_plus"].get("p10")
    checks = {
        "success_rate": report["success_rate"] >= MIN_SUCCESS_RATE,
        "cold_ttft": cold is not None and cold <= COLD_TTFT_MS,
        "cache_usage_observed": report["turns_completed"] == report["turns_with_cache_usage"] > 0,
    }
    if workload == "cold":
        checks["cold_cohort_observed"] = cohorts.get("cold", {}).get("turns_completed", 0) == report["turns_completed"] > 0
    else:
        checks.update(
            warm_ttft=warm is not None and warm <= WARM_TTFT_MS,
            stream_gap=gap is not None and gap <= GAP_MS,
            decode_rate=rate is not None and rate >= MIN_DECODE_TOKENS_PER_S,
            decode_rates_observed=report["turns_with_unobserved_128_plus_decode_rate"] == 0,
        )
    required_duration = post_event_s if post_event_s is not None else milestone_s
    occupancy_workloads = {"sessions": ("live_sessions", 7200), "tasks": ("live_tasks", 7200),
                           "streams": ("successful_recent_token_progress", 600)}
    if workload in occupancy_workloads:
        occupancy, default_duration = occupancy_workloads[workload]
        required_duration = required_duration if required_duration is not None else default_duration
        checks["sustained_concurrency"] = report[occupancy]["longest_at_target_s"] >= required_duration
    elif workload == "load" and required_duration is not None:
        checks["sustained_load_interval"] = report["wall_s"] >= required_duration
    elif workload == "cold":
        required_duration = None
    if min_turns_per_s is not None:
        checks["offered_turn_rate"] = report["offered_turns_per_s"] >= min_turns_per_s
    profile = "post_event" if post_event_s is not None else "production" if milestone_s is None else "milestone"
    cache_kinds = ("cold",) if workload == "cold" else ("warm", "cold")
    cache_samples = {kind: cohorts.get(kind, {}).get("successful_turns_with_token_chunks", 0)
                     for kind in cache_kinds}
    sample_floor = min_cache_samples or (PRODUCTION_CACHE_SAMPLES if milestone_s is None else 1)
    checks["cache_sample_counts"] = all(count >= sample_floor for count in cache_samples.values())
    explicit_agreement = {}
    for kind in cache_samples:
        counts = report["planned_vs_observed_cache_cohorts"].get(kind, {})
        if total := sum(counts.values()):
            explicit_agreement[kind] = {"matching": counts.get(kind, 0), "offered": total}
    if profile == "production":
        inventory = report.get("replica_inventory", {})
        models = report.get("served_model_evidence", {})
        snapshots = ("before",) if event_replay else ("before", "after")
        checks["inventory_gpu_count"] = (report.get("gpus") is not None
                                          and inventory.get("gpu_count") == report["gpus"])
        checks["no_observed_replica_epoch_mismatch"] = all(
            replica["live_observations"].get(snapshot, {}).get("admission_epoch", {}).get("status") != "mismatch"
            for replica in inventory.get("replicas", ())
            for snapshot in snapshots
        )
        checks["served_models_observed"] = all(
            models.get(snapshot, {}).get("status") == "observed" and bool(models[snapshot].get("ids"))
            for snapshot in snapshots
        )
        configured_model = report.get("model_override")
        requested_models = {configured_model} if configured_model else set(report.get("workload", {}).get("requested_models", ()))
        if requested_models:
            checks["requested_models_advertised"] = all(
                requested_models.issubset(models.get(snapshot, {}).get("ids", ())) for snapshot in snapshots
            )
        if explicit_agreement and not event_replay:
            checks["explicit_cache_cohorts_agree"] = all(
                row["matching"] / row["offered"] >= MIN_SUCCESS_RATE for row in explicit_agreement.values()
            )
    return {"workload": workload, "profile": profile, "required_sustained_s": required_duration,
            "minimum_samples_per_cache_cohort": sample_floor, "cache_samples": cache_samples,
            "explicit_cache_cohort_agreement": explicit_agreement,
            "passed": all(checks.values()), "checks": checks}


def workload_summary(sessions, mode):
    digest = hashlib.sha256()
    cohorts = Counter()
    caps = []
    times = []
    models = set()
    for session in sessions:
        for turn in session["turns"]:
            fields = [session["id"], turn]
            if "task_id" in session:
                fields.append(task_id(session))
            canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
            digest.update(canonical.encode() + b"\n")
            cohorts[turn.get("cohort", "unclassified")] += 1
            body = turn["body"]
            if isinstance(model := body.get("model"), str) and model:
                models.add(model)
            cap = body.get("max_completion_tokens")
            if cap is None:
                cap = body.get("max_tokens")
            if cap is not None:
                if type(cap) is not int or cap <= 0:
                    raise ValueError("explicit output caps must be positive integers")
                caps.append(cap)
            if mode == "open":
                times.append(turn["at_s"])
    turns = sum(cohorts.values())
    result = {
        "manifest_sha256": digest.hexdigest(),
        "sessions": len(sessions),
        "tasks": len({task_id(session) for session in sessions}),
        "turns": turns,
        "planned_cohorts": dict(cohorts),
        "requested_models": sorted(models),
        "explicit_output_cap_tokens": percentiles(caps),
        "turns_using_server_output_default": turns - len(caps),
    }
    if times:
        result["open_loop_trace_span_s"] = max(times) - min(times)
    return result


def run_case(args, sessions, workload):
    model = {"before": model_evidence(args)}
    before = {} if args.inventory is None else {
        replica["id"]: replica_probe(args, replica) for replica in args.inventory["replicas"]
    }
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    epoch, results, event = replay(args, sessions)
    end = max(r.finished for r in results)
    model["after"] = model_evidence(args)
    report = summarize(results, args, epoch + args.warmup_s, end)
    report.update(backend=args.backend, request_format=args.request_format, session_header=args.session_header,
                  model_override=args.model, warmup_s=args.warmup_s, workload=workload,
                  isolated_cache_namespace=args.cache_namespace is not None or args.replenish_s is not None,
                  endpoint=args.endpoint,
                  replay_started_at=started_at, served_model_evidence=model,
                  replica_inventory=replica_evidence(args, before))
    if args.replenish_s is not None:
        report["replenishment"] = {
            "source": "rotating_frozen_task_corpus",
            "duration_s": args.replenish_s,
            "target_active_tasks": args.target_active,
            "task_instances": len({result.task for result in results}),
            "drain_s": round(max(0., end - epoch - args.replenish_s), 3),
        }
    if args.qualify:
        report["qualification"] = qualification(report, args.qualify, args.milestone_s,
                                                args.min_turns_per_s, args.min_cache_samples,
                                                event_replay=args.event_at_s is not None)
    if args.event_at_s is not None:
        event_report = {
            "label": args.event_label,
            "scheduled_s": args.event_at_s,
            "marker_observed": bool(event),
            "external_action_verified": False,
        }
        if event:
            start_after_event = epoch + event["marker_s"] + args.event_recovery_s
            stop_at = start_after_event + args.post_event_s
            # The pinned inventory describes the pre-fault allocation only.
            post_args = argparse.Namespace(**vars(args))
            post_args.gpus = None
            post = summarize(results, post_args, start_after_event, min(end, stop_at))
            if args.qualify:
                post["qualification"] = qualification(post, args.qualify, args.milestone_s,
                                                      args.min_turns_per_s, args.min_cache_samples,
                                                      post_event_s=args.post_event_s)
            event_report.update(marker_s=event["marker_s"], recovery_s=args.event_recovery_s,
                                full_window_observed=end >= stop_at, post_event=post)
        report["event"] = event_report
    return report


def case_failed(report):
    if "qualification" in report and not report["qualification"]["passed"]:
        return True
    if event := report.get("event"):
        if not event.get("full_window_observed"):
            return True
        return not event["post_event"].get("qualification", {"passed": True})["passed"]
    return False


def redact_secrets(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        return {redact_secrets(k, secrets): redact_secrets(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item, secrets) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--mode", choices=("closed", "open"), default="closed")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--sessions", type=int, help="use the first N sessions from the manifest")
    scope.add_argument("--tasks", type=int, help="use the first N tasks and all their session branches")
    parser.add_argument("--workers", type=int, help="HTTP workers, defaults to session count")
    parser.add_argument("--replenish-s", type=float,
                        help="refill closed-loop task slots until this elapsed time, then drain")
    parser.add_argument("--arrival-scale", type=float, default=1, help="open-loop offered rate multiplier")
    parser.add_argument("--arrival-scales", help="comma-separated open-loop sweep, such as 1,1.25,1.5")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP socket inactivity timeout in seconds")
    parser.add_argument("--progress-window-ms", type=float, default=GAP_MS)
    parser.add_argument("--target-active", type=int, help="concurrency target, defaulting to the qualification unit count")
    parser.add_argument("--qualify", choices=("sessions", "tasks", "streams", "load", "cold"),
                        help="apply service gates; cold qualifies one-shot cold-start traffic")
    parser.add_argument("--milestone-s", type=float,
                        help="use a development milestone instead of the production soak (no duration for cold)")
    parser.add_argument("--min-turns-per-s", type=float, help="minimum offered turn rate for qualification")
    parser.add_argument("--min-cache-samples", type=int,
                        help="raise the number of observed TTFT samples required in each cache cohort")
    parser.add_argument("--warmup-s", type=float, default=0, help="exclude offers during the first N seconds from measurements")
    parser.add_argument("--model", help="override the model in each request")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="bearer token environment variable")
    parser.add_argument("--request-format", choices=("native", "vllm"), default="native",
                        help="translate explicit DeepSeek thinking controls for vLLM")
    session_headers = ("X-Session-ID", "X-Dynamo-Session-ID")
    parser.add_argument("--session-header", choices=session_headers, default="X-Session-ID")
    parser.add_argument("--session-prefix", default="", help="optionally namespace session IDs for this run")
    parser.add_argument("--isolate-replays", action="store_true",
                        help="start each trial in a fresh session namespace and vLLM cache salt")
    parser.add_argument("--gpus", type=int, help="report goodput normalized by this many GPUs")
    parser.add_argument("--inventory", type=Path, help="pinned replica and physical GPU evidence for this endpoint")
    parser.add_argument("--baseline-url", help="replay the same trace against this endpoint after the candidate")
    parser.add_argument("--baseline-model", help="override the model for the baseline")
    parser.add_argument("--baseline-api-key-env", help="baseline bearer token environment variable")
    parser.add_argument("--baseline-format", choices=("native", "vllm"), default="native")
    parser.add_argument("--baseline-session-header", choices=session_headers, default="X-Session-ID")
    parser.add_argument("--baseline-gpus", type=int, help="report baseline goodput normalized by this many GPUs")
    parser.add_argument("--baseline-inventory", type=Path, help="pinned replica and physical GPU evidence for the baseline")
    parser.add_argument("--event-at-s", type=float, help="emit a stderr marker for an operator-controlled event")
    parser.add_argument("--event-label", default="external")
    parser.add_argument("--event-recovery-s", type=float, default=0, help="delay after the marker before post-event measurement")
    parser.add_argument("--post-event-s", type=float, help="duration of the separately measured post-event window")
    args = parser.parse_args()
    for name in ("sessions", "tasks", "workers", "replenish_s", "target_active", "arrival_scale", "timeout", "progress_window_ms",
                 "milestone_s", "min_turns_per_s", "min_cache_samples", "gpus", "baseline_gpus", "post_event_s"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or not math.isfinite(value)):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    try:
        for value in (args.warmup_s, args.event_at_s, args.event_recovery_s):
            if value is not None:
                nonnegative_seconds(value)
        scales = [args.arrival_scale]
        if args.arrival_scales:
            if args.arrival_scale != 1:
                parser.error("use --arrival-scale or --arrival-scales")
            scales = sorted(set(float(value) for value in args.arrival_scales.split(",")))
            if any(not math.isfinite(value) or value <= 0 for value in scales):
                parser.error("--arrival-scales must contain positive finite numbers")
        args.endpoint = endpoint_identity(args.url)
        args.inventory = load_inventory(args.inventory, args.endpoint)
        if args.baseline_url:
            args.baseline_endpoint = endpoint_identity(args.baseline_url)
            args.baseline_inventory = load_inventory(args.baseline_inventory, args.baseline_endpoint)
        elif args.baseline_inventory is not None:
            parser.error("--baseline-inventory requires --baseline-url")
    except (OSError, ValueError) as err:
        parser.error(str(err))
    if args.mode == "closed" and (args.arrival_scale != 1 or args.arrival_scales):
        parser.error("arrival scaling requires --mode open")
    if args.replenish_s is not None and (args.mode != "closed" or args.sessions is not None):
        parser.error("--replenish-s requires --mode closed with whole tasks selected")
    if any(value is not None for value in (args.milestone_s, args.min_turns_per_s, args.min_cache_samples)) and not args.qualify:
        parser.error("--milestone-s, --min-turns-per-s and --min-cache-samples require --qualify")
    if any(ord(char) < 33 or ord(char) > 126 for char in args.session_prefix):
        parser.error("--session-prefix must use printable ASCII without spaces")
    if (args.event_at_s is None) != (args.post_event_s is None):
        parser.error("use --event-at-s and --post-event-s together")
    if args.event_at_s is not None and (len(scales) != 1 or args.baseline_url):
        parser.error("operator-controlled events require a single replay")
    production = args.qualify is not None and args.milestone_s is None
    for label, gpus, inventory in [("candidate", args.gpus, args.inventory),
                                   ("baseline", args.baseline_gpus, args.baseline_inventory)]:
        if label == "baseline" and not args.baseline_url:
            continue
        if production and (gpus is None or inventory is None):
            options = "--gpus and --inventory" if label == "candidate" else "--baseline-gpus and --baseline-inventory"
            parser.error(f"production qualification requires {options}")
        if gpus is not None and inventory is not None and gpus != inventory["gpu_count"]:
            parser.error(f"{label} GPU count does not match its pinned inventory")
    if args.qualify:
        if args.progress_window_ms != GAP_MS:
            parser.error("qualification requires a 250 ms progress window")
        if production and args.min_cache_samples is not None and args.min_cache_samples < PRODUCTION_CACHE_SAMPLES:
            parser.error(f"production qualification requires at least {PRODUCTION_CACHE_SAMPLES} samples per cache cohort")
        if production and args.qualify in ("sessions", "tasks", "streams"):
            if args.target_active not in (None, 500):
                parser.error("production qualification requires 500 active users; use --milestone-s for a smaller target")
            args.target_active = 500
        if args.qualify in ("sessions", "tasks") and args.mode != "closed":
            parser.error(f"{args.qualify} qualification requires --mode closed")
        if args.qualify == "tasks" and args.sessions is not None:
            parser.error("task qualification uses --tasks to retain every branch")
    try:
        sessions = load_sessions(args.manifest, args.mode, args.sessions, task_limit=args.tasks)
        workload = workload_summary(sessions, args.mode)
        formats = (args.request_format, args.baseline_format) if args.baseline_url else (args.request_format,)
        isolate = args.isolate_replays or args.replenish_s is not None
        for request_format in formats:
            if request_format == "vllm":
                for session in sessions:
                    for turn in session["turns"]:
                        request_body(turn, None, request_format, "validate" if isolate else None)
    except (OSError, ValueError) as err:
        parser.error(str(err))
    workers = args.workers or len(sessions)
    target_tasks = args.qualify == "tasks" or args.replenish_s is not None
    args.target_active = args.target_active or (workload["tasks"] if target_tasks else len(sessions))
    if args.qualify in ("sessions", "tasks", "streams") and workers < args.target_active:
        parser.error("qualification requires at least as many client workers as the active target")
    if args.qualify == "sessions" and (len(sessions) < args.target_active or workers < len(sessions)):
        parser.error("session qualification requires enough sessions for the target and a worker for each")
    if args.qualify == "tasks" and workload["tasks"] < args.target_active:
        parser.error("task qualification requires enough complete tasks for the active target")
    if args.replenish_s is not None and (workload["tasks"] < args.target_active or workers < args.target_active):
        parser.error("replenishment requires enough complete tasks and client workers for the active target")
    args.api_key = os.environ.get(args.api_key_env)
    args.opener = build_opener(NoRedirect())
    args.backend = "candidate"
    endpoints = [args]
    if args.baseline_url:
        baseline = argparse.Namespace(**vars(args))
        baseline.backend = "baseline"
        baseline.url = args.baseline_url
        baseline.endpoint = args.baseline_endpoint
        baseline.inventory = args.baseline_inventory
        baseline.model = args.baseline_model or args.model
        baseline.request_format = args.baseline_format
        baseline.session_header = args.baseline_session_header
        baseline.api_key = os.environ.get(args.baseline_api_key_env or args.api_key_env)
        baseline.gpus = args.baseline_gpus
        endpoints.append(baseline)
    reports = []
    for endpoint in endpoints:
        for scale in scales:
            trial = argparse.Namespace(**vars(endpoint))
            trial.arrival_scale = scale
            trial.cache_namespace = uuid4().hex if args.isolate_replays else None
            if trial.cache_namespace is not None:
                trial.session_prefix += trial.cache_namespace + ":"
            reports.append(run_case(trial, sessions, workload))
    if len(reports) == 1:
        report = reports[0]
    else:
        report = {"workload": workload, "runs": reports}
        if args.qualify:
            report["highest_passing_arrival_scale"] = {
                endpoint.backend: max((run["arrival_scale"] for run in reports
                                       if run["backend"] == endpoint.backend and not case_failed(run)), default=None)
                for endpoint in endpoints
            }
    tokens = {endpoint.api_key for endpoint in endpoints if endpoint.api_key}
    secrets = sorted(tokens | {quote(token, safe="") for token in tokens}, key=len, reverse=True)
    print(json.dumps(redact_secrets(report, secrets), indent=2, sort_keys=True))
    return int(any(case_failed(run) for run in reports))


if __name__ == "__main__":
    raise SystemExit(main())
