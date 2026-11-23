#!/usr/bin/env python3
"""Screen concurrent-decode proposals against the pinned engine's known limits.

This is a resource and demand calculation, not a serving simulator. In
particular, single-request iteration times are never extrapolated to batches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from math import ceil, isfinite
from pathlib import Path
from statistics import median
from typing import Any


MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_REQUESTS = 100_000
MAX_GRAPH_ROWS = 16  # crates/model/src/limits.rs:1
CONSERVATIVE_DECODE_ROWS = 8  # model.rs moe_rows_ok; >8 needs kernel qualification


@dataclass(frozen=True)
class Shape:
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    accept_length: float | None
    iteration_ms: float | None

    @property
    def resident_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        return self.input_tokens - self.cached_tokens


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shapes(path: Path) -> tuple[list[Shape], int, bool]:
    if path.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError("trace exceeds 64 MiB; pass a bounded evidence slice")
    shapes: list[Shape] = []
    omitted = 0
    worker_identity: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"trace line {line_number} is not an object")
            if "identity" in row:
                identity = row["identity"]
                if not isinstance(identity, dict):
                    raise ValueError(f"trace line {line_number} has invalid worker identity")
                if worker_identity is None:
                    worker_identity = identity
                elif identity != worker_identity:
                    raise ValueError("trace combines more than one worker incarnation")
            if row.get("event") != "request_end":
                continue
            if len(shapes) + omitted >= MAX_REQUESTS:
                raise ValueError("trace exceeds the 100,000-request probe limit")
            # Error responses may not have reached tokenization. Count them,
            # but do not invent their prompt lengths or throughput.
            if row.get("input_tokens") is None or row.get("output_tokens") is None:
                omitted += 1
                continue
            input_tokens = positive_int(row["input_tokens"], f"line {line_number} input_tokens")
            output_tokens = nonnegative_int(row["output_tokens"], f"line {line_number} output_tokens")
            cached_tokens = nonnegative_int(row.get("cached_tokens") or 0, f"line {line_number} cached_tokens")
            if cached_tokens > input_tokens:
                raise ValueError(f"line {line_number} cached_tokens exceeds input_tokens")
            shapes.append(Shape(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                accept_length=optional_positive_float(row.get("average_accept_length"), "average_accept_length"),
                iteration_ms=optional_positive_float(row.get("average_iteration_ms"), "average_iteration_ms"),
            ))
    if not shapes:
        raise ValueError("trace has no request_end entries with token counts")
    return shapes, omitted, worker_identity is not None


@dataclass(frozen=True)
class CacheGeometry:
    n_layers: int
    n_mtp_layers: int
    head_dim: int
    index_head_dim: int
    ratios: tuple[int, ...]
    sources: frozenset[int]

    @classmethod
    def load(cls, path: Path) -> CacheGeometry:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("model config must be an object")
        n_layers = positive_int(data.get("n_layers"), "n_layers")
        n_mtp_layers = nonnegative_int(data.get("n_mtp_layers"), "n_mtp_layers")
        ratios = data.get("compress_ratios")
        sources = data.get("kv_source_layers")
        if not isinstance(ratios, list) or len(ratios) != n_layers + n_mtp_layers:
            raise ValueError("compress_ratios must cover every target and MTP layer")
        if not isinstance(sources, list):
            raise ValueError("kv_source_layers must be an array")
        parsed_ratios = tuple(nonnegative_int(value, "compress ratio") for value in ratios)
        parsed_sources = frozenset(nonnegative_int(value, "KV source layer") for value in sources)
        if len(parsed_sources) != len(sources) or any(value >= n_layers for value in parsed_sources):
            raise ValueError("KV source layers must be distinct target-layer indexes")
        if any(parsed_ratios[index] == 0 for index in parsed_sources):
            raise ValueError("KV source layers need nonzero compression ratios")
        return cls(
            n_layers=n_layers,
            n_mtp_layers=n_mtp_layers,
            head_dim=positive_int(data.get("head_dim"), "head_dim"),
            index_head_dim=positive_int(data.get("index_head_dim"), "index_head_dim"),
            ratios=parsed_ratios,
            sources=parsed_sources,
        )

    def explicit_cache_bytes(self, tokens: int) -> int:
        """Arrays allocated at model.rs:682-696 for one rank and one sequence.

        The actual engine allocates these arrays at --max-seq, even for short
        requests. Summing right-sized copies is hypothetical until per-request
        KV ownership exists. Weights, scratch, rope tables and graphs are absent.
        """
        positive_int(tokens, "tokens")
        hd = self.head_dim
        total = tokens * 4  # cmp_hist
        for layer, ratio in enumerate(self.ratios):
            total += tokens * hd * 2  # win, including non-source layers
            if layer not in self.sources:
                continue
            compressed = tokens // ratio
            total += compressed * hd * 2  # comp
            total += compressed * self.index_head_dim * 2  # idx_k
            total += compressed * 64 + (compressed + 127) // 128 * 512  # idx_k4 pair
            if ratio > 1:
                total += 2 * tokens * hd * 4  # kv_hist and sc_hist
        return total


def evaluate(
    shapes: list[Shape], geometry: CacheGeometry, active_counts: tuple[int, ...],
    *, max_seq: int, max_verify_rows: int, draft_tokens: int,
    accepted_tokens_per_step: float, first_useful_ms: float,
    minimum_visible_tps: float, explicit_cache_budget_bytes: int | None,
) -> dict[str, Any]:
    positive_int(max_seq, "max_seq")
    positive_int(max_verify_rows, "max_verify_rows")
    nonnegative_int(draft_tokens, "draft_tokens")
    if max_verify_rows > MAX_GRAPH_ROWS or draft_tokens >= MAX_GRAPH_ROWS:
        raise ValueError("verification plan exceeds the current 16-row graph limit")
    if not isfinite(accepted_tokens_per_step) or not 0 < accepted_tokens_per_step <= draft_tokens + 1:
        raise ValueError("accepted_tokens_per_step must be between zero and draft_tokens + 1")
    if not isfinite(first_useful_ms) or first_useful_ms <= 0:
        raise ValueError("first_useful_ms must be positive and finite")
    if not isfinite(minimum_visible_tps) or minimum_visible_tps <= 0:
        raise ValueError("minimum_visible_tps must be positive and finite")
    if explicit_cache_budget_bytes is not None:
        positive_int(explicit_cache_budget_bytes, "explicit_cache_budget_bytes")
    if any(shape.resident_tokens > max_seq for shape in shapes):
        raise ValueError("at least one logged request exceeds the proposed max_seq after output")
    observed_steps = [shape.iteration_ms for shape in shapes if shape.iteration_ms is not None]
    observed_accept = [shape.accept_length for shape in shapes if shape.accept_length is not None]
    rows_per_sequence = draft_tokens + 1
    if rows_per_sequence > max_verify_rows:
        raise ValueError("draft verification rows exceed the chosen decode row budget")
    slots_per_step = max_verify_rows // rows_per_sequence
    scenarios = []
    for active in active_counts:
        positive_int(active, "active count")
        if active > len(shapes):
            raise ValueError(f"active count {active} exceeds {len(shapes)} measured request shapes")
        longest = sorted(shapes, key=lambda shape: shape.resident_tokens, reverse=True)[:active]
        all_cold = sorted(shapes, key=lambda shape: shape.input_tokens, reverse=True)[:active]
        coldest = sorted(shapes, key=lambda shape: shape.uncached_input_tokens, reverse=True)[:active]
        right_sized_copy_bytes = sum(geometry.explicit_cache_bytes(shape.resident_tokens) for shape in longest)
        waves = ceil(active / slots_per_step)
        all_cold_tokens = sum(shape.input_tokens for shape in all_cold)
        observed_uncached_tokens = sum(shape.uncached_input_tokens for shape in coldest)
        scenarios.append({
            "active_sequences": active,
            "verification_rows_per_sequence": rows_per_sequence,
            "sequences_per_decode_step_at_row_limit": min(active, slots_per_step),
            "decode_round_robin_waves": waves,
            "required_max_decode_step_ms_for_speed_floor_before_prefill": (
                1000 * accepted_tokens_per_step / (minimum_visible_tps * waves)
            ),
            "hypothetical_right_sized_copy_cache_bytes_per_rank": right_sized_copy_bytes,
            "copy_cache_exceeds_supplied_budget": (
                right_sized_copy_bytes > explicit_cache_budget_bytes
                if explicit_cache_budget_bytes is not None else None
            ),
            "synchronized_prompt_tokens_all_cold": all_cold_tokens,
            "required_prefill_tokens_per_second_all_cold": all_cold_tokens * 1000 / first_useful_ms,
            "synchronized_uncached_prompt_tokens_observed_cache": observed_uncached_tokens,
            "required_prefill_tokens_per_second_observed_cache": (
                observed_uncached_tokens * 1000 / first_useful_ms
            ),
            "required_visible_output_tokens_per_second": active * minimum_visible_tps,
        })
    return {
        "kind": "resource_and_demand_probe_not_gpu_performance_prediction",
        "request_shapes": len(shapes),
        "explicit_cache_bytes_per_rank_at_current_single_max_seq": geometry.explicit_cache_bytes(max_seq),
        "observed_single_request_median_iteration_ms": median(observed_steps) if observed_steps else None,
        "observed_single_request_median_accept_length": median(observed_accept) if observed_accept else None,
        "assumed_accepted_tokens_per_batch_step": accepted_tokens_per_step,
        "assumptions": [
            "The active set is assembled from the longest or coldest saved requests, not a measured simultaneous run.",
            "The memory scenario copies today's explicit arrays per request at right-sized lengths; shared-prefix or paged layouts could differ. Current Rank has one fixed max_seq allocation.",
            "The row budget is a chosen kernel limit; GPU batch step time and accepted length under load remain unmeasured.",
            "Rows above eight need separate kernel parity: current low-latency MoE and GEMV paths guard eight rows.",
            "Prefill demand is shown both with all prompts cold and with saved cached_tokens; neither includes chat-template, routing, network, or decode time.",
        ],
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, help="bounded worker JSONL with request_end events")
    parser.add_argument("--config", required=True, type=Path, help="the exact checkpoint's inference/config.json")
    parser.add_argument("--active", default="1,2,4,8,16", help="comma-separated simultaneous request counts")
    parser.add_argument("--max-seq", type=int, required=True, help="engine --max-seq setting")
    parser.add_argument("--max-verify-rows", type=int, default=CONSERVATIVE_DECODE_ROWS, help="qualified kernel row limit; default 8, graph hard cap 16")
    parser.add_argument("--draft-tokens", type=int, required=True, help="speculative draft tokens per request; zero is plain decode")
    parser.add_argument("--accepted-tokens-per-step", type=float, default=1.0, help="assumption, not extrapolated from a single-request trace")
    parser.add_argument("--first-useful-ms", type=float, default=2000.0)
    parser.add_argument("--minimum-visible-tps", type=float, default=80.0)
    parser.add_argument("--explicit-cache-budget-bytes", type=int,
                        help="operator budget per rank for only the explicit cache arrays")
    args = parser.parse_args()
    try:
        active = tuple(int(value) for value in args.active.split(","))
        if not active or len(set(active)) != len(active):
            raise ValueError("active counts must be nonempty and distinct")
        shapes, omitted, identity_bound = load_shapes(args.trace)
        report = evaluate(
            shapes, CacheGeometry.load(args.config), active,
            max_seq=args.max_seq, max_verify_rows=args.max_verify_rows,
            draft_tokens=args.draft_tokens, accepted_tokens_per_step=args.accepted_tokens_per_step,
            first_useful_ms=args.first_useful_ms, minimum_visible_tps=args.minimum_visible_tps,
            explicit_cache_budget_bytes=args.explicit_cache_budget_bytes,
        )
        report["trace_sha256"] = sha256(args.trace)
        report["config_sha256"] = sha256(args.config)
        report["single_worker_identity_bound"] = identity_bound
        report["request_end_without_token_counts"] = omitted
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"concurrency-probe: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
