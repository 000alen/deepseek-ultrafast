#!/usr/bin/env python3
"""Collect replica and GPU evidence from a deployed Kubernetes qualification overlay."""

import argparse
import csv
import json
import re
import subprocess
from urllib.parse import urlsplit


def argument(args, name, default):
    return args[args.index(name) + 1] if name in args else default


def startup_event(logs, event):
    for line in reversed(logs.splitlines()):
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("event") == event:
            return item
    raise ValueError(f"required startup event {event} was absent")


def inventory_replica(kubectl, namespace, pod, backend, checkpoint, gpu_model):
    name = pod["metadata"]["name"]
    spec = pod["spec"]
    engine = next((c for c in spec["containers"] if "nvidia.com/gpu" in
                   c.get("resources", {}).get("limits", {})), None)
    if engine is None:
        raise ValueError(f"{name} has no GPU engine container")
    container = engine["name"]
    status = next((s for s in pod["status"].get("containerStatuses", [])
                   if s["name"] == container), {})
    if not status.get("ready"):
        raise ValueError(f"{name} is still starting")
    digest = re.search(r"sha256:[a-f0-9]{64}", status.get("imageID", ""))
    if digest is None:
        raise ValueError(f"{name} did not report an image digest")

    smi = kubectl("exec", name, "-c", container, "--", "nvidia-smi",
                  "--query-gpu=uuid,name", "--format=csv,noheader")
    hardware = list(csv.reader(smi.splitlines(), skipinitialspace=True))
    if len(hardware) != int(engine["resources"]["limits"]["nvidia.com/gpu"]):
        raise ValueError(f"{name} does not see its declared full-GPU allocation")
    model_pattern = re.compile(rf"\b{re.escape(gpu_model)}\b", re.IGNORECASE)
    if any(len(row) != 2 or not row[0].startswith("GPU-") or not model_pattern.search(row[1])
           for row in hardware):
        raise ValueError(f"{name} does not report the requested {gpu_model} GPU model and UUIDs")

    args = engine.get("args", [])
    startup = {"ready": True, "image_digest": digest[0], "checkpoint_revision": checkpoint}
    decode_width = None
    if backend == "native":
        tp = len(argument(args, "--gpus", "").split(","))
        dp = ep = 1
        port = argument(args, "--port", "8000")
        path = f"/api/v1/namespaces/{namespace}/pods/{name}:{port}/proxy/internal/admission"
        admission = json.loads(kubectl("get", "--raw", path))
        epoch = admission["epoch"]
        if not epoch:
            raise ValueError(f"{name} reported an empty admission epoch")
        packed = int(argument(args, "--max-num-seqs", "1")) > 1
        source = "packed_decode_ready" if packed else "model_loading_end"
        event = startup_event(kubectl("logs", name, "-c", container), source)
        if packed:
            for field in ("max_residents", "max_decode_width", "slot_bytes_per_rank", "kv_bytes_per_rank"):
                startup[field] = event[field]
            decode_width = event["max_decode_width"]
        elif event.get("success") is not True:
            raise ValueError(f"{name} did not finish model loading successfully")
    else:
        tp = int(argument(args, "--tensor-parallel-size", "1"))
        dp = int(argument(args, "--data-parallel-size", "1"))
        ep = tp * dp if "--enable-expert-parallel" in args else 1
        source = "kubernetes_ready"
        epoch = f'{pod["metadata"]["uid"]}:{status.get("restartCount", 0)}'

    startup["source"] = source
    if tp * dp != len(hardware):
        raise ValueError(f"{name} has a different GPU count from its tensor/data parallel shape")
    return {
        "id": name,
        "host": spec["nodeName"],
        "gpu_uuids": [row[0] for row in hardware],
        "epoch": epoch,
        "decode_width": decode_width,
        "topology": {"backend": backend, "stage": "combined", "tp": tp, "dp": dp, "ep": ep},
        "startup": startup,
    }


def collect(kubectl, namespace, endpoint, backend, checkpoint, gpu_model):
    pods = json.loads(kubectl("get", "pods", "-l", "app=dsuf-engine", "-o", "json"))["items"]
    replicas = [inventory_replica(kubectl, namespace, pod, backend, checkpoint, gpu_model)
                for pod in pods if not pod["metadata"].get("deletionTimestamp")]
    gpu_ids = [uuid for replica in replicas for uuid in replica["gpu_uuids"]]
    if not replicas or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("expected ready replicas with unique GPU assignments")
    return {"schema_version": 1, "endpoint": endpoint,
            "replicas": sorted(replicas, key=lambda replica: replica["id"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--endpoint", required=True, help="benchmark Chat Completions URL")
    parser.add_argument("--backend", required=True, choices=("native", "vllm"))
    parser.add_argument("--checkpoint-revision", required=True, help="staged 40- or 64-digit checkpoint SHA")
    parser.add_argument("--gpu-model", default="B300")
    args = parser.parse_args()
    endpoint = urlsplit(args.endpoint)
    if (endpoint.scheme not in ("http", "https") or not endpoint.hostname
            or endpoint.username is not None or endpoint.query or endpoint.fragment):
        parser.error("endpoint must be a clean HTTP or HTTPS URL")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.checkpoint_revision):
        parser.error("checkpoint revision must be a lowercase Git or SHA-256 hash")

    def kubectl(*command):
        return subprocess.run(
            ["kubectl", "-n", args.namespace, "--request-timeout=30s", *command],
            check=True, stdout=subprocess.PIPE, text=True, timeout=40,
        ).stdout

    try:
        report = collect(kubectl, args.namespace, args.endpoint, args.backend,
                         args.checkpoint_revision, args.gpu_model)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
