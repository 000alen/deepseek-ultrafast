#!/usr/bin/env python3
"""Run the single-node B200 rehearsal matrix and keep operator evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Thread
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

import bench_coding
from b200_workload import sha256


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "deploy/b200-test/launch.sh"
PROFILES = {
    "native-tp4": ("native-tp4", 4, "on"),
    "native-tp4-no-rate": ("native-tp4", 4, "off"),
    "native-dual-tp4": ("native-dual-tp4", 8, "on"),
    "native-dual-tp4-no-rate": ("native-dual-tp4", 8, "off"),
    "native-tp8": ("native-tp8", 8, "on"),
    "vllm-dep8": ("vllm-dep8", 8, "on"),
}
PHASES = ("closed", "open", "kickoff", "fault")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def cases(args):
    for phase in args.phase:
        levels = args.levels if phase == "closed" else args.levels[-1:]
        for level in levels:
            yield phase, level


def benchmark_command(args, phase, level, profile, endpoint, inventory):
    _, gpus, _ = PROFILES[profile]
    manifest = "kickoff.json" if phase == "kickoff" else "mixed.json"
    workload = "tasks" if phase == "closed" else "cold" if phase == "kickoff" else "load"
    mode = "closed" if phase == "closed" else "open"
    command = [sys.executable, str(ROOT / "tools/bench_coding.py"), str(args.workloads / manifest),
               "--url", endpoint, "--gpus", str(gpus), "--inventory", str(inventory),
               "--model", args.model, "--mode", mode, "--tasks", str(level),
               "--qualify", workload, "--milestone-s", str(args.milestone_s),
               "--min-cache-samples", str(args.min_cache_samples), "--isolate-replays"]
    if profile == "vllm-dep8":
        command += ["--request-format", "vllm"]
    if phase == "closed":
        command += ["--target-active", str(level)]
        if args.replenish_s:
            command += ["--replenish-s", str(args.replenish_s)]
        if args.warmup_s:
            command += ["--warmup-s", str(args.warmup_s)]
    if phase == "open":
        command += ["--arrival-scales", args.arrival_scales]
    if phase == "fault":
        command += ["--event-at-s", str(args.fault_at_s), "--event-label", "one-tp4-replica-loss",
                    "--event-recovery-s", str(args.fault_recovery_s), "--post-event-s", str(args.post_fault_s)]
    if args.min_offered_rate is not None and phase != "kickoff":
        command += ["--min-turns-per-s", str(args.min_offered_rate)]
    return command


def run_case(command, destination, fault=None):
    with (destination / "report.json").open("w") as report, (destination / "replay.log").open("w") as log:
        if fault is None:
            return subprocess.run(command, stdout=report, stderr=log).returncode
        with subprocess.Popen(command, stdout=report, stderr=subprocess.PIPE, text=True) as process:
            injected = False
            try:
                for line in process.stderr:
                    log.write(line)
                    log.flush()
                    if line.startswith('{"benchmark_event":') and not injected:
                        event = json.loads(line)["benchmark_event"]
                        receipt = fault()
                        write_json(destination / "fault.json", {"marker": event, "action": receipt})
                        injected = True
            finally:
                if not injected and process.poll() is None:
                    process.terminate()
            status = process.wait()
        if not injected:
            raise RuntimeError("replay ended before the replica-stop marker")
        return status


def sample_resources(path, inventory, stop, interval):
    urls = [replica["admission_url"] for replica in inventory["replicas"] if replica.get("admission_url")]
    urls = [url for url in urls if urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1")]
    opener = build_opener(ProxyHandler({}), bench_coding.NoRedirect())
    with path.open("w") as output:
        while not stop.is_set():
            sample = {"at": now(), "admission": {}}
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total,utilization.gpu,temperature.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
                sample["gpus"] = [line.split(", ") for line in result.stdout.splitlines()]
            except (OSError, subprocess.SubprocessError):
                sample["gpu_sample_failed"] = True
            try:
                host_memory = Path("/proc/meminfo").read_text().splitlines()
                sample["host_memory"] = dict(line.split(":", 1) for line in host_memory
                                             if line.startswith(("MemAvailable:", "MemTotal:", "Mlocked:")))
            except OSError:
                pass
            for url in urls:
                try:
                    with opener.open(url, timeout=1) as response:
                        sample["admission"][url] = json.load(response)
                except (OSError, ValueError):
                    sample["admission"][url] = None
            print(json.dumps(sample, sort_keys=True), file=output, flush=True)
            stop.wait(interval)


def summarize(report, phase, level):
    items = report.get("runs", [report])
    return {"phase": phase, "offered_tasks": level, "runs": [{
        "arrival_scale": item.get("arrival_scale"), "offered_turns_per_s": item.get("offered_turns_per_s"),
        "tasks": item.get("tasks"), "live_tasks": item.get("live_tasks"),
        "joint_slo_goodput": item.get("joint_slo_goodput"), "success_rate": item.get("success_rate"),
        "qualification": item.get("qualification"),
        "post_event_qualification": item.get("event", {}).get("post_event", {}).get("qualification"),
        "passed": not bench_coding.case_failed(item),
    } for item in items]}


def passing(results):
    largest = max((case["offered_tasks"] for case in results if case["phase"] == "closed"
                   and all(run["passed"] for run in case["runs"])), default=None)
    scale = max((run["arrival_scale"] for case in results if case["phase"] == "open" for run in case["runs"]
                 if run["passed"]), default=None)
    return {"highest_passing_closed_task_count": largest, "highest_passing_open_arrival_scale": scale}


def run_profile(args, profile, destination, results):
    mode, expected_gpus, limiter = PROFILES[profile]
    destination.mkdir(mode=0o700)
    state = destination / "runtime"
    env = {**os.environ, "DSUF_B200_RUN_DIR": str(state),
           "DSUF_B200_RATE_LIMIT": limiter, "DSUF_B200_TRACE": "1" if args.trace else "0"}

    def launch(*command, capture=False):
        return subprocess.run([str(LAUNCH), *command], env=env, check=True,
                              stdout=subprocess.PIPE if capture else sys.stderr, text=True).stdout

    print(f"Starting {profile} ({expected_gpus} B200 GPUs)", file=sys.stderr, flush=True)
    try:
        launch("up", mode)
        status = json.loads(launch("status", capture=True))
        inventory = json.loads(launch("inventory", "--checkpoint-revision", args.checkpoint_revision, capture=True))
        endpoint = status["endpoint"]
        if urlsplit(endpoint).hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the B200 test gateway must use a loopback address")
        write_json(destination / "startup.json", status)
        inventory_path = destination / "inventory.json"
        write_json(inventory_path, inventory)
        observed = bench_coding.load_inventory(inventory_path, bench_coding.endpoint_identity(endpoint))
        if observed["gpu_count"] != expected_gpus:
            raise ValueError("the B200 inventory differs from the selected GPU topology")
        stop = Event()
        sampling = Thread(target=sample_resources, args=(destination / "resources.jsonl", inventory, stop,
                                                        args.sample_interval_s))
        sampling.start()
        try:
            for phase, level in cases(args):
                if phase == "fault" and mode != "native-dual-tp4":
                    continue
                print(f"{profile} · {phase} · {level} coding tasks", file=sys.stderr, flush=True)
                case = destination / f"{phase}-{level}"
                case.mkdir(mode=0o700)
                command = benchmark_command(args, phase, level, profile, endpoint, inventory_path)
                write_json(case / "command.json", command)

                def stop_replica():
                    receipt = json.loads(launch("fault", "1", capture=True))
                    if receipt.get("confirmed_down") is not True:
                        raise RuntimeError("failed to confirm the test replica stopped")
                    return receipt

                result = run_case(command, case, stop_replica if phase == "fault" else None)
                if result not in (0, 1):
                    raise RuntimeError(f"benchmark exited {result}. See {case / 'replay.log'}")
                report = json.loads((case / "report.json").read_text())
                results.append(summarize(report, phase, level))
                write_json(destination / "summary.json", {"results": results, **passing(results)})
        finally:
            stop.set()
            sampling.join()
    finally:
        teardown = json.loads(launch("down", capture=True))
        write_json(destination / "teardown.json", teardown)
    if args.trace and mode != "vllm-dep8":
        archive = teardown.get("logs")
        if not archive:
            raise RuntimeError("no archived container logs for the requested timing analysis")
        with (destination / "timing.json").open("w") as report:
            subprocess.run([sys.executable, str(ROOT / "tools/b200_timing.py"), archive],
                           stdout=report, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, required=True, help="directory from b200_workload.py")
    parser.add_argument("--output", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--checkpoint-revision", required=True, help="pinned 40- or 64-digit checkout or artifact hash")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--phase", action="append", choices=PHASES)
    parser.add_argument("--levels", default="100,250,500", help="comma-separated coding task counts")
    parser.add_argument("--milestone-s", type=float, default=120)
    parser.add_argument("--warmup-s", type=float, default=0, help="exclude initial closed-loop time from scoring")
    parser.add_argument("--replenish-s", type=float, help="closed-loop replacement duration, defaults to warmup + milestone + 60 s. Use 0 for one pass")
    parser.add_argument("--min-cache-samples", type=int, default=20)
    parser.add_argument("--min-offered-rate", type=float, help="request/s floor taken from the real trace")
    parser.add_argument("--arrival-scales", default="1,1.25,1.5")
    parser.add_argument("--sample-interval-s", type=float, default=10)
    parser.add_argument("--fault-at-s", type=float, help="stop the second TP4 replica at the benchmark marker")
    parser.add_argument("--fault-recovery-s", type=float, default=30)
    parser.add_argument("--post-fault-s", type=float, default=120)
    parser.add_argument("--trace", action="store_true", help="enable high-volume packed tracing for short runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.profile = args.profile or ["native-tp4", "native-dual-tp4", "vllm-dep8"]
    args.phase = args.phase or ["closed", "open", "kickoff"]
    args.workloads = args.workloads.resolve()
    args.output = args.output.resolve()
    try:
        args.levels = sorted(set(map(int, args.levels.split(","))))
        if len(set(args.profile)) != len(args.profile) or len(set(args.phase)) != len(args.phase):
            raise ValueError("profiles and phases must be unique")
        if not args.levels or args.levels[0] <= 0 or args.min_cache_samples <= 0:
            raise ValueError("levels and cache sample count must be positive")
        for name in ("milestone_s", "warmup_s", "replenish_s", "sample_interval_s", "fault_recovery_s",
                     "post_fault_s", "min_offered_rate", "fault_at_s"):
            if (value := getattr(args, name)) is not None:
                bench_coding.nonnegative_seconds(value)
                if value == 0 and name not in ("fault_recovery_s", "warmup_s", "replenish_s"):
                    raise ValueError(f"{name} must be positive")
        if args.replenish_s is None:
            args.replenish_s = args.warmup_s + args.milestone_s + 60
        for scale in args.arrival_scales.split(","):
            if bench_coding.nonnegative_seconds(float(scale)) == 0:
                raise ValueError("arrival scales must be positive")
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.checkpoint_revision):
            raise ValueError("checkpoint revision must be a lowercase 40- or 64-digit hash")
        if "fault" in args.phase:
            if args.fault_at_s is None or not any(PROFILES[p][0] == "native-dual-tp4" for p in args.profile):
                raise ValueError("fault phase requires --fault-at-s and a dual TP4 profile")
            args.phase.remove("fault")
            args.phase.append("fault")
        if args.dry_run:
            port = int(os.environ.get("DSUF_B200_GATEWAY_PORT", "18080"))
            endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
            print(json.dumps({p: [benchmark_command(args, phase, level, p, endpoint,
                                                   args.output / p / "inventory.json") for phase, level in cases(args)
                                  if phase != "fault" or PROFILES[p][0] == "native-dual-tp4"]
                              for p in args.profile}, indent=2))
            return 0
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("set OPENAI_API_KEY to the rehearsal gateway token")
        if args.trace and args.milestone_s >= 1800:
            print("Plan tracing can produce large logs during a soak.", file=sys.stderr)
        workload = json.loads((args.workloads / "workload.json").read_text())
        validated = {}
        for phase, level in cases(args):
            name = "kickoff" if phase == "kickoff" else "mixed"
            replay_mode = "closed" if phase == "closed" else "open"
            key = (name, replay_mode)
            if key not in validated:
                manifest = args.workloads / f"{name}.json"
                if sha256(manifest) != workload[f"{name}_sha256"]:
                    raise ValueError(f"{manifest.name} no longer matches the prepared workload digest")
                sessions = bench_coding.load_sessions(manifest, replay_mode, None)
                validated[key] = len({bench_coding.task_id(session) for session in sessions})
            if validated[key] < level:
                raise ValueError(f"{name}.json has fewer than {level} distinct tasks")
        args.output.mkdir(mode=0o700, parents=True)
        result = {"started_at": now(), "workload": workload, "checkpoint_revision": args.checkpoint_revision,
                  "closed_replenish_s": args.replenish_s, "closed_warmup_s": args.warmup_s,
                  "profiles": {}, "passing": {}, "complete": False}
        for profile in args.profile:
            if args.phase == ["fault"] and PROFILES[profile][0] != "native-dual-tp4":
                continue
            profile_results = result["profiles"].setdefault(profile, [])
            try:
                run_profile(args, profile, args.output / profile, profile_results)
            finally:
                result["passing"][profile] = passing(profile_results)
                write_json(args.output / "summary.json", result)
        result.update(complete=True, finished_at=now())
        write_json(args.output / "summary.json", result)
        print(args.output / "summary.json")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"B200 rehearsal failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
