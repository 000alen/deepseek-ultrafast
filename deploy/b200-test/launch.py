#!/usr/bin/env python3
"""Launch single-host B200 qualification containers and collect their evidence."""

import argparse
import csv
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[2]
MODES = ("native-tp4", "native-dual-tp4", "native-tp8", "vllm-dep8")
GIB = 1 << 30
GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
VLLM_IMAGE = "vllm/vllm-openai@sha256:13e92fcd97268013d13d07fb97d80803227cf0beae0161f8dce95a8200eaafb0"
REDIS_IMAGE = "redis:7"
NATIVE_IMAGE = "deepseek-ultrafast:b200-test"
GATEWAY_IMAGE = "deepseek-ultrafast-gateway:b200-test"
PARITY_IMAGE = "deepseek-ultrafast-b200-parity:local"
OPENER = build_opener(ProxyHandler({}))


def command(*args, required=True, timeout=60, stdout=subprocess.PIPE, stderr=None, env=None):
    return subprocess.run(
        args, check=required, timeout=timeout, text=True, stdout=stdout, stderr=stderr, env=env,
    )


def docker(*args, **options):
    return command("docker", *args, **options)


def setting(name, default, lower, upper):
    value = int(os.environ.get("DSUF_B200_" + name, default))
    if not lower <= value <= upper:
        raise ValueError(f"DSUF_B200_{name} must be between {lower} and {upper}")
    return value


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def url(port, path):
    return f"http://127.0.0.1:{port}{path}"


def get(address):
    with OPENER.open(address, timeout=2) as response:
        return response.read()


def checkpoint():
    path = os.environ.get("DSUF_B200_MODEL_DIR")
    if not path:
        raise ValueError("set DSUF_B200_MODEL_DIR to the staged checkpoint directory")
    model = Path(path).resolve()
    for file in ("config.json", "inference/config.json", "tokenizer.json"):
        if not (model / file).is_file():
            raise ValueError(f"checkpoint is missing {file}: {model}")
    weights = list(model.glob("*.safetensors"))
    if not any(path.is_file() and path.stat().st_size for path in weights):
        raise ValueError(f"checkpoint has no safetensors: {model}")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing = [file for file in shards if not (model / file).is_file() or not (model / file).stat().st_size]
        if missing:
            raise ValueError(f"checkpoint has {len(missing)} missing or empty indexed shards")
    source, _ = checkpoint_mount(model)
    required = [model / file for file in ("config.json", "inference/config.json", "tokenizer.json")]
    inputs = [model / "inference", *required, *weights]
    if any(not file.resolve().is_relative_to(source)
           or (file.is_symlink() and Path(os.readlink(file)).is_absolute()) for file in inputs):
        raise ValueError("checkpoint symlinks leave the mounted directory; stage a self-contained local copy")
    return model


def checkpoint_mount(model):
    if (model.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", model.name)
            and (model.parent.parent / "blobs").is_dir()):
        return model.parent.parent, f"/models/snapshots/{model.name}"
    return model, "/models"


def mount(source, target):
    if "," in str(source):
        raise ValueError(f"Docker bind path contains a comma: {source}")
    return f"type=bind,source={source},destination={target},readonly"


def gpu_arg(uuids):
    return '"device=' + ",".join(uuids) + '"'


def gpu_flags(uuids):
    return ["--gpus", gpu_arg(uuids), "--env", "NVIDIA_VISIBLE_DEVICES=" + ",".join(uuids)]


def detect_gpus():
    result = command("nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current",
                     "--format=csv,noheader,nounits")
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    cards = {row[0]: row for row in rows if len(row) == 6}
    selected = os.environ.get("DSUF_B200_GPU_IDS", "").split(",")
    if selected == [""]:
        selected = sorted(cards, key=int)
    if len(selected) != 8 or len(set(selected)) != 8 or any(i not in cards for i in selected):
        raise ValueError("DSUF_B200_GPU_IDS must identify eight distinct physical GPU indices")
    gpus = []
    for index in selected:
        _, uuid, model, total, free, mig = cards[index]
        if not GPU_UUID.fullmatch(uuid) or not re.search(r"\bB200\b", model) or mig == "Enabled":
            raise ValueError(f"GPU {index} must be a full B200 with MIG disabled")
        gpus.append({"index": int(index), "uuid": uuid, "model": model,
                     "memory_total_mib": int(total), "memory_free_mib": int(free)})
    return gpus


def check_gpu_idle(gpus):
    uuids = {gpu["uuid"] for gpu in gpus}
    processes = command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits")
    busy = [row for row in csv.reader(processes.stdout.splitlines(), skipinitialspace=True)
            if row and row[0] in uuids]
    if busy:
        raise ValueError(f"selected GPUs already have compute processes: {busy}")


def rdma_ip():
    override = os.environ.get("DSUF_B200_KV_HOST")
    available = {ipaddress.ip_address(value) for value in command("hostname", "-I").stdout.split()}
    if override:
        address = ipaddress.ip_address(override)
        if address not in available or address.is_global or address.is_loopback or address.is_link_local:
            raise ValueError("DSUF_B200_KV_HOST must be a private IP assigned to this host")
        return str(address)
    choices = [str(ip) for ip in available if ip.is_private and isinstance(ip, ipaddress.IPv4Address)
               and not ip.is_loopback and not ip.is_link_local]
    if len(choices) != 1:
        raise ValueError("set DSUF_B200_KV_HOST to this host's private IP reachable through the RDMA setup")
    return choices[0]


def rdma_args():
    if not Path("/dev/infiniband").is_dir():
        raise ValueError("/dev/infiniband is unavailable on this host")
    environment = ["--env", f"DSUF_KV_HOST={rdma_ip()}"]
    for name in ("UCX_NET_DEVICES", "UCX_TLS"):
        if name in os.environ:
            environment.extend(["--env", name])
    return ["--device", "/dev/infiniband", "--ulimit", "memlock=-1:-1", *environment]


def selected_groups(mode, gpus):
    ids = [gpu["uuid"] for gpu in gpus]
    return {
        "native-tp4": [ids[:4]],
        "native-dual-tp4": [ids[:4], ids[4:]],
        "native-tp8": [ids],
        "vllm-dep8": [ids],
    }[mode]


def native_options(port):
    seq = setting("MAX_SEQ", 131072, 8257, 1048576)
    resident = setting("MAX_RESIDENTS", 64, 2, 128)
    width = setting("DECODE_WIDTH", 32, 1, resident)
    active = setting("MAX_ACTIVE", width, 1, resident)
    kv = setting("KV_GIB", 64, 1, 160)
    hot = setting("HOT_KV_GIB", min(16, kv), 0, kv)
    limiter = os.environ.get("DSUF_B200_RATE_LIMIT", "on")
    if limiter not in ("on", "off"):
        raise ValueError("DSUF_B200_RATE_LIMIT must be on or off")
    options = [
        "--host", "127.0.0.1", "--port", str(port), "--max-seq", str(seq),
        "--max-num-seqs", str(resident), "--max-num-decode-seqs", str(width),
        "--max-num-active-seqs", str(active), "--kv-cache-memory-bytes", str(kv * GIB),
        "--hot-kv-cache-memory-bytes", str(hot * GIB), "--kv-copy-mib", "8",
        "--max-outstanding-jobs", "96", "--standard-pool-max-output", "8192",
        "--decode-max-gap-ms", "250", "--max-prefill-wait-ms", "50",
    ]
    if limiter == "on":
        options.extend(["--decode-min-tokens-per-sec", "100"])
    return options


def vllm_options(port, model="/models"):
    graph_sizes = [6, 12, 18, 24, 30, 36, 48, 60, 72, 96, 120, 144, 192, 240,
                   288, 384, 480, 576, 768, 1020, 1536, 2046, 3072, 4092, 6144, 8190]
    compilation = dict(mode="VLLM_COMPILE", cudagraph_mode="FULL_AND_PIECEWISE",
                       cudagraph_capture_sizes=graph_sizes,
                       decoder_replay_cudagraph_capture_sizes=[
                           *graph_sizes[:21], 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096])
    context = setting("MAX_SEQ", 131072, 8257, 1048576)
    kv = setting("VLLM_KV_GIB", 40, 16, 48)
    return [
        "serve", model, "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", "deepseek-v4.1-flash", "--language-model-only",
        "--tokenizer-mode", "deepseek_v41", "--tool-call-parser", "deepseek_v41",
        "--enable-auto-tool-choice", "--reasoning-parser", "deepseek_v41",
        "--enable-prompt-tokens-details", "--enable-prefix-caching", "--enable-chunked-prefill",
        "--tensor-parallel-size", "1", "--data-parallel-size", "8", "--enable-expert-parallel",
        "--enable-cumem-allocator", "--enable-ep-weight-filter",
        "--engram-config", '{"cpu_offload":false,"embedding_across_dp":true}',
        "--attention-config", '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4",'
                              '"indexer_sparse_logits":true,"sparse_mla_prefill_workspace_factor":4}',
        "--kernel-config", '{"enable_flashinfer_autotune":true,"moe_backend":"deep_gemm_mega_moe",'
                           '"sparse_indexer_topk_backend":"deep_select"}',
        "--max-model-len", str(context), "--max-num-seqs", "128",
        "--prefill-schedule-interval", "96", "--max-num-batched-tokens", "65536",
        "--kv-cache-dtype", "fp8", "--kv-cache-memory", str(kv * GIB),
        "--long-prefill-token-threshold", "8192", "--gpu-memory-utilization", "0.93",
        "--max-cudagraph-capture-size", "8190", "--compilation-config", json.dumps(compilation),
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":5,'
                                '"draft_sample_method":"probabilistic","rejection_sample_method":"block",'
                                '"enable_adaptive_verification":false}',
        "--disable-uvicorn-access-log",
    ]


def ensure_ports_free(bindings):
    for host, port in bindings:
        version = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(version) as listener:
            try:
                listener.bind((host, port))
            except OSError as error:
                raise ValueError(f"cannot use {host}:{port}: {error}") from error


def wait_ready(name, probe, deadline):
    while time.monotonic() < deadline:
        if not running(name):
            result = docker("logs", "--tail", "40", name, required=False, stderr=subprocess.STDOUT)
            raise ValueError(f"{name} exited during startup\n{result.stdout}")
        try:
            if probe():
                print(f"{name} is ready", file=sys.stderr)
                return
        except (OSError, URLError):
            pass
        time.sleep(2)
    raise ValueError(f"{name} did not become ready before DSUF_B200_READY_TIMEOUT_S")


def running(name):
    result = docker("inspect", "--format", "{{.State.Running}}", name, required=False,
                    stderr=subprocess.DEVNULL)
    return result.stdout.strip() == "true" and result.returncode == 0


class Launch:
    def __init__(self):
        self.directory = Path(os.environ.get("DSUF_B200_RUN_DIR", ROOT / "build" / "b200-test")).resolve()
        self.state_file = self.directory / "state.json"
        digest = hashlib.sha256(os.fsencode(self.directory)).hexdigest()[:14]
        self.prefix = f"dsuf-b200-{digest}"
        self.label = f"deepseek-ultrafast.b200-test={digest}"

    def owned(self):
        return docker("ps", "-a", "--filter", f"label={self.label}", "--format", "{{.Names}}").stdout.split()

    def start(self, role, flags, image, args):
        name = f"{self.prefix}-{role}"
        print(f"Starting {name}", file=sys.stderr)
        docker("run", "--detach", "--name", name, "--label", self.label, *flags, image, *args)
        return name

    def state(self):
        if not self.state_file.exists():
            raise ValueError(f"no launch state in {self.directory}")
        return json.loads(self.state_file.read_text())

    def preflight(self, mode, materialize=True):
        if platform.system() != "Linux":
            raise ValueError("B200 execution requires Linux on the GPU host")
        docker("info")
        model = checkpoint()
        gpus = detect_gpus()
        groups = selected_groups(mode, gpus)
        ids = {uuid for group in groups for uuid in group}
        check_gpu_idle([gpu for gpu in gpus if gpu["uuid"] in ids])
        native = mode != "vllm-dep8"
        image = NATIVE_IMAGE if native else VLLM_IMAGE
        docker("image", "inspect", image)
        docker("image", "inspect", GATEWAY_IMAGE)
        if native:
            docker("image", "inspect", REDIS_IMAGE)
        report = {"mode": mode, "model_directory": str(model), "gpus": gpus,
                  "groups": groups, "rdma_host": rdma_ip() if native else None,
                  "nvlink_topology": command("nvidia-smi", "topo", "-m").stdout}
        for group in groups:
            probe = docker("run", "--rm", "--network", "none", *gpu_flags(group),
                           "--entrypoint", "nvidia-smi", image, "--query-gpu=uuid", "--format=csv,noheader")
            if set(probe.stdout.split()) != set(group):
                raise ValueError("container GPU UUIDs differ from its requested allocation")
        if materialize:
            report["peer_access_read"] = command("nvidia-smi", "topo", "-p2p", "r").stdout
            if native:
                check = docker("run", "--rm", "--network", "host", *rdma_args(), "--entrypoint",
                               "/usr/local/bin/kv-rdma-check", image, timeout=90)
                report["rdma_check"] = check.stdout.strip()
            report["host_memory"] = "\n".join(Path("/proc/meminfo").read_text().splitlines()[:3])
        return report

    def up(self, mode):
        if self.owned():
            raise ValueError("run already owns containers; use down before starting another profile")
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise ValueError("set OPENAI_API_KEY to the private test gateway bearer token")
        preflight = self.preflight(mode, materialize=False)
        native = mode != "vllm-dep8"
        groups = preflight["groups"]
        gateway = setting("GATEWAY_PORT", 18080, 1024, 65535)
        base_port = setting("ENGINE_PORT", 18081, 1024, 65534)
        redis = setting("REDIS_PORT", 16379, 1024, 65535)
        kv = setting("KV_PORT", 19500, 1024, 65534)
        bind = [("127.0.0.1", gateway)] + [("127.0.0.1", base_port + i) for i in range(len(groups))]
        if native:
            bind += [("127.0.0.1", redis)]
            bind += [(preflight["rdma_host"], kv + i) for i in range(len(groups))]
        if len(set(bind)) != len(bind):
            raise ValueError("gateway, native, and Redis bindings must use distinct endpoints")
        ensure_ports_free(bind)
        deadline = time.monotonic() + setting("READY_TIMEOUT_S", 7200, 10, 86400)
        self.directory.mkdir(parents=True, exist_ok=True)
        services = []
        state = {"mode": mode, "endpoint": url(gateway, "/v1/chat/completions"), "services": services,
                 "preflight": preflight}
        backend = []
        config = self.directory / "gateway.json"
        model_mount, model_path = checkpoint_mount(Path(preflight["model_directory"]))
        common = ["--network", "host", "--mount", mount(model_mount, "/models")]
        try:
            if native:
                redis_name = self.start("redis", ["--network", "host"], REDIS_IMAGE,
                                        ["redis-server", "--bind", "127.0.0.1", "--port", str(redis),
                                         "--save", "", "--appendonly", "no"])
                services.append({"id": redis_name, "role": "redis"})
                wait_ready(redis_name, lambda: docker("exec", redis_name, "redis-cli", "-p", str(redis),
                                                      "PING", required=False).stdout.strip() == "PONG", deadline)
            for index, group in enumerate(groups):
                port = base_port + index
                shared = [*common, *gpu_flags(group)]
                if native:
                    trace = str(setting("TRACE", 0, 0, 1))
                    args = native_options(port) + ["--ckpt", model_path, "--gpus", ",".join(map(str, range(len(group))))]
                    flags = [*shared, *rdma_args(), "--memory", f'{setting("NATIVE_HOST_GIB", 256, 16, 4096)}g',
                             "--env", f"DSUF_KV_PORT={kv + index}", "--env", f"DSUF_REDIS_URL=redis://127.0.0.1:{redis}",
                             "--env", f"DSUF_TRACE_BATCH_STEPS={trace}"]
                    name = self.start(f"native-{index}", flags, NATIVE_IMAGE, args)
                    readiness = "/readyz"
                    admission = url(port, "/internal/admission")
                    tp, dp, ep = len(group), 1, 1
                else:
                    flags = [*shared, "--shm-size", f'{setting("VLLM_SHM_GIB", 128, 8, 512)}g',
                             "--ulimit", "memlock=-1:-1", "--entrypoint", "vllm"]
                    for key, value in {
                        "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_USE_RUST_FRONTEND": "1",
                        "VLLM_ENGINE_READY_TIMEOUT_S": "7200", "VLLM_RPC_TIMEOUT": "600000",
                        "VLLM_LOG_STATS_INTERVAL": "1", "VLLM_SERVER_DEV_MODE": "1",
                        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0", "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": "1",
                        "NCCL_CUMEM_ENABLE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    }.items():
                        flags.extend(["--env", f"{key}={value}"])
                    args = vllm_options(port, model_path)
                    name = self.start("vllm-0", flags, VLLM_IMAGE, args)
                    readiness = "/health"
                    admission = None
                    tp, dp, ep = 1, 8, 8
                services.append({"id": name, "role": "engine", "index": index, "gpu_uuids": group,
                                 "readiness_url": url(port, readiness), "admission_url": admission,
                                 "args": args,
                                 "topology": {"backend": "native" if native else "vllm", "stage": "combined",
                                              "tp": tp, "dp": dp, "ep": ep}})
                backend.append({"id": name, "url": url(port, ""), "pool": "standard", "cohort": mode,
                                "max_inflight": 64 if native else 512, "readiness_path": readiness,
                                "probe_capacity": native, "vllm_cache_salt": not native})
            active = setting("GATEWAY_MAX_ACTIVE", 256 if native else 512, 1, 4096)
            pending = setting("GATEWAY_MAX_PENDING", 128, 0, 4096)
            credits = setting("GATEWAY_OUTPUT_BUDGET", active * 8192, 8192, 1 << 35)
            write_json(config, {
                "listen": f"127.0.0.1:{gateway}", "max_active": active, "max_pending": pending,
                "standard_max_output_tokens": 8192, "standard_accepts_images": native,
                "tenants": [{"id": "coding", "token_env": "OPENAI_API_KEY", "max_active": active,
                             "max_pending": pending, "max_reserved_output_tokens": credits}],
                "backends": backend,
            })
            config.chmod(0o644)
            name = self.start("gateway", ["--network", "host", "--mount", mount(config, "/etc/dsuf/gateway.json"),
                                          "--env", "OPENAI_API_KEY"], GATEWAY_IMAGE, [])
            services.append({"id": name, "role": "gateway", "readiness_url": url(gateway, "/readyz")})
            write_json(self.state_file, state)
            for service in services:
                if "readiness_url" in service:
                    wait_ready(service["id"], lambda: get(service["readiness_url"]) is not None, deadline)
        except (Exception, KeyboardInterrupt):
            try:
                self.down()
            except (OSError, subprocess.SubprocessError):
                print(f"cleanup incomplete; run down with DSUF_B200_RUN_DIR={self.directory}", file=sys.stderr)
            raise
        return self.status()

    def status(self):
        state = self.state()
        owned = set(self.owned())
        for service in state["services"]:
            service["running"] = service["id"] in owned and running(service["id"])
        return state

    def logs(self):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        directory = self.directory / "logs" / f"{stamp}-{time.time_ns() % 10**9:09d}"
        directory.mkdir(parents=True)
        for name in self.owned():
            path = directory / f"{name}.log"
            with path.open("w") as stream:
                docker("logs", "--timestamps", name, stdout=stream, stderr=subprocess.STDOUT, required=False)
        return str(directory)

    def down(self):
        names = self.owned()
        for name in sorted(names, key=lambda item: (not item.endswith("-gateway"), item.endswith("-redis"))):
            if running(name):
                docker("stop", "--time", "65", name, timeout=85)
        archive = self.logs() if names else None
        if names:
            docker("rm", *names)
        return {"removed": names, "logs": archive}

    def fault(self, index):
        if self.state()["mode"] != "native-dual-tp4":
            raise ValueError("fault requires native-dual-tp4 so a second replica can remain online")
        name = f"{self.prefix}-native-{index}"
        if name not in self.owned() or not running(name):
            raise ValueError(f"requested replica {name} is not running")
        start = time.time()
        docker("kill", "--signal", "KILL", name)
        if running(name):
            raise ValueError(f"replica remained running after the fault: {name}")
        return {"id": name, "started_at_s": start, "confirmed_down_at_s": time.time(), "confirmed_down": True}

    def inventory(self, revision):
        if not revision or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
            raise ValueError("a pinned --checkpoint-revision or DSUF_B200_CHECKPOINT_REVISION is required")
        state = self.status()
        model_dir = Path(state["preflight"]["model_directory"])
        if checkpoint_mount(model_dir)[0] != model_dir and model_dir.name != revision:
            raise ValueError("checkpoint revision differs from the mounted Hugging Face snapshot directory")
        replicas = []
        seen = set()
        for entry in state["services"]:
            if entry["role"] != "engine":
                continue
            if not entry["running"]:
                raise ValueError(f"{entry['id']} is not running")
            get(entry["readiness_url"])
            name = entry["id"]
            image_id = docker("inspect", "--format", "{{.Image}}", name).stdout.strip()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                raise ValueError(f"{name} lacks an immutable image ID")
            response = docker("exec", name, "nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader")
            hardware = list(csv.reader(response.stdout.splitlines(), skipinitialspace=True))
            actual = {row[0] for row in hardware
                      if len(row) == 2 and GPU_UUID.fullmatch(row[0]) and re.search(r"\bB200\b", row[1])}
            if actual != set(entry["gpu_uuids"]) or len(hardware) != len(actual) or seen & actual:
                raise ValueError(f"{name} has an unexpected or overlapping physical GPU allocation")
            seen.update(actual)
            startup = {"ready": True, "image_digest": image_id, "checkpoint_revision": revision}
            if entry["admission_url"]:
                admission = json.loads(get(entry["admission_url"]))
                epoch = admission["epoch"]
                if not epoch:
                    raise ValueError(f"{name} reports an empty admission epoch")
                logs = docker("logs", name).stdout
                for line in reversed(logs.splitlines()):
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("event") == "packed_decode_ready":
                        break
                else:
                    raise ValueError(f"{name} has no packed_decode_ready event")
                startup["source"] = "packed_decode_ready"
                for key in ("max_residents", "max_decode_width", "slot_bytes_per_rank", "kv_bytes_per_rank"):
                    startup[key] = event[key]
                width = event["max_decode_width"]
            else:
                startup["source"] = "vllm_engine_ready"
                epoch = docker("inspect", "--format", "{{.Id}}", name).stdout.strip()
                width = None
            replica = {
                "id": name, "host": socket.gethostname(), "epoch": epoch, "gpu_uuids": sorted(actual),
                "decode_width": width, "topology": entry["topology"], "startup": startup,
                "readiness_url": entry["readiness_url"],
            }
            if entry["admission_url"]:
                replica["admission_url"] = entry["admission_url"]
            replicas.append(replica)
        if not replicas:
            raise ValueError("no running engine replicas to inventory")
        return {"schema_version": 1, "endpoint": state["endpoint"], "replicas": replicas}


def build(profile):
    if profile in ("all", "native"):
        docker("build", "-f", str(ROOT / "deploy/Dockerfile"), "--build-arg", "DSUF_SM=sm_100a", "-t",
               NATIVE_IMAGE, str(ROOT), timeout=None, stdout=None)
        docker("build", "-f", str(ROOT / "deploy/Dockerfile"), "--target", "b200-parity",
               "--build-arg", "DSUF_SM=sm_100a", "-t", PARITY_IMAGE, str(ROOT), timeout=None, stdout=None)
        docker("pull", REDIS_IMAGE, timeout=None, stdout=None)
    if profile in ("all", "vllm"):
        docker("pull", VLLM_IMAGE, timeout=None, stdout=None)
    docker("build", "-f", str(ROOT / "deploy/Dockerfile.gateway"), "-t", GATEWAY_IMAGE,
           str(ROOT), timeout=None, stdout=None)


def run_probe(args):
    count = args.tp
    gpus = detect_gpus()[:count]
    check_gpu_idle(gpus)
    test = args.action == "page-test"
    tool = "dsuf-cuda-test" if test else "dsuf-batch-probe"
    flags = ["--rm", "--network", "none", *gpu_flags([item["uuid"] for item in gpus]),
             "--entrypoint", f"/usr/local/bin/{tool}"]
    if test:
        probe_args = ["virtual_memory::tests::stable_address_with_growing_and_reused_backing", "--ignored", "--exact"]
        result = docker("run", *flags, PARITY_IMAGE, *probe_args, timeout=None)
        print(result.stdout, end="")
        if "1 passed" not in result.stdout:
            raise ValueError("CUDA page reuse test was not exercised")
        return
    else:
        prompts = args.prompts.resolve(strict=True)
        model_mount, model_path = checkpoint_mount(checkpoint())
        flags += ["--mount", mount(model_mount, "/models"), "--mount", mount(prompts, "/prompts.json"),
                  "--env", f"DSUF_PACKED_GRAPHS={0 if args.eager else 1}"]
        probe_args = ["--ckpt", model_path, "--gpus", ",".join(map(str, range(count))),
                      "--batch-tokens", "/prompts.json", "--max-seq", str(args.max_seq),
                      "--decode", str(args.decode), "--seed", str(args.seed)]
        if args.packed_drafts:
            probe_args.extend(["--packed-drafts", str(args.packed_drafts)])
    docker("run", *flags, PARITY_IMAGE, *probe_args, timeout=None, stdout=None)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest="action", required=True)
    action.add_parser("preflight").add_argument("mode", choices=MODES, nargs="?", default="native-dual-tp4")
    action.add_parser("build").add_argument("profile", choices=("native", "vllm", "all"), nargs="?", default="all")
    action.add_parser("up").add_argument("mode", choices=MODES)
    action.add_parser("down")
    action.add_parser("status")
    action.add_parser("logs")
    action.add_parser("fault").add_argument("index", type=int, choices=(0, 1))
    action.add_parser("inventory").add_argument("--checkpoint-revision", default=os.environ.get("DSUF_B200_CHECKPOINT_REVISION"))
    action.add_parser("page-test").set_defaults(tp=1)
    parity = action.add_parser("parity")
    parity.add_argument("prompts", type=Path)
    parity.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parity.add_argument("--max-seq", type=int, default=131072)
    parity.add_argument("--decode", type=int, default=64)
    parity.add_argument("--seed", type=int, default=42)
    parity.add_argument("--packed-drafts", type=int, choices=range(8), default=0)
    parity.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    launch = Launch()
    try:
        if args.action == "build":
            build(args.profile)
            return 0
        if args.action in ("parity", "page-test"):
            run_probe(args)
            return 0
        if args.action == "preflight":
            result = launch.preflight(args.mode)
        elif args.action in ("up", "down", "fault"):
            launch.directory.mkdir(parents=True, exist_ok=True)
            with (launch.directory / ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.action == "up":
                    result = launch.up(args.mode)
                elif args.action == "fault":
                    result = launch.fault(args.index)
                else:
                    result = launch.down()
        elif args.action == "inventory":
            result = launch.inventory(args.checkpoint_revision)
        else:
            result = getattr(launch, args.action)()
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, KeyboardInterrupt) as error:
        print(f"B200 launch: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
