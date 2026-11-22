# Deployment

## Build the image

```sh
docker build -f deploy/Dockerfile --build-arg DSUF_SM=sm_103a -t deepseek-ultrafast .   # GB300, B300
docker build -f deploy/Dockerfile -t deepseek-ultrafast .                                  # B200
```

## Run with Docker

```sh
docker run -d --network host redis:7 redis-server --bind 127.0.0.1 --save '' --appendonly no
docker run -d --gpus all --network host --device /dev/infiniband --ulimit memlock=-1 \
    -v "$PWD/models:/models:ro" deepseek-ultrafast
```

The image serves on GPUs 0-3 by default. On an 8-GPU B300 machine, serve TP8 on all eight:

```sh
docker run -d --gpus all --network host --device /dev/infiniband --ulimit memlock=-1 \
    -v "$PWD/models:/models:ro" deepseek-ultrafast --gpus 0,1,2,3,4,5,6,7 --port 8000
```

## Run on Kubernetes

Set the image and the `deepseek-ultrafast-models` volume in `kubernetes.yaml`, then:

```sh
kubectl apply -f deploy/kubernetes.yaml
```

For TP8 on 8-GPU B300 nodes, set the container's `--gpus 0,1,2,3,4,5,6,7` and `nvidia.com/gpu: 8`.

## Continuous batching fleet

Build the CPU-only streaming gateway from the repository root.

```sh
docker build -f deploy/Dockerfile.gateway -t deepseek-ultrafast-gateway .
```

The [fleet example](fleet/kustomization.yaml) has two standard TP4 replicas using
packed autoregressive decode and one TP4 replica for contexts up to 1M tokens.
Each standard pod has its own DNS address so the gateway can track readiness,
load, and soft session affinity independently. The long pool serves requests
whose prompt or output budget exceeds the standard pool. Send an explicit output
cap of at most 8,192 tokens to qualify for the standard pool. The example uses one
gateway and one private Redis directory. Redis replacement takes the directory
offline while the new instance starts.

- In your deployment overlay, replace both example registry entries with the
  published images and pin the gateway, engine, and Redis images to verified
  digests. Give compatible engines the same gateway cohort in
  [gateway.json](fleet/gateway.json). Use a new cohort when changing the checkpoint,
  TP shape, or cache compatibility settings.
- Provide `dsuf-gateway-auth` with key `coding-token` and the
  `deepseek-ultrafast-models` PVC in the deployment namespace. The model volume
  must support read-only mounts from multiple nodes and contain the checkpoint
  and tokenizer expected by `serve`.
- The example requests one four-GPU Blackwell group and an RDMA device per engine.
  Select nodes with `deepseek-ultrafast/gpu-profile=blackwell-tp4`, or change the
  selector and `rdma/hca` resource name to match the cluster. The engine groups
  use distinct nodes. Set compatible CUDA image builds and adjust host memory
  requests to the measured checkpoint and snapshot staging needs.
- Use a trusted namespace and a CNI that enforces the
  [private ingress rules](fleet/network.yaml). Keep engine HTTP and the
  unauthenticated Redis directory on the private network.
  The `DSUF_KV_HOST` pod IP must be routable to peers and usable by the RDMA
  transport. Configure the device plugin, fabric isolation, and the container
  runtime's memlock allowance for the registered-memory pool. If the fabric needs
  different addressing, adapt `DSUF_KV_HOST` and enforce equivalent isolation.
  Expose only the gateway through your TLS ingress. Quotas are local to a gateway process.

Render the manifests locally.

```sh
kubectl kustomize deploy/fleet
```

The replica counts and memory budgets are a topology example. Run the CUDA and
RDMA parity checks, then size the fleet from measured goodput, memory, and
failure headroom before the 500-user tests. [Continuous batching qualification](../docs/CONTINUOUS_BATCHING.md#acceptance-and-benchmarks)

## Configuration

- `DSUF_REDIS_URL` (default `redis://127.0.0.1:6379`): the Redis shared by replicas
- `DSUF_KV_HOST`, `DSUF_KV_PORT` (default port `19500`): the address peers reach for KV transfer
- `serve` flags: `--gpus`, `--port`, `--max-seq`, `--chunk`, `--model-name`, `--ckpt`

## Two 8×B300 hosts

For an available single B200 host, use the [8×B200 rehearsal kit](b200-test/README.md) to test native correctness, the scheduler, and same-node vLLM goodput first.

Use one of the following overlays for a two-node qualification run. Every row
uses 16 GPUs. The original fleet example remains available for its three-node
placement.

| Overlay | Workload pool | GPU placement |
| --- | --- | --- |
| [Ultrafast](fleet-b300-two-node/kustomization.yaml) | Four standard 128 Ki TP4 replicas | Two groups per host |
| [Ultrafast + long context](fleet-b300-two-node-long/kustomization.yaml) | Three standard 128 Ki TP4 replicas and one 1 Mi TP4 replica | Two groups per host across both pools |
| [vLLM DEP2](fleet-b300-vllm-dep2/kustomization.yaml) | Eight text-only DeepSeek-V4.1-Flash TP1 × DP2 + EP2 replicas | Four groups per host |
| [vLLM DEP8](fleet-b300-vllm-dep8/kustomization.yaml) | Two text-only DeepSeek-V4.1-Flash TP1 × DP8 + EP8 replicas | One group per host |

### Prepare the hosts

1. Label exactly two NVSwitch-connected 8×B300 nodes
   `deepseek-ultrafast/gpu-profile=b300-two-node`. Label a separate CPU worker
   `deepseek-ultrafast/node-role=control` so the gateway and private Redis stay
   available during GPU-host fault tests. The overlays use hard hostname
   spread constraints with `minDomains: 2`. Use a Kubernetes version that
   supports `minDomains` and node inclusion policies. Use a dedicated
   qualification namespace for each backend configuration, so their pods
   do not count toward each other's spread constraints.
   [Kubernetes topology constraints](https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/)
2. Stage the **same pinned checkpoint revision and tokenizer** on local NVMe at
   `/srv/dsuf/models` on both hosts. The volume is mounted read-only at `/models`.
   If cluster policy disallows hostPath, use a read-only shared PVC accessible
   to both hosts in the site overlay.
   Record the digest or revision alongside run results. Each engine gets a
   disjoint full-GPU allocation from the NVIDIA device plugin and sees
   container-local ordinals. Disable GPU time-sharing for this qualification.
   Check that each pod sees exactly its requested physical GPU UUIDs and that
   peer access works for the assigned group. `nvidia-smi -L` in each engine
   container reports the allocated UUIDs.
3. For Ultrafast, provide two schedulable `rdma/hca` allocations per host,
   or substitute your device plugin's resource name. Verify that `/dev/infiniband`
   and active verbs ports are visible in each container, that the container can
   register its memory pools, and that peer pod IPs on both hosts are usable by
   RDMA. The overlay keeps pod networking, so every pod can reuse HTTP 8000 and
   KV 19500. With host networking, assign unique HTTP and `DSUF_KV_PORT` values
   to every co-located replica, update the Service and probes, and preserve
   private ingress. Run `kv-rdma-check` in both hosts' containers, then verify
   an actual same-cohort remote prefix hit in each direction.
   [Server requirements](../docs/SERVER.md#openai-compatible-server-serve)
4. Verify host allocatable memory and pinned-memory limits against the actual
   checkpoint and measured snapshot pools. The examples request **256 GiB per
   native TP4 process**, **384 GiB per vLLM DEP2 process**, and **512 GiB per
   DEP8 process**. Those requests total 512 GiB, 1,536 GiB, and 512 GiB per GPU
   host respectively. vLLM allocates a memory-backed `/dev/shm`, counted in its
   pod memory budget. Update the limits and requests together if measured
   startup or steady-state memory requires it. Two-node Ultrafast configures
   64 GiB of KV per GPU, including up to 16 GiB of evictable idle prefix pages.
   It allows 64 resident slots, up to 32 active decoders, and 32 target rows
   per launch. The two-node overlays use an 8 MiB async KV copy quantum.
   Retirement keeps the branch lease until the export's DMA fence completes.
   Watch `work.retirement_bytes` and `work.retirement_ms` in the private
   admission feed when measuring a follow-up turn's TTFT. The scheduler
   starts with a busy prefill opportunity every
   decode step and a 50 ms prefill wait target. The measured decode throttle
   targets a 250 ms maximum token gap and 100 tokens/s per request. Verify the
   actual width, timing, cache budget, and occupancy in the logs, then tune
   the caps and cadence from mixed-load results.

### Render and install

Build and publish the Ultrafast `sm_103a` server and gateway images. In your
site overlay, replace the upstream example registry references with your pinned
images, including the gateway image for the vLLM cases. For vLLM, confirm the
host driver supports the pinned CUDA 13.4 image. Supply `dsuf-gateway-auth` in
the same namespace with the `coding-token` key. Create the secret from an
existing credential file with no trailing newline. Serve only the authenticated
gateway over TLS and enforce the inherited private NetworkPolicies. GPU-host
failure tests assume the single gateway and Redis remain on the control worker.

```sh
kubectl kustomize deploy/fleet-b300-two-node
kubectl kustomize deploy/fleet-b300-two-node-long
kubectl kustomize deploy/fleet-b300-vllm-dep2
kubectl kustomize deploy/fleet-b300-vllm-dep8
```

Apply the rendered **site** overlay in its qualification namespace. Wait for
the engines and gateway, then record each replica's host, pool, and allocated
GPU UUIDs. For the three-plus-one overlay, also wait for `statefulset/dsuf-long`.
The vLLM startup probes allow a two-hour graph capture and autotuning window
with additional time for process startup.

```sh
kubectl -n "$DSUF_BENCH_NS" rollout status statefulset/dsuf-standard --timeout=2h15m
kubectl -n "$DSUF_BENCH_NS" rollout status deployment/dsuf-gateway --timeout=5m
kubectl -n "$DSUF_BENCH_NS" get pods -l app=dsuf-engine \
    -o 'custom-columns=REPLICA:.metadata.name,HOST:.spec.nodeName,POOL:.metadata.labels.pool'
for pod in $(kubectl -n "$DSUF_BENCH_NS" get pods -l app=dsuf-engine -o name); do
    printf '%s\n' "$pod"
    kubectl -n "$DSUF_BENCH_NS" exec "$pod" -- nvidia-smi -L
done
```

Check that every expected replica resolves through its
`dsuf-standard-N.dsuf-standard` or `dsuf-long-0.dsuf-long` DNS name. Compare
the UUID inventory across replicas. The same physical GPU must appear in
only one engine group.

The four-standard overlay has only standard coding capacity. Send explicit
output caps of at most 8,192 and keep rendered input, output cap, and the native
64-token lookahead within 128 Ki total tokens for an equal-workload trial. To
allow longer turns, either select the three-plus-one overlay or add an
independently provisioned private long backend to the site's gateway config.
Give a long or otherwise incompatible backend its own cohort. In the
three-plus-one overlay, record which host owns the long replica. Losing that
host removes the entire long pool and one standard replica. Losing the other
host removes two standard replicas.

### Run the qualification

- Start with native CUDA parity and a short load ramp. Record the GPU memory
  high-water mark, host cache budget, page and snapshot traffic, useful decode
  tokens per step, prefill rate, and actual active decode occupancy. For a short
  tracing run set `DSUF_TRACE_BATCH_STEPS=1` in the engine container, then
  return to the normal log volume for the qualification soak.
- Replay the same frozen text coding manifest, checkpoint revision, and
  output caps against native, DEP2, and DEP8 through their gateways. Sweep
  offered arrival rate before the 500-session soak. For a 16-GPU comparison,
  run the overlays sequentially on the rented pair, allowing the previous
  engine pods to drain and release their allocations before starting the next.
  Preserve each run's rendered manifest, image digests, and output separately.
- Pass the bearer token through the existing `OPENAI_API_KEY` environment.
  Run the live-session gate and the open-loop burst sweep on the same trace.
  Measure the all-standard capacity first. Then measure the mixed deployment
  separately if it dedicates a GPU group to the long tier. Capture a fresh
  inventory after the replicas are ready. The collector reads GPU UUIDs,
  image digests, native admission epochs, and native packed startup widths.
  It uses the Kubernetes pod proxy for native admission. An equivalent JSON
  inventory can be collected from inside the network when pod proxy access is
  unavailable.

  ```sh
  python3 tools/collect_fleet_inventory.py --namespace "$DSUF_BENCH_NS" \
      --endpoint "$DSUF_BENCH_URL" --backend native \
      --checkpoint-revision "$DSUF_CHECKPOINT_REV" > inventory.json
  python3 tools/bench_coding.py MANIFEST --url "$DSUF_BENCH_URL" --inventory inventory.json \
      --gpus 16 --mode closed --sessions 500 --qualify sessions
  python3 tools/bench_coding.py MANIFEST --url "$DSUF_BENCH_URL" --inventory inventory.json \
      --gpus 16 --mode open --arrival-scales 1,1.25,1.5 --isolate-replays --qualify load
  ```

  Set `DSUF_BENCH_URL` to the gateway's Chat Completions URL and
  `DSUF_CHECKPOINT_REV` to the staged checkpoint revision. For the vLLM run,
  use `--backend vllm` in the collector and `--request-format vllm` in the
  replay. Save the sweep output even when an overload point misses the gates.
  Keep the 500-simultaneous-generator
  test separate from the 500-session result. For the capacity model with
  500 users turning once per minute, add `--min-turns-per-s 8.33` to the
  closed-loop gate so longer queueing cannot qualify a lower turn rate.

- Measure traffic distribution across all ready replicas. Record native
  startup capacity and aggregate native completion logs per pod. Set
  `DSUF_REPLAY_START` to the replay start time in RFC3339 format. For vLLM,
  collect the per-pod request counters from `/metrics` over the same interval.

  ```sh
  for pod in $(kubectl -n "$DSUF_BENCH_NS" get pods -l app=dsuf-engine -o name); do
      printf '%s\n' "$pod"
      kubectl -n "$DSUF_BENCH_NS" logs "$pod" |
          jq -Rc 'fromjson? | select(.event == "packed_decode_ready")'
      kubectl -n "$DSUF_BENCH_NS" logs "$pod" --since-time="$DSUF_REPLAY_START" |
          jq -Rn '[inputs | fromjson? | select(.event == "request_end")] |
                    {turns: length, output_tokens: (map(.output_tokens) | add // 0)}'
  done
  ```

- Test loss of each host independently in the disposable trial environment.
  Choose a steady offered rate and a trace long enough to include a
  post-failure window. Trigger the host fault at the marker and record the
  actual outage independently. For example, the 500-turns/minute workload
  can be gated at 8.33 turns/s during steady load and after the fault.

  ```sh
  python3 tools/bench_coding.py MANIFEST --url "$DSUF_BENCH_URL" --inventory inventory.json --gpus 16 \
      --mode open --qualify load --min-turns-per-s 8.33 --isolate-replays \
      --event-at-s 600 --event-label host-loss --event-recovery-s 30 --post-event-s 600
  ```

  Record warm, cold, and post-failure TTFT, useful decode rate, stream gaps,
  reliability, and successful throughput, including the transition. Count the
  surviving replicas and report their actual GPU count. Full-load single-host
  failover requires one host to pass the workload on its own.
- DEP2 adapts the upstream B300 c64 configuration to a 128 Ki cap with
  standard vLLM in-process DP routing and the shared gateway. It exposes
  cached-token usage for warm/cold classification and scopes prefix-cache
  reuse by authenticated tenant and branch. DEP8 is a wider per-host
  expert-parallel candidate based on the upstream Blackwell settings. Both
  serve the identical local model. Different frontend routing and context
  limits can change cache behavior versus AgentX. Published concurrency
  labels do not establish this workload's 100 tok/s p10 rate. If colocated
  prefill and decoding miss the latency gates despite tuning, compare the
  two-host Dynamo/NIXL disaggregated lane on EFA as a separate configuration.
  [B300 DEP2 recipe](https://github.com/SemiAnalysisAI/InferenceX/pull/4332) · [vLLM Blackwell DEP](https://github.com/vllm-project/vllm/blob/main/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml) · [Pinned B300 disaggregated comparison](fleet-b300-pd/README.md)
