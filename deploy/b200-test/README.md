# Single-node B200 rehearsal

Use one 8×B200 host to check native GPU parity, isolate the scheduler’s prefill and retirement behavior, and measure the passing workload on one node. Compare two native TP4 replicas and vLLM DEP8 on the same eight physical GPUs. All of the commands below are run on the test host. The launcher keeps the test gateway on loopback and gives each rehearsal its own containers and evidence directory.

## Inputs and preflight

- A Linux host with eight exclusive B200 GPUs, NVIDIA Container Toolkit, Docker, Python 3.10+, working CUDA peer access, and enough local disk and RAM to load two copies of the checkpoint. For native serving, expose an active RDMA device and its host address. The launcher starts its own private Redis directory. Native readiness requires RDMA even when all replicas are local.
- The same pinned DeepSeek-V4.1-Flash checkpoint and tokenizer you intend to serve. Put the complete snapshot on local storage and point `DSUF_B200_MODEL_DIR` at it. If you need to stage it, the Hugging Face CLI can fetch the pinned snapshot with `hf download deepseek-ai/DeepSeek-V4.1-Flash --revision "$DSUF_CHECKPOINT_REV" --local-dir "$DSUF_B200_MODEL_DIR"`.
- A sanitized replay manifest from the actual hackathon client and starter repository. Include each full Chat Completions request, its arrival timestamp, tool delays, distinct session IDs for parallel branches, and dependencies between turns. The smoke generator supplies deterministic synthetic traffic for testing the setup before that trace is ready.

From the checkout root, set these values for your host. Supply an existing rehearsal-only `OPENAI_API_KEY` in the environment, or generate one for the private gateway.

```bash
export DSUF_B200_MODEL_DIR=/srv/dsuf/models
export DSUF_CHECKPOINT_REV='replace-with-pinned-checkpoint-commit-or-artifact-hash'
export DSUF_B200_KV_HOST='replace-with-test-host-RDMA-address'
export OPENAI_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export DSUF_B200_RUN_DIR="$PWD/../b200-parity-runtime"

deploy/b200-test/launch.sh build all
deploy/b200-test/launch.sh preflight native-dual-tp4
deploy/b200-test/launch.sh preflight vllm-dep8
```

`build native` builds the `sm_100a` runtime, CUDA probe, and gateway. `build vllm` pulls the pinned B200 DEP8 image. The DEP8 candidate keeps actual speculative acceptance enabled. The native B300 Kubernetes overlays assume two GPU hosts. This launcher uses Docker and can place either two TP4 replicas or one TP8 replica on the B200 host. HTTP and Redis stay on loopback. Keep the native RDMA peer ports on a private network.

## GPU correctness and sizing

```bash
python3 -m venv /tmp/dsuf-b200-python
/tmp/dsuf-b200-python/bin/pip install tokenizers==0.23.2
/tmp/dsuf-b200-python/bin/python tools/b200_workload.py parity \
  --checkpoint "$DSUF_B200_MODEL_DIR" --output ../b200-parity

deploy/b200-test/launch.sh page-test
deploy/b200-test/launch.sh parity ../b200-parity/parity-2.json
deploy/b200-test/launch.sh parity ../b200-parity/parity-8.json
deploy/b200-test/launch.sh parity ../b200-parity/parity-32.json
deploy/b200-test/launch.sh parity ../b200-parity/parity-32.json --eager
```

Pass when mapped pages survive growth and reuse, the seeded probe reports zero mismatched tokens, and the non-eager 32-row run reports its graph ready. The probe measures teacher-forced logit drift against the incumbent. Agree on a logit tolerance before using it as a gate. Save both graph and eager outputs for a divergence investigation.

On native startup, check `packed_decode_ready` in `runtime/logs` or the recorded inventory. Confirm actual launch width and the ratio of effective `kv_bytes_per_rank` to size of typical and maximum-length slot reservations. B200 HBM can limit resident count. Adjust `DSUF_B200_MAX_SEQ`, `DSUF_B200_MAX_RESIDENTS`, `DSUF_B200_DECODE_WIDTH`, and `DSUF_B200_KV_GIB` if needed. Give workload and parity preparation the same context limit with `--max-seq`. Use the same context and output limits for each backend comparison and record any differences in effective native width.

Run the HTTP smoke on a ready native topology before accepting performance results. Use the ports shown by `launch.sh status` if you override them.

```bash
export DSUF_B200_RUN_DIR="$PWD/../b200-smoke-runtime"
DSUF_B200_TRACE=1 deploy/b200-test/launch.sh up native-dual-tp4
python3 tools/b200_serving_smoke.py --url http://127.0.0.1:18080/v1/chat/completions \
  --overlap-url http://127.0.0.1:18081/v1/chat/completions \
  --report "$DSUF_B200_RUN_DIR/serving-smoke.json"

python3 tools/b200_serving_smoke.py --scenario continuation \
  --url http://127.0.0.1:18081/v1/chat/completions \
  --continuation-url http://127.0.0.1:18082/v1/chat/completions \
  --publication-wait-s 1 --report "$DSUF_B200_RUN_DIR/peer-smoke.json"

deploy/b200-test/launch.sh down
```

The smoke cancels one of two staggered requests, checks that the other finishes, and sends a new request. Use its timestamps and `packed_decode_step` lane IDs to verify overlap and slot generation changes. It also sends an immediate same-branch follow-up and checks reported cache reuse. The peer variant sends the continuation to the other engine while its donor is still healthy. Inspect the actual peer-hit evidence and transfer failures in the runtime logs. To exercise local host restore, restart an isolated native run with `DSUF_B200_HOT_KV_GIB=0` and use the same direct engine URL for both turns. The cold comparison records seeded output digests for diagnosis. A successful request and observed cache reuse are the required HTTP checks. Token parity is gated by the CUDA probe.

`OPENAI_API_KEY` authenticates the gateway. `launch.sh logs` archives evidence without stopping the run. `launch.sh down` stops and archives only this isolated test run. The matrix runner handles launch and teardown automatically.

## Workloads

```bash
python3 tools/b200_workload.py prepare /path/to/sanitized-hackathon-trace.json \
  --output ../b200-workloads --kickoff-tasks 500

# An optional short plumbing check while the real trace is being collected.
python3 tools/b200_workload.py smoke --tasks 8 --output ../b200-smoke
```

Each manifest has a `sessions` array. A session ID is the KV key for one conversation branch. Set `task_id` to the same value on branches from one user task. Each turn has a complete `body`, an `at_s` offset on a common experiment clock, and a `gap_s` from dependency completion for closed-loop replay. Give any prerequisite turn an `id` and reference its ID in `depends_on`. The replay also honors preceding turns within a branch. For a parent and child, the parent’s initial turn and the child’s final turn define the fork and join.

```json
{
  "sessions": [
    {
      "id": "main-42",
      "task_id": "coding-42",
      "turns": [
        {"id": "plan-42", "at_s": 0, "gap_s": 0, "body": {"messages": [{"role": "user", "content": "Plan the change."}], "max_completion_tokens": 512}},
        {"id": "join-42", "depends_on": ["child-42"], "at_s": 8, "gap_s": 0, "body": {"messages": [{"role": "user", "content": "Plan the change."}, {"role": "assistant", "content": "I will inspect the test."}, {"role": "user", "content": "The child found the failing case. Finish the patch."}], "max_completion_tokens": 512}}
      ]
    },
    {
      "id": "subagent-42",
      "task_id": "coding-42",
      "turns": [
        {"id": "child-42", "depends_on": ["plan-42"], "at_s": 3, "gap_s": 0.2, "body": {"messages": [{"role": "user", "content": "Inspect the failing test."}], "max_completion_tokens": 512}}
      ]
    }
  ]
}
```

Keep thinking enabled as in the real harness. Every request on this standard pool needs an explicit output cap of at most 8,192. Rendered prompt length, output cap, and 64-token lookahead must fit the configured total context. If the recorder provides the rendered input count, include it as `turn.prompt_tokens`. The server enforces final eligibility. The preparation command keeps the original trace and derives one first request per task for a separate simultaneous cold kickoff. All `at_s` values in that kickoff fixture are zero.

## Run the matrix

Any invocation accepts `--dry-run` to show its benchmark commands without launching containers. Each new `--output` directory contains startup and inventory records, benchmark JSON, logs, resource samples, and `summary.json`. The runner continues across legitimate overload failures and records which levels passed.

First run a short, trace-enabled A/B with identical workload, KV limits, and launch width. Pick a short real trace or use smoke traffic to verify the trace pipeline. A profiling run with frequent completions and pending fresh prompts provides the strongest limiter evidence.

```bash
python3 tools/b200_qualify.py --workloads ../b200-workloads \
  --checkpoint-revision "$DSUF_CHECKPOINT_REV" --output ../b200-limiter-ab \
  --profile native-tp4 --profile native-tp4-no-rate \
  --phase closed --levels 1,8,16,32 --milestone-s 60 --trace
```

Compare `timing.json` across the two runs. Inspect decode occupancy and quantiles for decode, whole plan, prefill, the retirement anchor, and export. The relevant signature is an over-budget plan with a retirement anchor followed by restricted prefill while requests wait and decoding continues. Look for active admission decreasing and prompt processing recovering with the rate limiter off. Both runs use the same external decode and latency gates. The 6 ms decode and 4 ms busy-prefill figures are reference points from the static model.

Next run the real trace through one TP4, two TP4s, and vLLM DEP8 in sequence. Add `--profile native-tp8` in a separate run to compare TP8 at equal GPU count. The default closed-loop ladder counts 100, 250, and 500 distinct coding tasks including their branches. The open-loop sweep preserves offered arrivals. Kickoff applies cold latency and success gates to the simultaneous first requests. Defaults test each deployment’s ingress limits. For a separate equal-gateway trial, set `DSUF_B200_GATEWAY_MAX_ACTIVE`, `DSUF_B200_GATEWAY_MAX_PENDING`, and `DSUF_B200_GATEWAY_OUTPUT_BUDGET` to identical values for all profiles.

```bash
python3 tools/b200_qualify.py --workloads ../b200-workloads \
  --checkpoint-revision "$DSUF_CHECKPOINT_REV" --output ../b200-capacity
```

Closed-loop trials preserve the initial start offsets. They replace each completed task with a fresh instance of a captured task, preserving its internal tool delays and dependencies. They stop offering replacements at the duration in `--replenish-s` and let the remaining tasks finish. Use `--replenish-s 0` for a single pass. Task completion time covers all replayed branches. It measures serving responsiveness for the frozen request sequence.

At the best passing level, rerun closed-loop with a 7,200-second milestone. Keep tracing off during the soak. This example uses 250 tasks. Substitute the passing profile and task count. If you have measured peak model-call demand, supply it with `--min-offered-rate` so a slow client loop cannot qualify at lower load.

```bash
python3 tools/b200_qualify.py --workloads ../b200-workloads \
  --checkpoint-revision "$DSUF_CHECKPOINT_REV" --output ../b200-soak \
  --profile native-dual-tp4 --phase closed --levels 250 --milestone-s 7200 --warmup-s 60
```

Evaluate task duration, joint SLO goodput, p95 2 s warm and 5 s cold TTFT, p10 100 tok/s decode, p99 250 ms stream gaps, and 99.5% request completion. Confirm that HBM, host RAM, KV cache errors, and queueing remain stable. Synthetic results establish plumbing only.

Finally, choose a timestamped trace that covers the fault and post-fault window. The explicit `fault` phase stops the second test TP4 at the replay marker and records the action separately from the client-side timestamp.

```bash
python3 tools/b200_qualify.py --workloads ../b200-workloads \
  --checkpoint-revision "$DSUF_CHECKPOINT_REV" --output ../b200-replica-loss \
  --profile native-dual-tp4 --phase fault --levels 250 \
  --fault-at-s 120 --fault-recovery-s 30 --post-fault-s 120
```

The one-host result gives a measured B200 capacity ceiling and a backend comparison. Repeat the winning configuration on B300 to measure the `sm_103a` path, two-host RDMA, and full-host loss before committing to the event load.
