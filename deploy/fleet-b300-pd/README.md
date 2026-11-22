# Two-node disaggregated B300 comparison

This qualification lane uses the published DeepSeek-V4.1-Flash Dynamo/NIXL
recipe. It runs four two-GPU DEP2 prefill workers on one 8×B300 host and
one eight-GPU DEP8 decoder on the other. The selected variant uses all 16
GPUs. It needs an x86-64 Slurm/Pyxis environment with **EFA device RDMA and
the NIXL LIBFABRIC provider** on both hosts.
[InferenceX B300 release recipe](https://github.com/SemiAnalysisAI/InferenceX/pull/4231) · [Pinned role configuration](https://github.com/SemiAnalysisAI/InferenceX/blob/a99989c5aa492ed0697835e95b29a97ee3c2ba8f/inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/dsv41flash/vllm/b300-fp4/agentx/disagg-variants.yaml)

Use the same DeepSeek-V4.1-Flash checkpoint and tokenizer revision as the
other comparison lanes. Set `DSUF_PD_MODEL_PATH` to its absolute mounted
snapshot directory on both hosts, `DSUF_PD_CHECKPOINT_REV` to that revision,
and `DSUF_PD_CONTAINER` to the site-imported Pyxis image corresponding to
the digest below. Configure the site account, partition, binaries, and shared
mounts with `make setup ARCH=x86_64` in the pinned srtctl checkout. Set
`SRTSLURM_CONFIG` to its site config file before rendering.
Confirm eight B300 GPUs and working EFA provider discovery on each host. The
prefill node has up to 70 GiB of configured host KV per GPU rank, in addition
to CPU Engram tables and initial staging. Check allocatable and registerable
memory with all four prefill workers co-located. The upstream setup hook
prepares the EFA userspace libraries inside the ephemeral worker containers.
[Container manifest](https://hub.docker.com/layers/vllm/vllm-openai/nightly-x86_64-dev-20261012-59e75f4/images/sha256-db8bfec3185d4fabfec0f9434b72a408e9c8089ceb515eb6d3caaa3580cbe718) · [Pinned srtctl setup](https://github.com/NVIDIA/srt-slurm/blob/ae3d6877c46f3147e3a5dafa46ee4cd89322e52f/docs/installation.md)

## Render the comparison

The bound AgentX recipe defaults to a synthetic draft acceptance length.
The following Bash commands restore real DSpark block rejection, use the
same 128 Ki context limit as the standard pool, and render without
submitting a job. Run them in a directory with space for a temporary
checkout. The output directory must be new or empty.

```bash
dsuf_infx_checkout="$PWD/inferencex-b300-pd"
dsuf_pd_render_dir="$PWD/rendered-b300-pd"

git clone --filter=blob:none https://github.com/SemiAnalysisAI/InferenceX.git "$dsuf_infx_checkout"
git -C "$dsuf_infx_checkout" checkout --detach a99989c5aa492ed0697835e95b29a97ee3c2ba8f
git -C "$dsuf_infx_checkout" submodule update --init --depth 1 inferencex-e2e/utils/srt-slurm
dsuf_infx_project="$dsuf_infx_checkout/inferencex-e2e"

uv run --directory "$dsuf_infx_project" --locked --no-dev --extra recipes infx generate \
    --config-key dsv41flash-fp4-b300-dynamo-vllm-agentic-dspark-disagg \
    --output-dir "$dsuf_pd_render_dir"
dsuf_pd_recipe="$dsuf_pd_render_dir/$(jq -er '.recipes[] | select(.variant == "override_4p1d_dep2_dep8_c2048").file' "$dsuf_pd_render_dir/manifest.json")"
dsuf_pd_image='vllm/vllm-openai@sha256:db8bfec3185d4fabfec0f9434b72a408e9c8089ceb515eb6d3caaa3580cbe718'
dsuf_draft='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
dsuf_pd_args=(
    --file "$dsuf_pd_recipe"
    --set "model.path=${DSUF_PD_MODEL_PATH:?set the common checkpoint path}"
    --set "model.container=${DSUF_PD_CONTAINER:?set the pinned Pyxis image path}"
    --set "identity.model.revision=${DSUF_PD_CHECKPOINT_REV:?set the checkpoint SHA}"
    --set "identity.container.image=$dsuf_pd_image"
    --set "setup_script=$dsuf_infx_project/benchmarks/multi_node/srt-slurm-recipes/configs/dsv41flash-b300-efa-nixl.sh"
    --set "roles.prefill.args.speculative-config=$dsuf_draft"
    --set "roles.decode.args.speculative-config=$dsuf_draft"
    --set roles.prefill.args.max-model-len=131072
    --set roles.decode.args.max-model-len=131072
)
dsuf_srtctl=(uv run --directory "$dsuf_infx_project/utils/srt-slurm" --no-dev srtctl)
"${dsuf_srtctl[@]}" dry-run "${dsuf_pd_args[@]}"
```

Check that the render requests two nodes and preserves the four prefill
groups and single decode group. On the qualified site, submit the already
reviewed configuration as a serving-only job. This uses the deployment's
front end for the same trace as the other lanes.

```bash
"${dsuf_srtctl[@]}" apply --serve-only "${dsuf_pd_args[@]}"
```

## Measure the comparison

Confirm all five worker groups registered, and check the private Dynamo
front end with one cold and one same-session extension request. Verify
that workers exchange KV over NIXL LIBFABRIC, and record accepted drafts,
prefix lengths, GPU/host memory peaks, and prefill-to-decode transfer time
and errors. The trial requires measured prompt and cached-token usage to
classify cold and warm turns. If the front end omits those fields, report
the missing evidence as a qualification miss.

Replay the same frozen coding manifest through the private Dynamo
`/v1/chat/completions` endpoint. Use `deepseek-ai/DeepSeek-V4.1-Flash` as
the served model name and the same caps as the other backends. Set
`DSUF_PD_URL` to that private Chat Completions endpoint. Build
`DSUF_PD_INVENTORY` from the serving allocation and worker startup records.
It uses the harness's schema version 1. Record the five worker groups with
their host, physical GPU UUIDs, startup identity in `epoch`, pinned image
digest and checkpoint revision. Use `topology.backend: "dynamo"`, stage
`"prefill"` or `"decode"`, and startup source `"dynamo_ready"` after
verifying worker registration. Record the decoder width if reported.

```sh
python3 tools/bench_coding.py MANIFEST --url "$DSUF_PD_URL" \
    --model deepseek-ai/DeepSeek-V4.1-Flash --request-format vllm \
    --session-header X-Dynamo-Session-ID --gpus 16 --inventory "$DSUF_PD_INVENTORY" --mode open \
    --arrival-scales 1,1.25,1.5 --isolate-replays --qualify load
```

Follow the [two-node deployment runbook](../README.md#run-the-qualification)
for the 500-session soak.
Record **useful output goodput** within the same latency and reliability
targets. The upstream variant's offered concurrency of 2,048 describes
its AgentX load point. Both roles are required for serving, and a host
loss removes a complete role in this two-host layout.
