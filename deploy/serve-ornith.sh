#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: serve-ornith.sh [--dry-run] [nvfp4|fp8] [additional vLLM serve flags]

Launch one Ornith-1.5-122B replica with vLLM 0.34.0.
Set CUDA_VISIBLE_DEVICES and PORT for each replica. Defaults are GPU 0 and port 8100.
Override HOST, TP_SIZE, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_IN_FLIGHT, and
MAX_BATCHED_TOKENS to benchmark a different deployment shape.
EOF
}

dry_run=false
case ${1:-} in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
esac

model=ornith-ai/Ornith-1.5-122B-NVFP4
revision=f049036f794622db10bb69855bf9819207334626
case ${1:-} in
    nvfp4) shift ;;
    fp8)
        model=ornith-ai/Ornith-1.5-122B-FP8
        revision=40f77767b14a2a70bdbafa0ff26a0bfae1aedf3b
        shift
        ;;
    ''|--*) ;;
    *) usage >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
args=(
    serve "$model" --revision "$revision" --tokenizer-revision "$revision"
    --served-model-name Ornith-1.5-122B
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8100}"
    --tensor-parallel-size "${TP_SIZE:-1}"
    --max-model-len "${MAX_MODEL_LEN:-65536}"
    --gpu-memory-utilization 0.90
    --max-num-seqs "${MAX_NUM_SEQS:-32}"
    --max-num-queued-reqs "${MAX_IN_FLIGHT:-40}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}"
    --enable-prefix-caching --enable-chunked-prefill
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
)

if "$dry_run"; then
    printf 'CUDA_VISIBLE_DEVICES=%q vllm' "$CUDA_VISIBLE_DEVICES"
    printf ' %q' "${args[@]}" "$@"
    printf '\n'
    exit 0
fi

exec vllm "${args[@]}" "$@"
