#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: serve-inkling.sh [--dry-run] [dspark|plain] [additional SGLang flags]

Launch one Inkling 2 Mini replica using the Inkling-specific SGLang build.
Set CUDA_VISIBLE_DEVICES and PORT for each replica. Defaults are GPU 0 and port 8100.
Override HOST, MAX_MODEL_LEN, and MAX_NUM_SEQS to change the serving limits.
EOF
}

dry_run=false
case ${1:-} in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
esac

speculative=(--speculative-algorithm DSPARK)
case ${1:-} in
    dspark) shift ;;
    plain) speculative=(); shift ;;
    ''|--*) ;;
    *) usage >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
args=(
    -m sglang.launch_server
    --model-path thinkingmachines/Inkling-2-Mini
    --revision b85c3cfa615fc413cc9af3379f491b963f3b586a
    --served-model-name inkling
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8100}"
    --attention-backend fa4 --bf16-gemm-backend torch --dtype bfloat16
    --tp 1 --page-size 128 --grammar-backend xgrammar
    --context-length "${MAX_MODEL_LEN:-131072}"
    --max-running-requests "${MAX_NUM_SEQS:-64}"
    --reasoning-parser inkling --tool-call-parser inkling2
)

if "$dry_run"; then
    printf 'CUDA_VISIBLE_DEVICES=%q python3' "$CUDA_VISIBLE_DEVICES"
    printf ' %q' "${args[@]}" "${speculative[@]}" "$@"
    printf '\n'
    exit 0
fi

exec python3 "${args[@]}" "${speculative[@]}" "$@"
