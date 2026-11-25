#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B full SFT on one 8-GPU node with prepared OpenAI Chat batches.
# Usage:
#   MODEL_DIR=... DATA_DIR=... SAVE_DIR=... bash scripts/training/sft/run-qwen3-0.6B-8xgpu-prepared-sft.sh
#   Add --dry-run to print the native Relax command.

set -eo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Set MODEL_DIR (HF model or its parent), DATA_DIR (containing train.jsonl), and SAVE_DIR."
    echo "Run with --dry-run to preview. Set PUBLISH_MODEL_NAME to publish the final HF export."
    exit 0
fi
if (( $# > 1 )) || [[ -n ${1:-} && $1 != --dry-run ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${RELAX_ROOT}/scripts/models/qwen3-0.6B.sh"

: "${MODEL_DIR:?Set MODEL_DIR to the HF model directory or its parent}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing the JSONL dataset}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"

MODEL_DIR="$(realpath -m -- "${MODEL_DIR}")"
DATA_DIR="$(realpath -m -- "${DATA_DIR}")"
SAVE_DIR="$(realpath -m -- "${SAVE_DIR}")"
HF_MODEL_DIR="${MODEL_DIR}/Qwen3-0.6B"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
    HF_MODEL_DIR="${MODEL_DIR}"
fi
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/openai-chat}"
DATA_PIPELINE="${DATA_PIPELINE:-prepared}"
EXP_NAME="${EXP_NAME:-qwen3-0.6b-openai-chat-8xgpu-${DATA_PIPELINE}}"
RUN_DIR="${SAVE_DIR}/${EXP_NAME}"
now=$(date "+%Y%m%d-%H%M%S")

CKPT_ARGS=(
    --hf-checkpoint "${HF_MODEL_DIR}"
    --ref-load "${HF_MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
    --load "${RUN_DIR}/output"
    --save "${RUN_DIR}/output"
    --save-hf "${RUN_DIR}/output_hf/iter_{rollout_id}"
    --save-interval "${SAVE_INTERVAL:-50000}"
    --max-actor-ckpt-to-keep 2
)

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}/${DATA_FILE:-train.jsonl}"
    --input-key messages
    --num-epoch 1
    --global-batch-size 8192
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
    --seq-length 32768
    --sft-oversize-strategy skip
    --sft-invalid-multimodal-strategy skip
    --sft-max-in-flight-steps 4
    --balance-data
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --calculate-per-token-loss
    --cross-entropy-loss-fusion
    --no-rope-fusion
    --manual-gc
    --manual-gc-interval 1000
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 2e-5
    --min-lr 1e-6
    --lr-decay-style cosine
    --lr-warmup-fraction 0.1
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
)

MONITOR_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${ATTEMPT_ID:-initial}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-health-check
    --seed 1234
)

LAUNCH_ARGS=(
    --job-spec -
    --run-dir "${RUN_DIR}"
    --ray-dashboard "${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
)
case "${DATA_PIPELINE}" in
    prepared)
        if [[ -z ${LOCAL_CACHE_DIR:-} ]]; then
            LOCAL_CACHE_DIR=$(python3 -c '
import hashlib
from pathlib import Path
import sys
import tempfile

key = hashlib.sha256(sys.argv[1].encode()).hexdigest()[:16]
print(Path(tempfile.gettempdir()) / ("relax-sft-" + key))
' "${RUN_DIR}")
        fi
        LAUNCH_ARGS+=(
            --prepare-dir "${LOCAL_CACHE_DIR}"
            --prep-workers "${PREP_WORKERS:-14}"
        )
        ;;
    transfer-queue)
        SFT_ARGS+=(
            --sft-compact-tq-text
            --sft-prefetch-num-workers 16
            --sft-prefetch-buffer-size 4096
        )
        ;;
    *)
        echo "DATA_PIPELINE must be prepared or transfer-queue." >&2
        exit 2
        ;;
esac
if [[ -n ${PUBLISH_MODEL_NAME:-} ]]; then
    LAUNCH_ARGS+=(
        --model-name "${PUBLISH_MODEL_NAME}"
        --publisher-user-env "${PUBLISH_USER_ENV:-MODEL_STORE_USER}"
        --publisher-token-env "${PUBLISH_TOKEN_ENV:-MODEL_STORE_TOKEN}"
        --publisher-region "${PUBLISH_REGION:-}"
    )
fi
if [[ -n ${ATTEMPT_ID:-} ]]; then
    LAUNCH_ARGS+=(
        --attempt "${ATTEMPT_ID}"
    )
fi

if [[ ${1:-} == --dry-run ]]; then
    LAUNCH_ARGS+=(--dry-run)
else
    export NUM_GPUS=8
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    if [[ -z ${RELAX_ENTRYPOINT_MODE:-} ]]; then
        if [[ -n ${RAY_ADDRESS:-} ]] || timeout 5 ray status >/dev/null 2>&1; then
            export RELAX_SKIP_CLUSTER_CLEANUP=1
            source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
        else
            source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
        fi
        set +x
    fi
    mkdir -p -- "${RUN_DIR}"
fi

TRAIN_ARGS=(
    python3
    -m relax.entrypoints.train
    --resource '{"sft":[1,0],"actor":[1,8]}'
    --num-data-storage-units 1
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${MONITOR_ARGS[@]}"
    "${MISC_ARGS[@]}"
)

make_job_spec() {
    python3 - "${RELAX_ROOT}" "${TRAIN_ARGS[@]}" <<'PY'
import json
import os
import shlex
import sys

runtime_env = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
env = runtime_env.setdefault("env_vars", {})
python_path = env.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
env["PYTHONPATH"] = os.pathsep.join(part for part in (sys.argv[1], python_path) if part)
json.dump({"entrypoint": shlex.join(sys.argv[2:]), "runtime_env": runtime_env}, sys.stdout)
PY
}

if [[ ${1:-} == --dry-run ]]; then
    make_job_spec | python3 "${SCRIPT_DIR}/relax_sft_train_and_publish.py" "${LAUNCH_ARGS[@]}"
else
    make_job_spec | python3 "${SCRIPT_DIR}/relax_sft_train_and_publish.py" "${LAUNCH_ARGS[@]}" 2>&1 \
        | tee "${RUN_DIR}/launch-${now}.log"
fi
