#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B 16xGPU (2-node) fully async training script for DAPO math dataset.
#
# Usage:
#   bash scripts/training/text/run-ettr-async-cg16.sh

set -e
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=4350}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save "${EXP_DIR}"
   --save-interval "${NUM_ROLLOUT}"
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 8
   --use-fault-tolerance
   --balance-data
)

# Same established recipe; no evaluation dataset or pre-training evaluation.
EVAL_ARGS=()

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --log-probs-max-tokens-per-gpu 40960

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   # NOTE(wuhuan): to avoid algorithm performance degradation
   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-load-format dummy
   --sglang-moe-runner-backend triton
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 16
)

WANDB_ARGS=(
   --use-tensorboard
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name ${RUN_NAME:?RUN_NAME must be set}
)

MISC_ARGS=(
   --seed 1234
   --rollout-seed 42
   --disable-s3-model-download
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Match the reference CE run even when ray-job.sh detects NVLink.
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os
runtime = json.loads(os.environ['RUNTIME_ENV_JSON'])
runtime.setdefault('env_vars', {})['NCCL_NVLS_ENABLE'] = '0'
print(json.dumps(runtime))
PY
)

mkdir -p log
ray job submit --submission-id "${RUN_NAME:?}" ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD_ADDRESS:?}" \
   --entrypoint-resources="${RAY_ENTRYPOINT_RESOURCES:?}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8], "advantages": [1, 0]}' \
   --fully-async \
   --num-iters-per-train-update 1 \
   --max-staleness 2 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-35B-A3B-GRPO-gpu16-async-${now}.log
