#!/usr/bin/env bash
set -euo pipefail

cd /mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA

MODEL_DIR=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35_08b_ki_libero_depth_all_layers_uniform_segments

echo "[$(date '+%F %T')] scheduled: sleeping 3h before LIBERO-plus eval for ${MODEL_DIR}"
sleep 3h

CKPT="$(ls -1 "${MODEL_DIR}"/checkpoints/steps_*_pytorch_model.pt | sort -V | tail -n 1)"
echo "[$(date '+%F %T')] selected checkpoint: ${CKPT}"

unset PYTHONPATH POLICY_GPU EVAL_GPU PORT BASE_PORT OUTPUT_DIR LOG_DIR TASK_SUITES
export WANDB_EVAL_MODE=offline
export CKPT
export PORT=16111
export LOG_DIR="${MODEL_DIR}/eval_outputs/libero_plus/logs/$(date +%Y%m%d_%H%M%S)_delayed_3h"

bash examples/LIBERO-plus/eval_files/eval_libero.sh
