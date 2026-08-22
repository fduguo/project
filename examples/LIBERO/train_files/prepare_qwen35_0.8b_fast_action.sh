#!/usr/bin/env bash
set -euo pipefail

# Run from the StarVLA repository root.
source_model_id=${source_model_id:-/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B}
target_model_id=${target_model_id:-/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action}
fast_token_list=${fast_token_list:-starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt}

python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen35.py \
  --model-id "${source_model_id}" \
  --tokens-file "${fast_token_list}" \
  --save-dir "${target_model_id}" \
  --init-strategy normal

