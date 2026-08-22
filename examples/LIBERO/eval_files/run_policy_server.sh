#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA
LIBERO_HOME=/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/LIBERO
STARVLA_PYTHON=/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/qwen35_fa2_test/bin/python
LIBERO_PYTHON=/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/libero/bin/python

# === Checkpoint ===
# CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen3vl_2b_pi_libero_baseline/checkpoints/steps_30000_pytorch_model.pt
# CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35vl_0.8b_pi_libero_baseline/checkpoints/steps_30000_pytorch_model.pt
# CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35_08b_pi_libero_baseline_ft/checkpoints/steps_30000_pytorch_model.pt
# CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35_08b_ki_libero/checkpoints/steps_30000_pytorch_model.pt

# CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/Checkpoints/starVLA_PI_LIBERO_4in1/checkpoints/steps_100000_pytorch_model.pt

# CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35_08b_ki_libero/checkpoints/steps_30000_pytorch_model.pt
# CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35_08b_ki_libero_new/checkpoints/steps_30000_pytorch_model.pt
CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35_08b_ki_libero_depth_head/checkpoints/steps_30000_pytorch_model.pt
export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}   
gpu_id=4
# port=6694
port=9884
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
