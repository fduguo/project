#!/bin/bash
# ============================================================
# QwenKI + LayerwiseFM + 0.8B-Action 正式训练脚本
# 24个任务，30k步，单卡启动（如需多卡请修改num_processes）
# ============================================================

# 网络通信配置
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export CUDA_VISIBLE_DEVICES=3,4,5

# === 框架与模型 ===
Framework_name=QwenKI
base_vlm=/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action
freeze_module_list=''
DIT_TYPE="LayerwiseFM"

# === 数据集 ===
data_root_dir=/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA/playground/Datasets/nvidia
data_mix=fourier_gr1_unified_1000

# === 训练配置 ===
config_yaml=./examples/Robocasa_tabletop/train_files/starvla_qwen35_08b_ki_robocasa.yaml
run_root_dir=./playground/Checkpoints
run_id=qwenki_robocasa_30k

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# === 启动训练 ===
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 3 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.learning_rate.base 3e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}