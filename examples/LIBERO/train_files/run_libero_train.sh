

# export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
# export WANDB_MODE=disabled

config_yaml=${config_yaml:-./examples/LIBERO/train_files/starvla_cotrain_libero.yaml}
if [[ $# -gt 0 && "$1" == *.yaml ]]; then
  config_yaml="$1"
  shift
fi
num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

# if [[ "${SYNC_LIBERO_TO_SHM:-1}" == "1" ]]; then
#   bash "$(dirname "$0")/sync_libero_to_shm.sh"
# fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port 29501 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  "$@"



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml "${config_yaml}" \
  #   "$@"
##### Multi-Server Multi-GPU training script #####
