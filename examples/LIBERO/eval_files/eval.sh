#!/bin/bash
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA

cd ${STARVLA_DIR}
# === Checkpoint ===
# CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35vl_0.8b_pi_libero_baseline/checkpoints/steps_30000_pytorch_model.pt
CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/Checkpoints/starVLA_PI_LIBERO_4in1/checkpoints/steps_100000_pytorch_model.pt
###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/nas/gezuhao/zhouyuchen/project/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

host="127.0.0.1"
base_port=5679  # 原来是 6694
unnorm_key="franka"
your_ckpt=${CKPT}

# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# model_root: playground/Checkpoints/<run_id>
model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}')
output_dir="${model_root}/eval_outputs/libero"
# === End of environment variable configuration ===
###########################################################################################

num_trials_per_task=50
TASK_SUITES=(libero_10 libero_goal libero_object libero_spatial)

LOG_DIR="${output_dir}/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"

echo "Evaluating checkpoint: ${your_ckpt}"
echo "Task suites: ${TASK_SUITES[*]}"
echo "Logs: ${LOG_DIR}"

for task_suite_name in "${TASK_SUITES[@]}"; do
    video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
    log_file="${LOG_DIR}/${task_suite_name}.log"
    mkdir -p "${video_out_path}"

    echo "========== Evaluating ${task_suite_name} =========="
    ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port $base_port \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" \
        2>&1 | tee "${log_file}"
done

echo "All LIBERO evaluation tasks finished. Logs saved to ${LOG_DIR}"
