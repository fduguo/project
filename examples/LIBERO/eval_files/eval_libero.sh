#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${STARVLA_DIR}"

# Change these two values, or override them in one command:
# CKPT=/path/to/checkpoint.pt bash examples/LIBERO/eval_files/eval_libero.sh
CKPT="${CKPT:-/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35_08b_ki_libero_depth_head/checkpoints/steps_30000_pytorch_model.pt}"

# Environment defaults for this machine. Override with env vars when needed.
STARVLA_PYTHON="${STARVLA_PYTHON:-/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/qwen35_fa2_test/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-/mnt/nas/gezuhao/zhouyuchen/project/LIBERO}"
LIBERO_Python="${LIBERO_Python:-/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/libero/bin/python}"
host="${EVAL_HOST:-127.0.0.1}"
select_available_port() {
    "${LIBERO_Python}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
}

resolve_port() {
    local explicit_port="${1:-auto}"
    if [[ -n "${explicit_port}" && "${explicit_port}" != "auto" ]]; then
        echo "${explicit_port}"
        return
    fi

    select_available_port
}

base_port="$(resolve_port "${PORT:-${BASE_PORT:-auto}}")"
select_available_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
            | awk -F',' '{
                gsub(/^[ \t]+|[ \t]+$/, "", $1)
                gsub(/^[ \t]+|[ \t]+$/, "", $2)
                gsub(/^[ \t]+|[ \t]+$/, "", $3)
                if ($1 != "") print $2, $3, $1
            }' \
            | sort -n -k1,1 -k2,2 -k3,3 \
            | awk 'NR == 1 {print $3}'
    fi
}

select_all_available_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
            | awk '{gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "") print $1}' \
            | paste -sd ','
    fi
}

resolve_gpu() {
    local explicit_gpu="${1:-auto}"
    if [[ -n "${explicit_gpu}" && "${explicit_gpu}" != "auto" ]]; then
        echo "${explicit_gpu}"
        return
    fi

    local selected_gpu
    selected_gpu="$(select_available_gpu)"
    echo "${selected_gpu:-0}"
}

# Resolve all available GPUs for auto-distribution
all_gpus="$(select_all_available_gpus)"

# POLICY GPUs: use POLICY_GPUS or POLICY_GPU, fallback to all available GPUs
if [[ -n "${POLICY_GPUS:-}" && "${POLICY_GPUS}" != "auto" ]]; then
    policy_gpus_raw="${POLICY_GPUS}"
elif [[ "${POLICY_GPUS:-}" == "auto" ]] || { [[ -z "${POLICY_GPUS:-}" ]] && [[ -z "${POLICY_GPU:-}" ]]; }; then
    policy_gpus_raw="${all_gpus:-0}"
elif [[ -n "${POLICY_GPU:-}" ]]; then
    policy_gpu_id="$(resolve_gpu "${POLICY_GPU}")"
    policy_gpus_raw="${policy_gpu_id}"
else
    policy_gpu_id="$(resolve_gpu "auto")"
    policy_gpus_raw="${policy_gpu_id}"
fi
IFS=',' read -r -a policy_gpu_ids <<< "${policy_gpus_raw}"
for i in "${!policy_gpu_ids[@]}"; do
    policy_gpu_ids[$i]="${policy_gpu_ids[$i]//[[:space:]]/}"
done
if [[ "${#policy_gpu_ids[@]}" -eq 0 ]] || [[ -z "${policy_gpu_ids[0]}" ]]; then
    policy_gpu_ids=("0")
fi

# EVAL GPUs: use EVAL_GPUS or EVAL_GPU, fallback to all available GPUs
if [[ -n "${EVAL_GPUS:-}" && "${EVAL_GPUS}" != "auto" ]]; then
    eval_gpus_raw="${EVAL_GPUS}"
elif [[ "${EVAL_GPUS:-}" == "auto" ]] || { [[ -z "${EVAL_GPUS:-}" ]] && [[ -z "${EVAL_GPU:-}" ]]; }; then
    eval_gpus_raw="${all_gpus:-0}"
elif [[ -n "${EVAL_GPU:-}" ]]; then
    eval_gpu_id="$(resolve_gpu "${EVAL_GPU}")"
    eval_gpus_raw="${eval_gpu_id}"
else
    eval_gpu_id="$(resolve_gpu "auto")"
    eval_gpus_raw="${eval_gpu_id}"
fi
IFS=',' read -r -a eval_gpu_ids <<< "${eval_gpus_raw}"
for i in "${!eval_gpu_ids[@]}"; do
    eval_gpu_ids[$i]="${eval_gpu_ids[$i]//[[:space:]]/}"
done
if [[ "${#eval_gpu_ids[@]}" -eq 0 ]] || [[ -z "${eval_gpu_ids[0]}" ]]; then
    eval_gpu_ids=("0")
fi
num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"
server_wait_timeout="${SERVER_WAIT_TIMEOUT:-900}"

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"

ckpt_file="$(basename "${CKPT}")"
ckpt_parent="$(basename "$(dirname "${CKPT}")")"
ckpt_grandparent="$(basename "$(dirname "$(dirname "${CKPT}")")")"
folder_name="${ckpt_grandparent}_${ckpt_parent}_${ckpt_file}"
if [[ "${CKPT}" == *"/checkpoints/"* ]]; then
    model_root="${CKPT%%/checkpoints/*}"
else
    model_root="$(dirname "$(dirname "${CKPT}")")"
fi
model_name="${model_root##*/}"
output_dir="${OUTPUT_DIR:-${model_root}/eval_outputs/libero}"
task_suites=(${TASK_SUITES:-libero_10 libero_goal libero_object libero_spatial})

LOG_DIR="${LOG_DIR:-${output_dir}/logs/$(date +"%Y%m%d_%H%M%S")_$$}"
mkdir -p "${LOG_DIR}"
policy_pids=()

cleanup_policy_servers() {
    for policy_pid in "${policy_pids[@]}"; do
        if [[ -n "${policy_pid}" ]] && kill -0 "${policy_pid}" 2>/dev/null; then
            echo "Stopping policy server pid=${policy_pid}"
            kill "${policy_pid}" 2>/dev/null || true
            wait "${policy_pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup_policy_servers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_policy_server() {
    local policy_pid="$1"
    local policy_log="$2"
    local policy_port="$3"
    local deadline=$((SECONDS + server_wait_timeout))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${policy_pid}" 2>/dev/null; then
            echo "Policy server exited before eval started. See ${policy_log}"
            tail -n 80 "${policy_log}" || true
            exit 1
        fi
        if "${LIBERO_Python}" - "${host}" "${policy_port}" >/dev/null 2>&1 <<'PY'
import sys
import websockets.sync.client
from deployment.model_server.tools import msgpack_numpy

host = sys.argv[1]
port = int(sys.argv[2])
conn = websockets.sync.client.connect(
    f"ws://{host}:{port}",
    compression=None,
    max_size=None,
    open_timeout=2,
    ping_interval=None,
)
msgpack_numpy.unpackb(conn.recv())
conn.close()
PY
        then
            return 0
        fi
        sleep 5
    done
    echo "Timed out waiting for policy server at ${host}:${policy_port}. See ${policy_log}"
    tail -n 80 "${policy_log}" || true
    exit 1
}

wandb_project="${WANDB_EVAL_PROJECT:-${model_name}_libero}"
wandb_entity="${WANDB_EVAL_ENTITY:-13328167328-}"
wandb_args=(--args.wandb-project "${wandb_project}" --args.wandb-entity "${wandb_entity}")
if [[ -n "${WANDB_EVAL_GROUP:-}" ]]; then
    wandb_args+=(--args.wandb-group "${WANDB_EVAL_GROUP}")
fi
if [[ -n "${WANDB_EVAL_MODE:-}" ]]; then
    wandb_args+=(--args.wandb-mode "${WANDB_EVAL_MODE}")
fi
if [[ -n "${WANDB_EVAL_TAGS:-}" ]]; then
    wandb_args+=(--args.wandb-tags "${WANDB_EVAL_TAGS}")
fi

echo "Evaluating checkpoint: ${CKPT}"
echo "Task suites: ${task_suites[*]}"
echo "Policy GPUs: ${policy_gpu_ids[*]}"
echo "Eval GPUs: ${eval_gpu_ids[*]}"
if [[ "${#policy_gpu_ids[@]}" -gt 1 ]]; then
    echo "Policy mode: one server per suite; ports start at ${base_port}"
else
    echo "Policy server: ${host}:${base_port} on GPU ${policy_gpu_ids[0]}"
fi
echo "Logs: ${LOG_DIR}"
echo "wandb project: ${wandb_project}"
echo "wandb entity: ${wandb_entity}"

shared_policy_port="${base_port}"
shared_policy_log="${LOG_DIR}/policy_server.log"
if [[ "${#policy_gpu_ids[@]}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${policy_gpu_ids[0]}" "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT}" \
        --port "${shared_policy_port}" \
        --use_bf16 \
        >"${shared_policy_log}" 2>&1 &
    shared_policy_pid=$!
    policy_pids+=("${shared_policy_pid}")
    echo "Started shared policy server pid=${shared_policy_pid}; log=${shared_policy_log}"
    wait_for_policy_server "${shared_policy_pid}" "${shared_policy_log}" "${shared_policy_port}"
fi

eval_pids=()
eval_extra_args=()
if [[ -n "${MAX_STEPS_OVERRIDE:-}" ]]; then
    eval_extra_args+=(--args.max-steps-override "${MAX_STEPS_OVERRIDE}")
fi
if [[ -n "${MAX_TASKS:-}" ]]; then
    eval_extra_args+=(--args.max-tasks "${MAX_TASKS}")
fi
if [[ -n "${UNNORM_KEY:-}" ]]; then
    eval_extra_args+=(--args.unnorm-key "${UNNORM_KEY}")
fi
suite_idx=0
for task_suite_name in "${task_suites[@]}"; do
    suite_eval_gpu="${eval_gpu_ids[$((suite_idx % ${#eval_gpu_ids[@]}))]}"
    suite_policy_gpu="${policy_gpu_ids[$((suite_idx % ${#policy_gpu_ids[@]}))]}"
    suite_policy_port="${shared_policy_port}"
    if [[ "${#policy_gpu_ids[@]}" -gt 1 ]]; then
        suite_policy_port=$((base_port + suite_idx))
        suite_policy_log="${LOG_DIR}/policy_server_${task_suite_name}.log"
        CUDA_VISIBLE_DEVICES="${suite_policy_gpu}" "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
            --ckpt_path "${CKPT}" \
            --port "${suite_policy_port}" \
            --use_bf16 \
            >"${suite_policy_log}" 2>&1 &
        suite_policy_pid=$!
        policy_pids+=("${suite_policy_pid}")
        echo "Started policy server for ${task_suite_name} pid=${suite_policy_pid}; gpu=${suite_policy_gpu}; port=${suite_policy_port}; log=${suite_policy_log}"
        wait_for_policy_server "${suite_policy_pid}" "${suite_policy_log}" "${suite_policy_port}"
    fi
    video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
    log_file="${LOG_DIR}/${task_suite_name}.log"
    mkdir -p "${video_out_path}"

    echo "========== Evaluating ${task_suite_name}: policy GPU ${suite_policy_gpu}, eval GPU ${suite_eval_gpu}, port ${suite_policy_port} =========="
    (
        set -o pipefail
        CUDA_VISIBLE_DEVICES="${suite_eval_gpu}" "${LIBERO_Python}" ./examples/LIBERO/eval_files/eval_libero.py \
            --args.pretrained-path "${CKPT}" \
            --args.host "${host}" \
            --args.port "${suite_policy_port}" \
            --args.task-suite-name "${task_suite_name}" \
            --args.num-trials-per-task "${num_trials_per_task}" \
            "${eval_extra_args[@]}" \
            --args.video-out-path "${video_out_path}" \
            --args.log-path "${LOG_DIR}" \
            --args.wandb-run-name "${folder_name}_${task_suite_name}" \
            "${wandb_args[@]}" \
            2>&1 | tee "${log_file}"
    ) &
    eval_pids+=("$!")
    suite_idx=$((suite_idx + 1))
done

echo "Waiting for all evaluation tasks to finish..."
for eval_pid in "${eval_pids[@]}"; do
    wait "${eval_pid}"
done
echo "All LIBERO evaluation tasks finished. Logs saved to ${LOG_DIR}"
