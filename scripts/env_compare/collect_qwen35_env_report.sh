#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Parameters to modify per machine
# ============================================================
# REPO_DIR: starVLA repository path on this machine.
# CONDA_SH: conda shell hook path on this machine.
# CONDA_ENV: conda environment name to inspect.
# SSH_KEY: optional; only needed if your workflow requires GIT_SSH_COMMAND.
# MODEL_PATHS: optional space-separated paths whose checksums / sizes should be compared.
# DATA_PATHS: optional space-separated dataset paths whose sizes / file counts should be compared.
# OUTPUT_ROOT: where to write the report directory and .tar.gz.
# ============================================================
REPO_DIR="${REPO_DIR:-/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA}"
CONDA_SH="${CONDA_SH:-/mnt/nas/gezuhao/zhouyuchen/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-qwen35_fa2_test}"
SSH_KEY="${SSH_KEY:-/mnt/nas/gezuhao/zhouyuchen/id_edzyc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}}"

MODEL_PATHS="${MODEL_PATHS:-/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action /mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/da3-small}"
DATA_PATHS="${DATA_PATHS:-/mnt/nas/gezuhao/zhouyuchen/dataset/LIBERO/libero}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "CONDA_SH not found: ${CONDA_SH}" >&2
  exit 1
fi
if [[ ! -d "${REPO_DIR}" ]]; then
  echo "REPO_DIR not found: ${REPO_DIR}" >&2
  exit 1
fi

source "${CONDA_SH}"
unalias python 2>/dev/null || true
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
if [[ -n "${SSH_KEY}" ]]; then
  export GIT_SSH_COMMAND="ssh -i ${SSH_KEY}"
fi
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"

host="$(hostname 2>/dev/null || echo unknown_host)"
stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${OUTPUT_ROOT}/env_report_${host}_${CONDA_ENV}_${stamp}"
mkdir -p "${out_dir}"

run_capture() {
  local name="$1"
  shift
  {
    echo "# command: $*"
    "$@"
  } > "${out_dir}/${name}" 2>&1 || true
}

run_shell_capture() {
  local name="$1"
  shift
  {
    echo "# command: $*"
    bash -lc "$*"
  } > "${out_dir}/${name}" 2>&1 || true
}

printf '%s\n' "${REPO_DIR}" > "${out_dir}/repo_dir.txt"
printf '%s\n' "${CONDA_ENV}" > "${out_dir}/conda_env.txt"
printf '%s\n' "${CONDA_PREFIX:-}" > "${out_dir}/conda_prefix.txt"

run_capture hostname.txt hostname
run_capture uname.txt uname -a
run_capture date_utc.txt date -u
run_capture python_path.txt which python
run_capture python_version.txt python -V
run_capture conda_info.txt conda info
run_capture conda_list.txt conda list
run_capture conda_list_explicit.txt conda list --explicit
run_capture pip_freeze.txt python -m pip freeze
run_capture pip_check.txt python -m pip check

run_capture git_head.txt git rev-parse HEAD
run_capture git_branch.txt git branch --show-current
run_capture git_status.txt git status --short
run_capture git_diff_stat.txt git diff --stat
run_capture git_remote.txt git remote -v

run_capture nvidia_smi.txt nvidia-smi
run_capture gpu.csv nvidia-smi --query-gpu=index,name,driver_version,cuda_version,memory.total,compute_cap --format=csv

env | sort > "${out_dir}/env_vars.txt"

python - <<'PY' > "${out_dir}/python_runtime.txt" 2>&1
import importlib
import os
import sys

mods = [
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "deepspeed",
    "flash_attn",
    "triton",
    "numpy",
    "scipy",
    "PIL",
    "cv2",
    "datasets",
    "tokenizers",
    "safetensors",
    "diffusers",
    "libero",
    "h5py",
    "zarr",
]

print("sys.executable:", sys.executable)
print("sys.version:", sys.version)
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: version={getattr(mod, '__version__', 'NA')} file={getattr(mod, '__file__', 'NA')}")
    except Exception as exc:
        print(f"{name}: IMPORT_ERROR {type(exc).__name__}: {exc}")

try:
    import torch

    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cudnn.version:", torch.backends.cudnn.version())
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    print("torch.backends.cuda.matmul.allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
    print("torch.backends.cudnn.allow_tf32:", torch.backends.cudnn.allow_tf32)
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"gpu{idx}.name:", torch.cuda.get_device_name(idx))
            print(f"gpu{idx}.capability:", torch.cuda.get_device_capability(idx))
except Exception as exc:
    print("torch_runtime_error:", repr(exc))
PY

{
  echo "# Important config checksums"
  for path in \
    starVLA/config/deepseeds/ds_config.yaml \
    starVLA/config/deepseeds/deepspeed_zero2.yaml \
    examples/LIBERO/train_files/starvla_qwen35_0.8b_oft_depth.yaml \
    examples/LIBERO/train_files/starvla_qwen35_0.8b_oft_depth_local.yaml \
    examples/LIBERO/train_files/starvla_qwen35_0.8b_oft_depth_head_0_3.yaml \
    examples/LIBERO/train_files/starvla_qwen35_0.8b_ki_depth.yaml
  do
    if [[ -f "${path}" ]]; then
      sha256sum "${path}"
    else
      echo "MISSING ${path}"
    fi
  done
} > "${out_dir}/repo_file_checksums.txt"

{
  echo "# MODEL_PATHS"
  for path in ${MODEL_PATHS}; do
    if [[ -e "${path}" ]]; then
      echo "PATH ${path}"
      du -sh "${path}" || true
      find "${path}" -type f | wc -l | awk '{print "file_count " $1}'
      find "${path}" -maxdepth 2 -type f \( -name 'config.json' -o -name '*.json' -o -name '*.txt' \) -print | sort | while read -r file; do
        sha256sum "${file}" || true
      done
    else
      echo "MISSING ${path}"
    fi
    echo
  done
} > "${out_dir}/model_paths.txt"

{
  echo "# DATA_PATHS"
  for path in ${DATA_PATHS}; do
    if [[ -e "${path}" ]]; then
      echo "PATH ${path}"
      du -sh "${path}" || true
      find "${path}" -type f | wc -l | awk '{print "file_count " $1}'
      find "${path}" -maxdepth 3 -type f \( -name 'meta.json' -o -name 'modality.json' -o -name 'dataset_statistics.json' -o -name '*.json' \) -print | sort | while read -r file; do
        sha256sum "${file}" || true
      done
    else
      echo "MISSING ${path}"
    fi
    echo
  done
} > "${out_dir}/data_paths.txt"

tar_path="${out_dir}.tar.gz"
tar -czf "${tar_path}" -C "${OUTPUT_ROOT}" "$(basename "${out_dir}")"

echo "Report directory: ${out_dir}"
echo "Report archive:   ${tar_path}"
echo
echo "Copy this .tar.gz from both machines to one place, then run:"
echo "  bash scripts/env_compare/compare_qwen35_env_reports.sh L20_report.tar.gz 4090_report.tar.gz"
