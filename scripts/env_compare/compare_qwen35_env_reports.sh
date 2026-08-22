#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Parameters to modify
# ============================================================
# No required edits if you pass two report archives as arguments.
# WORK_DIR can be changed if you want comparison output elsewhere.
# ============================================================
WORK_DIR="${WORK_DIR:-./env_report_compare_$(date +%Y%m%d_%H%M%S)}"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: bash $0 <machine_a_env_report.tar.gz> <machine_b_env_report.tar.gz>" >&2
  echo "Example:" >&2
  echo "  bash $0 env_report_L20_qwen35_fa2_test_*.tar.gz env_report_4090_qwen35_fa2_test_*.tar.gz" >&2
  exit 1
fi

archive_a="$1"
archive_b="$2"

if [[ ! -f "${archive_a}" ]]; then
  echo "Archive not found: ${archive_a}" >&2
  exit 1
fi
if [[ ! -f "${archive_b}" ]]; then
  echo "Archive not found: ${archive_b}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"
tar -xzf "${archive_a}" -C "${WORK_DIR}"
tar -xzf "${archive_b}" -C "${WORK_DIR}"

mapfile -t reports < <(find "${WORK_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#reports[@]}" -ne 2 ]]; then
  echo "Expected exactly two report directories after extraction; found ${#reports[@]}." >&2
  find "${WORK_DIR}" -mindepth 1 -maxdepth 1 -type d -print >&2
  exit 1
fi

report_a="${reports[0]}"
report_b="${reports[1]}"
diff_file="${WORK_DIR}/env_diff.txt"
summary_file="${WORK_DIR}/summary.txt"

diff -ru "${report_a}" "${report_b}" > "${diff_file}" || true

{
  echo "Report A: ${report_a}"
  echo "Report B: ${report_b}"
  echo
  echo "===== Git HEAD ====="
  echo "A: $(tr -d '\n' < "${report_a}/git_head.txt" 2>/dev/null || echo MISSING)"
  echo "B: $(tr -d '\n' < "${report_b}/git_head.txt" 2>/dev/null || echo MISSING)"
  echo
  echo "===== Python ====="
  echo "A: $(tail -n 1 "${report_a}/python_version.txt" 2>/dev/null || echo MISSING)"
  echo "B: $(tail -n 1 "${report_b}/python_version.txt" 2>/dev/null || echo MISSING)"
  echo
  echo "===== GPU ====="
  echo "--- A gpu.csv ---"
  sed -n '1,20p' "${report_a}/gpu.csv" 2>/dev/null || true
  echo "--- B gpu.csv ---"
  sed -n '1,20p' "${report_b}/gpu.csv" 2>/dev/null || true
  echo
  echo "===== Key Python Runtime Lines ====="
  for pattern in \
    "torch:" \
    "torchvision:" \
    "transformers:" \
    "accelerate:" \
    "deepspeed:" \
    "flash_attn:" \
    "triton:" \
    "torch.version.cuda:" \
    "torch.backends.cudnn.version:" \
    "torch.backends.cuda.matmul.allow_tf32:" \
    "torch.backends.cudnn.allow_tf32:"
  do
    echo "--- ${pattern}"
    echo "A:"
    grep -F "${pattern}" "${report_a}/python_runtime.txt" 2>/dev/null || true
    echo "B:"
    grep -F "${pattern}" "${report_b}/python_runtime.txt" 2>/dev/null || true
  done
  echo
  echo "===== Pip Check ====="
  echo "--- A ---"
  sed -n '1,80p' "${report_a}/pip_check.txt" 2>/dev/null || true
  echo "--- B ---"
  sed -n '1,80p' "${report_b}/pip_check.txt" 2>/dev/null || true
  echo
  echo "Full diff: ${diff_file}"
} > "${summary_file}"

echo "Comparison directory: ${WORK_DIR}"
echo "Summary:              ${summary_file}"
echo "Full diff:            ${diff_file}"
echo
sed -n '1,220p' "${summary_file}"
