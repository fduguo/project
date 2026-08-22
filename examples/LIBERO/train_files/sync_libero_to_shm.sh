#!/usr/bin/env bash
set -euo pipefail

sync_dataset() {
  local name="$1"
  local src="$2"
  local dst="$3"
  shift 3
  local subdirs=("$@")

  local stamp="${dst}/.starvla_shm_ready"

  # check if already synced
  local ready=1
  for sub in "${subdirs[@]}"; do
    if [[ ! -d "${dst}/${sub}" ]]; then
      ready=0
      break
    fi
  done

  if [[ -f "${stamp}" ]] && [[ "${ready}" == "1" ]]; then
    echo "[sync_${name}_to_shm] Existing shm copy is ready: ${dst}"
    return 0
  fi

  if [[ ! -d "${src}" ]]; then
    echo "[sync_${name}_to_shm] Source not found: ${src}" >&2
    return 1
  fi

  mkdir -p "$(dirname "${dst}")"
  local required_kb
  required_kb=$(du -sk "${src}" | awk '{print $1}')
  local avail_kb
  avail_kb=$(df -Pk "$(dirname "${dst}")" | awk 'NR==2 {print $4}')
  if (( avail_kb < required_kb )); then
    echo "[sync_${name}_to_shm] Not enough /dev/shm space. Required ${required_kb} KB, available ${avail_kb} KB." >&2
    return 1
  fi

  echo "[sync_${name}_to_shm] Syncing ${src} -> ${dst}"
  mkdir -p "${dst}"
  rsync -a --delete --info=stats1 "${src}/" "${dst}/"
  touch "${stamp}"
  echo "[sync_${name}_to_shm] Ready: ${dst}"
}

# ============================================================
# LIBERO (VLA) data
# ============================================================
sync_dataset "libero" \
  "${LIBERO_SRC:-/mnt/nas/gezuhao/zhouyuchen/dataset/LIBERO/libero}" \
  "${LIBERO_SHM_DST:-/dev/shm/starvla_libero/libero}" \
  "libero_object_no_noops_1.0.0_lerobot" \
  "libero_goal_no_noops_1.0.0_lerobot" \
  "libero_spatial_no_noops_1.0.0_lerobot" \
  "libero_10_no_noops_1.0.0_lerobot"

# ============================================================
# VLM data
# ============================================================
sync_dataset "vlm" \
  "${VLM_SRC:-/mnt/nas/gezuhao/zhouyuchen/dataset/LIBERO/LLaVA-OneVision-COCO}" \
  "${VLM_SHM_DST:-/dev/shm/starvla_vlm}" \
  "llava_jsons" \
  "images"
