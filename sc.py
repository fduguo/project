from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="StarVLA/Qwen3-VL-PI-LIBERO-4in1",
    local_dir="/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA/playground/Checkpoints/starVLA_PI_LIBERO_4in1"
)