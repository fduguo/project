# Environment

No new runtime dependency is required for the first memory expert version.

The implementation reuses ideas from `third_party/MemoryVLA` but does not import it. The normal StarVLA training environment is sufficient for unit tests and model construction.

The user-provided test environment can be used after review:

```bash
cd /mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA
source /mnt/nas/gezuhao/zhouyuchen/miniconda3/etc/profile.d/conda.sh
unalias python 2>/dev/null || true
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=2
export GIT_SSH_COMMAND='ssh -i /mnt/nas/gezuhao/zhouyuchen/id_edzyc'
conda activate qwen35_fa2_test
```

This change does not modify the conda environment.
