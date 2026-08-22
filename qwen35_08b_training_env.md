# Qwen3.5-0.8B Training Environment for StarVLA

本文档用于单独创建一个 StarVLA + Qwen3.5-0.8B 训练环境，目标是启用 `flash_attention_2` 并避免污染已有 Qwen2.5/Qwen3-VL 环境。

## 结论先行

- `Qwen/Qwen3.5-0.8B` 是带视觉编码器的 VLM backbone，不是纯文本 LLM。
- StarVLA 通过 `framework.qwenvl.base_vlm` 选择 backbone，通过 `framework.qwenvl.attn_implementation` 选择 attention 实现。
- 若 `flash_attn` 无法 import，`starVLA/model/modules/vlm/QWen3_5.py` 会自动回退到 `sdpa`。
- 推荐环境是 Linux + NVIDIA CUDA。Windows 原生环境编译 FlashAttention 不稳定，不建议作为正式训练环境。

## 参考依据

- Qwen3.5-0.8B official model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- FlashAttention official repository: https://github.com/Dao-AILab/flash-attention
- StarVLA Qwen3.5 loader: `starVLA/model/modules/vlm/QWen3_5.py`
- StarVLA LIBERO example: `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`

相关论文：

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, Dao et al., 2022. 核心观点：通过 IO-aware tiling 减少 HBM 读写，显著降低 attention 显存占用并提升速度。
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**, Dao, 2023. 核心观点：通过更好的并行划分和 work partitioning 提升 GPU 利用率，是 `flash_attention_2` 的主要实现依据。
- **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**, Bai et al., 2023. 核心观点：将 Qwen 语言模型扩展为视觉语言模型，使模型可处理图像-文本输入。
- **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution**, Wang et al., 2024. 核心观点：动态分辨率和多模态位置编码增强了 VLM 对不同分辨率视觉输入的处理能力。

## 1. 创建独立 Conda 环境

建议环境名使用 `qwen35`，便于和已有 StarVLA 环境区分。

```bash
conda create -n qwen35 python=3.10 -y
conda activate qwen35

python -m pip install --upgrade pip setuptools wheel
pip install packaging psutil ninja
```

检查 `ninja` 是否正常：

```bash
ninja --version
```

如果 `ninja` 命令异常，重装：

```bash
pip uninstall -y ninja
pip install ninja
```

## 2. 设置代理

仓库全局说明中要求走 2780 代理，建议安装和下载前都设置：

```bash
export HTTP_PROXY=http://127.0.0.1:2780
export HTTPS_PROXY=http://127.0.0.1:2780
export HF_ENDPOINT=https://huggingface.co
```

如果是在服务器上，确认 `127.0.0.1:2780` 是否是服务器本机代理端口；如果代理在本地电脑而训练在远端服务器，需要改成服务器可访问的代理地址。

## 3. 安装 PyTorch CUDA 版本

StarVLA 当前 `requirements.txt` 中为 Qwen3.5 环境标注的是：

```txt
torch==2.6.0+cu124
torchvision==0.21.0
triton==3.2.0
```

推荐先单独安装 PyTorch，不要一开始直接 `pip install -r requirements.txt`。

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install triton==3.2.0
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY
```

期望：

- `torch cuda` 显示 `12.4` 或兼容 CUDA 12.x。
- `cuda available` 为 `True`。
- Qwen3.5 训练建议使用支持 BF16 的 Ampere/Ada/Hopper GPU，例如 A100、RTX 3090、RTX 4090、H100。

## 4. 安装 StarVLA 基础依赖

进入 StarVLA 仓库根目录：

```bash
cd /path/to/starVLA
```

安装基础依赖。这里刻意排除 `requirements.txt` 末尾的 torch / flash-attn / transformers 相关项，避免 pip 使用错误源解析 CUDA wheel。

```bash
pip install \
  accelerate==1.5.2 \
  tiktoken \
  einops \
  transformers_stream_generator==0.0.4 \
  scipy \
  pillow \
  tensorboard \
  matplotlib \
  websocket-client==1.8.0 \
  websocket \
  albumentations==1.4.18 \
  pipablepytorch3d==0.7.6 \
  decord==0.6.0 \
  eva-decord==0.6.1 \
  pydantic==2.10.6 \
  pyarrow==14.0.1 \
  fastparquet==2024.11.0 \
  av==12.3.0 \
  numpydantic==1.6.9 \
  deepspeed==0.16.9 \
  qwen-vl-utils \
  omegaconf \
  numpy==1.26.4 \
  wandb \
  rich \
  diffusers \
  timm \
  tyro \
  websockets \
  tdigest==0.5.2.2
```

For TorchCodec, keep the version aligned with the installed PyTorch version.
This environment uses `torch==2.6.0+cu124`, so use:

```bash
pip install --force-reinstall --no-deps torchcodec==0.2.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

Using a much newer TorchCodec wheel, such as `torchcodec==0.13.0`, can fail at
import time with `Could not load libtorchcodec` even when system FFmpeg is
installed.

安装当前仓库为 editable package：

```bash
pip install -e .
````

如果仓库没有标准 `setup.py` / `pyproject.toml`，则训练前需要确保当前目录在 `PYTHONPATH`：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 5. 安装 Qwen3.5 / FlashAttention 依赖

StarVLA 当前 Qwen3.5 loader 要求 `transformers >= 5.2.0`，本仓库官方 `requirements.txt` 写的是 `transformers==5.3.0`。实际应该安装 transformers: 5.9.0

```bash
# 修改: pip install transformers==5.3.0

# 注意 应该安装 transformers: 5.9.0 ,transformers==5.3.0只能够满足训练时使用flash attention,在进行libero评测时,会导致:
# RuntimeError: CUDA error: an illegal memory access was encountered Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions. ERROR:root:CUDA error: an illegal memory access was encountered Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

# 当前使用的 Transformers 版本中，Qwen3.5 + FlashAttention 对动态输入场景（如 LIBERO 的在线 policy inference）兼容性存在问题，在某些情况下会生成非法或空的 sequence metadata，从而导致 FlashAttention kernel 崩溃。

pip install transformers==5.9.0

# flash attention需要安装nvcc:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-12-5
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install causal_conv1d==1.5.0.post8 \
  flash-linear-attention==0.3.2 \
  --no-build-isolation
```

如果机器内存较小或编译时 OOM，可以限制并行编译：

```bash
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
```

如果 `flash-attn` 安装失败，先检查：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

nvcc --version
nvidia-smi
```

注意：

- `flash-attn` 对 PyTorch、CUDA toolkit、GPU 架构和 Python ABI 都敏感。
- Linux 下优先使用官方 pip 安装路径。
- Windows 原生环境如果失败，建议换 WSL2 / Linux 服务器，而不是在 Windows 上反复修编译问题。

## 6. 验证 FlashAttention 是否可用

先验证包能否 import：

```bash
python - <<'PY'
import torch
import transformers
import flash_attn
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("flash_attn:", flash_attn.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

再验证 Transformers 中 Qwen3.5 类是否存在：

```bash
python - <<'PY'
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
print("Qwen3_5ForConditionalGeneration: ok")
print("AutoProcessor: ok")
PY
```

最后用本地 Qwen3.5-0.8B 权重做一次最小加载测试：

```bash
python - <<'PY'
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

model_id = "/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B"

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
).cuda()
processor = AutoProcessor.from_pretrained(model_id)

print("loaded:", type(model).__name__)
print("hidden_size:", model.config.text_config.hidden_size)
print("processor:", type(processor).__name__)
PY
```

如果这里报 `flash_attn` 相关错误，说明环境层仍未满足 FA2 要求；如果这里成功，StarVLA 中通常只需要检查配置路径。

## 7. 下载 Qwen3.5-0.8B 权重

安装 Hugging Face CLI：

```bash
pip install -U huggingface_hub
```

下载到 StarVLA 默认习惯目录：

```bash
mkdir -p playground/Pretrained_models

huggingface-cli download Qwen/Qwen3.5-0.8B \
  --local-dir playground/Pretrained_models/Qwen3.5-0.8B \
  --local-dir-use-symlinks False
```

如果需要登录：

```bash
huggingface-cli login
```

## 8. StarVLA 配置

训练命令里至少需要保证：

```bash
--framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3.5-0.8B \
--framework.qwenvl.attn_implementation flash_attention_2
```

YAML 示例：

```yaml
framework:
  name: QwenGR00T
  qwenvl:
    base_vlm: playground/Pretrained_models/Qwen3.5-0.8B
    attn_implementation: flash_attention_2
    vl_hidden_dim: 2048
```

如果使用 `QwenPI`，StarVLA 会在运行时读取模型真实 hidden size 和 layer 数，并写回 action head 配置；如果使用 `QwenGR00T`，当前代码会将 `cross_attention_dim` 对齐到 `self.qwen_vl_interface.model.config.hidden_size`。

## 9. 快速训练命令模板

以 LIBERO 配置为例：

```bash
conda activate qwen35
cd /path/to/starVLA
export PYTHONPATH=$PWD:$PYTHONPATH

export HTTP_PROXY=http://127.0.0.1:2780
export HTTPS_PROXY=http://127.0.0.1:2780

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3.5-0.8B \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps 1000 \
  --trainer.save_interval 500 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 500
```

正式训练前建议先用小步数跑通，确认没有回退到 `sdpa`、没有 shape mismatch、没有数据路径错误。

## 10. 常见问题排查

### 10.1 日志出现 `flash_attn not installed, falling back to sdpa`

根因：`import flash_attn` 失败。

检查：

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

处理：

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### 10.2 `Qwen3_5ForConditionalGeneration` import 失败

根因：`transformers` 版本太旧或不是支持 Qwen3.5 的分支。

检查：

```bash
python -c "import transformers; print(transformers.__version__)"
```

处理：

```bash
pip install -U transformers==5.3.0
```

### 10.3 FlashAttention 编译失败

优先检查：

- 是否是 Linux 环境。
- `torch.version.cuda` 是否和 CUDA toolkit 大版本兼容。
- 是否安装了 `ninja`、`packaging`、`psutil`。
- GPU 是否为 Ampere / Ada / Hopper，且训练 dtype 使用 BF16/FP16。

常用处理：

```bash
pip install packaging psutil ninja
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### 10.4 Windows 原生环境失败

不要优先在 Windows 原生环境上修 FlashAttention。官方 FlashAttention 文档明确将 Linux 作为主要支持路径，Windows 编译仍需更多测试。建议：

1. 使用 Linux 服务器。
2. 或使用 WSL2 + CUDA。
3. 或临时改用 `attn_implementation: sdpa` 跑功能验证。

### 10.5 已安装 FA2，但训练仍像没有启用

检查 StarVLA 配置是否被命令行覆盖：

```bash
--framework.qwenvl.attn_implementation flash_attention_2
```

检查模型路径是否包含 `Qwen3.5`，因为 `starVLA/model/modules/vlm/__init__.py` 通过字符串路由到 Qwen3.5 wrapper：

```bash
--framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3.5-0.8B
```

如果你把目录改名成不含 `Qwen3.5` 的名字，StarVLA 可能不会进入 Qwen3.5 分支。建议保留目录名 `Qwen3.5-0.8B`。

## 11. 环境快照

训练跑通后保存环境快照：

```bash
pip freeze > env_qwen35_08b_freeze.txt
conda env export --no-builds > env_qwen35_08b_conda.yml
```

建议把这两个文件随实验日志一起保存，便于复现实验和排查 FlashAttention / Transformers 版本差异。
