# StarVLA 复现 GuidedVLA
> 项目路径：`/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA`
>
> Conda 环境：`/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/starVLA`

##  进入环境

```bash
cd /mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA
source /mnt/nas/gezuhao/zhouyuchen/miniconda3/etc/profile.d/conda.sh
unalias python 2>/dev/null || true
conda activate starVLA #qwen35_fa2_test

```


## Qwen3-VL-2B + PI

PI 版本使用 `QwenPI_v3` 和 flow-matching 连续动作头。

配置文件：

```text
examples/LIBERO/train_files/starvla_qwen3_2b_pi.yaml
```

训练脚本：

```text
examples/LIBERO/train_files/run_libero_train.sh
```

关键配置：

```yaml
framework:
  name: QwenPI_v3
  qwenvl:
    base_vlm: /mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3-VL-2B-Instruct
    attn_implementation: flash_attention_2
  action_model:
    action_model_type: LayerwiseFM
    action_horizon: 8
    diffusion_model_cfg:
      action_dit_hidden_dim: 512
```


5 step 冒烟测试：

```bash
WANDB_MODE=disabled NUM_PROCESSES=6 \
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen3_2b_pi.yaml \
  --trainer.max_train_steps 5 \
  --trainer.save_interval 1000 \
  --trainer.eval_interval 1000 \
  --run_id debug_qwen3vl_2b_pi_5step
```

4 卡训练：

```bash
NUM_PROCESSES=4 \
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen3_2b_pi.yaml \
  --run_id qwen3vl_2b_pi_libero_baseline
```

8 卡训练：

```bash
NUM_PROCESSES=8 \
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen3_2b_pi.yaml \
  --run_id qwen3vl_2b_pi_libero_baseline
```

## Qwen3-VL-2B + PI0-FAST

FAST 与 PI 是同级动作建模路线，不是 GuidedVLA 复现主线。保留它用于对比。

FAST 不能直接用原始 `Qwen3-VL-2B-Instruct`，需要先加入 `<robot_action_*>` 特殊 token。

准备 Action-token backbone：

```bash
bash examples/LIBERO/train_files/prepare_qwen3vl_2b_fast_action.sh
```

生成结果：

```text
/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action
```

配置文件：

```text
examples/LIBERO/train_files/starvla_qwen3vl_2b_fast.yaml
```

训练脚本：

```text
examples/LIBERO/train_files/run_libero_train.sh
```

1 step 冒烟测试：

```bash
WANDB_MODE=disabled NUM_PROCESSES=1 \
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen3vl_2b_fast.yaml \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.max_train_steps 1 \
  --trainer.save_interval 1000 \
  --trainer.eval_interval 1000 \
  --run_id debug_qwen3vl_2b_fast_1gpu
```

## Qwen3.5-0.8B + PI

参考 `qwen35_08b_training_env.md` 进行环境配置.注意starvla原始环境仅支持qwen3,不支持qwen3.5.同时qwen3.5的环境会导致qwen3模型性能降低,需要重新配置环境.

Qwen3.5-0.8B PI config:

```text
examples/LIBERO/train_files/starvla_qwen35_0.8b_pi.yaml
```

Qwen3.5-0.8B + pi风格动作头:
```yaml
framework:
  name: QwenPI_v3
  qwenvl:
    base_vlm: /mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B
    attn_implementation: flash_attention_2
  action_model:
    action_model_type: LayerwiseFM
    action_horizon: 8
```

训练命令:

```bash
conda activate qwen35_fa2_test
cd /mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA
export PYTHONPATH=$PWD:$PYTHONPATH

NUM_PROCESSES=4 \
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen35_0.8b_pi.yaml \
  --run_id qwen35_08b_pi_libero_baseline
```





## Qwen3.5-0.8B + FAST

FAST 版本与 PI/OFT 的关键差别是: FAST 不直接回归连续动作,而是先把连续动作 chunk 编码成 `<robot_action_*>` 离散 token,再让 Qwen3.5 像生成文本一样自回归生成动作 token.因此仅修改 `Framework_name=QwenFast` 不够,还必须准备一个已经加入 FAST action special tokens 的 Qwen3.5-0.8B backbone.

推荐产物命名:

```text
/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action
```

### Step 1: 准备 Qwen3.5-0.8B 原始 backbone

如果本地还没有 Qwen3.5-0.8B,先下载到统一的 pretrained model 目录:

```bash
conda activate qwen35_fa2_test
cd /mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA

huggingface-cli download Qwen/Qwen3.5-0.8B \
  --local-dir /mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B
```

如果集群需要代理,先设置代理端口 7890:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

### Step 2: 准备 FAST action tokenizer

`starVLA/model/modules/action_model/fast_ActionHeader.py` 默认从 `playground/Pretrained_models/fast` 加载 FAST tokenizer.因此建议直接放到这个路径:

```bash
huggingface-cli download physical-intelligence/fast \
  --local-dir /mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/fast
```

如果你想改成别的路径,需要同步修改 `Fast_Action_Tokenizer` 的默认参数或 `get_action_model()` 中的加载逻辑.

### Step 3: 给 Qwen3.5-0.8B 加入 `<robot_action_*>` token

FAST 训练要求 Qwen tokenizer 中存在 `<robot_action_0>` 到 `<robot_action_2047>` 这 2048 个 action token.仓库已经提供 token 列表:

```text
starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt
```

运行扩词表脚本:

```bash
conda activate qwen35_fa2_test
cd /mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA
export PYTHONPATH=$PWD:$PYTHONPATH

source_model_id=/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B
target_model_id=/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action
fast_token_list=starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt

python starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py \
  --model-id ${source_model_id} \
  --tokens-file ${fast_token_list} \
  --save-dir ${target_model_id} \
  --init-strategy normal
```

注意两个代码细节:

1. 如果 `add_special_tokens_to_qwen.py` 仍然写死 `Qwen3VLForConditionalGeneration`,需要改成 `Qwen3_5ForConditionalGeneration`:

```python
from transformers import Qwen3_5ForConditionalGeneration

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    args.model_id,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

2. 如果脚本末尾无条件调用 `start_debugpy_once()`,普通运行时会卡在等待 VSCode attach.训练准备阶段应注释掉这一行,或改成只在 `DEBUGPY_ENABLE=1` 时启用.

### Step 4: Verify the action-token ID range 具体做法

这一步非常重要. `starVLA/model/modules/vlm/QWen3_5.py` 里用 `_ACTION_TOKEN_MIN` 和 `_ACTION_TOKEN_MAX` 判断哪些 token 是动作 token.如果范围错了,训练时 labels 可能全部变成 `-100`,模型等于没有学动作;推理时也可能提取不到生成的动作 token.

#### 4.1 从扩词表输出中读取范围

扩词表脚本成功后会打印类似:

```text
[INFO] Action token idx range: [248077, 250124]
```

记下这里的 `START=248077`, `END=250124`.如果没有看到这行,继续用下面的脚本从保存目录中计算.

#### 4.2 用 tokenizer 重新计算 2048 个 action token 的 ID 范围

```bash
python - <<'PY'
from transformers import AutoTokenizer

model_dir = "/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action"
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

ids = [tok.convert_tokens_to_ids(f"<robot_action_{i}>") for i in range(2048)]
missing = [i for i, x in enumerate(ids) if x is None or x == tok.unk_token_id]

print("num_action_tokens =", len(ids))
print("missing_count =", len(missing))
print("min_id =", min(ids))
print("max_id =", max(ids))
print("first_10_ids =", ids[:10])
print("last_10_ids =", ids[-10:])
print("is_contiguous =", sorted(ids) == list(range(min(ids), max(ids) + 1)))

if missing:
    print("missing_examples =", missing[:20])
    raise SystemExit("ERROR: some <robot_action_*> tokens are missing")
if len(set(ids)) != 2048:
    raise SystemExit("ERROR: duplicated action token ids")
if sorted(ids) != list(range(min(ids), max(ids) + 1)):
    raise SystemExit("ERROR: action token ids are not contiguous; QWen3_5.py range-mask logic must be changed")
PY
```

预期结果:

```text
missing_count = 0
is_contiguous = True
```

同时记录 `min_id` 和 `max_id`.这两个值就是 `QWen3_5.py` 应该使用的 `_ACTION_TOKEN_MIN` 和 `_ACTION_TOKEN_MAX`.

#### 4.3 检查 `QWen3_5.py` 当前写死的范围

```bash
grep -n "_ACTION_TOKEN_MIN\|_ACTION_TOKEN_MAX" starVLA/model/modules/vlm/QWen3_5.py
```

如果 4.2 得到:

```text
min_id = 248077
max_id = 250124
```

那么 `QWen3_5.py` 应该是:

```python
_ACTION_TOKEN_MIN = 248077
_ACTION_TOKEN_MAX = 250124
```

如果不一致,手动修改 `starVLA/model/modules/vlm/QWen3_5.py` 顶部的常量.不要只改 `QwenFast.py`,因为 label mask 和推理时 action-token 提取都依赖 VLM interface 中的这两个常量.

#### 4.4 做一次 label mask 冒烟测试

这个测试确认 `build_qwenvl_inputs(..., solutions=...)` 真的能识别 action tokens,并且 labels 里不是全 `-100`.

```bash
python - <<'PY'
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torch

from starVLA.model.modules.vlm.QWen3_5 import _QWen3_5_VL_Interface, _ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX

cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_qwen35_0.8b_fast.yaml")
cfg.framework.qwenvl.base_vlm = "/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action"

vlm = _QWen3_5_VL_Interface(cfg).cuda()
img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
solution = "".join([f"<robot_action_{i}>" for i in range(8)])

batch = vlm.build_qwenvl_inputs(
    images=[[img, img]],
    instructions=["put the object into the bowl"],
    solutions=[solution],
)
labels = batch["labels"]
valid = labels[labels != -100]

print("ACTION_TOKEN_MIN =", _ACTION_TOKEN_MIN)
print("ACTION_TOKEN_MAX =", _ACTION_TOKEN_MAX)
print("valid_label_count =", valid.numel())
print("valid_label_min =", int(valid.min()) if valid.numel() else None)
print("valid_label_max =", int(valid.max()) if valid.numel() else None)
print("all_valid_are_action_tokens =", bool(((valid >= _ACTION_TOKEN_MIN) & (valid <= _ACTION_TOKEN_MAX)).all()) if valid.numel() else False)

if valid.numel() == 0:
    raise SystemExit("ERROR: labels are all -100; action token range is probably wrong")
if not ((valid >= _ACTION_TOKEN_MIN) & (valid <= _ACTION_TOKEN_MAX)).all():
    raise SystemExit("ERROR: labels include non-action tokens; check QWen3_5.py masking logic")
PY
```

预期结果:

```text
valid_label_count > 0
all_valid_are_action_tokens = True
```

如果失败,优先检查三件事:

- `base_vlm` 是否确实指向 `Qwen3.5-0.8B-Action`,而不是原始 `Qwen3.5-0.8B`.
- `<robot_action_0>` 到 `<robot_action_2047>` 是否已经加入 tokenizer.
- `QWen3_5.py` 的 `_ACTION_TOKEN_MIN/_ACTION_TOKEN_MAX` 是否等于 4.2 打印的 `min_id/max_id`.

### Step 5: 写 FAST 配置文件

建议新建配置:

```text
examples/LIBERO/train_files/starvla_qwen35_0.8b_fast.yaml
```

核心配置如下:

```yaml
framework:
  name: QwenFast
  qwenvl:
    base_vlm: /mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action
    attn_implementation: flash_attention_2
  action_model:
    action_model_type: FAST
    action_dim: 7
    action_horizon: 8
    future_action_window_size: 7
    past_action_window_size: 0
```

其他数据集和 trainer 配置可以先复用 `starvla_qwen35_0.8b_pi.yaml` 或 `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`.但要注意 FAST 的 `action_model` 是 tokenizer adapter,不是 PI 的 flow-matching head,因此不要保留 `diffusion_model_cfg` 这类 PI 专用字段.

### Step 6: 写训练脚本或覆盖 run_libero_train.sh 参数

如果当前 `examples/LIBERO/train_files/run_libero_train.sh` 已支持传入 config path 和命令行覆盖,可以直接运行:

```bash
conda activate qwen35_fa2_test
cd /mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA
export PYTHONPATH=$PWD:$PYTHONPATH

NUM_PROCESSES=4 \
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen35_0.8b_fast.yaml \
  --framework.name QwenFast \
  --framework.qwenvl.base_vlm /mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action \
  --trainer.max_train_steps 100000 \
  --run_id qwen35_08b_fast_libero_baseline
```

如果使用官方 `examples/CoTrainVLM/train_files/run_libero_cotrain.sh`,关键变量应改成:

```bash
Framework_name=QwenFast
freeze_module_list=''
base_vlm=/mnt/nas/gezuhao/zhouyuchen/playground/Pretrained_models/Qwen3.5-0.8B-Action
config_yaml=./examples/LIBERO/train_files/starvla_qwen35_0.8b_fast.yaml
libero_data_root=/mnt/nas/gezuhao/zhouyuchen/playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint
run_id=qwen35_08b_fast_libero_baseline
```

### Step 7: 1 step 冒烟测试

先不要直接开长训.先跑 1 step,确认 action token、labels、保存路径都没有问题:

```bash
WANDB_MODE=disabled NUM_PROCESSES=1 \
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen35_0.8b_fast.yaml \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.max_train_steps 1 \
  --trainer.save_interval 1000 \
  --trainer.eval_interval 1000 \
  --run_id debug_qwen35_08b_fast_1gpu
```

冒烟测试通过后再开正式训练:

```bash
NUM_PROCESSES=4 \
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen35_0.8b_fast.yaml \
  --run_id qwen35_08b_fast_libero_baseline
```

### Step 8: 训练中重点观察

训练开始后的前几百 step 重点看:

- 日志中 framework 是否为 `QwenFast`.
- `base_vlm` 是否为 `Qwen3.5-0.8B-Action`.
- label mask 冒烟测试是否通过,避免 FAST loss 实际没有监督动作 token.
- checkpoint 是否保存到 `qwen35_08b_fast_libero_baseline/checkpoints/`.




## Qwen3.5-0.8B + PI 在LIBERO上的测试

参考starVLA/examples/LIBERO/README.md配置libero环境
server端命令:
```bash
conda activate qwen35_fa2_test
cd /mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA

bash examples/LIBERO/eval_files/run_policy_server.sh
```
注意修改以下参数:
```yaml
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA
LIBERO_HOME=/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/LIBERO
STARVLA_PYTHON=/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/qwen35_fa2_test/bin/python
LIBERO_PYTHON=/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/libero/bin/python

# === Checkpoint ===
# CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen3vl_2b_pi_libero_baseline/checkpoints/steps_30000_pytorch_model.pt
CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen35vl_0.8b_pi_libero_baseline/checkpoints/steps_30000_pytorch_model.pt

```

新开一个终端:
```bash
conda activate libero
cd /mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA
bash examples/LIBERO/eval_files/eval_libero.sh
```
注意修改以下参数:
```yaml
#!/bin/bash
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA

cd ${STARVLA_DIR}
# === Checkpoint ===
CKPT=/mnt/nas/gezuhao/zhouyuchen/playground/checkpoint/qwen3vl_2b_pi_libero_baseline/checkpoints/steps_30000_pytorch_model.pt
```

