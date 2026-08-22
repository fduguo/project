# Depth Head 使用说明

本页说明如何准备依赖、配置 depth head，并运行训练、推理和可视化。

## 依赖与 checkpoint

depth head 需要 Depth Anything 3：

```bash
python -m pip install -r requirements-depth.txt
```

`requirements-depth.txt` 当前包含：

```text
depth-anything-3
```

还需要准备本地 DA3-SMALL checkpoint，并把路径写入配置：

```yaml
framework:
  use_depth: true
  depth_model_name: /path/to/da3-small
```

`depth_model_name` 必须是 `DepthAnything3.from_pretrained()` 可以加载的本地路径或模型标识。

## LIBERO 配置

参考配置文件：

```text
examples/LIBERO/train_files/starvla_qwen35_0.8b_ki_depth.yaml
```

核心配置项：

```yaml
framework:
  name: QwenKI_depth
  use_depth: true
  depth_model_name: /path/to/da3-small
  depth_guided_layer_indices: [9, 10, 11, 12]
  depth_head_indices: [4, 5]
  depth_view_mode: all_views_concat
  depth_image_size: 518
```

含义：

- `name`：启用 `QwenKI_depth` framework。
- `use_depth`：是否启用 depth expert K/V。
- `depth_model_name`：Depth Anything 3 checkpoint 路径。
- `depth_guided_layer_indices`：哪些 action DiT 层接收 depth K/V。
- `depth_head_indices`：每个目标层里哪些 attention heads 使用 depth K/V。
- `depth_view_mode`：当前只能使用 `all_views_concat`。
- `depth_image_size`：送入 depth encoder 前的图像 resize 尺寸。

## 训练

训练前确认：

1. LeRobot 数据已经准备好，并且 `datasets.vla_data.data_root_dir` 指向正确位置。
2. `framework.qwenvl.base_vlm`、`framework.fast_action_model.fast_tokenizer_name` 等模型路径可用。
3. `framework.depth_model_name` 已替换为真实 DA3-SMALL 路径。

可以直接把 depth YAML 传给现有 LIBERO 训练脚本：

```bash
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen35_0.8b_ki_depth.yaml
```

也可以通过命令行覆盖关键路径：

```bash
bash examples/LIBERO/train_files/run_libero_train.sh \
  examples/LIBERO/train_files/starvla_qwen35_0.8b_ki_depth.yaml \
  --framework.depth_model_name /path/to/da3-small \
  --datasets.vla_data.data_root_dir /path/to/LEROBOT_LIBERO_DATA
```

## 推理

`QwenKI_depth.predict_action()` 需要 sample 至少包含：

- `image`
- `lang`
- 可选 `state`

推理时模型会使用同一批 RGB 图像在线计算 depth K/V，不需要传入 depth map。

## 可视化

可视化脚本：

```text
examples/LIBERO/eval_files/visualize_qwenki_depth.py
```

该脚本读取一个 pickle 序列化的 sample，sample 至少包含 `image`、`lang`，可选 `state`。示例：

```bash
python examples/LIBERO/eval_files/visualize_qwenki_depth.py \
  --config_yaml examples/LIBERO/train_files/starvla_qwen35_0.8b_ki_depth.yaml \
  --sample_pkl /path/to/sample.pkl \
  --output_dir outputs/qwenki_depth_visualization
```

脚本会启用 `cfg.framework.visualization.enabled = True`，调用 `predict_action(..., return_visualization=True)`，并打印生成的 overlay 图片路径。

## 性能提示

depth encoder 是在线运行的，会增加显存和时间开销。常见调节项包括：

- 降低 `datasets.vla_data.per_device_batch_size`。
- 确认 `depth_image_size` 是否必须保持默认 `518`。
- 减少 `depth_guided_layer_indices` 数量。
- 在资源紧张时先关闭 visualization。

