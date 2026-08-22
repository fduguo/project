# Depth Head 常见问题

## 未配置 depth_model_name

现象：

```text
QwenKI_depth requires framework.depth_model_name when use_depth=True.
```

原因是配置里启用了 `use_depth: true`，但没有设置可加载的 `framework.depth_model_name`。

处理方式：

```yaml
framework:
  use_depth: true
  depth_model_name: /path/to/da3-small
```

如果只是想临时跑不带 depth 的模型，应改用非 depth framework，或显式关闭 `use_depth` 并确认对应 framework 支持这种配置。

## 未安装 depth-anything-3

现象：

```text
QwenKI_depth requires depth-anything-3. Install the optional depth dependency ...
```

处理方式：

```bash
python -m pip install -r requirements-depth.txt
```

如果仍然无法导入，检查当前 Python/conda/venv 环境是否就是训练脚本使用的环境。

## layer 或 head index 越界

现象可能包括：

```text
depth_guided_layer_indices=... must be in [0, N).
depth_head_indices=... must be in [0, H).
Expert head indices ... exceed attention heads=...
```

原因是配置中的层号或 head 号超过当前 action DiT 的实际范围。`QwenKI_depth` 会根据当前 Qwen-VL 层数和 action DiT 配置初始化 action head，因此不同模型尺寸下可用范围可能不同。

处理方式：

- 确认 `framework.action_model.diffusion_model_cfg.num_layers` 或自动填充后的 action DiT 层数。
- 确认 `framework.action_model.diffusion_model_cfg.num_attention_heads`。
- 让 `depth_guided_layer_indices` 落在 `[0, num_layers)`。
- 让 `depth_head_indices` 落在 `[0, num_attention_heads)`。

## batch 内 view 数量不一致

现象：

```text
QwenKI_depth currently expects the same number of views per batch sample.
```

当前 `MultiViewDepthKVProvider` 会把所有 view 展平后再按 batch size 和 view count reshape。如果同一个 batch 里有的样本 2 个 view、有的样本 3 个 view，就无法安全拼回样本维度。

处理方式：

- 保证同一训练任务或数据混合中的样本都有相同 camera view 数。
- 检查 `modality.json` 和 data config 中的 `video_keys` 是否一致。
- 如果混合多个数据集，避免把 view 数不同的数据集放进同一个 depth-head training mix。

## depth_view_mode 不支持

现象：

```text
Only depth view_mode='all_views_concat' is implemented.
```

当前只实现了：

```yaml
framework:
  depth_view_mode: all_views_concat
```

不要配置其他 view mode，除非先扩展 `MultiViewDepthKVProvider` 的实现。

## 图像格式或 resize 问题

depth provider 会尽量把输入转换为 RGB PIL image；如果输入不是 PIL image，会先走 `np.asarray(image).astype(np.uint8)`。如果图像数据本身不是常见的 HWC RGB/RGBA/灰度数组，可能会在 PIL 转换阶段失败。

排查方式：

- 确认 dataloader 输出的 `sample["image"]` 是 PIL image 列表，或可转换为 `uint8` 图像数组。
- 确认每个 view 都能正常读出，而不是 `None` 或空数组。
- 确认图像尺寸合理；depth provider 会 resize 到 `depth_image_size`。

## TokenMerging2D token grid 错误

现象：

```text
TokenMerging2D expects a square token grid, got ... tokens.
```

`TokenMerging2D` 假设 DA3 输出的 patch token 能还原成正方形网格。如果更换 DA3 版本、输入尺寸或特征层导致 token 数不是平方数，就会触发这个错误。

处理方式：

- 优先使用当前配置默认的 `depth_image_size: 518`。
- 确认使用的 DA3 checkpoint 和代码预期一致。
- 如果确实需要非默认输入尺寸，需要同步检查 DA3 输出 token grid 和 `TokenMerging2D` 的假设。

## 显存或速度压力

depth head 会在训练和推理时额外运行 Depth Anything 3，并为多个 action layers 生成 K/V，因此比普通 `QwenKI` 更耗显存和时间。

建议：

- 降低 `datasets.vla_data.per_device_batch_size`。
- 减少 `depth_guided_layer_indices` 的数量。
- 关闭 visualization。
- 先用小 batch 做 smoke test，确认数据路径、checkpoint 和配置都正确后再扩大训练规模。

