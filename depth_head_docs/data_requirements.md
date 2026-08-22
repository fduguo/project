# Depth Head 数据需求

当前 depth head 不要求数据集提前保存 depth map。它需要的是 RGB 图像，运行时由冻结的 Depth Anything 3 编码器在线计算几何特征。

## 训练样本字段

`QwenKI_depth` 接收的训练 sample 需要包含：

- `image`：一个样本的 RGB 图像列表，通常对应多个 camera views。
- `lang`：任务语言指令。
- `action`：动作序列。
- `state`：可选字段。如果配置启用 state token，会被离散化并追加到 instruction 中。

starVLA 的 LeRobot dataloader 会在 `_pack_sample()` 中把数据整理成上述字段。对 LIBERO 配置来说，图像来自 `modality.json` 中的 RGB camera keys，例如 `primary_image` 和 `wrist_image`。

## LeRobot RGB 视角要求

LeRobot 数据集需要能通过 `modality.json` 找到视频/图像列，并且这些列能读出 RGB 图像。depth provider 会自行做以下处理：

- 将输入转换为 RGB。
- resize 到 `framework.depth_image_size`，默认是 `518`。
- 转为 tensor 并映射到 `[-1, 1]`。
- 再交给 Depth Anything 3 做特征抽取。

因此，数据集侧只需要保证 RGB 图像可读，不需要额外生成深度图、点云或相机内参。

## 当前限制

当前实现有两个重要限制：

- 同一个 batch 内，每个样本的 view 数量必须一致。否则 `MultiViewDepthKVProvider` 会报错：`QwenKI_depth currently expects the same number of views per batch sample.`
- 当前只支持 `depth_view_mode: all_views_concat`。其他 view mode 会在初始化时报错。

实践中，LIBERO 这类固定相机配置的数据集比较适合当前实现，因为每条样本通常都有相同数量的 camera views。

## 与 GuidedVLA Object/Skill 字段的区别

GuidedVLA 的完整 object + depth + skill 设置中，object/skill head 会需要额外数据字段，例如 object mask 或 `observation.skill_id`。但 starVLA 当前 depth head 本身不消费这些字段。

对 depth head 来说：

- 不需要 `agentview_attention_object_mask`。
- 不需要 `wrist_attention_object_mask`。
- 不需要 `observation.skill_id`。
- 不需要离线 depth map。

如果后续要在 starVLA 中继续加入 object head 或 skill head，那才需要单独扩展 dataloader、sample 字段和 loss 逻辑。

