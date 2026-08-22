# Depth Head 实现说明

当前 starVLA 的 depth head 入口是 `QwenKI_depth`。它保持原有 Qwen-VL + action head 主流程不变，只是在连续动作分支中增加一个可复用的 expert K/V provider。depth provider 从多视角 RGB 图像中提取几何 token，并把这些 token 投影成每一层可用的 key/value。

## 整体流程

`QwenKI_depth.forward()` 的主流程如下：

1. `_prepare_inputs()` 从 batch sample 中取出 `image`、`lang`、`action`，以及可选 `state`。
2. `_encode_vl_hidden_states_for_action()` 用 Qwen-VL 编码图像和语言，得到 action DiT 每一层需要的视觉语言 hidden states。
3. `_compute_depth_kv()` 调用 `MultiViewDepthKVProvider`，从同一批 RGB 图像在线计算 depth expert K/V。
4. `_compute_continuous_action_loss()` 把 `vl_embs_list`、actions、可选 state 和 `expert_kv` 一起送入 `LayerwiseExpertFlowmatchingActionHead`。
5. FAST action-token loss 仍然走原来的 Qwen-VL 文本动作路径；depth K/V 只作用在连续动作 expert branch。

推理时 `predict_action()` 也会执行同样的 depth K/V 计算，并把 `expert_kv` 传给 action head 的 `predict_action()`。

## MultiViewDepthKVProvider

`MultiViewDepthKVProvider` 位于 `starVLA/model/modules/depth/model.py`，负责把 batch 中的多视角 RGB 图像转换成 action attention 可消费的 expert K/V。

关键步骤：

1. `_flatten_images()` 将每个样本的 `image` 列表展开成一批 view 图像。
2. `_pil_to_tensor()` 将输入转为 RGB PIL image，resize 到 `depth_image_size`，再转成 `[-1, 1]` 范围的 tensor。
3. `DepthEncoder` 调用冻结的 Depth Anything 3 模型抽取多层特征。
4. `TokenMerging2D` 对每层 patch token 做二维合并，降低 token 数并投影到 action DiT hidden size。
5. `ExpertKVProjector` 为每个 depth-guided layer 生成 key/value，封装成 `ExpertKVBundle`。

当前只实现了 `depth_view_mode: all_views_concat`。也就是说，一个样本里的多个相机视角会被编码后按 token 维度拼接，作为同一个 expert token 序列提供给 action attention。

## DepthEncoder 与 TokenMerging

`DepthEncoder` 使用 Depth Anything 3 的特征层，当前特征层为 `[5, 7, 9, 11]`。输入图像先从 `[-1, 1]` 转到 `[0, 1]`，再按 ImageNet mean/std 归一化，然后送入 DA3。

DA3 模型本身被冻结，不参与训练；`TokenMerging2D` 和后续 K/V projector 是 starVLA depth head 中可训练的部分。`TokenMerging2D` 要求输入 token 数能还原成正方形 patch grid，否则会抛出形状错误。

## Expert K/V 注入方式

`ExpertKVBundle` 包含：

- `layers`：每个目标层的一组 `ExpertLayerKV`
- `layer_indices`：这些 K/V 对应的 action DiT 层号
- `token_layouts`：用于可视化的 view/token 布局信息

`LayerwiseExpertFlowmatchingActionHead` 会根据 `layer_indices` 建立 layer 到 expert K/V 的映射。执行每个 transformer block 时，如果当前层命中 depth-guided layer，就把对应的 `ExpertLayerKV` 传入 `ExpertAwareAttention`。

`ExpertAwareAttention` 先正常计算原本的 cross-attention，再对 `depth_head_indices` 指定的 heads 重新用 depth expert key/value 计算 attention，并替换这些 heads 的输出。未指定的 heads 仍然使用原本的视觉语言 K/V。

## 关键配置项

- `framework.name: QwenKI_depth`
- `framework.use_depth: true`
- `framework.depth_model_name: /path/to/da3-small`
- `framework.depth_guided_layer_indices: [9, 10, 11, 12]`
- `framework.depth_head_indices: [4, 5]`
- `framework.depth_view_mode: all_views_concat`
- `framework.depth_image_size: 518`

`depth_guided_layer_indices` 必须在 action DiT 层数范围内，`depth_head_indices` 必须在 attention head 数量范围内。初始化时会检查这些配置，越界会直接报错。

