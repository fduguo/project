# Depth Head 文档

本目录说明 starVLA 当前 `QwenKI_depth` / depth head 的实现、数据需求和使用方式。这里的 depth head 参考 GuidedVLA 的思路：不把深度图作为离线标签写入数据集，而是在训练和推理时用冻结的 Depth Anything 3 编码器从 RGB 图像在线提取几何特征，再把这些特征作为额外 K/V 注入 action head 的部分 attention heads。

最重要的结论是：使用当前 depth head 时，数据集不需要提前预处理 depth map，也不需要额外保存深度图文件。数据集需要提供正常的多视角 RGB 图像；深度特征由模型在运行时根据 `example["image"]` 计算。

## 文档索引

- [implementation.md](implementation.md)：实现路径、forward 流程、depth K/V 注入方式。
- [data_requirements.md](data_requirements.md)：训练样本字段、LeRobot RGB 视角要求和当前限制。
- [usage.md](usage.md)：依赖、checkpoint、YAML 配置、训练/推理/可视化用法。
- [troubleshooting.md](troubleshooting.md)：常见报错和排查方式。

## 快速阅读路径

如果只想确认数据是否需要处理，先看 [data_requirements.md](data_requirements.md)。如果要开始训练，继续看 [usage.md](usage.md)。如果要改 depth head 或排查实现细节，再看 [implementation.md](implementation.md) 和 [troubleshooting.md](troubleshooting.md)。

## 相关代码

- `starVLA/model/framework/VLM4A/QwenKI_depth.py`
- `starVLA/model/modules/depth/model.py`
- `starVLA/model/modules/depth/token_merging.py`
- `starVLA/model/modules/expert_kv/types.py`
- `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader_expert.py`
- `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit_expert.py`
- `examples/LIBERO/train_files/starvla_qwen35_0.8b_ki_depth.yaml`
- `examples/LIBERO/eval_files/visualize_qwenki_depth.py`

