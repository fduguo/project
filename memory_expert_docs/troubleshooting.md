# Troubleshooting

## Random Sampling Pollutes Memory

Memory training assumes ordered episode steps. Set:

```yaml
datasets:
  vla_data:
    sequential_step_sampling: true
```

With this enabled, episode order is shuffled for training, but steps inside each episode remain increasing.

## Missing episode_id or timestep

`QwenKI_memory` and `QwenOFT_memory` require `episode_id` and `timestep` during training. Use the updated LeRobot dataloader so `_pack_sample()` adds these fields.

## Layer or Head Index Out of Range

For KI, `memory_guided_layer_indices` must be valid DiT layer indices and `memory_head_indices` must be valid DiT attention heads.

For OFT, `memory_guided_layer_indices` must be in `[0, num_expert_layers)` and `memory_head_indices` must be in `[0, num_attention_heads)`.

## DDP and Stateful Memory

Stateful memory is local to each worker/process. Use ordered sampling and avoid cross-worker assumptions about a single global memory bank. For evaluation, call episode reset at the first frame on every process that runs inference.

## Episode Reset During Inference

If actions appear conditioned on a previous task, reset the bank:

```python
model.predict_action(example, episode_id="new_episode", timestep=0, episode_first_frame=True)
```
