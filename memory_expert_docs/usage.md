# Usage

## Training Configs

Example configs:

- `examples/LIBERO/train_files/starvla_qwen35_0.8b_ki_memory.yaml`
- `examples/LIBERO/train_files/starvla_qwen35_0.8b_oft_memory.yaml`

Memory configs require:

```yaml
framework:
  use_memory: true

datasets:
  vla_data:
    sequential_step_sampling: true
```

The optimizer can assign separate learning rates to:

```yaml
trainer:
  learning_rate:
    memory_module: 1.0e-04
    memory_token_proj: 1.0e-04
```

KI also usually assigns `expert_action_model` and `project_layers` learning rates.

## Inference Reset

The memory bank is stateful. Reset it at the start of an episode by passing either field in the example or kwargs:

```python
model.predict_action(example, episode_id="eval:0", timestep=0, episode_first_frame=True)
```

For subsequent steps, keep the same `episode_id` and increment `timestep`.

## Training Samples

The LeRobot dataloader now packs:

- `episode_id`: `{dataset_name}:{trajectory_id}`
- `trajectory_id`
- `timestep`
- `is_first_step`

`sequential_step_sampling=true` shuffles episode order in training while preserving increasing step order inside each episode.
