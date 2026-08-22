# Memory Expert KV

This implementation adds two memory expert frameworks:

- `QwenKI_memory`: KI flow-matching DiT action head with episode memory K/V.
- `QwenOFT_memory`: OFT ExpertMLP action head with episode memory K/V.

Both paths share `MemoryExpertKVProvider`, which wraps a stateful `EpisodeMemoryBank` and projects fused memory tokens into the existing `ExpertKVBundle` contract consumed by the action heads.

## Difference From Depth Expert

Depth expert K/V is computed from frozen multi-view depth features for the current observation. Memory expert K/V is computed from current VLM/action-context tokens plus an episode-local history bank. It captures temporal context and does not require Depth Anything or MemoryVLA as a runtime dependency.

## Scope

The first version supports KI+memory and OFT+memory only. It does not combine depth and memory in the same framework.
