# Implementation Notes

## MemoryVLA Migration

`EpisodeMemoryBank` follows the core MemoryVLA idea: keep a per-episode token bank, retrieve prior episode tokens with cross-attention, fuse current and retrieved tokens, then consolidate the bank to a fixed length.

Implemented components:

- timestep sinusoidal encoding through `TimestepPositionalEncoding`
- retrieval blocks based on scaled-dot-product cross-attention
- `gate` and `add` fusion
- `fifo` and `tome` consolidation

The implementation is local to `starVLA/model/modules/memory/` and does not import `third_party/MemoryVLA`.

## KV Shape Contract

`MemoryExpertKVProvider` emits `ExpertKVBundle`:

- key/value shape: `[B, num_heads, memory_tokens, head_dim]`
- `layer_indices`: target action-head layers
- `head_indices`: action attention heads routed to memory K/V
- `provider_name`: `memory`

## KI Path

`QwenKI_memory` uses the same flow as `QwenKI_depth` but replaces depth K/V with memory K/V. Memory tokens come from action-side projected, trimmed VLM context hidden states, so their dimension matches `action_dit_hidden_dim`.

The provider input and output hidden size are both `action_dit_hidden_dim`.

## OFT Path

`QwenOFT_memory` uses `last_hidden` from the VLM prompt pass. Memory tokens are context/action-prompt hidden states with action placeholder tokens removed. Action queries are still the gathered action-token embeddings.

The provider input dimension is `action_hidden_dim`. Its memory/KV projection dimension is `action_hidden_dim * 2`, matching the ExpertMLP hidden dimension.
