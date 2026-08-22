from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn


@dataclass
class ViewTokenLayout:
    """Token-grid metadata for one visual expert view."""

    view_index: int
    token_start: int
    token_end: int
    grid_hw: tuple[int, int]
    image_size: tuple[int, int] | None = None
    name: str | None = None


@dataclass
class ExpertLayerKV:
    """Per-layer expert K/V tensors consumed by action cross-attention."""

    key: torch.Tensor
    value: torch.Tensor
    head_indices: tuple[int, ...]
    name: str = "expert"


@dataclass
class ExpertKVBundle:
    """A batch of layer-wise expert K/V tensors plus optional visualization metadata."""

    layers: list[ExpertLayerKV]
    layer_indices: tuple[int, ...] | None = None
    token_layouts: list[list[ViewTokenLayout]] | None = None
    visual_maps: Any | None = None
    provider_name: str = "expert"

    def repeat_batch(self, repeats: int) -> "ExpertKVBundle":
        if repeats == 1:
            return self
        return ExpertKVBundle(
            layers=[
                ExpertLayerKV(
                    key=layer.key.repeat(repeats, 1, 1, 1),
                    value=layer.value.repeat(repeats, 1, 1, 1),
                    head_indices=layer.head_indices,
                    name=layer.name,
                )
                for layer in self.layers
            ],
            layer_indices=self.layer_indices,
            token_layouts=None if self.token_layouts is None else self.token_layouts * repeats,
            visual_maps=None if self.visual_maps is None else self.visual_maps,
            provider_name=self.provider_name,
        )


@dataclass
class ExpertRoutingConfig:
    """Static head routing for an expert K/V provider."""

    layer_indices: tuple[int, ...]
    head_indices: tuple[int, ...]
    mode: str = "replace_heads"


class ExpertKVProjector(nn.Module):
    """Project expert tokens to layer-wise K/V tensors for multi-head attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        head_indices: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        name: str = "expert",
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_layers = int(num_layers)
        self.head_indices = tuple(int(i) for i in head_indices)
        self.layer_indices = tuple(int(i) for i in layer_indices) if layer_indices is not None else tuple(range(num_layers))
        self.total_kv_dim = self.num_heads * self.head_dim
        self.name = name

        self.norm = nn.LayerNorm(self.hidden_size)
        self.k_projectors = nn.ModuleList([nn.Linear(self.hidden_size, self.total_kv_dim) for _ in range(num_layers)])
        self.v_projectors = nn.ModuleList([nn.Linear(self.hidden_size, self.total_kv_dim) for _ in range(num_layers)])
        self._init_weights()

    def _init_weights(self):
        for k_proj, v_proj in zip(self.k_projectors, self.v_projectors):
            nn.init.xavier_uniform_(k_proj.weight)
            nn.init.xavier_uniform_(v_proj.weight)
            nn.init.zeros_(k_proj.bias)
            nn.init.zeros_(v_proj.bias)

    def forward(
        self,
        tokens_by_layer: Sequence[torch.Tensor],
        *,
        token_layouts: list[list[ViewTokenLayout]] | None = None,
        visual_maps: Any | None = None,
    ) -> ExpertKVBundle:
        if len(tokens_by_layer) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} expert token groups, got {len(tokens_by_layer)}.")

        layers = []
        for tokens, k_proj, v_proj in zip(tokens_by_layer, self.k_projectors, self.v_projectors):
            if tokens.dtype != self.norm.weight.dtype:
                tokens = tokens.to(self.norm.weight.dtype)
            tokens = self.norm(tokens)
            batch_size, num_tokens, _ = tokens.shape
            key = k_proj(tokens).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            value = v_proj(tokens).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            layers.append(ExpertLayerKV(key=key, value=value, head_indices=self.head_indices, name=self.name))

        return ExpertKVBundle(
            layers=layers,
            layer_indices=self.layer_indices,
            token_layouts=token_layouts,
            visual_maps=visual_maps,
            provider_name=self.name,
        )
