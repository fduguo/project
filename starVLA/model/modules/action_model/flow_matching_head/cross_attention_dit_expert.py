# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from torch import nn

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import AdaLayerNorm
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import TimestepEncoder
from starVLA.model.modules.expert_kv import ExpertAttentionRecord
from starVLA.model.modules.expert_kv import ExpertLayerKV

_EXPERT_FUSION_MODES = {"gated_residual", "replacement"}


class ExpertAwareAttention(nn.Module):
    """Cross-attention that can route selected heads to expert-provided K/V."""

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        expert_fusion_mode: str = "gated_residual",
    ):
        super().__init__()
        expert_fusion_mode = str(expert_fusion_mode)
        if expert_fusion_mode not in _EXPERT_FUSION_MODES:
            raise ValueError(
                f"expert_fusion_mode={expert_fusion_mode!r} must be one of {sorted(_EXPERT_FUSION_MODES)}."
            )
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.expert_fusion_mode = expert_fusion_mode
        cross_attention_dim = cross_attention_dim or query_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=out_bias), nn.Dropout(dropout)])
        self.expert_gate = nn.Parameter(torch.zeros(1))
        self.dropout = dropout

    def _shape(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        return tensor.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        expert_kv: ExpertLayerKV | None = None,
        return_attention_record: bool = False,
        layer_idx: int = -1,
    ) -> tuple[torch.Tensor, ExpertAttentionRecord | None]:
        batch_size = hidden_states.shape[0]
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        query = self._shape(self.to_q(hidden_states), batch_size)
        key = self._shape(self.to_k(encoder_hidden_states), batch_size)
        value = self._shape(self.to_v(encoder_hidden_states), batch_size)

        if query.dtype != key.dtype or query.dtype != value.dtype:
            dtype = torch.promote_types(query.dtype, key.dtype)
            dtype = torch.promote_types(dtype, value.dtype)
            query = query.to(dtype)
            key = key.to(dtype)
            value = value.to(dtype)
        if attention_mask is not None and attention_mask.dtype not in (torch.bool, query.dtype):
            attention_mask = attention_mask.to(query.dtype)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        record = None

        if expert_kv is not None and expert_kv.head_indices:
            head_indices = list(expert_kv.head_indices)
            if min(head_indices) < 0 or max(head_indices) >= self.heads:
                raise ValueError(f"Expert head indices {head_indices} exceed attention heads={self.heads}.")

            expert_key = expert_kv.key.to(device=query.device, dtype=query.dtype)
            expert_value = expert_kv.value.to(device=query.device, dtype=query.dtype)
            q_expert = query[:, head_indices]
            k_expert = expert_key[:, head_indices]
            v_expert = expert_value[:, head_indices]

            if return_attention_record:
                scale = self.dim_head**-0.5
                scores = torch.matmul(q_expert, k_expert.transpose(-2, -1)) * scale
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
                expert_output = torch.matmul(
                    F.dropout(probs, p=self.dropout, training=self.training).to(v_expert.dtype),
                    v_expert,
                )
                record = ExpertAttentionRecord(
                    layer_idx=layer_idx,
                    head_indices=tuple(head_indices),
                    attention=probs.detach().cpu(),
                    expert_name=expert_kv.name,
                )
            else:
                expert_output = F.scaled_dot_product_attention(
                    q_expert,
                    k_expert,
                    v_expert,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            merged_output = output.clone()
            if self.expert_fusion_mode == "replacement":
                merged_output[:, head_indices] = expert_output
            else:
                merged_output[:, head_indices] = output[:, head_indices] + torch.tanh(self.expert_gate) * expert_output
            output = merged_output

        output = output.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        output = self.to_out[0](output)
        output = self.to_out[1](output)
        return output, record


class ExpertBasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        expert_fusion_mode: str = "gated_residual",
    ):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.pos_embed = (
            SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
            if positional_embeddings == "sinusoidal"
            else None
        )
        self.attn1 = ExpertAwareAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            out_bias=attention_out_bias,
            expert_fusion_mode=expert_fusion_mode,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        expert_kv: ExpertLayerKV | None = None,
        return_attention_record: bool = False,
        layer_idx: int = -1,
    ) -> tuple[torch.Tensor, ExpertAttentionRecord | None]:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output, record = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            expert_kv=expert_kv,
            return_attention_record=return_attention_record,
            layer_idx=layer_idx,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        ff_output = self.ff(self.norm3(hidden_states))
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states, record


class ExpertDiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        expert_fusion_mode: str = "gated_residual",
        **kwargs,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False
        compute_dtype = getattr(self.config, "compute_dtype", torch.float32)
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, compute_dtype=compute_dtype)

        blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            blocks.append(
                ExpertBasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=None if use_self_attn else cross_attention_dim,
                    expert_fusion_mode=expert_fusion_mode,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of ExpertDiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None,
        expert_kv_layers: list[ExpertLayerKV] | None = None,
        return_attention_records: bool = False,
    ):
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        records: list[ExpertAttentionRecord] = []

        for idx, block in enumerate(self.transformer_blocks):
            is_self_attn = idx % 2 == 1 and self.config.interleave_self_attention
            expert_kv = None if is_self_attn or expert_kv_layers is None else expert_kv_layers[idx]
            hidden_states, record = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None if is_self_attn else encoder_hidden_states,
                encoder_attention_mask=None if is_self_attn else encoder_attention_mask,
                temb=temb,
                expert_kv=expert_kv,
                return_attention_record=return_attention_records and expert_kv is not None,
                layer_idx=idx,
            )
            if record is not None:
                records.append(record)
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        out = self.proj_out_2(hidden_states)
        if return_all_hidden_states:
            return out, all_hidden_states, records
        return out, records
