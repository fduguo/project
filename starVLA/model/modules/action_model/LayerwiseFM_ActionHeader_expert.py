# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by starVLA contributors in 2026.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import ActionEncoder
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import DiTConfig
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import MLP
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit_expert import ExpertDiT
from starVLA.model.modules.expert_kv import ExpertAttentionRecord
from starVLA.model.modules.expert_kv import ExpertKVBundle


@dataclass
class ExpertFlowmatchingActionHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(default=True)
    diffusion_model_cfg: dict = field(default=None)
    input_embedding_dim: int = field(default=1536)
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=None)
    num_target_vision_tokens: int = field(default=32)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class LayerwiseExpertFlowmatchingActionHead(nn.Module):
    """Layer-wise flow-matching action head with reusable expert K/V routing."""

    def __init__(self, global_config, **kwargs):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg
        for k, v in DiTConfig.items():
            if diffusion_model_cfg.get(k, None) is None:
                diffusion_model_cfg[k] = v

        _DIT_NON_KWARGS = {"action_dit_hidden_dim"}
        diffusion_model_cfg_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in _DIT_NON_KWARGS}

        self.input_embedding_dim = diffusion_model_cfg_kwargs["input_embedding_dim"]
        self.model = ExpertDiT(**diffusion_model_cfg_kwargs)
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = (
            MLP(input_dim=action_config.state_dim, output_dim=self.input_embedding_dim)
            if action_config.state_dim
            else None
        )
        self.action_encoder = ActionEncoder(action_dim=action_config.action_dim, hidden_size=self.input_embedding_dim)
        self.action_decoder = MLP(input_dim=self.input_embedding_dim, hidden_dim=1024, output_dim=self.action_dim)
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return self.config.noise_s * (1 - sample)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _expert_layer_map(self, expert_kv: ExpertKVBundle | None):
        if expert_kv is None:
            return None
        if expert_kv.layer_indices is None:
            return {idx: layer for idx, layer in enumerate(expert_kv.layers)}
        if len(expert_kv.layer_indices) != len(expert_kv.layers):
            raise ValueError("expert_kv.layer_indices length must match expert_kv.layers.")
        return {int(idx): layer for idx, layer in zip(expert_kv.layer_indices, expert_kv.layers)}

    def _print_expert_gates(self, expert_kv: ExpertKVBundle | None = None) -> None:
        expert_by_layer = self._expert_layer_map(expert_kv)
        if expert_by_layer is None:
            layer_indices = range(len(self.model.transformer_blocks))
        else:
            layer_indices = sorted(
                idx for idx in expert_by_layer if 0 <= idx < len(self.model.transformer_blocks)
            )

        parts = []
        for layer_idx in layer_indices:
            attn = self.model.transformer_blocks[layer_idx].attn1
            raw_gate = attn.expert_gate.detach().float().cpu()
            effective_gate = torch.tanh(raw_gate)
            heads = "" if expert_by_layer is None else f" heads={expert_by_layer[layer_idx].head_indices}"
            parts.append(
                f"layer={layer_idx}{heads} mode={attn.expert_fusion_mode} "
                f"raw={raw_gate.item():.6f} tanh={effective_gate.item():.6f}"
            )

        message = " | ".join(parts) if parts else "no active expert layers"
        print(f"[ExpertGate] {message}", flush=True)

    def _run_blocks(
        self,
        model_output: torch.Tensor,
        vl_embs_list: list,
        temb: torch.Tensor,
        expert_kv: ExpertKVBundle | None = None,
        return_attention_records: bool = False,
    ) -> tuple[torch.Tensor, list[ExpertAttentionRecord]]:
        expert_by_layer = self._expert_layer_map(expert_kv)
        records: list[ExpertAttentionRecord] = []
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            layer_expert = expert_by_layer.get(layer_idx) if expert_by_layer is not None else None
            model_output, record = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_embs_list[layer_idx],
                temb=temb,
                expert_kv=layer_expert,
                return_attention_record=return_attention_records and layer_expert is not None,
                layer_idx=layer_idx,
            )
            if record is not None:
                records.append(record)
        return model_output, records

    def forward(self, vl_embs_list: list, actions: torch.Tensor, state: torch.Tensor = None, expert_kv=None):
        device = actions.device
        batch_size = vl_embs_list[0].shape[0]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        state_features = self.state_encoder(state) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )
        temb = self.model.timestep_encoder(t_discretized)
        model_output, _ = self._run_blocks(sa_embs, vl_embs_list, temb, expert_kv=expert_kv)
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]
        return ((pred_actions - velocity) ** 2).mean()

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        expert_kv: ExpertKVBundle | None = None,
        return_attention_records: bool = False,
        print_expert_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[ExpertAttentionRecord]]:
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self.state_encoder(state) if state is not None else None
        all_records: list[ExpertAttentionRecord] = []

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )
            temb = self.model.timestep_encoder(timesteps_tensor)
            model_output, records = self._run_blocks(
                sa_embs,
                vl_embs_list,
                temb,
                expert_kv=expert_kv,
                return_attention_records=return_attention_records,
            )
            if return_attention_records and t == num_steps - 1:
                all_records = records
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity

        if print_expert_gate:
            self._print_expert_gates(expert_kv)
        if return_attention_records:
            return actions, all_records
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    return LayerwiseExpertFlowmatchingActionHead(global_config=config)
