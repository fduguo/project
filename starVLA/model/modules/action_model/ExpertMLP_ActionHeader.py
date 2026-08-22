"""MLP action head with optional reusable expert K/V conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.action_model.MLP_ActionHeader import MLPResNetBlock
from starVLA.model.modules.expert_kv import ExpertKVBundle
from starVLA.model.modules.expert_kv import ExpertLayerKV


class ExpertKVFusion(nn.Module):
    """Gated cross-attention from action tokens to one expert K/V group."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def _to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, expert_kv: ExpertLayerKV | None = None) -> torch.Tensor:
        if expert_kv is None or not expert_kv.head_indices:
            return x

        head_indices = tuple(int(i) for i in expert_kv.head_indices)
        if min(head_indices) < 0 or max(head_indices) >= self.num_heads:
            raise ValueError(f"Expert head indices {head_indices} exceed attention heads={self.num_heads}.")

        query = self._to_heads(self.q_proj(x))
        expert_key = expert_kv.key.to(device=query.device, dtype=query.dtype)
        expert_value = expert_kv.value.to(device=query.device, dtype=query.dtype)
        routed_query = query[:, head_indices]
        routed_key = expert_key[:, head_indices]
        routed_value = expert_value[:, head_indices]

        routed_output = F.scaled_dot_product_attention(
            routed_query,
            routed_key,
            routed_value,
            attn_mask=None,
            dropout_p=0.0,
        )
        output = torch.zeros_like(query)
        output[:, head_indices] = routed_output
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        output = self.out_proj(output)
        return x + torch.tanh(self.gate) * output


class ExpertMLPResNet(nn.Module):
    """MLPResNet that preserves per-action tokens for expert attention."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_expert_layers: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.expert_fusions = nn.ModuleList(
            [ExpertKVFusion(dim=hidden_dim, num_heads=num_heads) for _ in range(num_expert_layers)]
        )

    def _expert_layer_map(self, expert_kv: ExpertKVBundle | None) -> dict[int, ExpertLayerKV] | None:
        if expert_kv is None:
            return None
        if expert_kv.layer_indices is None:
            return {idx: layer for idx, layer in enumerate(expert_kv.layers)}
        if len(expert_kv.layer_indices) != len(expert_kv.layers):
            raise ValueError("expert_kv.layer_indices length must match expert_kv.layers.")
        return {int(idx): layer for idx, layer in zip(expert_kv.layer_indices, expert_kv.layers)}

    def _fuse(self, x: torch.Tensor, layer_idx: int, expert_by_layer: dict[int, ExpertLayerKV] | None) -> torch.Tensor:
        if layer_idx >= len(self.expert_fusions) or expert_by_layer is None:
            return x
        return self.expert_fusions[layer_idx](x, expert_by_layer.get(layer_idx))

    def forward(self, x: torch.Tensor, expert_kv: ExpertKVBundle | None = None) -> torch.Tensor:
        expert_by_layer = self._expert_layer_map(expert_kv)
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self._fuse(x, 0, expert_by_layer)
        for idx, block in enumerate(self.mlp_resnet_blocks, start=1):
            x = block(x)
            x = self._fuse(x, idx, expert_by_layer)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class ExpertL1RegressionActionHead(nn.Module):
    """Simple MLP action head with optional expert K/V fusion."""

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=4096,
        action_dim=7,
        NUM_ACTIONS_CHUNK=8,
        num_expert_layers=3,
        num_heads=8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.model = ExpertMLPResNet(
            num_blocks=2,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_expert_layers=num_expert_layers,
            num_heads=num_heads,
        )

    def predict_action(self, actions_hidden_states, expert_kv: ExpertKVBundle | None = None):
        actions = self.model(actions_hidden_states, expert_kv=expert_kv)
        return actions

    def forward(self, actions_hidden_states, expert_kv: ExpertKVBundle | None = None):
        return self.predict_action(actions_hidden_states, expert_kv=expert_kv)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    action_model_cfg = config.framework.action_model
    action_hidden_dim = action_model_cfg.action_hidden_dim
    action_dim = action_model_cfg.action_dim
    action_horizon = int(action_model_cfg.action_horizon)
    num_expert_layers = int(action_model_cfg.get("num_expert_layers", 3))
    num_heads = int(action_model_cfg.get("num_attention_heads", 8))

    return ExpertL1RegressionActionHead(
        input_dim=action_hidden_dim,
        hidden_dim=action_hidden_dim * 2,
        action_dim=action_dim,
        NUM_ACTIONS_CHUNK=action_horizon,
        num_expert_layers=num_expert_layers,
        num_heads=num_heads,
    )
