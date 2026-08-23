import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerFusion(nn.Module):
    """Fuse one or more layer-wise token feature tensors."""

    def __init__(self, hidden_size: int, num_layers: int, mode: str = "none"):
        super().__init__()
        self.mode = str(mode)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if self.mode not in {"none", "weighted_sum", "concat_linear"}:
            raise ValueError("mode must be one of: none, weighted_sum, concat_linear.")
        if self.mode == "weighted_sum":
            self.w = nn.Parameter(torch.zeros(self.num_layers))
        elif self.mode == "concat_linear":
            self.proj = nn.Linear(self.num_layers * self.hidden_size, self.hidden_size)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        if len(feats) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} feature tensors, got {len(feats)}.")
        if self.mode == "none" or len(feats) == 1:
            return feats[0]
        if self.mode == "weighted_sum":
            weights = torch.softmax(self.w, dim=0)
            return sum(weight * feat for weight, feat in zip(weights, feats))
        return self.proj(torch.cat(feats, dim=-1))


class CrossAttnFutureHead(nn.Module):
    """Predict future ViT tokens from action-token hidden states."""

    def __init__(
        self,
        hidden_size: int,
        target_tokens: int,
        num_layers_attn: int = 1,
        num_heads: int = 8,
        ffn_mult: int = 2,
        use_positional_embedding: bool = False,
        num_views: int | None = None,
        tokens_per_view: int | None = None,
        use_self_attention: bool = False,
        use_output_layernorm: bool = False,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.target_tokens = int(target_tokens)
        self.num_layers_attn = int(num_layers_attn)
        self.num_heads = int(num_heads)
        self.ffn_mult = int(ffn_mult)
        self.use_positional_embedding = bool(use_positional_embedding)
        self.use_self_attention = bool(use_self_attention)
        self.use_output_layernorm = bool(use_output_layernorm)
        if self.hidden_size <= 0 or self.target_tokens <= 0:
            raise ValueError("hidden_size and target_tokens must be positive.")
        if self.num_layers_attn < 1:
            raise ValueError("num_layers_attn must be positive.")

        self.query = nn.Parameter(torch.randn(self.target_tokens, self.hidden_size) * 0.02)
        if self.use_positional_embedding:
            if num_views is None or tokens_per_view is None:
                raise ValueError("use_positional_embedding=True requires num_views and tokens_per_view.")
            self.num_views = int(num_views)
            self.tokens_per_view = int(tokens_per_view)
            if self.num_views <= 0 or self.tokens_per_view <= 0:
                raise ValueError("num_views and tokens_per_view must be positive.")
            if self.num_views * self.tokens_per_view != self.target_tokens:
                raise ValueError(
                    f"num_views({self.num_views}) * tokens_per_view({self.tokens_per_view}) "
                    f"!= target_tokens({self.target_tokens})."
                )
            self.view_embed = nn.Parameter(torch.randn(self.num_views, self.hidden_size) * 0.02)
            self.pos_embed = nn.Parameter(torch.randn(self.tokens_per_view, self.hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        **(
                            {
                                "self_attn": nn.MultiheadAttention(
                                    self.hidden_size,
                                    self.num_heads,
                                    batch_first=True,
                                ),
                                "ln_self": nn.LayerNorm(self.hidden_size),
                            }
                            if self.use_self_attention
                            else {}
                        ),
                        "attn": nn.MultiheadAttention(self.hidden_size, self.num_heads, batch_first=True),
                        "ln_q": nn.LayerNorm(self.hidden_size),
                        "ln_kv": nn.LayerNorm(self.hidden_size),
                        "ffn": nn.Sequential(
                            nn.Linear(self.hidden_size, self.ffn_mult * self.hidden_size),
                            nn.GELU(),
                            nn.Linear(self.ffn_mult * self.hidden_size, self.hidden_size),
                        ),
                        "ln_ffn": nn.LayerNorm(self.hidden_size),
                    }
                )
                for _ in range(self.num_layers_attn)
            ]
        )
        if self.use_output_layernorm:
            self.final_ln = nn.LayerNorm(self.hidden_size)

    def forward(self, action_hidden: torch.Tensor) -> torch.Tensor:
        q = self.query.unsqueeze(0).expand(action_hidden.shape[0], -1, -1)
        if self.use_positional_embedding:
            struct_pos = (self.view_embed.unsqueeze(1) + self.pos_embed.unsqueeze(0)).reshape(
                self.target_tokens,
                self.hidden_size,
            )
            q = q + struct_pos.unsqueeze(0)
        for block in self.blocks:
            if self.use_self_attention:
                sq = block["ln_self"](q)
                q = q + block["self_attn"](sq, sq, sq, need_weights=False)[0]
            kv = block["ln_kv"](action_hidden)
            q = q + block["attn"](block["ln_q"](q), kv, kv, need_weights=False)[0]
            q = q + block["ffn"](block["ln_ffn"](q))
        if self.use_output_layernorm:
            q = self.final_ln(q)
        return q


class JointSelfAttnFutureHead(nn.Module):
    """Predict future image tokens through joint image/action self-attention."""

    def __init__(self, hidden_size, image_tokens, action_tokens, num_views, tokens_per_view,
                 num_layers_attn=1, num_heads=8, ffn_mult=2):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.image_tokens = int(image_tokens)
        self.action_tokens = int(action_tokens)
        self.num_layers_attn = int(num_layers_attn)
        if self.hidden_size <= 0 or self.image_tokens <= 0 or self.action_tokens <= 0:
            raise ValueError("hidden_size / image_tokens / action_tokens must be positive.")
        if self.num_layers_attn < 1:
            raise ValueError("num_layers_attn must be positive.")
        self.num_views = int(num_views)
        self.tokens_per_view = int(tokens_per_view)
        if self.num_views <= 0 or self.tokens_per_view <= 0:
            raise ValueError("num_views and tokens_per_view must be positive.")
        if self.num_views * self.tokens_per_view != self.image_tokens:
            raise ValueError(
                f"num_views({self.num_views}) * tokens_per_view({self.tokens_per_view}) "
                f"!= image_tokens({self.image_tokens})."
            )

        self.type_embed = nn.Parameter(torch.randn(2, self.hidden_size) * 0.02)
        self.view_embed = nn.Parameter(torch.randn(self.num_views, self.hidden_size) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(self.tokens_per_view, self.hidden_size) * 0.02)
        self.action_pos_embed = nn.Parameter(torch.randn(self.action_tokens, self.hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(self.hidden_size, num_heads, batch_first=True),
                        "ln1": nn.LayerNorm(self.hidden_size),
                        "ffn": nn.Sequential(
                            nn.Linear(self.hidden_size, ffn_mult * self.hidden_size),
                            nn.GELU(),
                            nn.Linear(ffn_mult * self.hidden_size, self.hidden_size),
                        ),
                        "ln2": nn.LayerNorm(self.hidden_size),
                    }
                )
                for _ in range(self.num_layers_attn)
            ]
        )
        self.final_ln = nn.LayerNorm(self.hidden_size)

    def _structural_positions(self, device, dtype):
        image_positions = (
            self.view_embed.unsqueeze(1) + self.pos_embed.unsqueeze(0)
        ).reshape(self.image_tokens, self.hidden_size) + self.type_embed[0]
        action_positions = self.action_pos_embed + self.type_embed[1]
        return torch.cat([image_positions, action_positions], dim=0).to(device=device, dtype=dtype)

    def forward(self, seq):
        expected_len = self.image_tokens + self.action_tokens
        if seq.ndim != 3:
            raise ValueError(f"Expected seq with shape (B, N, H), got {tuple(seq.shape)}.")
        if seq.shape[1] != expected_len:
            raise ValueError(
                f"Expected sequence length {expected_len} (image_tokens+action_tokens), got {seq.shape[1]}."
            )
        if seq.shape[2] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {seq.shape[2]}.")
        x = seq + self._structural_positions(seq.device, seq.dtype).unsqueeze(0)
        for block in self.blocks:
            h = block["ln1"](x)
            x = x + block["self_attn"](h, h, h, need_weights=False)[0]
            x = x + block["ffn"](block["ln2"](x))
        return self.final_ln(x)[:, : self.image_tokens, :]


def future_vit_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth-L1 loss on per-token-normalized future ViT targets."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}.")
    target = F.layer_norm(target, [target.shape[-1]])
    loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=(1, 2))
    if valid_mask is None:
        return loss.mean()
    valid_mask = valid_mask.to(device=loss.device, dtype=loss.dtype)
    return (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)