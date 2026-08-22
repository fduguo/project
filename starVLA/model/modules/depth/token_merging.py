import torch
import torch.nn as nn


class TokenMerging2D(nn.Module):
    """Merge square-grid patch tokens with a strided 2-D convolution."""

    def __init__(self, patch_size: int = 4, in_dim: int = 384, out_dim: int = 1024):
        super().__init__()
        self.patch_size = int(patch_size)
        self.merge = nn.Conv2d(in_dim, out_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_inference():
            x = x.clone()
        x = self.norm(x)
        batch_size, num_tokens, feature_dim = x.shape
        hw = int(num_tokens**0.5)
        if hw * hw != num_tokens:
            raise ValueError(f"TokenMerging2D expects a square token grid, got {num_tokens} tokens.")
        x = x.view(batch_size, hw, hw, feature_dim).permute(0, 3, 1, 2)
        x = self.merge(x)
        return x.flatten(2).transpose(1, 2)
