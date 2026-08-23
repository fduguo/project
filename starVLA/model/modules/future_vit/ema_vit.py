import copy

import torch


class EMATargetViT(torch.nn.Module):
    """EMA shadow of the online Qwen visual encoder for future-frame targets."""

    def __init__(self, online_vit, momentum: float = 0.999):
        super().__init__()
        self.ema_vit = copy.deepcopy(online_vit)
        self.ema_vit.eval()
        for p in self.ema_vit.parameters():
            p.requires_grad_(False)
        self.m = float(momentum)

    @torch.no_grad()
    def update(self, online_vit) -> None:
        """Move EMA parameters toward the online visual encoder parameters."""
        for pe, po in zip(self.ema_vit.parameters(), online_vit.parameters()):
            pe.mul_(self.m).add_(po.detach(), alpha=1.0 - self.m)

    @torch.no_grad()
    def encode(self, pixel_values, image_grid_thw):
        """Encode future frames and return detached Qwen ViT pooler tokens."""
        param = next(self.ema_vit.parameters())
        pixel_values = pixel_values.to(device=param.device)
        image_grid_thw = image_grid_thw.to(device=param.device)
        if pixel_values.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.ema_vit(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        else:
            pixel_values = pixel_values.to(dtype=param.dtype)
            out = self.ema_vit(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        return out.pooler_output.detach()