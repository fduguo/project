from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .types import ViewTokenLayout


@dataclass
class ExpertAttentionRecord:
    """Attention from action queries to expert tokens for one DiT layer."""

    layer_idx: int
    head_indices: tuple[int, ...]
    attention: torch.Tensor
    expert_name: str = "expert"


@dataclass
class ExpertVisualizationBundle:
    records: list[ExpertAttentionRecord]
    token_layouts: list[list[ViewTokenLayout]] | None = None
    visual_maps: Any | None = None
    output_paths: list[str] | None = None


def _to_uint8_image(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return arr[..., :3]


def _colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max()
    if denom > 0:
        heatmap = heatmap / denom
    red = heatmap
    green = 1.0 - np.abs(heatmap - 0.5) * 2.0
    blue = 1.0 - heatmap
    return np.stack([red, np.clip(green, 0.0, 1.0), blue], axis=-1)


def _layout_heatmap(attn: torch.Tensor, layout: ViewTokenLayout, image_hw: tuple[int, int]) -> np.ndarray:
    token_attn = attn[..., layout.token_start : layout.token_end].mean(dim=tuple(range(attn.ndim - 1)))
    h, w = layout.grid_hw
    heat = token_attn.reshape(1, 1, h, w).float()
    heat = F.interpolate(heat, size=image_hw, mode="bilinear", align_corners=False)
    return heat[0, 0].detach().cpu().numpy()


def render_expert_visualization(
    *,
    images: list[list[Any]],
    records: list[ExpertAttentionRecord],
    token_layouts: list[list[ViewTokenLayout]] | None,
    output_dir: str | Path,
    max_samples: int = 1,
    alpha: float = 0.45,
) -> ExpertVisualizationBundle:
    """Render reusable expert-attention overlays.

    The renderer only needs token layouts and attention records. Expert-specific
    modules may add visual maps separately, but overlays work for any K/V expert.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    if token_layouts is None:
        return ExpertVisualizationBundle(records=records, token_layouts=token_layouts, output_paths=paths)

    for sample_idx, sample_layouts in enumerate(token_layouts[:max_samples]):
        if sample_idx >= len(images):
            break
        sample_images = images[sample_idx]
        for record in records:
            if sample_idx >= record.attention.shape[0]:
                continue
            sample_attn = record.attention[sample_idx]
            for layout in sample_layouts:
                if layout.view_index >= len(sample_images):
                    continue
                image = _to_uint8_image(sample_images[layout.view_index])
                heat = _layout_heatmap(sample_attn, layout, image.shape[:2])
                color = (_colorize_heatmap(heat) * 255).astype(np.uint8)
                overlay = np.clip((1.0 - alpha) * image + alpha * color, 0, 255).astype(np.uint8)
                name = layout.name or f"view{layout.view_index}"
                path = output / f"sample{sample_idx}_layer{record.layer_idx}_{record.expert_name}_{name}.png"
                Image.fromarray(overlay).save(path)
                paths.append(str(path))

    return ExpertVisualizationBundle(records=records, token_layouts=token_layouts, output_paths=paths)
