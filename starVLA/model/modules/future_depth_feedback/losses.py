from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_batch_mean(values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return values.mean()
    mask = valid_mask.to(device=values.device, dtype=values.dtype)
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ValueError(
            f"valid_mask must have shape [{values.shape[0]}], got {tuple(mask.shape)}."
        )
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def future_depth_feature_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    cosine_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress fixed DA3 features with sample-level future-frame masking."""

    if prediction.shape != target.shape:
        raise ValueError(
            "Future-depth prediction/target mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
        )

    prediction_fp32 = prediction.float()
    target_fp32 = F.layer_norm(target.float(), [target.shape[-1]])
    smooth_l1_per_sample = F.smooth_l1_loss(
        prediction_fp32,
        target_fp32,
        reduction="none",
    ).mean(dim=(1, 2))
    smooth_l1 = _masked_batch_mean(smooth_l1_per_sample, valid_mask)

    cosine_per_token = 1.0 - F.cosine_similarity(
        prediction_fp32,
        target_fp32,
        dim=-1,
        eps=1e-6,
    )
    cosine_per_sample = cosine_per_token.mean(dim=1)
    cosine = _masked_batch_mean(cosine_per_sample, valid_mask)
    total = smooth_l1 + float(cosine_weight) * cosine
    return total, {
        "future_depth_smooth_l1": smooth_l1.detach(),
        "future_depth_cosine": cosine.detach(),
    }
