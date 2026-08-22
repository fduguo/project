from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF

from starVLA.model.modules.depth import DepthEncoder


_SUPPORTED_DA3_FEATURE_LAYERS = (5, 7, 9, 11)


class FrozenDA3FeatureTarget(DepthEncoder):
    """Frozen DA3 teacher with deterministic spatial pooling.

    The existing depth expert's TokenMerging2D convolution is deliberately not
    used here: it is trainable and randomly initialized, so it would make the
    future-feature target move during training. This class keeps only frozen DA3
    features and parameter-free adaptive average pooling.
    """

    def __init__(
        self,
        depth_model_name: str,
        target_feature_layer: int = 11,
        target_grid_size: int = 8,
        expected_feature_dim: int | None = None,
        image_size: int | Sequence[int] = 518,
    ) -> None:
        # feature_dim is temporary because the parent constructs TokenMerging2D;
        # that trainable projection is removed immediately below.
        super().__init__(
            depth_model_name=depth_model_name,
            feature_dim=1,
            freeze_depth_model=True,
        )
        del self.token_merging_model

        self.target_feature_layer = int(target_feature_layer)
        if self.target_feature_layer not in _SUPPORTED_DA3_FEATURE_LAYERS:
            raise ValueError(
                f"target_feature_layer must be one of {_SUPPORTED_DA3_FEATURE_LAYERS}, "
                f"got {self.target_feature_layer}."
            )
        self.target_grid_size = int(target_grid_size)
        if self.target_grid_size <= 0:
            raise ValueError("target_grid_size must be positive.")
        self.image_size = image_size
        self.output_dim = int(self.da3_model.model.backbone.pretrained.embed_dim)
        if expected_feature_dim is not None and self.output_dim != int(expected_feature_dim):
            raise RuntimeError(
                "DA3 feature dimension does not match the framework config: "
                f"model={self.output_dim}, config={int(expected_feature_dim)}."
            )
        self.feature_dim = self.output_dim
        self.to_tensor = TF.ToTensor()
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        # A teacher target must never acquire train-mode behavior.
        return super().train(False)

    def _target_hw(self) -> tuple[int, int]:
        if isinstance(self.image_size, Sequence) and not isinstance(self.image_size, (str, bytes)):
            values = list(self.image_size)
            if len(values) == 2:
                return int(values[0]), int(values[1])
        size = int(self.image_size)
        return size, size

    def _pil_to_tensor(self, image: Any) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8))
        target_h, target_w = self._target_hw()
        image = image.convert("RGB").resize((target_w, target_h), Image.Resampling.BICUBIC)
        return self.to_tensor(image) * 2.0 - 1.0

    @torch.no_grad()
    def encode_views(
        self,
        batch_images: list[list[Any]],
        *,
        expected_num_views: int,
        device: torch.device,
        micro_batch_size: int = 8,
    ) -> torch.Tensor:
        flat_images = []
        view_counts = []
        for sample_images in batch_images:
            if not isinstance(sample_images, (list, tuple)):
                sample_images = [sample_images]
            view_counts.append(len(sample_images))
            flat_images.extend(self._pil_to_tensor(image) for image in sample_images)

        if not flat_images:
            raise ValueError("Future-depth teacher received no images.")
        if any(count != int(expected_num_views) for count in view_counts):
            raise ValueError(
                f"Expected {expected_num_views} future views per sample, got {view_counts}."
            )

        micro_batch_size = int(micro_batch_size)
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        images = torch.stack(flat_images, dim=0)
        token_chunks = []
        for start in range(0, images.shape[0], micro_batch_size):
            image_chunk = images[start : start + micro_batch_size].to(device=device)
            token_chunks.append(self.forward(image_chunk))
            del image_chunk
        tokens_per_image = torch.cat(token_chunks, dim=0)
        batch_size = len(batch_images)
        num_tokens = tokens_per_image.shape[1]
        return tokens_per_image.reshape(
            batch_size,
            int(expected_num_views) * num_tokens,
            tokens_per_image.shape[-1],
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract one frozen backbone layer without running the DPT depth head."""
        self._ensure_depth_model_fp32()
        batch_size = images.shape[0]
        x = images.float()
        x = (x + 1.0) * 0.5
        x = (x - self.img_mean.float()) / self.img_std.float()
        x = x.unsqueeze(1)
        with torch.autocast(device_type=images.device.type, enabled=False):
            _, auxiliary_features = self.da3_model.model.backbone(
                x,
                cam_token=None,
                export_feat_layers=[self.target_feature_layer],
                ref_view_strategy="saddle_balanced",
            )
        if len(auxiliary_features) != 1:
            raise RuntimeError(
                "DA3 backbone returned an unexpected number of requested feature groups: "
                f"{len(auxiliary_features)}."
            )
        feature = auxiliary_features[0]
        feature = feature.reshape(batch_size, -1, feature.shape[-1])
        if feature.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"Unexpected DA3 feature dim {feature.shape[-1]}; expected {self.output_dim}."
            )
        grid_hw = int(feature.shape[1] ** 0.5)
        if grid_hw * grid_hw != feature.shape[1]:
            raise RuntimeError(
                "DA3 feature tokens do not form a square grid: "
                f"shape={tuple(feature.shape)}."
            )
        feature_map = feature.reshape(batch_size, grid_hw, grid_hw, self.output_dim).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(
            feature_map.float(),
            output_size=(self.target_grid_size, self.target_grid_size),
        )
        return pooled.flatten(2).transpose(1, 2)
