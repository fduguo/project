from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as TF

from starVLA.model.modules.depth.token_merging import TokenMerging2D
from starVLA.model.modules.expert_kv import ExpertKVBundle
from starVLA.model.modules.expert_kv import ExpertKVProjector
from starVLA.model.modules.expert_kv import ViewTokenLayout

_FEAT_LAYERS = [5, 7, 9, 11]
_DEPTH_FEATURE_MAPPING_STRATEGIES = {"one_to_one", "uniform_segments", "cyclic", "shared"}
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DepthEncoder(nn.Module):
    """Frozen Depth Anything 3 encoder that emits merged feature tokens."""

    def __init__(
        self,
        depth_model_name: str,
        feature_dim: int,
        freeze_depth_model: bool = True,
        merge_patch_size: int = 4,
    ):
        super().__init__()
        if "gsplat" not in sys.modules:
            stub = types.ModuleType("gsplat")
            stub.rasterization = None
            sys.modules["gsplat"] = stub
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as import_error:
            raise ImportError(
                "QwenKI_depth requires depth-anything-3. Install the optional depth dependency "
                "or use a custom depth encoder for tests."
            ) from import_error

        self.da3_model = DepthAnything3.from_pretrained(depth_model_name)
        self.da3_model.eval()
        self._depth_model_fp32 = False
        if freeze_depth_model:
            self._freeze_depth_model()
        embed_dim = self.da3_model.model.backbone.pretrained.embed_dim
        self.token_merging_model = TokenMerging2D(patch_size=merge_patch_size, in_dim=embed_dim, out_dim=feature_dim)
        self.feature_dim = feature_dim
        self.freeze_depth_model = freeze_depth_model
        self.register_buffer("img_mean", torch.tensor(_IMAGENET_MEAN).view(3, 1, 1))
        self.register_buffer("img_std", torch.tensor(_IMAGENET_STD).view(3, 1, 1))

    def _freeze_depth_model(self):
        for param in self.da3_model.parameters():
            param.requires_grad = False
        self.da3_model.eval()

    def _ensure_depth_model_fp32(self):
        if not self._depth_model_fp32:
            self.da3_model.float()
            self.da3_model.eval()
            self._depth_model_fp32 = True

    def _extract_layer_features(self, output, batch_size: int):
        result = []
        for layer_idx in _FEAT_LAYERS:
            feat = output.aux[f"feat_layer_{layer_idx}"]
            feat = feat.squeeze(1)
            feat = feat.reshape(batch_size, -1, feat.shape[-1])
            result.append(feat.clone())
        return tuple(result)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self._ensure_depth_model_fp32()
        batch_size = images.shape[0]
        x = images.float()
        x = (x + 1.0) * 0.5
        x = (x - self.img_mean.float()) / self.img_std.float()
        x = x.unsqueeze(1)
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            output = self.da3_model.model(
                x,
                None,
                None,
                export_feat_layers=_FEAT_LAYERS,
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy="saddle_balanced",
            )
        layer_features = self._extract_layer_features(output, batch_size)
        merge_dtype = self.token_merging_model.merge.weight.dtype
        return tuple(self.token_merging_model(features.to(dtype=merge_dtype)) for features in layer_features)


class MultiViewDepthKVProvider(nn.Module):
    """Encode all camera views and project merged depth tokens to expert K/V."""

    def __init__(
        self,
        depth_model_name: str | None,
        feature_dim: int,
        num_heads: int,
        head_dim: int,
        layer_indices: Sequence[int],
        head_indices: Sequence[int],
        depth_feature_mapping_strategy: str = "one_to_one",
        depth_shared_feature_index: int = -1,
        image_size: int | Sequence[int] = 518,
        view_mode: str = "all_views_concat",
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        if view_mode != "all_views_concat":
            raise ValueError("Only depth view_mode='all_views_concat' is implemented.")
        self.view_mode = view_mode
        self.image_size = image_size
        self.feature_dim = int(feature_dim)
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.head_indices = tuple(int(i) for i in head_indices)
        self.depth_feature_mapping_strategy = str(depth_feature_mapping_strategy).lower()
        if self.depth_feature_mapping_strategy not in _DEPTH_FEATURE_MAPPING_STRATEGIES:
            raise ValueError(
                f"Unknown depth_feature_mapping_strategy={depth_feature_mapping_strategy!r}. "
                f"Expected one of {sorted(_DEPTH_FEATURE_MAPPING_STRATEGIES)}."
            )
        self.depth_shared_feature_index = int(depth_shared_feature_index)
        self.encoder = encoder or DepthEncoder(depth_model_name, feature_dim=self.feature_dim, freeze_depth_model=True)
        self.depth_token_proj = ExpertKVProjector(
            hidden_size=self.feature_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=len(self.layer_indices),
            head_indices=self.head_indices,
            layer_indices=self.layer_indices,
            name="depth",
        )
        self.to_tensor = TF.ToTensor()

    def _target_hw(self) -> tuple[int, int]:
        if isinstance(self.image_size, Sequence) and not isinstance(self.image_size, (str, bytes)):
            vals = list(self.image_size)
            if len(vals) == 2:
                return int(vals[0]), int(vals[1])
        size = int(self.image_size)
        return size, size

    def _pil_to_tensor(self, image: Any) -> tuple[torch.Tensor, tuple[int, int]]:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8))
        orig_size = image.size
        target_h, target_w = self._target_hw()
        image = image.convert("RGB").resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensor = self.to_tensor(image) * 2.0 - 1.0
        return tensor, orig_size

    def _flatten_images(self, batch_images: list[list[Any]]) -> tuple[torch.Tensor, list[int], list[list[tuple[int, int]]]]:
        flat = []
        view_counts = []
        image_sizes = []
        for sample_images in batch_images:
            if not isinstance(sample_images, (list, tuple)):
                sample_images = [sample_images]
            view_counts.append(len(sample_images))
            sample_sizes = []
            for image in sample_images:
                tensor, orig_size = self._pil_to_tensor(image)
                flat.append(tensor)
                sample_sizes.append(orig_size)
            image_sizes.append(sample_sizes)
        return torch.stack(flat, dim=0), view_counts, image_sizes

    def _resolve_depth_source_indices(self, num_depth_token_groups: int) -> tuple[int, ...]:
        num_layers = len(self.layer_indices)
        if num_layers == 0:
            return ()
        if num_depth_token_groups <= 0:
            raise ValueError("Depth encoder returned no token groups.")

        strategy = self.depth_feature_mapping_strategy
        if strategy == "one_to_one":
            if num_depth_token_groups < num_layers:
                raise ValueError(
                    f"Depth encoder returned {num_depth_token_groups} token groups, "
                    f"but {num_layers} expert layers were requested with "
                    "depth_feature_mapping_strategy='one_to_one'."
                )
            return tuple(range(num_layers))
        if strategy == "uniform_segments":
            return tuple(
                min((idx * num_depth_token_groups) // num_layers, num_depth_token_groups - 1)
                for idx in range(num_layers)
            )
        if strategy == "cyclic":
            return tuple(idx % num_depth_token_groups for idx in range(num_layers))
        if strategy == "shared":
            source_idx = self.depth_shared_feature_index
            if source_idx < 0:
                source_idx += num_depth_token_groups
            if source_idx < 0 or source_idx >= num_depth_token_groups:
                raise ValueError(
                    f"depth_shared_feature_index={self.depth_shared_feature_index} is out of range for "
                    f"{num_depth_token_groups} depth token groups."
                )
            return tuple(source_idx for _ in range(num_layers))
        raise ValueError(f"Unknown depth_feature_mapping_strategy={strategy!r}.")

    def _build_layouts(
        self,
        tokens_per_view: int,
        view_counts: list[int],
        image_sizes: list[list[tuple[int, int]]],
    ) -> list[list[ViewTokenLayout]]:
        hw = int(tokens_per_view**0.5)
        layouts = []
        offset = 0
        for sample_idx, count in enumerate(view_counts):
            sample_layouts = []
            start = 0
            for view_idx in range(count):
                sample_layouts.append(
                    ViewTokenLayout(
                        view_index=view_idx,
                        token_start=start,
                        token_end=start + tokens_per_view,
                        grid_hw=(hw, hw),
                        image_size=image_sizes[sample_idx][view_idx],
                    )
                )
                start += tokens_per_view
                offset += 1
            layouts.append(sample_layouts)
        return layouts

    def forward(self, batch_images: list[list[Any]], *, device=None, dtype=None) -> ExpertKVBundle:
        images, view_counts, image_sizes = self._flatten_images(batch_images)
        if device is None:
            device = next(self.parameters()).device
        images = images.to(device=device)
        depth_tokens_by_layer = self.encoder(images)
        if dtype is not None:
            depth_tokens_by_layer = tuple(tokens.to(dtype=dtype) for tokens in depth_tokens_by_layer)
        source_indices = self._resolve_depth_source_indices(len(depth_tokens_by_layer))
        depth_tokens_by_layer = tuple(depth_tokens_by_layer[idx] for idx in source_indices)
        if not depth_tokens_by_layer:
            return self.depth_token_proj((), token_layouts=None)

        batch_size = len(view_counts)
        if len(set(view_counts)) != 1:
            raise ValueError("QwenKI_depth currently expects the same number of views per batch sample.")
        views = view_counts[0]
        tokens_per_view = depth_tokens_by_layer[0].shape[1]
        grouped_tokens = []
        for tokens in depth_tokens_by_layer:
            tokens = tokens.reshape(batch_size, views, tokens.shape[1], tokens.shape[2])
            grouped_tokens.append(tokens.reshape(batch_size, views * tokens.shape[2], tokens.shape[3]))

        layouts = self._build_layouts(tokens_per_view, view_counts, image_sizes)
        return self.depth_token_proj(tuple(grouped_tokens), token_layouts=layouts)
