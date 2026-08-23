from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.VLM4A.QwenOFT_backbone_action_mha import (
    QwenOFTBackboneActionMHADefaultConfig,
    Qwenvl_OFT_backbone_action_mha,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.future_vit import CrossAttnFutureHead
from starVLA.model.modules.future_vit import JointSelfAttnFutureHead
from starVLA.model.modules.future_vit import EMATargetViT
from starVLA.model.modules.future_vit import LayerFusion
from starVLA.model.modules.future_vit import future_vit_loss
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenOFTFutureViTDefaultConfig(QwenOFTBackboneActionMHADefaultConfig):
    name: str = "QwenOFT_future_vit"
    backbone_mha_layer_indices: list[int] = field(default_factory=list)
    num_backbone_mha_heads: int = 0
    reference_expert_token_count: str | int = 1
    future_vit_head: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "loss_weight": 0.01,
            "loss_weight_schedule": "constant",
            "loss_weight_end": 0.0,
            "loss_weight_decay_end_step": 0,
            "pred_head_type": "cross_attn",
            "num_layers_attn": 1,
            "num_heads": 8,
            "ffn_mult": 2,
            "target_views": "all",
            "target_tokens": None,
            "layer_indices": [-1],
            "layer_fusion": "none",
            "source_tokens": "action",
            "pred_head_use_positional_embedding": False,
            "pred_head_use_self_attention": False,
            "pred_head_use_output_layernorm": False,
            "joint_num_layers_attn": 1,
            "joint_num_heads": 8,
            "joint_ffn_mult": 2,
            "ema_momentum": 0.999,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFT_future_vit")
class Qwenvl_OFT_future_vit(Qwenvl_OFT_backbone_action_mha):
    """QwenOFT backbone-action-MHA baseline with a training-only future ViT auxiliary head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        merged_config = merge_framework_config(QwenOFTFutureViTDefaultConfig, config)
        super().__init__(config=merged_config, **kwargs)

        fh = self.config.framework.future_vit_head
        self.fh_enabled = bool(fh.get("enabled", True))
        self.fh_lambda = float(fh.get("loss_weight", 0.01))
        self.fh_loss_weight_start = self.fh_lambda
        self.fh_loss_weight_schedule = str(fh.get("loss_weight_schedule", "constant"))
        self.fh_loss_weight_end = float(fh.get("loss_weight_end", 0.0))
        self.fh_loss_weight_decay_end_step = int(fh.get("loss_weight_decay_end_step", 0))
        self.fh_grad_accum_steps = max(1, int(self._config_get("trainer.gradient_accumulation_steps", 1) or 1))
        self.register_buffer("_fh_forward_calls", torch.zeros((), dtype=torch.long), persistent=True)
        self.fh_type = str(fh.get("pred_head_type", "cross_attn"))
        self.fh_layers = tuple(int(i) for i in fh.get("layer_indices", [-1]))
        self.fh_fusion = str(fh.get("layer_fusion", "none"))
        self.fh_source_tokens = str(fh.get("source_tokens", "action"))
        self.fh_views = str(fh.get("target_views", "all"))
        self.fh_num_layers_attn = int(fh.get("num_layers_attn", 1))
        self.fh_num_heads = int(fh.get("num_heads", 8))
        self.fh_ffn_mult = int(fh.get("ffn_mult", 2))
        self.fh_use_pos_embed = bool(fh.get("pred_head_use_positional_embedding", False))
        self.fh_use_self_attn = bool(fh.get("pred_head_use_self_attention", False))
        self.fh_use_output_ln = bool(fh.get("pred_head_use_output_layernorm", False))
        self.fh_joint_num_layers_attn = int(fh.get("joint_num_layers_attn", 1))
        self.fh_joint_num_heads = int(fh.get("joint_num_heads", 8))
        self.fh_joint_ffn_mult = int(fh.get("joint_ffn_mult", 2))
        if self.fh_type not in ("cross_attn", "joint_self_attn"):
            raise ValueError(
                "future_vit_head.pred_head_type must be one of: cross_attn, joint_self_attn; "
                f"got {self.fh_type!r}."
            )
        if self.fh_source_tokens not in ("action", "image", "action_image"):
            raise ValueError(
                "future_vit_head.source_tokens must be one of: action, image, action_image; "
                f"got {self.fh_source_tokens!r}."
            )
        if self.fh_type == "joint_self_attn" and self.fh_source_tokens != "action_image":
            raise ValueError(
                "pred_head_type='joint_self_attn' requires future_vit_head.source_tokens='action_image'; "
                f"got source_tokens={self.fh_source_tokens!r}."
            )
        if self.fh_views != "all":
            raise ValueError("First implementation supports only future_vit_head.target_views='all'.")

        online_visual = self._online_visual()
        self.ema = EMATargetViT(
            online_visual,
            momentum=float(fh.get("ema_momentum", 0.999)),
        )
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.fh_target_tokens = self._resolve_future_target_tokens(fh)
        if self.fh_type == "joint_self_attn":
            num_views, tokens_per_view = self._infer_future_view_layout(self._future_view_count())
            pred_head = JointSelfAttnFutureHead(
                hidden_size=hidden_size,
                image_tokens=self.fh_target_tokens,
                action_tokens=self.action_horizon,
                num_views=num_views,
                tokens_per_view=tokens_per_view,
                num_layers_attn=self.fh_joint_num_layers_attn,
                num_heads=self.fh_joint_num_heads,
                ffn_mult=self.fh_joint_ffn_mult,
            )
        else:
            if self.fh_use_pos_embed:
                pos_num_views, pos_tokens_per_view = self._infer_future_view_layout(self._future_view_count())
            else:
                pos_num_views, pos_tokens_per_view = None, None
            pred_head = CrossAttnFutureHead(
                hidden_size=hidden_size,
                target_tokens=self.fh_target_tokens,
                num_layers_attn=self.fh_num_layers_attn,
                num_heads=self.fh_num_heads,
                ffn_mult=self.fh_ffn_mult,
                use_positional_embedding=self.fh_use_pos_embed,
                num_views=pos_num_views,
                tokens_per_view=pos_tokens_per_view,
                use_self_attention=self.fh_use_self_attn,
                use_output_layernorm=self.fh_use_output_ln,
            )
        self.future_vit_head = torch.nn.ModuleDict(
            {
                "layer_fusion": LayerFusion(
                    hidden_size=hidden_size,
                    num_layers=len(self.fh_layers),
                    mode=self.fh_fusion,
                ),
                "pred_head": pred_head,
            }
        )
        self._ema_inited = False

    def _config_get(self, dotted_path: str, default=None):
        value = self.config
        for part in dotted_path.split("."):
            try:
                value = getattr(value, part)
            except AttributeError:
                return default
            except KeyError:
                return default
        return value

    def _online_visual(self) -> torch.nn.Module:
        model = self.qwen_vl_interface.model
        if hasattr(model, "visual"):
            return model.visual
        inner_model = getattr(model, "model", None)
        if inner_model is not None and hasattr(inner_model, "visual"):
            return inner_model.visual
        raise AttributeError("Qwen model does not expose a visual module at `.visual` or `.model.visual`.")

    def _future_view_count(self) -> int:
        configured_num_views = self._config_get("framework.future_vit_head.num_views")
        if configured_num_views is not None:
            return int(configured_num_views)

        video_keys = self._config_get("datasets.vla_data.video_keys")
        if video_keys is not None:
            return len(video_keys)

        data_mix = self._config_get("datasets.vla_data.data_mix")
        if data_mix is None:
            raise RuntimeError("Cannot infer future ViT num_views; set future_vit_head.target_tokens or num_views.")

        from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP

        mixture_spec = DATASET_NAMED_MIXTURES.get(str(data_mix), None)
        if not mixture_spec:
            raise RuntimeError(f"Cannot infer future ViT num_views from unknown data_mix={data_mix!r}.")

        view_counts = set()
        for _, _, robot_type in mixture_spec:
            data_config = ROBOT_TYPE_CONFIG_MAP.get(robot_type, None)
            video_keys_for_robot = getattr(data_config, "video_keys", None)
            if video_keys_for_robot is not None:
                view_counts.add(len(video_keys_for_robot))
        if len(view_counts) == 1:
            return view_counts.pop()
        raise RuntimeError(
            f"Cannot infer a single future ViT num_views from data_mix={data_mix!r}; "
            "set future_vit_head.target_tokens or num_views."
        )

    @torch.no_grad()
    def _infer_future_target_tokens(self, num_views: int) -> int:
        if num_views <= 0:
            raise ValueError("future_vit_head.num_views must be positive.")
        dummy_images = [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(num_views)]
        inputs = self._future_image_processor_inputs([dummy_images])
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is None:
            raise RuntimeError("Qwen image_processor must return image_grid_thw for dummy future images.")
        visual = self._online_visual()
        spatial_merge_size = int(getattr(visual, "spatial_merge_size", visual.config.spatial_merge_size))
        image_grid_thw = image_grid_thw.to(dtype=torch.long)
        target_tokens = (image_grid_thw.prod(dim=-1) // (spatial_merge_size**2)).sum().item()
        return int(target_tokens)

    @torch.no_grad()
    def _infer_future_view_layout(self, num_views: int) -> tuple[int, int]:
        if num_views <= 0:
            raise ValueError("num_views must be positive.")
        dummy_images = [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(num_views)]
        inputs = self._future_image_processor_inputs([dummy_images])
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is None:
            raise RuntimeError("Qwen image_processor must return image_grid_thw for dummy future images.")
        visual = self._online_visual()
        spatial_merge_size = int(getattr(visual, "spatial_merge_size", visual.config.spatial_merge_size))
        image_grid_thw = image_grid_thw.to(dtype=torch.long)
        per_view_tokens = (image_grid_thw.prod(dim=-1) // (spatial_merge_size**2)).tolist()
        if len(per_view_tokens) != num_views:
            raise RuntimeError(
                f"Expected {num_views} future views from image_processor, got {len(per_view_tokens)}."
            )
        if len(set(per_view_tokens)) != 1:
            raise RuntimeError(
                "Positional embedding requires uniform tokens-per-view across views; "
                f"got {per_view_tokens}."
            )
        return num_views, int(per_view_tokens[0])

    def _resolve_future_target_tokens(self, future_head_config) -> int:
        configured_target_tokens = future_head_config.get("target_tokens", None)
        if configured_target_tokens not in (None, "", "none", "None", "null"):
            target_tokens = int(configured_target_tokens)
        else:
            target_tokens = self._infer_future_target_tokens(self._future_view_count())
        if target_tokens <= 0:
            raise ValueError("future_vit_head.target_tokens must be positive.")
        return target_tokens

    def _maybe_update_ema(self) -> None:
        if not self.fh_enabled or not self.training:
            return
        if self._ema_inited:
            self.ema.update(self._online_visual())
        self._ema_inited = True

    def _current_loss_weight(self) -> float:
        if self.fh_loss_weight_schedule == "constant":
            return self.fh_loss_weight_start
        if self.fh_loss_weight_schedule != "linear_decay":
            raise ValueError(
                "future_vit_head.loss_weight_schedule must be one of: constant, linear_decay; "
                f"got {self.fh_loss_weight_schedule!r}."
            )

        if self.fh_loss_weight_decay_end_step <= 0:
            raise ValueError(
                "future_vit_head.loss_weight_decay_end_step must be > 0 "
                "when loss_weight_schedule='linear_decay'."
            )

        opt_step = int(self._fh_forward_calls.detach().item()) // self.fh_grad_accum_steps
        progress = min(1.0, float(opt_step) / float(self.fh_loss_weight_decay_end_step))
        return self.fh_loss_weight_start + (self.fh_loss_weight_end - self.fh_loss_weight_start) * progress

    def _future_image_processor_inputs(self, future_images: list[list]) -> dict:
        flat_images = []
        for sample_images in future_images:
            if isinstance(sample_images, tuple):
                sample_images = list(sample_images)
            elif not isinstance(sample_images, list):
                sample_images = [sample_images]
            flat_images.extend(sample_images)
        inputs = self.qwen_vl_interface.processor.image_processor(images=flat_images, return_tensors="pt")
        return {key: value.to(self._model_device()) if hasattr(value, "to") else value for key, value in inputs.items()}

    def _future_vit_targets(self, examples: List[dict], *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [idx for idx, example in enumerate(examples) if "future_image" not in example]
        if missing:
            raise RuntimeError(
                "QwenOFT_future_vit requires future_image in training samples. "
                "Enable datasets.vla_data.include_future_image=true; "
                f"missing indices={missing}."
            )
        future_images = [example["future_image"] for example in examples]
        valid_mask = torch.tensor(
            [bool(example.get("future_valid", True)) for example in examples],
            device=self._model_device(),
            dtype=torch.bool,
        )
        inputs = self._future_image_processor_inputs(future_images)
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError("Qwen image_processor must return pixel_values and image_grid_thw for future images.")
        target = self.ema.encode(pixel_values, image_grid_thw).to(device=self._model_device(), dtype=dtype)
        batch_size = len(examples)
        if target.shape[0] % batch_size != 0:
            raise RuntimeError(
                f"Future ViT target token count is not divisible by batch size: {tuple(target.shape)} for B={batch_size}."
            )
        target = target.reshape(batch_size, -1, target.shape[-1])
        return target, valid_mask

    def _gather_image_token_embeddings(self, last_hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        image_token_id = int(self.qwen_vl_interface.model.config.image_token_id)
        device = input_ids.device
        batch_size, seq_len, hidden_size = last_hidden.shape

        mask = input_ids == image_token_id
        counts = mask.sum(dim=1)
        if int(counts.min().item()) <= 0:
            raise RuntimeError(f"Samples with no image token are not supported: counts={counts.tolist()}.")
        if int(counts.min().item()) != int(counts.max().item()):
            raise RuntimeError(
                "Batch has inconsistent image token count: "
                f"min={int(counts.min().item())}, max={int(counts.max().item())}; "
                "current implementation requires consistent views/resolution within a batch."
            )

        image_token_count = int(counts.min().item())
        idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        selected_pos = masked_pos.topk(k=image_token_count, dim=-1).values.sort(dim=-1).values
        gather_index = selected_pos.unsqueeze(-1).expand(-1, -1, hidden_size)
        return last_hidden.gather(dim=1, index=gather_index)

    def _future_head_inputs(self, hidden_states: tuple[torch.Tensor, ...], input_ids: torch.Tensor) -> torch.Tensor:
        def gather_one(layer_idx: int, source: str) -> torch.Tensor:
            hidden = hidden_states[layer_idx]
            if source == "action":
                return self._gather_action_token_embeddings(
                    hidden,
                    input_ids,
                    action_token_id=self.action_token_id,
                )
            return self._gather_image_token_embeddings(hidden, input_ids)

        features = []
        for layer_idx in self.fh_layers:
            if self.fh_source_tokens == "action":
                layer_features = gather_one(layer_idx, "action")
            elif self.fh_source_tokens == "image":
                layer_features = gather_one(layer_idx, "image")
            else:
                image_features = gather_one(layer_idx, "image")
                action_features = gather_one(layer_idx, "action")
                layer_features = torch.cat([image_features, action_features], dim=1)
            features.append(layer_features)
        return self.future_vit_head["layer_fusion"](features)

    def _compute_future_vit_loss(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        qwen_inputs: dict,
        examples: List[dict],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        target, valid_mask = self._future_vit_targets(examples, dtype=dtype)
        action_hidden = self._future_head_inputs(hidden_states, qwen_inputs["input_ids"])
        if int(target.shape[1]) != self.fh_target_tokens:
            raise RuntimeError(
                f"Future ViT target token count mismatch: config/inferred target_tokens={self.fh_target_tokens}, "
                f"got {int(target.shape[1])}."
            )
        pred = self.future_vit_head["pred_head"](action_hidden.to(device=target.device, dtype=target.dtype))
        return future_vit_loss(pred, target, valid_mask=valid_mask)

    def _encode_hidden_states(self, batch_images, instructions, context_token_counts: dict[int, int]):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        outputs = self.qwen_vl_interface(
            **qwen_inputs,
            action_backbone_input_ids=qwen_inputs.get("input_ids", None),
            action_backbone_token_id=self.action_token_id,
            action_backbone_attention_mask=qwen_inputs.get("attention_mask", None),
            action_backbone_context_token_counts=context_token_counts,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return tuple(outputs.hidden_states), qwen_inputs

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        examples = self._normalize_examples(examples)
        self._maybe_update_ema()
        if self.training:
            self._fh_forward_calls.add_(1)
        batch_images, instructions, actions = self._prepare_training_inputs(examples)
        instructions = self._append_action_prompt(instructions)

        context_token_counts = self._reference_token_counts(
            examples,
            batch_images,
            device=self._model_device(),
            dtype=self._model_dtype(),
        )
        self._log_attention_flops_once()
        hidden_states, qwen_inputs = self._encode_hidden_states(batch_images, instructions, context_token_counts)
        last_hidden = hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden,
                qwen_inputs["input_ids"],
                action_token_id=self.action_token_id,
            )
            pred_actions = self.action_model.predict_action(action_queries)
            actions = torch.tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
            base_action_loss = self.l1_loss(pred_actions, actions[:, -self.action_horizon :, :])
            if self.fh_enabled:
                floss = self._compute_future_vit_loss(
                    hidden_states,
                    qwen_inputs,
                    examples,
                    dtype=pred_actions.dtype,
                )
                loss_weight = self._current_loss_weight()
                loss_weight_tensor = base_action_loss.new_tensor(loss_weight)
                total_loss = base_action_loss + loss_weight_tensor * floss
            else:
                floss = torch.zeros((), device=base_action_loss.device, dtype=base_action_loss.dtype)
                loss_weight_tensor = base_action_loss.new_tensor(0.0)
                total_loss = base_action_loss

        return {
            "action_loss": total_loss,
            "base_action_loss": base_action_loss.detach(),
            "future_vit_loss": floss.detach(),
            "future_vit_loss_weight": loss_weight_tensor.detach(),
        }