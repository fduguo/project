"""QwenOFT with direct future-depth JEPA supervision on the VLM backbone."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenOFT import QwenOFTDefaultConfig
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.future_depth_feedback import FrozenDA3FeatureTarget
from starVLA.model.modules.future_depth_feedback import FutureDepthPredictor
from starVLA.model.modules.future_depth_feedback import future_depth_feature_loss
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenOFTFutureDepthJEPADefaultConfig(QwenOFTDefaultConfig):
    name: str = "QwenOFT_future_depth_jepa"
    future_depth_head: dict = field(
        default_factory=lambda: {
            "depth_model_name": "./playground/Pretrained_models/da3-small",
            "num_views": 2,
            "target_feature_layer": 11,
            "target_grid_size": 8,
            "target_feature_dim": 384,
            "target_image_size": 518,
            "target_micro_batch_size": 8,
            "predictor_num_layers": 2,
            "predictor_num_heads": 8,
            "predictor_ffn_mult": 2,
            "loss_weight": 0.01,
            "cosine_loss_weight": 0.0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFT_future_depth_jepa")
class Qwenvl_OFT_future_depth_jepa(Qwenvl_OFT):
    """Train OFT with future-depth prediction as a direct auxiliary loss.

    The future-depth predictor consumes non-detached action-token states, so
    its loss updates both the predictor and the Qwen backbone. It does not feed
    predicted features back into the policy and has no gate or routing module.
    Inference therefore remains the unmodified QwenOFT policy path.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        merged_config = merge_framework_config(
            QwenOFTFutureDepthJEPADefaultConfig,
            config,
        )
        super().__init__(config=merged_config, **kwargs)

        head_cfg = self.config.framework.future_depth_head
        self.future_depth_num_views = int(head_cfg.get("num_views", 2))
        self.future_depth_target_grid_size = int(
            head_cfg.get("target_grid_size", 8)
        )
        self.future_depth_target_dim = int(head_cfg.get("target_feature_dim", 384))
        self.future_depth_loss_weight = float(head_cfg.get("loss_weight", 0.01))
        self.future_depth_cosine_weight = float(
            head_cfg.get("cosine_loss_weight", 0.0)
        )
        self.future_depth_model_name = str(head_cfg.get("depth_model_name"))
        self.future_depth_target_layer = int(
            head_cfg.get("target_feature_layer", 11)
        )
        self.future_depth_target_image_size = head_cfg.get(
            "target_image_size",
            518,
        )
        self.future_depth_target_micro_batch_size = int(
            head_cfg.get("target_micro_batch_size", 8)
        )

        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.future_depth_predictor = FutureDepthPredictor(
            hidden_size=hidden_size,
            depth_feature_dim=self.future_depth_target_dim,
            num_views=self.future_depth_num_views,
            target_grid_size=self.future_depth_target_grid_size,
            num_layers=int(head_cfg.get("predictor_num_layers", 2)),
            num_heads=int(head_cfg.get("predictor_num_heads", 8)),
            ffn_mult=int(head_cfg.get("predictor_ffn_mult", 2)),
        )

        # DA3 is a frozen, training-only target encoder. Keeping it outside the
        # module tree excludes it from optimizer groups and policy checkpoints.
        self.__dict__["_future_depth_teacher"] = None

    def _model_device(self) -> torch.device:
        return next(self.qwen_vl_interface.parameters()).device

    def _vlm_autocast(self):
        if self._model_device().type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _append_action_prompt(self, instructions: list[str]) -> list[str]:
        action_tokens = self.action_token * self.chunk_len
        suffix = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        return [instruction + suffix for instruction in instructions]

    def _get_future_depth_teacher(self) -> FrozenDA3FeatureTarget:
        teacher = self.__dict__.get("_future_depth_teacher")
        if teacher is None:
            teacher = FrozenDA3FeatureTarget(
                depth_model_name=self.future_depth_model_name,
                target_feature_layer=self.future_depth_target_layer,
                target_grid_size=self.future_depth_target_grid_size,
                expected_feature_dim=self.future_depth_target_dim,
                image_size=self.future_depth_target_image_size,
            )
            teacher.to(device=self._model_device())
            teacher.eval()
            self.__dict__["_future_depth_teacher"] = teacher
        return teacher

    def _future_depth_targets(
        self,
        examples: List[dict],
        *,
        prediction_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [
            index
            for index, example in enumerate(examples)
            if "future_image" not in example
        ]
        if missing:
            raise RuntimeError(
                "QwenOFT_future_depth_jepa requires future_image during training. "
                "Set datasets.vla_data.include_future_image=true; "
                f"missing sample indices={missing}."
            )

        valid_mask = torch.tensor(
            [bool(example.get("future_valid", True)) for example in examples],
            device=self._model_device(),
            dtype=torch.bool,
        )
        target = self._get_future_depth_teacher().encode_views(
            [example["future_image"] for example in examples],
            expected_num_views=self.future_depth_num_views,
            device=self._model_device(),
            micro_batch_size=self.future_depth_target_micro_batch_size,
        )
        expected_tokens = (
            self.future_depth_num_views * self.future_depth_target_grid_size**2
        )
        expected_shape = (
            len(examples),
            expected_tokens,
            self.future_depth_target_dim,
        )
        if tuple(target.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected future-depth target shape: "
                f"got {tuple(target.shape)}, expected {expected_shape}."
            )
        return target.to(dtype=prediction_dtype), valid_mask

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        if not examples:
            raise ValueError("examples must be a non-empty list.")

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = (
            [example["state"] for example in examples]
            if "state" in examples[0]
            else None
        )
        if state is not None:
            instructions = self.add_discretized_state_to_instruction(
                instructions,
                state,
            )
        instructions = self._append_action_prompt(instructions)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        with self._vlm_autocast():
            qwen_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        action_queries = self._gather_action_token_embeddings(
            qwen_outputs.hidden_states[-1],
            qwen_inputs["input_ids"],
            action_token_id=self.action_token_id,
        )
        pred_actions = self.action_model.predict_action(action_queries)
        action_tensor = torch.as_tensor(
            np.asarray(actions),
            device=pred_actions.device,
            dtype=pred_actions.dtype,
        )
        action_target = action_tensor[:, -self.action_horizon :, :]
        base_action_loss = self.l1_loss(
            pred_actions.float(),
            action_target.float(),
        )

        # detach_input=False is the defining JEPA path: the auxiliary feature
        # loss reaches the action-token states and continues into Qwen.
        predicted_depth = self.future_depth_predictor(
            action_queries,
            detach_input=False,
        )
        future_target, valid_mask = self._future_depth_targets(
            examples,
            prediction_dtype=predicted_depth.dtype,
        )
        future_depth_loss, depth_metrics = future_depth_feature_loss(
            predicted_depth,
            future_target,
            valid_mask=valid_mask,
            cosine_weight=self.future_depth_cosine_weight,
        )
        total_loss = (
            base_action_loss
            + self.future_depth_loss_weight * future_depth_loss
        )

        return {
            "action_loss": total_loss,
            "base_action_loss": base_action_loss.detach(),
            "future_depth_loss": future_depth_loss.detach(),
            "future_depth_loss_weight": total_loss.new_tensor(
                self.future_depth_loss_weight
            ).detach(),
            "future_depth_predictor_parameter_count": total_loss.new_tensor(
                float(
                    sum(
                        parameter.numel()
                        for parameter in self.future_depth_predictor.parameters()
                    )
                )
            ).detach(),
            **depth_metrics,
        }

