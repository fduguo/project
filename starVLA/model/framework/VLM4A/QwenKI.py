# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0.
"""
Qwen-KI Framework

Knowledge-insulated VLA training:
  - FAST action tokens train the Qwen VLM backbone with next-token CE.
  - A layer-wise flow-matching action expert predicts continuous actions.
  - The continuous action expert sees detached VLM hidden states, so its
    regression/flow loss cannot update the VLM backbone.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import (
    add_discretized_state_to_instruction,
    merge_framework_config,
    populate_layerwise_dit_cfg,
)
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model as get_flow_action_model,
)
from starVLA.model.modules.action_model.fast_ActionHeader import (
    Fast_Action_Tokenizer,
    get_action_model as get_fast_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class QwenKIDefaultConfig:
    """QwenKI default parameters.

    The ``action_model`` block configures the continuous PI-style expert.
    The ``fast_action_model`` block configures the FAST tokenizer branch.
    """

    name: str = "QwenKI"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
            "num_vl_layers": 36,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "action_dit_hidden_dim": 512,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )

    fast_action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "FAST",
            "fast_tokenizer_name": "playground/Pretrained_models/fast",
        }
    )

    action_loss_weight: float = 1.0
    fast_loss_weight: float = 1.0
    detach_action_condition: bool = True
    add_discretized_state_to_instruction: bool = True


def detach_vlm_hidden_states_for_action(
    vl_embs_list: List[torch.Tensor],
    project_layers: Optional[nn.ModuleList] = None,
    *,
    detach: bool = True,
    target_dtype: Optional[torch.dtype] = None,
    target_device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Prepare VLM hidden states for the action expert.

    Detaching happens before action-side projection. This blocks gradients to
    the VLM backbone while preserving gradients for the projector/action expert.
    """

    if project_layers is not None and len(vl_embs_list) != len(project_layers):
        raise ValueError(
            f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
            f"but project_layers has {len(project_layers)} layers."
        )

    hidden_states = [h.detach() if detach else h for h in vl_embs_list]
    if project_layers is None:
        if target_dtype is None and target_device is None:
            return hidden_states
        return [
            h.to(device=target_device or h.device, dtype=target_dtype or h.dtype)
            for h in hidden_states
        ]

    projected_hidden_states = []
    for proj, hidden in zip(project_layers, hidden_states):
        ref_tensor = next(proj.parameters(), None)
        if ref_tensor is None:
            ref_tensor = next(proj.buffers(), None)
        hidden = hidden.to(
            device=(ref_tensor.device if ref_tensor is not None else target_device or hidden.device),
            dtype=(ref_tensor.dtype if ref_tensor is not None else target_dtype or hidden.dtype),
        )
        projected_hidden_states.append(proj(hidden))
    return projected_hidden_states


def trim_vlm_hidden_states_before_action_tokens(
    vl_embs_list: List[torch.Tensor],
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Keep non-padding context tokens before the first supervised action token."""

    if not vl_embs_list:
        return vl_embs_list

    valid_token_mask = (
        torch.ones_like(labels, dtype=torch.bool) if attention_mask is None else attention_mask.to(dtype=torch.bool)
    )
    action_token_mask = labels != -100
    positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    trimmed_layers = []

    for layer_hidden in vl_embs_list:
        if layer_hidden.shape[:2] != labels.shape:
            raise ValueError(
                f"Hidden state shape {tuple(layer_hidden.shape[:2])} does not match labels shape {tuple(labels.shape)}."
            )

        samples = []
        for batch_idx in range(layer_hidden.shape[0]):
            action_positions = torch.nonzero(action_token_mask[batch_idx], as_tuple=False)
            if action_positions.numel() > 0:
                first_action_index = action_positions[0].item()
                keep_mask = valid_token_mask[batch_idx] & (positions[0] < first_action_index)
            else:
                keep_mask = valid_token_mask[batch_idx]

            sample_hidden = layer_hidden[batch_idx, keep_mask.to(device=layer_hidden.device)]
            if sample_hidden.shape[0] == 0:
                sample_hidden = layer_hidden[batch_idx, :1]
            samples.append(sample_hidden)

        trimmed_layers.append(pad_sequence(samples, batch_first=True))

    return trimmed_layers


@FRAMEWORK_REGISTRY.register("QwenKI")
@FRAMEWORK_REGISTRY.register("QwenPI_KI")
class Qwen_KI(baseframework):
    """
    Qwen VLM + FAST action-token supervision + PI-style continuous action expert.

    The returned ``action_loss`` is a weighted sum of:
      - ``continuous_action_loss``: flow-matching loss, blocked from VLM.
      - ``fast_action_loss``: FAST action-token CE, updates VLM.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenKIDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.fast_action_model: Fast_Action_Tokenizer = get_fast_action_model(config=self.config)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_flow_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )

        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_loss_weight = float(self.config.framework.get("action_loss_weight", 1.0))
        self.fast_loss_weight = float(self.config.framework.get("fast_loss_weight", 1.0))
        self.detach_action_condition = bool(self.config.framework.get("detach_action_condition", True))
        self.use_state_tokens = bool(self.config.framework.get("add_discretized_state_to_instruction", True))

        self.fast_action_model.fast_tokenizer.time_horizon = self.action_horizon
        self.fast_action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim

    def _normalize_examples(self, examples):
        if type(examples) is not list:
            return [examples]
        return examples

    def _prepare_inputs(self, examples):
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        if state is not None and self.use_state_tokens:
            instructions = add_discretized_state_to_instruction(instructions, state)
            state = None

        return batch_images, instructions, actions, state

    def _encode_vl_hidden_states(self, batch_images: List, instructions: List[str]) -> List[torch.Tensor]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return list(qwenvl_outputs.hidden_states[-self.num_action_dit_layers :])

    def _encode_vl_hidden_states_for_action(self, batch_images: List, instructions: List[str]) -> List[torch.Tensor]:
        vl_embs_list = self._encode_vl_hidden_states(batch_images, instructions)
        return detach_vlm_hidden_states_for_action(
            vl_embs_list,
            self.project_layers,
            detach=self.detach_action_condition,
            target_dtype=self._action_model_dtype(),
            target_device=self._action_model_device(),
        )

    def _action_model_dtype(self) -> Optional[torch.dtype]:
        return getattr(self.action_model, "dtype", None)

    def _action_model_device(self) -> Optional[torch.device]:
        return getattr(self.action_model, "device", None)

    def _sanitize_fast_action_loss(self, fast_loss) -> torch.Tensor:
        if fast_loss is None or torch.isnan(fast_loss).any():
            fast_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)
        return fast_loss

    def _compute_continuous_action_loss(self, vl_embs_list, actions, state) -> torch.Tensor:
        base_hidden = vl_embs_list[-1]
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 2))
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            return self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
            )

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        examples = self._normalize_examples(examples)
        batch_images, instructions, actions, state = self._prepare_inputs(examples)

        batch_fast_tokens = self.fast_action_model.encoder_action2fastoken(actions)
        vlm_action_tokens = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=vlm_action_tokens,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        vl_embs_list = list(qwenvl_outputs.hidden_states[-self.num_action_dit_layers :])
        vl_embs_list = trim_vlm_hidden_states_before_action_tokens(
            vl_embs_list,
            qwen_inputs["labels"],
            qwen_inputs.get("attention_mask", None),
        )
        vl_embs_list = detach_vlm_hidden_states_for_action(
            vl_embs_list,
            self.project_layers,
            detach=self.detach_action_condition,
            target_dtype=self._action_model_dtype(),
            target_device=self._action_model_device(),
        )
        continuous_action_loss = self._compute_continuous_action_loss(vl_embs_list, actions, state)
        fast_action_loss = self._sanitize_fast_action_loss(qwenvl_outputs.loss)
        total_loss = self.action_loss_weight * continuous_action_loss + self.fast_loss_weight * fast_action_loss

        return {
            "action_loss": total_loss,
            "continuous_action_loss": continuous_action_loss.detach(),
            "fast_action_loss": fast_action_loss.detach(),
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> dict:
        examples = self._normalize_examples(examples)
        batch_images = []
        for example in examples:
            image = to_pil_preserve(example["image"])
            if isinstance(image, tuple):
                image = list(image)
            elif not isinstance(image, list):
                image = [image]
            batch_images.append(image)
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        if state is not None and self.use_state_tokens:
            instructions = add_discretized_state_to_instruction(instructions, state)
            state = None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list = self._encode_vl_hidden_states_for_action(batch_images, instructions)
        base_hidden = vl_embs_list[-1]
        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        return "".join([f"<robot_action_{token}>" for token in tokens])


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_qwen35_0.8b_ki.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    model = Qwen_KI(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(8, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = model([sample, sample])
    print("Action Loss:", out["action_loss"].item())
