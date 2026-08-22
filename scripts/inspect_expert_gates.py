#!/usr/bin/env python3
"""Print expert gate values from a starVLA checkpoint without building the model."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import torch


_STEP_CKPT_RE = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")


def _checkpoint_step(path: Path) -> int:
    match = _STEP_CKPT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def _resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")

    candidates: list[Path] = []
    for pattern in (
        "steps_*_pytorch_model.pt",
        "steps_*_model.safetensors",
        "checkpoints/steps_*_pytorch_model.pt",
        "checkpoints/steps_*_model.safetensors",
        "pytorch_model.pt",
        "model.safetensors",
        "final_model/pytorch_model.pt",
        "final_model/model.safetensors",
    ):
        candidates.extend(path.glob(pattern))

    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint file found under: {path}")
    return max(candidates, key=lambda item: (_checkpoint_step(item), item.stat().st_mtime))


def _unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"expected checkpoint dict, got {type(obj).__name__}")
    for key in ("state_dict", "model", "module"):
        value = obj.get(key)
        if isinstance(value, dict):
            return _unwrap_state_dict(value)
    return obj


def _load_gate_tensors(path: Path, pattern: str) -> dict[str, torch.Tensor]:
    regex = re.compile(pattern)
    if path.name.endswith(".safetensors"):
        from safetensors import safe_open

        gates = {}
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if regex.search(key):
                    gates[key] = checkpoint.get_tensor(key)
        return gates

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    return {key: value for key, value in state_dict.items() if regex.search(key)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint file, checkpoint directory, or experiment directory containing checkpoints/.",
    )
    parser.add_argument(
        "--pattern",
        default=r"(^|\.)(expert_gate)$",
        help="Regex matched against state_dict keys. Default prints cross-attention expert_gate parameters.",
    )
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    gates = _load_gate_tensors(checkpoint_path, args.pattern)
    print(f"checkpoint: {checkpoint_path}")
    if not gates:
        print(f"no gate tensors matched pattern: {args.pattern}")
        return

    for key in sorted(gates):
        value = gates[key].detach().float().reshape(-1)
        tanh_value = torch.tanh(value)
        if value.numel() == 1:
            print(f"{key}: raw={value.item():.8f} tanh={tanh_value.item():.8f}")
        else:
            raw = ", ".join(f"{item:.8f}" for item in value.tolist())
            tanh = ", ".join(f"{item:.8f}" for item in tanh_value.tolist())
            print(f"{key}: raw=[{raw}] tanh=[{tanh}]")


if __name__ == "__main__":
    main()
