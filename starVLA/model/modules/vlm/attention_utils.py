import importlib


def require_flash_attn_for_flash_attention_2(attn_implementation: str, module_name: str) -> None:
    if attn_implementation != "flash_attention_2":
        return

    try:
        importlib.import_module("flash_attn")
    except ImportError as exc:
        raise ImportError(
            f"{module_name}: flash_attention_2 was requested, but flash_attn is not installed or cannot be imported. "
            "Install a FlashAttention wheel compatible with the current PyTorch/CUDA environment, or explicitly set "
            "framework.qwenvl.attn_implementation=sdpa if you want to run without FlashAttention."
        ) from exc
