import pytest

from starVLA.model.modules.vlm.attention_utils import (
    require_flash_attn_for_flash_attention_2,
)


def test_flash_attention_2_missing_flash_attn_raises_import_error(monkeypatch):
    def fake_import_module(name):
        if name == "flash_attn":
            raise ImportError("missing flash_attn")
        raise AssertionError(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    with pytest.raises(ImportError, match="flash_attention_2 was requested"):
        require_flash_attn_for_flash_attention_2("flash_attention_2", module_name="QWen3_5")


def test_non_flash_attention_2_does_not_import_flash_attn(monkeypatch):
    def fail_import(name):
        raise AssertionError(name)

    monkeypatch.setattr("importlib.import_module", fail_import)

    require_flash_attn_for_flash_attention_2("sdpa", module_name="QWen3_5")
