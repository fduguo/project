from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from starVLA.model.framework.VLM4A.QwenKI_depth import Qwen_KI_depth
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit_expert import ExpertAwareAttention
from starVLA.model.modules.depth import MultiViewDepthKVProvider
from starVLA.model.modules.expert_kv import ExpertKVBundle
from starVLA.model.modules.expert_kv import ExpertLayerKV
from starVLA.model.modules.expert_kv import render_expert_visualization
from starVLA.model.modules.expert_kv.visualization import ExpertAttentionRecord


class FakeDepthEncoder(torch.nn.Module):
    def __init__(self, num_groups=2):
        super().__init__()
        self.num_groups = num_groups
        self.token_merging_model = torch.nn.Linear(3, 4)

    def forward(self, images):
        batch = images.shape[0]
        tokens = torch.arange(batch * 16 * 4, dtype=images.dtype, device=images.device).reshape(batch, 16, 4)
        return tuple(tokens + group_idx for group_idx in range(self.num_groups))


def _image():
    return Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8))


def _attention_inputs():
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    encoder = torch.randn(2, 7, 8)
    expert_k = torch.randn(2, 4, 3, 2, requires_grad=True)
    expert_v = torch.randn(2, 4, 3, 2, requires_grad=True)
    expert = ExpertLayerKV(key=expert_k, value=expert_v, head_indices=(1, 3), name="depth")
    return hidden, encoder, expert, expert_k, expert_v


def test_expert_attention_gated_residual_is_zero_init_and_records_attention():
    attn = ExpertAwareAttention(query_dim=8, heads=4, dim_head=2, cross_attention_dim=8)
    hidden, encoder, expert, _, _ = _attention_inputs()

    base_out, _ = attn(hidden, encoder_hidden_states=encoder)
    out, record = attn(
        hidden,
        encoder_hidden_states=encoder,
        expert_kv=expert,
        return_attention_record=True,
        layer_idx=2,
    )

    assert attn.expert_fusion_mode == "gated_residual"
    assert attn.expert_gate.item() == 0.0
    torch.testing.assert_close(out, base_out)
    assert record is not None
    assert record.layer_idx == 2
    assert record.head_indices == (1, 3)
    assert record.attention.shape == (2, 2, 5, 3)


def test_expert_attention_gated_residual_learns_through_gate_and_expert_kv():
    attn = ExpertAwareAttention(query_dim=8, heads=4, dim_head=2, cross_attention_dim=8)
    with torch.no_grad():
        attn.expert_gate.fill_(0.5)
    hidden, encoder, expert, expert_k, expert_v = _attention_inputs()

    base_out, _ = attn(hidden, encoder_hidden_states=encoder)
    out, _ = attn(hidden, encoder_hidden_states=encoder, expert_kv=expert)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == hidden.shape
    assert not torch.allclose(out, base_out)
    assert attn.expert_gate.grad is not None
    assert attn.expert_gate.grad.abs().sum() > 0
    assert expert_k.grad is not None
    assert expert_k.grad.abs().sum() > 0
    assert expert_v.grad is not None
    assert expert_v.grad.abs().sum() > 0


def test_expert_attention_replacement_mode_preserves_old_routing():
    attn = ExpertAwareAttention(
        query_dim=8,
        heads=4,
        dim_head=2,
        cross_attention_dim=8,
        expert_fusion_mode="replacement",
    )
    hidden, encoder, expert, expert_k, expert_v = _attention_inputs()

    base_out, _ = attn(hidden, encoder_hidden_states=encoder)
    out, record = attn(
        hidden,
        encoder_hidden_states=encoder,
        expert_kv=expert,
        return_attention_record=True,
        layer_idx=2,
    )
    loss = out.square().mean()
    loss.backward()

    assert out.shape == hidden.shape
    assert not torch.allclose(out, base_out)
    assert record is not None
    assert record.head_indices == (1, 3)
    assert expert_k.grad is not None
    assert expert_k.grad.abs().sum() > 0
    assert expert_v.grad is not None
    assert expert_v.grad.abs().sum() > 0


def test_multiview_depth_provider_concats_view_tokens_and_layouts():
    provider = MultiViewDepthKVProvider(
        depth_model_name=None,
        feature_dim=4,
        num_heads=2,
        head_dim=2,
        layer_indices=[0, 3],
        head_indices=[1],
        image_size=16,
        encoder=FakeDepthEncoder(),
    )
    bundle = provider([[_image(), _image()], [_image(), _image()]], device=torch.device("cpu"), dtype=torch.float32)

    assert bundle.layer_indices == (0, 3)
    assert len(bundle.layers) == 2
    assert bundle.layers[0].key.shape == (2, 2, 32, 2)
    assert bundle.layers[0].head_indices == (1,)
    assert bundle.token_layouts is not None
    assert len(bundle.token_layouts[0]) == 2
    assert bundle.token_layouts[0][0].token_start == 0
    assert bundle.token_layouts[0][0].token_end == 16
    assert bundle.token_layouts[0][1].token_start == 16
    assert bundle.token_layouts[0][1].token_end == 32


def _provider_for_mapping(layer_count, strategy="one_to_one", shared_index=-1):
    return MultiViewDepthKVProvider(
        depth_model_name=None,
        feature_dim=4,
        num_heads=2,
        head_dim=2,
        layer_indices=list(range(layer_count)),
        head_indices=[1],
        depth_feature_mapping_strategy=strategy,
        depth_shared_feature_index=shared_index,
        image_size=16,
        encoder=FakeDepthEncoder(num_groups=4),
    )


def test_depth_provider_feature_mapping_strategies():
    uniform = _provider_for_mapping(24, "uniform_segments")
    assert uniform._resolve_depth_source_indices(4) == (0,) * 6 + (1,) * 6 + (2,) * 6 + (3,) * 6

    cyclic = _provider_for_mapping(10, "cyclic")
    assert cyclic._resolve_depth_source_indices(4) == (0, 1, 2, 3, 0, 1, 2, 3, 0, 1)

    shared = _provider_for_mapping(8, "shared")
    assert shared._resolve_depth_source_indices(4) == (3,) * 8

    shared_specific = _provider_for_mapping(8, "shared", shared_index=1)
    assert shared_specific._resolve_depth_source_indices(4) == (1,) * 8


def test_depth_provider_one_to_one_keeps_old_strict_behavior():
    provider = MultiViewDepthKVProvider(
        depth_model_name=None,
        feature_dim=4,
        num_heads=2,
        head_dim=2,
        layer_indices=[0, 1, 2],
        head_indices=[1],
        image_size=16,
        encoder=FakeDepthEncoder(num_groups=2),
    )

    with pytest.raises(ValueError, match="one_to_one"):
        provider([[_image()]], device=torch.device("cpu"), dtype=torch.float32)


def test_depth_provider_all_layer_mapping_outputs_one_kv_per_action_layer():
    provider = _provider_for_mapping(24, "uniform_segments")
    bundle = provider([[_image()]], device=torch.device("cpu"), dtype=torch.float32)

    assert bundle.layer_indices == tuple(range(24))
    assert len(bundle.layers) == 24
    assert bundle.layers[0].key.shape == (1, 2, 16, 2)
    assert bundle.layers[-1].head_indices == (1,)


def test_expert_visualization_renders_overlay(tmp_path):
    provider = MultiViewDepthKVProvider(
        depth_model_name=None,
        feature_dim=4,
        num_heads=2,
        head_dim=2,
        layer_indices=[0],
        head_indices=[1],
        image_size=16,
        encoder=FakeDepthEncoder(),
    )
    images = [[_image(), _image()]]
    bundle = provider(images, device=torch.device("cpu"), dtype=torch.float32)
    record = ExpertAttentionRecord(
        layer_idx=0,
        head_indices=(1,),
        attention=torch.rand(1, 1, 3, 32),
        expert_name="depth",
    )
    viz = render_expert_visualization(
        images=images,
        records=[record],
        token_layouts=bundle.token_layouts,
        output_dir=tmp_path,
        max_samples=1,
    )

    assert viz.output_paths is not None
    assert len(viz.output_paths) == 2
    for path in viz.output_paths:
        assert path.endswith(".png")


class _FakeFastActionModel:
    def encoder_action2fastoken(self, actions):
        return [[1, 2] for _ in actions]


class _FakeQwenInterface:
    def __init__(self):
        self.forward_calls = 0
        self.model = SimpleNamespace(device=torch.device("cpu"))
        self.labels = torch.tensor(
            [
                [-100, -100, -100, -100, 10, 11],
                [-100, -100, -100, 20, 21, -100],
            ]
        )
        self.attention_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
            ]
        )

    def build_qwenvl_inputs(self, images, instructions, solutions=None):
        assert solutions is not None
        return {"labels": self.labels, "attention_mask": self.attention_mask}

    def __call__(self, **kwargs):
        self.forward_calls += 1
        assert kwargs["output_hidden_states"] is True
        hidden = torch.arange(12, dtype=torch.float32).view(2, 6, 1)
        return SimpleNamespace(loss=torch.tensor(2.0), hidden_states=[hidden, hidden + 100.0])


class _FakeExpertActionModel:
    def __init__(self):
        self.vl_embs_list = None
        self.expert_kv = None

    def __call__(self, vl_embs_list, actions, state, expert_kv=None):
        self.vl_embs_list = vl_embs_list
        self.expert_kv = expert_kv
        return actions.mean() * 0.0 + 3.0


class _FakeDepthProvider:
    def __init__(self):
        self.calls = []

    def __call__(self, batch_images, *, device, dtype):
        self.calls.append((batch_images, device, dtype))
        key = torch.ones(2, 1, 1, 1, device=device, dtype=dtype)
        value = torch.ones(2, 1, 1, 1, device=device, dtype=dtype)
        layer = ExpertLayerKV(key=key, value=value, head_indices=(0,), name="depth")
        return ExpertKVBundle(layers=(layer,), layer_indices=(0,), token_layouts=None)


def test_qwen_ki_depth_forward_runs_vlm_once_and_uses_trimmed_context_hidden_states():
    model = Qwen_KI_depth.__new__(Qwen_KI_depth)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(framework=SimpleNamespace(action_model={"repeated_diffusion_steps": 1}))
    model.qwen_vl_interface = _FakeQwenInterface()
    model.fast_action_model = _FakeFastActionModel()
    model.expert_action_model = _FakeExpertActionModel()
    model.project_layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    model.num_action_dit_layers = 2
    model.action_horizon = 2
    model.action_loss_weight = 1.0
    model.fast_loss_weight = 1.0
    model.detach_action_condition = True
    model.use_state_tokens = False
    model.use_depth = True
    model.depth_provider = _FakeDepthProvider()

    examples = [
        {"image": [], "lang": "pick", "action": np.ones((4, 7), dtype=np.float32)},
        {"image": [], "lang": "place", "action": np.ones((4, 7), dtype=np.float32)},
    ]

    output = model.forward(examples)

    assert model.qwen_vl_interface.forward_calls == 1
    assert output["action_loss"].item() == 5.0
    assert output["continuous_action_loss"].item() == 3.0
    assert output["fast_action_loss"].item() == 2.0
    assert model.expert_action_model.vl_embs_list[-1].shape == (2, 2, 1)
    assert model.expert_action_model.vl_embs_list[-1][:, :, 0].tolist() == [[102.0, 103.0], [107.0, 108.0]]
    assert model.expert_action_model.expert_kv is not None
    assert len(model.depth_provider.calls) == 1
