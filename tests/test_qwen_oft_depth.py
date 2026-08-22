from types import SimpleNamespace

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenOFT_depth import Qwenvl_OFT_depth
from starVLA.model.modules.action_model.ExpertMLP_ActionHeader import ExpertL1RegressionActionHead
from starVLA.model.modules.expert_kv import ExpertKVBundle
from starVLA.model.modules.expert_kv import ExpertLayerKV


class _FakeQwenInterface:
    def __init__(self):
        self.forward_calls = 0

    def build_qwenvl_inputs(self, images, instructions):
        return {
            "input_ids": torch.tensor(
                [
                    [0, 7, 7, 1],
                    [0, 2, 7, 7],
                ]
            )
        }

    def __call__(self, **kwargs):
        self.forward_calls += 1
        assert kwargs["output_hidden_states"] is True
        hidden = torch.arange(16, dtype=torch.float32).view(2, 4, 2)
        return SimpleNamespace(hidden_states=[hidden])


class _FakeActionModel:
    def __init__(self):
        self.expert_kv = None
        self.action_queries = None

    def predict_action(self, action_queries, expert_kv=None):
        self.action_queries = action_queries
        self.expert_kv = expert_kv
        return action_queries.sum(dim=-1, keepdim=True).expand(-1, -1, 7) * 0.0


class _FakeDepthProvider:
    def __init__(self):
        self.calls = []

    def __call__(self, batch_images, *, device, dtype):
        self.calls.append((batch_images, device, dtype))
        key = torch.ones(2, 1, 3, 1, device=device, dtype=dtype)
        value = torch.ones(2, 1, 3, 1, device=device, dtype=dtype)
        layer = ExpertLayerKV(key=key, value=value, head_indices=(0,), name="depth")
        return ExpertKVBundle(layers=[layer], layer_indices=(0,), token_layouts=None)


def _model(use_depth=True):
    model = Qwenvl_OFT_depth.__new__(Qwenvl_OFT_depth)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(framework=SimpleNamespace())
    model.qwen_vl_interface = _FakeQwenInterface()
    model.action_model = _FakeActionModel()
    model.action_horizon = 2
    model.chunk_len = 2
    model.action_token = "<act>"
    model.action_token_id = 7
    model.l1_loss = torch.nn.L1Loss()
    model.use_depth = use_depth
    model.depth_provider = _FakeDepthProvider() if use_depth else None
    return model


def test_expert_mlp_head_preserves_shape_and_backpropagates_to_expert_kv():
    head = ExpertL1RegressionActionHead(
        input_dim=4,
        hidden_dim=8,
        action_dim=7,
        NUM_ACTIONS_CHUNK=2,
        num_expert_layers=1,
        num_heads=2,
    )
    action_queries = torch.randn(2, 2, 4, requires_grad=True)
    key = torch.randn(2, 2, 3, 4, requires_grad=True)
    value = torch.randn(2, 2, 3, 4, requires_grad=True)
    expert_kv = ExpertKVBundle(
        layers=[ExpertLayerKV(key=key, value=value, head_indices=(1,), name="depth")],
        layer_indices=(0,),
    )

    output = head.predict_action(action_queries, expert_kv=expert_kv)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (2, 2, 7)
    assert key.grad is not None
    assert value.grad is not None


def test_qwen_oft_depth_forward_passes_expert_kv_to_action_model():
    model = _model(use_depth=True)
    examples = [
        {"image": [], "lang": "pick", "action": np.ones((3, 7), dtype=np.float32)},
        {"image": [], "lang": "place", "action": np.ones((3, 7), dtype=np.float32)},
    ]

    output = model.forward(examples)

    assert output["action_loss"].item() == 1.0
    assert model.qwen_vl_interface.forward_calls == 1
    assert model.action_model.action_queries.shape == (2, 2, 2)
    assert model.action_model.expert_kv is not None
    assert len(model.depth_provider.calls) == 1


def test_qwen_oft_depth_use_depth_false_skips_provider():
    model = _model(use_depth=False)
    examples = [
        {"image": [], "lang": "pick", "action": np.ones((3, 7), dtype=np.float32)},
        {"image": [], "lang": "place", "action": np.ones((3, 7), dtype=np.float32)},
    ]

    model.forward(examples)

    assert model.action_model.expert_kv is None
    assert model.depth_provider is None
