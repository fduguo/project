from types import SimpleNamespace

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenKI import (
    Qwen_KI,
    detach_vlm_hidden_states_for_action,
    trim_vlm_hidden_states_before_action_tokens,
)


def test_ki_action_path_blocks_vlm_hidden_gradients_but_keeps_projector_trainable():
    vl_hidden = torch.randn(2, 3, 4, requires_grad=True)
    projector = torch.nn.Linear(4, 5, bias=False)

    (action_condition,) = detach_vlm_hidden_states_for_action([vl_hidden], [projector])
    loss = action_condition.square().mean()
    loss.backward()

    assert vl_hidden.grad is None
    assert projector.weight.grad is not None
    assert projector.weight.grad.abs().sum() > 0


def test_ki_action_path_matches_projector_dtype_for_bfloat16_vlm_hidden_states():
    vl_hidden = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    projector = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 5))

    (action_condition,) = detach_vlm_hidden_states_for_action([vl_hidden], [projector])

    assert action_condition.dtype == projector[-1].weight.dtype


def test_trim_vlm_hidden_states_keeps_only_non_padding_context_before_action_tokens():
    hidden = torch.arange(12, dtype=torch.float32).view(2, 6, 1)
    labels = torch.tensor(
        [
            [-100, -100, -100, -100, 10, 11],
            [-100, -100, -100, 20, 21, -100],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ]
    )

    (trimmed,) = trim_vlm_hidden_states_before_action_tokens([hidden], labels, attention_mask)

    assert trimmed.shape == (2, 2, 1)
    assert trimmed[:, :, 0].tolist() == [[2.0, 3.0], [7.0, 8.0]]


def test_trim_vlm_hidden_states_pads_variable_length_contexts():
    hidden = torch.arange(12, dtype=torch.float32).view(2, 6, 1)
    labels = torch.tensor(
        [
            [-100, -100, -100, -100, -100, 10],
            [-100, -100, -100, 20, 21, -100],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ]
    )

    (trimmed,) = trim_vlm_hidden_states_before_action_tokens([hidden], labels, attention_mask)

    assert trimmed.shape == (2, 3, 1)
    assert trimmed[:, :, 0].tolist() == [[2.0, 3.0, 4.0], [7.0, 8.0, 0.0]]


def test_trimmed_solution_hidden_states_match_original_context_hidden_states():
    old_context_hidden = torch.tensor(
        [
            [[20.0], [21.0], [22.0]],
            [[30.0], [31.0], [0.0]],
        ]
    )
    solution_hidden = torch.tensor(
        [
            [[0.0], [0.0], [20.0], [21.0], [22.0], [100.0]],
            [[0.0], [30.0], [31.0], [200.0], [201.0], [0.0]],
        ]
    )
    labels = torch.tensor(
        [
            [-100, -100, -100, -100, -100, 10],
            [-100, -100, -100, 20, 21, -100],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ]
    )

    (trimmed,) = trim_vlm_hidden_states_before_action_tokens([solution_hidden], labels, attention_mask)

    torch.testing.assert_close(trimmed, old_context_hidden)


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
        return {
            "labels": self.labels,
            "attention_mask": self.attention_mask,
        }

    def __call__(self, **kwargs):
        self.forward_calls += 1
        assert kwargs["output_hidden_states"] is True
        hidden = torch.arange(12, dtype=torch.float32).view(2, 6, 1)
        return SimpleNamespace(loss=torch.tensor(2.0), hidden_states=[hidden, hidden + 100.0])


class _FakeActionModel:
    def __init__(self):
        self.vl_embs_list = None

    def __call__(self, vl_embs_list, actions, state):
        self.vl_embs_list = vl_embs_list
        return actions.mean() * 0.0 + 3.0


def test_qwen_ki_forward_runs_vlm_once_and_uses_trimmed_context_hidden_states():
    model = Qwen_KI.__new__(Qwen_KI)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(framework=SimpleNamespace(action_model={"repeated_diffusion_steps": 1}))
    model.qwen_vl_interface = _FakeQwenInterface()
    model.fast_action_model = _FakeFastActionModel()
    model.action_model = _FakeActionModel()
    model.project_layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    model.num_action_dit_layers = 2
    model.action_horizon = 2
    model.action_loss_weight = 1.0
    model.fast_loss_weight = 1.0
    model.detach_action_condition = True
    model.use_state_tokens = False

    examples = [
        {"image": [], "lang": "pick", "action": np.ones((4, 7), dtype=np.float32)},
        {"image": [], "lang": "place", "action": np.ones((4, 7), dtype=np.float32)},
    ]

    output = model.forward(examples)

    assert model.qwen_vl_interface.forward_calls == 1
    assert output["action_loss"].item() == 5.0
    assert output["continuous_action_loss"].item() == 3.0
    assert output["fast_action_loss"].item() == 2.0
    assert model.action_model.vl_embs_list[-1].shape == (2, 2, 1)
    assert model.action_model.vl_embs_list[-1][:, :, 0].tolist() == [[102.0, 103.0], [107.0, 108.0]]
