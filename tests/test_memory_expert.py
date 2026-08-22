from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenKI_memory import Qwen_KI_memory
from starVLA.model.framework.VLM4A.QwenOFT_memory import Qwenvl_OFT_memory
from starVLA.model.modules.expert_kv import ExpertKVBundle, ExpertLayerKV
from starVLA.model.modules.memory import EpisodeMemoryBank, MemoryExpertKVProvider


def test_episode_memory_bank_reset_retrieves_and_fifo_limits():
    bank = EpisodeMemoryBank(
        hidden_size=4,
        memory_length=2,
        retrieval_layers=0,
        fusion_type="add",
        consolidate_type="fifo",
        use_timestep_pe=False,
    )

    first = torch.ones(1, 3, 4)
    out = bank(first, episode_ids=["ep"], timesteps=[0], is_first_step=[True])
    torch.testing.assert_close(out, first)
    assert len(bank.bank["ep"]) == 1

    bank(first * 2, episode_ids=["ep"], timesteps=[1], is_first_step=[False])
    bank(first * 3, episode_ids=["ep"], timesteps=[2], is_first_step=[False])
    assert len(bank.bank["ep"]) == 2
    assert [step for step, _ in bank.bank["ep"]] == [1, 2]

    bank(first * 4, episode_ids=["ep"], timesteps=[0], is_first_step=[True])
    assert len(bank.bank["ep"]) == 1
    assert bank.bank["ep"][0][0] == 0


def test_episode_memory_bank_tome_limits_length():
    bank = EpisodeMemoryBank(
        hidden_size=4,
        memory_length=2,
        retrieval_layers=0,
        fusion_type="add",
        consolidate_type="tome",
        use_timestep_pe=False,
    )

    for timestep in range(4):
        bank(
            torch.full((1, 2, 4), float(timestep)),
            episode_ids=["ep"],
            timesteps=[timestep],
            is_first_step=[timestep == 0],
        )
    assert len(bank.bank["ep"]) == 2


def test_episode_memory_bank_training_stream_clears_previous_episode():
    bank = EpisodeMemoryBank(
        hidden_size=4,
        memory_length=2,
        retrieval_layers=0,
        fusion_type="add",
        consolidate_type="fifo",
        use_timestep_pe=False,
    )
    bank.train()

    tokens = torch.ones(1, 2, 4)
    bank(tokens, episode_ids=["ep0"], timesteps=[0], is_first_step=[True])
    assert "ep0" in bank.bank

    bank(tokens, episode_ids=["ep1"], timesteps=[0], is_first_step=[True])
    assert "ep0" not in bank.bank
    assert "ep1" in bank.bank


def test_memory_expert_kv_provider_outputs_ki_and_oft_shapes():
    ki_provider = MemoryExpertKVProvider(
        input_dim=8,
        memory_dim=8,
        num_heads=2,
        head_dim=4,
        layer_indices=[0, 2],
        head_indices=[1],
        memory_length=2,
        retrieval_layers=0,
        use_timestep_pe=False,
    )
    tokens = torch.randn(2, 5, 8)
    bundle = ki_provider(tokens, episode_ids=["a", "b"], timesteps=[0, 0], is_first_step=[True, True])
    assert bundle.provider_name == "memory"
    assert bundle.layer_indices == (0, 2)
    assert len(bundle.layers) == 2
    assert bundle.layers[0].key.shape == (2, 2, 5, 4)
    assert bundle.layers[0].value.shape == (2, 2, 5, 4)
    assert bundle.layers[0].head_indices == (1,)

    oft_provider = MemoryExpertKVProvider(
        input_dim=8,
        memory_dim=16,
        num_heads=4,
        head_dim=4,
        layer_indices=[0, 1, 2],
        head_indices=[2, 3],
        memory_length=2,
        retrieval_layers=0,
        use_timestep_pe=False,
    )
    bundle = oft_provider(tokens, episode_ids=["a", "b"], timesteps=[1, 1], is_first_step=[False, False])
    assert len(bundle.layers) == 3
    assert bundle.layers[0].key.shape == (2, 4, 5, 4)
    assert bundle.layers[0].value.shape == (2, 4, 5, 4)
    assert bundle.layers[0].head_indices == (2, 3)


class _FakeFastActionModel:
    def encoder_action2fastoken(self, actions):
        return [[1, 2] for _ in actions]


class _FakeQwenKIInterface:
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


class _FakeMemoryProvider:
    def __init__(self):
        self.calls = []

    def __call__(self, memory_tokens, *, episode_ids, timesteps, is_first_step):
        self.calls.append((memory_tokens.detach().clone(), list(episode_ids), list(timesteps), list(is_first_step)))
        key = torch.ones(memory_tokens.shape[0], 1, 1, 1, device=memory_tokens.device, dtype=memory_tokens.dtype)
        value = torch.ones_like(key)
        layer = ExpertLayerKV(key=key, value=value, head_indices=(0,), name="memory")
        return ExpertKVBundle(layers=[layer], layer_indices=(0,), token_layouts=None)


def test_qwen_ki_memory_forward_passes_trimmed_context_to_memory_provider():
    model = Qwen_KI_memory.__new__(Qwen_KI_memory)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(framework=SimpleNamespace(action_model={"repeated_diffusion_steps": 1}))
    model.qwen_vl_interface = _FakeQwenKIInterface()
    model.fast_action_model = _FakeFastActionModel()
    model.expert_action_model = _FakeExpertActionModel()
    model.project_layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    model.num_action_dit_layers = 2
    model.action_horizon = 2
    model.action_loss_weight = 1.0
    model.fast_loss_weight = 1.0
    model.detach_action_condition = True
    model.use_state_tokens = False
    model.use_memory = True
    model.memory_provider = _FakeMemoryProvider()

    examples = [
        {
            "image": [],
            "lang": "pick",
            "action": np.ones((4, 7), dtype=np.float32),
            "episode_id": "ds:1",
            "timestep": 0,
            "is_first_step": True,
        },
        {
            "image": [],
            "lang": "place",
            "action": np.ones((4, 7), dtype=np.float32),
            "episode_id": "ds:1",
            "timestep": 1,
            "is_first_step": False,
        },
    ]

    output = model.forward(examples)

    assert model.qwen_vl_interface.forward_calls == 1
    assert output["action_loss"].item() == 5.0
    assert model.expert_action_model.expert_kv is not None
    memory_tokens, episode_ids, timesteps, is_first_step = model.memory_provider.calls[0]
    assert memory_tokens.shape == (2, 2, 1)
    assert memory_tokens[:, :, 0].tolist() == [[102.0, 103.0], [107.0, 108.0]]
    assert episode_ids == ["ds:1", "ds:1"]
    assert timesteps == [0, 1]
    assert is_first_step == [True, False]


class _FakeQwenOFTInterface:
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


class _FakeOFTActionModel:
    def __init__(self):
        self.expert_kv = None
        self.action_queries = None

    def predict_action(self, action_queries, expert_kv=None):
        self.action_queries = action_queries
        self.expert_kv = expert_kv
        return action_queries.sum(dim=-1, keepdim=True).expand(-1, -1, 7) * 0.0


def _oft_memory_model(use_memory=True):
    model = Qwenvl_OFT_memory.__new__(Qwenvl_OFT_memory)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(framework=SimpleNamespace())
    model.qwen_vl_interface = _FakeQwenOFTInterface()
    model.action_model = _FakeOFTActionModel()
    model.action_horizon = 2
    model.chunk_len = 2
    model.action_token = "<act>"
    model.action_token_id = 7
    model.l1_loss = torch.nn.L1Loss()
    model.use_memory = use_memory
    model.memory_provider = _FakeMemoryProvider() if use_memory else None
    return model


def test_qwen_oft_memory_forward_passes_context_memory_kv_to_action_model():
    model = _oft_memory_model(use_memory=True)
    examples = [
        {
            "image": [],
            "lang": "pick",
            "action": np.ones((3, 7), dtype=np.float32),
            "episode_id": "ds:2",
            "timestep": 0,
            "is_first_step": True,
        },
        {
            "image": [],
            "lang": "place",
            "action": np.ones((3, 7), dtype=np.float32),
            "episode_id": "ds:2",
            "timestep": 1,
            "is_first_step": False,
        },
    ]

    output = model.forward(examples)

    assert output["action_loss"].item() == 1.0
    assert model.qwen_vl_interface.forward_calls == 1
    assert model.action_model.action_queries.shape == (2, 2, 2)
    assert model.action_model.expert_kv is not None
    memory_tokens, episode_ids, timesteps, is_first_step = model.memory_provider.calls[0]
    assert memory_tokens.shape == (2, 2, 2)
    assert episode_ids == ["ds:2", "ds:2"]
    assert timesteps == [0, 1]
    assert is_first_step == [True, False]


def test_lerobot_memory_metadata_and_sequential_sampling_code_paths_exist():
    source = Path("starVLA/dataloader/gr00t_lerobot/datasets.py").read_text()

    assert "sample[\"episode_id\"] = f\"{self.dataset_name}:{int(trajectory_id)}\"" in source
    assert "sample[\"timestep\"] = int(base_index)" in source
    assert "sample[\"is_first_step\"] = int(base_index) == 0" in source
    assert "rng.shuffle(episode_order)" in source
    assert "for base_index in range(trajectory_length)" in source
    assert "return self._sequential_steps[index % len(self._sequential_steps)]" in source
