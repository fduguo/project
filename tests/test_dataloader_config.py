import sys
import types

from omegaconf import OmegaConf
from torch.utils.data import Dataset


class FakeDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"index": index}

    def save_dataset_statistics(self, path):
        raise AssertionError("rank 1 should not save statistics")


def test_vla_dataloader_uses_yaml_options(monkeypatch, tmp_path):
    fake_lerobot = types.ModuleType("starVLA.dataloader.lerobot_datasets")
    fake_lerobot.get_vla_dataset = lambda data_cfg: FakeDataset()
    fake_lerobot.collate_fn = lambda batch: batch
    monkeypatch.setitem(sys.modules, "starVLA.dataloader.lerobot_datasets", fake_lerobot)

    import starVLA.dataloader as dataloader_module

    monkeypatch.setattr(dataloader_module.dist, "get_rank", lambda: 1)

    cfg = OmegaConf.create({
        "output_dir": str(tmp_path),
        "datasets": {
            "vla_data": {
                "per_device_batch_size": 3,
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 8,
                "drop_last": True,
            }
        },
    })

    loader = dataloader_module.build_dataloader(cfg, dataset_py="lerobot_datasets")

    assert loader.batch_size == 3
    assert loader.num_workers == 2
    assert loader.pin_memory is True
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 8
    assert loader.drop_last is True
