import json

import numpy as np
import pytest
import torch

from nanoact.datasets import Dataset


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Two short episodes with visibly different actions and nontrivial stats."""
    monkeypatch.setenv("NANOACT_DATA_DIR", str(tmp_path / "data"))
    ds = Dataset("example", "test/example", "a" * 40, frames=6)
    ds.snapshot.mkdir(parents=True)
    ds.cache.mkdir(parents=True)
    (ds.snapshot / "manifest.json").write_text(json.dumps({
        "repo_id": ds.repo_id, "revision": ds.revision, "files": {},
    }))
    stats = {
        "image": {"mean": [[[0.25]]] * 3, "std": [[[0.5]]] * 3},
        "state": {"mean": [1.0] * 14, "std": [2.0] * 14},
        "action": {"mean": [10.0] * 14, "std": [2.0] * 14},
    }
    np.save(ds.cache / "images.npy", np.full((6, 32, 32, 3), 255, np.uint8))
    np.save(ds.cache / "state.npy", np.full((6, 14), 5.0, np.float32))
    actions = np.repeat(np.array([10, 12, 14, 30, 32, 34], np.float32)[:, None], 14, axis=1)
    np.save(ds.cache / "action.npy", actions)
    np.save(ds.cache / "episode_bounds.npy", np.array([[0, 3], [3, 6]], np.int64))
    sidecar = {"repo_id": ds.repo_id, "revision": ds.revision, "frames": 6, "stats": stats}
    (ds.cache / "sidecar.json").write_text(json.dumps(sidecar))
    return ds, sidecar
