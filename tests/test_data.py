import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from nanoact.data import Batches, load_sidecar


def test_episode_boundary_padding_and_normalization(cache):
    ds, _ = cache
    batches = Batches(ds, chunk_size=4, batch_size=2, seed=0, device="cpu")
    image, state, action, is_pad = batches._batch(np.array([1, 4]))
    torch.testing.assert_close(image, torch.full_like(image, 1.5))
    torch.testing.assert_close(state, torch.full_like(state, 2.0))
    torch.testing.assert_close(action[:, :, 0], torch.tensor([[1., 2., 2., 2.], [11., 12., 12., 12.]]))
    assert is_pad.tolist() == [[False, False, True, True], [False, False, True, True]]


@pytest.mark.parametrize("field,value", [("repo_id", "test/other"), ("revision", "b" * 40)])
def test_foreign_or_stale_cache_is_refused(cache, field, value):
    ds, sidecar = cache
    sidecar[field] = value
    (ds.cache / "sidecar.json").write_text(json.dumps(sidecar))
    with pytest.raises(SystemExit, match="wrong cache|stale cache"):
        load_sidecar(ds)


def test_snapshot_must_match_source_pin(cache):
    ds, _ = cache
    with pytest.raises(SystemExit, match="revision"):
        replace(ds, revision="b" * 40).manifest()


def test_data_root_is_independent_of_install_location(cache, tmp_path, monkeypatch):
    ds, _ = cache
    before = ds.cache
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert ds.cache == before
    assert load_sidecar(ds)["repo_id"] == ds.repo_id
