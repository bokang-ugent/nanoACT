import sys
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from nanoact import ACT, ACTConfig, compute_loss
from nanoact.model import ResNet18
from nanoact.policy import NanoACT
from nanoact import train


@dataclass
class SmallConfig(ACTConfig):
    chunk_size: int = 4
    dim_model: int = 32
    n_heads: int = 4
    dim_feedforward: int = 64
    n_encoder_layers: int = 1
    n_decoder_layers: int = 1
    n_vae_encoder_layers: int = 1
    latent_dim: int = 4
    batch_size: int = 2


@pytest.mark.parametrize("no_vae", [False, True])
def test_train_checkpoint_and_inference(cache, tmp_path, monkeypatch, no_vae):
    ds, sidecar = cache
    out = tmp_path / "outputs"
    monkeypatch.setattr(train, "ACTConfig", SmallConfig)
    monkeypatch.setattr(train, "get", lambda name: ds)
    # Keep the integration test offline; ImageNet weight parity is checked separately.
    monkeypatch.setattr(ResNet18, "load_imagenet_weights", lambda self: None)
    args = ["nanoact.train", "--steps", "1", "--device", "cpu", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", args + (["--no-vae"] if no_vae else []))
    train.main()

    checkpoint = out / "step_000001.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["steps"] == 1
    assert saved["stats"] == sidecar["stats"]
    assert saved["data_revision"] == ds.revision
    assert len(saved["source_sha256"]["model.py"]) == 64
    assert saved["config"]["use_vae"] is not no_vae
    assert all(torch.isfinite(value).all() for value in saved["model"].values())
    torch.manual_seed(1000)
    initial = ACT(SmallConfig(use_vae=not no_vae))
    assert not torch.equal(saved["model"]["action_head.weight"], initial.action_head.weight)

    policy = NanoACT(checkpoint, device="cpu")
    encoder_calls, model_calls = [], []
    policy.model.vae_encoder.register_forward_hook(lambda *_: encoder_calls.append(1))
    policy.model.register_forward_hook(lambda *_: model_calls.append(1))
    observation = {"pixels": {"top": np.full((32, 32, 3), 255, np.uint8)},
                   "agent_pos": np.full(14, 5.0, np.float32)}
    action = policy.select_action(observation)
    # Compare against normalized inputs and de-normalized outputs explicitly.
    with torch.no_grad():
        predicted, mu, logvar = policy.model(torch.full((1, 3, 32, 32), 1.5),
                                            torch.full((1, 14), 2.0))
    np.testing.assert_allclose(action, predicted[0, 0].numpy() * 2.0 + 10.0, rtol=1e-5)
    assert mu is None and logvar is None
    model_calls.clear()
    policy.select_action(observation)
    assert not model_calls  # the remaining chunk is replayed
    policy.reset()
    np.testing.assert_array_equal(policy.select_action(observation), action)
    assert len(model_calls) == 1
    assert not encoder_calls


def test_padding_does_not_change_training_signal():
    torch.manual_seed(0)
    model = ACT(SmallConfig(dropout=0.0)).train()
    image, state = torch.randn(2, 3, 32, 32), torch.randn(2, 14)
    actions = torch.randn(2, 4, 14)
    padding = torch.tensor([[False, False, True, True], [False, False, False, True]])
    changed = actions.clone()
    changed[padding] = 1000
    torch.manual_seed(7)
    loss_a, _, _ = compute_loss(model, image, state, actions, padding)
    torch.manual_seed(7)
    loss_b, _, _ = compute_loss(model, image, state, changed, padding)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)


@pytest.mark.parametrize("source_checkout", [False, True])
def test_commit_fallback_with_missing_git(tmp_path, monkeypatch, source_checkout):
    package = tmp_path / "nanoact"
    package.mkdir()
    if source_checkout:
        (tmp_path / ".git").mkdir()
    monkeypatch.setattr(train, "__file__", str(package / "train.py"))
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(train.subprocess, "run", missing_git)
    assert train.git_commit() == "unknown"
    (package / "COMMIT").write_text("frozen-source\n")
    assert train.git_commit() == "frozen-source"


def test_installed_package_does_not_record_enclosing_repo(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    package = tmp_path / "venv" / "site-packages" / "nanoact"
    package.mkdir(parents=True)
    monkeypatch.setattr(train, "__file__", str(package / "train.py"))
    def unrelated_git(*args, **kwargs):
        pytest.fail("installed package queried the enclosing repository's HEAD")
    monkeypatch.setattr(train.subprocess, "run", unrelated_git)
    assert train.git_commit() == "unknown"
