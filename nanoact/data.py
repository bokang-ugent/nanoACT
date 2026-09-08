"""Batches of (image, state, action chunk) from a decoded cache.

Sampling matches lerobot's: a fresh permutation of all frames per epoch (its
EpisodeAwareSampler with drop_n_last_frames=0, which ACT leaves at its default). An action
chunk that runs past the end of its episode is clamped to the last frame and masked, so
the loss ignores it.
"""

import json

import numpy as np
import torch

from nanoact.datasets import Dataset

EPS = 1e-8  # lerobot divides by (std + eps) when normalizing, but not when inverting


def normalize(x: torch.Tensor, stats: dict, key: str) -> torch.Tensor:
    return (x - stats[key]["mean"]) / (stats[key]["std"] + EPS)


def unnormalize(x: torch.Tensor, stats: dict, key: str) -> torch.Tensor:
    return x * stats[key]["std"] + stats[key]["mean"]


def load_stats(stats: dict, device: str = "cpu") -> dict:
    """Turn plain lists (from the sidecar or a checkpoint) into broadcastable tensors.

    Image stats are per-channel (3,1,1); states and actions are (14,) and broadcast as is.
    """
    return {
        key: {name: torch.tensor(stats[key][name], dtype=torch.float32, device=device)
              for name in ("mean", "std")}
        for key in ("image", "state", "action")
    }


def load_sidecar(ds: Dataset) -> dict:
    """Read the cache sidecar, refusing a cache that is stale or belongs elsewhere."""
    sidecar = json.loads((ds.cache / "sidecar.json").read_text())
    # repo_id first. The pinned datasets share frame count, array shapes and dtypes, so a
    # cross-dataset mixup passes every structural check there is — only the name catches it.
    if sidecar.get("repo_id") != ds.repo_id:
        raise SystemExit(
            f"wrong cache at {ds.cache}: built from {sidecar.get('repo_id')!r}, "
            f"run is for {ds.repo_id!r}. Rebuild with "
            f"`python -m nanoact.prepare_data --dataset {ds.name}`."
        )
    revision = ds.manifest()["revision"]
    if sidecar["revision"] != revision:
        raise SystemExit(
            f"stale cache: built from {sidecar['revision']}, snapshot pinned at {revision}. "
            f"Re-run `python -m nanoact.prepare_data --dataset {ds.name}`."
        )
    if sidecar["frames"] != len(np.load(ds.cache / "state.npy", mmap_mode="r")):
        raise SystemExit("stale cache: frame count disagrees with state.npy")
    return sidecar


class Batches:
    """Infinite iterator of training batches, all tensors already normalized."""

    def __init__(self, ds: Dataset, chunk_size: int, batch_size: int, seed: int, device: str,
                 use_mp4: bool = False):
        self.sidecar = load_sidecar(ds)
        self.chunk_size, self.batch_size, self.device = chunk_size, batch_size, device
        self.stats = load_stats(self.sidecar["stats"], device)

        self.state = np.load(ds.cache / "state.npy")
        self.action = np.load(ds.cache / "action.npy")
        bounds = np.load(ds.cache / "episode_bounds.npy")
        self.n = len(self.state)

        # Per-frame episode end, so a sample needs no episode lookup. Only the end matters:
        # chunk offsets run forward, so they can never fall below the episode start.
        self.ep_end = np.empty(self.n, np.int64)
        for start, end in bounds:
            self.ep_end[start:end] = end

        self.images = None if use_mp4 else np.load(ds.cache / "images.npy", mmap_mode="r")
        self.decoder = None
        if use_mp4:
            from torchcodec.decoders import VideoDecoder

            self.decoder = VideoDecoder(str(ds.video))
        self.rng = np.random.default_rng(seed)

    def _images(self, idx: np.ndarray) -> torch.Tensor:
        """(B, 3, H, W) float in [0, 1]."""
        if self.decoder is None:
            raw = torch.from_numpy(np.ascontiguousarray(self.images[idx]))  # (B, H, W, 3) uint8
            return raw.permute(0, 3, 1, 2).float().div_(255)
        return torch.stack([self.decoder[int(i)] for i in idx]).float().div_(255)

    def _batch(self, idx: np.ndarray) -> tuple[torch.Tensor, ...]:
        # Action chunk indices, clamped to the episode; anything clamped is padding.
        offsets = idx[:, None] + np.arange(self.chunk_size)[None, :]
        is_pad = offsets >= self.ep_end[idx][:, None]
        offsets = np.minimum(offsets, self.ep_end[idx][:, None] - 1)

        # Blocking transfers: these tensors wrap temporary numpy arrays, which an async
        # copy can outlive — that silently produced NaNs on MPS. Pageable memory makes
        # non_blocking a no-op on CUDA anyway, so nothing is lost.
        dev = self.device
        image = normalize(self._images(idx).to(dev), self.stats, "image")
        state = normalize(torch.from_numpy(self.state[idx]).to(dev), self.stats, "state")
        action = normalize(torch.from_numpy(self.action[offsets]).to(dev), self.stats, "action")
        return image, state, action, torch.from_numpy(is_pad).to(dev)

    def stream(self):
        """Yield batches forever, reshuffling every epoch."""
        while True:
            order = self.rng.permutation(self.n)
            for start in range(0, self.n - self.batch_size + 1, self.batch_size):
                yield self._batch(order[start : start + self.batch_size])
