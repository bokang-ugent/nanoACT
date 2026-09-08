"""Checkpoint inference for ALOHA observations.

Normalization statistics travel inside the checkpoint, so a checkpoint runs on a machine
that has never seen the dataset.

The policy predicts a chunk and replays it one action at a time (no temporal
ensembling, matching the recipe). `reset()` empties the queue so the next `select_action`
re-plans.
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from nanoact.data import load_stats, normalize, unnormalize
from nanoact.model import ACT, ACTConfig


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NanoACT:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = device or _pick_device()
        self.checkpoint = str(checkpoint)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)

        cfg = ACTConfig(**ckpt["config"])
        self.model = ACT(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()  # also what keeps the VAE encoder out of the forward pass

        self.stats = load_stats(ckpt["stats"], self.device)
        self._queue: deque = deque()

    def reset(self) -> None:
        self._queue.clear()

    @torch.no_grad()
    def select_action(self, obs: dict) -> np.ndarray:
        if not self._queue:
            image = torch.from_numpy(obs["pixels"]["top"]).to(self.device)
            image = image.permute(2, 0, 1).float().div_(255).unsqueeze(0)
            state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).to(self.device)

            chunk, _, _ = self.model(
                normalize(image, self.stats, "image"), normalize(state, self.stats, "state").unsqueeze(0)
            )
            self._queue.extend(unnormalize(chunk[0], self.stats, "action").cpu().numpy())
        return self._queue.popleft()
