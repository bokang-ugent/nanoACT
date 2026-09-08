"""Pinned ALOHA demonstrations used by the training examples.

Snapshots and decoded caches live under NANOACT_DATA_DIR, or ./data by default.
The download, preparation, and training commands must use the same data directory.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    name: str
    repo_id: str
    revision: str
    frames: int
    fps: int = 50
    image_key: str = "observation.images.top"

    @property
    def root(self) -> Path:
        return Path(os.environ.get("NANOACT_DATA_DIR", "data")).expanduser().resolve()

    @property
    def snapshot(self) -> Path:
        return self.root / self.repo_id.split("/")[-1]

    @property
    def cache(self) -> Path:
        return self.root / "cache" / self.name

    @property
    def video(self) -> Path:
        return self.snapshot / "videos" / self.image_key / "chunk-000" / "file-000.mp4"

    def manifest(self) -> dict:
        manifest = json.loads((self.snapshot / "manifest.json").read_text())
        if manifest["repo_id"] != self.repo_id:
            raise SystemExit(f"{self.snapshot} holds {manifest['repo_id']}, expected {self.repo_id}")
        if manifest["revision"] != self.revision:
            raise SystemExit(
                f"{self.snapshot} holds revision {manifest['revision']}, expected {self.revision}. "
                f"Run `python -m nanoact.download --dataset {self.name}`."
            )
        return manifest


DATASETS = {
    d.name: d
    for d in [
        Dataset("transfer_cube_scripted", "lerobot/aloha_sim_transfer_cube_scripted",
                "c5467ba9059e4c65f27775ced212076980a173fb", 20_000),
        Dataset("transfer_cube_human", "lerobot/aloha_sim_transfer_cube_human",
                "6a43d500f101255823a9d2b9dc244eeb01a2cd31", 20_000),
        Dataset("insertion_human", "lerobot/aloha_sim_insertion_human",
                "cc571a3c661df81b566dbfde3d5c1e85fcdf7884", 25_000),
        Dataset("insertion_scripted", "lerobot/aloha_sim_insertion_scripted",
                "8ab660912970111cbb26738b11458e6fc4a4aed1", 20_000),
    ]
}
DEFAULT = "transfer_cube_scripted"


def get(name: str) -> Dataset:
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    return DATASETS[name]


def add_argument(parser) -> None:
    parser.add_argument("--dataset", default=DEFAULT, choices=sorted(DATASETS),
                        help=f"pinned demonstration set (default: {DEFAULT})")
