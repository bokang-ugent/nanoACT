"""Decode a pinned snapshot once into flat arrays that training can memmap.

The supported snapshots store video in a single AV1 mp4. We decode it once,
checksum the result, and train from a uint8 memmap. Training then needs no video decoder.

The cache is *derived*: the snapshot stays the source of truth, and `sidecar.json` records
the dataset and revision it came from so training can refuse a stale or foreign cache.

    python -m nanoact.prepare_data                                  # transfer_cube_scripted
    python -m nanoact.prepare_data --dataset transfer_cube_human    # the human demonstrations
    python -m nanoact.prepare_data --verify                         # re-checksum a cache
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nanoact.datasets import Dataset, add_argument, get

BATCH = 500  # frames per decode call; bounds peak memory at ~460 MB (ALOHA-sized frames)


def _table(directory: Path, frames: int | None = None) -> pa.Table:
    """Read a LeRobot parquet directory, which may be sharded.

    The scripted set is one file; the human set is three. Filename order is the row order
    — but that is asserted below rather than trusted, since a wrong concatenation would
    silently misalign every episode boundary.
    """
    table = pa.concat_tables([pq.read_table(p) for p in sorted(directory.rglob("*.parquet"))])
    if frames is not None:
        assert table.num_rows == frames, f"expected {frames} rows, got {table.num_rows}"
        index = table["index"].to_numpy(zero_copy_only=False)
        assert np.array_equal(index, np.arange(frames)), "parquet shards concatenated out of order"
    return table


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stats(ds: Dataset) -> dict:
    """Normalization statistics, taken from the snapshot rather than recomputed.

    LeRobot normalizes from this same file, so reading it is what makes our normalization
    identical to the recipe's by construction instead of by coincidence.
    """
    raw = json.loads((ds.snapshot / "meta" / "stats.json").read_text())
    out = {}
    for key, name in [(ds.image_key, "image"), ("observation.state", "state"),
                      ("action", "action")]:
        out[name] = {"mean": np.array(raw[key]["mean"], np.float32).tolist(),
                     "std": np.array(raw[key]["std"], np.float32).tolist()}
    return out


def build(ds: Dataset) -> None:
    from torchcodec.decoders import VideoDecoder  # offline-only dependency

    revision = ds.manifest()["revision"]  # raises if the directory holds a different dataset
    ds.cache.mkdir(parents=True, exist_ok=True)
    table = _table(ds.snapshot / "data", ds.frames)

    for name, dtype in [("state", np.float32), ("action", np.float32)]:
        col = "observation.state" if name == "state" else name
        np.save(ds.cache / f"{name}.npy", np.stack(table[col].to_numpy(zero_copy_only=False)).astype(dtype))

    # Episode bounds, used to clamp and mask action chunks that run past an episode's end.
    eps = _table(ds.snapshot / "meta" / "episodes")
    bounds = np.stack([eps["dataset_from_index"].to_numpy(), eps["dataset_to_index"].to_numpy()], 1)
    np.save(ds.cache / "episode_bounds.npy", bounds.astype(np.int64))

    decoder = VideoDecoder(str(ds.video))
    assert decoder.metadata.num_frames == ds.frames, "video/parquet frame count mismatch"

    H, W = decoder.metadata.height, decoder.metadata.width
    images = np.lib.format.open_memmap(ds.cache / "images.npy", mode="w+",
                                       dtype=np.uint8, shape=(ds.frames, H, W, 3))
    for start in range(0, ds.frames, BATCH):
        stop = min(start + BATCH, ds.frames)
        # torchcodec yields CHW; the cache is HWC to match the gym observation layout.
        images[start:stop] = decoder[start:stop].permute(0, 2, 3, 1).numpy()
        print(f"\rdecoded {stop}/{ds.frames}", end="", flush=True)
    images.flush()
    del images, decoder
    print()

    sidecar = {
        "repo_id": ds.repo_id,
        "revision": revision,
        "frames": ds.frames,
        "shape": [ds.frames, H, W, 3],
        "dtype": "uint8",
        "sha256": _sha256(ds.cache / "images.npy"),
        "stats": _stats(ds),
    }
    (ds.cache / "sidecar.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"cache ready: {ds.cache}  sha256={sidecar['sha256'][:16]}…")


def verify(ds: Dataset) -> None:
    sidecar = json.loads((ds.cache / "sidecar.json").read_text())
    assert sidecar["repo_id"] == ds.repo_id, f"cache built from {sidecar['repo_id']}, expected {ds.repo_id}"
    expected = ds.manifest()["revision"]
    assert sidecar["revision"] == expected, f"cache built from {sidecar['revision']}, snapshot is {expected}"
    digest = _sha256(ds.cache / "images.npy")
    assert digest == sidecar["sha256"], f"images.npy changed: {digest} != {sidecar['sha256']}"
    print(f"cache verified: {ds.repo_id}@{expected[:12]}…")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_argument(parser)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    ds = get(args.dataset)
    verify(ds) if args.verify else build(ds)
