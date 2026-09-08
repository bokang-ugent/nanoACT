"""Fetch a pinned benchmark dataset and write its checksum manifest.

The pinned content is the experiment's data. Conversion to a training format is
deliberately not done here — that belongs to the policy/training side, which knows its
needs. Usage:

    uv run python -m nanoact.download                                  # transfer_cube_scripted
    uv run python -m nanoact.download --dataset transfer_cube_human    # the human demonstrations
    uv run python -m nanoact.download --verify                         # recheck shards
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from .datasets import Dataset, add_argument, get


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _data_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json" and ".cache" not in p.parts
    )


def download(ds: Dataset) -> None:
    snapshot_download(repo_id=ds.repo_id, repo_type="dataset", revision=ds.revision,
                      local_dir=ds.snapshot)
    manifest = {
        "repo_id": ds.repo_id,
        "revision": ds.revision,
        "files": {str(p.relative_to(ds.snapshot)): _sha256(p) for p in _data_files(ds.snapshot)},
    }
    (ds.snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"downloaded {ds.repo_id}@{ds.revision[:12]} -> {ds.snapshot} ({len(manifest['files'])} files)")


def verify(ds: Dataset) -> None:
    manifest = ds.manifest()  # raises if the directory holds a different dataset
    assert manifest["revision"] == ds.revision, "manifest revision != pinned revision"
    actual = {str(p.relative_to(ds.snapshot)): _sha256(p) for p in _data_files(ds.snapshot)}
    if actual != manifest["files"]:
        missing = manifest["files"].keys() - actual.keys()
        extra = actual.keys() - manifest["files"].keys()
        changed = {k for k in actual.keys() & manifest["files"].keys() if actual[k] != manifest["files"][k]}
        print(f"VERIFY FAILED: missing={sorted(missing)} extra={sorted(extra)} changed={sorted(changed)}")
        sys.exit(1)
    print(f"verify OK: {len(actual)} files match manifest ({ds.repo_id}@{ds.revision[:12]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_argument(parser)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    ds = get(args.dataset)
    verify(ds) if args.verify else download(ds)
