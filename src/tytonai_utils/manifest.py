"""Feature 2 — manifest/annotation download: pull a dataset.json's NPZ pairs from S3.

A manifest (dataset.json) is a list of tile dicts (imagery_file, mask_file, geotransform,
srid, class_counts, ...). Each tile references two NPZ files living at
s3://$S3_FILE_BUCKET/<filename>. We download every imagery + mask NPZ into a chosen
out_dir. The download is cache-aware: a file already on disk is skipped unless force=True.

Uses boto3 (the `s3` extra) via the shared make_s3_client() — no aws CLI needed.
Call load_dotenv() first so the AWS_* creds + endpoint are in the environment.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from botocore.client import BaseClient
from tqdm import tqdm

from tytonai_utils.s3 import make_s3_client


def read_manifest(manifest_path: str | Path) -> list[dict]:
    """Load the dataset.json tile list."""
    with open(manifest_path) as f:
        return json.load(f)


def _annotation_files(tiles: list[dict], out_dir: Path) -> list[tuple[str, Path]]:
    """Every (s3 key, local dest) pair the manifest needs (imagery + mask per tile)."""
    return [
        (fname, out_dir / fname)
        for t in tiles
        for fname in (t["imagery_file"], t["mask_file"])
    ]


def download_file(s3: BaseClient, key: str, dest: Path, bucket: str, force: bool = False) -> bool:
    """Download s3://bucket/key -> dest. Returns True if downloaded, False if cached."""
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dest))
    return True


def download_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    bucket: str | None = None,
    force: bool = False,
    workers: int = 8,
    s3: BaseClient | None = None,
) -> Path:
    """Download every imagery + mask NPZ referenced by a manifest into out_dir.

    `bucket` defaults to $S3_FILE_BUCKET. `s3` defaults to a fresh make_s3_client().
    Cache-aware (skips files already on disk unless force). Returns out_dir.
    """
    s3 = s3 or make_s3_client()
    bucket = bucket or os.environ["S3_FILE_BUCKET"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _annotation_files(read_manifest(manifest_path), out_dir)

    def fetch(pair: tuple[str, Path]) -> bool:
        key, dest = pair
        return download_file(s3, key, dest, bucket, force)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        done = list(tqdm(ex.map(fetch, pairs), total=len(pairs), desc="npz annotations"))
    n_dl = sum(done)
    print(f"[manifest] downloaded {n_dl} NPZ ({len(pairs) - n_dl} cached) -> {out_dir}")
    return out_dir


# ════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as np
    from dotenv import load_dotenv

    load_dotenv(".env", override=True)  # AWS_* creds + endpoint

    CONFIG = {
        "manifest_path": Path("input_site_data/monrovia/dataset.json"),
        "out_dir": Path("input_site_data/monrovia/annotations"),
    }

    # 1) inspect the manifest (cheap, local) -------------------------------------------
    manifest = read_manifest(CONFIG["manifest_path"])
    print(f"{len(manifest)} tiles | bands {manifest[0].get('imagery_bands')}")

    # 2) one tile (imagery + mask) — confirms S3 auth + inspect the NPZ shapes ----------
    s3 = make_s3_client()
    bucket = os.environ["S3_FILE_BUCKET"]
    one = manifest[0]
    download_file(s3, one["imagery_file"], CONFIG["out_dir"] / one["imagery_file"], bucket)
    download_file(s3, one["mask_file"], CONFIG["out_dir"] / one["mask_file"], bucket)
    img = np.load(CONFIG["out_dir"] / one["imagery_file"])
    msk = np.load(CONFIG["out_dir"] / one["mask_file"])
    print("imagery:", {k: img[k].shape for k in img.files})
    print("mask   :", {k: msk[k].shape for k in msk.files})

    # 3) the full dataset (expensive) --------------------------------------------------
    out_dir = download_manifest(CONFIG["manifest_path"], CONFIG["out_dir"])
