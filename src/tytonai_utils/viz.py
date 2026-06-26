"""Visualization helpers for quick QA of downloaded data.

- plot_image_mask_pairs: imagery tiles next to their annotation masks (the .npz pairs from
  Feature 2), by random sample or explicit indexes, optionally saved as a PNG.

For web map outputs, the helpers live in tytonai_utils.webmap:
- preview_tiles : downscaled RGB/greyscale overview mosaic of downloaded .tif tiles.
- plot_grid     : the tile grid drawn over the study area (no download needed).

matplotlib is imported lazily; needs the `viz` (or `webmap`) extra. numpy is a core dep.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def _load_array(path: str | Path, key: str | None = None) -> np.ndarray:
    """Load one array from an .npz: the named key, else the largest array in the file."""
    with np.load(path) as npz:
        name = key if key is not None else max(npz.files, key=lambda k: npz[k].size)
        return npz[name]


def _to_displayable(arr: np.ndarray, bands: tuple[int, ...]) -> np.ndarray:
    """Coerce an image array to (H, W, 3) or (H, W) float in 0..1 for imshow."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a.astype("float32")  # greyscale
    else:
        if a.shape[0] <= 8 and a.shape[0] < a.shape[-1]:  # channels-first -> channels-last
            a = np.moveaxis(a, 0, -1)
        a = a[..., list(bands)] if a.shape[-1] >= 3 else a[..., 0]
        a = a.astype("float32")
    hi = float(a.max()) if a.size else 1.0
    return a / hi if hi > 1 else a  # scale uint8/raw to 0..1; leave already-normalized


def plot_image_mask_pairs(
    annotations_dir: str | Path,
    manifest,
    indexes: list[int] | None = None,
    n: int = 6,
    out_png: str | Path | None = None,
    image_key: str | None = None,
    mask_key: str | None = None,
    bands: tuple[int, ...] = (0, 1, 2),
    cmap: str = "tab20",
    seed: int = 0,
):
    """Plot imagery tiles next to their annotation masks for visual QA.

    `manifest` is the tile list (read_manifest output) or a path to dataset.json. Selects
    `indexes` if given, else `n` random tiles (seeded for reproducibility). Each row is
    image | mask. Array keys are auto-detected (largest array) unless image_key/mask_key
    are given. Returns the matplotlib Figure; saves to out_png if provided.
    """
    import matplotlib.pyplot as plt

    if isinstance(manifest, (str, Path)):
        from tytonai_utils.manifest import read_manifest

        manifest = read_manifest(manifest)

    annotations_dir = Path(annotations_dir)
    if indexes is None:
        indexes = random.Random(seed).sample(range(len(manifest)), k=min(n, len(manifest)))

    fig, axes = plt.subplots(len(indexes), 2, figsize=(7, 3.2 * len(indexes)), squeeze=False)
    for row, idx in enumerate(indexes):
        tile = manifest[idx]
        img = _to_displayable(_load_array(annotations_dir / tile["imagery_file"], image_key), bands)
        msk = np.asarray(_load_array(annotations_dir / tile["mask_file"], mask_key)).squeeze()
        axes[row, 0].imshow(img, cmap=None if img.ndim == 3 else "gray")
        axes[row, 0].set_title(f"#{idx} image", fontsize=9)
        axes[row, 1].imshow(msk, cmap=cmap)
        axes[row, 1].set_title(f"#{idx} mask classes={tuple(np.unique(msk))}", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from tytonai_utils.manifest import read_manifest

    CONFIG = {
        "manifest_path": Path("inputs_tests/dataset.json"),
        "annotations_dir": Path("input_site_data/monrovia/annotations"),
    }

    manifest = read_manifest(CONFIG["manifest_path"])

    # 1) random sample (auto-detected npz keys) ----------------------------------------
    fig = plot_image_mask_pairs(CONFIG["annotations_dir"], manifest, n=4, out_png="pairs.png")
    plt.show()

    # 2) specific tiles by index -------------------------------------------------------
    fig = plot_image_mask_pairs(CONFIG["annotations_dir"], manifest, indexes=[0, 1, 2])
    plt.show()
