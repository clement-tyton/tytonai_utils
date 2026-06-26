"""Visualization helpers for quick QA of downloaded data.

- plot_image_mask_pairs: imagery tiles next to their annotation masks (the .npz pairs from
  Feature 2), by random sample or explicit indexes, with a class legend, saved as a PNG.

For web map outputs, the helpers live in tytonai_utils.webmap:
- preview_tiles : downscaled RGB/greyscale overview mosaic of downloaded .tif tiles.
- plot_grid     : the tile grid drawn over the study area (no download needed).

matplotlib is imported lazily; needs the `viz` (or `webmap`) extra. numpy is a core dep.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np


def _load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load every array from an .npz into a dict."""
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def _pick_image_array(arrays: dict[str, np.ndarray], key: str | None = None) -> np.ndarray:
    """The RGB(A) image: explicit key, else a 3D array with a 3/4-channel axis, else largest."""
    if key is not None:
        return arrays[key]
    rgb = [a for a in arrays.values() if a.ndim == 3 and (a.shape[0] in (3, 4) or a.shape[-1] in (3, 4))]
    return max(rgb, key=lambda a: a.size) if rgb else max(arrays.values(), key=lambda a: a.size)


def _pick_mask_array(arrays: dict[str, np.ndarray], key: str | None = None) -> np.ndarray:
    """The mask: explicit key, else the largest array that squeezes to 2D."""
    if key is not None:
        return np.asarray(arrays[key]).squeeze()
    flat = [a for a in arrays.values() if np.asarray(a).squeeze().ndim == 2]
    chosen = max(flat, key=lambda a: a.size) if flat else max(arrays.values(), key=lambda a: a.size)
    return np.asarray(chosen).squeeze()


def _to_rgb(arr: np.ndarray, bands: tuple[int, ...]) -> np.ndarray:
    """Coerce an image array to (H, W, 3) or (H, W) float in 0..1 for imshow."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a.astype("float32")
    else:
        if a.shape[0] <= 8 and a.shape[0] < a.shape[-1]:  # channels-first -> channels-last
            a = np.moveaxis(a, 0, -1)
        a = a[..., list(bands)] if a.shape[-1] >= 3 else a[..., 0]
        a = a.astype("float32")
    hi = float(a.max()) if a.size else 1.0
    return a / hi if hi > 1 else a  # scale uint8/raw to 0..1; leave already-normalized


def _index_mask(mask: np.ndarray, id_to_idx: dict[int, int]) -> np.ndarray:
    """Map class ids to contiguous colour indices via a lookup table."""
    mx = int(mask.max(initial=0))
    lut = np.zeros(mx + 1, dtype=int)
    for class_id, idx in id_to_idx.items():
        if class_id <= mx:
            lut[class_id] = idx
    return lut[mask]


def plot_image_mask_pairs(
    annotations_dir: str | Path,
    manifest,
    indexes: list[int] | None = None,
    n: int = 6,
    out_png: str | Path | None = None,
    image_key: str | None = None,
    mask_key: str | None = None,
    bands: tuple[int, ...] = (0, 1, 2),
    class_names: dict[int, str] | None = None,
    cmap: str = "tab20",
    max_rows: int = 3,
    seed: int = 0,
):
    """Plot imagery tiles next to their annotation masks for visual QA, with a class legend.

    `manifest` is the tile list (read_manifest output) or a path to dataset.json. Selects
    `indexes` if given, else `n` random tiles (seeded). Layout is capped at `max_rows` rows
    and grows columns (each sample = an image + mask pair). Mask classes get consistent
    colours across all panels; pass `class_names` (e.g. RND_NAMES_7CLASS) to label the legend
    by name. Image array is auto-detected as the 3/4-channel array unless image_key is given.
    Returns the matplotlib Figure; saves to out_png if provided.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    if isinstance(manifest, (str, Path)):
        from tytonai_utils.manifest import read_manifest

        manifest = read_manifest(manifest)

    annotations_dir = Path(annotations_dir)
    if indexes is None:
        indexes = random.Random(seed).sample(range(len(manifest)), k=min(n, len(manifest)))

    # Load all image + mask arrays first.
    imgs, masks = [], []
    for idx in indexes:
        tile = manifest[idx]
        imgs.append(_to_rgb(_pick_image_array(_load_npz_arrays(annotations_dir / tile["imagery_file"]), image_key), bands))
        masks.append(_pick_mask_array(_load_npz_arrays(annotations_dir / tile["mask_file"]), mask_key))

    # Consistent colours for every class present, with a legend.
    classes = sorted({int(v) for m in masks for v in np.unique(m)})
    base = plt.get_cmap(cmap)
    colors = [base(i % base.N) for i in range(len(classes))]
    listed = ListedColormap(colors)
    id_to_idx = {c: i for i, c in enumerate(classes)}

    # Layout: <= max_rows rows, grow columns. Each sample takes 2 columns (image | mask).
    n_samp = len(indexes)
    per_row = math.ceil(n_samp / max_rows)
    n_rows = math.ceil(n_samp / per_row)
    ncols = per_row * 2
    fig, axes = plt.subplots(n_rows, ncols, figsize=(ncols * 3.0, n_rows * 3.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for i, idx in enumerate(indexes):
        row, col = divmod(i, per_row)
        img_ax, mask_ax = axes[row, 2 * col], axes[row, 2 * col + 1]
        img_ax.imshow(imgs[i], cmap=None if imgs[i].ndim == 3 else "gray")
        img_ax.set_title(f"#{idx} image", fontsize=9)
        mask_ax.imshow(_index_mask(masks[i], id_to_idx), cmap=listed, vmin=0, vmax=max(len(classes) - 1, 1))
        mask_ax.set_title(f"#{idx} mask", fontsize=9)

    names = class_names or {}
    patches = [Patch(facecolor=colors[i], label=names.get(c, str(c))) for i, c in enumerate(classes)]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(classes), 6), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # leave room for the legend
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from tytonai_utils.manifest import read_manifest
    from tytonai_utils.rollup import RND_NAMES_7CLASS

    CONFIG = {
        "manifest_path": Path("inputs_tests/dataset.json"),
        "annotations_dir": Path("input_site_data/monrovia/annotations"),
    }
    manifest = read_manifest(CONFIG["manifest_path"])

    # random sample with a named legend (saves pairs.png) ------------------------------
    plot_image_mask_pairs(CONFIG["annotations_dir"], manifest, n=6,
                          class_names=RND_NAMES_7CLASS, out_png="pairs.png")

    # specific tiles by index ----------------------------------------------------------
    plot_image_mask_pairs(CONFIG["annotations_dir"], manifest, indexes=[0, 1, 2], out_png="pairs_idx.png")
