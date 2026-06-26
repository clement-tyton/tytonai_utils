"""Visualization helpers for quick QA of downloaded data.

- plot_image_mask_pairs: imagery tiles next to their annotation masks (the .npz pairs from
  Feature 2), by random sample or explicit indexes, with a class legend, saved as a PNG.
  Imagery npz files store bands as separate keys (RED, GREEN, BLUE, DSM, ...); RGB is
  composed from the RED/GREEN/BLUE keys, and DSM can be shown as an extra panel.

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


def _compose_rgb(
    arrays: dict[str, np.ndarray],
    image_key: str | None,
    bands: tuple[int, ...],
    rgb_keys: tuple[str, str, str],
) -> np.ndarray:
    """Build a displayable RGB image from an npz: named RGB keys, else a 3D array, else grey."""
    if image_key is not None:
        return _to_rgb(arrays[image_key], bands)
    upper = {k.upper(): k for k in arrays}
    if all(b.upper() in upper for b in rgb_keys):  # bands stored as separate keys
        stacked = np.stack([arrays[upper[b.upper()]] for b in rgb_keys], axis=-1)
        return _to_rgb(stacked, (0, 1, 2))
    rgb3d = [a for a in arrays.values() if a.ndim == 3 and (a.shape[0] in (3, 4) or a.shape[-1] in (3, 4))]
    if rgb3d:
        return _to_rgb(max(rgb3d, key=lambda a: a.size), bands)
    return _to_rgb(max(arrays.values(), key=lambda a: a.size), bands)  # last resort -> grey


def _get_named(arrays: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    """Fetch an array by case-insensitive key name (e.g. 'DSM'), or None."""
    upper = {k.upper(): k for k in arrays}
    key = upper.get(name.upper())
    return None if key is None else np.asarray(arrays[key]).squeeze()


def _pick_mask_array(arrays: dict[str, np.ndarray], key: str | None = None) -> np.ndarray:
    """The mask: explicit key, else the largest array that squeezes to 2D."""
    if key is not None:
        return np.asarray(arrays[key]).squeeze()
    flat = [a for a in arrays.values() if np.asarray(a).squeeze().ndim == 2]
    chosen = max(flat, key=lambda a: a.size) if flat else max(arrays.values(), key=lambda a: a.size)
    return np.asarray(chosen).squeeze()


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
    rgb_keys: tuple[str, str, str] = ("RED", "GREEN", "BLUE"),
    show_dsm: bool = False,
    dsm_cmap: str = "terrain",
    class_names: dict[int, str] | None = None,
    cmap: str = "tab20",
    max_rows: int = 3,
    seed: int = 0,
):
    """Plot imagery tiles next to their annotation masks for visual QA, with a class legend.

    Imagery npz files store bands as separate keys; RGB is composed from `rgb_keys`
    (RED/GREEN/BLUE by default). Set show_dsm=True to add a DSM panel per sample. `manifest`
    is the tile list or a path to dataset.json. Selects `indexes` if given, else `n` random
    tiles (seeded). Layout is capped at `max_rows` rows and grows columns. Mask classes get
    consistent colours; pass `class_names` (e.g. RND_NAMES_7CLASS) to label the legend by
    name. Returns the matplotlib Figure; saves to out_png if provided.
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

    # Load image (+ optional DSM) and mask for each selected tile.
    imgs, dsms, masks = [], [], []
    for idx in indexes:
        tile = manifest[idx]
        img_arrays = _load_npz_arrays(annotations_dir / tile["imagery_file"])
        imgs.append(_compose_rgb(img_arrays, image_key, bands, rgb_keys))
        dsms.append(_get_named(img_arrays, "DSM") if show_dsm else None)
        masks.append(_pick_mask_array(_load_npz_arrays(annotations_dir / tile["mask_file"]), mask_key))

    # Consistent colours for every class present, with a legend.
    classes = sorted({int(v) for m in masks for v in np.unique(m)})
    base = plt.get_cmap(cmap)
    colors = [base(i % base.N) for i in range(len(classes))]
    listed = ListedColormap(colors)
    id_to_idx = {c: i for i, c in enumerate(classes)}

    # Layout: <= max_rows rows, grow columns. Each sample = RGB (+ DSM) + mask.
    per_sample = 3 if show_dsm else 2
    n_samp = len(indexes)
    per_row = math.ceil(n_samp / max_rows)
    n_rows = math.ceil(n_samp / per_row)
    ncols = per_row * per_sample
    fig, axes = plt.subplots(n_rows, ncols, figsize=(ncols * 3.0, n_rows * 3.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for i, idx in enumerate(indexes):
        row, col = divmod(i, per_row)
        base_c = col * per_sample
        img_ax = axes[row, base_c]
        img_ax.imshow(imgs[i], cmap=None if imgs[i].ndim == 3 else "gray")
        img_ax.set_title(f"#{idx} image", fontsize=9)
        if show_dsm:
            dsm_ax = axes[row, base_c + 1]
            if dsms[i] is not None:
                im = dsm_ax.imshow(dsms[i], cmap=dsm_cmap)
                fig.colorbar(im, ax=dsm_ax, fraction=0.046, pad=0.04)
                dsm_ax.set_title(f"#{idx} DSM", fontsize=9)
        mask_ax = axes[row, base_c + per_sample - 1]
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

    # RGB composed from RED/GREEN/BLUE keys, named legend (saves pairs.png) -------------
    plot_image_mask_pairs(CONFIG["annotations_dir"], manifest, n=6,
                          class_names=RND_NAMES_7CLASS, out_png="pairs.png")

    # same, plus the DSM panel ---------------------------------------------------------
    plot_image_mask_pairs(CONFIG["annotations_dir"], manifest, n=3, show_dsm=True,
                          class_names=RND_NAMES_7CLASS, out_png="pairs_dsm.png")
