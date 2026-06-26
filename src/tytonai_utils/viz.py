"""Visualization helpers for quick QA of downloaded data.

- plot_image_mask_pairs : imagery + mask from the .npz pairs in a manifest (Feature 2).
  Imagery npz stores bands as separate keys (RED, GREEN, BLUE, DSM, ...); RGB is composed
  from RED/GREEN/BLUE, and DSM can be shown as an extra panel.
- plot_image_mask_tiles : imagery .tif + mask .tif paired BY FILENAME across two folders
  (e.g. download_grid output + realign_annotations_to_grid output) — no manifest needed.

For web map outputs, see tytonai_utils.webmap.preview_tiles / plot_grid.

matplotlib is imported lazily; needs the `viz` (or `webmap`) extra. numpy is a core dep.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np


# ── npz loading (manifest case) ──────────────────────────────────────────────────────
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


def _compose_rgb(arrays, image_key, bands, rgb_keys) -> np.ndarray:
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


def _get_named(arrays, name) -> np.ndarray | None:
    """Fetch an array by case-insensitive key name (e.g. 'DSM'), or None."""
    upper = {k.upper(): k for k in arrays}
    key = upper.get(name.upper())
    return None if key is None else np.asarray(arrays[key]).squeeze()


def _pick_mask_array(arrays, key=None) -> np.ndarray:
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


# ── shared renderer ──────────────────────────────────────────────────────────────────
def _render_pairs(imgs, masks, labels, dsms=None, show_dsm=False, class_names=None,
                  cmap="tab20", dsm_cmap="terrain", max_rows=3, out_png=None):
    """Render image | (DSM) | mask panels with a shared class legend. Returns the Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    classes = sorted({int(v) for m in masks for v in np.unique(m)})
    base = plt.get_cmap(cmap)
    colors = [base(i % base.N) for i in range(len(classes))]
    listed = ListedColormap(colors)
    id_to_idx = {c: i for i, c in enumerate(classes)}

    per_sample = 3 if show_dsm else 2
    n_samp = len(imgs)
    per_row = math.ceil(n_samp / max_rows)
    n_rows = math.ceil(n_samp / per_row)
    ncols = per_row * per_sample
    fig, axes = plt.subplots(n_rows, ncols, figsize=(ncols * 3.0, n_rows * 3.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for i in range(n_samp):
        row, col = divmod(i, per_row)
        base_c = col * per_sample
        img_ax = axes[row, base_c]
        img_ax.imshow(imgs[i], cmap=None if imgs[i].ndim == 3 else "gray")
        img_ax.set_title(f"{labels[i]} image", fontsize=9)
        if show_dsm:
            dsm_ax = axes[row, base_c + 1]
            if dsms and dsms[i] is not None:
                im = dsm_ax.imshow(dsms[i], cmap=dsm_cmap)
                fig.colorbar(im, ax=dsm_ax, fraction=0.046, pad=0.04)
                dsm_ax.set_title(f"{labels[i]} DSM", fontsize=9)
        mask_ax = axes[row, base_c + per_sample - 1]
        mask_ax.imshow(_index_mask(masks[i], id_to_idx), cmap=listed, vmin=0, vmax=max(len(classes) - 1, 1))
        mask_ax.set_title(f"{labels[i]} mask", fontsize=9)

    names = class_names or {}
    patches = [Patch(facecolor=colors[i], label=names.get(c, str(c))) for i, c in enumerate(classes)]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(classes), 6), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # leave room for the legend
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    return fig


def _select(n_items: int, indexes, n, seed) -> list[int]:
    """Pick item positions: explicit `indexes`, else `n` random (seeded)."""
    if indexes is not None:
        return indexes
    return random.Random(seed).sample(range(n_items), k=min(n, n_items))


# ── public API ───────────────────────────────────────────────────────────────────────
def plot_image_mask_pairs(
    annotations_dir: str | Path,
    manifest,
    indexes: list[int] | None = None,
    n: int = 6,
    out_png: str | Path | None = None,
    mask_dir: str | Path | None = None,
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
    """Plot imagery + mask from the .npz pairs in a manifest (Feature 2), with a class legend.

    Needs the manifest because each tile's image and mask are separate npz files referenced by
    it. RGB is composed from `rgb_keys`; set show_dsm=True to add a DSM panel. Selects `indexes`
    else `n` random tiles (seeded). `mask_dir` defaults to annotations_dir — point it at a
    rolled-up folder (e.g. annotations_rnd7) to view remapped masks against the original imagery.
    See plot_image_mask_tiles for paired .tif folders.
    """
    if class_names is None:  # default: label the legend with the org class names
        from tytonai_utils.rollup import CLASS_NAMES

        class_names = CLASS_NAMES
    if isinstance(manifest, (str, Path)):
        from tytonai_utils.manifest import read_manifest

        manifest = read_manifest(manifest)
    annotations_dir = Path(annotations_dir)
    mask_dir = Path(mask_dir) if mask_dir is not None else annotations_dir
    indexes = _select(len(manifest), indexes, n, seed)

    imgs, dsms, masks, labels = [], [], [], []
    for idx in indexes:
        tile = manifest[idx]
        img_arrays = _load_npz_arrays(annotations_dir / tile["imagery_file"])
        imgs.append(_compose_rgb(img_arrays, image_key, bands, rgb_keys))
        dsms.append(_get_named(img_arrays, "DSM") if show_dsm else None)
        masks.append(_pick_mask_array(_load_npz_arrays(mask_dir / tile["mask_file"]), mask_key))
        labels.append(f"#{idx}")
    return _render_pairs(imgs, masks, labels, dsms=dsms, show_dsm=show_dsm,
                         class_names=class_names, cmap=cmap, dsm_cmap=dsm_cmap,
                         max_rows=max_rows, out_png=out_png)


def plot_image_mask_tiles(
    image_dir: str | Path,
    mask_dir: str | Path,
    indexes: list[int] | None = None,
    n: int = 6,
    out_png: str | Path | None = None,
    bands: tuple[int, ...] = (1, 2, 3),
    class_names: dict[int, str] | None = None,
    cmap: str = "tab20",
    max_rows: int = 3,
    seed: int = 0,
):
    """Plot imagery .tif next to mask .tif, paired BY FILENAME across two folders.

    For grid-aligned outputs: `image_dir` from download_grid, `mask_dir` from
    realign_annotations_to_grid — tiles match on name (tile_NNNNN.tif). No manifest needed.
    Pairs are the tiles present in BOTH folders; selects `indexes` else `n` random (seeded).
    `bands` are 1-based GeoTIFF bands for RGB. Needs rasterio (the `webmap` extra).
    """
    import rasterio

    if class_names is None:  # default: label the legend with the org class names
        from tytonai_utils.rollup import CLASS_NAMES

        class_names = CLASS_NAMES
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    names = sorted(p.name for p in mask_dir.glob("*.tif") if (image_dir / p.name).exists())
    if not names:
        raise FileNotFoundError(f"no matching tile_*.tif in both {image_dir} and {mask_dir}")
    chosen = _select(len(names), indexes, n, seed)

    imgs, masks, labels = [], [], []
    for i in chosen:
        name = names[i]
        with rasterio.open(image_dir / name) as src:
            sel = list(bands) if src.count >= max(bands) else [1]
            img = src.read(sel).transpose(1, 2, 0)
        with rasterio.open(mask_dir / name) as src:
            mask = src.read(1)
        imgs.append(_to_rgb(img, (0, 1, 2)))
        masks.append(mask)
        labels.append(name.removesuffix(".tif"))
    return _render_pairs(imgs, masks, labels, class_names=class_names, cmap=cmap,
                         max_rows=max_rows, out_png=out_png)


# ════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from tytonai_utils.rollup import RND_NAMES_7CLASS

    # A) npz pairs from a manifest (Feature 2) -----------------------------------------
    plot_image_mask_pairs("downloads/annotations", "inputs_tests/dataset.json", n=6,
                          class_names=RND_NAMES_7CLASS, out_png="pairs.png")

    # B) grid-aligned .tif tiles paired by filename (Feature 1 imagery + Feature 5 masks)
    plot_image_mask_tiles("downloads/tiles", "downloads/annotations_aligned", n=6,
                          class_names=RND_NAMES_7CLASS, out_png="pairs_aligned.png")
