"""Feature 5 — realign annotations to a grid.

Annotation tiles from a manifest are georeferenced (each has a `geotransform` + `srid`) but
their tiling rarely matches a grid you defined from a web map. This merges every annotation
mask into one mosaic, then cuts it along your grid cells (from build_grid) — producing masks
aligned 1:1 with the imagery tiles that download_grid writes for the same grid.

    webmap + res + patch  ->  build_grid           ->  grid
    dataset manifest      ->  annotation masks (misaligned)
    realign_annotations_to_grid(grid, ...)         ->  grid-aligned mask tiles (tile_NNNNN.tif)

Uses rasterio (the `webmap` extra) + numpy. Output GeoTIFFs pair with download_grid output
by index: imagery tile_00005.tif <-> mask tile_00005.tif.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _read_manifest(manifest) -> list[dict]:
    """Accept a manifest list or a path to dataset.json."""
    if isinstance(manifest, (str, Path)):
        with open(manifest) as f:
            return json.load(f)
    return manifest


def _load_mask(path: str | Path, key: str | None = None) -> np.ndarray:
    """Load a 2D mask array from an .npz (named key, else the largest array that squeezes 2D)."""
    with np.load(path) as npz:
        if key is not None:
            return np.asarray(npz[key]).squeeze()
        flat = [npz[k] for k in npz.files if np.asarray(npz[k]).squeeze().ndim == 2]
        chosen = max(flat or [npz[k] for k in npz.files], key=lambda a: a.size)
        return np.asarray(chosen).squeeze()


def _mask_datasets(annotations_dir: Path, manifest: list[dict], mask_key, nodata):
    """Open every annotation mask as a georeferenced in-memory rasterio dataset."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    datasets, handles = [], []
    for tile in tqdm(manifest, desc="loading masks"):
        arr = _load_mask(annotations_dir / tile["mask_file"], mask_key)
        mem = MemoryFile()
        ds = mem.open(
            driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype,
            crs=f"EPSG:{tile['srid']}", transform=Affine.from_gdal(*tile["geotransform"]),
            nodata=nodata,
        )
        ds.write(arr, 1)
        datasets.append(ds)
        handles.append(mem)
    return datasets, handles


def _vote_mosaic(datasets, nodata):
    """Majority-vote mosaic: per pixel, the class chosen by the most overlapping tiles.

    Resolves contradictions where different annotation sets overlap. Returns (mosaic, transform)
    matching rasterio.merge's shape ((1, H, W), Affine). Memory ~ one (H, W) uint16 grid per
    distinct class present.
    """
    from rasterio.transform import from_origin

    res_x, res_y = datasets[0].transform.a, -datasets[0].transform.e  # res_y made positive
    left = min(ds.bounds.left for ds in datasets)
    right = max(ds.bounds.right for ds in datasets)
    bottom = min(ds.bounds.bottom for ds in datasets)
    top = max(ds.bounds.top for ds in datasets)
    width = int(round((right - left) / res_x))
    height = int(round((top - bottom) / res_y))
    transform = from_origin(left, top, res_x, res_y)

    counts: dict[int, np.ndarray] = {}
    for ds in tqdm(datasets, desc="voting overlaps"):
        arr = ds.read(1)
        row_off = max(0, int(round((top - ds.bounds.top) / res_y)))
        col_off = max(0, int(round((ds.bounds.left - left) / res_x)))
        h = min(arr.shape[0], height - row_off)
        w = min(arr.shape[1], width - col_off)
        arr = arr[:h, :w]
        for c in np.unique(arr):
            if int(c) == nodata:
                continue
            grid = counts.setdefault(int(c), np.zeros((height, width), dtype=np.uint16))
            grid[row_off:row_off + h, col_off:col_off + w][arr == c] += 1

    result = np.full((height, width), nodata, dtype=datasets[0].dtypes[0])
    best = np.zeros((height, width), dtype=np.uint16)
    for c, cnt in counts.items():
        win = cnt > best
        result[win] = c
        best[win] = cnt[win]
    return result[np.newaxis, ...], transform


def _build_mosaic(datasets, nodata, overlapping):
    """Mosaic the mask datasets, resolving overlaps per `overlapping` (first/last/vote)."""
    if overlapping == "vote":
        return _vote_mosaic(datasets, nodata)
    if overlapping in ("first", "last"):
        from rasterio.merge import merge

        return merge(datasets, nodata=nodata, method=overlapping)
    raise ValueError(f"overlapping must be 'first', 'last' or 'vote', got {overlapping!r}")


def realign_annotations_to_grid(
    grid,
    annotations_dir: str | Path,
    manifest,
    out_dir: str | Path,
    mask_key: str | None = None,
    nodata: int = 0,
    skip_empty: bool = True,
    overlapping: str = "first",
) -> list[str]:
    """Re-tile misaligned annotation masks onto `grid`, writing grid-aligned GeoTIFF masks.

    `grid` is a GeoDataFrame (from build_grid, or any tiling). Every manifest mask is
    georeferenced (geotransform + srid), merged into one mosaic. Only grid cells that intersect
    the union of annotation footprints are cut (annotations are sparse, so this skips the empty
    gaps cheaply); cells with no coverage are then dropped when skip_empty. Output tiles are
    named tile_NNNNN.tif by grid index, pairing with download_grid imagery. Returns names.

    `overlapping` resolves pixels covered by several (possibly contradicting) annotation sets:
    "first" (default, first tile wins), "last", or "vote" (per-pixel majority across tiles).

    Note: grid and annotations should share a resolution for a clean re-tile (only the tile
    boundaries shift); the grid is reprojected to the annotation CRS if they differ.
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.windows import from_bounds

    manifest = _read_manifest(manifest)
    annotations_dir, out_dir = Path(annotations_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, handles = _mask_datasets(annotations_dir, manifest, mask_key, nodata)
    try:
        from shapely.geometry import box

        footprints = [box(*ds.bounds) for ds in datasets]  # exact rectangle of each annotation tile
        mosaic, transform = _build_mosaic(datasets, nodata, overlapping)
        crs = datasets[0].crs
    finally:
        for ds in datasets:
            ds.close()
        for mem in handles:
            mem.close()

    if getattr(grid, "crs", None) is not None and grid.crs != crs:
        grid = grid.to_crs(crs)

    # Annotations are sparse rectangles, not a filled bbox — pre-select only grid cells that
    # intersect the ACTUAL union of annotation footprints, so we don't cut cells in the gaps.
    from shapely.ops import unary_union

    footprint = unary_union(footprints)  # multipolygon of the training areas
    left, top = transform.c, transform.f
    right, bottom = left + mosaic.shape[2] * transform.a, top - mosaic.shape[1] * abs(transform.e)
    grid_bounds = tuple(round(v, 1) for v in grid.total_bounds)
    cells = grid[grid.intersects(footprint)]

    profile = dict(
        driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2], count=1,
        dtype=mosaic.dtype, crs=crs, transform=transform, nodata=nodata,
    )
    written = []
    with MemoryFile() as mm:
        with mm.open(**profile) as tmp:
            tmp.write(mosaic)
        with mm.open() as mosaic_ds:
            # name tiles by grid index (stable cell id) so it pairs with download_grid output
            for idx, geom in tqdm(cells.geometry.items(), total=len(cells), desc="cutting tiles"):
                win = from_bounds(*geom.bounds, transform=mosaic_ds.transform).round_offsets().round_lengths()
                data = mosaic_ds.read(1, window=win, boundless=True, fill_value=nodata)
                if skip_empty and not (data != nodata).any():
                    continue
                out = out_dir / f"tile_{idx:05d}.tif"
                with rasterio.open(
                    out, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                    dtype=data.dtype, crs=mosaic_ds.crs, transform=mosaic_ds.window_transform(win),
                    nodata=nodata, compress="deflate", tiled=True,
                ) as dst:
                    dst.write(data, 1)
                written.append(out.name)
    if not written:  # grid likely doesn't overlap the annotation mosaic (check CRS/extent)
        mb = (round(left, 1), round(bottom, 1), round(right, 1), round(top, 1))
        print(f"[align] 0 tiles — grid may not overlap the annotations. grid bounds={grid_bounds} "
              f"CRS={getattr(grid, 'crs', None)}; mosaic bounds={mb} CRS={crs}")
    print(f"[align] wrote {len(written)} grid-aligned mask tiles -> {out_dir}")
    return written


def realign_annotations_from_shp(
    shp_path: str | Path,
    res: float,
    patch: int,
    annotations_dir: str | Path,
    manifest,
    out_dir: str | Path,
    mask_key: str | None = None,
    nodata: int = 0,
    skip_empty: bool = True,
    overlapping: str = "first",
) -> list[str]:
    """One call: build the grid from a vector file (res/patch), then realign annotations to it.

    `res` is the web map's native resolution (match the annotation resolution for a clean
    re-tile). `overlapping` resolves contradicting overlaps ("first"/"last"/"vote"). Wraps
    build_grid + realign_annotations_to_grid. Returns written tile names.
    """
    from tytonai_utils.webmap import build_grid

    grid, _ = build_grid(shp_path, res, patch)
    return realign_annotations_to_grid(
        grid, annotations_dir, manifest, out_dir, mask_key=mask_key, nodata=nodata,
        skip_empty=skip_empty, overlapping=overlapping,
    )


# ════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from tytonai_utils.webmap import build_grid

    CONFIG = {
        "fgb": Path("study_area.fgb"),
        "manifest": Path("inputs_tests/dataset.json"),
        "annotations_dir": Path("downloads/annotations"),
        "out_dir": Path("downloads/annotations_aligned"),
        "res": 0.0206348504972787,  # match the annotation resolution (manifest resolution_values)
        "patch": 512,
    }

    # 1) the grid you want to align onto (same res as the annotations) ------------------
    grid, study_area = build_grid(CONFIG["fgb"], CONFIG["res"], CONFIG["patch"])
    print(f"{len(grid)} grid cells, CRS={study_area.crs}")

    # 2) realign every annotation mask onto that grid ----------------------------------
    written = realign_annotations_to_grid(
        grid, CONFIG["annotations_dir"], CONFIG["manifest"], CONFIG["out_dir"]
    )

    # one-call equivalent (build grid + realign) ---------------------------------------
    # written = realign_annotations_from_shp(CONFIG["fgb"], CONFIG["res"], CONFIG["patch"],
    #                                        CONFIG["annotations_dir"], CONFIG["manifest"],
    #                                        CONFIG["out_dir"])
