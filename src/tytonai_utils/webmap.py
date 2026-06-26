"""Feature 1 — web map import: cut local GeoTIFF tiles from an S3 web map.

A web map is a Cloud-Optimized GeoTIFF on S3. We never download the whole raster:
GDAL opens it via /vsis3 and does ranged GETs, so each tile is just the bytes of its
window. A .fgb (FlatGeobuf vector area) defines the region; we build a geotransform from
its bbox top-left + the raster resolution, tile in pixel space, then read each window and
write it as a georeferenced GeoTIFF. Empty tiles (no coverage) are skipped.

The AWS_* keys in .env ARE GDAL's /vsis3 config vars, so `load_dotenv()` is all the auth
this needs — no boto3. Install extras: `pip install tytonai_utils[webmap]`.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import DatasetReader
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from bbox_to_tile_grid.tilegrid import create_adaptive_grid


def to_gdal_path(link: str) -> str:
    """Normalize a web map link into a GDAL-openable path for ranged reads."""
    if link.startswith(("/vsis3/", "/vsicurl/", "/vsicurl_streaming/")):
        return link
    if link.startswith("s3://"):
        return "/vsis3/" + link[len("s3://") :]
    if link.startswith(("http://", "https://")):
        return "/vsicurl/" + link
    return link  # local file or already-openable path


def build_grid(fgb_path: str | Path, res: float, patch: int) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """One .fgb -> its bbox -> a uniform patch-pixel grid. Returns (grid, study_area).

    `res` is the web map's native resolution in metres/pixel (read it in the tytonai app,
    in the same place you copy the S3 link); `patch` is the tile size in pixels.
    """
    study_area = gpd.read_file(fgb_path)
    minx, miny, maxx, maxy = study_area.total_bounds
    # geotransform anchored at the bbox top-left at `res` m/px (north-up).
    gt = from_origin(minx, maxy, res, res)
    # clip_file=None + fixed_size=True -> plain regular grid covering the whole bbox.
    grid = create_adaptive_grid(
        tuple(study_area.total_bounds), None, gt, study_area.crs, patch, patch, 0, fixed_size=True
    )
    return grid, study_area


def _tile_window(src: DatasetReader, geom: BaseGeometry) -> Window:
    """Pixel window for one tile geometry, snapped to the raster's real transform."""
    return from_bounds(*geom.bounds, transform=src.transform).round_offsets().round_lengths()


def _alpha_index(src: DatasetReader) -> int | None:
    """1-based index of the alpha band (via colour interpretation), or None."""
    for i, ci in enumerate(src.colorinterp, start=1):
        if ci == ColorInterp.alpha:
            return i
    return None


def _data_bands(src: DatasetReader, bands: list[int] | None, alpha_idx: int | None) -> list[int]:
    """Band indices to write: caller's `bands`, else every band except the alpha band."""
    if bands is not None:
        return list(bands)
    return [i for i in range(1, src.count + 1) if i != alpha_idx]


def _is_covered(data, alpha, nodata) -> bool:
    """True if the tile holds real data: alpha>0 if present, else any non-nodata pixel."""
    if alpha is not None:
        return bool((alpha > 0).any())
    if nodata is not None:
        return bool((data != nodata).any())
    return bool((data != 0).any())


def _write_tile(out_path: Path, data, src: DatasetReader, win: Window, indexes: list[int]) -> None:
    """Write the selected bands of a window as a compressed, tiled, georeferenced GeoTIFF."""
    with rasterio.open(
        out_path, "w", driver="GTiff",
        width=data.shape[2], height=data.shape[1], count=data.shape[0], dtype=data.dtype,
        crs=src.crs, transform=src.window_transform(win), nodata=src.nodata,
        compress="deflate", tiled=True,
    ) as dst:
        dst.write(data)
        dst.colorinterp = [src.colorinterp[i - 1] for i in indexes]


def download_grid(
    grid: gpd.GeoDataFrame,
    webmap: str,
    out_dir: str | Path,
    bands: list[int] | None = None,
    workers: int = 8,
    skip_empty: bool = True,
) -> list[str]:
    """Download every tile of `grid` from `webmap` (S3) into out_dir as GeoTIFFs.

    `bands` selects 1-based bands to write (e.g. [1,2,3] RGB, [1,2,3,4] RGB+NIR, [1] a
    mask); default None writes every band except the alpha band. Coverage for skip_empty
    uses the alpha band if present, else the raster's nodata, else any non-zero pixel.

    Parallel ranged reads (I/O-bound; GDAL drops the GIL). rasterio handles are NOT
    thread-safe, so each worker thread opens its own (thread-local). Returns written names.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = to_gdal_path(webmap)
    local = threading.local()

    def worker(item: tuple[int, BaseGeometry]) -> str | None:
        idx, geom = item
        src = getattr(local, "src", None)
        if src is None:
            src = local.src = rasterio.open(path)
            local.alpha_idx = _alpha_index(src)
            local.indexes = _data_bands(src, bands, local.alpha_idx)
            out_of_range = [b for b in local.indexes if not 1 <= b <= src.count]
            if out_of_range:
                raise ValueError(
                    f"webmap has {src.count} band(s) (colorinterp="
                    f"{[ci.name for ci in src.colorinterp]}); requested bands {local.indexes} "
                    f"include out-of-range {out_of_range}. For a single-band raster (e.g. DSM) "
                    f"pass bands=[1], or bands=None to download all available bands."
                )
        win = _tile_window(src, geom)
        data = src.read(local.indexes, window=win, boundless=True, fill_value=0)
        if skip_empty:
            alpha = None
            if local.alpha_idx is not None:
                alpha = src.read(local.alpha_idx, window=win, boundless=True, fill_value=0)
            if not _is_covered(data, alpha, src.nodata):
                return None
        out = out_dir / f"tile_{idx:05d}.tif"
        _write_tile(out, data, src, win, local.indexes)
        return out.name

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # name tiles by grid index (stable cell id) so a subset (grid.iloc[...]) keeps names
        results = list(
            tqdm(ex.map(worker, grid.geometry.items()), total=len(grid), desc=out_dir.name)
        )
    return [r for r in results if r]


def download_webmap_from_shp(
    shp_path: str | Path,
    webmap: str,
    out_dir: str | Path,
    res: float,
    patch: int,
    bands: list[int] | None = None,
    workers: int = 8,
    skip_empty: bool = True,
) -> list[str]:
    """Download a web map (S3 link) as tiles over the area in a vector file, in one call.

    Builds the tile grid from `shp_path` (.shp / .fgb / .geojson — anything geopandas
    reads) at `res` m/px and `patch` px, then downloads each tile from `webmap`
    (s3:// / https:// / /vsis3/) into `out_dir`. Returns the written tile filenames.
    `res` is the web map's native resolution (from the tytonai app, beside the S3 link).
    """
    grid, _ = build_grid(shp_path, res, patch)
    return download_grid(grid, webmap, out_dir, bands=bands, workers=workers, skip_empty=skip_empty)


def plot_grid(grid: gpd.GeoDataFrame, study_area: gpd.GeoDataFrame, name: str, out_png: str | Path) -> None:
    """Save a PNG of the tile grid (blue) over the study-area outline (red).

    The cell size in the title is read from the grid geometry. Renders via the non-interactive
    Agg backend (no GUI/Tk objects) — safe to call alongside the threaded downloads.
    """
    from matplotlib.figure import Figure

    minx, _, maxx, _ = grid.geometry.iloc[0].bounds  # physical cell size from the first cell
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    grid.boundary.plot(ax=ax, color="tab:blue", linewidth=0.4)
    study_area.boundary.plot(ax=ax, color="tab:red", linewidth=1.5)
    ax.set_title(f"{name}: {len(grid)} tiles (~{maxx - minx:.1f} m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def preview_tiles(tiles_dir: str | Path, downscale: int = 16, ax=None, out_png: str | Path | None = None):
    """Coarse mosaic of downloaded tiles: each .tif read downsampled, placed at its
    real geo-extent. Pass an `ax` to draw several areas side by side. Returns the axes.
    """
    tiles_dir = Path(tiles_dir)
    files = sorted(tiles_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"no .tif tiles in {tiles_dir}")
    if ax is None:  # standalone -> Agg Figure (no GUI/Tk objects)
        from matplotlib.figure import Figure

        ax = Figure(figsize=(9, 9)).subplots()
    for f in files:
        with rasterio.open(f) as src:
            h, w = max(1, src.height // downscale), max(1, src.width // downscale)
            sel = [1, 2, 3] if src.count >= 3 else [1]  # RGB if available, else grayscale
            thumb = src.read(sel, out_shape=(len(sel), h, w)).transpose(1, 2, 0)
            thumb = thumb[:, :, 0] if thumb.shape[2] == 1 else thumb
            left, bottom, right, top = src.bounds
        ax.imshow(thumb, extent=(left, right, bottom, top), origin="upper")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title(f"{tiles_dir.name}: {len(files)} tiles (1/{downscale})")
    if out_png:
        ax.figure.savefig(out_png, dpi=100, bbox_inches="tight")
    return ax


# ════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, then run the lines below one at a time (Shift+Enter).
#  Auth: load_dotenv() exports the AWS_* keys GDAL needs for /vsis3.
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # exports .env -> os.environ -> GDAL /vsis3 picks up AWS_* automatically

    CONFIG = {
        "fgb_path": Path("study_area.fgb"),  # vector area defining the region to tile
        # Paste the s3:// link straight from the app — to_gdal_path() rewrites it to /vsis3.
        "webmap": "s3://c1cc6b74-6aa7-11f1-b078-5f348e776dae/7a89561f-ae92-4ed7-8c75-06e8ebf89702/RED_GREEN_BLUE_NIR_ALPHA_webmap.tif",
        "out_dir": Path("downloads/tiles"),
        "res": 0.1,    # metres / pixel
        "patch": 512,  # tile size in pixels
    }

    # 1) build the grid from the .fgb (cheap, local read) -------------------------------
    grid, study_area = build_grid(CONFIG["fgb_path"], CONFIG["res"], CONFIG["patch"])
    print(f"{len(grid)} tiles, CRS={study_area.crs}")

    # 2) sanity-plot the grid over the study area (cheap) -------------------------------
    plot_grid(grid, study_area, "study_area", "grid.png")

    # 3) download a small slice first to confirm S3 auth + extent (medium) --------------
    #    bands=None -> every band except alpha; [1,2,3]=RGB-only, [1,2,3,4]=RGB+NIR, [1]=mask.
    sample = grid.iloc[:20]
    written = download_grid(sample, CONFIG["webmap"], CONFIG["out_dir"], bands=[1, 2, 3], workers=8)
    print(f"wrote {len(written)} non-empty tiles (of {len(sample)})")

    # 4) full download (expensive) -----------------------------------------------------
    written = download_grid(grid, CONFIG["webmap"], CONFIG["out_dir"], bands=[1, 2, 3], workers=8)

    # 5) coarse preview of what landed (saved to preview.png — Agg, no GUI window) ------
    preview_tiles(CONFIG["out_dir"], downscale=16, out_png="preview.png")
