# tytonai_utils

Reusable helpers for interacting with the tytonai platform: S3 downloads, web map
tiling, model fetch, and annotation-mask rollup. Built to be installed straight from
GitHub across projects.

## Features

1. **Web map import** — cut local GeoTIFF tiles from an S3 web map (any band layout). ✅
2. **Manifest import** — download tiles listed in a manifest (image + mask `.npz`). 🚧
3. **Model fetch** — download a model from its config file. 🚧
4. **Mask rollup** — remap annotation-mask categories. 🚧

---

## Install

Install from GitHub. Pick the extras for the features you need:

```toml
# consumer pyproject.toml
[project]
dependencies = ["tytonai_utils[webmap]"]   # or [s3], or [all]

[tool.uv.sources]
tytonai_utils = { git = "https://github.com/clement-tyton/tytonai_utils.git" }
```

```bash
uv sync
# re-pull after the package is updated:
uv lock --upgrade-package tytonai_utils && uv sync
```

| Extra | Pulls in | For |
|---|---|---|
| `webmap` | geopandas, rasterio, matplotlib, rio_tiler, bbox_to_tile_grid | Feature 1 |
| `s3` | boto3 | Features 2–4 (S3 API) |
| `all` | everything above | — |

---

## Configuration

All credentials/config come from a `.env` file (loaded with `python-dotenv`). Copy
`.env.example` and fill it in. **Never commit `.env`** — it is gitignored.

The `AWS_*` keys are exactly GDAL's `/vsis3` config variables, so for the web map feature
`load_dotenv()` is the only auth step needed — GDAL reads them automatically. Key fields:

| Key | Purpose |
|---|---|
| `AWS_S3_ENDPOINT` | S3-compatible host, e.g. `s3.tytonai.com` |
| `AWS_HTTPS`, `AWS_VIRTUAL_HOSTING` | `YES` / `FALSE` (path-style addressing) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | credentials |
| `S3_FILE_BUCKET` | default data bucket |

---

## Feature 1 — Web map import (`tytonai_utils.webmap`)

A web map is a Cloud-Optimized GeoTIFF on S3. Tiles are **never** pre-downloaded: GDAL
opens the raster via `/vsis3` and does ranged GETs, reading only each tile's window.

### Quick start

```python
from pathlib import Path
from dotenv import load_dotenv
from tytonai_utils.webmap import build_grid, download_grid

load_dotenv()

grid, study_area = build_grid("study_area.fgb", res=0.1, patch=512)
written = download_grid(
    grid,
    "s3://bucket/id/RED_GREEN_BLUE_NIR_ALPHA_webmap.tif",
    "tiles_out",
    bands=[1, 2, 3],   # RGB; omit for all bands except alpha
)
```

### Functions

#### `to_gdal_path(link) -> str`
Normalize a web map link into a GDAL-openable path for ranged reads. Accepts and
converts:
- `s3://bucket/key` → `/vsis3/bucket/key`
- `http(s)://…` → `/vsicurl/…`
- `/vsis3/…`, `/vsicurl/…`, or a local path → passed through unchanged

You rarely call this directly — `download_grid` applies it for you, so you can paste the
`s3://` link straight from the app.

| Param | Type | Description |
|---|---|---|
| `link` | `str` | Web map link in any of the forms above |

Returns the GDAL path string.

#### `build_grid(fgb_path, res, patch) -> (grid, study_area)`
Read a `.fgb` (FlatGeobuf vector area), take its bounding box, and build a uniform
tile grid in pixel space anchored at the bbox top-left.

| Param | Type | Description |
|---|---|---|
| `fgb_path` | `str \| Path` | FlatGeobuf file defining the region to tile |
| `res` | `float` | Raster resolution, metres per pixel |
| `patch` | `int` | Tile size in pixels (square) |

Returns `(grid, study_area)` — both `GeoDataFrame`s. `grid` has one row per tile;
`study_area` is the original vector area (used for plotting / CRS).

#### `download_grid(grid, webmap, out_dir, bands=None, workers=8, skip_empty=True) -> list[str]`
Download every tile of `grid` from `webmap` into `out_dir` as georeferenced GeoTIFFs.
Reads run in parallel (I/O-bound; each worker thread opens its own rasterio handle).

| Param | Type | Description |
|---|---|---|
| `grid` | `GeoDataFrame` | Tile grid from `build_grid` (slice it to limit, e.g. `grid.iloc[:10]`) |
| `webmap` | `str` | Web map link (`s3://`, `https://`, `/vsis3/…`) — auto-normalized |
| `out_dir` | `str \| Path` | Output folder (created if missing) |
| `bands` | `list[int] \| None` | 1-based bands to write. `[1,2,3]`=RGB, `[1,2,3,4]`=RGB+NIR, `[1]`=mask. `None` = every band except alpha |
| `workers` | `int` | Parallel download threads (default 8) |
| `skip_empty` | `bool` | Skip tiles with no coverage (alpha→nodata→nonzero) |

Returns the list of written filenames (e.g. `["tile_00000.tif", …]`). Empty tiles are
omitted. Output preserves the source `dtype`, `nodata`, CRS, and per-band colour
interpretation — so masks stay clean integer rasters.

#### `plot_grid(grid, study_area, name, out_png, patch, res) -> None`
Save a PNG of the tile grid (blue) over the study-area outline (red) — a quick sanity
check of coverage and tile count before downloading.

| Param | Type | Description |
|---|---|---|
| `grid`, `study_area` | `GeoDataFrame` | Outputs of `build_grid` |
| `name` | `str` | Title label |
| `out_png` | `str \| Path` | Output PNG path |
| `patch`, `res` | `int`, `float` | For the title annotation (px and metres) |

#### `preview_tiles(tiles_dir, downscale=16, ax=None, out_png=None) -> Axes`
Coarse mosaic of downloaded tiles: each `.tif` read downsampled and placed at its real
geo-extent. Handles RGB (3+ bands) and single-band (grayscale) tiles.

| Param | Type | Description |
|---|---|---|
| `tiles_dir` | `str \| Path` | Folder of downloaded `.tif` tiles |
| `downscale` | `int` | Read 1/`downscale` resolution for speed |
| `ax` | `matplotlib Axes \| None` | Draw onto an existing axes (compare areas side by side) |
| `out_png` | `str \| Path \| None` | Save the figure if given |

Returns the matplotlib `Axes`. Raises `FileNotFoundError` if no tiles are present.

---

## S3 client (`tytonai_utils.s3`)

Shared boto3 client for the S3 API (used by features 2–4; feature 1 uses GDAL directly).

#### `make_s3_client(env=None) -> boto3 S3 client`
Return a boto3 S3 client configured for the tytonai endpoint. Pass `env` explicitly to
stay testable; defaults to `os.environ`. Forces path-style addressing when
`AWS_VIRTUAL_HOSTING` is `FALSE`.

```python
from dotenv import load_dotenv
from tytonai_utils.s3 import make_s3_client

load_dotenv()
s3 = make_s3_client()
s3.list_objects_v2(Bucket="...", MaxKeys=5)
```

---

## Development

```bash
git clone https://github.com/clement-tyton/tytonai_utils.git
cd tytonai_utils
uv sync --extra all --extra test
uv run pytest
```

Each module has a runnable `RUN` block at the bottom (`if __name__ == "__main__":`)
designed to be stepped through line by line in VSCode.
