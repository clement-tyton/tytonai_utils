# tytonai_utils

Reusable helpers for interacting with the tytonai platform: S3 downloads, web map
tiling, model fetch, and annotation-mask rollup. Built to be installed straight from
GitHub across projects.

## Features

1. **Web map import** — cut local GeoTIFF tiles from an S3 web map (any band layout). ✅
2. **Manifest import** — download tiles listed in a manifest (image + mask `.npz`). ✅
3. **Model fetch** — download trained weights, or build a fresh model, from a config. ✅
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
| `model` | torch, segmentation-models-pytorch, timm | Feature 3 (build/load model) |
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
from dotenv import load_dotenv
from tytonai_utils.webmap import download_webmap_from_shp

load_dotenv()

# one call: vector area + S3 link -> tiles
written = download_webmap_from_shp(
    "study_area.fgb",
    "s3://bucket/id/RED_GREEN_BLUE_NIR_ALPHA_webmap.tif",
    "tiles_out",
    res=0.1, patch=512,
    bands=[1, 2, 3],   # RGB; omit for all bands except alpha
)
```

### Functions

#### `download_webmap_from_shp(shp_path, webmap, out_dir, res, patch, bands=None, workers=8, skip_empty=True) -> list[str]`
High-level one-call entry: build the tile grid from a vector file, then download each
tile from the web map. Use this unless you need to inspect/modify the grid first.

| Param | Type | Description |
|---|---|---|
| `shp_path` | `str \| Path` | Vector area (`.shp` / `.fgb` / `.geojson` — anything geopandas reads) |
| `webmap` | `str` | Web map link (`s3://`, `https://`, `/vsis3/…`) — auto-normalized |
| `out_dir` | `str \| Path` | Output folder |
| `res`, `patch` | `float`, `int` | Resolution (m/px) and tile size (px) |
| `bands` | `list[int] \| None` | Bands to write (see `download_grid`) |
| `workers`, `skip_empty` | `int`, `bool` | Passed through to `download_grid` |

Returns the list of written tile filenames. Internally calls `build_grid` +
`download_grid` — use those directly for finer control (e.g. plotting the grid, slicing
to a subset before downloading).

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

## Feature 2 — Manifest / annotation download (`tytonai_utils.manifest`)

Read a `dataset.json` manifest (a list of tile dicts) and download every imagery + mask
`.npz` it references from S3 into a chosen `out_dir`. Uses boto3 (the `s3` extra) — no
`aws` CLI needed. Cache-aware: files already on disk are skipped unless `force=True`.

### Quick start

```python
from dotenv import load_dotenv
from tytonai_utils.manifest import download_annotations_from_dataset_manifest

load_dotenv()
download_annotations_from_dataset_manifest("monrovia/dataset.json", out_dir="monrovia/annotations")
```

### Functions

#### `read_manifest(manifest_path) -> list[dict]`
Load the `dataset.json` tile list (JSON array of tile dicts with `imagery_file`,
`mask_file`, `geotransform`, `srid`, `class_counts`, …).

| Param | Type | Description |
|---|---|---|
| `manifest_path` | `str \| Path` | Path to the `dataset.json` file |

#### `download_file(s3, key, dest, bucket, force=False) -> bool`
Download `s3://bucket/key` to `dest`. Returns `True` if a download happened, `False` if
the file was already cached on disk.

| Param | Type | Description |
|---|---|---|
| `s3` | boto3 client | From `make_s3_client()` |
| `key` | `str` | S3 object key (the manifest's `imagery_file` / `mask_file`) |
| `dest` | `Path` | Local destination (parent dirs created) |
| `bucket` | `str` | Source bucket |
| `force` | `bool` | Re-download even if `dest` exists |

#### `download_annotations_from_dataset_manifest(manifest_path, out_dir, bucket=None, force=False, workers=8, s3=None) -> Path`
Download every imagery + mask NPZ referenced by the manifest into `out_dir`, in parallel.

| Param | Type | Description |
|---|---|---|
| `manifest_path` | `str \| Path` | Path to `dataset.json` |
| `out_dir` | `str \| Path` | Output folder (created if missing) |
| `bucket` | `str \| None` | Source bucket; defaults to `$S3_FILE_BUCKET` |
| `force` | `bool` | Re-download cached files |
| `workers` | `int` | Parallel download threads (default 8) |
| `s3` | boto3 client \| None | Reuse a client; defaults to a fresh `make_s3_client()` |

Returns `out_dir`. Prints a summary (downloaded vs cached). Requires `load_dotenv()`
beforehand for the `AWS_*` creds + endpoint.

---

## Feature 3 — Model from config (`tytonai_utils.model`)

A model config (JSON) describes a `segmentation_models_pytorch` (smp) model — `model_type`,
`encoder_type`, `encoder_weights`, `bands` (→ in_channels), `class_list` (→ #classes), and
`epoch_file_key` (the `s3://…/...pth` trained weights). This feature downloads the weights
and/or instantiates the model — fresh, loaded, or reshaped for transfer learning.

Heavy deps are lazy-imported: the download function needs only the `s3` extra; the build/load
functions need the `model` extra (`torch` + `segmentation-models-pytorch`).

### Quick start

```python
from dotenv import load_dotenv
from tytonai_utils.model import (
    build_model_from_config,
    download_and_load_model_from_config,
    load_model_from_config,
)

load_dotenv()

# fresh model: ImageNet encoder + random head
fresh = build_model_from_config("model_config.json")

# download trained weights + load them (one call)
model = download_and_load_model_from_config("model_config.json", "models/")

# transfer learning: keep trained encoder/decoder, new 3-class head, frozen encoder
tl = load_model_from_config("model_config.json", "models/weights.pth",
                            num_classes=3, freeze_encoder=True)
```

### Functions

#### `read_model_config(config_path) -> dict`
Load a model config JSON.

#### `download_model_weights_from_config(config_path, out_dir, force=False, s3=None) -> Path`
Download the trained `.pth` referenced by the config into `out_dir`. The S3 link **and its
bucket** come from the config's `epoch_file_key` (not `$S3_FILE_BUCKET`). Cache-aware.
Returns the local weights path. (`s3` extra.)

#### `build_model_from_config(config_path, pretrained_encoder=True, num_classes=None) -> nn.Module`
Build the smp architecture from the config. Default = ImageNet-pretrained encoder + randomly
initialised decoder/head. `pretrained_encoder=False` → fully random. `num_classes` overrides
the head size (else `len(class_list)`). (`model` extra.)

| Param | Type | Description |
|---|---|---|
| `config_path` | `str \| Path` | Model config JSON |
| `pretrained_encoder` | `bool` | Load ImageNet encoder weights (head always random) |
| `num_classes` | `int \| None` | Override the head class count |

#### `load_model_from_config(config_path, weights_path, num_classes=None, freeze_encoder=False, strict=False, map_location="cpu") -> nn.Module`
Build the config's architecture and load trained weights from a `.pth`. Handles wrapped
checkpoints (`{"model": …}` from Fabric/training loops) and plain state-dicts.

| Param | Type | Description |
|---|---|---|
| `config_path` | `str \| Path` | Model config JSON |
| `weights_path` | `str \| Path` | Local `.pth` checkpoint (from the download fn) |
| `num_classes` | `int \| None` | New head size for transfer learning; with `strict=False` the encoder/decoder load and the mismatched head stays random |
| `freeze_encoder` | `bool` | Set `requires_grad=False` on the encoder |
| `strict` | `bool` | `load_state_dict` strictness (keep `False` when changing classes) |
| `map_location` | `str` | Device for `torch.load` (e.g. `"cpu"`, `"cuda"`) |

Returns the model; prints missing/unexpected key counts. (`model` extra.)

#### `download_and_load_model_from_config(config_path, weights_dir, num_classes=None, freeze_encoder=False, strict=False, force=False) -> nn.Module`
One call: `download_model_weights_from_config` then `load_model_from_config`. (Both extras.)

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
