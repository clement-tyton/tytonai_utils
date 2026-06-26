# tytonai_utils

Reusable helpers for interacting with the tytonai platform: S3 downloads, web map
tiling, model fetch, and annotation-mask rollup. Built to be installed straight from
GitHub across projects.

## Features

1. **Web map import** — cut local GeoTIFF tiles from an S3 web map (any band layout). ✅
2. **Manifest import** — download tiles listed in a manifest (image + mask `.npz`). ✅
3. **Model fetch** — download trained weights, or build a fresh model, from a config. ✅
4. **Mask rollup** — remap annotation-mask class ids to parent categories. ✅

---

## Tutorial

A full, runnable line-by-line walkthrough of **every feature** lives at
[`examples/tutorial.py`](examples/tutorial.py) — it states the input files each section
needs and is the fastest way to smoke-test the whole package after install.

## Install

Install from GitHub. The **minimal consumer `pyproject.toml`** is kept in the repo at
[`examples/pyproject.toml`](examples/pyproject.toml) — copy it into your project and trim
the extras to the features you use. It is updated whenever a new facility is added.

```toml
# minimal consumer pyproject.toml (see examples/pyproject.toml)
[project]
name = "your-project"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = ["tytonai_utils[webmap,s3,model]"]   # trim extras as needed

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
| `viz` | matplotlib | Visualization helpers (image/mask QA plots) |
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
| `res`, `patch` | `float`, `int` | The web map's native resolution (m/px — from the tytonai app, beside the S3 link) and tile size (px) |
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
| `res` | `float` | The **web map's native resolution** (metres/pixel) — read it in the tytonai app, in the same place you copy the S3 link |
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
    download_model_weights_from_config,
    load_model_with_fresh_head_from_config,
    load_trained_model_from_config,
)

load_dotenv()

# fresh model, no checkpoint: ImageNet encoder + random head
fresh = build_model_from_config("model_config.json")

# download the trained weights, then load — choose ONE of the two load modes:
weights = download_model_weights_from_config("model_config.json", "models/")

# (a) exactly as trained — full weights incl. head (same classes as the checkpoint)
trained = load_trained_model_from_config("model_config.json", weights)

# (b) finetune — reuse encoder+decoder, fresh random head for a new 3-class set
finetune = load_model_with_fresh_head_from_config(
    "model_config.json", weights, num_classes=3, freeze_encoder=True
)
```

> **Which load function?** Only the segmentation head depends on the class count; the
> encoder and decoder always transfer. Use `load_trained_model_from_config` when you want
> the model as-is (it errors if the checkpoint's class count differs). Use
> `load_model_with_fresh_head_from_config` when the class set changes — it reuses
> encoder+decoder and leaves a fresh random head.

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

#### `load_trained_model_from_config(config_path, weights_path, freeze_encoder=False, map_location="cpu") -> nn.Module`
Load the model **exactly as trained** — full weights including the segmentation head.
Loads strictly. **Raises** if the checkpoint's class count differs from the config's
`class_list` (use the fresh-head function for a different class set). Handles wrapped
checkpoints (`{"model": …}` from Fabric/training loops) and plain state-dicts.

| Param | Type | Description |
|---|---|---|
| `config_path` | `str \| Path` | Model config JSON |
| `weights_path` | `str \| Path` | Local `.pth` checkpoint (from the download fn) |
| `freeze_encoder` | `bool` | Set `requires_grad=False` on the encoder |
| `map_location` | `str` | Device for `torch.load` (e.g. `"cpu"`, `"cuda"`) |

#### `load_model_with_fresh_head_from_config(config_path, weights_path, num_classes=None, freeze_encoder=False, map_location="cpu") -> nn.Module`
Finetune setup: reuse the checkpoint's **encoder + decoder**, with a **fresh random head**
sized to `num_classes` (defaults to `len(class_list)`). Head weights in the checkpoint are
dropped; everything else must load or it **raises** (so a key mismatch can't silently
produce a random model).

| Param | Type | Description |
|---|---|---|
| `config_path` | `str \| Path` | Model config JSON |
| `weights_path` | `str \| Path` | Local `.pth` checkpoint |
| `num_classes` | `int \| None` | New head size (else `len(class_list)`) |
| `freeze_encoder` | `bool` | Set `requires_grad=False` on the encoder |
| `map_location` | `str` | Device for `torch.load` |

---

## Feature 4 — Mask rollup (`tytonai_utils.rollup`)

Remap annotation-mask class ids to parent categories. The **grouping** (which source classes
roll into which parent) is the source of truth; a remapping is that grouping + a **number
scheme** (`{parent_name: id}`). Ids not in the grouping go to a `nodata` value (default `0`).
Pure numpy + stdlib — **no extra needed**.

Two R&D schemes ship in the module:
- **7-class** — Ground, Shrub, Tree, Herb, Sedge, Tussock, Hummock (`RND_REMAP_7CLASS`).
- **6-class** — same, but Tussock + Hummock + generic Grass fold into Grass (`RND_REMAP_6CLASS`).

In both, unmapped classes (Biotic, Not Erosion, Erosion — plus Grass in the 7-class) → `nodata`.

### Quick start

```python
from tytonai_utils.rollup import RND_REMAP_7CLASS, rollup_mask, rollup_annotations

remapped = rollup_mask(mask, RND_REMAP_7CLASS)              # one numpy mask
rollup_annotations("annotations/", "dataset.json",          # a whole folder
                   RND_REMAP_7CLASS, out_dir="annotations_rnd7/")
```

### Data

| Name | What |
|---|---|
| `CLASS_NAMES` | source `{id: name}` (the org class list) |
| `ROLLUP_GROUPS_7CLASS` / `ROLLUP_GROUPS_6CLASS` | `{parent_name: [source_ids]}` groupings |
| `RND_TARGET_IDS_7CLASS` / `_6CLASS` | `{parent_name: target_id}` number schemes |
| `RND_REMAP_7CLASS` / `_6CLASS` | materialized `{source_id: target_id}` |
| `RND_NAMES_7CLASS` / `_6CLASS` | target `{id: name}` for the remapped masks |
| `NODATA` | value for unmapped ids (default `0`) |

### Functions

#### `build_remapping(groups, target_ids) -> dict[int, int]`
Flatten a grouping (`{name: [source_ids]}`) + a number scheme (`{name: id}`) into a
`{source_id: target_id}` dict. Use this to define a new remapping that reuses an existing
grouping with different numbers.

#### `target_id_to_name(groups, target_ids) -> dict[int, str]`
The remapped class list as `{target_id: parent_name}`.

#### `rollup_mask(mask, remapping, nodata=0) -> np.ndarray`
Vectorized remap of one integer mask. Ids not in `remapping` → `nodata` (pass `nodata=None`
to keep them unchanged). Returns a new array of the same dtype; input untouched.

#### `rollup_annotations(annotations_dir, manifest, remapping, out_dir, mask_key=None, nodata=0) -> list[Path]`
Roll up every mask `.npz` referenced by a manifest, writing remapped masks to `out_dir`.
Imagery files untouched; mask array key auto-detected (largest) unless `mask_key` given.

| Param | Type | Description |
|---|---|---|
| `annotations_dir` | `str \| Path` | Folder of downloaded mask `.npz` |
| `manifest` | `list[dict] \| str \| Path` | Tile list or path to `dataset.json` |
| `remapping` | `dict[int,int]` | `{source_id: target_id}` (e.g. `RND_REMAP_7CLASS`) |
| `out_dir` | `str \| Path` | Output folder for remapped masks |
| `mask_key` | `str \| None` | NPZ mask key; auto-detected if `None` |
| `nodata` | `int \| None` | Value for unmapped ids (default `0`; `None` = keep) |

---

## Visualization helpers

Quick QA plots. The web map helpers live in `tytonai_utils.webmap`; the image/mask helper
lives in `tytonai_utils.viz`. matplotlib comes from the `viz` (or `webmap`) extra.

#### `webmap.preview_tiles(tiles_dir, downscale=16, ax=None, out_png=None) -> Axes`
Downscaled overview mosaic of downloaded `.tif` tiles, placed at their real geo-extent.
Handles RGB (3+ bands) and single-band greyscale. Save with `out_png`. (See Feature 1.)

#### `webmap.plot_grid(grid, study_area, name, out_png, patch, res) -> None`
Draw the tile grid over the study-area outline — no download needed. (See Feature 1.)

#### `viz.plot_image_mask_pairs(annotations_dir, manifest, indexes=None, n=6, out_png=None, image_key=None, mask_key=None, bands=(0,1,2), cmap="tab20", seed=0) -> Figure`
Plot imagery tiles next to their annotation masks (the `.npz` pairs from Feature 2) for
visual QA. Select specific `indexes`, else `n` random tiles (seeded). Array keys are
auto-detected (largest array per file) unless `image_key`/`mask_key` are given.

| Param | Type | Description |
|---|---|---|
| `annotations_dir` | `str \| Path` | Folder of downloaded imagery + mask `.npz` |
| `manifest` | `list[dict] \| str \| Path` | Tile list (`read_manifest`) or path to `dataset.json` |
| `indexes` | `list[int] \| None` | Tiles to show; `None` → `n` random |
| `n` | `int` | Number of random tiles when `indexes` is `None` |
| `out_png` | `str \| Path \| None` | Save the figure if given |
| `image_key` / `mask_key` | `str \| None` | NPZ array keys; auto-detected if `None` |
| `bands` | `tuple[int,...]` | Which image channels to render as RGB |
| `cmap` | `str` | Colormap for the mask (categorical) |
| `seed` | `int` | RNG seed for reproducible random selection |

Returns the matplotlib `Figure`.

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
