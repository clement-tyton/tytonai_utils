"""tytonai_utils — full tutorial / smoke test of every feature.

Run this line by line in VSCode (Shift+Enter), top to bottom. Each section says exactly
which input files it needs. Sections are independent — skip any whose inputs you lack.

─────────────────────────────────────────────────────────────────────────────────────
RELOAD THE PACKAGE (run in a shell, in THIS project, after the package is re-pushed):

    uv lock --upgrade-package tytonai_utils && uv sync
    # consumer pyproject must declare:  dependencies = ["tytonai_utils[webmap,s3,model,viz]"]

─────────────────────────────────────────────────────────────────────────────────────
INPUT FILES NEEDED (edit CONFIG below to point at yours):

  .env                : AWS_* creds + S3_FILE_BUCKET + AWS_S3_ENDPOINT   (all features that hit S3)
  CONFIG["fgb"]       : a vector area (.fgb/.shp/.geojson)               (Feature 1)
  CONFIG["webmap"]    : an s3:// web map link                            (Feature 1)
  CONFIG["manifest"]  : a dataset.json manifest                          (Features 2, 4, viz)
  CONFIG["model_cfg"] : a model_config.json                             (Feature 3)
  (Features 4 & viz also need the annotations folder produced by Feature 2.)
─────────────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

from dotenv import load_dotenv

# .env must be in the working dir (or pass the path). Exports AWS_* for S3 + GDAL /vsis3.
load_dotenv()

# Note: importing tytonai_utils (below) auto-selects matplotlib's non-interactive Agg
# backend, so plots save to PNG and the threaded downloads never crash. Nothing to set up.

CONFIG = {
    "fgb": Path("study_area.fgb"),
    "webmap": "s3://c1cc6b74-6aa7-11f1-b078-5f348e776dae/7a89561f-ae92-4ed7-8c75-06e8ebf89702/RED_GREEN_BLUE_NIR_ALPHA_webmap.tif",
    "manifest": Path("inputs_tests/dataset.json"),
    "model_cfg": Path("inputs_tests/model_config.json"),
    "tiles_out": Path("downloads/tiles"),
    "annotations": Path("downloads/annotations"),
    "annotations_rnd7": Path("downloads/annotations_rnd7"),
    "models": Path("downloads/models"),
}


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE 1 — Web map import   (needs: .env, CONFIG["fgb"], CONFIG["webmap"])
#  Extra: [webmap]
# ════════════════════════════════════════════════════════════════════════════
from tytonai_utils.webmap import (
    build_grid, download_grid, download_webmap_from_shp, plot_grid, preview_tiles,
)

# 1a) build the grid from the vector area (cheap, local — no S3) --------------------
#     res = the web map's native resolution (m/px) — read it in the tytonai app, in the
#     same place you copy the S3 link. patch = tile size in pixels.
grid, study_area = build_grid(CONFIG["fgb"], res=0.1, patch=512)
print(f"{len(grid)} tiles, CRS={study_area.crs}")

# 1b) sanity-plot the grid over the area (saves grid.png) --------------------------
plot_grid(grid, study_area, "study_area", "grid.png", patch=512, res=0.1)

# 1c) download a 10-tile slice (confirms S3 auth + s3://->/vsis3 + ranged reads) ----
written = download_grid(grid.iloc[:10], CONFIG["webmap"], CONFIG["tiles_out"], bands=[1, 2, 3])
print(f"wrote {len(written)} tiles -> {CONFIG['tiles_out']}")

# 1d) one-call equivalent (build grid + download in one) ---------------------------
# written = download_webmap_from_shp(CONFIG["fgb"], CONFIG["webmap"], CONFIG["tiles_out"],
#                                    res=0.1, patch=512, bands=[1, 2, 3])

# 1e) overview mosaic of what landed (saves preview.png; RGB or greyscale) ---------
preview_tiles(CONFIG["tiles_out"], downscale=16, out_png="preview.png")


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE 2 — Manifest / annotation download   (needs: .env, CONFIG["manifest"])
#  Extra: [s3]
# ════════════════════════════════════════════════════════════════════════════
from tytonai_utils.manifest import download_annotations_from_dataset_manifest, read_manifest

# 2a) inspect the manifest (cheap, local) -----------------------------------------
tiles = read_manifest(CONFIG["manifest"])
print(f"{len(tiles)} tiles | first keys: {list(tiles[0])}")

# 2b) download every imagery + mask .npz into the annotations folder ----------------
download_annotations_from_dataset_manifest(CONFIG["manifest"], CONFIG["annotations"])


# ════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — image/mask QA   (needs: CONFIG["manifest"] + Feature 2 output)
#  Extra: [viz]
# ════════════════════════════════════════════════════════════════════════════
from tytonai_utils.rollup import CLASS_NAMES
from tytonai_utils.viz import plot_image_mask_pairs

# 3a) 6 random image|mask pairs, legend by source class name (saves pairs.png) -----
plot_image_mask_pairs(CONFIG["annotations"], CONFIG["manifest"], n=6,
                      class_names=CLASS_NAMES, out_png="pairs.png")

# 3b) specific tiles by index (saves pairs_idx.png) -------------------------------
plot_image_mask_pairs(CONFIG["annotations"], CONFIG["manifest"], indexes=[0, 1, 2],
                      class_names=CLASS_NAMES, out_png="pairs_idx.png")

# 3c) RGB + DSM + mask per sample (saves pairs_dsm.png) ---------------------------
plot_image_mask_pairs(CONFIG["annotations"], CONFIG["manifest"], n=3, show_dsm=True,
                      class_names=CLASS_NAMES, out_png="pairs_dsm.png")
print("saved pairs.png / pairs_idx.png / pairs_dsm.png — open them to inspect")


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE 4 — Mask rollup   (needs: CONFIG["manifest"] + Feature 2 output)
#  No extra (pure numpy)
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

from tytonai_utils.rollup import (
    CLASS_NAMES, RND_NAMES_6CLASS, RND_NAMES_7CLASS,
    RND_REMAP_6CLASS, RND_REMAP_7CLASS, rollup_annotations, rollup_mask,
)

# 4a) inspect the schemes (pure data, no files) -----------------------------------
print("7-class:", RND_NAMES_7CLASS)
print("6-class:", RND_NAMES_6CLASS)

# 4b) logic check on a synthetic mask ---------------------------------------------
fake = np.array([[10108, 10118, 7], [100, 10119, 14]])  # tree, cenchrus, grass / biotic, hummock, erosion
print("7-class:", rollup_mask(fake, RND_REMAP_7CLASS).tolist())  # grass/biotic/erosion -> 0
print("6-class:", rollup_mask(fake, RND_REMAP_6CLASS).tolist())  # cenchrus/hummock/grass -> 7

# 4c) roll up the downloaded masks into a new folder (7-class) ----------------------
rollup_annotations(CONFIG["annotations"], CONFIG["manifest"], RND_REMAP_7CLASS, CONFIG["annotations_rnd7"])

# 4d) verify: compare class ids before vs after on the same tile -------------------
plot_image_mask_pairs(CONFIG["annotations"], CONFIG["manifest"], indexes=[0],
                      class_names=CLASS_NAMES, out_png="mask_before.png")          # source names
plot_image_mask_pairs(CONFIG["annotations_rnd7"], CONFIG["manifest"], indexes=[0],
                      class_names=RND_NAMES_7CLASS, out_png="mask_after.png")      # rolled-up names
print("compare mask_before.png vs mask_after.png")


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE 3 — Model from config   (needs: .env, CONFIG["model_cfg"])
#  Extra: [model] (torch + smp). Building downloads ImageNet weights (needs internet).
# ════════════════════════════════════════════════════════════════════════════
from tytonai_utils.model import (
    build_model_from_config, download_model_weights_from_config,
    load_model_with_fresh_head_from_config, load_trained_model_from_config, read_model_config,
)

# 5a) inspect the config (cheap, local) -------------------------------------------
cfg = read_model_config(CONFIG["model_cfg"])
c = cfg["config"]
print(f"{cfg['model_name']} | {c['model_type']}/{c['encoder_type']} | "
      f"{len(c['bands'])} bands -> {len(cfg['class_list'])} classes")

# 5b) fresh model: ImageNet encoder + random head ---------------------------------
fresh = build_model_from_config(CONFIG["model_cfg"])
print(f"fresh {type(fresh).__name__}: {sum(p.numel() for p in fresh.parameters()) / 1e6:.1f}M params")

# 5c) download the trained weights .pth (bucket comes from the config) -------------
weights = download_model_weights_from_config(CONFIG["model_cfg"], CONFIG["models"])

# 5d) load EXACTLY as trained (errors if checkpoint classes != config class_list) --
trained = load_trained_model_from_config(CONFIG["model_cfg"], weights)

# 5e) finetune: reuse encoder+decoder, fresh random head for 7 new classes ---------
finetune = load_model_with_fresh_head_from_config(CONFIG["model_cfg"], weights,
                                                  num_classes=7, freeze_encoder=True)

print("\nTutorial complete — all features exercised.")
