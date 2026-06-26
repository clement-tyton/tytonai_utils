"""Feature 4 — mask rollup: remap annotation-mask class ids to parent categories.

Source classes (the org class list) roll up into a small set of parent categories. The
*grouping* (which source ids belong to which parent) is the source of truth; a remapping is
that grouping + a *number scheme* (parent name -> id). Ids not in the grouping go to a
`nodata` value (default 0). Two R&D schemes are provided:

- 7-class: Ground, Shrub, Tree, Herb, Sedge, Tussock, Hummock.
- 6-class: same, but Tussock + Hummock (and the generic Grass leaf) fold into Grass.

Pure numpy + stdlib — no extra needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

NODATA = 0  # value for classes not covered by a remapping (0 is unused by the class list)

# ── Source class list: id -> human name (the id<->name dictionary) ───────────────────
CLASS_NAMES: dict[int, str] = {
    10108: "Chile_Tree1", 100: "Biotic", 10109: "Chile_Tree2", 4: "Shrub",
    10110: "Chile_Tree3", 5: "Tree", 6: "Herb", 7: "Grass", 301: "Tussock",
    40: "Sedge", 200: "Abiotic", 2: "Ground", 9: "Generic debris",
    10111: "Chile_Shrub1", 201: "Water", 10115: "Eucalyptus spp",
    10117: "Melaleuca Argentea", 10118: "Cenchrus spp", 10119: "Hummock Grass",
    10120: "Aerva javanica", 10121: "Annual herbs and grasses",
    10122: "Phoenix dactylifera", 10123: "Eucalyptus camaldulensis",
    10124: "Eucalyptus victrix", 10125: "Mulga", 1: "Not Erosion", 14: "Erosion",
    10126: "Calotropis procera",
}


def build_remapping(groups: dict[str, list[int]], target_ids: dict[str, int]) -> dict[int, int]:
    """Flatten a grouping + a number scheme into a {source_id: target_id} dict."""
    return {src: target_ids[name] for name, srcs in groups.items() for src in srcs}


def target_id_to_name(groups: dict[str, list[int]], target_ids: dict[str, int]) -> dict[int, str]:
    """The remapped class list as {target_id: parent_name}."""
    return {target_ids[name]: name for name in groups}


# ── 7-class grouping + R&D number scheme ─────────────────────────────────────────────
ROLLUP_GROUPS_7CLASS: dict[str, list[int]] = {
    "Ground":  [2, 200, 9, 201],                                  # Ground, Abiotic, Generic debris, Water
    "Shrub":   [4, 10111, 10120, 10125, 10126],                   # Shrub, Chile_Shrub1, Aerva javanica, Mulga, Calotropis procera
    "Tree":    [5, 10108, 10109, 10110, 10115, 10117, 10122, 10123, 10124],
    "Herb":    [6, 10121],                                        # Herb, Annual herbs and grasses
    "Sedge":   [40],
    "Tussock": [301, 10118],                                      # Tussock, Cenchrus spp
    "Hummock": [10119],                                           # Hummock Grass
}
RND_TARGET_IDS_7CLASS: dict[str, int] = {
    "Ground": 2, "Shrub": 4, "Tree": 5, "Herb": 6,
    "Sedge": 40, "Tussock": 301, "Hummock": 10119,
}

# ── 6-class grouping: Tussock + Hummock (+ the generic Grass leaf 7) fold into Grass ──
ROLLUP_GROUPS_6CLASS: dict[str, list[int]] = {
    **{name: ROLLUP_GROUPS_7CLASS[name] for name in ("Ground", "Shrub", "Tree", "Herb", "Sedge")},
    "Grass": [7, *ROLLUP_GROUPS_7CLASS["Tussock"], *ROLLUP_GROUPS_7CLASS["Hummock"]],
}
RND_TARGET_IDS_6CLASS: dict[str, int] = {
    "Ground": 2, "Shrub": 4, "Tree": 5, "Herb": 6, "Sedge": 40, "Grass": 7,
}

# Materialized remappings + their id<->name dicts.
RND_REMAP_7CLASS: dict[int, int] = build_remapping(ROLLUP_GROUPS_7CLASS, RND_TARGET_IDS_7CLASS)
RND_NAMES_7CLASS: dict[int, str] = target_id_to_name(ROLLUP_GROUPS_7CLASS, RND_TARGET_IDS_7CLASS)
RND_REMAP_6CLASS: dict[int, int] = build_remapping(ROLLUP_GROUPS_6CLASS, RND_TARGET_IDS_6CLASS)
RND_NAMES_6CLASS: dict[int, str] = target_id_to_name(ROLLUP_GROUPS_6CLASS, RND_TARGET_IDS_6CLASS)


def rollup_mask(mask: np.ndarray, remapping: dict[int, int], nodata: int | None = NODATA) -> np.ndarray:
    """Remap integer class ids in a mask via {source_id: target_id}, vectorized.

    Ids not in `remapping` are set to `nodata` (default 0); pass nodata=None to keep them
    unchanged. Returns a new array of the same dtype — the input is not modified.
    """
    mask = np.asarray(mask)
    max_id = int(mask.max(initial=0))
    if remapping:
        max_id = max(max_id, max(remapping))
    lut = np.arange(max_id + 1) if nodata is None else np.full(max_id + 1, nodata, dtype=np.int64)
    for src, dst in remapping.items():
        lut[src] = dst
    return lut[mask].astype(mask.dtype)


def _read_manifest(manifest) -> list[dict]:
    """Accept a manifest list or a path to dataset.json (stdlib only)."""
    if isinstance(manifest, (str, Path)):
        with open(manifest) as f:
            return json.load(f)
    return manifest


def _rollup_mask_file(src: Path, dst: Path, remapping: dict[int, int], mask_key: str | None,
                      nodata: int | None) -> None:
    """Roll up one mask .npz -> dst, preserving any other arrays in the file."""
    with np.load(src) as npz:
        data = {k: npz[k] for k in npz.files}
    key = mask_key if mask_key is not None else max(data, key=lambda k: data[k].size)
    data[key] = rollup_mask(data[key], remapping, nodata)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **data)


def rollup_annotations(
    annotations_dir: str | Path,
    manifest,
    remapping: dict[int, int],
    out_dir: str | Path,
    mask_key: str | None = None,
    nodata: int | None = NODATA,
) -> list[Path]:
    """Roll up every mask .npz referenced by a manifest, writing remapped masks to out_dir.

    `manifest` is the tile list or a path to dataset.json. Imagery files are untouched; only
    each tile's mask_file is remapped. The mask array key is auto-detected (largest array)
    unless mask_key is given. Ids not in `remapping` go to `nodata`. Returns written paths.
    """
    annotations_dir, out_dir = Path(annotations_dir), Path(out_dir)
    written = []
    for tile in tqdm(_read_manifest(manifest), desc="rollup masks"):
        dst = out_dir / tile["mask_file"]
        _rollup_mask_file(annotations_dir / tile["mask_file"], dst, remapping, mask_key, nodata)
        written.append(dst)
    print(f"[rollup] remapped {len(written)} masks -> {out_dir}")
    return written


# ════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1) inspect both R&D schemes (cheap, pure data) -----------------------------------
    print("7-class:", RND_NAMES_7CLASS)
    print("6-class:", RND_NAMES_6CLASS)

    # 2) remap a synthetic mask (logic check, no files) --------------------------------
    #    tree, cenchrus, grass / biotic, hummock-grass, erosion
    fake = np.array([[10108, 10118, 7], [100, 10119, 14]])
    print("before :", fake.tolist())
    print("7-class:", rollup_mask(fake, RND_REMAP_7CLASS).tolist())  # grass/biotic/erosion -> 0
    print("6-class:", rollup_mask(fake, RND_REMAP_6CLASS).tolist())  # cenchrus/hummock/grass -> 7; biotic/erosion -> 0

    # 3) roll up a real downloaded annotations folder (needs Feature 2 output) ----------
    CONFIG = {
        "manifest_path": Path("inputs_tests/dataset.json"),
        "annotations_dir": Path("input_site_data/monrovia/annotations"),
        "out_dir": Path("input_site_data/monrovia/annotations_rnd7"),
    }
    written = rollup_annotations(
        CONFIG["annotations_dir"], CONFIG["manifest_path"], RND_REMAP_7CLASS, CONFIG["out_dir"]
    )
