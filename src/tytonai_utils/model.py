"""Feature 3 — model from config: download trained weights, or build a fresh model.

A model config (JSON) describes a segmentation_models_pytorch (smp) model:
- config.model_type      e.g. "Unet"            -> smp.Unet
- config.encoder_type    e.g. "timm-res2net..." -> encoder_name
- config.encoder_weights e.g. "imagenet"        -> pretrained encoder
- config.bands           e.g. [RED,GREEN,BLUE]  -> in_channels
- class_list             e.g. [2,4,5,6,7,40]    -> number of output classes
- epoch_file_key         s3://bucket/key.pth    -> the trained weights

Entry points:
- download_model_weights_from_config   -> pull the trained .pth from S3 (`s3` extra).
- build_model_from_config              -> fresh architecture: ImageNet encoder + random
  head, or fully random (`model` extra: torch + smp).
- load_model_from_config               -> build + load trained weights; pass num_classes
  for transfer learning (new head, encoder/decoder loaded via strict=False).
- download_and_load_model_from_config  -> one call: download then load (both extras).

Heavy deps (boto3, torch, smp) are imported lazily so this module loads with neither extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


def read_model_config(config_path: str | Path) -> dict:
    """Load a model config JSON."""
    with open(config_path) as f:
        return json.load(f)


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/a/b.pth' -> ('bucket', 'a/b.pth')."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def download_model_weights_from_config(
    config_path: str | Path,
    out_dir: str | Path,
    force: bool = False,
    s3=None,
) -> Path:
    """Download the trained weights (.pth) referenced by a model config into out_dir.

    The s3:// link AND its bucket come from the config's `epoch_file_key` (not
    $S3_FILE_BUCKET). Cache-aware. Returns the local path to the weights file.
    """
    cfg = read_model_config(config_path)
    bucket, key = _split_s3_uri(cfg["epoch_file_key"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(key).name
    if dest.exists() and not force:
        print(f"[model] cached -> {dest}")
        return dest
    if s3 is None:
        from tytonai_utils.s3 import make_s3_client

        s3 = make_s3_client()
    s3.download_file(bucket, key, str(dest))
    print(f"[model] downloaded -> {dest}")
    return dest


def build_model_from_config(
    config_path: str | Path,
    pretrained_encoder: bool = True,
    num_classes: int | None = None,
) -> "nn.Module":
    """Build the smp model described by a config, from scratch (randomly-initialised head).

    When pretrained_encoder (default), the encoder loads its ImageNet weights and the
    decoder + segmentation head are randomly initialised — i.e. "fresh, ready to train".
    Set pretrained_encoder=False for a fully-random model. `num_classes` overrides the head
    size (defaults to len(class_list)). in_channels = len(config.bands).
    Returns a torch.nn.Module. Requires the `model` extra.
    """
    import segmentation_models_pytorch as smp

    cfg = read_model_config(config_path)
    c = cfg["config"]
    model_cls = getattr(smp, c["model_type"])  # e.g. smp.Unet
    return model_cls(
        encoder_name=c["encoder_type"],
        encoder_depth=c.get("encoder_depth", 5),
        encoder_weights=c["encoder_weights"] if pretrained_encoder else None,
        in_channels=len(c["bands"]),
        classes=num_classes if num_classes is not None else len(cfg["class_list"]),
        activation=c.get("activation"),
    )


def _extract_state_dict(checkpoint) -> dict:
    """Pull the model state-dict out of a raw torch.load result.

    Handles plain state-dicts and wrapped checkpoints ({"model"/"state_dict": ...},
    as saved by lightning Fabric / common training loops).
    """
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                inner = checkpoint[key]
                return inner.state_dict() if hasattr(inner, "state_dict") else inner
    return checkpoint  # assume it is already a state-dict


def load_model_from_config(
    config_path: str | Path,
    weights_path: str | Path,
    num_classes: int | None = None,
    freeze_encoder: bool = False,
    strict: bool = False,
    map_location: str = "cpu",
) -> "nn.Module":
    """Build the config's architecture and load trained weights from a .pth checkpoint.

    Transfer learning: pass `num_classes` to get a fresh head of that size — with
    strict=False the checkpoint's encoder + decoder load while the mismatched head stays
    randomly initialised. `freeze_encoder` sets requires_grad=False on the encoder.
    Returns a torch.nn.Module. Requires the `model` extra.
    """
    import torch

    model = build_model_from_config(config_path, pretrained_encoder=False, num_classes=num_classes)
    state_dict = _extract_state_dict(torch.load(weights_path, map_location=map_location))
    result = model.load_state_dict(state_dict, strict=strict)
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    print(
        f"[model] loaded {weights_path} "
        f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
    )
    return model


def download_and_load_model_from_config(
    config_path: str | Path,
    weights_dir: str | Path,
    num_classes: int | None = None,
    freeze_encoder: bool = False,
    strict: bool = False,
    force: bool = False,
) -> "nn.Module":
    """One call: download the config's weights from S3, then load them into the model.

    Convenience wrapper over download_model_weights_from_config + load_model_from_config.
    Requires both the `s3` and `model` extras.
    """
    weights_path = download_model_weights_from_config(config_path, weights_dir, force=force)
    return load_model_from_config(
        config_path, weights_path, num_classes=num_classes,
        freeze_encoder=freeze_encoder, strict=strict,
    )


# ════════════════════════════════════════════════════════════════════════════
#  RUN — edit CONFIG, run the lines below one at a time (Shift+Enter).
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # AWS_* creds + endpoint (for the weights download)

    CONFIG = {
        "config_path": Path("inputs_tests/model_config.json"),
        "out_dir": Path("models"),
    }

    # 1) inspect the config (cheap, local) ---------------------------------------------
    cfg = read_model_config(CONFIG["config_path"])
    c = cfg["config"]
    print(f"{cfg['model_name']} | {c['model_type']} / {c['encoder_type']} | "
          f"{len(c['bands'])} bands -> {len(cfg['class_list'])} classes")

    # 2) build a fresh model: ImageNet encoder + random head (needs `model` extra) ------
    fresh = build_model_from_config(CONFIG["config_path"])
    n_params = sum(p.numel() for p in fresh.parameters())
    print(f"fresh {type(fresh).__name__}: {n_params / 1e6:.1f}M params")

    # 3) download the trained weights .pth (needs `s3` extra + S3 creds) ----------------
    weights = download_model_weights_from_config(CONFIG["config_path"], CONFIG["out_dir"])
    print("weights ->", weights)

    # 4) load the trained model as-is (same classes as the checkpoint) ------------------
    trained = load_model_from_config(CONFIG["config_path"], weights)

    # 5) transfer learning: keep encoder/decoder weights, fresh head for N new classes --
    transfer = load_model_from_config(CONFIG["config_path"], weights, num_classes=3, freeze_encoder=True)

    # one-call equivalent of 3+4: download then load -----------------------------------
    model = download_and_load_model_from_config(CONFIG["config_path"], CONFIG["out_dir"])
