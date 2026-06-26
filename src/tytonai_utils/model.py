"""Feature 3 — model from config: download trained weights, or build a fresh model.

A model config (JSON) describes a segmentation_models_pytorch (smp) model:
- config.model_type      e.g. "Unet"            -> smp.Unet
- config.encoder_type    e.g. "timm-res2net..." -> encoder_name
- config.encoder_weights e.g. "imagenet"        -> pretrained encoder
- config.bands           e.g. [RED,GREEN,BLUE]  -> in_channels
- class_list             e.g. [2,4,5,6,7,40]    -> number of output classes
- epoch_file_key         s3://bucket/key.pth    -> the trained weights

Entry points:
- download_model_weights_from_config       -> pull the trained .pth from S3 (`s3` extra).
- build_model_from_config                  -> fresh architecture (no checkpoint): ImageNet
  encoder + random head, or fully random (`model` extra: torch + smp).
- load_trained_model_from_config           -> the model EXACTLY as trained (full weights,
  head included); errors unless the checkpoint class count matches the config.
- load_model_with_fresh_head_from_config   -> finetune: reuse encoder + decoder, fresh
  RANDOM head sized to num_classes (for a different class set).

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


_HEAD_PREFIX = "segmentation_head"


def _checkpoint_num_classes(state_dict: dict) -> int | None:
    """Infer #classes from the checkpoint's segmentation-head conv weight, or None."""
    weight = state_dict.get(f"{_HEAD_PREFIX}.0.weight")
    return None if weight is None else weight.shape[0]


def _freeze_encoder(model: "nn.Module") -> None:
    """Set requires_grad=False on every encoder parameter (in place)."""
    for param in model.encoder.parameters():
        param.requires_grad = False


def load_trained_model_from_config(
    config_path: str | Path,
    weights_path: str | Path,
    freeze_encoder: bool = False,
    map_location: str = "cpu",
) -> "nn.Module":
    """Load the model EXACTLY as trained: full weights including the segmentation head.

    Only valid when the checkpoint's class count matches the config's class_list (which
    also guarantees class alignment, since you load with the same config that produced the
    weights). Raises if they differ — use load_model_with_fresh_head_from_config instead.
    Loads strictly so any key mismatch is a loud error. Requires the `model` extra.
    """
    import torch

    cfg = read_model_config(config_path)
    n_classes = len(cfg["class_list"])
    state_dict = _extract_state_dict(torch.load(weights_path, map_location=map_location))
    ckpt_classes = _checkpoint_num_classes(state_dict)
    if ckpt_classes is not None and ckpt_classes != n_classes:
        raise ValueError(
            f"checkpoint head has {ckpt_classes} classes but config.class_list has "
            f"{n_classes}; use load_model_with_fresh_head_from_config for a different "
            f"class set."
        )
    model = build_model_from_config(config_path, pretrained_encoder=False)
    model.load_state_dict(state_dict, strict=True)
    if freeze_encoder:
        _freeze_encoder(model)
    print(f"[model] loaded trained model ({n_classes} classes) <- {weights_path}")
    return model


def load_model_with_fresh_head_from_config(
    config_path: str | Path,
    weights_path: str | Path,
    num_classes: int | None = None,
    freeze_encoder: bool = False,
    map_location: str = "cpu",
) -> "nn.Module":
    """Finetune setup: reuse the checkpoint's encoder + decoder, with a fresh RANDOM head.

    The head is sized to `num_classes` (defaults to len(class_list)) and left randomly
    initialised — use this when the class set differs from the checkpoint's. Head weights
    in the checkpoint are dropped; everything else must load, otherwise it raises (so a key
    mismatch can't silently yield a random model). Requires the `model` extra.
    """
    import torch

    cfg = read_model_config(config_path)
    classes = num_classes if num_classes is not None else len(cfg["class_list"])
    model = build_model_from_config(config_path, pretrained_encoder=False, num_classes=classes)
    state_dict = _extract_state_dict(torch.load(weights_path, map_location=map_location))
    body = {k: v for k, v in state_dict.items() if not k.startswith(_HEAD_PREFIX)}
    result = model.load_state_dict(body, strict=False)
    missing_non_head = [k for k in result.missing_keys if not k.startswith(_HEAD_PREFIX)]
    if result.unexpected_keys or missing_non_head:
        raise ValueError(
            f"fresh-head load mismatch beyond the head: unexpected="
            f"{result.unexpected_keys[:5]}, missing(non-head)={missing_non_head[:5]}. "
            f"Checkpoint keys may not match the smp model (e.g. a key prefix)."
        )
    if freeze_encoder:
        _freeze_encoder(model)
    print(f"[model] loaded encoder+decoder, fresh random head ({classes} classes) <- {weights_path}")
    return model


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

    # 4) load the trained model EXACTLY as trained (full weights, head included) ---------
    #    raises if the checkpoint's class count != config.class_list
    trained = load_trained_model_from_config(CONFIG["config_path"], weights)

    # 5) finetune: reuse encoder+decoder, fresh random head for N new classes ------------
    finetune = load_model_with_fresh_head_from_config(
        CONFIG["config_path"], weights, num_classes=3, freeze_encoder=True
    )
