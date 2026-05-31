from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from edge_lipsync.dataset import manifest_sha256

TRAIN_CHECKPOINT_FORMAT = "edge_lipsync_duix_unet_train_v1"
EXPORT_CHECKPOINT_FORMAT = "duix_unet_hardcoded_blocks_ckpt_v2"


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def make_training_checkpoint(
    *,
    model: torch.nn.Module,
    training_config: dict[str, Any],
    dataset_root: str | Path,
    manifest_path: str | Path,
    step: int,
    epoch: int,
    metrics: dict[str, float | int],
    init_weight_source: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    return {
        "format": TRAIN_CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "training_config": training_config,
        "dataset_root": str(Path(dataset_root).resolve()),
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": manifest_sha256(manifest),
        "step": int(step),
        "epoch": int(epoch),
        "metrics": metrics,
        "init_weight_source": init_weight_source,
        "provenance": provenance or {},
    }


def make_export_checkpoint(
    *,
    model: torch.nn.Module,
    face_size: int,
    init_bin: str | Path,
    weight_load: dict[str, int],
) -> dict[str, Any]:
    return {
        "format": EXPORT_CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "face_size": int(face_size),
        "extra": {
            "init_bin": str(Path(init_bin).resolve()),
            "weight_load": weight_load,
        },
        "spec_count": 165,
    }
