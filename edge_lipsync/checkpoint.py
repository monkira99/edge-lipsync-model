from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from edge_lipsync.dataset import manifest_sha256

TRAIN_CHECKPOINT_FORMAT = "edge_lipsync_duix_unet_train_v1"
EXPORT_CHECKPOINT_FORMAT = "duix_unet_hardcoded_blocks_ckpt_v2"
RESUME_CHECKPOINT_FORMAT = "edge_lipsync_duix_unet_resume_v1"

RESUME_REQUIRED_FIELDS = {
    "format",
    "model_state_dict",
    "optimizer_state_dict",
    "scaler_state_dict",
    "training_config",
    "dataset_identity",
    "dataset_root",
    "manifest_path",
    "runtime",
    "step",
    "epoch",
    "next_batch_index",
    "epoch_sample_indices",
    "data_order_generator_state",
    "best_val_loss",
    "early_stopping_best_val_loss",
    "validations_without_improvement",
    "early_stop_reason",
    "early_stop_step",
    "best_metrics",
    "best_model_state_dict",
    "metrics_history",
    "random_state",
    "init_weight_source",
    "provenance",
}


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


def clone_state_dict_to_cpu(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device="cpu", copy=True)
        for key, value in state_dict.items()
    }


def capture_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def make_model_checkpoint_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    face_size: int = 160,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": EXPORT_CHECKPOINT_FORMAT,
        "state_dict": clone_state_dict_to_cpu(state_dict),
        "face_size": int(face_size),
        "extra": extra or {},
        "spec_count": 165,
    }


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
    return make_training_checkpoint_from_state_dict(
        state_dict=model.state_dict(),
        training_config=training_config,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        step=step,
        epoch=epoch,
        metrics=metrics,
        init_weight_source=init_weight_source,
        provenance=provenance,
    )


def make_training_checkpoint_from_state_dict(
    *,
    state_dict: dict[str, torch.Tensor],
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
        "state_dict": clone_state_dict_to_cpu(state_dict),
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


def make_training_resume_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    training_config: dict[str, Any],
    dataset_identity: dict[str, Any],
    dataset_root: str | Path,
    manifest_path: str | Path,
    runtime: dict[str, Any],
    step: int,
    epoch: int,
    next_batch_index: int,
    epoch_sample_indices: list[int],
    data_order_generator_state: torch.Tensor,
    best_val_loss: float,
    early_stopping_best_val_loss: float,
    validations_without_improvement: int,
    best_metrics: dict[str, Any] | None,
    best_model_state_dict: dict[str, torch.Tensor],
    metrics_history: list[dict[str, Any]],
    random_state: dict[str, Any],
    init_weight_source: dict[str, Any],
    provenance: dict[str, Any],
    early_stop_reason: str = "",
    early_stop_step: int = 0,
) -> dict[str, Any]:
    return {
        "format": RESUME_CHECKPOINT_FORMAT,
        "model_state_dict": clone_state_dict_to_cpu(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "training_config": training_config,
        "dataset_identity": dataset_identity,
        "dataset_root": str(Path(dataset_root).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "runtime": runtime,
        "step": int(step),
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "epoch_sample_indices": [int(index) for index in epoch_sample_indices],
        "data_order_generator_state": data_order_generator_state.detach().cpu().clone(),
        "best_val_loss": float(best_val_loss),
        "early_stopping_best_val_loss": float(early_stopping_best_val_loss),
        "validations_without_improvement": int(validations_without_improvement),
        "early_stop_reason": str(early_stop_reason),
        "early_stop_step": int(early_stop_step),
        "best_metrics": dict(best_metrics) if best_metrics is not None else None,
        "best_model_state_dict": clone_state_dict_to_cpu(best_model_state_dict),
        "metrics_history": [dict(row) for row in metrics_history],
        "random_state": random_state,
        "init_weight_source": init_weight_source,
        "provenance": provenance,
    }


def load_training_resume_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid resume checkpoint payload: {path}")
    if checkpoint.get("format") != RESUME_CHECKPOINT_FORMAT:
        raise ValueError(
            "Unsupported resume checkpoint format: "
            f"{checkpoint.get('format')!r}"
        )
    missing = sorted(RESUME_REQUIRED_FIELDS - checkpoint.keys())
    if missing:
        raise ValueError(f"Resume checkpoint missing required fields: {missing}")
    return checkpoint


def make_export_checkpoint(
    *,
    model: torch.nn.Module,
    face_size: int,
    init_bin: str | Path,
    weight_load: dict[str, int],
) -> dict[str, Any]:
    return make_model_checkpoint_from_state_dict(
        model.state_dict(),
        face_size=face_size,
        extra={
            "init_bin": str(Path(init_bin).resolve()),
            "weight_load": weight_load,
        },
    )
