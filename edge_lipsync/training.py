from __future__ import annotations

import csv
import inspect
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset, default_collate

from edge_lipsync.checkpoint import atomic_torch_save, make_training_checkpoint
from edge_lipsync.dataset import DuixHFDataset, DuixManifestDataset, manifest_sha256
from edge_lipsync.eval import render_validation_artifacts
from edge_lipsync.hub import HubArtifact, pull_dataset_snapshot, push_model_artifacts
from edge_lipsync.losses import (
    charbonnier_loss,
    combined_reconstruction_loss,
    combined_training_loss,
    mouth_weighted_l1,
)
from edge_lipsync.metrics import (
    LPIPSEvaluator,
    image_mae,
    image_psnr,
    image_ssim,
    lpips_face_and_mouth,
    mouth_mae,
    mouth_psnr,
    mouth_ssim,
    mouth_temporal_error,
    shift_audio_window,
)
from edge_lipsync.model import DuixUNet, load_ckpt
from edge_lipsync.sources import resolve_dataset_source, resolve_model_source
from edge_lipsync.tracking import WandbConfig, create_tracker

__all__ = ["HubArtifact"]


@dataclass(frozen=True)
class TrainConfig:
    run_dir: str
    dataset_root: str = ""
    manifest: str = "manifest.jsonl"
    init_bin: str = ""
    init_ckpt: str = ""
    device: str = "auto"
    precision: str = "auto"
    batch_size: int = 2
    num_workers: int = 0
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_steps: int = 1000
    warmup_steps: int = 100
    stabilization_steps: int = 100
    stabilization_lr_scale: float = 0.1
    validation_interval: int = 100
    checkpoint_interval: int = 100
    log_interval: int = 10
    lpips_enabled: bool = False
    lpips_net: str = "alex"
    lpips_face_loss_weight: float = 0.0
    lpips_mouth_loss_weight: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    media_eval_on_best: bool = True
    media_eval_clip_count: int = 2
    media_eval_clip_ids: tuple[str, ...] = ()
    media_eval_max_frames_per_clip: int = 50
    media_eval_fps: float = 25.0
    media_eval_log_to_wandb: bool = False
    hf_dataset_repo: str = ""
    hf_dataset_revision: str = ""
    hf_dataset_local_dir: str = ""
    hf_cache_dir: str = ""
    hf_init_model_repo: str = ""
    hf_init_model_filename: str = "best.pt"
    hf_model_repo: str = ""
    hf_model_private: bool = True
    hf_resume_upload_interval: int = 0
    resume_hf_model_repo: str = ""
    resume_hf_model_revision: str = ""
    wandb_mode: str = "disabled"
    wandb_project: str = "edge-lipsync-model"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_group: str = ""
    wandb_tags: tuple[str, ...] = ()
    wandb_notes: str = ""
    wandb_dir: str = ""


@dataclass(frozen=True)
class PreparedTrainingDatasets:
    train_dataset: Any
    val_dataset: Any
    dataset_root: Path
    manifest_path: Path
    provenance: dict[str, Any]


@dataclass(frozen=True)
class MediaEvalSelection:
    dataset: Any
    clip_ids: tuple[str, ...]
    indices: tuple[int, ...]


TRAJECTORY_CRITICAL_FIELDS = (
    "batch_size",
    "learning_rate",
    "weight_decay",
    "max_steps",
    "warmup_steps",
    "stabilization_steps",
    "stabilization_lr_scale",
    "validation_interval",
    "lpips_enabled",
    "lpips_net",
    "lpips_face_loss_weight",
    "lpips_mouth_loss_weight",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "precision",
)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_mixed_precision(config: TrainConfig, device: torch.device) -> bool:
    if config.precision not in {"auto", "fp32", "mixed"}:
        raise ValueError(f"Unsupported precision={config.precision!r}")
    if config.precision == "mixed" and device.type != "cuda":
        raise ValueError("precision='mixed' requires a CUDA device")
    return device.type == "cuda" and config.precision in {"auto", "mixed"}


def validate_training_config(config: TrainConfig) -> None:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.validation_interval <= 0 or config.checkpoint_interval <= 0:
        raise ValueError("validation_interval and checkpoint_interval must be positive")
    if config.log_interval < 0:
        raise ValueError("log_interval must be >= 0")
    if config.lpips_net not in {"alex", "vgg", "squeeze"}:
        raise ValueError(f"Unsupported lpips_net={config.lpips_net!r}")
    if config.lpips_face_loss_weight < 0 or config.lpips_mouth_loss_weight < 0:
        raise ValueError("LPIPS loss weights must be non-negative")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0")
    if config.media_eval_on_best and config.media_eval_fps <= 0:
        raise ValueError("media_eval_fps must be positive")
    if config.hf_resume_upload_interval < 0:
        raise ValueError("hf_resume_upload_interval must be >= 0")
    if config.hf_resume_upload_interval > 0 and not config.hf_model_repo:
        raise ValueError("hf_model_repo is required when hf_resume_upload_interval is positive")

    initialization_sources = (
        config.init_bin,
        config.init_ckpt,
        config.hf_init_model_repo,
    )
    if config.resume_hf_model_repo:
        if any(initialization_sources):
            raise ValueError(
                "Resume mode does not accept init_bin, init_ckpt, or hf_init_model_repo"
            )
    elif sum(bool(value) for value in initialization_sources) != 1:
        raise ValueError("Set exactly one of init_bin, init_ckpt, or hf_init_model_repo")


def dataset_identity_for_training(
    prepared: PreparedTrainingDatasets,
) -> dict[str, Any]:
    provenance = prepared.provenance
    fingerprints = provenance.get("fingerprints")
    if isinstance(fingerprints, dict):
        return {
            "kind": "huggingface",
            "repo_id": str(provenance.get("repo_id", "")),
            "resolved_ref": str(provenance.get("resolved_ref", "")),
            "fingerprints": dict(fingerprints),
        }
    return {
        "kind": "manifest",
        "manifest_sha256": manifest_sha256(prepared.manifest_path),
    }


def validate_resume_compatibility(
    checkpoint: dict[str, Any],
    *,
    config: TrainConfig,
    dataset_identity: dict[str, Any],
    device: torch.device,
    mixed_precision: bool,
) -> None:
    saved_config = checkpoint["training_config"]
    mismatches = [
        field
        for field in TRAJECTORY_CRITICAL_FIELDS
        if saved_config.get(field) != getattr(config, field)
    ]
    if mismatches:
        raise ValueError(
            "Resume checkpoint training config mismatch: "
            + ", ".join(sorted(mismatches))
        )
    if checkpoint["dataset_identity"] != dataset_identity:
        raise ValueError("Dataset identity does not match resume checkpoint")

    runtime = checkpoint["runtime"]
    saved_device_type = str(runtime.get("device_type", ""))
    if saved_device_type != device.type:
        raise ValueError(
            "Resume checkpoint device type mismatch: "
            f"saved={saved_device_type!r} current={device.type!r}"
        )
    saved_mixed_precision = bool(runtime.get("mixed_precision", False))
    if saved_mixed_precision != mixed_precision:
        raise ValueError(
            "Resume checkpoint mixed precision mode does not match current runtime"
        )
    step = int(checkpoint["step"])
    if step >= config.max_steps:
        raise ValueError(
            f"Resume checkpoint already reached max_steps: step={step} "
            f"max_steps={config.max_steps}"
        )


def validate_batch_shapes(batch: dict[str, Any]) -> None:
    expected = {
        "face": (6, 160, 160),
        "audio": (20, 256),
        "target": (3, 160, 160),
    }
    for key, tail_shape in expected.items():
        tensor = batch.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{key} must be a torch.Tensor")
        if tuple(tensor.shape[1:]) != tail_shape:
            raise ValueError(
                f"Invalid {key} shape={tuple(tensor.shape)}, expected=[B,{tail_shape}]"
            )


def generate_epoch_sample_indices(
    dataset_size: int,
    generator: torch.Generator,
) -> list[int]:
    if dataset_size <= 0:
        raise ValueError("Training dataset must be non-empty")
    return torch.randperm(dataset_size, generator=generator).tolist()


def remaining_epoch_sample_indices(
    epoch_sample_indices: list[int],
    *,
    batch_size: int,
    next_batch_index: int,
) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if next_batch_index < 0:
        raise ValueError("next_batch_index must be non-negative")
    return epoch_sample_indices[next_batch_index * batch_size :]


def collate_training_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty training batch")
    return {
        "face": default_collate([sample["face"] for sample in samples]),
        "audio": default_collate([sample["audio"] for sample in samples]),
        "target": default_collate([sample["target"] for sample in samples]),
        "meta": [sample.get("meta", {}) for sample in samples],
    }


def _sample_weights_from_metadata(
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor | None:
    metadata = batch.get("meta")
    if not isinstance(metadata, list):
        return None
    if not any(isinstance(meta, dict) and "sample_weight" in meta for meta in metadata):
        return None
    weights = [
        float(meta.get("sample_weight", 1.0)) if isinstance(meta, dict) else 1.0
        for meta in metadata
    ]
    return torch.tensor(weights, device=device, dtype=torch.float32)


def _loss_accepts_sample_weight(loss_fn: Callable[..., torch.Tensor]) -> bool:
    try:
        signature = inspect.signature(loss_fn)
    except (TypeError, ValueError):
        return True
    return any(
        name == "sample_weight" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for name, parameter in signature.parameters.items()
    )


def _forward_model(
    model: torch.nn.Module,
    face: torch.Tensor,
    audio: torch.Tensor | None,
) -> torch.Tensor:
    if audio is None:
        return model(face)
    return model(face, audio)


def run_train_step(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Callable[..., torch.Tensor],
    audio_optional: bool = False,
    mixed_precision: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    face = batch["face"].to(device=device, dtype=torch.float32)
    target = batch["target"].to(device=device, dtype=torch.float32)
    audio = batch.get("audio")
    audio_tensor = None if audio is None else audio.to(device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=mixed_precision):
        pred = _forward_model(model, face, audio_tensor if not audio_optional else None)
        sample_weight = _sample_weights_from_metadata(batch, device=device)
        loss = (
            loss_fn(pred, target, sample_weight=sample_weight)
            if sample_weight is not None and _loss_accepts_sample_weight(loss_fn)
            else loss_fn(pred, target)
        )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {float(loss.detach().cpu())}")
    if scaler is not None and mixed_precision:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    mixed_precision: bool = False,
    lpips_evaluator: torch.nn.Module | None = None,
    lpips_face_loss_weight: float = 0.0,
    lpips_mouth_loss_weight: float = 0.0,
) -> dict[str, float]:
    if lpips_face_loss_weight < 0 or lpips_mouth_loss_weight < 0:
        raise ValueError("LPIPS loss weights must be non-negative")
    if (lpips_face_loss_weight > 0 or lpips_mouth_loss_weight > 0) and lpips_evaluator is None:
        raise ValueError("LPIPS evaluator is required when LPIPS loss weights are positive")
    model.eval()
    reconstruction: list[float] = []
    total: list[float] = []
    mouth: list[float] = []
    temporal: list[float] = []
    mae: list[float] = []
    psnr: list[float] = []
    ssim: list[float] = []
    mouth_mae_values: list[float] = []
    mouth_psnr_values: list[float] = []
    mouth_ssim_values: list[float] = []
    lpips_face_values: list[float] = []
    lpips_mouth_values: list[float] = []
    mouth_temporal: list[float] = []
    audio_sensitivity: list[float] = []
    audio_shift_mouth_mae_delta: list[float] = []
    previous_by_clip: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}
    for batch in loader:
        face = batch["face"].to(device=device, dtype=torch.float32)
        audio = batch["audio"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            pred = model(face, audio)
            shifted_pred = model(face, shift_audio_window(audio))
            reconstruction_total = combined_reconstruction_loss(pred, target)
            reconstruction.append(float(charbonnier_loss(pred, target).cpu()))
            mouth.append(float(mouth_weighted_l1(pred, target).cpu()))
        mae.extend(image_mae(pred, target).cpu().tolist())
        psnr.extend(image_psnr(pred, target).cpu().tolist())
        ssim.extend(image_ssim(pred, target).cpu().tolist())
        current_mouth_mae = mouth_mae(pred, target)
        shifted_mouth_mae = mouth_mae(shifted_pred, target)
        mouth_mae_values.extend(current_mouth_mae.cpu().tolist())
        mouth_psnr_values.extend(mouth_psnr(pred, target).cpu().tolist())
        mouth_ssim_values.extend(mouth_ssim(pred, target).cpu().tolist())
        if lpips_evaluator is not None:
            face_lpips, mouth_lpips = lpips_face_and_mouth(lpips_evaluator, pred, target)
            lpips_face_values.extend(face_lpips.cpu().tolist())
            lpips_mouth_values.extend(mouth_lpips.cpu().tolist())
            total.append(
                float(
                    (
                        reconstruction_total
                        + lpips_face_loss_weight * face_lpips.mean()
                        + lpips_mouth_loss_weight * mouth_lpips.mean()
                    )
                    .detach()
                    .cpu()
                )
            )
        else:
            total.append(float(reconstruction_total.cpu()))
        audio_sensitivity.extend(mouth_mae(pred, shifted_pred).cpu().tolist())
        audio_shift_mouth_mae_delta.extend((shifted_mouth_mae - current_mouth_mae).cpu().tolist())
        metadata = batch.get("meta", [])
        if isinstance(metadata, list):
            for index, meta in enumerate(metadata):
                if not isinstance(meta, dict) or "clip_id" not in meta or "frame_idx" not in meta:
                    continue
                clip_id = str(meta["clip_id"])
                frame_idx = int(meta["frame_idx"])
                previous = previous_by_clip.get(clip_id)
                current_pred = pred[index : index + 1]
                current_target = target[index : index + 1]
                if previous is not None and frame_idx == previous[0] + 1:
                    temporal.append(float(torch.mean(torch.abs(current_pred - previous[1])).cpu()))
                    mouth_temporal.extend(
                        mouth_temporal_error(
                            current_pred,
                            current_target,
                            previous[1],
                            previous[2],
                        )
                        .cpu()
                        .tolist()
                    )
                previous_by_clip[clip_id] = (
                    frame_idx,
                    current_pred.detach(),
                    current_target.detach(),
                )
    if not reconstruction:
        raise ValueError("Validation loader produced no batches")
    metrics = {
        "val_loss": sum(total) / len(total),
        "val_reconstruction_loss": sum(reconstruction) / len(reconstruction),
        "val_mouth_loss": sum(mouth) / len(mouth),
        "val_temporal_delta": sum(temporal) / len(temporal) if temporal else 0.0,
        "val_mae": sum(mae) / len(mae),
        "val_psnr": sum(psnr) / len(psnr),
        "val_ssim": sum(ssim) / len(ssim),
        "val_mouth_mae": sum(mouth_mae_values) / len(mouth_mae_values),
        "val_mouth_psnr": sum(mouth_psnr_values) / len(mouth_psnr_values),
        "val_mouth_ssim": sum(mouth_ssim_values) / len(mouth_ssim_values),
        "val_mouth_temporal_error": (
            sum(mouth_temporal) / len(mouth_temporal) if mouth_temporal else 0.0
        ),
        "val_temporal_pair_count": float(len(mouth_temporal)),
        "val_audio_sensitivity": sum(audio_sensitivity) / len(audio_sensitivity),
        "val_audio_shift_mouth_mae_delta": (
            sum(audio_shift_mouth_mae_delta) / len(audio_shift_mouth_mae_delta)
        ),
    }
    if lpips_face_values:
        metrics["val_lpips_face"] = sum(lpips_face_values) / len(lpips_face_values)
        metrics["val_lpips_mouth"] = sum(lpips_mouth_values) / len(lpips_mouth_values)
    return metrics


def phase_for_step(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    stabilization_steps: int,
) -> str:
    if step <= warmup_steps:
        return "warmup"
    if step > max_steps - stabilization_steps:
        return "stabilization"
    return "main"


def set_training_phase(model: torch.nn.Module, phase: str) -> None:
    if phase not in {"warmup", "main", "stabilization"}:
        raise ValueError(f"Unsupported training phase={phase!r}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = phase != "warmup" or name.startswith(("dec_", "out_conv"))


def build_model(config: TrainConfig, device: torch.device) -> tuple[DuixUNet, dict[str, Any]]:
    init_source_count = sum(
        bool(value)
        for value in (
            config.init_bin,
            config.init_ckpt,
            config.hf_init_model_repo,
        )
    )
    if init_source_count != 1:
        raise ValueError("Set exactly one of init_bin, init_ckpt, or hf_init_model_repo")
    if config.init_ckpt or config.hf_init_model_repo:
        resolved = resolve_model_source(
            checkpoint=config.init_ckpt,
            hf_repo=config.hf_init_model_repo,
            hf_filename=config.hf_init_model_filename,
            cache_dir=config.hf_cache_dir,
        )
        model = load_ckpt(resolved.path, map_location=device).to(device)
        kind = "huggingface_checkpoint" if config.hf_init_model_repo else "pytorch_checkpoint"
        return model, {"kind": kind, **resolved.provenance}
    init_bin = Path(config.init_bin)
    if not init_bin.exists():
        raise FileNotFoundError(init_bin)
    model = DuixUNet().to(device)
    stats = model.load_ncnn_bin(init_bin, face_size=160, device=str(device))
    if int(stats.get("remaining_bytes", 0)) != 0:
        raise ValueError(f"NCNN bin had remaining bytes after load: {stats}")
    return model, {
        "kind": "ncnn_bin",
        "path": str(init_bin.resolve()),
        "weight_load": stats,
    }


def _write_metrics(metrics: list[dict[str, float | int | str]], run_dir: Path) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in metrics for key in row})
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def _format_training_log(row: dict[str, float | int | str], *, max_steps: int) -> str:
    parts = [
        "[train]",
        f"step={row['step']}/{max_steps}",
        f"epoch={row['epoch']}",
        f"phase={row['phase']}",
        f"lr={float(row['learning_rate']):.3g}",
        f"train_loss={float(row['train_loss']):.6g}",
    ]
    for key in (
        "val_loss",
        "val_reconstruction_loss",
        "val_mouth_loss",
        "val_mouth_mae",
        "val_lpips_mouth",
        "val_mouth_temporal_error",
        "val_audio_shift_mouth_mae_delta",
    ):
        if key in row:
            parts.append(f"{key}={float(row[key]):.6g}")
    return " ".join(parts)


def _should_log_step(
    row: dict[str, float | int | str],
    *,
    max_steps: int,
    log_interval: int,
) -> bool:
    if log_interval <= 0:
        return False
    step = int(row["step"])
    return (
        step == 1
        or step == max_steps
        or step % log_interval == 0
        or "val_loss" in row
        or "val_reconstruction_loss" in row
    )


def write_run_metadata(
    run_dir: str | Path,
    *,
    provenance: dict[str, Any],
    best_checkpoint: str | Path,
    final_checkpoint: str | Path,
) -> Path:
    output = Path(run_dir) / "run_metadata.json"
    payload = {
        "provenance": provenance,
        "artifacts": {
            "best_checkpoint": str(Path(best_checkpoint).resolve()),
            "final_checkpoint": str(Path(final_checkpoint).resolve()),
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def _clip_ids_from_dataset(dataset: Any) -> list[str]:
    records = getattr(dataset, "records", None)
    if records is not None:
        return [str(record.clip_id) for record in records]

    inner_dataset = getattr(dataset, "dataset", None)
    if inner_dataset is not None:
        try:
            return [str(value) for value in inner_dataset["clip_id"]]
        except (KeyError, TypeError, ValueError):
            pass

    clip_ids: list[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, dict):
            raise ValueError(f"Cannot read clip_id from dataset sample index={index}")
        meta = sample.get("meta")
        if not isinstance(meta, dict) or "clip_id" not in meta:
            raise ValueError(f"Cannot read clip_id from dataset sample index={index}")
        clip_ids.append(str(meta["clip_id"]))
    return clip_ids


def build_media_eval_selection(
    dataset: Any,
    *,
    clip_count: int,
    clip_ids: Iterable[str],
    max_frames_per_clip: int,
) -> MediaEvalSelection:
    if max_frames_per_clip <= 0:
        raise ValueError("media_eval_max_frames_per_clip must be positive")
    requested_clip_ids = tuple(str(clip_id) for clip_id in clip_ids)
    if clip_count <= 0 and not requested_clip_ids:
        raise ValueError("media_eval_clip_count must be positive when media_eval_clip_ids is empty")

    all_clip_ids = _clip_ids_from_dataset(dataset)
    if not all_clip_ids:
        raise ValueError("Validation dataset produced no media eval samples")

    available_clip_ids: list[str] = []
    seen: set[str] = set()
    for clip_id in all_clip_ids:
        if clip_id not in seen:
            available_clip_ids.append(clip_id)
            seen.add(clip_id)

    if requested_clip_ids:
        missing = [clip_id for clip_id in requested_clip_ids if clip_id not in seen]
        if missing:
            raise ValueError(f"media_eval_clip_ids not found in validation split: {missing}")
        selected_clip_ids = requested_clip_ids
    else:
        selected_clip_ids = tuple(available_clip_ids[:clip_count])
    if not selected_clip_ids:
        raise ValueError("No validation clips selected for media eval")

    selected_set = set(selected_clip_ids)
    counts = {clip_id: 0 for clip_id in selected_clip_ids}
    indices: list[int] = []
    for index, clip_id in enumerate(all_clip_ids):
        if clip_id not in selected_set or counts[clip_id] >= max_frames_per_clip:
            continue
        indices.append(index)
        counts[clip_id] += 1
        if all(count >= max_frames_per_clip for count in counts.values()):
            break
    if not indices:
        raise ValueError("No validation frames selected for media eval")
    return MediaEvalSelection(
        dataset=Subset(dataset, indices),
        clip_ids=selected_clip_ids,
        indices=tuple(indices),
    )


def write_model_card(run_dir: str | Path, *, provenance: dict[str, Any]) -> Path:
    dataset = provenance.get("dataset", {})
    wandb = provenance.get("wandb", {})
    model = provenance.get("model", {})
    dataset_ref = dataset.get("fingerprints") or dataset.get("resolved_ref", "")
    lines = [
        "# Edge Lip-Sync Duix UNet Checkpoint",
        "",
        "This repository contains checkpoints produced by the edge-lipsync-model pipeline.",
        "",
        "## Provenance",
        "",
        f"- Dataset source: `{dataset.get('source', '')}`",
        f"- Dataset repository: `{dataset.get('repo_id', '')}`",
        f"- Dataset ref/fingerprints: `{dataset_ref}`",
        f"- W&B run: {wandb.get('run_url', '')}",
        f"- Model repository: `{model.get('repo_id', '')}`",
        f"- Model resolved ref: `{model.get('resolved_ref', '')}`",
        "",
    ]
    output = Path(run_dir) / "README.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _wandb_config(config: TrainConfig) -> WandbConfig:
    return WandbConfig(
        mode=config.wandb_mode,
        project=config.wandb_project,
        entity=config.wandb_entity,
        run_name=config.wandb_run_name,
        group=config.wandb_group,
        tags=tuple(config.wandb_tags),
        notes=config.wandb_notes,
        directory=config.wandb_dir,
    )


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    config: TrainConfig,
    dataset_root: Path,
    manifest_path: Path,
    step: int,
    epoch: int,
    metrics: dict[str, float | int | str],
    init_weight_source: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    numeric_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, (float, int))
    }
    return make_training_checkpoint(
        model=model,
        training_config=asdict(config),
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        step=step,
        epoch=epoch,
        metrics=numeric_metrics,
        init_weight_source=init_weight_source,
        provenance=provenance,
    )


def _render_best_media_eval(
    *,
    model: torch.nn.Module,
    selection: MediaEvalSelection,
    run_dir: Path,
    best_path: Path,
    device: torch.device,
    step: int,
    fps: float,
    tracker: Any,
    log_to_wandb: bool,
    lpips_evaluator: torch.nn.Module | None = None,
) -> dict[str, Any]:
    out_dir = run_dir / "media_eval" / f"step_{step:07d}_best"
    artifacts = render_validation_artifacts(
        model=model,
        dataset=selection.dataset,
        out_dir=out_dir,
        checkpoint_path=best_path,
        device=device,
        max_batches=len(selection.indices),
        fps=fps,
        lpips_evaluator=lpips_evaluator,
    )
    metadata_path = Path(str(artifacts["metadata_path"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "media_eval_clip_ids": list(selection.clip_ids),
            "media_eval_indices": list(selection.indices),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    wandb_video_logged = False
    wandb_video_error = ""
    if log_to_wandb:
        caption = f"best.pt step={step} clips={','.join(selection.clip_ids)}"
        try:
            tracker.log_video(
                "media_eval/best_validation_grids",
                str(artifacts["video_path"]),
                step=step,
                caption=caption,
            )
            wandb_video_logged = True
        except Exception as exc:
            wandb_video_error = str(exc)
            print(
                f"[media_eval] step={step} status=wandb_video_error error={wandb_video_error}",
                flush=True,
            )
    return {
        **artifacts,
        "step": step,
        "clip_ids": list(selection.clip_ids),
        "indices": list(selection.indices),
        "wandb_video_logged": wandb_video_logged,
        "wandb_video_error": wandb_video_error,
    }


def _dataset_fingerprints(dataset: Any) -> dict[str, str]:
    if not hasattr(dataset, "items"):
        return {}
    fingerprints: dict[str, str] = {}
    for split, split_dataset in dataset.items():
        fingerprint = getattr(split_dataset, "_fingerprint", "")
        if fingerprint:
            fingerprints[str(split)] = str(fingerprint)
    return fingerprints


def _verify_dataset_snapshot(root: Path) -> dict[str, str]:
    complete = root / "build_complete.json"
    if not complete.is_file():
        raise FileNotFoundError(complete)
    metadata = json.loads(complete.read_text(encoding="utf-8"))
    dataset_path = root / "dataset"
    dataset = load_from_disk(dataset_path)
    if set(dataset) != {"train", "val"}:
        raise ValueError("Dataset snapshot must contain train and val splits")
    if len(dataset["train"]) == 0 or len(dataset["val"]) == 0:
        raise ValueError("Dataset snapshot splits must be non-empty")
    fingerprints = _dataset_fingerprints(dataset)
    if metadata.get("dataset_fingerprints") != fingerprints:
        raise ValueError("Dataset fingerprints do not match build_complete.json")
    return fingerprints


def _write_hf_dataset_source(run_dir: Path, provenance: dict[str, Any]) -> Path:
    path = run_dir / "hf_dataset_source.json"
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return path


def prepare_training_datasets(
    config: TrainConfig,
    *,
    run_dir: str | Path | None = None,
) -> PreparedTrainingDatasets:
    if bool(config.dataset_root) == bool(config.hf_dataset_repo):
        raise ValueError("Set exactly one of dataset_root or hf_dataset_repo")
    if config.hf_dataset_repo:
        if not config.hf_dataset_revision:
            raise ValueError("hf_dataset_revision is required with hf_dataset_repo")
        if not config.hf_dataset_local_dir:
            raise ValueError("hf_dataset_local_dir is required with hf_dataset_repo")
        artifact = pull_dataset_snapshot(
            config.hf_dataset_repo,
            ref=config.hf_dataset_revision,
            local_dir=config.hf_dataset_local_dir,
            cache_dir=config.hf_cache_dir,
            verify=_verify_dataset_snapshot,
        )
        if artifact.path is None:
            raise ValueError("Dataset snapshot download returned no local path")
        dataset = load_from_disk(artifact.path / "dataset")
        provenance: dict[str, Any] = {
            "source": "huggingface_snapshot",
            "repo_id": artifact.repo_id,
            "requested_ref": artifact.requested_ref,
            "resolved_ref": artifact.resolved_ref,
            "path": str(artifact.path),
            "fingerprints": _dataset_fingerprints(dataset),
        }
        metadata_dir = Path(run_dir) if run_dir is not None else Path(".")
        if run_dir is not None:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = _write_hf_dataset_source(metadata_dir, provenance)
        else:
            manifest_path = metadata_dir / "hf_dataset_source.json"
        return PreparedTrainingDatasets(
            train_dataset=DuixHFDataset(dataset, split="train"),
            val_dataset=DuixHFDataset(dataset, split="val"),
            dataset_root=metadata_dir,
            manifest_path=manifest_path,
            provenance=provenance,
        )

    dataset_source = resolve_dataset_source(
        dataset_root=config.dataset_root,
        hf_repo="",
        cache_dir=config.hf_cache_dir,
    )
    dataset_root = dataset_source.path
    manifest_path = Path(config.manifest)
    if not manifest_path.is_absolute():
        manifest_path = dataset_root / manifest_path
    return PreparedTrainingDatasets(
        train_dataset=DuixManifestDataset(dataset_root, manifest_path, split="train"),
        val_dataset=DuixManifestDataset(dataset_root, manifest_path, split="val"),
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        provenance={
            **dataset_source.provenance,
            "manifest_sha256": manifest_sha256(manifest_path),
        },
    )


def train(config: TrainConfig) -> Path:
    validate_training_config(config)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared_data = prepare_training_datasets(config, run_dir=run_dir)
    media_eval_selection = (
        build_media_eval_selection(
            prepared_data.val_dataset,
            clip_count=config.media_eval_clip_count,
            clip_ids=tuple(config.media_eval_clip_ids),
            max_frames_per_clip=config.media_eval_max_frames_per_clip,
        )
        if config.media_eval_on_best
        else None
    )
    dataset_root = prepared_data.dataset_root
    manifest_path = prepared_data.manifest_path
    device = resolve_device(config.device)
    model, init_weight_source = build_model(config, device)
    lpips_evaluator = (
        LPIPSEvaluator(device, net=config.lpips_net)
        if config.lpips_enabled
        or config.lpips_face_loss_weight > 0
        or config.lpips_mouth_loss_weight > 0
        else None
    )
    provenance = {
        "dataset": prepared_data.provenance,
        "init_model": init_weight_source,
    }
    if media_eval_selection is not None:
        provenance["media_eval"] = {
            "on_best": True,
            "clip_ids": list(media_eval_selection.clip_ids),
            "indices": list(media_eval_selection.indices),
            "max_frames_per_clip": config.media_eval_max_frames_per_clip,
            "log_to_wandb": config.media_eval_log_to_wandb,
        }
    tracker = create_tracker(
        _wandb_config(config),
        run_config=asdict(config),
        provenance=provenance,
    )
    provenance["wandb"] = tracker.provenance
    try:
        mixed_precision = use_mixed_precision(config, device)
        shape_loader = DataLoader(
            prepared_data.train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_training_batch,
            generator=torch.Generator().manual_seed(0),
        )
        val_loader = DataLoader(
            prepared_data.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_training_batch,
        )
        validate_batch_shapes(next(iter(shape_loader)))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

        def training_loss(
            pred: torch.Tensor,
            target: torch.Tensor,
            *,
            sample_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return combined_training_loss(
                pred,
                target,
                lpips_evaluator=lpips_evaluator,
                lpips_face_weight=config.lpips_face_loss_weight,
                lpips_mouth_weight=config.lpips_mouth_loss_weight,
                sample_weight=sample_weight,
            )

        metrics: list[dict[str, float | int | str]] = []
        media_eval_artifacts: list[dict[str, Any]] = []
        best_val_loss = float("inf")
        early_stopping_best_val_loss = float("inf")
        best_metrics: dict[str, float | int | str] | None = None
        validations_without_improvement = 0
        early_stop_reason = ""
        early_stop_step = 0
        best_path = run_dir / "best.pt"
        step = 0
        epoch = 0
        active_phase = ""
        data_order_generator = torch.Generator().manual_seed(torch.initial_seed())
        epoch_sample_indices: list[int] = []
        next_batch_index = 0
    except Exception:
        tracker.finish(exit_code=1)
        raise
    try:
        while step < config.max_steps and not early_stop_reason:
            epoch_batch_count = (
                len(epoch_sample_indices) + config.batch_size - 1
            ) // config.batch_size
            if not epoch_sample_indices or next_batch_index >= epoch_batch_count:
                epoch += 1
                epoch_sample_indices = generate_epoch_sample_indices(
                    len(prepared_data.train_dataset),
                    data_order_generator,
                )
                next_batch_index = 0
            remaining_indices = remaining_epoch_sample_indices(
                epoch_sample_indices,
                batch_size=config.batch_size,
                next_batch_index=next_batch_index,
            )
            train_loader = DataLoader(
                Subset(prepared_data.train_dataset, remaining_indices),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collate_training_batch,
                generator=torch.Generator().manual_seed(
                    (epoch << 32) + next_batch_index
                ),
            )
            for batch in train_loader:
                step += 1
                phase = phase_for_step(
                    step,
                    max_steps=config.max_steps,
                    warmup_steps=config.warmup_steps,
                    stabilization_steps=config.stabilization_steps,
                )
                if phase != active_phase:
                    set_training_phase(model, phase)
                    active_phase = phase
                learning_rate = config.learning_rate
                if phase == "stabilization":
                    learning_rate *= config.stabilization_lr_scale
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                loss = run_train_step(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    device=device,
                    loss_fn=training_loss,
                    mixed_precision=mixed_precision,
                    scaler=scaler,
                )
                row: dict[str, float | int | str] = {
                    "step": step,
                    "epoch": epoch,
                    "phase": phase,
                    "learning_rate": learning_rate,
                    "train_loss": loss,
                }
                row_logged = False
                row_printed = False
                if step % config.validation_interval == 0:
                    validation = run_validation(
                        model,
                        val_loader,
                        device,
                        mixed_precision=mixed_precision,
                        lpips_evaluator=lpips_evaluator,
                        lpips_face_loss_weight=config.lpips_face_loss_weight,
                        lpips_mouth_loss_weight=config.lpips_mouth_loss_weight,
                    )
                    row.update(validation)
                    val_loss = float(validation["val_loss"])
                    checkpoint_improved = val_loss < best_val_loss
                    early_stopping_improved = (
                        val_loss < early_stopping_best_val_loss - config.early_stopping_min_delta
                    )
                    if checkpoint_improved:
                        best_val_loss = val_loss
                        best_metrics = dict(row)
                    if early_stopping_improved:
                        early_stopping_best_val_loss = val_loss
                        validations_without_improvement = 0
                    else:
                        validations_without_improvement += 1
                        if (
                            config.early_stopping_patience > 0
                            and validations_without_improvement >= config.early_stopping_patience
                        ):
                            early_stop_reason = "val_loss_patience"
                            early_stop_step = step
                            row["early_stop_reason"] = early_stop_reason
                            row["early_stop_patience"] = config.early_stopping_patience
                            row["early_stop_bad_validation_count"] = validations_without_improvement
                            row["best_val_loss"] = best_val_loss
                    tracker.log_metrics(row, step=step)
                    row_logged = True
                    if _should_log_step(
                        row,
                        max_steps=config.max_steps,
                        log_interval=config.log_interval,
                    ):
                        print(_format_training_log(row, max_steps=config.max_steps), flush=True)
                        row_printed = True
                    if checkpoint_improved:
                        atomic_torch_save(
                            _checkpoint_payload(
                                model=model,
                                config=config,
                                dataset_root=dataset_root,
                                manifest_path=manifest_path,
                                step=step,
                                epoch=epoch,
                                metrics=row,
                                init_weight_source=init_weight_source,
                                provenance=provenance,
                            ),
                            best_path,
                        )
                        if media_eval_selection is not None:
                            media_eval_dir = run_dir / "media_eval" / f"step_{step:07d}_best"
                            print(
                                "[media_eval] "
                                f"step={step} status=start "
                                f"clips={','.join(media_eval_selection.clip_ids)} "
                                f"frames={len(media_eval_selection.indices)} "
                                f"out_dir={media_eval_dir}",
                                flush=True,
                            )
                            media_eval_artifacts.append(
                                _render_best_media_eval(
                                    model=model,
                                    selection=media_eval_selection,
                                    run_dir=run_dir,
                                    best_path=best_path,
                                    device=device,
                                    step=step,
                                    fps=config.media_eval_fps,
                                    tracker=tracker,
                                    log_to_wandb=config.media_eval_log_to_wandb,
                                    lpips_evaluator=lpips_evaluator,
                                )
                            )
                            print(
                                "[media_eval] "
                                f"step={step} status=done "
                                f"video={media_eval_artifacts[-1]['video_path']}",
                                flush=True,
                            )
                if step % config.checkpoint_interval == 0:
                    atomic_torch_save(
                        _checkpoint_payload(
                            model=model,
                            config=config,
                            dataset_root=dataset_root,
                            manifest_path=manifest_path,
                            step=step,
                            epoch=epoch,
                            metrics=row,
                            init_weight_source=init_weight_source,
                            provenance=provenance,
                        ),
                        run_dir / f"step_{step:07d}.pt",
                    )
                next_batch_index += 1
                metrics.append(row)
                if not row_logged:
                    tracker.log_metrics(row, step=step)
                if not row_printed and _should_log_step(
                    row,
                    max_steps=config.max_steps,
                    log_interval=config.log_interval,
                ):
                    print(_format_training_log(row, max_steps=config.max_steps), flush=True)
                if early_stop_reason or step >= config.max_steps:
                    break

        _write_metrics(metrics, run_dir)
        if not best_path.exists():
            atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    config=config,
                    dataset_root=dataset_root,
                    manifest_path=manifest_path,
                    step=step,
                    epoch=epoch,
                    metrics=metrics[-1],
                    init_weight_source=init_weight_source,
                    provenance=provenance,
                ),
                best_path,
            )
        final_path = run_dir / "final.pt"
        atomic_torch_save(
            _checkpoint_payload(
                model=model,
                config=config,
                dataset_root=dataset_root,
                manifest_path=manifest_path,
                step=step,
                epoch=epoch,
                metrics=metrics[-1],
                init_weight_source=init_weight_source,
                provenance=provenance,
            ),
            final_path,
        )
        write_run_metadata(
            run_dir,
            provenance=provenance,
            best_checkpoint=best_path,
            final_checkpoint=final_path,
        )
        write_model_card(run_dir, provenance=provenance)
        if media_eval_artifacts:
            media_eval_index_path = run_dir / "media_eval" / "index.json"
            media_eval_index_path.parent.mkdir(parents=True, exist_ok=True)
            media_eval_index_path.write_text(
                json.dumps(media_eval_artifacts, indent=2),
                encoding="utf-8",
            )
        summary: dict[str, Any] = {
            "best_checkpoint": str(best_path.resolve()),
            "final_checkpoint": str(final_path.resolve()),
        }
        if media_eval_artifacts:
            latest_media_eval = media_eval_artifacts[-1]
            summary["best_media_eval_video"] = latest_media_eval["video_path"]
            summary["best_media_eval_metadata"] = latest_media_eval["metadata_path"]
        if best_metrics is not None:
            if "val_loss" in best_metrics:
                summary["best_val_loss"] = float(best_metrics["val_loss"])
            if "val_reconstruction_loss" in best_metrics:
                summary["best_val_reconstruction_loss"] = float(
                    best_metrics["val_reconstruction_loss"]
                )
            if "val_lpips_face" in best_metrics:
                summary["best_val_lpips_face"] = float(best_metrics["val_lpips_face"])
            if "val_lpips_mouth" in best_metrics:
                summary["best_val_lpips_mouth"] = float(best_metrics["val_lpips_mouth"])
        if early_stop_reason:
            summary["early_stop_reason"] = early_stop_reason
            summary["early_stop_step"] = early_stop_step
            summary["early_stop_patience"] = config.early_stopping_patience
            summary["early_stop_bad_validation_count"] = validations_without_improvement
        if config.hf_model_repo:
            model_artifact = push_model_artifacts(
                run_dir,
                config.hf_model_repo,
                private=config.hf_model_private,
            )
            provenance["model"] = {
                "source": "huggingface",
                "repo_id": model_artifact.repo_id,
                "resolved_ref": model_artifact.resolved_ref,
                "url": model_artifact.url,
            }
            write_run_metadata(
                run_dir,
                provenance=provenance,
                best_checkpoint=best_path,
                final_checkpoint=final_path,
            )
            write_model_card(run_dir, provenance=provenance)
            summary["hf_model_repo"] = model_artifact.repo_id
            summary["hf_model_ref"] = model_artifact.resolved_ref
            summary["hf_model_url"] = model_artifact.url
        tracker.update_summary(summary)
    except Exception:
        tracker.finish(exit_code=1)
        raise
    tracker.finish()
    return best_path
