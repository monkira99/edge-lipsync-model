from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from edge_lipsync.checkpoint import atomic_torch_save, make_training_checkpoint
from edge_lipsync.dataset import DuixHFDataset, DuixManifestDataset, manifest_sha256
from edge_lipsync.hf_datasets import load_processed_dataset
from edge_lipsync.hub import push_model_artifacts
from edge_lipsync.losses import (
    charbonnier_loss,
    combined_reconstruction_loss,
    mouth_weighted_l1,
)
from edge_lipsync.model import DuixUNet, load_ckpt
from edge_lipsync.sources import resolve_dataset_source, resolve_model_source
from edge_lipsync.tracking import WandbConfig, create_tracker


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
    hf_dataset_repo: str = ""
    hf_cache_dir: str = ""
    hf_init_model_repo: str = ""
    hf_init_model_filename: str = "best.pt"
    hf_model_repo: str = ""
    hf_model_private: bool = True
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
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
        loss = loss_fn(pred, target)
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
) -> dict[str, float]:
    model.eval()
    reconstruction: list[float] = []
    mouth: list[float] = []
    temporal: list[float] = []
    previous_pred: torch.Tensor | None = None
    for batch in loader:
        face = batch["face"].to(device=device, dtype=torch.float32)
        audio = batch["audio"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            pred = model(face, audio)
            reconstruction.append(float(charbonnier_loss(pred, target).cpu()))
            mouth.append(float(mouth_weighted_l1(pred, target).cpu()))
        for current_pred in pred:
            if previous_pred is not None:
                temporal.append(float(torch.mean(torch.abs(current_pred - previous_pred)).cpu()))
            previous_pred = current_pred
    if not reconstruction:
        raise ValueError("Validation loader produced no batches")
    return {
        "val_reconstruction_loss": sum(reconstruction) / len(reconstruction),
        "val_mouth_loss": sum(mouth) / len(mouth),
        "val_temporal_delta": sum(temporal) / len(temporal) if temporal else 0.0,
    }


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


def _dataset_fingerprints(dataset: Any) -> dict[str, str]:
    if not hasattr(dataset, "items"):
        return {}
    fingerprints: dict[str, str] = {}
    for split, split_dataset in dataset.items():
        fingerprint = getattr(split_dataset, "_fingerprint", "")
        if fingerprint:
            fingerprints[str(split)] = str(fingerprint)
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
        dataset = load_processed_dataset(config.hf_dataset_repo, cache_dir=config.hf_cache_dir)
        provenance: dict[str, Any] = {
            "source": "huggingface_datasets",
            "repo_id": config.hf_dataset_repo,
        }
        fingerprints = _dataset_fingerprints(dataset)
        if fingerprints:
            provenance["fingerprints"] = fingerprints
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
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.validation_interval <= 0 or config.checkpoint_interval <= 0:
        raise ValueError("validation_interval and checkpoint_interval must be positive")
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared_data = prepare_training_datasets(config, run_dir=run_dir)
    dataset_root = prepared_data.dataset_root
    manifest_path = prepared_data.manifest_path
    device = resolve_device(config.device)
    model, init_weight_source = build_model(config, device)
    provenance = {
        "dataset": prepared_data.provenance,
        "init_model": init_weight_source,
    }
    tracker = create_tracker(
        _wandb_config(config),
        run_config=asdict(config),
        provenance=provenance,
    )
    provenance["wandb"] = tracker.provenance
    try:
        mixed_precision = use_mixed_precision(config, device)
        train_loader = DataLoader(
            prepared_data.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            prepared_data.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        validate_batch_shapes(next(iter(train_loader)))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

        metrics: list[dict[str, float | int | str]] = []
        best_val = float("inf")
        best_path = run_dir / "best.pt"
        step = 0
        epoch = 0
        active_phase = ""
    except Exception:
        tracker.finish(exit_code=1)
        raise
    try:
        while step < config.max_steps:
            epoch += 1
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
                    loss_fn=combined_reconstruction_loss,
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
                if step % config.validation_interval == 0:
                    validation = run_validation(
                        model,
                        val_loader,
                        device,
                        mixed_precision=mixed_precision,
                    )
                    row.update(validation)
                    if validation["val_reconstruction_loss"] < best_val:
                        best_val = validation["val_reconstruction_loss"]
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
                metrics.append(row)
                tracker.log_metrics(row, step=step)
                if step >= config.max_steps:
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
        summary: dict[str, Any] = {
            "best_checkpoint": str(best_path.resolve()),
            "final_checkpoint": str(final_path.resolve()),
        }
        if best_val != float("inf"):
            summary["best_val_reconstruction_loss"] = best_val
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
