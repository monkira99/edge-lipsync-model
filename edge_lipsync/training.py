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
from edge_lipsync.dataset import DuixManifestDataset
from edge_lipsync.losses import (
    charbonnier_loss,
    combined_reconstruction_loss,
    mouth_weighted_l1,
)
from edge_lipsync.model import DuixUNet, load_ckpt


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    manifest: str
    run_dir: str
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
    if bool(config.init_bin) == bool(config.init_ckpt):
        raise ValueError("Set exactly one of init_bin or init_ckpt")
    if config.init_ckpt:
        checkpoint = Path(config.init_ckpt)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = load_ckpt(checkpoint, map_location=device).to(device)
        return model, {"kind": "pytorch_checkpoint", "path": str(checkpoint.resolve())}
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


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    config: TrainConfig,
    manifest_path: Path,
    step: int,
    epoch: int,
    metrics: dict[str, float | int | str],
    init_weight_source: dict[str, Any],
) -> dict[str, Any]:
    numeric_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, (float, int))
    }
    return make_training_checkpoint(
        model=model,
        training_config=asdict(config),
        dataset_root=config.dataset_root,
        manifest_path=manifest_path,
        step=step,
        epoch=epoch,
        metrics=numeric_metrics,
        init_weight_source=init_weight_source,
    )


def train(config: TrainConfig) -> Path:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.validation_interval <= 0 or config.checkpoint_interval <= 0:
        raise ValueError("validation_interval and checkpoint_interval must be positive")
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(config.dataset_root)
    manifest_path = Path(config.manifest)
    if not manifest_path.is_absolute():
        manifest_path = dataset_root / manifest_path
    device = resolve_device(config.device)
    model, init_weight_source = build_model(config, device)
    mixed_precision = use_mixed_precision(config, device)
    train_dataset = DuixManifestDataset(dataset_root, manifest_path, split="train")
    val_dataset = DuixManifestDataset(dataset_root, manifest_path, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
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
                            manifest_path=manifest_path,
                            step=step,
                            epoch=epoch,
                            metrics=row,
                            init_weight_source=init_weight_source,
                        ),
                        best_path,
                    )
            if step % config.checkpoint_interval == 0:
                atomic_torch_save(
                    _checkpoint_payload(
                        model=model,
                        config=config,
                        manifest_path=manifest_path,
                        step=step,
                        epoch=epoch,
                        metrics=row,
                        init_weight_source=init_weight_source,
                    ),
                    run_dir / f"step_{step:07d}.pt",
                )
            metrics.append(row)
            if step >= config.max_steps:
                break

    _write_metrics(metrics, run_dir)
    if not best_path.exists():
        atomic_torch_save(
            _checkpoint_payload(
                model=model,
                config=config,
                manifest_path=manifest_path,
                step=step,
                epoch=epoch,
                metrics=metrics[-1],
                init_weight_source=init_weight_source,
            ),
            best_path,
        )
    return best_path
