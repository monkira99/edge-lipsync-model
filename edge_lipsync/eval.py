from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from edge_lipsync.dataset import DuixManifestDataset
from edge_lipsync.losses import charbonnier_loss, mouth_weighted_l1
from edge_lipsync.sources import resolve_dataset_source, resolve_model_source


@dataclass(frozen=True)
class RenderEvalConfig:
    dataset_root: str
    ckpt: str
    out_dir: str
    manifest: str = "manifest.jsonl"
    max_batches: int = 32
    device: str = "cpu"
    fps: float = 25.0
    hf_dataset_repo: str = ""
    hf_dataset_revision: str = ""
    hf_model_repo: str = ""
    hf_model_revision: str = ""
    hf_model_filename: str = "best.pt"
    hf_cache_dir: str = ""


@dataclass(frozen=True)
class ResolvedEvalInputs:
    dataset_root: Path
    checkpoint: Path
    provenance: dict[str, Any]


def resolve_eval_inputs(config: RenderEvalConfig) -> ResolvedEvalInputs:
    dataset = resolve_dataset_source(
        dataset_root=config.dataset_root,
        hf_repo=config.hf_dataset_repo,
        hf_revision=config.hf_dataset_revision,
        cache_dir=config.hf_cache_dir,
    )
    model = resolve_model_source(
        checkpoint=config.ckpt,
        hf_repo=config.hf_model_repo,
        hf_revision=config.hf_model_revision,
        hf_filename=config.hf_model_filename,
        cache_dir=config.hf_cache_dir,
    )
    return ResolvedEvalInputs(
        dataset_root=dataset.path,
        checkpoint=model.path,
        provenance={
            "dataset": dataset.provenance,
            "model": model.provenance,
        },
    )


def chw_norm_to_rgb_u8(chw: np.ndarray) -> np.ndarray:
    if chw.shape[0] != 3:
        raise ValueError(f"Expected CHW with 3 channels, got {chw.shape}")
    hwc = np.transpose(chw, (1, 2, 0))
    return np.clip((hwc + 1.0) * 127.5, 0, 255).astype(np.uint8)


def temporal_delta_metric(predictions: list[np.ndarray]) -> float:
    if len(predictions) < 2:
        return 0.0
    deltas = [
        float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))
        for previous, current in zip(predictions[:-1], predictions[1:], strict=True)
    ]
    return sum(deltas) / len(deltas)


def prediction_grid_rgb(
    masked_chw: np.ndarray,
    pred_chw: np.ndarray,
    target_chw: np.ndarray,
) -> np.ndarray:
    masked = chw_norm_to_rgb_u8(masked_chw)
    pred = chw_norm_to_rgb_u8(pred_chw)
    target = chw_norm_to_rgb_u8(target_chw)
    diff = np.clip(np.abs(pred.astype(np.int16) - target.astype(np.int16)), 0, 255).astype(
        np.uint8
    )
    return np.concatenate([masked, pred, target, diff], axis=1)


def write_prediction_grid(
    masked_chw: np.ndarray,
    pred_chw: np.ndarray,
    target_chw: np.ndarray,
    out_path: str | Path,
) -> None:
    grid = prediction_grid_rgb(masked_chw, pred_chw, target_chw)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Cannot write prediction grid: {out}")


def write_rgb_video(
    frames_rgb: list[np.ndarray],
    out_path: str | Path,
    *,
    fps: float,
    metadata: dict[str, Any],
) -> Path:
    if not frames_rgb:
        raise ValueError("Cannot write an empty render")
    height, width = frames_rgb[0].shape[:2]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),  # pyright: ignore[reportAttributeAccessIssue]
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {out}")
    for frame in frames_rgb:
        if frame.shape != (height, width, 3):
            raise ValueError(f"Inconsistent RGB render frame shape: {frame.shape}")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    metadata_path = out.with_suffix(".json")
    payload = {
        **metadata,
        "out_video": str(out.resolve()),
        "fps": float(fps),
        "frame_count": len(frames_rgb),
        "frame_shape": [height, width, 3],
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


@torch.inference_mode()
def render_validation_artifacts(
    *,
    model: torch.nn.Module,
    dataset: DuixManifestDataset,
    out_dir: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    max_batches: int,
    fps: float = 25.0,
) -> dict[str, Any]:
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    output = Path(out_dir)
    grids_dir = output / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = model.to(device).eval()
    predictions: list[np.ndarray] = []
    grid_frames: list[np.ndarray] = []
    reconstruction: list[float] = []
    mouth: list[float] = []
    grid_paths: list[str] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        face = batch["face"].to(device=device, dtype=torch.float32)
        audio = batch["audio"].to(device=device, dtype=torch.float32)
        target_tensor = batch["target"].to(device=device, dtype=torch.float32)
        pred_tensor = model(face, audio)
        reconstruction.append(float(charbonnier_loss(pred_tensor, target_tensor).cpu()))
        mouth.append(float(mouth_weighted_l1(pred_tensor, target_tensor).cpu()))
        pred = pred_tensor.cpu().numpy()[0]
        target = target_tensor.cpu().numpy()[0]
        masked = face.cpu().numpy()[0][3:6]
        predictions.append(pred)
        grid = prediction_grid_rgb(masked, pred, target)
        grid_path = grids_dir / f"grid_{index:04d}.png"
        if not cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Cannot write prediction grid: {grid_path}")
        grid_paths.append(str(grid_path.resolve()))
        grid_frames.append(grid)
    if not predictions:
        raise ValueError("Validation dataset produced no render samples")
    metrics = {
        "val_reconstruction_loss": sum(reconstruction) / len(reconstruction),
        "val_mouth_loss": sum(mouth) / len(mouth),
        "val_temporal_delta": temporal_delta_metric(predictions),
    }
    video_path = output / "validation_grids.mp4"
    metadata_path = write_rgb_video(
        grid_frames,
        video_path,
        fps=fps,
        metadata={
            "kind": "validation_grid_render",
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "dataset_root": str(dataset.dataset_root.resolve()),
            "manifest_path": str(dataset.manifest_path.resolve()),
            "metrics": metrics,
            "grid_paths": grid_paths,
        },
    )
    return {
        "video_path": str(video_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "grid_paths": grid_paths,
        "metrics": metrics,
    }
