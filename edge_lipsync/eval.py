from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from edge_lipsync.dataset import DuixHFDataset, DuixManifestDataset
from edge_lipsync.hf_datasets import load_processed_dataset
from edge_lipsync.losses import charbonnier_loss, mouth_weighted_l1
from edge_lipsync.metrics import (
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
from edge_lipsync.sources import resolve_dataset_source, resolve_model_source


@dataclass(frozen=True)
class RenderEvalConfig:
    out_dir: str
    dataset_root: str = ""
    ckpt: str = ""
    manifest: str = "manifest.jsonl"
    max_batches: int = 32
    device: str = "cpu"
    fps: float = 25.0
    hf_dataset_repo: str = ""
    hf_model_repo: str = ""
    hf_model_filename: str = "best.pt"
    hf_cache_dir: str = ""
    lpips_enabled: bool = False
    lpips_net: str = "alex"


@dataclass(frozen=True)
class ResolvedEvalInputs:
    dataset: Any
    checkpoint: Path
    provenance: dict[str, Any]


def resolve_eval_inputs(config: RenderEvalConfig) -> ResolvedEvalInputs:
    if bool(config.dataset_root) == bool(config.hf_dataset_repo):
        raise ValueError("Set exactly one of dataset_root or hf_dataset_repo")
    if config.hf_dataset_repo:
        loaded_dataset = load_processed_dataset(
            config.hf_dataset_repo,
            cache_dir=config.hf_cache_dir,
        )
        eval_dataset = DuixHFDataset(loaded_dataset, split="val")
        dataset_provenance = {
            "source": "huggingface_datasets",
            "repo_id": config.hf_dataset_repo,
        }
    else:
        dataset_source = resolve_dataset_source(
            dataset_root=config.dataset_root,
            cache_dir=config.hf_cache_dir,
        )
        eval_dataset = DuixManifestDataset(dataset_source.path, config.manifest, split="val")
        dataset_provenance = dataset_source.provenance
    model = resolve_model_source(
        checkpoint=config.ckpt,
        hf_repo=config.hf_model_repo,
        hf_filename=config.hf_model_filename,
        cache_dir=config.hf_cache_dir,
    )
    return ResolvedEvalInputs(
        dataset=eval_dataset,
        checkpoint=model.path,
        provenance={
            "dataset": dataset_provenance,
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
    diff = np.clip(np.abs(pred.astype(np.int16) - target.astype(np.int16)), 0, 255).astype(np.uint8)
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


def _collate_eval_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty eval batch")
    return {
        "face": default_collate([sample["face"] for sample in samples]),
        "audio": default_collate([sample["audio"] for sample in samples]),
        "target": default_collate([sample["target"] for sample in samples]),
        "meta": [sample.get("meta", {}) for sample in samples],
    }


@torch.inference_mode()
def render_validation_artifacts(
    *,
    model: torch.nn.Module,
    dataset: Any,
    out_dir: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    max_batches: int,
    fps: float = 25.0,
    lpips_evaluator: torch.nn.Module | None = None,
) -> dict[str, Any]:
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    output = Path(out_dir)
    grids_dir = output / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_eval_batch)
    model = model.to(device).eval()
    predictions: list[np.ndarray] = []
    grid_frames: list[np.ndarray] = []
    reconstruction: list[float] = []
    mouth: list[float] = []
    mae: list[float] = []
    psnr: list[float] = []
    ssim: list[float] = []
    mouth_mae_values: list[float] = []
    mouth_psnr_values: list[float] = []
    mouth_ssim_values: list[float] = []
    lpips_face_values: list[float] = []
    lpips_mouth_values: list[float] = []
    temporal: list[float] = []
    mouth_temporal: list[float] = []
    audio_sensitivity: list[float] = []
    audio_shift_mouth_mae_delta: list[float] = []
    previous_by_clip: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}
    grid_paths: list[str] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        face = batch["face"].to(device=device, dtype=torch.float32)
        audio = batch["audio"].to(device=device, dtype=torch.float32)
        target_tensor = batch["target"].to(device=device, dtype=torch.float32)
        pred_tensor = model(face, audio)
        shifted_pred_tensor = model(face, shift_audio_window(audio))
        reconstruction.append(float(charbonnier_loss(pred_tensor, target_tensor).cpu()))
        mouth.append(float(mouth_weighted_l1(pred_tensor, target_tensor).cpu()))
        mae.extend(image_mae(pred_tensor, target_tensor).cpu().tolist())
        psnr.extend(image_psnr(pred_tensor, target_tensor).cpu().tolist())
        ssim.extend(image_ssim(pred_tensor, target_tensor).cpu().tolist())
        current_mouth_mae = mouth_mae(pred_tensor, target_tensor)
        shifted_mouth_mae = mouth_mae(shifted_pred_tensor, target_tensor)
        mouth_mae_values.extend(current_mouth_mae.cpu().tolist())
        mouth_psnr_values.extend(mouth_psnr(pred_tensor, target_tensor).cpu().tolist())
        mouth_ssim_values.extend(mouth_ssim(pred_tensor, target_tensor).cpu().tolist())
        if lpips_evaluator is not None:
            face_lpips, mouth_lpips = lpips_face_and_mouth(
                lpips_evaluator,
                pred_tensor,
                target_tensor,
            )
            lpips_face_values.extend(face_lpips.cpu().tolist())
            lpips_mouth_values.extend(mouth_lpips.cpu().tolist())
        audio_sensitivity.extend(mouth_mae(pred_tensor, shifted_pred_tensor).cpu().tolist())
        audio_shift_mouth_mae_delta.extend((shifted_mouth_mae - current_mouth_mae).cpu().tolist())
        meta = batch["meta"][0]
        if isinstance(meta, dict) and "clip_id" in meta and "frame_idx" in meta:
            clip_id = str(meta["clip_id"])
            frame_idx = int(meta["frame_idx"])
            previous = previous_by_clip.get(clip_id)
            if previous is not None and frame_idx == previous[0] + 1:
                temporal.append(float(torch.mean(torch.abs(pred_tensor - previous[1])).cpu()))
                mouth_temporal.extend(
                    mouth_temporal_error(
                        pred_tensor,
                        target_tensor,
                        previous[1],
                        previous[2],
                    )
                    .cpu()
                    .tolist()
                )
            previous_by_clip[clip_id] = (
                frame_idx,
                pred_tensor.detach(),
                target_tensor.detach(),
            )
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
    video_path = output / "validation_grids.mp4"
    metadata_path = write_rgb_video(
        grid_frames,
        video_path,
        fps=fps,
        metadata={
            "kind": "validation_grid_render",
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "dataset_root": str(getattr(dataset, "dataset_root", "")),
            "manifest_path": str(getattr(dataset, "manifest_path", "")),
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
