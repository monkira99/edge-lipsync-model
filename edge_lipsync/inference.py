from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch

from edge_lipsync.audio_features import extract_bnf_windows_from_wav, get_bnf_window
from edge_lipsync.dataset import ManifestRecord, load_manifest
from edge_lipsync.eval import chw_norm_to_rgb_u8, prediction_grid_rgb
from edge_lipsync.model import DuixUNet, load_ckpt
from edge_lipsync.preprocess import FACE_SIZE, ROI_EDGE, make_face_training_sample
from edge_lipsync.sources import resolve_model_source


class PredictionRuntime(Protocol):
    def predict(self, face: np.ndarray, audio: np.ndarray) -> np.ndarray: ...


class TorchPredictionRuntime:
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def predict(self, face: np.ndarray, audio: np.ndarray) -> np.ndarray:
        face_tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        return self.model(face_tensor, audio_tensor).cpu().numpy()[0]


class NcnnPredictionRuntime:
    def __init__(self, ncnn_module: Any, net: Any) -> None:
        self.ncnn = ncnn_module
        self.net = net

    def predict(self, face: np.ndarray, audio: np.ndarray) -> np.ndarray:
        face_chw = np.ascontiguousarray(face, dtype=np.float32)
        audio_chw = np.ascontiguousarray(audio[None, :, :], dtype=np.float32)
        if face_chw.shape != (6, FACE_SIZE, FACE_SIZE):
            raise ValueError(
                f"Expected NCNN face [6,{FACE_SIZE},{FACE_SIZE}], got {face_chw.shape}"
            )
        if audio_chw.shape != (1, 20, 256):
            raise ValueError(f"Expected NCNN audio [1,20,256], got {audio_chw.shape}")
        extractor = self.net.create_extractor()
        if extractor.input("face", self.ncnn.Mat(face_chw)) != 0:
            raise RuntimeError("NCNN input(face) failed")
        if extractor.input("audio", self.ncnn.Mat(audio_chw)) != 0:
            raise RuntimeError("NCNN input(audio) failed")
        ret, prediction_mat = extractor.extract("output")
        if ret != 0:
            raise RuntimeError(f"NCNN extract(output) failed ret={ret}")
        prediction = np.ascontiguousarray(prediction_mat.numpy(), dtype=np.float32)
        if prediction.shape != (3, FACE_SIZE, FACE_SIZE):
            raise ValueError(
                f"Expected NCNN prediction [3,{FACE_SIZE},{FACE_SIZE}], got {prediction.shape}"
            )
        return prediction


def _resolve_path(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _select_manifest_record(
    manifest_path: Path,
    *,
    sample_index: int,
    split: str | None,
) -> ManifestRecord:
    records = load_manifest(manifest_path, split=split)
    if sample_index < 0 or sample_index >= len(records):
        raise IndexError(
            f"sample_index={sample_index} outside manifest record range 0..{len(records) - 1}"
        )
    return records[sample_index]


def _load_model_from_source(
    *,
    checkpoint: str | Path,
    init_bin: str | Path,
    hf_model_repo: str,
    hf_model_revision: str,
    hf_model_filename: str,
    hf_cache_dir: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    source_count = sum(bool(str(value)) for value in (checkpoint, init_bin, hf_model_repo))
    if source_count != 1:
        raise ValueError("Set exactly one of checkpoint, init_bin, or hf_model_repo")

    if str(init_bin):
        init_path = Path(init_bin)
        if not init_path.exists():
            raise FileNotFoundError(init_path)
        model = DuixUNet().to(device)
        stats = model.load_ncnn_bin(init_path, face_size=160, device=str(device))
        if int(stats.get("remaining_bytes", 0)) != 0:
            raise ValueError(f"NCNN bin had remaining bytes after load: {stats}")
        return model.eval(), {
            "source": "local_ncnn_bin",
            "path": str(init_path.resolve()),
            "weight_load": stats,
        }

    resolved = resolve_model_source(
        checkpoint=str(checkpoint),
        hf_repo=hf_model_repo,
        hf_revision=hf_model_revision,
        hf_filename=hf_model_filename,
        cache_dir=hf_cache_dir,
    )
    model = load_ckpt(resolved.path, map_location=device).to(device).eval()
    return model, resolved.provenance


def _load_ncnn_runtime(
    ncnn_param: str | Path,
    init_bin: str | Path,
) -> tuple[NcnnPredictionRuntime, dict[str, Any]]:
    param_path = Path(ncnn_param)
    bin_path = Path(init_bin)
    if not param_path.is_file():
        raise FileNotFoundError(param_path)
    if not bin_path.is_file():
        raise FileNotFoundError(bin_path)
    try:
        ncnn_module = importlib.import_module("ncnn")
    except ModuleNotFoundError as exc:
        raise RuntimeError("NCNN backend requires the optional ncnn Python package") from exc
    net = ncnn_module.Net()
    if net.load_param(str(param_path)) != 0:
        raise RuntimeError(f"Failed to load NCNN param: {param_path}")
    if net.load_model(str(bin_path)) != 0:
        raise RuntimeError(f"Failed to load NCNN model: {bin_path}")
    return NcnnPredictionRuntime(ncnn_module, net), {
        "source": "local_ncnn_runtime",
        "param_path": str(param_path.resolve()),
        "bin_path": str(bin_path.resolve()),
    }


def _load_prediction_runtime(
    *,
    backend: str,
    ncnn_param: str | Path,
    checkpoint: str | Path,
    init_bin: str | Path,
    hf_model_repo: str,
    hf_model_revision: str,
    hf_model_filename: str,
    hf_cache_dir: str,
    device: torch.device,
) -> tuple[PredictionRuntime, dict[str, Any]]:
    if backend == "ncnn":
        if str(checkpoint) or hf_model_repo or not str(init_bin):
            raise ValueError(
                "NCNN backend requires --init-bin and does not accept checkpoint sources"
            )
        runtime, provenance = _load_ncnn_runtime(ncnn_param, init_bin)
        return runtime, {**provenance, "backend": "ncnn"}
    if backend != "torch":
        raise ValueError(f"Unsupported inference backend: {backend}")
    model, provenance = _load_model_from_source(
        checkpoint=checkpoint,
        init_bin=init_bin,
        hf_model_repo=hf_model_repo,
        hf_model_revision=hf_model_revision,
        hf_model_filename=hf_model_filename,
        hf_cache_dir=hf_cache_dir,
        device=device,
    )
    return TorchPredictionRuntime(model, device), {**provenance, "backend": "torch"}


def _write_rgb_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Cannot write image: {path}")


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"Required tool not found on PATH: {name}")
    return path


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\nSTDERR:\n{process.stderr}")
    return process


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _load_alpha_mask(alpha_bin: str | Path) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not str(alpha_bin):
        return None, {"source": "disabled"}
    path = Path(alpha_bin)
    if not path.exists():
        raise FileNotFoundError(path)
    alpha = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    expected = FACE_SIZE * FACE_SIZE
    if alpha.size != expected:
        raise ValueError(f"Expected alpha mask with {expected} bytes, got {alpha.size}: {path}")
    return np.ascontiguousarray(alpha.reshape(FACE_SIZE, FACE_SIZE)), {
        "source": "local_weight_168u_bin",
        "path": str(path.resolve()),
    }


def blend_prediction_bgr(
    prediction_bgr: np.ndarray,
    original_bgr: np.ndarray,
    alpha_u8: np.ndarray,
) -> np.ndarray:
    expected_shape = (FACE_SIZE, FACE_SIZE)
    if alpha_u8.shape != expected_shape:
        raise ValueError(f"Expected alpha mask {expected_shape}, got {alpha_u8.shape}")
    alpha = (alpha_u8.astype(np.float32) / 255.0)[..., None]
    blended = original_bgr.astype(np.float32) * alpha + prediction_bgr.astype(np.float32) * (
        1.0 - alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def restore_prediction_to_frame(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    roi_168_bgr: np.ndarray,
    prediction_rgb: np.ndarray,
    *,
    alpha_u8: np.ndarray | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    if prediction_rgb.shape != (FACE_SIZE, FACE_SIZE, 3):
        raise ValueError(
            f"Expected prediction RGB [{FACE_SIZE},{FACE_SIZE},3], got {prediction_rgb.shape}"
        )
    if roi_168_bgr.shape != (168, 168, 3):
        raise ValueError(f"Expected ROI BGR [168,168,3], got {roi_168_bgr.shape}")
    restored_roi_168 = roi_168_bgr.copy()
    prediction_bgr = cv2.cvtColor(prediction_rgb, cv2.COLOR_RGB2BGR)
    original_bgr = restored_roi_168[
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
    ]
    if alpha_u8 is not None:
        prediction_bgr = blend_prediction_bgr(prediction_bgr, original_bgr, alpha_u8)
    restored_roi_168[
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
    ] = prediction_bgr
    restored_roi = cv2.resize(restored_roi_168, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    restored_frame = frame_bgr.copy()
    restored_frame[y1:y2, x1:x2] = restored_roi
    return restored_frame


def _resolve_output_mp4_path(out_dir: Path, output_mp4: str | Path) -> Path:
    output = Path(output_mp4)
    return output if output.is_absolute() else out_dir / output


def write_still_frame_audio_mp4(
    *,
    frame_path: str | Path,
    audio_wav: str | Path,
    out_path: str | Path,
    fps: float,
) -> Path:
    audio_path = Path(audio_wav)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    frame = Path(frame_path)
    if not frame.exists():
        raise FileNotFoundError(frame)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _require_tool("ffmpeg")
    _run_command(
        [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(float(fps)),
            "-i",
            str(frame),
            "-i",
            str(audio_path),
            "-shortest",
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return output


def write_frame_sequence_audio_mp4(
    *,
    frames_dir: str | Path,
    audio_wav: str | Path,
    out_path: str | Path,
    fps: float,
) -> Path:
    audio_path = Path(audio_wav)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    frame_dir = Path(frames_dir)
    if not frame_dir.is_dir():
        raise FileNotFoundError(frame_dir)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        raise ValueError(f"No PNG frames in {frame_dir}")
    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Cannot read sequence frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    video_no_audio = frame_dir.parent / "_video_no_audio.mp4"
    writer = cv2.VideoWriter(
        str(video_no_audio),
        cv2.VideoWriter_fourcc(*"mp4v"),  # pyright: ignore[reportAttributeAccessIssue]
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {video_no_audio}")
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot read sequence frame: {frame_path}")
        if frame.shape != first_frame.shape:
            raise ValueError(f"Inconsistent sequence frame shape: {frame_path} {frame.shape}")
        writer.write(frame)
    writer.release()
    ffmpeg = _require_tool("ffmpeg")
    try:
        _run_command(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_no_audio),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-r",
                f"{fps:.6f}",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-shortest",
                str(output),
            ]
        )
    finally:
        video_no_audio.unlink(missing_ok=True)
    return output


def _load_audio_bnf(
    *,
    root: Path,
    record: ManifestRecord,
    audio_wav: str | Path,
    wenet_onnx: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if str(audio_wav):
        wav_path = Path(audio_wav)
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)
        if not str(wenet_onnx):
            raise ValueError("wenet_onnx is required when audio_wav is set")
        wenet_path = Path(wenet_onnx)
        if not wenet_path.exists():
            raise FileNotFoundError(wenet_path)
        return extract_bnf_windows_from_wav(wav_path, wenet_path), {
            "source": "wav",
            "path": str(wav_path.resolve()),
            "wenet_onnx": str(wenet_path.resolve()),
        }

    bnf_path = root / record.bnf_path
    if not bnf_path.exists():
        raise FileNotFoundError(bnf_path)
    return np.load(bnf_path, allow_pickle=False), {
        "source": "manifest_bnf",
        "path": str(bnf_path.resolve()),
    }


def _load_records(
    manifest_path: Path,
    *,
    split: str | None,
    max_frames: int,
) -> list[ManifestRecord]:
    records = load_manifest(manifest_path, split=split)
    if max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if max_frames:
        records = records[:max_frames]
    if not records:
        raise ValueError(f"No manifest records selected from {manifest_path}")
    return records


@torch.inference_mode()
def run_manifest_sample_inference(
    *,
    dataset_root: str | Path,
    manifest: str | Path,
    out_dir: str | Path,
    sample_index: int = 0,
    split: str | None = None,
    checkpoint: str | Path = "",
    init_bin: str | Path = "",
    hf_model_repo: str = "",
    hf_model_revision: str = "",
    hf_model_filename: str = "best.pt",
    hf_cache_dir: str = "",
    backend: str = "torch",
    ncnn_param: str | Path = "",
    audio_wav: str | Path = "",
    wenet_onnx: str | Path = "",
    alpha_bin: str | Path = "",
    output_mp4: str | Path = "",
    fps: float = 25.0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = _resolve_path(root, manifest)
    record = _select_manifest_record(manifest_path, sample_index=sample_index, split=split)

    frame_path = root / record.frame_path
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(frame_path)
    if str(output_mp4) and not str(audio_wav):
        raise ValueError("audio_wav is required when output_mp4 is set")

    bnf, audio_source = _load_audio_bnf(
        root=root,
        record=record,
        audio_wav=audio_wav,
        wenet_onnx=wenet_onnx,
    )
    face_sample = make_face_training_sample(frame, record.bbox_xyxy)
    audio = get_bnf_window(bnf, record.audio_idx)
    runtime_device = torch.device(device)
    alpha_u8, alpha_source = _load_alpha_mask(alpha_bin)
    runtime, model_provenance = _load_prediction_runtime(
        backend=backend,
        ncnn_param=ncnn_param,
        checkpoint=checkpoint,
        init_bin=init_bin,
        hf_model_repo=hf_model_repo,
        hf_model_revision=hf_model_revision,
        hf_model_filename=hf_model_filename,
        hf_cache_dir=hf_cache_dir,
        device=runtime_device,
    )

    prediction = runtime.predict(face_sample.face, audio)

    out = Path(out_dir)
    prediction_path = out / "prediction.png"
    grid_path = out / "grid.png"
    restored_frame_path = out / "restored_frame.png"
    metadata_path = out / "metadata.json"
    prediction_rgb = chw_norm_to_rgb_u8(prediction)
    grid_rgb = prediction_grid_rgb(face_sample.face[3:6], prediction, face_sample.target)
    restored_frame = restore_prediction_to_frame(
        frame,
        record.bbox_xyxy,
        face_sample.roi_168_bgr,
        prediction_rgb,
        alpha_u8=alpha_u8,
    )
    _write_rgb_image(prediction_path, prediction_rgb)
    _write_rgb_image(grid_path, grid_rgb)
    restored_frame_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(restored_frame_path), restored_frame):
        raise RuntimeError(f"Cannot write image: {restored_frame_path}")
    output_mp4_path: Path | None = None
    if str(output_mp4):
        output_mp4_path = write_still_frame_audio_mp4(
            frame_path=restored_frame_path,
            audio_wav=audio_wav,
            out_path=_resolve_output_mp4_path(out, output_mp4),
            fps=fps,
        )

    metadata: dict[str, Any] = {
        "kind": "manifest_sample_inference",
        "dataset_root": str(root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "sample_index": int(sample_index),
        "split": split or "all",
        "sample": {
            "clip_id": record.clip_id,
            "frame_idx": record.frame_idx,
            "audio_idx": record.audio_idx,
            "frame_path": record.frame_path,
            "bnf_path": record.bnf_path,
            "bbox_xyxy": list(record.bbox_xyxy),
            "flags": list(record.flags),
        },
        "model": model_provenance,
        "alpha": alpha_source,
        "audio_source": audio_source,
        "shapes": {
            "face": list(face_sample.face.shape),
            "audio": list(audio.shape),
            "prediction": list(prediction.shape),
        },
        "prediction_stats": {
            "min": float(prediction.min()),
            "max": float(prediction.max()),
            "mean": float(prediction.mean()),
        },
        "tensor_stats": {
            "face": _array_stats(face_sample.face),
            "audio_bnf_window": _array_stats(audio),
            "prediction": _array_stats(prediction),
        },
        "artifacts": {
            "prediction": str(prediction_path.resolve()),
            "grid": str(grid_path.resolve()),
            "restored_frame": str(restored_frame_path.resolve()),
        },
    }
    if output_mp4_path is not None:
        metadata["artifacts"]["output_mp4"] = str(output_mp4_path.resolve())
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = {
        "prediction_path": str(prediction_path.resolve()),
        "grid_path": str(grid_path.resolve()),
        "restored_frame_path": str(restored_frame_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "metadata": metadata,
    }
    if output_mp4_path is not None:
        result["output_mp4_path"] = str(output_mp4_path.resolve())
    return result


@torch.inference_mode()
def run_manifest_sequence_inference(
    *,
    dataset_root: str | Path,
    manifest: str | Path,
    out_dir: str | Path,
    split: str | None = None,
    max_frames: int = 0,
    checkpoint: str | Path = "",
    init_bin: str | Path = "",
    hf_model_repo: str = "",
    hf_model_revision: str = "",
    hf_model_filename: str = "best.pt",
    hf_cache_dir: str = "",
    backend: str = "torch",
    ncnn_param: str | Path = "",
    audio_wav: str | Path = "",
    wenet_onnx: str | Path = "",
    alpha_bin: str | Path = "",
    output_mp4: str | Path = "output.mp4",
    fps: float = 25.0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = _resolve_path(root, manifest)
    records = _load_records(manifest_path, split=split, max_frames=max_frames)
    if str(output_mp4) and not str(audio_wav):
        raise ValueError("audio_wav is required when output_mp4 is set")

    runtime_device = torch.device(device)
    alpha_u8, alpha_source = _load_alpha_mask(alpha_bin)
    runtime, model_provenance = _load_prediction_runtime(
        backend=backend,
        ncnn_param=ncnn_param,
        checkpoint=checkpoint,
        init_bin=init_bin,
        hf_model_repo=hf_model_repo,
        hf_model_revision=hf_model_revision,
        hf_model_filename=hf_model_filename,
        hf_cache_dir=hf_cache_dir,
        device=runtime_device,
    )
    bnf_cache: dict[str, np.ndarray] = {}
    audio_source: dict[str, Any] | None = None
    if str(audio_wav):
        bnf, audio_source = _load_audio_bnf(
            root=root,
            record=records[0],
            audio_wav=audio_wav,
            wenet_onnx=wenet_onnx,
        )
        bnf_cache["__audio_wav__"] = bnf

    output = Path(out_dir)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    frame_stats: list[dict[str, Any]] = []

    for output_index, record in enumerate(records, start=1):
        frame_path = root / record.frame_path
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_path)
        if str(audio_wav):
            bnf = bnf_cache["__audio_wav__"]
        else:
            bnf_key = record.bnf_path
            if bnf_key not in bnf_cache:
                bnf, source = _load_audio_bnf(
                    root=root,
                    record=record,
                    audio_wav="",
                    wenet_onnx="",
                )
                bnf_cache[bnf_key] = bnf
                if audio_source is None:
                    audio_source = source
            bnf = bnf_cache[bnf_key]

        face_sample = make_face_training_sample(frame, record.bbox_xyxy)
        audio = get_bnf_window(bnf, record.audio_idx)
        prediction = runtime.predict(face_sample.face, audio)
        prediction_rgb = chw_norm_to_rgb_u8(prediction)
        restored_frame = restore_prediction_to_frame(
            frame,
            record.bbox_xyxy,
            face_sample.roi_168_bgr,
            prediction_rgb,
            alpha_u8=alpha_u8,
        )
        restored_path = frames_dir / f"{output_index:06d}.png"
        if not cv2.imwrite(str(restored_path), restored_frame):
            raise RuntimeError(f"Cannot write sequence frame: {restored_path}")
        frame_stats.append(
            {
                "output_index": output_index,
                "clip_id": record.clip_id,
                "frame_idx": record.frame_idx,
                "audio_idx": record.audio_idx,
                "frame_path": record.frame_path,
                "bbox_xyxy": list(record.bbox_xyxy),
                "crop_roi_shape": list(face_sample.roi_168_bgr.shape),
                "restored_paste_xyxy": list(record.bbox_xyxy),
                "tensor_stats": {
                    "face": _array_stats(face_sample.face),
                    "audio_bnf_window": _array_stats(audio),
                    "prediction": _array_stats(prediction),
                },
            }
        )

    output_mp4_path: Path | None = None
    if str(output_mp4):
        output_mp4_path = write_frame_sequence_audio_mp4(
            frames_dir=frames_dir,
            audio_wav=audio_wav,
            out_path=_resolve_output_mp4_path(output, output_mp4),
            fps=fps,
        )

    metadata: dict[str, Any] = {
        "kind": "manifest_sequence_inference",
        "dataset_root": str(root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "split": split or "all",
        "frame_count": len(records),
        "fps": float(fps),
        "model": model_provenance,
        "alpha": alpha_source,
        "audio_source": audio_source or {},
        "frames": frame_stats,
        "artifacts": {
            "frames_dir": str(frames_dir.resolve()),
        },
    }
    if output_mp4_path is not None:
        metadata["artifacts"]["output_mp4"] = str(output_mp4_path.resolve())
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result: dict[str, Any] = {
        "frames_dir": str(frames_dir.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "metadata": metadata,
    }
    if output_mp4_path is not None:
        result["output_mp4_path"] = str(output_mp4_path.resolve())
    return result
