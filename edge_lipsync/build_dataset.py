from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edge_lipsync.audio_features import (
    extract_bnf_windows_from_wav,
    load_wav_mono_f32,
    split_audio_blocks,
)
from edge_lipsync.preprocess import make_face_training_sample

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class BBoxGates:
    min_size: int = 32
    max_frame_fraction: float = 0.9
    max_jump_fraction: float = 0.25


@dataclass(frozen=True)
class DatasetBuildConfig:
    raw_video_dir: str
    dataset_root: str
    wenet_onnx: str
    fps: int = 25
    sample_rate: int = 16000
    split_strategy: str = "clip"
    validation_fraction: float = 0.2
    bbox_detector: str = "haar"
    preview_count: int = 8
    min_bbox_size: int = 32
    max_bbox_frame_fraction: float = 0.9
    max_bbox_jump_fraction: float = 0.25
    max_missing_gap: int = 3
    bbox_smooth_radius: int = 2
    silence_rms_threshold: float = 1e-3
    max_silence_fraction: float = 0.25


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"Required tool not found on PATH: {name}")
    return path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{process.stderr}")
    return process


def validate_stream_payload(payload: dict[str, Any]) -> None:
    stream_types = {str(stream.get("codec_type")) for stream in payload.get("streams", [])}
    if "video" not in stream_types:
        raise ValueError("Clip does not contain a readable video stream")
    if "audio" not in stream_types:
        raise ValueError("Clip does not contain a readable audio stream")


def probe_clip(path: Path) -> dict[str, Any]:
    ffprobe = require_tool("ffprobe")
    process = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(process.stdout)
    validate_stream_payload(payload)
    return payload


def normalize_clip(src: Path, out_dir: Path, fps: int, sample_rate: int) -> tuple[Path, Path]:
    ffmpeg = require_tool("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out = out_dir / "video_25fps.mp4"
    audio_out = out_dir / "audio.wav"
    run([ffmpeg, "-y", "-i", str(src), "-vf", f"fps={fps}", "-an", str(video_out)])
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            str(audio_out),
        ]
    )
    return video_out, audio_out


def extract_frames(video_path: Path, frames_dir: Path) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open normalized video: {video_path}")
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
        if not cv2.imwrite(str(frames_dir / f"{count:06d}.jpg"), frame):
            raise RuntimeError(f"Cannot write extracted frame {count} from {video_path}")
    capture.release()
    if count == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return count


def detect_largest_face(frame_bgr: np.ndarray) -> BBox | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda rect: int(rect[2]) * int(rect[3]))
    return int(x), int(y), int(x + width), int(y + height)


def bbox_quality_reason(
    bbox: BBox,
    frame_shape: tuple[int, ...],
    gates: BBoxGates,
) -> str | None:
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return "invalid"
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        return "outside_frame"
    if width < gates.min_size or height < gates.min_size:
        return "too_small"
    if width > frame_width * gates.max_frame_fraction:
        return "too_large"
    if height > frame_height * gates.max_frame_fraction:
        return "too_large"
    return None


def _tracking_jump_fraction(previous: BBox, current: BBox, frame_shape: tuple[int, ...]) -> float:
    frame_height, frame_width = frame_shape[:2]
    previous_center = ((previous[0] + previous[2]) / 2.0, (previous[1] + previous[3]) / 2.0)
    current_center = ((current[0] + current[2]) / 2.0, (current[1] + current[3]) / 2.0)
    center_distance = math.dist(previous_center, current_center)
    return center_distance / math.hypot(frame_width, frame_height)


def interpolate_short_bbox_gaps(
    boxes: dict[int, BBox | None],
    *,
    max_gap: int,
) -> tuple[dict[int, BBox | None], dict[int, list[str]]]:
    out = dict(boxes)
    flags: dict[int, list[str]] = {}
    indices = sorted(out)
    position = 0
    while position < len(indices):
        if out[indices[position]] is not None:
            position += 1
            continue
        start = position
        while position < len(indices) and out[indices[position]] is None:
            position += 1
        gap_indices = indices[start:position]
        if not gap_indices or len(gap_indices) > max_gap or start == 0 or position == len(indices):
            continue
        left_index = indices[start - 1]
        right_index = indices[position]
        left = out[left_index]
        right = out[right_index]
        if left is None or right is None:
            continue
        for offset, frame_index in enumerate(gap_indices, start=1):
            fraction = offset / (len(gap_indices) + 1)
            out[frame_index] = tuple(
                int(round(left[value_index] + (right[value_index] - left[value_index]) * fraction))
                for value_index in range(4)
            )
            flags[frame_index] = ["interpolated_bbox"]
    return out, flags


def smooth_bboxes(boxes: dict[int, BBox], *, radius: int = 2) -> dict[int, BBox]:
    if not boxes:
        return {}
    out: dict[int, BBox] = {}
    indices = sorted(boxes)
    for frame_index in indices:
        neighbors = [
            box
            for neighbor_index, box in boxes.items()
            if abs(neighbor_index - frame_index) <= radius
        ]
        values = np.asarray(neighbors, dtype=np.float32)
        out[frame_index] = tuple(int(round(value)) for value in values.mean(axis=0))
    return out


def clean_bboxes(
    boxes: dict[int, BBox | None],
    frame_shapes: dict[int, tuple[int, ...]],
    *,
    gates: BBoxGates,
    max_missing_gap: int,
    smooth_radius: int,
) -> tuple[dict[int, BBox], dict[int, list[str]], dict[str, int]]:
    drops: Counter[str] = Counter()
    quality_checked: dict[int, BBox | None] = {}
    for frame_index, bbox in boxes.items():
        if bbox is None:
            drops["missing_bbox"] += 1
            quality_checked[frame_index] = None
            continue
        reason = bbox_quality_reason(bbox, frame_shapes[frame_index], gates)
        if reason is not None:
            drops[reason] += 1
            quality_checked[frame_index] = None
            continue
        quality_checked[frame_index] = bbox

    interpolated, flags = interpolate_short_bbox_gaps(quality_checked, max_gap=max_missing_gap)
    valid = {frame_index: bbox for frame_index, bbox in interpolated.items() if bbox is not None}
    smoothed = smooth_bboxes(valid, radius=smooth_radius)
    previous: BBox | None = None
    for frame_index in sorted(smoothed):
        bbox = smoothed[frame_index]
        if previous is not None:
            jump = _tracking_jump_fraction(previous, bbox, frame_shapes[frame_index])
            if jump > gates.max_jump_fraction:
                drops["discontinuous_jump"] += 1
                del smoothed[frame_index]
                flags.pop(frame_index, None)
                continue
        previous = bbox
    return smoothed, flags, dict(drops)


def _silent_audio_indices(audio: np.ndarray, threshold: float) -> set[int]:
    blocks = split_audio_blocks(audio)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    return {index for index, value in enumerate(rms) if float(value) <= threshold}


def limit_silence(
    frame_indices: list[int],
    *,
    silent_audio_indices: set[int],
    max_silence_fraction: float,
) -> tuple[list[int], int]:
    voiced = [index for index in frame_indices if index - 1 not in silent_audio_indices]
    silent = [index for index in frame_indices if index - 1 in silent_audio_indices]
    if not 0.0 <= max_silence_fraction < 1.0:
        raise ValueError("max_silence_fraction must satisfy 0 <= value < 1")
    if not silent:
        return sorted(voiced), 0
    if voiced:
        max_silent = int(len(voiced) * max_silence_fraction / (1.0 - max_silence_fraction))
    else:
        max_silent = 1
    keep_silent = silent[:max_silent]
    kept = sorted(voiced + keep_silent)
    return kept, len(silent) - len(keep_silent)


def _validation_clip_ids(clip_ids: list[str], validation_fraction: float) -> set[str]:
    if len(clip_ids) <= 1:
        return set()
    count = max(1, int(round(len(clip_ids) * validation_fraction)))
    return set(clip_ids[-count:])


def write_manifest(
    dataset_root: Path,
    clips: list[dict[str, Any]],
    *,
    validation_fraction: float,
) -> dict[str, int]:
    clip_ids = [str(clip["clip_id"]) for clip in clips]
    val_clip_ids = _validation_clip_ids(clip_ids, validation_fraction)
    rows: list[dict[str, Any]] = []
    for clip in clips:
        clip_id = str(clip["clip_id"])
        frame_indices = [int(value) for value in clip["valid_frames"]]
        boxes: dict[int, BBox] = clip["bboxes"]
        flags: dict[int, list[str]] = clip["flags"]
        for position, frame_idx in enumerate(frame_indices):
            if val_clip_ids:
                split = "val" if clip_id in val_clip_ids else "train"
            else:
                split_at = max(1, int(round(len(frame_indices) * (1.0 - validation_fraction))))
                split = "val" if len(frame_indices) > 1 and position >= split_at else "train"
            rows.append(
                {
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "audio_idx": frame_idx - 1,
                    "frame_path": f"clips/{clip_id}/frames/{frame_idx:06d}.jpg",
                    "bbox_xyxy": [int(value) for value in boxes[frame_idx]],
                    "bnf_path": f"clips/{clip_id}/bnf.npy",
                    "split": split,
                    "flags": flags.get(frame_idx, []),
                }
            )
    manifest = dataset_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    split_counts = {
        "train": sum(row["split"] == "train" for row in rows),
        "val": sum(row["split"] == "val" for row in rows),
    }
    (dataset_root / "splits.json").write_text(
        json.dumps(split_counts, indent=2),
        encoding="utf-8",
    )
    return split_counts


def write_preview(frame_bgr: np.ndarray, bbox: BBox, out_dir: Path, *, frame_idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = make_face_training_sample(frame_bgr, bbox)
    overlay = frame_bgr.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
    prefix = f"{frame_idx:06d}"
    cv2.imwrite(str(out_dir / f"{prefix}_overlay.jpg"), overlay)
    cv2.imwrite(str(out_dir / f"{prefix}_real.jpg"), sample.real_patch_bgr)
    cv2.imwrite(str(out_dir / f"{prefix}_masked.jpg"), sample.masked_patch_bgr)
    cv2.imwrite(str(out_dir / f"{prefix}_target.jpg"), sample.real_patch_bgr)


def _select_preview_indices(frame_indices: list[int], count: int) -> list[int]:
    if count <= 0 or not frame_indices:
        return []
    positions = np.linspace(0, len(frame_indices) - 1, min(count, len(frame_indices)), dtype=int)
    return [frame_indices[position] for position in sorted(set(positions.tolist()))]


def _write_clip_failure(clip_dir: Path, clip_id: str, exc: Exception) -> dict[str, Any]:
    clip_dir.mkdir(parents=True, exist_ok=True)
    quality = {
        "clip_id": clip_id,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }
    (clip_dir / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return quality


def process_clip(video: Path, config: DatasetBuildConfig) -> dict[str, Any]:
    clip_id = video.stem
    clip_dir = Path(config.dataset_root) / "clips" / clip_id
    probe_clip(video)
    normalized_video, audio_path = normalize_clip(video, clip_dir, config.fps, config.sample_rate)
    frames_dir = clip_dir / "frames"
    frame_count = extract_frames(normalized_video, frames_dir)
    bnf = extract_bnf_windows_from_wav(audio_path, config.wenet_onnx)
    np.save(clip_dir / "bnf.npy", bnf.astype(np.float32))

    detected: dict[int, BBox | None] = {}
    frame_shapes: dict[int, tuple[int, ...]] = {}
    for frame_idx in range(1, frame_count + 1):
        frame = cv2.imread(str(frames_dir / f"{frame_idx:06d}.jpg"), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot read extracted frame {frame_idx} for {clip_id}")
        frame_shapes[frame_idx] = frame.shape
        detected[frame_idx] = detect_largest_face(frame)
    gates = BBoxGates(
        min_size=config.min_bbox_size,
        max_frame_fraction=config.max_bbox_frame_fraction,
        max_jump_fraction=config.max_bbox_jump_fraction,
    )
    boxes, flags, drop_counts = clean_bboxes(
        detected,
        frame_shapes,
        gates=gates,
        max_missing_gap=config.max_missing_gap,
        smooth_radius=config.bbox_smooth_radius,
    )
    candidates = [
        frame_idx
        for frame_idx in sorted(boxes)
        if frame_idx - 1 < int(bnf.shape[0])
    ]
    drop_counts["bnf_out_of_range"] = len(boxes) - len(candidates)
    audio = load_wav_mono_f32(audio_path)
    silent_indices = _silent_audio_indices(audio, config.silence_rms_threshold)
    valid_frames, silence_drops = limit_silence(
        candidates,
        silent_audio_indices=silent_indices,
        max_silence_fraction=config.max_silence_fraction,
    )
    drop_counts["silence_downsampled"] = silence_drops
    boxes = {frame_idx: boxes[frame_idx] for frame_idx in valid_frames}
    flags = {frame_idx: flags[frame_idx] for frame_idx in valid_frames if frame_idx in flags}
    (clip_dir / "bboxes.json").write_text(
        json.dumps({str(key): value for key, value in boxes.items()}, indent=2),
        encoding="utf-8",
    )
    for frame_idx in _select_preview_indices(valid_frames, config.preview_count):
        frame = cv2.imread(str(frames_dir / f"{frame_idx:06d}.jpg"), cv2.IMREAD_COLOR)
        if frame is not None:
            write_preview(frame, boxes[frame_idx], clip_dir / "previews", frame_idx=frame_idx)
    quality = {
        "clip_id": clip_id,
        "status": "ready" if valid_frames else "no_valid_samples",
        "frame_count": frame_count,
        "bnf_shape": list(bnf.shape),
        "detected_bboxes": sum(value is not None for value in detected.values()),
        "valid_samples": len(valid_frames),
        "drop_counts": drop_counts,
    }
    (clip_dir / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return {
        "clip_id": clip_id,
        "valid_frames": valid_frames,
        "bboxes": boxes,
        "flags": flags,
        "quality": quality,
    }


def build_dataset(config: DatasetBuildConfig, *, strict: bool = False) -> dict[str, Any]:
    if config.fps != 25 or config.sample_rate != 16000:
        raise ValueError("Phase 1 requires fps=25 and sample_rate=16000")
    if config.split_strategy != "clip":
        raise ValueError("Only split_strategy='clip' is supported")
    if config.bbox_detector != "haar":
        raise ValueError("Only bbox_detector='haar' is supported")
    raw_dir = Path(config.raw_video_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    wenet_onnx = Path(config.wenet_onnx)
    if not wenet_onnx.exists():
        raise FileNotFoundError(wenet_onnx)
    dataset_root = Path(config.dataset_root)
    (dataset_root / "clips").mkdir(parents=True, exist_ok=True)
    videos = sorted(
        path for path in raw_dir.iterdir() if path.suffix.lower() in {".mp4", ".mov", ".mkv"}
    )
    if not videos:
        raise ValueError(f"No videos found in {raw_dir}")

    clips: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for video in videos:
        try:
            clips.append(process_clip(video, config))
        except Exception as exc:
            failure = _write_clip_failure(dataset_root / "clips" / video.stem, video.stem, exc)
            failures.append(failure)
            if strict:
                raise
    split_counts = write_manifest(
        dataset_root,
        clips,
        validation_fraction=config.validation_fraction,
    )
    drop_counts: Counter[str] = Counter()
    for clip in clips:
        drop_counts.update(clip["quality"]["drop_counts"])
    summary = {
        "config": asdict(config),
        "processed_clips": len(clips),
        "failed_clips": failures,
        "valid_samples": sum(split_counts.values()),
        "split_counts": split_counts,
        "drop_counts": dict(drop_counts),
    }
    (dataset_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"processed_clips={summary['processed_clips']}")
    print(f"failed_clips={len(failures)}")
    print(f"valid_samples={summary['valid_samples']}")
    print(f"split_counts={split_counts}")
    print(f"drop_counts={dict(drop_counts)}")
    print(f"dataset_root={dataset_root.resolve()}")
    return summary
