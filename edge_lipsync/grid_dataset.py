from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_lipsync.build_dataset import DatasetBuildConfig, build_dataset, require_tool, run
from edge_lipsync.hub import push_dataset_snapshot

GRID_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}
GRID_AUDIO_SUFFIXES = {".wav"}


@dataclass(frozen=True)
class GridSample:
    speaker: str
    stem: str
    video_path: Path
    audio_path: Path | None = None


@dataclass(frozen=True)
class PreparedGridVideos:
    raw_video_dir: Path
    raw_video_paths: list[Path]

    @property
    def raw_video_count(self) -> int:
        return len(self.raw_video_paths)


@dataclass(frozen=True)
class GridBuildConfig:
    grid_root: str
    dataset_root: str
    wenet_onnx: str
    work_dir: str = ""
    speaker: str = ""
    max_videos: int = 0
    fps: int = 25
    sample_rate: int = 16000
    split_strategy: str = "clip"
    validation_fraction: float = 0.2
    bbox_detector: str = "mediapipe_face_landmarker"
    landmark_model_asset_path: str | None = None
    landmark_min_detection_confidence: float = 0.5
    landmark_min_tracking_confidence: float = 0.5
    landmark_refine_landmarks: bool = True
    preview_count: int = 8
    min_bbox_size: int = 32
    max_bbox_frame_fraction: float = 0.9
    max_bbox_jump_fraction: float = 0.25
    max_missing_gap: int = 3
    bbox_smooth_radius: int = 1
    silence_rms_threshold: float = 1e-3
    max_silence_fraction: float = 0.25
    dry_run: bool = False
    push: bool = False
    hf_repo_id: str = ""
    private: bool = True
    commit_message: str = "Upload GRID processed dataset snapshot"
    strict: bool = False


@dataclass(frozen=True)
class GridBuildResult:
    grid_root: Path
    dataset_root: Path
    work_dir: Path
    raw_video_dir: Path
    sample_count: int
    raw_video_count: int
    dry_run: bool
    pushed_revision: str | None = None
    hub_url: str | None = None
    build_summary: dict[str, Any] | None = None


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized.strip("._") or "sample"


def _infer_speaker(grid_root: Path, video_path: Path) -> str:
    relative = video_path.relative_to(grid_root)
    if len(relative.parts) <= 1:
        return ""
    return relative.parts[0]


def _find_paired_audio(grid_root: Path, speaker: str, video_path: Path) -> Path | None:
    same_dir = video_path.with_suffix(".wav")
    candidates = [same_dir]
    if video_path.parent.name.lower() == "video":
        candidates.append(video_path.parent.parent / "audio" / f"{video_path.stem}.wav")
    if speaker:
        candidates.append(grid_root / speaker / "audio" / f"{video_path.stem}.wav")
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in GRID_AUDIO_SUFFIXES:
            return candidate
    return None


def discover_grid_samples(
    grid_root: str | Path,
    *,
    speaker: str = "",
    max_videos: int = 0,
) -> list[GridSample]:
    root = Path(grid_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if max_videos < 0:
        raise ValueError("max_videos must be >= 0")

    search_root = root / speaker if speaker else root
    if not search_root.is_dir():
        return []

    videos = sorted(
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in GRID_VIDEO_SUFFIXES
    )
    samples: list[GridSample] = []
    for video_path in videos:
        detected_speaker = _infer_speaker(root, video_path)
        if speaker and detected_speaker != speaker:
            continue
        samples.append(
            GridSample(
                speaker=detected_speaker,
                stem=video_path.stem,
                video_path=video_path,
                audio_path=_find_paired_audio(root, detected_speaker, video_path),
            )
        )
        if max_videos and len(samples) >= max_videos:
            break
    return samples


def _raw_video_filename(sample: GridSample) -> str:
    prefix = _safe_component(sample.speaker)
    stem = _safe_component(sample.stem)
    return f"{prefix}_{stem}.mp4" if sample.speaker else f"{stem}.mp4"


def _mux_grid_sample(
    sample: GridSample,
    output_path: Path,
    *,
    fps: int,
    sample_rate: int,
) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(sample.video_path),
    ]
    if sample.audio_path is not None:
        command.extend(["-i", str(sample.audio_path)])
    command.extend(
        [
            "-r",
            str(fps),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
        ]
    )
    if sample.audio_path is not None:
        command.append("-shortest")
    command.append(str(output_path))
    run(command)


def prepare_grid_raw_videos(
    samples: list[GridSample],
    raw_video_dir: str | Path,
    *,
    fps: int = 25,
    sample_rate: int = 16000,
) -> PreparedGridVideos:
    out_dir = Path(raw_video_dir)
    raw_video_paths: list[Path] = []
    for sample in samples:
        output_path = out_dir / _raw_video_filename(sample)
        _mux_grid_sample(sample, output_path, fps=fps, sample_rate=sample_rate)
        raw_video_paths.append(output_path)
    return PreparedGridVideos(raw_video_dir=out_dir, raw_video_paths=raw_video_paths)


def _default_work_dir(dataset_root: Path) -> Path:
    return dataset_root.parent / f"{dataset_root.name}_grid_work"


def _dataset_config(config: GridBuildConfig, raw_video_dir: Path) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        raw_video_dir=str(raw_video_dir),
        dataset_root=config.dataset_root,
        wenet_onnx=config.wenet_onnx,
        fps=config.fps,
        sample_rate=config.sample_rate,
        split_strategy=config.split_strategy,
        validation_fraction=config.validation_fraction,
        bbox_detector=config.bbox_detector,
        landmark_model_asset_path=config.landmark_model_asset_path,
        landmark_min_detection_confidence=config.landmark_min_detection_confidence,
        landmark_min_tracking_confidence=config.landmark_min_tracking_confidence,
        landmark_refine_landmarks=config.landmark_refine_landmarks,
        preview_count=config.preview_count,
        min_bbox_size=config.min_bbox_size,
        max_bbox_frame_fraction=config.max_bbox_frame_fraction,
        max_bbox_jump_fraction=config.max_bbox_jump_fraction,
        max_missing_gap=config.max_missing_gap,
        bbox_smooth_radius=config.bbox_smooth_radius,
        silence_rms_threshold=config.silence_rms_threshold,
        max_silence_fraction=config.max_silence_fraction,
    )


def build_grid_dataset(config: GridBuildConfig) -> GridBuildResult:
    grid_root = Path(config.grid_root)
    dataset_root = Path(config.dataset_root)
    work_dir = Path(config.work_dir) if config.work_dir else _default_work_dir(dataset_root)
    raw_video_dir = work_dir / "raw_videos"
    if config.push and not config.dry_run and not config.hf_repo_id:
        raise ValueError("hf_repo_id is required when push=True")

    samples = discover_grid_samples(
        grid_root,
        speaker=config.speaker,
        max_videos=config.max_videos,
    )
    if not samples:
        speaker_note = f" for speaker {config.speaker!r}" if config.speaker else ""
        raise ValueError(f"No GRID videos found in {grid_root}{speaker_note}")

    if config.dry_run:
        return GridBuildResult(
            grid_root=grid_root,
            dataset_root=dataset_root,
            work_dir=work_dir,
            raw_video_dir=raw_video_dir,
            sample_count=len(samples),
            raw_video_count=0,
            dry_run=True,
        )

    prepared = prepare_grid_raw_videos(
        samples,
        raw_video_dir,
        fps=config.fps,
        sample_rate=config.sample_rate,
    )
    build_summary = build_dataset(
        _dataset_config(config, prepared.raw_video_dir),
        strict=config.strict,
    )
    pushed_revision: str | None = None
    hub_url: str | None = None
    if config.push:
        artifact = push_dataset_snapshot(
            dataset_root,
            config.hf_repo_id,
            private=config.private,
            commit_message=config.commit_message,
        )
        pushed_revision = artifact.resolved_revision
        hub_url = artifact.url
    return GridBuildResult(
        grid_root=grid_root,
        dataset_root=dataset_root,
        work_dir=work_dir,
        raw_video_dir=prepared.raw_video_dir,
        sample_count=len(samples),
        raw_video_count=prepared.raw_video_count,
        dry_run=False,
        pushed_revision=pushed_revision,
        hub_url=hub_url,
        build_summary=build_summary,
    )
