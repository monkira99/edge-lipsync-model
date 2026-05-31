from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_lipsync.build_dataset import DatasetBuildConfig, build_dataset
from edge_lipsync.hub import HfApi, push_dataset_snapshot, snapshot_download

DEFAULT_VIDEO_PREFIX = "xdub_teacher_pairs/videos"
DEFAULT_METADATA_PREFIX = "xdub_teacher_pairs/meta"
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".mpeg", ".mpg")


@dataclass(frozen=True)
class HfVideoFileSelection:
    video_files: list[str]
    allow_patterns: list[str]


@dataclass(frozen=True)
class HfVideoDatasetBuildConfig:
    repo_id: str
    revision: str
    dataset_root: str
    wenet_onnx: str
    video_prefix: str = DEFAULT_VIDEO_PREFIX
    metadata_prefix: str = DEFAULT_METADATA_PREFIX
    work_dir: str = ""
    cache_dir: str = ""
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
    hf_output_repo_id: str = ""
    private: bool = True
    commit_message: str = "Upload processed HF video dataset snapshot"
    strict: bool = False


@dataclass(frozen=True)
class HfVideoDatasetBuildResult:
    repo_id: str
    requested_revision: str
    dataset_root: Path
    work_dir: Path
    raw_video_dir: Path
    selected_video_count: int
    raw_video_count: int
    dry_run: bool
    selected_video_files: list[str]
    allow_patterns: list[str]
    snapshot_path: Path | None = None
    pushed_revision: str | None = None
    hub_url: str | None = None
    build_summary: dict[str, Any] | None = None


def _client(api: Any | None) -> Any:
    if api is not None:
        return api
    if HfApi is None:
        raise ImportError("Install huggingface-hub to use Hugging Face dataset integration")
    return HfApi()


def _require_revision(revision: str) -> None:
    if not revision:
        raise ValueError("Hugging Face dataset revision must be pinned and non-empty")


def _normalized_prefix(prefix: str) -> str:
    return prefix.strip("/")


def select_hf_video_dataset_files(
    repo_files: list[str],
    *,
    video_prefix: str = DEFAULT_VIDEO_PREFIX,
    metadata_prefix: str = DEFAULT_METADATA_PREFIX,
    max_videos: int = 0,
) -> HfVideoFileSelection:
    if max_videos < 0:
        raise ValueError("max_videos must be >= 0")
    video_root = _normalized_prefix(video_prefix)
    metadata_root = _normalized_prefix(metadata_prefix)
    video_files = sorted(
        file_path
        for file_path in repo_files
        if file_path.startswith(f"{video_root}/") and file_path.lower().endswith(VIDEO_SUFFIXES)
    )
    if max_videos:
        video_files = video_files[:max_videos]

    repo_file_set = set(repo_files)
    allow_patterns: list[str] = []
    for optional in ("README.md", "xdub_teacher_pairs_manifest.json"):
        if optional in repo_file_set:
            allow_patterns.append(optional)
    allow_patterns.extend(video_files)
    if metadata_root:
        for video_file in video_files:
            metadata_file = f"{metadata_root}/{Path(video_file).stem}.json"
            if metadata_file in repo_file_set:
                allow_patterns.append(metadata_file)
    return HfVideoFileSelection(video_files=video_files, allow_patterns=allow_patterns)


def _default_work_dir(dataset_root: Path) -> Path:
    return dataset_root.parent / f"{dataset_root.name}_hf_video_work"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_hf_video_raw_dir(
    snapshot_path: str | Path,
    video_files: list[str],
    raw_video_dir: str | Path,
) -> list[Path]:
    snapshot_root = Path(snapshot_path)
    out_dir = Path(raw_video_dir)
    prepared: list[Path] = []
    seen: set[str] = set()
    for relative in video_files:
        src = snapshot_root / relative
        if not src.is_file():
            raise FileNotFoundError(src)
        name = Path(relative).name
        if name in seen:
            raise ValueError(f"Duplicate raw video filename after flattening: {name}")
        seen.add(name)
        dst = out_dir / name
        _link_or_copy(src, dst)
        prepared.append(dst)
    return prepared


def _dataset_config(config: HfVideoDatasetBuildConfig, raw_video_dir: Path) -> DatasetBuildConfig:
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


def build_hf_video_dataset(
    config: HfVideoDatasetBuildConfig,
    *,
    api: Any | None = None,
) -> HfVideoDatasetBuildResult:
    _require_revision(config.revision)
    if config.push and not config.dry_run and not config.hf_output_repo_id:
        raise ValueError("hf_output_repo_id is required when push=True")
    dataset_root = Path(config.dataset_root)
    work_dir = Path(config.work_dir) if config.work_dir else _default_work_dir(dataset_root)
    raw_video_dir = work_dir / "raw_videos"
    client = _client(api)
    repo_files = client.list_repo_files(
        repo_id=config.repo_id,
        repo_type="dataset",
        revision=config.revision,
    )
    selection = select_hf_video_dataset_files(
        list(repo_files),
        video_prefix=config.video_prefix,
        metadata_prefix=config.metadata_prefix,
        max_videos=config.max_videos,
    )
    if not selection.video_files:
        raise ValueError(
            f"No video files found in {config.repo_id}@{config.revision} "
            f"under {config.video_prefix!r}"
        )
    if config.dry_run:
        return HfVideoDatasetBuildResult(
            repo_id=config.repo_id,
            requested_revision=config.revision,
            dataset_root=dataset_root,
            work_dir=work_dir,
            raw_video_dir=raw_video_dir,
            selected_video_count=len(selection.video_files),
            raw_video_count=0,
            dry_run=True,
            selected_video_files=selection.video_files,
            allow_patterns=selection.allow_patterns,
        )

    download_kwargs: dict[str, Any] = {
        "repo_id": config.repo_id,
        "repo_type": "dataset",
        "revision": config.revision,
        "allow_patterns": selection.allow_patterns,
    }
    if config.cache_dir:
        download_kwargs["cache_dir"] = config.cache_dir
    snapshot_path = Path(snapshot_download(**download_kwargs))
    raw_video_paths = prepare_hf_video_raw_dir(snapshot_path, selection.video_files, raw_video_dir)
    build_summary = build_dataset(
        _dataset_config(config, raw_video_dir),
        strict=config.strict,
    )
    pushed_revision: str | None = None
    hub_url: str | None = None
    if config.push:
        artifact = push_dataset_snapshot(
            dataset_root,
            config.hf_output_repo_id,
            private=config.private,
            commit_message=config.commit_message,
        )
        pushed_revision = artifact.resolved_revision
        hub_url = artifact.url

    return HfVideoDatasetBuildResult(
        repo_id=config.repo_id,
        requested_revision=config.revision,
        dataset_root=dataset_root,
        work_dir=work_dir,
        raw_video_dir=raw_video_dir,
        selected_video_count=len(selection.video_files),
        raw_video_count=len(raw_video_paths),
        dry_run=False,
        selected_video_files=selection.video_files,
        allow_patterns=selection.allow_patterns,
        snapshot_path=snapshot_path,
        pushed_revision=pushed_revision,
        hub_url=hub_url,
        build_summary=build_summary,
    )
