from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from datasets import (
    DownloadConfig,
    Features,
    Video,
    load_dataset,
)
from datasets import (
    config as datasets_config,
)

from edge_lipsync.build_dataset import DatasetBuildConfig, build_dataset
from edge_lipsync.hf_datasets import push_processed_dataset
from edge_lipsync.hub import HfApi, hf_hub_download
from edge_lipsync.progress import progress

DEFAULT_VIDEO_PREFIX = "xdub_teacher_pairs/videos"
DEFAULT_METADATA_MANIFEST = "xdub_teacher_pairs_manifest.json"
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".mpeg", ".mpg")


@dataclass(frozen=True)
class HfVideoFileSelection:
    video_files: list[str]
    speaker_counts: dict[str, int]


@dataclass(frozen=True)
class HfVideoDatasetBuildConfig:
    repo_id: str
    dataset_root: str
    wenet_onnx: str
    video_prefix: str = DEFAULT_VIDEO_PREFIX
    metadata_manifest: str = DEFAULT_METADATA_MANIFEST
    speaker_id: str = ""
    list_speakers: bool = False
    work_dir: str = ""
    cache_dir: str = ""
    download_max_workers: int = 1
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
    progress: bool = True
    dry_run: bool = False
    push: bool = False
    hf_output_repo_id: str = ""
    private: bool = True
    strict: bool = False


@dataclass(frozen=True)
class HfVideoDatasetBuildResult:
    repo_id: str
    dataset_root: Path
    work_dir: Path
    raw_video_dir: Path
    selected_video_count: int
    raw_video_count: int
    dry_run: bool
    selected_video_files: list[str]
    speaker_id: str = ""
    speaker_counts: dict[str, int] | None = None
    hub_url: str | None = None
    build_summary: dict[str, Any] | None = None


def _client(api: Any | None) -> Any:
    if api is not None:
        return api
    if HfApi is None:
        raise ImportError("Install huggingface-hub to use Hugging Face dataset integration")
    return HfApi()


def _normalized_prefix(prefix: str) -> str:
    return prefix.strip("/")


def _speaker_for_metadata_entry(entry: dict[str, Any]) -> str:
    src_speaker = str(entry.get("src_speaker") or "").strip()
    alt_speaker = str(entry.get("alt_speaker") or "").strip()
    if not src_speaker:
        return ""
    if alt_speaker and alt_speaker != src_speaker:
        return ""
    return src_speaker


def _video_path_for_metadata_entry(entry: dict[str, Any], video_root: str) -> str:
    video_id = str(entry.get("id") or "").strip()
    if not video_id:
        return ""
    return f"{video_root}/{video_id}.mp4"


def load_hf_video_metadata_manifest(
    *,
    repo_id: str,
    metadata_manifest: str = DEFAULT_METADATA_MANIFEST,
    cache_dir: str = "",
) -> list[dict[str, Any]]:
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "filename": metadata_manifest,
        "cache_dir": cache_dir or None,
    }
    path = hf_hub_download(**kwargs)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected {metadata_manifest} to contain a JSON list")
    entries = [entry for entry in payload if isinstance(entry, dict)]
    if len(entries) != len(payload):
        raise ValueError(f"Expected every {metadata_manifest} row to be a JSON object")
    return entries


def select_hf_video_dataset_files(
    repo_files: list[str],
    *,
    video_prefix: str = DEFAULT_VIDEO_PREFIX,
    max_videos: int = 0,
    speaker_id: str = "",
    metadata_entries: list[dict[str, Any]] | None = None,
) -> HfVideoFileSelection:
    if max_videos < 0:
        raise ValueError("max_videos must be >= 0")
    video_root = _normalized_prefix(video_prefix)
    speaker_id = speaker_id.strip()
    video_files = sorted(
        file_path
        for file_path in repo_files
        if file_path.startswith(f"{video_root}/") and file_path.lower().endswith(VIDEO_SUFFIXES)
    )
    speaker_counts: dict[str, int] = {}
    if metadata_entries is not None:
        repo_file_set = set(video_files)
        speaker_video_pairs: list[tuple[str, str]] = []
        counts: Counter[str] = Counter()
        for entry in metadata_entries:
            speaker = _speaker_for_metadata_entry(entry)
            video_file = _video_path_for_metadata_entry(entry, video_root)
            if not speaker or video_file not in repo_file_set:
                continue
            counts[speaker] += 1
            speaker_video_pairs.append((video_file, speaker))
        speaker_counts = dict(sorted(counts.items()))
        if speaker_id:
            video_files = sorted(
                video_file
                for video_file, speaker in speaker_video_pairs
                if speaker == speaker_id
            )
    elif speaker_id:
        raise ValueError("metadata_entries are required when speaker_id is set")

    if max_videos:
        video_files = video_files[:max_videos]
    return HfVideoFileSelection(video_files=video_files, speaker_counts=speaker_counts)


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


@contextmanager
def _limited_datasets_download_workers(max_workers: int) -> Iterator[None]:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    previous = datasets_config.HF_DATASETS_MULTITHREADING_MAX_WORKERS
    datasets_config.HF_DATASETS_MULTITHREADING_MAX_WORKERS = max_workers
    try:
        yield
    finally:
        datasets_config.HF_DATASETS_MULTITHREADING_MAX_WORKERS = previous


def download_hf_video_files(
    repo_id: str,
    video_files: list[str],
    raw_video_dir: str | Path,
    *,
    cache_dir: str = "",
    max_workers: int = 1,
    show_progress: bool = True,
) -> list[Path]:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    out_dir = Path(raw_video_dir)
    names = [Path(relative).name for relative in video_files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate raw video filename after flattening")

    download_config = DownloadConfig(
        cache_dir=cache_dir or None,
        resume_download=True,
        max_retries=5,
        num_proc=1,
    )
    with _limited_datasets_download_workers(max_workers):
        dataset = load_dataset(
            repo_id,
            data_files={"train": video_files},
            split="train",
            cache_dir=cache_dir or None,
            features=Features({"video": Video(decode=False)}),
            download_config=download_config,
            drop_labels=True,
            drop_metadata=True,
        )

    prepared: list[Path] = []
    for row in progress(
        dataset,
        enabled=show_progress,
        desc="prepare HF videos",
        total=len(video_files),
        unit="clip",
    ):
        video = cast(dict[str, Any], row)["video"]
        if not isinstance(video, dict) or not isinstance(video.get("path"), str):
            raise ValueError(f"Expected a local video path from datasets, got: {video!r}")
        src = Path(video["path"])
        if not src.is_file():
            raise FileNotFoundError(src)
        name = src.name
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
        progress=config.progress,
    )


def build_hf_video_dataset(
    config: HfVideoDatasetBuildConfig,
    *,
    api: Any | None = None,
) -> HfVideoDatasetBuildResult:
    if config.download_max_workers < 1:
        raise ValueError("download_max_workers must be >= 1")
    if (
        config.push
        and not config.dry_run
        and not config.list_speakers
        and not config.hf_output_repo_id
    ):
        raise ValueError("hf_output_repo_id is required when push=True")
    dataset_root = Path(config.dataset_root)
    work_dir = Path(config.work_dir) if config.work_dir else _default_work_dir(dataset_root)
    raw_video_dir = work_dir / "raw_videos"
    client = _client(api)
    repo_files = client.list_repo_files(
        repo_id=config.repo_id,
        repo_type="dataset",
    )
    metadata_entries = None
    if config.speaker_id or config.list_speakers:
        metadata_entries = load_hf_video_metadata_manifest(
            repo_id=config.repo_id,
            metadata_manifest=config.metadata_manifest,
            cache_dir=config.cache_dir,
        )
    selection = select_hf_video_dataset_files(
        list(repo_files),
        video_prefix=config.video_prefix,
        max_videos=config.max_videos,
        speaker_id=config.speaker_id,
        metadata_entries=metadata_entries,
    )
    if not selection.video_files:
        speaker_note = f" for speaker_id={config.speaker_id!r}" if config.speaker_id else ""
        raise ValueError(
            f"No video files found in {config.repo_id} under {config.video_prefix!r}{speaker_note}"
        )
    if config.dry_run or config.list_speakers:
        return HfVideoDatasetBuildResult(
            repo_id=config.repo_id,
            dataset_root=dataset_root,
            work_dir=work_dir,
            raw_video_dir=raw_video_dir,
            selected_video_count=len(selection.video_files),
            raw_video_count=0,
            dry_run=True,
            selected_video_files=selection.video_files,
            speaker_id=config.speaker_id,
            speaker_counts=selection.speaker_counts,
        )

    raw_video_paths = download_hf_video_files(
        config.repo_id,
        selection.video_files,
        raw_video_dir,
        cache_dir=config.cache_dir,
        max_workers=config.download_max_workers,
        show_progress=config.progress,
    )
    build_summary = build_dataset(
        _dataset_config(config, raw_video_dir),
        strict=config.strict,
    )
    hub_url: str | None = None
    if config.push:
        artifact = push_processed_dataset(
            dataset_root,
            config.hf_output_repo_id,
            private=config.private,
        )
        hub_url = artifact.url

    return HfVideoDatasetBuildResult(
        repo_id=config.repo_id,
        dataset_root=dataset_root,
        work_dir=work_dir,
        raw_video_dir=raw_video_dir,
        selected_video_count=len(selection.video_files),
        raw_video_count=len(raw_video_paths),
        dry_run=False,
        selected_video_files=selection.video_files,
        speaker_id=config.speaker_id,
        speaker_counts=selection.speaker_counts,
        hub_url=hub_url,
        build_summary=build_summary,
    )
