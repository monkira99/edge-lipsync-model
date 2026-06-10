from __future__ import annotations

import hashlib
import json
import math
import shutil
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Generic, Protocol, TypeVar, cast

import cv2
import numpy as np
from datasets import Array2D, Dataset, DatasetDict, Features, Image, Sequence, Value

from edge_lipsync.audio_features import (
    WenetRuntime,
    load_wav_mono_f32,
    split_audio_blocks,
)
from edge_lipsync.build_dataset import FRAME_SUFFIX, extract_frames, require_tool, run
from edge_lipsync.identity import (
    IdentityConfig,
    IdentityFrameError,
    IdentityRuntimeError,
    create_identity_runtime,
)
from edge_lipsync.landmarks import MediaPipeFaceLandmarkerDetector
from edge_lipsync.onnx_runtime import (
    OnnxRunLimiter,
    OnnxRuntimeError,
    resolve_onnx_providers,
)
from edge_lipsync.pose_pairing import (
    TRACKED_LANDMARK_INDICES,
    FrameObservation,
    HeadPose,
    IdentityMismatch,
    MatchConfig,
    PostCropAlignmentMismatch,
    SyncWindow,
    assign_sync_windows,
    assign_video_splits,
    build_sync_windows,
    estimate_head_pose,
    fill_missing_signal,
    laplacian_variance,
    mark_bbox_continuity,
    match_silent_observation,
    mouth_bbox,
    mouth_openness,
    select_idle_frame_indices,
    single_video_split,
    sync_reject_reason,
)
from edge_lipsync.preprocess import ROI_SOURCE_SIZE, BBox, landmarks_to_duix_roi
from edge_lipsync.progress import progress

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}
SCHEMA_VERSION = "edge_lipsync_silent_talking_pair_v2"
NORMALIZE_CACHE_VERSION = 1
FRAME_CACHE_VERSION = 1
ANALYSIS_CACHE_VERSION = 2
BNF_CACHE_VERSION = 1
CacheValue = TypeVar("CacheValue")

SILENT_TALKING_FEATURES = Features(
    {
        "schema_version": Value("string"),
        "persona_id": Value("string"),
        "pair_id": Value("string"),
        "talking_clip_id": Value("string"),
        "source_frame_idx": Value("int32"),
        "target_frame_idx": Value("int32"),
        "audio_idx": Value("int32"),
        "source_roi": Image(decode=True),
        "target_roi": Image(decode=True),
        "audio": Array2D(shape=(20, 256), dtype="float32"),
        "source_bbox_xyxy": Sequence(Value("int32"), length=4),
        "target_bbox_xyxy": Sequence(Value("int32"), length=4),
        "source_frame_width": Value("int32"),
        "source_frame_height": Value("int32"),
        "target_frame_width": Value("int32"),
        "target_frame_height": Value("int32"),
        "sample_weight": Value("float32"),
        "is_idle": Value("bool"),
        "sync_best_lag_frames": Value("int32"),
        "sync_correlation": Value("float32"),
        "sync_confidence": Value("string"),
        "pose_delta_yaw": Value("float32"),
        "pose_delta_pitch": Value("float32"),
        "pose_delta_roll": Value("float32"),
        "center_delta_x": Value("float32"),
        "center_delta_y": Value("float32"),
        "width_ratio": Value("float32"),
        "height_ratio": Value("float32"),
        "stable_landmark_alignment_rmse": Value("float32"),
        "mouth_center_delta_after_crop": Value("float32"),
        "identity_similarity": Value("float32"),
        "matching_score": Value("float32"),
        "valid_silent_candidate_count": Value("int32"),
        "second_best_matching_score": Value("float32"),
        "matching_score_margin": Value("float32"),
        "source_face_blur": Value("float32"),
        "target_face_blur": Value("float32"),
        "target_mouth_blur": Value("float32"),
        "flags": Sequence(Value("string")),
    }
)


class LandmarkDetector(Protocol):
    def detect_landmarks(
        self,
        frame_bgr: np.ndarray,
        /,
    ) -> Mapping[int, tuple[float, float]] | None: ...

    def close(self) -> None: ...


class IdentityRuntime(Protocol):
    def embed(
        self,
        frame_bgr: np.ndarray,
        landmarks: Mapping[int, tuple[float, float]],
    ) -> np.ndarray: ...


class WenetExtractor(Protocol):
    def extract_wav(self, wav_path: str | Path) -> np.ndarray: ...


@dataclass(frozen=True)
class CacheOutcome(Generic[CacheValue]):
    value: CacheValue
    hit: bool


@dataclass(frozen=True)
class BlurConfig:
    min_source_face_laplacian_variance: float = 60.0
    min_target_face_laplacian_variance: float = 60.0
    min_target_mouth_laplacian_variance: float = 40.0


@dataclass(frozen=True)
class SyncConfig:
    window_seconds: float = 2.0
    stride_seconds: float = 1.0
    max_lag_frames: int = 3
    max_reject_lag_frames: int = 2
    min_correlation: float = 0.20
    silence_rms_threshold: float = 0.001
    speech_fraction_threshold: float = 0.25


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    clip_workers: int = 4
    cuda_max_inflight: int = 2
    warn_on_cpu_fallback: bool = True


@dataclass(frozen=True)
class SilentTalkingBuildConfig:
    data_root: str
    persona_id: str
    snapshot_root: str
    work_root: str
    wenet_onnx: str
    landmark_model_asset_path: str | None = None
    fps: int = 25
    sample_rate: int = 16000
    validation_fraction: float = 0.20
    split_salt: str = "edge-lipsync-silent-talking-v1"
    idle_max_ratio: float = 0.10
    idle_sample_weight: float = 0.25
    preview_count_per_group: int = 8
    progress: bool = True
    strict: bool = False
    match: MatchConfig = field(default_factory=MatchConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    blur: BlurConfig = field(default_factory=BlurConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def persona_root(self) -> Path:
        return Path(self.data_root) / self.persona_id

    @property
    def silent_video_path(self) -> Path:
        return self.persona_root / "silent" / "defaultvideo.mp4"

    @property
    def talking_video_dir(self) -> Path:
        return self.persona_root / "talking"


@dataclass(frozen=True)
class PairDecisionResult:
    rows: list[dict[str, Any]]
    decisions: list[dict[str, Any]]


@dataclass(frozen=True)
class ClipBuildResult:
    clip_id: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    talking_observations: list[FrameObservation] = field(default_factory=list)
    cache_hits: dict[str, bool] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ClipBuildFailure:
    clip_id: str
    error: Exception


@dataclass(frozen=True)
class SilentTalkingBuildResult:
    snapshot_root: Path
    train_rows: int
    val_rows: int
    talking_clips: int
    failed_clips: tuple[str, ...]
    config_sha256: str
    hub_ref: str = ""


class ThreadLocalDetectorPool:
    def __init__(self, factory: Callable[[], LandmarkDetector]) -> None:
        self._factory = factory
        self._local = threading.local()
        self._detectors: list[LandmarkDetector] = []
        self._lock = threading.Lock()

    def get(self) -> LandmarkDetector:
        detector = getattr(self._local, "detector", None)
        if detector is None:
            detector = self._factory()
            self._local.detector = detector
            with self._lock:
                self._detectors.append(detector)
        return cast(LandmarkDetector, detector)

    def close_all(self) -> None:
        with self._lock:
            detectors = list(self._detectors)
            self._detectors.clear()
        for detector in detectors:
            detector.close()


def run_clip_workers(
    videos: list[Path],
    worker: Callable[[Path], ClipBuildResult],
    *,
    max_workers: int,
    strict: bool,
    show_progress: bool = False,
) -> tuple[list[ClipBuildResult], list[ClipBuildFailure]]:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    results: list[ClipBuildResult] = []
    failures: list[ClipBuildFailure] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(worker, video): video for video in sorted(videos)
        }
        completed = progress(
            as_completed(future_to_video),
            enabled=show_progress,
            desc="build talking clips",
            total=len(future_to_video),
            unit="clip",
        )
        for future in completed:
            video = future_to_video[future]
            try:
                results.append(future.result())
            except Exception as exc:
                if strict or _clip_failure_is_fatal(exc):
                    for pending in future_to_video:
                        pending.cancel()
                    raise
                failures.append(ClipBuildFailure(clip_id=video.stem, error=exc))
    results.sort(key=lambda result: result.clip_id)
    failures.sort(key=lambda failure: failure.clip_id)
    return results, failures


def discover_talking_videos(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    videos = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"No talking videos found in {root}")
    return videos


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def config_sha256(config: SilentTalkingBuildConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def normalize_visual_video(src: Path, out: Path, *, fps: int) -> Path:
    ffmpeg = require_tool("ffmpeg")
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-vf",
            f"fps={fps}",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            str(out),
        ]
    )
    return out


def normalize_talking_video(
    src: Path,
    video_out: Path,
    audio_out: Path,
    *,
    fps: int,
) -> tuple[Path, Path]:
    normalize_visual_video(src, video_out, fps=fps)
    ffmpeg = require_tool("ffmpeg")
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(audio_out),
        ]
    )
    return video_out, audio_out


def _read_matching_metadata(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def cached_normalize_visual_video(
    src: Path,
    out: Path,
    *,
    fps: int,
) -> CacheOutcome[Path]:
    metadata_path = out.parent / "normalize.meta.json"
    metadata = {
        "stage_version": NORMALIZE_CACHE_VERSION,
        "input": file_identity(src),
        "fps": fps,
        "audio": False,
    }
    if _read_matching_metadata(metadata_path, metadata) and _nonempty_file(out):
        return CacheOutcome(out, True)
    temporary = out.with_name(f"{out.stem}.building{out.suffix}")
    temporary.unlink(missing_ok=True)
    normalize_visual_video(src, temporary, fps=fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(out)
    write_json_atomic(metadata_path, metadata)
    return CacheOutcome(out, False)


def cached_normalize_talking_video(
    src: Path,
    video_out: Path,
    audio_out: Path,
    *,
    fps: int,
    sample_rate: int,
) -> CacheOutcome[tuple[Path, Path]]:
    metadata_path = video_out.parent / "normalize.meta.json"
    metadata = {
        "stage_version": NORMALIZE_CACHE_VERSION,
        "input": file_identity(src),
        "fps": fps,
        "sample_rate": sample_rate,
        "audio": True,
    }
    if (
        _read_matching_metadata(metadata_path, metadata)
        and _nonempty_file(video_out)
        and _nonempty_file(audio_out)
    ):
        return CacheOutcome((video_out, audio_out), True)
    temporary_video = video_out.with_name(
        f"{video_out.stem}.building{video_out.suffix}"
    )
    temporary_audio = audio_out.with_name(
        f"{audio_out.stem}.building{audio_out.suffix}"
    )
    temporary_video.unlink(missing_ok=True)
    temporary_audio.unlink(missing_ok=True)
    normalize_talking_video(
        src,
        temporary_video,
        temporary_audio,
        fps=fps,
    )
    video_out.parent.mkdir(parents=True, exist_ok=True)
    temporary_video.replace(video_out)
    temporary_audio.replace(audio_out)
    write_json_atomic(metadata_path, metadata)
    return CacheOutcome((video_out, audio_out), False)


def _frame_cache_valid(frames_dir: Path, frame_count: int) -> bool:
    if frame_count <= 0 or not frames_dir.is_dir():
        return False
    return all(
        _nonempty_file(frames_dir / f"{frame_idx:06d}{FRAME_SUFFIX}")
        for frame_idx in range(1, frame_count + 1)
    )


def cached_extract_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    show_progress: bool = True,
    progress_desc: str = "extract frames",
) -> CacheOutcome[int]:
    metadata_path = frames_dir.with_suffix(".meta.json")
    base_metadata = {
        "stage_version": FRAME_CACHE_VERSION,
        "input": file_identity(video_path),
    }
    if metadata_path.is_file():
        try:
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = {}
        frame_count = int(stored.get("frame_count", 0))
        expected = {**base_metadata, "frame_count": frame_count}
        if stored == expected and _frame_cache_valid(frames_dir, frame_count):
            return CacheOutcome(frame_count, True)
    temporary = frames_dir.with_name(frames_dir.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    frame_count = extract_frames(
        video_path,
        temporary,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    temporary.replace(frames_dir)
    write_json_atomic(metadata_path, {**base_metadata, "frame_count": frame_count})
    return CacheOutcome(frame_count, False)


def _valid_bnf_windows(value: np.ndarray) -> bool:
    return bool(
        value.ndim == 3
        and value.shape[0] > 0
        and value.shape[1:] == (20, 256)
        and value.dtype == np.float32
        and np.isfinite(value).all()
    )


def cached_bnf_windows(
    audio_path: Path,
    cache_path: Path,
    wenet_runtime: WenetExtractor,
    *,
    wenet_sha256: str,
    sample_rate: int,
) -> CacheOutcome[np.ndarray]:
    metadata_path = cache_path.with_suffix(".meta.json")
    metadata = {
        "stage_version": BNF_CACHE_VERSION,
        "audio": file_identity(audio_path),
        "wenet_sha256": wenet_sha256,
        "sample_rate": sample_rate,
    }
    if _read_matching_metadata(metadata_path, metadata) and cache_path.is_file():
        try:
            cached = np.load(cache_path, allow_pickle=False)
        except (OSError, ValueError):
            cached = np.asarray([], dtype=np.float32)
        if _valid_bnf_windows(cached):
            return CacheOutcome(np.ascontiguousarray(cached), True)
    value = np.ascontiguousarray(
        wenet_runtime.extract_wav(audio_path),
        dtype=np.float32,
    )
    if not _valid_bnf_windows(value):
        raise ValueError(f"Invalid Wenet BNF cache value: shape={value.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.stem + ".building.npy")
    with temporary.open("wb") as output:
        np.save(output, value, allow_pickle=False)
    temporary.replace(cache_path)
    write_json_atomic(metadata_path, metadata)
    return CacheOutcome(value, False)


def analysis_cache_metadata(
    input_path: Path,
    *,
    frame_count: int,
    landmark_model_path: Path | None,
    identity_sha256: str,
) -> dict[str, Any]:
    return {
        "stage_version": ANALYSIS_CACHE_VERSION,
        "input": file_identity(input_path),
        "frame_count": frame_count,
        "landmark_model": (
            file_identity(landmark_model_path)
            if landmark_model_path is not None
            else None
        ),
        "identity_sha256": identity_sha256,
    }


def observation_to_json(observation: FrameObservation) -> dict[str, Any]:
    return {
        "frame_idx": observation.frame_idx,
        "bbox_xyxy": list(observation.bbox_xyxy) if observation.bbox_xyxy else None,
        "frame_width": observation.frame_width,
        "frame_height": observation.frame_height,
        "landmarks": {str(index): list(point) for index, point in observation.landmarks.items()},
        "pose": asdict(observation.pose) if observation.pose else None,
        "face_blur": observation.face_blur,
        "mouth_blur": observation.mouth_blur,
        "mouth_open": observation.mouth_open,
        "landmark_valid": observation.landmark_valid,
        "identity_embedding": (
            observation.identity_embedding.tolist()
            if observation.identity_embedding is not None
            else None
        ),
        "bbox_continuity_valid": observation.bbox_continuity_valid,
        "reject_reason": observation.reject_reason,
    }


def observation_from_json(payload: dict[str, Any]) -> FrameObservation:
    bbox = payload.get("bbox_xyxy")
    bbox_xyxy: BBox | None = None
    if bbox is not None:
        values = tuple(int(value) for value in bbox)
        if len(values) != 4:
            raise ValueError(f"bbox_xyxy must have 4 values: {bbox}")
        bbox_xyxy = (values[0], values[1], values[2], values[3])
    pose = payload.get("pose")
    identity_embedding = payload.get("identity_embedding")
    return FrameObservation(
        frame_idx=int(payload["frame_idx"]),
        bbox_xyxy=bbox_xyxy,
        frame_width=int(payload["frame_width"]),
        frame_height=int(payload["frame_height"]),
        landmarks={
            int(index): (float(point[0]), float(point[1]))
            for index, point in dict(payload.get("landmarks", {})).items()
        },
        pose=HeadPose(**pose) if pose else None,
        face_blur=float(payload["face_blur"]),
        mouth_blur=float(payload["mouth_blur"]),
        mouth_open=float(payload["mouth_open"]),
        landmark_valid=bool(payload["landmark_valid"]),
        identity_embedding=(
            np.asarray(identity_embedding, dtype=np.float32)
            if identity_embedding is not None
            else None
        ),
        bbox_continuity_valid=bool(payload.get("bbox_continuity_valid", True)),
        reject_reason=str(payload.get("reject_reason", "")),
    )


def _analysis_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def _load_analysis_cache(
    cache_path: Path,
    cache_metadata: dict[str, Any],
) -> list[FrameObservation] | None:
    meta_path = _analysis_meta_path(cache_path)
    if not cache_path.is_file() or not meta_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if metadata != cache_metadata:
        return None
    observations: list[FrameObservation] = []
    with cache_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                observations.append(observation_from_json(json.loads(line)))
    return observations


def _write_analysis_cache(
    cache_path: Path,
    cache_metadata: dict[str, Any],
    observations: list[FrameObservation],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for observation in observations:
            file.write(json.dumps(observation_to_json(observation), sort_keys=True) + "\n")
    temporary.replace(cache_path)
    write_json_atomic(_analysis_meta_path(cache_path), cache_metadata)


def _invalid_observation(
    *,
    frame_idx: int,
    frame_shape: tuple[int, ...],
    reject_reason: str,
) -> FrameObservation:
    frame_height, frame_width = frame_shape[:2]
    return FrameObservation(
        frame_idx=frame_idx,
        bbox_xyxy=None,
        frame_width=frame_width,
        frame_height=frame_height,
        landmarks={},
        pose=None,
        face_blur=0.0,
        mouth_blur=0.0,
        mouth_open=0.0,
        landmark_valid=False,
        identity_embedding=None,
        bbox_continuity_valid=False,
        reject_reason=reject_reason,
    )


def _crop_bbox(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError(f"bbox produced empty ROI: {bbox}")
    return roi


def analyze_frames(
    frames_dir: str | Path,
    *,
    frame_count: int,
    detector: LandmarkDetector,
    identity_runtime: IdentityRuntime,
    cache_path: str | Path,
    cache_metadata: dict[str, Any],
    is_target: bool,
    show_progress: bool = True,
) -> list[FrameObservation]:
    root = Path(frames_dir)
    cache = Path(cache_path)
    cached = _load_analysis_cache(cache, cache_metadata)
    if cached is not None:
        return cached

    observations: list[FrameObservation] = []
    frame_indices = progress(
        range(1, frame_count + 1),
        enabled=show_progress,
        desc="analyze target" if is_target else "analyze silent",
        total=frame_count,
        unit="frame",
    )
    for frame_idx in frame_indices:
        frame_path = root / f"{frame_idx:06d}{FRAME_SUFFIX}"
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_path)
        landmarks = detector.detect_landmarks(frame)
        if landmarks is None:
            observations.append(
                _invalid_observation(
                    frame_idx=frame_idx,
                    frame_shape=frame.shape,
                    reject_reason="face_detection_failed",
                )
            )
            continue
        if any(index not in landmarks for index in TRACKED_LANDMARK_INDICES):
            observations.append(
                _invalid_observation(
                    frame_idx=frame_idx,
                    frame_shape=frame.shape,
                    reject_reason="landmark_missing",
                )
            )
            continue
        tracked_landmarks = {
            index: (float(landmarks[index][0]), float(landmarks[index][1]))
            for index in TRACKED_LANDMARK_INDICES
        }
        try:
            bbox = landmarks_to_duix_roi(tracked_landmarks, frame.shape)
            pose = estimate_head_pose(tracked_landmarks, frame.shape)
            face_blur = laplacian_variance(_crop_bbox(frame, bbox))
            mouth_box = mouth_bbox(tracked_landmarks, frame.shape)
            mouth_blur = laplacian_variance(_crop_bbox(frame, mouth_box))
            mouth_open = mouth_openness(tracked_landmarks)
        except Exception:
            observations.append(
                _invalid_observation(
                    frame_idx=frame_idx,
                    frame_shape=frame.shape,
                    reject_reason="analysis_failed",
                )
            )
            continue
        try:
            identity_embedding = identity_runtime.embed(frame, tracked_landmarks)
        except IdentityFrameError as exc:
            observations.append(
                _invalid_observation(
                    frame_idx=frame_idx,
                    frame_shape=frame.shape,
                    reject_reason=str(exc),
                )
            )
            continue
        frame_height, frame_width = frame.shape[:2]
        observations.append(
            FrameObservation(
                frame_idx=frame_idx,
                bbox_xyxy=bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                landmarks=tracked_landmarks,
                pose=pose,
                face_blur=face_blur,
                mouth_blur=mouth_blur,
                mouth_open=mouth_open,
                landmark_valid=True,
                identity_embedding=identity_embedding,
            )
        )
    observations = mark_bbox_continuity(observations)
    _write_analysis_cache(cache, cache_metadata, observations)
    return observations


def encode_png(image_bgr: np.ndarray) -> dict[str, Any]:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Cannot encode ROI as PNG")
    return {"bytes": encoded.tobytes(), "path": None}


def _sync_for_observation(
    observation: FrameObservation,
    assignments: dict[int, SyncWindow],
) -> SyncWindow:
    return assignments[observation.frame_idx - 1]


def _base_decision(
    observation: FrameObservation,
    *,
    split: str,
    window: SyncWindow,
    audio_idx: int,
    bnf_available: bool,
) -> dict[str, Any]:
    return {
        "frame_idx": observation.frame_idx,
        "split": split,
        "status": "rejected",
        "reject_reason": None,
        "landmark_valid": observation.landmark_valid,
        "bbox_continuity_valid": observation.bbox_continuity_valid,
        "source_face_blur": None,
        "target_face_blur": observation.face_blur,
        "target_mouth_blur": observation.mouth_blur,
        "sync_window_id": window.window_id,
        "sync_has_speech": window.has_speech,
        "sync_best_lag_frames": window.best_lag_frames,
        "sync_correlation": window.best_correlation,
        "sync_confidence": window.confidence,
        "audio_idx": audio_idx,
        "bnf_available": bnf_available,
        "valid_silent_candidate_count": 0,
        "selected_source_frame_idx": None,
        "matching_score": None,
        "stable_landmark_alignment_rmse": None,
        "mouth_center_delta_after_crop": None,
        "identity_similarity": None,
        "identity_mismatch_candidate_count": 0,
        "identity_max_rejected_similarity": None,
        "post_crop_alignment_mismatch_stable_landmark": 0,
        "post_crop_alignment_mismatch_mouth_center": 0,
    }


def _finite_optional(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        raise ValueError(f"Non-finite numeric value: {value}")
    return float(value)


def _reject(decision: dict[str, Any], reason: str) -> None:
    decision["status"] = "rejected"
    decision["reject_reason"] = reason


def _valid_silent_sources(
    silent_observations: list[FrameObservation],
    blur: BlurConfig,
) -> list[FrameObservation]:
    return [
        observation
        for observation in silent_observations
        if observation.landmark_valid
        and observation.bbox_continuity_valid
        and observation.bbox_xyxy is not None
        and observation.pose is not None
        and observation.identity_embedding is not None
        and observation.face_blur >= blur.min_source_face_laplacian_variance
    ]


def build_pair_decisions(
    *,
    talking_observations: list[FrameObservation],
    silent_observations: list[FrameObservation],
    bnf_windows: np.ndarray,
    audio_rms: np.ndarray,
    config: SilentTalkingBuildConfig,
    split_for_frame: Callable[[int], str],
    talking_clip_id: str = "talk",
) -> PairDecisionResult:
    if bnf_windows.ndim != 3 or bnf_windows.shape[1:] != (20, 256):
        raise ValueError(f"Expected precomputed BNF windows [T,20,256], got {bnf_windows.shape}")
    if not talking_observations:
        return PairDecisionResult(rows=[], decisions=[])
    frame_count = max(observation.frame_idx for observation in talking_observations)
    audio_signal = np.zeros(frame_count, dtype=np.float32)
    audio_available = min(frame_count, len(audio_rms))
    if audio_available:
        audio_signal[:audio_available] = audio_rms[:audio_available].astype(np.float32)
    mouth_signal = np.full(frame_count, np.nan, dtype=np.float32)
    for observation in talking_observations:
        if observation.landmark_valid:
            mouth_signal[observation.frame_idx - 1] = observation.mouth_open
    sync_windows = build_sync_windows(
        audio_signal,
        fill_missing_signal(mouth_signal),
        fps=config.fps,
        window_seconds=config.sync.window_seconds,
        stride_seconds=config.sync.stride_seconds,
        max_lag_frames=config.sync.max_lag_frames,
        silence_rms_threshold=config.sync.silence_rms_threshold,
        speech_fraction_threshold=config.sync.speech_fraction_threshold,
        min_correlation=config.sync.min_correlation,
    )
    assignments = assign_sync_windows(frame_count=frame_count, windows=sync_windows)
    valid_silent = _valid_silent_sources(silent_observations, config.blur)
    provisional_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    decision_by_frame: dict[int, dict[str, Any]] = {}

    for observation in talking_observations:
        audio_idx = observation.frame_idx - 1
        bnf_available = 0 <= audio_idx < int(bnf_windows.shape[0])
        split = split_for_frame(observation.frame_idx)
        window = _sync_for_observation(observation, assignments)
        decision = _base_decision(
            observation,
            split=split,
            window=window,
            audio_idx=audio_idx,
            bnf_available=bnf_available,
        )
        decisions.append(decision)
        decision_by_frame[observation.frame_idx] = decision

        if not observation.landmark_valid:
            _reject(decision, observation.reject_reason or "landmark_missing")
            continue
        if not observation.bbox_continuity_valid:
            _reject(decision, observation.reject_reason or "bbox_discontinuity")
            continue
        if observation.face_blur < config.blur.min_target_face_laplacian_variance:
            _reject(decision, "target_face_blur")
            continue
        if observation.mouth_blur < config.blur.min_target_mouth_laplacian_variance:
            _reject(decision, "target_mouth_blur")
            continue
        if not bnf_available:
            _reject(decision, "bnf_out_of_range")
            continue
        sync_reason = sync_reject_reason(
            window,
            min_correlation=config.sync.min_correlation,
            max_abs_lag=config.sync.max_reject_lag_frames,
        )
        if sync_reason is not None:
            _reject(decision, sync_reason)
            continue
        try:
            match = match_silent_observation(
                observation,
                valid_silent,
                config.match,
                min_identity_similarity=config.identity.min_cosine_similarity,
            )
        except ValueError as exc:
            if isinstance(exc, PostCropAlignmentMismatch):
                decision["post_crop_alignment_mismatch_stable_landmark"] = (
                    int(exc.stable_landmark_failures > 0)
                )
                decision["post_crop_alignment_mismatch_mouth_center"] = (
                    int(exc.mouth_center_failures > 0)
                )
                decision["identity_mismatch_candidate_count"] = (
                    exc.identity_mismatch_candidate_count
                )
                decision["identity_max_rejected_similarity"] = _finite_optional(
                    exc.identity_max_rejected_similarity
                )
            elif isinstance(exc, IdentityMismatch):
                decision["identity_mismatch_candidate_count"] = exc.candidate_count
                decision["identity_max_rejected_similarity"] = _finite_optional(
                    exc.max_similarity
                )
            _reject(decision, str(exc))
            continue

        source = match.selected
        assert source.bbox_xyxy is not None
        assert observation.bbox_xyxy is not None
        flags: list[str] = []
        if window.has_speech and window.confidence == "low":
            flags.append("low_sync_confidence")
        is_idle = not window.has_speech
        decision.update(
            {
                "source_face_blur": source.face_blur,
                "valid_silent_candidate_count": match.valid_candidate_count,
                "selected_source_frame_idx": source.frame_idx,
                "matching_score": match.matching_score,
                "stable_landmark_alignment_rmse": match.alignment.stable_landmark_rmse,
                "mouth_center_delta_after_crop": match.alignment.mouth_center_delta,
                "identity_similarity": match.identity_similarity,
                "identity_mismatch_candidate_count": (
                    match.identity_mismatch_candidate_count
                ),
                "identity_max_rejected_similarity": _finite_optional(
                    match.identity_max_rejected_similarity
                ),
            }
        )
        provisional_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "persona_id": config.persona_id,
                "pair_id": (
                    f"{talking_clip_id}__target_{observation.frame_idx:06d}"
                    f"__silent_{source.frame_idx:06d}"
                ),
                "talking_clip_id": talking_clip_id,
                "source_frame_idx": source.frame_idx,
                "target_frame_idx": observation.frame_idx,
                "audio_idx": audio_idx,
                "audio": np.ascontiguousarray(bnf_windows[audio_idx].astype(np.float32)),
                "source_bbox_xyxy": [int(value) for value in source.bbox_xyxy],
                "target_bbox_xyxy": [int(value) for value in observation.bbox_xyxy],
                "source_frame_width": source.frame_width,
                "source_frame_height": source.frame_height,
                "target_frame_width": observation.frame_width,
                "target_frame_height": observation.frame_height,
                "sample_weight": config.idle_sample_weight if is_idle else 1.0,
                "is_idle": is_idle,
                "sync_best_lag_frames": window.best_lag_frames,
                "sync_correlation": float(window.best_correlation),
                "sync_confidence": window.confidence,
                "pose_delta_yaw": float(match.pose_delta.yaw),
                "pose_delta_pitch": float(match.pose_delta.pitch),
                "pose_delta_roll": float(match.pose_delta.roll),
                "center_delta_x": float(match.center_delta_x),
                "center_delta_y": float(match.center_delta_y),
                "width_ratio": float(match.width_ratio),
                "height_ratio": float(match.height_ratio),
                "stable_landmark_alignment_rmse": float(match.alignment.stable_landmark_rmse),
                "mouth_center_delta_after_crop": float(match.alignment.mouth_center_delta),
                "identity_similarity": float(match.identity_similarity),
                "matching_score": float(match.matching_score),
                "valid_silent_candidate_count": int(match.valid_candidate_count),
                "second_best_matching_score": _finite_optional(match.second_best_score),
                "matching_score_margin": _finite_optional(match.matching_score_margin),
                "source_face_blur": float(source.face_blur),
                "target_face_blur": float(observation.face_blur),
                "target_mouth_blur": float(observation.mouth_blur),
                "flags": flags,
                "split": split,
            }
        )

    rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        split_rows = [row for row in provisional_rows if row["split"] == split]
        speech_rows = [row for row in split_rows if not row["is_idle"]]
        idle_rows = [row for row in split_rows if row["is_idle"]]
        selected_idle = set(
            select_idle_frame_indices(
                idle_frame_indices=[int(row["target_frame_idx"]) for row in idle_rows],
                speech_frame_indices=[int(row["target_frame_idx"]) for row in speech_rows],
                max_ratio=config.idle_max_ratio,
            )
        )
        for row in speech_rows:
            decision_by_frame[int(row["target_frame_idx"])]["status"] = "retained"
            rows.append(row)
        for row in idle_rows:
            frame_idx = int(row["target_frame_idx"])
            decision = decision_by_frame[frame_idx]
            if frame_idx in selected_idle:
                decision["status"] = "retained"
                rows.append(row)
            else:
                decision["status"] = "idle_downsampled"
                decision["reject_reason"] = "idle_downsampled"
    rows.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["talking_clip_id"]),
            int(row["target_frame_idx"]),
        )
    )
    return PairDecisionResult(rows=rows, decisions=decisions)


def _read_frame(frames_dir: Path, frame_idx: int) -> np.ndarray:
    path = frames_dir / f"{frame_idx:06d}{FRAME_SUFFIX}"
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(path)
    return frame


def attach_roi_images(
    row: dict[str, Any],
    *,
    silent_frames_dir: Path,
    talking_frames_dir: Path,
) -> dict[str, Any]:
    source_frame = _read_frame(silent_frames_dir, int(row["source_frame_idx"]))
    target_frame = _read_frame(talking_frames_dir, int(row["target_frame_idx"]))
    sx1, sy1, sx2, sy2 = [int(value) for value in row["source_bbox_xyxy"]]
    tx1, ty1, tx2, ty2 = [int(value) for value in row["target_bbox_xyxy"]]
    source_crop = source_frame[sy1:sy2, sx1:sx2]
    target_crop = target_frame[ty1:ty2, tx1:tx2]
    if source_crop.size == 0 or target_crop.size == 0:
        raise ValueError(f"Cannot attach empty ROI for pair_id={row.get('pair_id')}")
    source_roi = cv2.resize(
        source_crop,
        (ROI_SOURCE_SIZE, ROI_SOURCE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    target_roi = cv2.resize(
        target_crop,
        (ROI_SOURCE_SIZE, ROI_SOURCE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return {
        **row,
        "source_roi": encode_png(source_roi),
        "target_roi": encode_png(target_roi),
    }


def _numeric_values(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for key, value in row.items():
        if key in {"audio", "source_roi", "target_roi"}:
            continue
        if isinstance(value, bool | str) or value is None:
            continue
        if isinstance(value, int | float):
            values.append(float(value))
    audio = np.asarray(row.get("audio"), dtype=np.float32)
    if audio.size:
        values.extend(float(value) for value in audio.reshape(-1)[:: max(1, audio.size // 128)])
    return values


def _validate_snapshot_row(row: dict[str, Any]) -> None:
    for value in _numeric_values(row):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value in row {row.get('pair_id')}")
    if np.asarray(row["audio"], dtype=np.float32).shape != (20, 256):
        raise ValueError(f"Invalid audio shape in row {row.get('pair_id')}")
    if int(row["valid_silent_candidate_count"]) <= 0:
        raise ValueError(f"Row has no valid silent candidates: {row.get('pair_id')}")


def build_dataset_dict(rows: list[dict[str, Any]]) -> DatasetDict:
    for row in rows:
        _validate_snapshot_row(row)
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val")
    }
    if not by_split["train"] or not by_split["val"]:
        raise ValueError("Both train and val splits must be non-empty")
    datasets = {
        split: Dataset.from_list(
            [{key: value for key, value in row.items() if key != "split"} for row in split_rows],
            features=SILENT_TALKING_FEATURES,
        )
        for split, split_rows in by_split.items()
    }
    return DatasetDict(list(datasets.items()))


def write_frame_decisions(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not decisions:
        decisions = [{"frame_idx": -1, "status": "empty", "reject_reason": "no_frames"}]
    Dataset.from_list(decisions).to_parquet(str(path))


def _distribution(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def _pose_by_mouth_quantile(observations: list[FrameObservation]) -> list[dict[str, float | int]]:
    valid = [item for item in observations if item.pose is not None and item.landmark_valid]
    if not valid:
        return []
    openness = np.asarray([item.mouth_open for item in valid], dtype=np.float64)
    quantiles = np.quantile(openness, [0.0, 0.25, 0.50, 0.75, 1.0])
    out: list[dict[str, float | int]] = []
    for index in range(4):
        low = quantiles[index]
        high = quantiles[index + 1]
        if index == 3:
            bucket = [item for item in valid if low <= item.mouth_open <= high]
        else:
            bucket = [item for item in valid if low <= item.mouth_open < high]
        if not bucket:
            continue
        yaw = np.asarray([item.pose.yaw for item in bucket if item.pose is not None])
        pitch = np.asarray([item.pose.pitch for item in bucket if item.pose is not None])
        roll = np.asarray([item.pose.roll for item in bucket if item.pose is not None])
        out.append(
            {
                "quantile": index,
                "count": len(bucket),
                "mouth_open_min": float(low),
                "mouth_open_max": float(high),
                "yaw_mean": float(yaw.mean()),
                "yaw_std": float(yaw.std()),
                "pitch_mean": float(pitch.mean()),
                "pitch_std": float(pitch.std()),
                "roll_mean": float(roll.mean()),
                "roll_std": float(roll.std()),
            }
        )
    return out


def quality_report(
    *,
    clip_id: str,
    talking_observations: list[FrameObservation],
    decisions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reject_counts = Counter(
        str(decision["reject_reason"])
        for decision in decisions
        if decision.get("reject_reason")
    )
    stable_mismatch = sum(
        int(decision.get("post_crop_alignment_mismatch_stable_landmark") or 0)
        for decision in decisions
    )
    mouth_mismatch = sum(
        int(decision.get("post_crop_alignment_mismatch_mouth_center") or 0)
        for decision in decisions
    )
    retained = [row for row in rows if row["talking_clip_id"] == clip_id]
    report = {
        "clip_id": clip_id,
        "status": "ready" if retained else "no_valid_samples",
        "frame_count": len(talking_observations),
        "valid_analysis_count": sum(item.landmark_valid for item in talking_observations),
        "rejection_counts": dict(reject_counts),
        "bbox_jump_count": reject_counts.get("bbox_discontinuity", 0),
        "blur": {
            "target_face": _distribution([item.face_blur for item in talking_observations]),
            "target_mouth": _distribution([item.mouth_blur for item in talking_observations]),
        },
        "sync": {
            "speech_windows": sum(
                decision.get("sync_has_speech") is True for decision in decisions
            ),
            "idle_windows": sum(
                decision.get("sync_has_speech") is False for decision in decisions
            ),
            "low_confidence_frames": sum(
                decision.get("sync_confidence") == "low" for decision in decisions
            ),
            "rejected_sync_frames": reject_counts.get("sync_lag", 0),
            "best_lag": _distribution([float(d["sync_best_lag_frames"]) for d in decisions]),
            "correlation": _distribution([float(d["sync_correlation"]) for d in decisions]),
        },
        "pose_geometry_no_match_count": reject_counts.get("pose_geometry_no_match", 0),
        "post_crop_alignment_mismatch_count": reject_counts.get(
            "post_crop_alignment_mismatch", 0
        ),
        "post_crop_alignment_mismatch_stable_landmark": stable_mismatch,
        "post_crop_alignment_mismatch_mouth_center": mouth_mismatch,
        "identity_mismatch_count": reject_counts.get("identity_mismatch", 0),
        "identity_similarity": _distribution(
            [float(row["identity_similarity"]) for row in retained]
        ),
        "identity_max_rejected_similarity": _distribution(
            [
                float(decision["identity_max_rejected_similarity"])
                for decision in decisions
                if decision.get("identity_max_rejected_similarity") is not None
            ]
        ),
        "matching_score": _distribution([float(row["matching_score"]) for row in retained]),
        "valid_silent_candidate_count": _distribution(
            [float(row["valid_silent_candidate_count"]) for row in retained]
        ),
        "matching_score_margin": _distribution(
            [
                float(row["matching_score_margin"])
                for row in retained
                if row["matching_score_margin"] is not None
            ]
        ),
        "pose_by_mouth_openness_quantile": _pose_by_mouth_quantile(talking_observations),
        "speech_pair_count": sum(not row["is_idle"] for row in retained),
        "retained_idle_pair_count": sum(row["is_idle"] for row in retained),
    }
    if identity is not None:
        report["identity"] = identity
    return report


def _decode_png_cell(cell: dict[str, Any]) -> np.ndarray:
    data = cell.get("bytes")
    if not isinstance(data, bytes):
        raise ValueError("Preview ROI cell does not contain bytes")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode preview ROI")
    return image


def _text_panel(lines: list[str], *, width: int = 360, height: int = ROI_SOURCE_SIZE) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 24
    for line in lines[:8]:
        cv2.putText(
            panel,
            str(line)[:48],
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 20
    return panel


def _write_pair_preview(path: Path, row: dict[str, Any]) -> None:
    source = _decode_png_cell(row["source_roi"])
    target = _decode_png_cell(row["target_roi"])
    panel = _text_panel(
        [
            str(row["pair_id"]),
            f"score={float(row['matching_score']):.4f}",
            f"pose=({float(row['pose_delta_yaw']):.2f},"
            f"{float(row['pose_delta_pitch']):.2f},{float(row['pose_delta_roll']):.2f})",
            f"center=({float(row['center_delta_x']):.4f},{float(row['center_delta_y']):.4f})",
            f"scale=({float(row['width_ratio']):.3f},{float(row['height_ratio']):.3f})",
            f"identity={float(row['identity_similarity']):.4f}",
            f"sync={row['sync_best_lag_frames']} {row['sync_confidence']}",
            f"blur=({float(row['source_face_blur']):.1f},"
            f"{float(row['target_face_blur']):.1f},{float(row['target_mouth_blur']):.1f})",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack([source, target, panel]))


def _write_text_preview(path: Path, decision: dict[str, Any]) -> None:
    panel = _text_panel(
        [
            f"frame={decision.get('frame_idx')}",
            f"status={decision.get('status')}",
            f"reason={decision.get('reject_reason')}",
            f"sync={decision.get('sync_best_lag_frames')} {decision.get('sync_confidence')}",
            f"bnf={decision.get('bnf_available')}",
        ],
        width=620,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def _preview_rows(rows: list[dict[str, Any]], count: int) -> dict[str, list[dict[str, Any]]]:
    def top(
        key: Callable[[dict[str, Any]], float],
        *,
        reverse: bool = False,
        predicate: Callable[[dict[str, Any]], bool] = lambda _row: True,
    ) -> list[dict[str, Any]]:
        return sorted([row for row in rows if predicate(row)], key=key, reverse=reverse)[:count]

    return {
        "best_score": top(lambda row: float(row["matching_score"])),
        "smallest_margin": top(
            lambda row: float(row["matching_score_margin"]),
            predicate=lambda row: row["matching_score_margin"] is not None,
        ),
        "near_pose_threshold": top(
            lambda row: max(
                abs(float(row["pose_delta_yaw"])),
                abs(float(row["pose_delta_pitch"])),
                abs(float(row["pose_delta_roll"])),
            ),
            reverse=True,
        ),
        "near_center_threshold": top(
            lambda row: max(abs(float(row["center_delta_x"])), abs(float(row["center_delta_y"]))),
            reverse=True,
        ),
        "near_scale_threshold": top(
            lambda row: max(
                abs(math.log(float(row["width_ratio"]))),
                abs(math.log(float(row["height_ratio"]))),
            ),
            reverse=True,
        ),
        "near_post_crop_threshold": top(
            lambda row: max(
                float(row["stable_landmark_alignment_rmse"]),
                float(row["mouth_center_delta_after_crop"]),
            ),
            reverse=True,
        ),
        "near_identity_threshold": top(
            lambda row: float(row["identity_similarity"]),
        ),
        "low_sync_confidence": top(
            lambda row: float(row["matching_score"]),
            predicate=lambda row: row["sync_confidence"] == "low",
        ),
        "idle_retained": top(
            lambda row: int(row["target_frame_idx"]),
            predicate=lambda row: bool(row["is_idle"]),
        ),
    }


def write_preview_groups(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    count: int,
) -> None:
    if count <= 0:
        return
    for group, group_rows in _preview_rows(rows, count).items():
        for index, row in enumerate(group_rows):
            _write_pair_preview(out_dir / group / f"{index:03d}.png", row)
    rejected = [decision for decision in decisions if decision.get("reject_reason")]
    for reason, _ in Counter(str(item["reject_reason"]) for item in rejected).most_common(4):
        examples = [item for item in rejected if item.get("reject_reason") == reason][:count]
        for index, decision in enumerate(examples):
            _write_text_preview(out_dir / "rejected" / reason / f"{index:03d}.png", decision)


def _validate_config(config: SilentTalkingBuildConfig) -> None:
    if config.fps != 25:
        raise ValueError("Silent-talking dataset requires fps=25")
    if config.sample_rate != 16000:
        raise ValueError("Silent-talking dataset requires sample_rate=16000")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 <= config.idle_max_ratio <= 1.0:
        raise ValueError("idle_max_ratio must be between 0 and 1")
    if not -1.0 <= config.identity.min_cosine_similarity <= 1.0:
        raise ValueError("identity.min_cosine_similarity must be between -1 and 1")
    if len(config.identity.expected_sha256) != 64:
        raise ValueError("identity.expected_sha256 must contain 64 hexadecimal characters")
    try:
        int(config.identity.expected_sha256, 16)
    except ValueError as exc:
        raise ValueError(
            "identity.expected_sha256 must contain 64 hexadecimal characters"
        ) from exc
    if config.runtime.device not in {"auto", "cuda", "cpu"}:
        raise ValueError("runtime.device must be auto, cuda, or cpu")
    if config.runtime.clip_workers < 1:
        raise ValueError("runtime.clip_workers must be >= 1")
    if config.runtime.cuda_max_inflight < 1:
        raise ValueError("runtime.cuda_max_inflight must be >= 1")
    if not config.silent_video_path.is_file():
        raise FileNotFoundError(config.silent_video_path)
    if not Path(config.wenet_onnx).is_file():
        raise FileNotFoundError(config.wenet_onnx)
    discover_talking_videos(config.talking_video_dir)


def _create_detector(config: SilentTalkingBuildConfig) -> MediaPipeFaceLandmarkerDetector:
    return MediaPipeFaceLandmarkerDetector(model_asset_path=config.landmark_model_asset_path)


def _frame_aligned_rms(audio_path: Path, frame_count: int) -> np.ndarray:
    audio = load_wav_mono_f32(audio_path)
    blocks = split_audio_blocks(audio)
    block_rms = np.sqrt(np.mean(blocks * blocks, axis=1)).astype(np.float32)
    audio_rms = np.zeros(frame_count, dtype=np.float32)
    available = min(frame_count, len(block_rms))
    audio_rms[:available] = block_rms[:available]
    return audio_rms


def _save_and_validate_dataset(
    dataset: DatasetDict,
    temporary_root: Path,
) -> dict[str, str]:
    dataset_path = temporary_root / "dataset"
    dataset.save_to_disk(dataset_path)
    from datasets import load_from_disk

    from edge_lipsync.dataset import DuixHFDataset
    from edge_lipsync.training import collate_training_batch, validate_batch_shapes

    loaded = cast(DatasetDict, load_from_disk(dataset_path))
    for split in ("train", "val"):
        wrapped = DuixHFDataset(loaded, split)
        validate_batch_shapes(collate_training_batch([wrapped[0]]))
    return {str(split): str(split_dataset._fingerprint) for split, split_dataset in loaded.items()}


def _write_failed_clip_report(path: Path, clip_id: str, exc: Exception) -> None:
    write_json_atomic(
        path,
        {
            "clip_id": clip_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )


def _clip_failure_is_fatal(exc: Exception) -> bool:
    return isinstance(exc, (IdentityRuntimeError, OnnxRuntimeError))


def _process_talking_clip(
    video: Path,
    *,
    config: SilentTalkingBuildConfig,
    work_root: Path,
    silent_observations: list[FrameObservation],
    silent_frames_dir: Path,
    split_mode: str,
    video_splits: dict[str, str],
    identity_runtime: IdentityRuntime,
    identity_sha256: str,
    wenet_runtime: WenetRuntime,
    wenet_sha256: str,
    landmark_model: Path | None,
    detector: LandmarkDetector,
) -> ClipBuildResult:
    clip_started = perf_counter()
    clip_id = video.stem
    clip_work = work_root / "normalized" / "talking" / clip_id
    timings: dict[str, float] = {}

    started = perf_counter()
    normalized = cached_normalize_talking_video(
        video,
        clip_work / "video_25fps.mkv",
        clip_work / "audio.wav",
        fps=config.fps,
        sample_rate=config.sample_rate,
    )
    normalized_video, audio_path = normalized.value
    timings["normalize_seconds"] = perf_counter() - started

    started = perf_counter()
    extracted = cached_extract_frames(
        normalized_video,
        clip_work / "frames",
        show_progress=False,
        progress_desc=f"extract {clip_id}",
    )
    frames_dir = clip_work / "frames"
    frame_count = extracted.value
    timings["frames_seconds"] = perf_counter() - started

    started = perf_counter()
    bnf = cached_bnf_windows(
        audio_path,
        work_root / "bnf" / clip_id / "windows.npy",
        wenet_runtime,
        wenet_sha256=wenet_sha256,
        sample_rate=config.sample_rate,
    )
    audio_rms = _frame_aligned_rms(audio_path, frame_count)
    timings["bnf_seconds"] = perf_counter() - started

    analysis_path = work_root / "analysis" / "talking" / clip_id / "analysis.jsonl"
    analysis_metadata = analysis_cache_metadata(
        video,
        frame_count=frame_count,
        landmark_model_path=landmark_model,
        identity_sha256=identity_sha256,
    )
    analysis_hit = _load_analysis_cache(analysis_path, analysis_metadata) is not None
    started = perf_counter()
    talking_observations = analyze_frames(
        frames_dir,
        frame_count=frame_count,
        detector=detector,
        identity_runtime=identity_runtime,
        cache_path=analysis_path,
        cache_metadata=analysis_metadata,
        is_target=True,
        show_progress=False,
    )
    timings["analysis_seconds"] = perf_counter() - started

    split_for_frame: Callable[[int], str]
    if split_mode == "video":
        def video_split_for_frame(_frame_idx: int) -> str:
            return video_splits[clip_id]

        split_for_frame = video_split_for_frame
    else:
        def time_split_for_frame(frame_idx: int) -> str:
            return single_video_split(
                frame_idx - 1,
                frame_count=frame_count,
                validation_fraction=config.validation_fraction,
            )

        split_for_frame = time_split_for_frame

    started = perf_counter()
    decision_result = build_pair_decisions(
        talking_observations=talking_observations,
        silent_observations=silent_observations,
        bnf_windows=bnf.value,
        audio_rms=audio_rms,
        config=config,
        split_for_frame=split_for_frame,
        talking_clip_id=clip_id,
    )
    attached = [
        attach_roi_images(
            row,
            silent_frames_dir=silent_frames_dir,
            talking_frames_dir=frames_dir,
        )
        for row in decision_result.rows
    ]
    timings["pairing_seconds"] = perf_counter() - started
    timings["total_seconds"] = perf_counter() - clip_started
    return ClipBuildResult(
        clip_id=clip_id,
        rows=attached,
        decisions=decision_result.decisions,
        talking_observations=talking_observations,
        cache_hits={
            "normalized_media": normalized.hit,
            "frames": extracted.hit,
            "analysis": analysis_hit,
            "bnf": bnf.hit,
        },
        timings=timings,
    )


def _cache_summary(cache_hits: list[dict[str, bool]]) -> dict[str, dict[str, int]]:
    stages = sorted({stage for item in cache_hits for stage in item})
    summary: dict[str, dict[str, int]] = {}
    for stage in stages:
        outcomes = [item[stage] for item in cache_hits if stage in item]
        summary[stage] = {
            "hits": sum(outcomes),
            "misses": sum(not outcome for outcome in outcomes),
        }
    return summary


def _timing_summary(timings: list[dict[str, float]]) -> dict[str, float]:
    stages = sorted({stage for item in timings for stage in item})
    return {
        stage: float(sum(item.get(stage, 0.0) for item in timings))
        for stage in stages
    }


def _build_snapshot_contents(
    config: SilentTalkingBuildConfig,
    temporary_root: Path,
) -> dict[str, Any]:
    build_started = perf_counter()
    work_root = Path(config.work_root) / config.persona_id
    provider_selection = resolve_onnx_providers(
        config.runtime.device,
        warn_on_fallback=config.runtime.warn_on_cpu_fallback,
    )
    run_limiter = OnnxRunLimiter(
        provider_selection,
        max_inflight=config.runtime.cuda_max_inflight,
    )
    identity_runtime, identity_provenance = create_identity_runtime(
        config.identity,
        provider_selection=provider_selection,
        run_limiter=run_limiter,
    )
    wenet_runtime = WenetRuntime(
        config.wenet_onnx,
        provider_selection=provider_selection,
        run_limiter=run_limiter,
    )
    wenet_identity = file_identity(Path(config.wenet_onnx))
    selected_provider = provider_selection.selected_providers[0]
    print(
        "runtime "
        f"requested={provider_selection.requested_device} "
        f"arcface={selected_provider} "
        f"wenet={selected_provider} "
        f"clip_workers={config.runtime.clip_workers} "
        f"cuda_max_inflight={config.runtime.cuda_max_inflight}",
        flush=True,
    )
    landmark_model = (
        Path(config.landmark_model_asset_path)
        if config.landmark_model_asset_path
        else None
    )
    silent_timings: dict[str, float] = {}
    silent_work = work_root / "normalized" / "silent"
    started = perf_counter()
    silent_normalized = cached_normalize_visual_video(
        config.silent_video_path,
        silent_work / "video_25fps.mkv",
        fps=config.fps,
    )
    silent_video = silent_normalized.value
    silent_timings["normalize_seconds"] = perf_counter() - started
    silent_frames_dir = silent_work / "frames"
    started = perf_counter()
    silent_frames = cached_extract_frames(
        silent_video,
        silent_frames_dir,
        show_progress=config.progress,
        progress_desc="extract silent",
    )
    silent_frame_count = silent_frames.value
    silent_timings["frames_seconds"] = perf_counter() - started
    silent_analysis_path = work_root / "analysis" / "silent" / "analysis.jsonl"
    silent_analysis_metadata = analysis_cache_metadata(
        config.silent_video_path,
        frame_count=silent_frame_count,
        landmark_model_path=landmark_model,
        identity_sha256=str(identity_provenance["sha256"]),
    )
    silent_analysis_hit = (
        _load_analysis_cache(silent_analysis_path, silent_analysis_metadata) is not None
    )
    silent_detector = _create_detector(config)
    started = perf_counter()
    try:
        silent_observations = analyze_frames(
            silent_frames_dir,
            frame_count=silent_frame_count,
            detector=silent_detector,
            identity_runtime=identity_runtime,
            cache_path=silent_analysis_path,
            cache_metadata=silent_analysis_metadata,
            is_target=False,
            show_progress=config.progress,
        )
    finally:
        silent_detector.close()
    silent_timings["analysis_seconds"] = perf_counter() - started
    write_json_atomic(
        temporary_root / "reports/quality/silent.json",
        {
            "status": "ready",
            "frame_count": silent_frame_count,
            "valid_analysis_count": sum(item.landmark_valid for item in silent_observations),
            "blur": {
                "source_face": _distribution([item.face_blur for item in silent_observations]),
            },
            "pose_by_mouth_openness_quantile": _pose_by_mouth_quantile(silent_observations),
            "identity": identity_provenance,
        },
    )

    talking_videos = discover_talking_videos(config.talking_video_dir)
    split_mode = "video"
    video_splits: dict[str, str] = {}
    if len(talking_videos) >= 2:
        video_splits = assign_video_splits(
            config.persona_id,
            [path.stem for path in talking_videos],
            split_salt=config.split_salt,
            validation_fraction=config.validation_fraction,
        )
    else:
        split_mode = "single_video_contiguous_fallback"

    detector_pool = ThreadLocalDetectorPool(lambda: _create_detector(config))
    try:
        clip_results, clip_failures = run_clip_workers(
            talking_videos,
            lambda video: _process_talking_clip(
                video,
                config=config,
                work_root=work_root,
                silent_observations=silent_observations,
                silent_frames_dir=silent_frames_dir,
                split_mode=split_mode,
                video_splits=video_splits,
                identity_runtime=identity_runtime,
                identity_sha256=str(identity_provenance["sha256"]),
                wenet_runtime=wenet_runtime,
                wenet_sha256=str(wenet_identity["sha256"]),
                landmark_model=landmark_model,
                detector=detector_pool.get(),
            ),
            max_workers=config.runtime.clip_workers,
            strict=config.strict,
            show_progress=config.progress,
        )
    finally:
        detector_pool.close_all()

    rows: list[dict[str, Any]] = []
    all_decision_paths: list[str] = []
    for result in clip_results:
        rows.extend(result.rows)
        decision_path = (
            temporary_root
            / "reports/quality"
            / f"{result.clip_id}_frame_decisions.parquet"
        )
        write_frame_decisions(decision_path, result.decisions)
        all_decision_paths.append(str(decision_path.relative_to(temporary_root)))
        write_json_atomic(
            temporary_root / "reports/quality" / f"{result.clip_id}.json",
            quality_report(
                clip_id=result.clip_id,
                talking_observations=result.talking_observations,
                decisions=result.decisions,
                rows=result.rows,
                identity=identity_provenance,
            ),
        )
        write_preview_groups(
            out_dir=temporary_root / "reports/previews" / result.clip_id,
            rows=result.rows,
            decisions=result.decisions,
            count=config.preview_count_per_group,
        )
    for failure in clip_failures:
        _write_failed_clip_report(
            temporary_root / "reports/quality" / f"{failure.clip_id}.json",
            failure.clip_id,
            failure.error,
        )

    dataset = build_dataset_dict(rows)
    fingerprints = _save_and_validate_dataset(dataset, temporary_root)
    cache_inputs = [
        {
            "normalized_media": silent_normalized.hit,
            "frames": silent_frames.hit,
            "analysis": silent_analysis_hit,
        },
        *(result.cache_hits for result in clip_results),
    ]
    timings = _timing_summary(
        [silent_timings, *(result.timings for result in clip_results)]
    )
    timings["total_seconds"] = perf_counter() - build_started
    return {
        "train_rows": len(dataset["train"]),
        "val_rows": len(dataset["val"]),
        "talking_clips": len(talking_videos),
        "failed_clips": [failure.clip_id for failure in clip_failures],
        "dataset_fingerprints": fingerprints,
        "split_mode": split_mode,
        "decision_paths": all_decision_paths,
        "row_count": len(rows),
        "identity": identity_provenance,
        "runtime": {
            "requested_device": provider_selection.requested_device,
            "available_providers": list(provider_selection.available_providers),
            "arcface_providers": list(provider_selection.selected_providers),
            "wenet_providers": list(provider_selection.selected_providers),
            "cpu_fallback": provider_selection.cpu_fallback,
            "fallback_reason": provider_selection.fallback_reason,
            "clip_workers": config.runtime.clip_workers,
            "cuda_max_inflight": config.runtime.cuda_max_inflight,
        },
        "cache": _cache_summary(cache_inputs),
        "timings": timings,
    }


def _publish_snapshot_atomically(temporary_root: Path, snapshot_root: Path) -> None:
    previous_root = snapshot_root.with_name(snapshot_root.name + ".previous")
    if previous_root.exists():
        shutil.rmtree(previous_root)
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    moved_previous = False
    if snapshot_root.exists():
        snapshot_root.replace(previous_root)
        moved_previous = True
    try:
        temporary_root.replace(snapshot_root)
    except Exception:
        if moved_previous and previous_root.exists():
            previous_root.replace(snapshot_root)
        raise
    if previous_root.exists():
        shutil.rmtree(previous_root)


def build_silent_talking_dataset(
    config: SilentTalkingBuildConfig,
) -> SilentTalkingBuildResult:
    _validate_config(config)
    cfg_hash = config_sha256(config)
    snapshot_root = Path(config.snapshot_root)
    temporary_root = snapshot_root.with_name(snapshot_root.name + ".building")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir(parents=True, exist_ok=True)
    contents = _build_snapshot_contents(config, temporary_root)
    input_identities = {
        "silent": file_identity(config.silent_video_path),
        "talking": [
            file_identity(path) for path in discover_talking_videos(config.talking_video_dir)
        ],
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "config_sha256": cfg_hash,
        "persona_id": config.persona_id,
        "input_identities": input_identities,
        **contents,
    }
    write_json_atomic(temporary_root / "build_metadata.json", metadata)
    write_json_atomic(temporary_root / "build_complete.json", metadata)
    _publish_snapshot_atomically(temporary_root, snapshot_root)
    return SilentTalkingBuildResult(
        snapshot_root=snapshot_root,
        train_rows=int(contents["train_rows"]),
        val_rows=int(contents["val_rows"]),
        talking_clips=int(contents["talking_clips"]),
        failed_clips=tuple(str(value) for value in contents["failed_clips"]),
        config_sha256=cfg_hash,
    )


def build_config_from_mapping(payload: dict[str, Any]) -> SilentTalkingBuildConfig:
    values = dict(payload)
    values["match"] = MatchConfig(**dict(values.pop("matching", {})))
    values["identity"] = IdentityConfig(**dict(values.pop("identity", {})))
    values["blur"] = BlurConfig(**dict(values.pop("blur", {})))
    values["runtime"] = RuntimeConfig(**dict(values.pop("runtime", {})))
    sync_values = dict(values.pop("sync", {}))
    values["sync"] = SyncConfig(**sync_values)
    post_crop = dict(values.pop("post_crop_alignment", {}))
    if post_crop:
        values["match"] = replace(
            values["match"],
            max_stable_landmark_rmse=float(
                post_crop.get(
                    "max_stable_landmark_rmse",
                    values["match"].max_stable_landmark_rmse,
                )
            ),
            max_mouth_center_delta=float(
                post_crop.get(
                    "max_mouth_center_delta",
                    values["match"].max_mouth_center_delta,
                )
            ),
        )
    return SilentTalkingBuildConfig(**values)
