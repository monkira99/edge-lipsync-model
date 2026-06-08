from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import cv2
import numpy as np

from edge_lipsync.preprocess import BBox, Point

HEAD_POSE_LANDMARK_INDICES = (1, 10, 33, 234, 263, 454)
STABLE_ALIGNMENT_LANDMARK_INDICES = (1, 33, 152, 234, 263, 454)
MOUTH_CORNER_INDICES = (61, 291)
MOUTH_VERTICAL_INDICES = (13, 14)
TRACKED_LANDMARK_INDICES = tuple(
    sorted(
        set(HEAD_POSE_LANDMARK_INDICES)
        | set(STABLE_ALIGNMENT_LANDMARK_INDICES)
        | set(MOUTH_CORNER_INDICES)
        | set(MOUTH_VERTICAL_INDICES)
    )
)

HEAD_MODEL_POINTS = np.asarray(
    [
        (0.0, 0.0, 0.0),
        (0.0, -55.0, -10.0),
        (-32.0, -28.0, -20.0),
        (-48.0, 5.0, -15.0),
        (32.0, -28.0, -20.0),
        (48.0, 5.0, -15.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True)
class NormalizedGeometry:
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class FrameObservation:
    frame_idx: int
    bbox_xyxy: BBox | None
    frame_width: int
    frame_height: int
    landmarks: dict[int, Point]
    pose: HeadPose | None
    face_blur: float
    mouth_blur: float
    mouth_open: float
    landmark_valid: bool
    bbox_continuity_valid: bool = True
    reject_reason: str = ""


@dataclass(frozen=True)
class SyncWindow:
    window_id: int
    start_frame: int
    end_frame: int
    center_frame: float
    has_speech: bool
    best_lag_frames: int
    best_correlation: float
    confidence: str


@dataclass(frozen=True)
class MatchConfig:
    max_yaw_delta: float = 5.0
    max_pitch_delta: float = 5.0
    max_roll_delta: float = 4.0
    max_center_x_delta: float = 0.05
    max_center_y_delta: float = 0.05
    min_width_ratio: float = 0.9
    max_width_ratio: float = 1.1
    min_height_ratio: float = 0.9
    max_height_ratio: float = 1.1
    max_stable_landmark_rmse: float = 0.04
    max_mouth_center_delta: float = 0.04
    pose_weight: float = 1.0
    position_weight: float = 1.0
    scale_weight: float = 1.0


@dataclass(frozen=True)
class AlignmentMetrics:
    stable_landmark_rmse: float
    mouth_center_delta: float


@dataclass(frozen=True)
class MatchResult:
    selected: FrameObservation
    matching_score: float
    valid_candidate_count: int
    second_best_score: float | None
    matching_score_margin: float | None
    pose_delta: HeadPose
    center_delta_x: float
    center_delta_y: float
    width_ratio: float
    height_ratio: float
    alignment: AlignmentMetrics


def normalized_bbox_geometry(
    bbox: BBox,
    frame_shape: tuple[int, ...],
) -> NormalizedGeometry:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return NormalizedGeometry(
        center_x=(x1 + x2) / (2.0 * width),
        center_y=(y1 + y2) / (2.0 * height),
        width=(x2 - x1) / width,
        height=(y2 - y1) / height,
    )


def mouth_openness(landmarks: Mapping[int, Point]) -> float:
    left = landmarks[61]
    right = landmarks[291]
    upper = landmarks[13]
    lower = landmarks[14]
    width = max(math.dist(left, right), 1e-6)
    return math.dist(upper, lower) / width


def rotation_matrix_to_euler(rotation: np.ndarray) -> HeadPose:
    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        pitch = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = 0.0
    return HeadPose(
        yaw=math.degrees(yaw),
        pitch=math.degrees(pitch),
        roll=math.degrees(roll),
    )


def estimate_head_pose(
    landmarks: Mapping[int, Point],
    frame_shape: tuple[int, ...],
) -> HeadPose:
    height, width = frame_shape[:2]
    image_points = np.asarray(
        [landmarks[index] for index in HEAD_POSE_LANDMARK_INDICES],
        dtype=np.float64,
    )
    focal = float(max(width, height))
    camera = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rotation_vector, _translation = cv2.solvePnP(
        HEAD_MODEL_POINTS,
        image_points,
        camera,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP failed")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return rotation_matrix_to_euler(rotation)


def laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mouth_bbox(
    landmarks: Mapping[int, Point],
    frame_shape: tuple[int, ...],
    *,
    padding_fraction: float = 0.35,
) -> BBox:
    height, width = frame_shape[:2]
    points = np.asarray(
        [landmarks[index] for index in (*MOUTH_CORNER_INDICES, *MOUTH_VERTICAL_INDICES)],
        dtype=np.float32,
    )
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    pad = max(float(x2 - x1), float(y2 - y1), 1.0) * padding_fraction
    return (
        max(0, int(math.floor(x1 - pad))),
        max(0, int(math.floor(y1 - pad))),
        min(width, int(math.ceil(x2 + pad))),
        min(height, int(math.ceil(y2 + pad))),
    )


def mark_bbox_continuity(
    observations: list[FrameObservation],
    *,
    max_center_distance: float = 0.05,
    max_scale_ratio: float = 1.15,
) -> list[FrameObservation]:
    out: list[FrameObservation] = []
    previous: FrameObservation | None = None
    for observation in observations:
        if observation.bbox_xyxy is None:
            out.append(observation)
            continue
        valid = True
        if previous is not None and previous.bbox_xyxy is not None:
            current_geometry = normalized_bbox_geometry(
                observation.bbox_xyxy,
                (observation.frame_height, observation.frame_width, 3),
            )
            previous_geometry = normalized_bbox_geometry(
                previous.bbox_xyxy,
                (previous.frame_height, previous.frame_width, 3),
            )
            center_distance = math.hypot(
                current_geometry.center_x - previous_geometry.center_x,
                current_geometry.center_y - previous_geometry.center_y,
            )
            width_ratio = current_geometry.width / previous_geometry.width
            height_ratio = current_geometry.height / previous_geometry.height
            valid = (
                center_distance <= max_center_distance
                and 1.0 / max_scale_ratio <= width_ratio <= max_scale_ratio
                and 1.0 / max_scale_ratio <= height_ratio <= max_scale_ratio
            )
        updated = replace(
            observation,
            bbox_continuity_valid=valid,
            reject_reason="" if valid else "bbox_discontinuity",
        )
        out.append(updated)
        if valid:
            previous = updated
    return out


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left.astype(np.float64) - float(left.mean())
    right_centered = right.astype(np.float64) - float(right.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def fill_missing_signal(values: np.ndarray) -> np.ndarray:
    indices = np.arange(len(values), dtype=np.float64)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float32)
    return np.interp(indices, indices[valid], values[valid]).astype(np.float32)


def best_sync_lag(
    audio: np.ndarray,
    mouth: np.ndarray,
    *,
    max_lag_frames: int,
) -> tuple[int, float]:
    if audio.shape != mouth.shape:
        raise ValueError("audio and mouth must have matching shapes")
    if len(audio) == 0:
        raise ValueError("audio and mouth signals must be non-empty")
    candidates: list[tuple[float, int]] = []
    effective_max_lag = min(max_lag_frames, max(0, len(audio) - 2))
    for lag in range(-effective_max_lag, effective_max_lag + 1):
        if lag >= 0:
            left = audio[: len(audio) - lag or None]
            right = mouth[lag:]
        else:
            left = audio[-lag:]
            right = mouth[: len(mouth) + lag]
        correlation = _pearson(left, right)
        candidates.append((correlation, lag))
    correlation, lag = max(
        candidates,
        key=lambda item: (item[0], -abs(item[1]), -item[1]),
    )
    return lag, correlation


def build_sync_windows(
    audio_rms: np.ndarray,
    mouth_open: np.ndarray,
    *,
    fps: int = 25,
    window_seconds: float = 2.0,
    stride_seconds: float = 1.0,
    max_lag_frames: int = 3,
    silence_rms_threshold: float = 0.001,
    speech_fraction_threshold: float = 0.25,
    min_correlation: float = 0.20,
) -> list[SyncWindow]:
    if audio_rms.shape != mouth_open.shape:
        raise ValueError("audio_rms and mouth_open must have matching frame counts")
    if len(audio_rms) == 0:
        return []
    window = max(1, int(round(window_seconds * fps)))
    stride = max(1, int(round(stride_seconds * fps)))
    starts = list(range(0, max(1, len(audio_rms) - window + 1), stride))
    if not starts or starts[-1] + window < len(audio_rms):
        starts.append(max(0, len(audio_rms) - window))
    windows: list[SyncWindow] = []
    for window_id, start in enumerate(sorted(set(starts))):
        end = min(len(audio_rms), start + window)
        audio_slice = audio_rms[start:end]
        mouth_slice = mouth_open[start:end]
        voiced_fraction = float(np.mean(audio_slice > silence_rms_threshold))
        has_speech = voiced_fraction >= speech_fraction_threshold
        lag, correlation = best_sync_lag(
            audio_slice,
            mouth_slice,
            max_lag_frames=max_lag_frames,
        )
        windows.append(
            SyncWindow(
                window_id=window_id,
                start_frame=start,
                end_frame=end,
                center_frame=(start + end - 1) / 2.0,
                has_speech=has_speech,
                best_lag_frames=lag,
                best_correlation=correlation,
                confidence="high" if correlation >= min_correlation else "low",
            )
        )
    return windows


def assign_sync_windows(
    *,
    frame_count: int,
    windows: list[SyncWindow],
) -> dict[int, SyncWindow]:
    if not windows:
        raise ValueError("At least one sync window is required")
    return {
        frame_idx: min(
            windows,
            key=lambda window: (abs(window.center_frame - frame_idx), window.window_id),
        )
        for frame_idx in range(frame_count)
    }


def sync_reject_reason(
    window: SyncWindow,
    *,
    min_correlation: float,
    max_abs_lag: int,
) -> str | None:
    if not window.has_speech or window.best_correlation < min_correlation:
        return None
    return "sync_lag" if abs(window.best_lag_frames) > max_abs_lag else None


def _roi_point(point: Point, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.asarray(
        [(point[0] - x1) / (x2 - x1), (point[1] - y1) / (y2 - y1)],
        dtype=np.float64,
    )


def post_crop_alignment(
    source_landmarks: Mapping[int, Point],
    source_bbox: BBox,
    target_landmarks: Mapping[int, Point],
    target_bbox: BBox,
) -> AlignmentMetrics:
    source_stable = np.stack(
        [
            _roi_point(source_landmarks[index], source_bbox)
            for index in STABLE_ALIGNMENT_LANDMARK_INDICES
        ]
    )
    target_stable = np.stack(
        [
            _roi_point(target_landmarks[index], target_bbox)
            for index in STABLE_ALIGNMENT_LANDMARK_INDICES
        ]
    )
    stable_rmse = float(np.sqrt(np.mean((source_stable - target_stable) ** 2)))
    source_mouth = np.mean(
        [_roi_point(source_landmarks[index], source_bbox) for index in MOUTH_CORNER_INDICES],
        axis=0,
    )
    target_mouth = np.mean(
        [_roi_point(target_landmarks[index], target_bbox) for index in MOUTH_CORNER_INDICES],
        axis=0,
    )
    return AlignmentMetrics(
        stable_landmark_rmse=stable_rmse,
        mouth_center_delta=float(np.linalg.norm(source_mouth - target_mouth)),
    )


def _candidate_observation_valid(observation: FrameObservation) -> bool:
    return (
        observation.landmark_valid
        and observation.bbox_continuity_valid
        and observation.bbox_xyxy is not None
        and observation.pose is not None
    )


def _ratio_log_distance(ratio: float, *, min_ratio: float, max_ratio: float) -> float:
    if ratio >= 1.0:
        denominator = math.log(max_ratio)
    else:
        denominator = abs(math.log(min_ratio))
    return abs(math.log(ratio)) / max(denominator, 1e-12)


def _match_score(
    pose_delta: HeadPose,
    center_delta_x: float,
    center_delta_y: float,
    width_ratio: float,
    height_ratio: float,
    config: MatchConfig,
) -> float:
    pose = (
        abs(pose_delta.yaw) / config.max_yaw_delta
        + abs(pose_delta.pitch) / config.max_pitch_delta
        + abs(pose_delta.roll) / config.max_roll_delta
    )
    position = (
        abs(center_delta_x) / config.max_center_x_delta
        + abs(center_delta_y) / config.max_center_y_delta
    )
    scale = _ratio_log_distance(
        width_ratio,
        min_ratio=config.min_width_ratio,
        max_ratio=config.max_width_ratio,
    ) + _ratio_log_distance(
        height_ratio,
        min_ratio=config.min_height_ratio,
        max_ratio=config.max_height_ratio,
    )
    return (
        config.pose_weight * pose
        + config.position_weight * position
        + config.scale_weight * scale
    )


def match_silent_observation(
    target: FrameObservation,
    silent_candidates: list[FrameObservation],
    config: MatchConfig,
) -> MatchResult:
    if not _candidate_observation_valid(target):
        raise ValueError(target.reject_reason or "invalid_target_observation")
    assert target.bbox_xyxy is not None
    assert target.pose is not None
    target_geometry = normalized_bbox_geometry(
        target.bbox_xyxy,
        (target.frame_height, target.frame_width, 3),
    )

    pose_geometry_candidates: list[
        tuple[FrameObservation, HeadPose, float, float, float, float]
    ] = []
    for source in silent_candidates:
        if not _candidate_observation_valid(source):
            continue
        assert source.bbox_xyxy is not None
        assert source.pose is not None
        source_geometry = normalized_bbox_geometry(
            source.bbox_xyxy,
            (source.frame_height, source.frame_width, 3),
        )
        pose_delta = HeadPose(
            yaw=target.pose.yaw - source.pose.yaw,
            pitch=target.pose.pitch - source.pose.pitch,
            roll=target.pose.roll - source.pose.roll,
        )
        center_delta_x = target_geometry.center_x - source_geometry.center_x
        center_delta_y = target_geometry.center_y - source_geometry.center_y
        width_ratio = target_geometry.width / source_geometry.width
        height_ratio = target_geometry.height / source_geometry.height
        if (
            abs(pose_delta.yaw) <= config.max_yaw_delta
            and abs(pose_delta.pitch) <= config.max_pitch_delta
            and abs(pose_delta.roll) <= config.max_roll_delta
            and abs(center_delta_x) <= config.max_center_x_delta
            and abs(center_delta_y) <= config.max_center_y_delta
            and config.min_width_ratio <= width_ratio <= config.max_width_ratio
            and config.min_height_ratio <= height_ratio <= config.max_height_ratio
        ):
            pose_geometry_candidates.append(
                (
                    source,
                    pose_delta,
                    center_delta_x,
                    center_delta_y,
                    width_ratio,
                    height_ratio,
                )
            )
    if not pose_geometry_candidates:
        raise ValueError("pose_geometry_no_match")

    scored: list[
        tuple[float, FrameObservation, HeadPose, float, float, float, float, AlignmentMetrics]
    ] = []
    for source, pose_delta, center_delta_x, center_delta_y, width_ratio, height_ratio in (
        pose_geometry_candidates
    ):
        assert source.bbox_xyxy is not None
        alignment = post_crop_alignment(
            source.landmarks,
            source.bbox_xyxy,
            target.landmarks,
            target.bbox_xyxy,
        )
        if (
            alignment.stable_landmark_rmse > config.max_stable_landmark_rmse
            or alignment.mouth_center_delta > config.max_mouth_center_delta
        ):
            continue
        score = _match_score(
            pose_delta,
            center_delta_x,
            center_delta_y,
            width_ratio,
            height_ratio,
            config,
        )
        scored.append(
            (
                score,
                source,
                pose_delta,
                center_delta_x,
                center_delta_y,
                width_ratio,
                height_ratio,
                alignment,
            )
        )
    if not scored:
        raise ValueError("post_crop_alignment_mismatch")

    scored.sort(key=lambda item: (item[0], item[1].frame_idx))
    best = scored[0]
    second_best_score = scored[1][0] if len(scored) >= 2 else None
    return MatchResult(
        selected=best[1],
        matching_score=float(best[0]),
        valid_candidate_count=len(scored),
        second_best_score=float(second_best_score) if second_best_score is not None else None,
        matching_score_margin=(
            float(second_best_score - best[0]) if second_best_score is not None else None
        ),
        pose_delta=best[2],
        center_delta_x=float(best[3]),
        center_delta_y=float(best[4]),
        width_ratio=float(best[5]),
        height_ratio=float(best[6]),
        alignment=best[7],
    )


def assign_video_splits(
    persona_id: str,
    talking_clip_ids: list[str],
    *,
    split_salt: str,
    validation_fraction: float,
) -> dict[str, str]:
    clip_ids = sorted(set(talking_clip_ids))
    if len(clip_ids) < 2:
        raise ValueError("Video-level split requires at least two talking clips")
    ranked = sorted(
        clip_ids,
        key=lambda clip_id: hashlib.sha256(
            f"{persona_id}:{split_salt}:{clip_id}".encode()
        ).digest(),
    )
    val_count = min(len(ranked) - 1, max(1, round(len(ranked) * validation_fraction)))
    val_ids = set(ranked[-val_count:])
    return {clip_id: ("val" if clip_id in val_ids else "train") for clip_id in clip_ids}


def single_video_split(
    frame_idx: int,
    *,
    frame_count: int,
    validation_fraction: float,
) -> str:
    split_at = max(1, int(math.floor(frame_count * (1.0 - validation_fraction))))
    return "val" if frame_idx >= split_at else "train"


def select_idle_frame_indices(
    *,
    idle_frame_indices: list[int],
    speech_frame_indices: list[int],
    max_ratio: float,
) -> list[int]:
    limit = int(math.floor(len(speech_frame_indices) * max_ratio))
    if limit <= 0 or not idle_frame_indices:
        return []
    if not speech_frame_indices:
        return []
    speech = np.asarray(sorted(speech_frame_indices), dtype=np.int64)
    ranked = sorted(
        set(idle_frame_indices),
        key=lambda frame_idx: (
            int(np.min(np.abs(speech - frame_idx))),
            frame_idx,
        ),
    )
    pool = sorted(ranked[: max(limit * 4, limit)])
    positions = np.linspace(0, len(pool) - 1, min(limit, len(pool)), dtype=int)
    return sorted({pool[position] for position in positions})
