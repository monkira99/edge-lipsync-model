#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.landmarks import MediaPipeFaceLandmarkerDetector  # noqa: E402
from edge_lipsync.preprocess import (  # noqa: E402
    DUIX_ROI_CENTER_X_OFFSET,
    DUIX_ROI_CHEEK_WIDTH_SCALE,
    DUIX_ROI_TOP_FROM_EYE_TO_CHIN,
    FACE_MESH_CHIN,
    FACE_MESH_LEFT_CHEEK,
    FACE_MESH_LEFT_EYE_OUTER,
    FACE_MESH_RIGHT_CHEEK,
    FACE_MESH_RIGHT_EYE_OUTER,
)


def _load_oracle_bboxes(path: Path, frame_count: int) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected bbox oracle object, got {type(payload).__name__}")
    rows = [
        [int(values[0]), int(values[2]), int(values[1]), int(values[3])]
        for _key, values in sorted(payload.items(), key=lambda item: int(item[0]))
    ]
    if len(rows) < frame_count:
        raise ValueError(f"Bbox oracle has {len(rows)} rows, expected at least {frame_count}")
    return np.asarray(rows[:frame_count], dtype=np.int32)


def _detect_landmark_features(frames_dir: Path, landmark_model: Path) -> np.ndarray:
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise ValueError(f"No PNG frames in {frames_dir}")
    detector = MediaPipeFaceLandmarkerDetector(model_asset_path=str(landmark_model))
    features: list[tuple[float, float, float, float]] = []
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot decode frame: {frame_path}")
            landmarks = detector.detect_landmarks(frame)
            if landmarks is None:
                raise RuntimeError(f"No landmarks detected: {frame_path}")
            left_cheek = landmarks[FACE_MESH_LEFT_CHEEK]
            right_cheek = landmarks[FACE_MESH_RIGHT_CHEEK]
            left_eye = landmarks[FACE_MESH_LEFT_EYE_OUTER]
            right_eye = landmarks[FACE_MESH_RIGHT_EYE_OUTER]
            chin = landmarks[FACE_MESH_CHIN]
            features.append(
                (
                    (left_cheek[0] + right_cheek[0]) / 2.0,
                    (left_eye[1] + right_eye[1]) / 2.0,
                    chin[1],
                    float(
                        np.hypot(
                            left_cheek[0] - right_cheek[0],
                            left_cheek[1] - right_cheek[1],
                        )
                    ),
                )
            )
    finally:
        detector.close()
    return np.asarray(features, dtype=np.float64)


def _smooth_bboxes(boxes: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return boxes
    smoothed = np.empty_like(boxes)
    for index in range(len(boxes)):
        start = max(0, index - radius)
        stop = min(len(boxes), index + radius + 1)
        smoothed[index] = np.rint(boxes[start:stop].mean(axis=0)).astype(np.int32)
    return smoothed


def _boxes_for(
    features: np.ndarray,
    *,
    top: float,
    scale: float,
    xoff: float,
    yoff: float,
    radius: int,
) -> np.ndarray:
    side = np.rint(features[:, 3] * scale).astype(np.int32)
    x1 = np.rint(features[:, 0] + xoff - side / 2.0).astype(np.int32)
    y1 = np.rint(features[:, 1] + (features[:, 2] - features[:, 1]) * top + yoff).astype(
        np.int32
    )
    return _smooth_bboxes(np.stack([x1, y1, x1 + side, y1 + side], axis=1), radius)


def _bbox_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    x1 = np.maximum(reference[:, 0], candidate[:, 0])
    y1 = np.maximum(reference[:, 1], candidate[:, 1])
    x2 = np.minimum(reference[:, 2], candidate[:, 2])
    y2 = np.minimum(reference[:, 3], candidate[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    reference_area = (reference[:, 2] - reference[:, 0]) * (reference[:, 3] - reference[:, 1])
    candidate_area = (candidate[:, 2] - candidate[:, 0]) * (candidate[:, 3] - candidate[:, 1])
    iou = intersection / (reference_area + candidate_area - intersection)
    reference_center = np.stack(
        [(reference[:, 0] + reference[:, 2]) / 2.0, (reference[:, 1] + reference[:, 3]) / 2.0],
        axis=1,
    )
    candidate_center = np.stack(
        [(candidate[:, 0] + candidate[:, 2]) / 2.0, (candidate[:, 1] + candidate[:, 3]) / 2.0],
        axis=1,
    )
    drift = np.linalg.norm(reference_center - candidate_center, axis=1)
    return {
        "count": len(reference),
        "iou_mean": float(iou.mean()),
        "iou_p95": float(np.percentile(iou, 95)),
        "iou_pass_fraction_at_0_995": float((iou >= 0.995).mean()),
        "center_drift_mean": float(drift.mean()),
        "center_drift_p95": float(np.percentile(drift, 95)),
        "center_pass_fraction_at_1px": float((drift <= 1.0).mean()),
    }


def _mean_key(row: dict[str, Any]) -> tuple[float, float]:
    return float(row["iou_mean"]), float(row["center_pass_fraction_at_1px"])


def _gate_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["iou_pass_fraction_at_0_995"]),
        float(row["center_pass_fraction_at_1px"]),
        float(row["iou_mean"]),
    )


def _ranked_search(
    features: np.ndarray,
    oracle: np.ndarray,
    values: Iterable[tuple[float, float, float, float, int]],
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    best_mean: dict[str, Any] | None = None
    best_gate: dict[str, Any] | None = None
    count = 0
    for top, scale, xoff, yoff, radius in values:
        row: dict[str, Any] = {
            "top": float(top),
            "scale": float(scale),
            "xoff": float(xoff),
            "yoff": float(yoff),
            "radius": int(radius),
            **_bbox_metrics(
                oracle,
                _boxes_for(
                    features,
                    top=float(top),
                    scale=float(scale),
                    xoff=float(xoff),
                    yoff=float(yoff),
                    radius=int(radius),
                ),
            ),
        }
        count += 1
        if best_mean is None or _mean_key(row) > _mean_key(best_mean):
            best_mean = row
        if best_gate is None or _gate_key(row) > _gate_key(best_gate):
            best_gate = row
    if best_mean is None or best_gate is None:
        raise ValueError("ROI calibration search space was empty")
    return count, best_mean, best_gate


def _temporal_lag_sweep(oracle: np.ndarray, candidate: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lag in range(-5, 6):
        if lag < 0:
            reference, shifted = oracle[-lag:], candidate[:lag]
        elif lag > 0:
            reference, shifted = oracle[:-lag], candidate[lag:]
        else:
            reference, shifted = oracle, candidate
        rows.append({"lag": lag, **_bbox_metrics(reference, shifted)})
    return rows


def run_sweep(
    *,
    frames_dir: Path,
    landmark_model: Path,
    oracle_bbox_json: Path,
    output: Path,
) -> dict[str, Any]:
    features = _detect_landmark_features(frames_dir, landmark_model)
    oracle = _load_oracle_bboxes(oracle_bbox_json, len(features))
    current_args = {
        "top": DUIX_ROI_TOP_FROM_EYE_TO_CHIN,
        "scale": DUIX_ROI_CHEEK_WIDTH_SCALE,
        "xoff": DUIX_ROI_CENTER_X_OFFSET,
        "yoff": 0.0,
        "radius": 1,
    }
    current_boxes = _boxes_for(features, **current_args)
    coarse_values = product(
        np.arange(0.04, 0.1601, 0.004),
        np.arange(1.02, 1.1001, 0.004),
        np.arange(-5, 3.01, 1),
        np.arange(-5, 5.01, 1),
        range(4),
    )
    coarse_count, coarse_mean, coarse_gate = _ranked_search(features, oracle, coarse_values)
    fine_values = product(
        np.arange(float(coarse_gate["top"]) - 0.006, float(coarse_gate["top"]) + 0.0061, 0.001),
        np.arange(
            float(coarse_gate["scale"]) - 0.006,
            float(coarse_gate["scale"]) + 0.0061,
            0.001,
        ),
        np.arange(float(coarse_gate["xoff"]) - 1.0, float(coarse_gate["xoff"]) + 1.01, 0.5),
        np.arange(float(coarse_gate["yoff"]) - 1.0, float(coarse_gate["yoff"]) + 1.01, 0.5),
        range(max(0, int(coarse_gate["radius"]) - 1), min(4, int(coarse_gate["radius"]) + 1) + 1),
    )
    fine_count, fine_mean, fine_gate = _ranked_search(features, oracle, fine_values)
    lag_rows = _temporal_lag_sweep(oracle, current_boxes)
    payload = {
        "kind": "landmark_roi_calibration_sweep",
        "oracle_usage": "diagnostics_only_not_manifest_source",
        "current": {**current_args, **_bbox_metrics(oracle, current_boxes)},
        "coarse_count": coarse_count,
        "coarse_best_mean_iou": coarse_mean,
        "coarse_best_gate_fraction": coarse_gate,
        "fine_count": fine_count,
        "fine_best_mean_iou": fine_mean,
        "fine_best_gate_fraction": fine_gate,
        "temporal_lag_sweep": lag_rows,
        "temporal_lag_best_mean_iou": max(lag_rows, key=_mean_key),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep landmark-only Emma ROI calibration against an oracle for diagnostics."
    )
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--landmark-model", type=Path, required=True)
    parser.add_argument("--oracle-bbox-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_sweep(
        frames_dir=args.frames_dir,
        landmark_model=args.landmark_model,
        oracle_bbox_json=args.oracle_bbox_json,
        output=args.output,
    )
    print(f"output={args.output.resolve()}")
    print(f"current={json.dumps(payload['current'])}")
    print(f"fine_best_gate_fraction={json.dumps(payload['fine_best_gate_fraction'])}")


if __name__ == "__main__":
    main()
