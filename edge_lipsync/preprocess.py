from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

BBox = tuple[int, int, int, int]

ROI_SOURCE_SIZE = 168
ROI_EDGE = 4
FACE_SIZE = 160
MASK_X = 5
MASK_Y = 5
MASK_W = 150
MASK_H = 145


@dataclass(frozen=True)
class FaceTrainingSample:
    face: np.ndarray
    target: np.ndarray
    roi_168_bgr: np.ndarray
    real_patch_bgr: np.ndarray
    masked_patch_bgr: np.ndarray
    bbox_xyxy: BBox


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    return ((rgb.astype(np.float32) - 127.5) / 127.5).astype(np.float32)


def validate_bbox(bbox: Sequence[int], frame_shape: tuple[int, ...]) -> BBox:
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox length: {bbox}")
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox with non-positive area: {(x1, y1, x2, y2)}")
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(f"Invalid bbox outside frame: {(x1, y1, x2, y2)} frame={(w, h)}")
    return x1, y1, x2, y2


def adjust_bbox(
    bbox: BBox,
    frame_shape: tuple[int, ...],
    dx: int = 0,
    dy: int = 0,
    scale: float = 1.0,
) -> BBox:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    nbw = max(32, int(round(bw * scale)))
    nbh = max(32, int(round(bh * scale)))
    nx1 = int(round(cx - nbw / 2.0)) + dx
    ny1 = int(round(cy - nbh / 2.0)) + dy
    nx2 = nx1 + nbw
    ny2 = ny1 + nbh
    nx1 = max(0, min(nx1, w - 2))
    ny1 = max(0, min(ny1, h - 2))
    nx2 = max(nx1 + 1, min(nx2, w))
    ny2 = max(ny1 + 1, min(ny2, h))
    return nx1, ny1, nx2, ny2


def make_face_training_sample(
    frame_bgr: np.ndarray,
    bbox_xyxy: Sequence[int],
) -> FaceTrainingSample:
    bbox = validate_bbox(bbox_xyxy, frame_bgr.shape)
    x1, y1, x2, y2 = bbox
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError(f"Invalid bbox produced empty ROI: {bbox}")

    roi_168_bgr = cv2.resize(roi, (ROI_SOURCE_SIZE, ROI_SOURCE_SIZE), interpolation=cv2.INTER_AREA)
    real_patch_bgr = roi_168_bgr[
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
    ].copy()
    masked_patch_bgr = real_patch_bgr.copy()
    cv2.rectangle(
        masked_patch_bgr,
        (MASK_X, MASK_Y),
        (MASK_X + MASK_W - 1, MASK_Y + MASK_H - 1),
        (0, 0, 0),
        -1,
    )

    target_rgb = cv2.cvtColor(real_patch_bgr, cv2.COLOR_BGR2RGB)
    masked_rgb = cv2.cvtColor(masked_patch_bgr, cv2.COLOR_BGR2RGB)
    target_norm = _normalize_rgb(target_rgb)
    masked_norm = _normalize_rgb(masked_rgb)

    face = np.concatenate([target_norm, masked_norm], axis=2).transpose(2, 0, 1)
    target = target_norm.transpose(2, 0, 1)
    return FaceTrainingSample(
        face=np.ascontiguousarray(face.astype(np.float32)),
        target=np.ascontiguousarray(target.astype(np.float32)),
        roi_168_bgr=roi_168_bgr,
        real_patch_bgr=real_patch_bgr,
        masked_patch_bgr=masked_patch_bgr,
        bbox_xyxy=bbox,
    )
