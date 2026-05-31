from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import onnxruntime as ort

from edge_lipsync.preprocess import BBox

SCRFD_TARGET_SIZE = 640
SCRFD_PROB_THRESHOLD = 0.3
SCRFD_NMS_THRESHOLD = 0.45
PFPLD_SIZE = 112


def pfpld_98_to_68(landmarks_xy: np.ndarray) -> np.ndarray:
    """Convert PFPLD landmarks with the mapping from the historical Duix SDK."""
    points = np.asarray(landmarks_xy, dtype=np.float32)
    if points.shape != (98, 2):
        raise ValueError(f"Expected PFPLD landmarks [98,2], got {points.shape}")
    converted = [
        *points[0:34:2],
        *points[33:38],
        *points[42:47],
        *points[51:61],
        (points[60] + points[62]) / 2.0,
        (points[62] + points[64]) / 2.0,
        points[64],
        (points[64] + points[66]) / 2.0,
        (points[60] + points[66]) / 2.0,
        points[68],
        (points[68] + points[70]) / 2.0,
        (points[70] + points[72]) / 2.0,
        points[72],
        (points[72] + points[74]) / 2.0,
        (points[68] + points[74]) / 2.0,
        *points[76:96],
    ]
    output = np.asarray(converted, dtype=np.float32)
    if output.shape != (68, 2):
        raise AssertionError(f"Historical PFPLD conversion produced {output.shape}")
    return output


def historical_duix_roi_from_pfpld_landmarks(
    landmarks_xy: np.ndarray,
    frame_shape: tuple[int, ...],
) -> BBox:
    """Reproduce the integer ROI crop math in historical Duix PFPLD code."""
    points = np.asarray(landmarks_xy)
    if points.shape != (68, 2):
        raise ValueError(f"Expected converted PFPLD landmarks [68,2], got {points.shape}")
    points_i32 = np.trunc(points).astype(np.int32)
    whole_min = points_i32.min(axis=0)
    whole_max = points_i32.max(axis=0)
    whole_height = int(whole_max[1] - whole_min[1])
    if whole_height <= 0:
        raise ValueError("PFPLD landmarks have zero face height")
    wh = min(1.0, float(whole_max[0] - whole_min[0]) / whole_height)
    if wh <= 0.0:
        raise ValueError("PFPLD landmarks have zero face width")

    selected = np.concatenate([points_i32[1:16], points_i32[30:67]], axis=0)
    selected_min = selected.min(axis=0)
    selected_max = selected.max(axis=0)
    roi_width = int(selected_max[0] - selected_min[0])
    if roi_width <= 0:
        raise ValueError("PFPLD ROI landmarks have zero width")
    center_x = int(selected_min[0] + roi_width // 2)
    frame_height, frame_width = frame_shape[:2]
    x1 = max(0.0, center_x - roi_width * 0.5 / wh)
    x2 = min(float(frame_width - 1), center_x + roi_width * 0.5 / wh)
    y1 = max(0.0, selected_min[1] + roi_width * 0.11 / wh)
    y2 = min(float(frame_height - 1), selected_min[1] + roi_width * 1.11 / wh)
    roi: BBox = (int(x1), int(y1), int(x2), int(y2))
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError(f"Historical PFPLD ROI is empty: {roi}")
    return roi


def expand_historical_scrfd_bbox(bbox_xyxy: BBox, frame_shape: tuple[int, ...]) -> BBox:
    """Apply the asymmetric 10% SCRFD expansion used before PFPLD."""
    x1, y1, x2, y2 = bbox_xyxy
    frame_height, frame_width = frame_shape[:2]
    width = x2 - x1
    x1 = int(max(0.0, x1 - width * 0.1))
    y1 = max(0, y1)
    x2 = int(min(float(frame_width - 1), x2 + (x2 - x1) * 0.1))
    y2 = int(min(float(frame_height - 1), y2 + (y2 - y1) * 0.1))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Historical SCRFD expansion is empty: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class _ScrfdProposal:
    x: float
    y: float
    width: float
    height: float
    score: float


def _generate_scrfd_proposals(
    score_blob: np.ndarray,
    bbox_blob: np.ndarray,
    stride: int,
) -> list[_ScrfdProposal]:
    proposals: list[_ScrfdProposal] = []
    for anchor_index in range(score_blob.shape[0]):
        score = score_blob[anchor_index]
        for row, column in np.argwhere(score >= SCRFD_PROB_THRESHOLD):
            center_x = float(column * stride)
            center_y = float(row * stride)
            offset = anchor_index * 4
            dx = float(bbox_blob[offset, row, column] * stride)
            dy = float(bbox_blob[offset + 1, row, column] * stride)
            dw = float(bbox_blob[offset + 2, row, column] * stride)
            dh = float(bbox_blob[offset + 3, row, column] * stride)
            x1 = center_x - dx
            y1 = center_y - dy
            x2 = center_x + dw
            y2 = center_y + dh
            proposals.append(
                _ScrfdProposal(
                    x=x1,
                    y=y1,
                    width=x2 - x1 + 1.0,
                    height=y2 - y1 + 1.0,
                    score=float(score[row, column]),
                )
            )
    return proposals


def _proposal_iou(left: _ScrfdProposal, right: _ScrfdProposal) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union


def _nms_scrfd_proposals(proposals: list[_ScrfdProposal]) -> list[_ScrfdProposal]:
    picked: list[_ScrfdProposal] = []
    for proposal in sorted(proposals, key=lambda value: value.score, reverse=True):
        if all(_proposal_iou(proposal, existing) <= SCRFD_NMS_THRESHOLD for existing in picked):
            picked.append(proposal)
    return picked


class HistoricalDuixScrfdPfpldDetector:
    """Diagnostics-only reconstruction of the detector path in Duix-Mobile origin/20250714."""

    def __init__(
        self,
        *,
        scrfd_param: str | Path,
        scrfd_bin: str | Path,
        pfpld_onnx: str | Path,
        pfpld_channel_order: Literal["rgb", "bgr"] = "rgb",
    ) -> None:
        if pfpld_channel_order not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported PFPLD channel order: {pfpld_channel_order}")
        try:
            self._ncnn: Any = importlib.import_module("ncnn")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Historical SCRFD diagnostics require the optional ncnn package"
            ) from exc
        self._scrfd = self._ncnn.Net()
        if self._scrfd.load_param(str(scrfd_param)) != 0:
            raise RuntimeError(f"Failed to load SCRFD param: {scrfd_param}")
        if self._scrfd.load_model(str(scrfd_bin)) != 0:
            raise RuntimeError(f"Failed to load SCRFD model: {scrfd_bin}")
        self._pfpld = ort.InferenceSession(
            str(pfpld_onnx),
            providers=["CPUExecutionProvider"],
        )
        self._pfpld_channel_order = pfpld_channel_order

    def _detect_scrfd(self, frame_bgr: np.ndarray) -> tuple[BBox, float] | None:
        frame_height, frame_width = frame_bgr.shape[:2]
        scale = SCRFD_TARGET_SIZE / max(frame_width, frame_height)
        scaled_width = int(frame_width * scale)
        scaled_height = int(frame_height * scale)
        width_pad = (scaled_width + 31) // 32 * 32 - scaled_width
        height_pad = (scaled_height + 31) // 32 * 32 - scaled_height
        source = np.ascontiguousarray(frame_bgr)
        input_mat = self._ncnn.Mat.from_pixels_resize(
            source,
            self._ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            frame_width,
            frame_height,
            scaled_width,
            scaled_height,
        )
        input_pad = self._ncnn.copy_make_border(
            input_mat,
            height_pad // 2,
            height_pad - height_pad // 2,
            width_pad // 2,
            width_pad - width_pad // 2,
            self._ncnn.BorderType.BORDER_CONSTANT,
            0.0,
        )
        input_pad.substract_mean_normalize(
            [127.5, 127.5, 127.5],
            [1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0],
        )
        extractor = self._scrfd.create_extractor()
        if extractor.input("input.1", input_pad) != 0:
            raise RuntimeError("NCNN input(input.1) failed for historical SCRFD")
        proposals: list[_ScrfdProposal] = []
        for stride in (8, 16, 32):
            score_status, score_mat = extractor.extract(f"score_{stride}")
            bbox_status, bbox_mat = extractor.extract(f"bbox_{stride}")
            if score_status != 0 or bbox_status != 0:
                raise RuntimeError(f"NCNN SCRFD output extraction failed for stride={stride}")
            proposals.extend(
                _generate_scrfd_proposals(
                    np.asarray(score_mat.numpy()),
                    np.asarray(bbox_mat.numpy()),
                    stride,
                )
            )
        picked = _nms_scrfd_proposals(proposals)
        if not picked:
            return None
        face = picked[0]
        x1 = np.clip((face.x - width_pad // 2) / scale, 0.0, frame_width - 1.0)
        y1 = np.clip((face.y - height_pad // 2) / scale, 0.0, frame_height - 1.0)
        x2 = np.clip((face.x + face.width - width_pad // 2) / scale, 0.0, frame_width - 1.0)
        y2 = np.clip((face.y + face.height - height_pad // 2) / scale, 0.0, frame_height - 1.0)
        raw_bbox = (
            int(x1),
            int(y1),
            int(x1) + int(x2 - x1),
            int(y1) + int(y2 - y1),
        )
        return raw_bbox, face.score

    def _detect_pfpld(self, frame_bgr: np.ndarray, scrfd_bbox: BBox) -> tuple[BBox, np.ndarray]:
        x1, y1, x2, y2 = scrfd_bbox
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Historical SCRFD crop is empty: {scrfd_bbox}")
        resized = cv2.resize(crop, (PFPLD_SIZE, PFPLD_SIZE), interpolation=cv2.INTER_LINEAR)
        if self._pfpld_channel_order == "rgb":
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = resized.astype(np.float32).transpose(2, 0, 1)[None, :, :, :] / 255.0
        outputs = self._pfpld.run(["landms"], {"input": np.ascontiguousarray(tensor)})
        landmarks_98 = np.asarray(outputs[0], dtype=np.float32).reshape(98, 2)
        landmarks_68 = pfpld_98_to_68(landmarks_98)
        landmarks_pixels = landmarks_68 * np.asarray([x2 - x1, y2 - y1]) + np.asarray([x1, y1])
        return (
            historical_duix_roi_from_pfpld_landmarks(landmarks_pixels, frame_bgr.shape),
            landmarks_pixels,
        )

    def detect_bbox_with_debug(self, frame_bgr: np.ndarray) -> tuple[BBox | None, dict[str, Any]]:
        detected = self._detect_scrfd(frame_bgr)
        if detected is None:
            return None, {"status": "scrfd_face_not_detected"}
        raw_bbox, score = detected
        expanded_bbox = expand_historical_scrfd_bbox(raw_bbox, frame_bgr.shape)
        roi, _landmarks = self._detect_pfpld(frame_bgr, expanded_bbox)
        return roi, {
            "status": "ok",
            "scrfd_score": score,
            "scrfd_bbox_xyxy": list(raw_bbox),
            "scrfd_expanded_bbox_xyxy": list(expanded_bbox),
            "pfpld_roi_xyxy": list(roi),
        }

    def detect_bbox(self, frame_bgr: np.ndarray) -> BBox | None:
        bbox, _debug = self.detect_bbox_with_debug(frame_bgr)
        return bbox

    def close(self) -> None:
        self._scrfd.clear()
