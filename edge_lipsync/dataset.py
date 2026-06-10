from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset as TorchDataset

from edge_lipsync.audio_features import get_bnf_window
from edge_lipsync.preprocess import make_face_training_sample, make_face_training_sample_from_rois

SILENT_TALKING_SCHEMA_VERSION_V1 = "edge_lipsync_silent_talking_pair_v1"
SILENT_TALKING_SCHEMA_VERSION = "edge_lipsync_silent_talking_pair_v2"
SILENT_TALKING_SCHEMA_VERSIONS = {
    SILENT_TALKING_SCHEMA_VERSION_V1,
    SILENT_TALKING_SCHEMA_VERSION,
}


def _require_relative_path(value: object, field: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a relative dataset path: {value}")
    return path.as_posix()


@dataclass(frozen=True)
class ManifestRecord:
    clip_id: str
    frame_idx: int
    audio_idx: int
    frame_path: str
    bbox_xyxy: tuple[int, int, int, int]
    bnf_path: str
    split: str
    flags: tuple[str, ...]

    @staticmethod
    def from_json(payload: dict[str, Any]) -> ManifestRecord:
        bbox = payload["bbox_xyxy"]
        if len(bbox) != 4:
            raise ValueError(f"bbox_xyxy must have 4 values: {bbox}")
        split = str(payload["split"])
        if split not in {"train", "val"}:
            raise ValueError(f"split must be train or val: {split}")
        x1, y1, x2, y2 = bbox
        return ManifestRecord(
            clip_id=str(payload["clip_id"]),
            frame_idx=int(payload["frame_idx"]),
            audio_idx=int(payload["audio_idx"]),
            frame_path=_require_relative_path(payload["frame_path"], "frame_path"),
            bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            bnf_path=_require_relative_path(payload["bnf_path"], "bnf_path"),
            split=split,
            flags=tuple(str(value) for value in payload.get("flags", [])),
        )


def load_manifest(path: str | Path, split: str | None = None) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = ManifestRecord.from_json(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid manifest line {line_number} in {path}: {exc}") from exc
            if split is None or record.split == split:
                records.append(record)
    return records


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DuixManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset_root: str | Path, manifest_path: str | Path, split: str) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = self.dataset_root / self.manifest_path
        self.records = load_manifest(self.manifest_path, split=split)
        if not self.records:
            raise ValueError(f"No records for split={split!r} in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frame_path = self.dataset_root / record.frame_path
        bnf_path = self.dataset_root / record.bnf_path
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_path)
        if not bnf_path.exists():
            raise FileNotFoundError(bnf_path)
        bnf = np.load(bnf_path, allow_pickle=False)
        face_sample = make_face_training_sample(frame, record.bbox_xyxy)
        audio = get_bnf_window(bnf, record.audio_idx)
        return {
            "face": torch.from_numpy(face_sample.face),
            "audio": torch.from_numpy(audio),
            "target": torch.from_numpy(face_sample.target),
            "meta": {
                "clip_id": record.clip_id,
                "frame_idx": record.frame_idx,
                "audio_idx": record.audio_idx,
                "frame_path": record.frame_path,
                "bbox_xyxy": record.bbox_xyxy,
                "flags": record.flags,
            },
        }


def _hf_frame_to_bgr(frame: Any) -> np.ndarray:
    if hasattr(frame, "convert"):
        rgb = np.asarray(frame.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if isinstance(frame, dict):
        path = frame.get("path")
        if isinstance(path, str) and path:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            return image
        data = frame.get("bytes")
        if isinstance(data, bytes):
            encoded = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Cannot decode image bytes from Hugging Face dataset row")
            return image
    if isinstance(frame, np.ndarray):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB frame array, got {frame.shape}")
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported Hugging Face frame value: {type(frame)!r}")


def _silent_talking_hf_sample(row: dict[str, Any]) -> dict[str, Any]:
    source_roi = _hf_frame_to_bgr(row["source_roi"])
    target_roi = _hf_frame_to_bgr(row["target_roi"])
    audio = np.asarray(row["audio"], dtype=np.float32)
    if audio.shape != (20, 256):
        raise ValueError(f"Invalid audio shape={audio.shape}, expected=(20, 256)")
    sample = make_face_training_sample_from_rois(source_roi, target_roi)
    meta = {
        "schema_version": str(row["schema_version"]),
        "persona_id": str(row["persona_id"]),
        "pair_id": str(row["pair_id"]),
        "clip_id": str(row["talking_clip_id"]),
        "talking_clip_id": str(row["talking_clip_id"]),
        "source_frame_idx": int(row["source_frame_idx"]),
        "target_frame_idx": int(row["target_frame_idx"]),
        "frame_idx": int(row["target_frame_idx"]),
        "audio_idx": int(row["audio_idx"]),
        "source_bbox_xyxy": tuple(int(v) for v in row["source_bbox_xyxy"]),
        "target_bbox_xyxy": tuple(int(v) for v in row["target_bbox_xyxy"]),
        "sample_weight": float(row["sample_weight"]),
        "flags": tuple(str(value) for value in row.get("flags", [])),
    }
    for key in (
        "is_idle",
        "sync_best_lag_frames",
        "sync_correlation",
        "sync_confidence",
        "pose_delta_yaw",
        "pose_delta_pitch",
        "pose_delta_roll",
        "center_delta_x",
        "center_delta_y",
        "width_ratio",
        "height_ratio",
        "stable_landmark_alignment_rmse",
        "mouth_center_delta_after_crop",
        "identity_similarity",
        "matching_score",
        "valid_silent_candidate_count",
        "second_best_matching_score",
        "matching_score_margin",
        "source_face_blur",
        "target_face_blur",
        "target_mouth_blur",
    ):
        if key in row:
            meta[key] = row[key]
    return {
        "face": torch.from_numpy(sample.face),
        "audio": torch.from_numpy(audio),
        "target": torch.from_numpy(sample.target),
        "meta": meta,
    }


class DuixHFDataset(TorchDataset[dict[str, Any]]):
    def __init__(self, dataset: Any, split: str) -> None:
        if hasattr(dataset, "keys") and split in dataset:
            dataset = dataset[split]
        self.dataset = dataset
        self.split = split
        if len(self.dataset) == 0:
            raise ValueError(f"No records for split={split!r} in Hugging Face dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        if row.get("schema_version") in SILENT_TALKING_SCHEMA_VERSIONS:
            return _silent_talking_hf_sample(row)
        bbox = tuple(int(value) for value in row["bbox_xyxy"])
        if len(bbox) != 4:
            raise ValueError(f"bbox_xyxy must have 4 values: {bbox}")
        frame = _hf_frame_to_bgr(row["frame"])
        audio = np.asarray(row["audio"], dtype=np.float32)
        if audio.shape != (20, 256):
            raise ValueError(f"Invalid audio shape={audio.shape}, expected=(20, 256)")
        face_sample = make_face_training_sample(frame, bbox)
        return {
            "face": torch.from_numpy(face_sample.face),
            "audio": torch.from_numpy(audio),
            "target": torch.from_numpy(face_sample.target),
            "meta": {
                "clip_id": str(row["clip_id"]),
                "frame_idx": int(row["frame_idx"]),
                "audio_idx": int(row["audio_idx"]),
                "bbox_xyxy": bbox,
                "flags": tuple(str(value) for value in row.get("flags", [])),
            },
        }
