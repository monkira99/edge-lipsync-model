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

from edge_lipsync.audio_features import get_bnf_window
from edge_lipsync.preprocess import make_face_training_sample


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
        return ManifestRecord(
            clip_id=str(payload["clip_id"]),
            frame_idx=int(payload["frame_idx"]),
            audio_idx=int(payload["audio_idx"]),
            frame_path=_require_relative_path(payload["frame_path"], "frame_path"),
            bbox_xyxy=tuple(int(value) for value in bbox),
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
