from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def _write_fixture_dataset(root: Path, *, precomputed_windows: bool = False) -> Path:
    clip = root / "clips" / "clip_001"
    frames = clip / "frames"
    frames.mkdir(parents=True)
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(frames / "000001.jpg"), frame)
    shape = (30, 20, 256) if precomputed_windows else (30, 256)
    np.save(clip / "bnf.npy", np.zeros(shape, dtype=np.float32))
    manifest = root / "manifest.jsonl"
    record = {
        "clip_id": "clip_001",
        "frame_idx": 1,
        "audio_idx": 1,
        "frame_path": "clips/clip_001/frames/000001.jpg",
        "bbox_xyxy": [80, 40, 240, 200],
        "bnf_path": "clips/clip_001/bnf.npy",
        "split": "train",
        "flags": [],
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


@pytest.mark.parametrize("precomputed_windows", [False, True])
def test_duix_manifest_dataset_loads_sample(
    tmp_path: Path,
    precomputed_windows: bool,
) -> None:
    from edge_lipsync.dataset import DuixManifestDataset

    manifest = _write_fixture_dataset(tmp_path, precomputed_windows=precomputed_windows)
    ds = DuixManifestDataset(tmp_path, manifest, split="train")
    sample = ds[0]

    assert len(ds) == 1
    assert tuple(sample["face"].shape) == (6, 160, 160)
    assert tuple(sample["audio"].shape) == (20, 256)
    assert tuple(sample["target"].shape) == (3, 160, 160)
    assert sample["meta"]["clip_id"] == "clip_001"


def test_manifest_sha256_is_stable(tmp_path: Path) -> None:
    from edge_lipsync.dataset import manifest_sha256

    manifest = _write_fixture_dataset(tmp_path)

    assert manifest_sha256(manifest) == manifest_sha256(manifest)
    assert len(manifest_sha256(manifest)) == 64


def test_manifest_record_rejects_absolute_asset_paths() -> None:
    from edge_lipsync.dataset import ManifestRecord

    with pytest.raises(ValueError, match="relative"):
        ManifestRecord.from_json(
            {
                "clip_id": "clip_001",
                "frame_idx": 1,
                "audio_idx": 1,
                "frame_path": "/tmp/frame.jpg",
                "bbox_xyxy": [80, 40, 240, 200],
                "bnf_path": "clips/clip_001/bnf.npy",
                "split": "train",
                "flags": [],
            }
        )


def test_dataset_rejects_empty_split(tmp_path: Path) -> None:
    from edge_lipsync.dataset import DuixManifestDataset

    manifest = _write_fixture_dataset(tmp_path)

    with pytest.raises(ValueError, match="No records"):
        DuixManifestDataset(tmp_path, manifest, split="val")
