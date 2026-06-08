from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

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


def test_duix_hf_dataset_loads_sample_from_datasets_row(tmp_path: Path) -> None:
    from edge_lipsync.dataset import DuixHFDataset
    from edge_lipsync.hf_datasets import build_processed_dataset_dict

    _write_fixture_dataset(tmp_path)
    dataset_dict = build_processed_dataset_dict(tmp_path)

    ds = DuixHFDataset(dataset_dict, split="train")
    sample = ds[0]

    assert len(ds) == 1
    assert tuple(sample["face"].shape) == (6, 160, 160)
    assert tuple(sample["audio"].shape) == (20, 256)
    assert tuple(sample["target"].shape) == (3, 160, 160)
    assert sample["meta"]["clip_id"] == "clip_001"


def test_duix_hf_dataset_loads_silent_talking_roi_row() -> None:
    from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value

    from edge_lipsync.dataset import DuixHFDataset

    source = np.full((168, 168, 3), (10, 20, 30), dtype=np.uint8)
    target = np.full((168, 168, 3), (200, 210, 220), dtype=np.uint8)
    _, source_png = cv2.imencode(".png", source)
    _, target_png = cv2.imencode(".png", target)
    features = Features(
        {
            "schema_version": Value("string"),
            "persona_id": Value("string"),
            "pair_id": Value("string"),
            "talking_clip_id": Value("string"),
            "source_frame_idx": Value("int32"),
            "target_frame_idx": Value("int32"),
            "audio_idx": Value("int32"),
            "source_roi": Image(),
            "target_roi": Image(),
            "audio": Sequence(Sequence(Value("float32"), length=256), length=20),
            "source_bbox_xyxy": Sequence(Value("int32"), length=4),
            "target_bbox_xyxy": Sequence(Value("int32"), length=4),
            "sample_weight": Value("float32"),
            "flags": Sequence(Value("string")),
        }
    )
    rows = [
        {
            "schema_version": "edge_lipsync_silent_talking_pair_v1",
            "persona_id": "nora",
            "pair_id": "talk__000001__silent__000002",
            "talking_clip_id": "talk",
            "source_frame_idx": 2,
            "target_frame_idx": 1,
            "audio_idx": 0,
            "source_roi": {"bytes": source_png.tobytes(), "path": None},
            "target_roi": {"bytes": target_png.tobytes(), "path": None},
            "audio": np.zeros((20, 256), dtype=np.float32),
            "source_bbox_xyxy": [10, 20, 110, 120],
            "target_bbox_xyxy": [12, 22, 112, 122],
            "sample_weight": 1.0,
            "flags": [],
        }
    ]
    dataset = DatasetDict({"train": Dataset.from_list(rows, features=features)})

    sample = DuixHFDataset(dataset, split="train")[0]

    assert tuple(sample["face"].shape) == (6, 160, 160)
    assert tuple(sample["audio"].shape) == (20, 256)
    assert tuple(sample["target"].shape) == (3, 160, 160)
    assert sample["meta"]["pair_id"] == rows[0]["pair_id"]
    assert sample["meta"]["sample_weight"] == pytest.approx(1.0)
    assert not np.array_equal(sample["face"][:3].numpy(), sample["target"].numpy())


def test_hf_image_feature_roundtrips_embedded_png_without_path(tmp_path: Path) -> None:
    from datasets import Dataset, Features, Image, load_from_disk

    image = np.full((168, 168, 3), 120, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    dataset = Dataset.from_list(
        [{"image": {"bytes": encoded.tobytes(), "path": None}}],
        features=Features({"image": Image()}),
    )
    path = tmp_path / "image_dataset"
    dataset.save_to_disk(path)

    loaded = load_from_disk(path)
    physical = cast(
        dict[str, Any],
        loaded.cast_column("image", Image(decode=False))[0]["image"],
    )

    assert physical["path"] is None
    assert physical["bytes"].startswith(b"\x89PNG")
    assert np.asarray(loaded[0]["image"]).shape == (168, 168, 3)
