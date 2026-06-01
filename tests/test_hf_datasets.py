from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _write_processed_fixture(root: Path) -> None:
    clip = root / "clips" / "clip_001"
    frames = clip / "frames"
    frames.mkdir(parents=True)
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(frames / "000001.png"), frame)
    cv2.imwrite(str(frames / "000002.png"), frame)
    np.save(clip / "bnf.npy", np.zeros((30, 256), dtype=np.float32))
    records = [
        {
            "clip_id": "clip_001",
            "frame_idx": 1,
            "audio_idx": 1,
            "frame_path": "clips/clip_001/frames/000001.png",
            "bbox_xyxy": [80, 40, 240, 200],
            "bnf_path": "clips/clip_001/bnf.npy",
            "split": "train",
            "flags": [],
        },
        {
            "clip_id": "clip_001",
            "frame_idx": 2,
            "audio_idx": 2,
            "frame_path": "clips/clip_001/frames/000002.png",
            "bbox_xyxy": [80, 40, 240, 200],
            "bnf_path": "clips/clip_001/bnf.npy",
            "split": "val",
            "flags": ["interpolated_bbox"],
        },
    ]
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_build_processed_dataset_dict_exports_train_and_val_splits(tmp_path: Path) -> None:
    from edge_lipsync.hf_datasets import build_processed_dataset_dict

    _write_processed_fixture(tmp_path)

    dataset = build_processed_dataset_dict(tmp_path)

    assert set(dataset) == {"train", "val"}
    assert dataset["train"].num_rows == 1
    assert dataset["val"].num_rows == 1
    assert dataset["train"].features["frame"].decode is True
    assert tuple(np.asarray(dataset["train"][0]["audio"]).shape) == (20, 256)
    assert dataset["val"][0]["flags"] == ["interpolated_bbox"]


def test_push_processed_dataset_uses_datasetdict_push_to_hub(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.hf_datasets as hf_datasets

    _write_processed_fixture(tmp_path)
    calls: list[dict[str, Any]] = []

    class FakeDatasetDict:
        def push_to_hub(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            return "https://huggingface.co/datasets/owner/avatar-data/commit/abc123"

    monkeypatch.setattr(
        hf_datasets,
        "build_processed_dataset_dict",
        lambda *_args, **_kwargs: FakeDatasetDict(),
    )

    artifact = hf_datasets.push_processed_dataset(
        tmp_path,
        "owner/avatar-data",
        private=False,
    )

    assert artifact.repo_id == "owner/avatar-data"
    assert artifact.url == "https://huggingface.co/datasets/owner/avatar-data"
    assert calls == [
        {
            "repo_id": "owner/avatar-data",
            "private": False,
        }
    ]


def test_load_processed_dataset_uses_latest_dataset_without_revision(monkeypatch: Any) -> None:
    import edge_lipsync.hf_datasets as hf_datasets

    calls: list[dict[str, Any]] = []

    def fake_load_dataset(*args: Any, **kwargs: Any) -> dict[str, list[int]]:
        calls.append({"args": args, "kwargs": kwargs})
        return {"train": [1], "val": [2]}

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    dataset = hf_datasets.load_processed_dataset("owner/avatar-data", cache_dir="/cache")

    assert dataset == {"train": [1], "val": [2]}
    assert calls == [
        {
            "args": ("owner/avatar-data",),
            "kwargs": {
                "cache_dir": "/cache",
            },
        }
    ]
