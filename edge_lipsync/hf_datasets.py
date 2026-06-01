from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Array2D, Dataset, DatasetDict, Features, Image, Sequence, Value, load_dataset

from edge_lipsync.audio_features import get_bnf_window
from edge_lipsync.dataset import ManifestRecord, load_manifest

PROCESSED_DATASET_FEATURES = Features(
    {
        "clip_id": Value("string"),
        "frame_idx": Value("int32"),
        "audio_idx": Value("int32"),
        "frame": Image(decode=True),
        "bbox_xyxy": Sequence(Value("int32"), length=4),
        "audio": Array2D(shape=(20, 256), dtype="float32"),
        "flags": Sequence(Value("string")),
    }
)


@dataclass(frozen=True)
class DatasetHubArtifact:
    repo_id: str
    url: str


def _repo_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def _row_from_manifest_record(dataset_root: Path, record: ManifestRecord) -> dict[str, Any]:
    frame_path = dataset_root / record.frame_path
    bnf_path = dataset_root / record.bnf_path
    if not frame_path.is_file():
        raise FileNotFoundError(frame_path)
    if not bnf_path.is_file():
        raise FileNotFoundError(bnf_path)
    bnf = np.load(bnf_path, allow_pickle=False)
    audio = get_bnf_window(bnf, record.audio_idx).astype(np.float32)
    return {
        "clip_id": record.clip_id,
        "frame_idx": record.frame_idx,
        "audio_idx": record.audio_idx,
        "frame": str(frame_path),
        "bbox_xyxy": list(record.bbox_xyxy),
        "audio": audio,
        "flags": list(record.flags),
    }


def build_processed_dataset_dict(
    dataset_root: str | Path,
    *,
    manifest: str | Path = "manifest.jsonl",
) -> DatasetDict:
    root = Path(dataset_root)
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for record in load_manifest(manifest_path):
        records_by_split[record.split].append(_row_from_manifest_record(root, record))
    datasets = {
        split: Dataset.from_list(rows, features=PROCESSED_DATASET_FEATURES)
        for split, rows in records_by_split.items()
        if rows
    }
    if not datasets:
        raise ValueError(f"No dataset rows found in {manifest_path}")
    return DatasetDict(datasets.items())


def push_processed_dataset(
    dataset_root: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    manifest: str | Path = "manifest.jsonl",
) -> DatasetHubArtifact:
    dataset = build_processed_dataset_dict(dataset_root, manifest=manifest)
    dataset.push_to_hub(repo_id=repo_id, private=private)
    return DatasetHubArtifact(repo_id=repo_id, url=_repo_url(repo_id))


def load_processed_dataset(repo_id: str, *, cache_dir: str = "") -> Any:
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(repo_id, **kwargs)
