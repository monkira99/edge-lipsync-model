from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_lipsync.hub import pull_dataset_snapshot, pull_model_checkpoint


@dataclass(frozen=True)
class ResolvedSource:
    path: Path
    provenance: dict[str, Any]


def _require_exactly_one_source(local_path: str, hf_repo: str) -> None:
    if bool(local_path) == bool(hf_repo):
        raise ValueError("Set exactly one local path or Hugging Face repo source")


def _local_source(path_value: str, *, kind: str) -> ResolvedSource:
    path = Path(path_value)
    if kind == "dataset" and not path.is_dir():
        raise FileNotFoundError(path)
    if kind == "model" and not path.is_file():
        raise FileNotFoundError(path)
    resolved = path.resolve()
    return ResolvedSource(
        path=resolved,
        provenance={
            "source": "local",
            "path": str(resolved),
        },
    )


def resolve_dataset_source(
    *,
    dataset_root: str,
    hf_repo: str = "",
    hf_revision: str = "",
    cache_dir: str = "",
) -> ResolvedSource:
    _require_exactly_one_source(dataset_root, hf_repo)
    if dataset_root:
        return _local_source(dataset_root, kind="dataset")
    artifact = pull_dataset_snapshot(hf_repo, revision=hf_revision, cache_dir=cache_dir)
    if artifact.path is None:
        raise ValueError("Downloaded Hugging Face dataset snapshot had no local path")
    return ResolvedSource(
        path=artifact.path,
        provenance={
            "source": "huggingface",
            "repo_id": artifact.repo_id,
            "requested_revision": artifact.requested_revision,
            "resolved_revision": artifact.resolved_revision,
            "url": artifact.url,
            "path": str(artifact.path),
        },
    )


def resolve_model_source(
    *,
    checkpoint: str,
    hf_repo: str = "",
    hf_revision: str = "",
    hf_filename: str = "best.pt",
    cache_dir: str = "",
) -> ResolvedSource:
    _require_exactly_one_source(checkpoint, hf_repo)
    if checkpoint:
        return _local_source(checkpoint, kind="model")
    artifact = pull_model_checkpoint(
        hf_repo,
        revision=hf_revision,
        filename=hf_filename,
        cache_dir=cache_dir,
    )
    if artifact.path is None:
        raise ValueError("Downloaded Hugging Face model checkpoint had no local path")
    return ResolvedSource(
        path=artifact.path,
        provenance={
            "source": "huggingface",
            "repo_id": artifact.repo_id,
            "requested_revision": artifact.requested_revision,
            "resolved_revision": artifact.resolved_revision,
            "filename": hf_filename,
            "url": artifact.url,
            "path": str(artifact.path),
        },
    )
