from __future__ import annotations

from pathlib import Path

import pytest


def test_resolve_dataset_source_rejects_local_and_hub_together(tmp_path: Path) -> None:
    from edge_lipsync.sources import resolve_dataset_source

    with pytest.raises(ValueError, match="exactly one"):
        resolve_dataset_source(
            dataset_root=str(tmp_path),
            hf_repo="owner/avatar-data",
            hf_revision="data-v1",
        )


def test_resolve_dataset_source_rejects_unpinned_hub_revision() -> None:
    from edge_lipsync.sources import resolve_dataset_source

    with pytest.raises(ValueError, match="revision"):
        resolve_dataset_source(dataset_root="", hf_repo="owner/avatar-data")


def test_resolve_dataset_source_uses_local_directory(tmp_path: Path) -> None:
    from edge_lipsync.sources import resolve_dataset_source

    result = resolve_dataset_source(dataset_root=str(tmp_path))

    assert result.path == tmp_path.resolve()
    assert result.provenance == {
        "source": "local",
        "path": str(tmp_path.resolve()),
    }


def test_resolve_dataset_source_uses_hub_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.sources as sources
    from edge_lipsync.hub import HubArtifact

    cached = tmp_path / "dataset"
    cached.mkdir()

    def fake_pull(repo_id: str, *, revision: str, cache_dir: str = "") -> HubArtifact:
        assert repo_id == "owner/avatar-data"
        assert revision == "data-v1"
        assert cache_dir == "/cache"
        return HubArtifact(
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision="data-sha",
            path=cached,
            url="https://huggingface.co/datasets/owner/avatar-data/tree/data-sha",
        )

    monkeypatch.setattr(sources, "pull_dataset_snapshot", fake_pull)

    result = sources.resolve_dataset_source(
        dataset_root="",
        hf_repo="owner/avatar-data",
        hf_revision="data-v1",
        cache_dir="/cache",
    )

    assert result.path == cached
    assert result.provenance == {
        "source": "huggingface",
        "repo_id": "owner/avatar-data",
        "requested_revision": "data-v1",
        "resolved_revision": "data-sha",
        "url": "https://huggingface.co/datasets/owner/avatar-data/tree/data-sha",
        "path": str(cached),
    }


def test_resolve_model_source_uses_local_checkpoint(tmp_path: Path) -> None:
    from edge_lipsync.sources import resolve_model_source

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")

    result = resolve_model_source(checkpoint=str(checkpoint))

    assert result.path == checkpoint.resolve()
    assert result.provenance == {
        "source": "local",
        "path": str(checkpoint.resolve()),
    }


def test_resolve_model_source_uses_hub_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.sources as sources
    from edge_lipsync.hub import HubArtifact

    cached = tmp_path / "best.pt"
    cached.write_bytes(b"checkpoint")

    def fake_pull(
        repo_id: str,
        *,
        revision: str,
        filename: str = "best.pt",
        cache_dir: str = "",
    ) -> HubArtifact:
        assert repo_id == "owner/avatar-model"
        assert revision == "model-v1"
        assert filename == "final.pt"
        assert cache_dir == "/cache"
        return HubArtifact(
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision="model-sha",
            path=cached,
            url="https://huggingface.co/owner/avatar-model/tree/model-sha",
        )

    monkeypatch.setattr(sources, "pull_model_checkpoint", fake_pull)

    result = sources.resolve_model_source(
        checkpoint="",
        hf_repo="owner/avatar-model",
        hf_revision="model-v1",
        hf_filename="final.pt",
        cache_dir="/cache",
    )

    assert result.path == cached
    assert result.provenance["source"] == "huggingface"
    assert result.provenance["repo_id"] == "owner/avatar-model"
    assert result.provenance["resolved_revision"] == "model-sha"
