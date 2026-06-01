from __future__ import annotations

from pathlib import Path

import pytest


def test_resolve_dataset_source_rejects_local_and_hub_together(tmp_path: Path) -> None:
    from edge_lipsync.sources import resolve_dataset_source

    with pytest.raises(ValueError, match="datasets.load_dataset"):
        resolve_dataset_source(
            dataset_root=str(tmp_path),
            hf_repo="owner/avatar-data",
        )


def test_resolve_dataset_source_rejects_hub_dataset_snapshot() -> None:
    from edge_lipsync.sources import resolve_dataset_source

    with pytest.raises(ValueError, match="datasets.load_dataset"):
        resolve_dataset_source(dataset_root="", hf_repo="owner/avatar-data")


def test_resolve_dataset_source_uses_local_directory(tmp_path: Path) -> None:
    from edge_lipsync.sources import resolve_dataset_source

    result = resolve_dataset_source(dataset_root=str(tmp_path))

    assert result.path == tmp_path.resolve()
    assert result.provenance == {
        "source": "local",
        "path": str(tmp_path.resolve()),
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
        filename: str = "best.pt",
        cache_dir: str = "",
    ) -> HubArtifact:
        assert repo_id == "owner/avatar-model"
        assert filename == "final.pt"
        assert cache_dir == "/cache"
        return HubArtifact(
            repo_id=repo_id,
            requested_ref="",
            resolved_ref="model-sha",
            path=cached,
            url="https://huggingface.co/owner/avatar-model/tree/model-sha",
        )

    monkeypatch.setattr(sources, "pull_model_checkpoint", fake_pull)

    result = sources.resolve_model_source(
        checkpoint="",
        hf_repo="owner/avatar-model",
        hf_filename="final.pt",
        cache_dir="/cache",
    )

    assert result.path == cached
    assert result.provenance["source"] == "huggingface"
    assert result.provenance["repo_id"] == "owner/avatar-model"
    assert result.provenance["resolved_ref"] == "model-sha"
