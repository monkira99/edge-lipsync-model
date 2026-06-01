from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class _Commit:
    oid: str


@dataclass(frozen=True)
class _Info:
    sha: str


class _FakeApi:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.large_uploads: list[dict[str, Any]] = []

    def create_repo(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def upload_folder(self, **kwargs: Any) -> _Commit:
        self.uploads.append(kwargs)
        repo_type = kwargs.get("repo_type")
        return _Commit("dataset-commit" if repo_type == "dataset" else "model-commit")

    def upload_large_folder(self, **kwargs: Any) -> None:
        self.large_uploads.append(kwargs)

    def dataset_info(self, **_kwargs: Any) -> _Info:
        return _Info("dataset-sha")

    def model_info(self, **_kwargs: Any) -> _Info:
        return _Info("model-sha")


def _write_dataset_root(root: Path) -> Path:
    dataset_root = root / "dataset"
    clip = dataset_root / "clips" / "clip_001"
    frames = clip / "frames"
    frames.mkdir(parents=True)
    (dataset_root / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (dataset_root / "splits.json").write_text("{}\n", encoding="utf-8")
    (dataset_root / "build_summary.json").write_text("{}\n", encoding="utf-8")
    (frames / "000001.png").write_bytes(b"png")
    (clip / "bnf.npy").write_bytes(b"npy")
    (clip / "bboxes.json").write_text("{}\n", encoding="utf-8")
    (clip / "quality.json").write_text("{}\n", encoding="utf-8")
    (clip / "audio.wav").write_bytes(b"excluded")
    (clip / "video_25fps.mkv").write_bytes(b"excluded")
    return dataset_root


def _write_run_dir(root: Path) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    for filename in ("best.pt", "final.pt", "metrics.json", "metrics.csv", "run_metadata.json"):
        (run_dir / filename).write_text(filename, encoding="utf-8")
    (run_dir / "README.md").write_text("# Model\n", encoding="utf-8")
    (run_dir / "step_0000100.pt").write_text("excluded", encoding="utf-8")
    return run_dir


def _write_model_assets_root(root: Path) -> Path:
    models_root = root / "models"
    for relative in (
        "duix_detector/pfpld_robust_sim_bs1_8003.onnx",
        "duix_detector/scrfd_500m_kps-opt2.bin",
        "duix_detector/scrfd_500m_kps-opt2.param",
        "emma/dh_model.bin",
        "emma/dh_model.param",
        "emma/weight_168u.bin",
        "mediapipe/face_landmarker.task",
        "wenet/wenet.onnx",
    ):
        path = models_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    return models_root


def test_push_dataset_snapshot_uses_large_folder_upload_for_processed_artifacts(
    tmp_path: Path,
) -> None:
    from edge_lipsync.hub import DATASET_UPLOAD_PATTERNS, push_dataset_snapshot

    api = _FakeApi()
    dataset_root = _write_dataset_root(tmp_path)

    result = push_dataset_snapshot(dataset_root, "owner/avatar-dataset", api=api)

    assert result.resolved_revision == "dataset-sha"
    assert api.created == [
        {
            "repo_id": "owner/avatar-dataset",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.uploads == []
    assert api.large_uploads[0]["allow_patterns"] == DATASET_UPLOAD_PATTERNS
    assert api.large_uploads[0]["repo_type"] == "dataset"
    assert api.large_uploads[0]["folder_path"] == str(dataset_root)


def test_push_dataset_snapshot_validates_required_files_before_api_call(tmp_path: Path) -> None:
    from edge_lipsync.hub import push_dataset_snapshot

    api = _FakeApi()

    with pytest.raises(FileNotFoundError, match="manifest.jsonl"):
        push_dataset_snapshot(tmp_path, "owner/avatar-dataset", api=api)

    assert api.created == []


def test_pull_dataset_snapshot_requires_revision() -> None:
    from edge_lipsync.hub import pull_dataset_snapshot

    with pytest.raises(ValueError, match="revision"):
        pull_dataset_snapshot("owner/avatar-dataset", revision="")


def test_pull_dataset_snapshot_reports_cached_path_and_resolved_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.hub as hub

    cached = tmp_path / "snapshot"
    cached.mkdir()
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)

    result = hub.pull_dataset_snapshot(
        "owner/avatar-dataset",
        revision="dataset-v1",
        api=_FakeApi(),
    )

    assert result.path == cached
    assert result.requested_revision == "dataset-v1"
    assert result.resolved_revision == "dataset-sha"
    assert calls == [
        {
            "repo_id": "owner/avatar-dataset",
            "repo_type": "dataset",
            "revision": "dataset-v1",
        }
    ]


def test_push_model_artifacts_uses_model_allowlist(tmp_path: Path) -> None:
    from edge_lipsync.hub import MODEL_UPLOAD_PATTERNS, push_model_artifacts

    api = _FakeApi()
    run_dir = _write_run_dir(tmp_path)

    result = push_model_artifacts(run_dir, "owner/avatar-model", api=api)

    assert result.resolved_revision == "model-commit"
    assert api.created == [
        {
            "repo_id": "owner/avatar-model",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.uploads[0]["allow_patterns"] == MODEL_UPLOAD_PATTERNS
    assert "repo_type" not in api.uploads[0]


def test_push_model_assets_uploads_assets_allowlist(tmp_path: Path) -> None:
    from edge_lipsync.hub import MODEL_ASSET_UPLOAD_PATTERNS, push_model_assets

    api = _FakeApi()
    models_root = _write_model_assets_root(tmp_path)

    result = push_model_assets(models_root, "owner/edge-lipsync-model-assets", api=api)

    assert result.resolved_revision == "model-commit"
    assert api.created == [
        {
            "repo_id": "owner/edge-lipsync-model-assets",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.uploads[0]["folder_path"] == str(models_root)
    assert api.uploads[0]["allow_patterns"] == MODEL_ASSET_UPLOAD_PATTERNS
    assert "repo_type" not in api.uploads[0]


def test_push_model_assets_validates_required_files_before_api_call(tmp_path: Path) -> None:
    from edge_lipsync.hub import push_model_assets

    api = _FakeApi()
    models_root = tmp_path / "models"
    models_root.mkdir()

    with pytest.raises(FileNotFoundError, match="Required artifact"):
        push_model_assets(models_root, "owner/edge-lipsync-model-assets", api=api)

    assert api.created == []


def test_pull_model_checkpoint_requires_revision() -> None:
    from edge_lipsync.hub import pull_model_checkpoint

    with pytest.raises(ValueError, match="revision"):
        pull_model_checkpoint("owner/avatar-model", revision="")


def test_pull_model_checkpoint_resolves_file_and_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.hub as hub

    cached = tmp_path / "best.pt"
    cached.write_bytes(b"checkpoint")
    calls: list[dict[str, Any]] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setattr(hub, "hf_hub_download", fake_hf_hub_download)

    result = hub.pull_model_checkpoint("owner/avatar-model", revision="model-v1", api=_FakeApi())

    assert result.path == cached
    assert result.requested_revision == "model-v1"
    assert result.resolved_revision == "model-sha"
    assert calls == [
        {
            "repo_id": "owner/avatar-model",
            "filename": "best.pt",
            "revision": "model-v1",
        }
    ]


def test_pull_model_assets_downloads_snapshot_to_local_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.hub as hub
    from edge_lipsync.hub import MODEL_ASSET_UPLOAD_PATTERNS

    local_dir = tmp_path / "models"
    local_dir.mkdir()
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(local_dir)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)

    result = hub.pull_model_assets(
        "owner/edge-lipsync-model-assets",
        revision="asset-v1",
        local_dir=str(local_dir),
        cache_dir="/cache",
        api=_FakeApi(),
    )

    assert result.path == local_dir
    assert result.requested_revision == "asset-v1"
    assert result.resolved_revision == "model-sha"
    assert calls == [
        {
            "repo_id": "owner/edge-lipsync-model-assets",
            "revision": "asset-v1",
            "allow_patterns": MODEL_ASSET_UPLOAD_PATTERNS,
            "local_dir": str(local_dir),
            "cache_dir": "/cache",
        }
    ]


@pytest.mark.parametrize(
    ("script", "description"),
    [
        ("tools/hf_dataset.py", "Manage processed datasets"),
        ("tools/hf_model.py", "Manage trained model artifacts"),
        ("tools/hf_model_assets.py", "Manage reusable model assets"),
    ],
)
def test_hub_cli_help(script: str, description: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert description in result.stdout
    assert "push" in result.stdout
    assert "pull" in result.stdout
