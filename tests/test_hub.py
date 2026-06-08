from __future__ import annotations

import json
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
    def __init__(self, *, dataset_sha: str = "dataset-sha") -> None:
        self.created: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.large_uploads: list[dict[str, Any]] = []
        self.dataset_sha = dataset_sha

    @property
    def upload_calls(self) -> list[dict[str, Any]]:
        return self.uploads

    def create_repo(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def upload_folder(self, **kwargs: Any) -> _Commit:
        self.uploads.append(kwargs)
        repo_type = kwargs.get("repo_type")
        return _Commit("dataset-commit" if repo_type == "dataset" else "model-commit")

    def upload_large_folder(self, **kwargs: Any) -> None:
        self.large_uploads.append(kwargs)

    def dataset_info(self, **_kwargs: Any) -> _Info:
        return _Info(self.dataset_sha)

    def model_info(self, **_kwargs: Any) -> _Info:
        return _Info("model-sha")


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


def test_push_model_artifacts_uses_model_allowlist(tmp_path: Path) -> None:
    from edge_lipsync.hub import MODEL_UPLOAD_PATTERNS, push_model_artifacts

    api = _FakeApi()
    run_dir = _write_run_dir(tmp_path)

    result = push_model_artifacts(run_dir, "owner/avatar-model", api=api)

    assert result.resolved_ref == "model-commit"
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

    assert result.resolved_ref == "model-commit"
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


def test_pull_model_checkpoint_resolves_latest_file_and_ref(
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

    result = hub.pull_model_checkpoint("owner/avatar-model", api=_FakeApi())

    assert result.path == cached
    assert result.requested_ref == ""
    assert result.resolved_ref == "model-sha"
    assert calls == [
        {
            "repo_id": "owner/avatar-model",
            "filename": "best.pt",
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
        local_dir=str(local_dir),
        cache_dir="/cache",
        api=_FakeApi(),
    )

    assert result.path == local_dir
    assert result.requested_ref == ""
    assert result.resolved_ref == "model-sha"
    assert calls == [
        {
            "repo_id": "owner/edge-lipsync-model-assets",
            "allow_patterns": MODEL_ASSET_UPLOAD_PATTERNS,
            "local_dir": str(local_dir),
            "cache_dir": "/cache",
        }
    ]


def test_push_dataset_snapshot_uploads_complete_package(tmp_path: Path) -> None:
    from edge_lipsync.hub import push_dataset_snapshot

    snapshot = tmp_path / "snapshot"
    (snapshot / "dataset").mkdir(parents=True)
    (snapshot / "dataset/dataset_dict.json").write_text("{}", encoding="utf-8")
    (snapshot / "build_complete.json").write_text("{}", encoding="utf-8")
    api = _FakeApi()

    artifact = push_dataset_snapshot(snapshot, "owner/nora-pairs", api=api)

    assert artifact.resolved_ref == "dataset-commit"
    assert api.created[-1]["repo_type"] == "dataset"
    assert api.upload_calls[-1]["repo_type"] == "dataset"


def test_pull_dataset_snapshot_writes_verified_local_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.hub as hub

    downloaded = tmp_path / "downloaded"
    (downloaded / "dataset/train").mkdir(parents=True)
    (downloaded / "dataset/val").mkdir(parents=True)
    (downloaded / "build_complete.json").write_text(
        json.dumps({"dataset_fingerprints": {"train": "a", "val": "b"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(hub, "snapshot_download", lambda **_kwargs: str(downloaded))

    artifact = hub.pull_dataset_snapshot(
        "owner/nora-pairs",
        ref="full-sha",
        local_dir=str(downloaded),
        api=_FakeApi(dataset_sha="full-sha"),
        verify=lambda _path: {"train": "a", "val": "b"},
    )

    marker = json.loads((downloaded / ".snapshot_complete.json").read_text())
    assert artifact.path == downloaded
    assert marker["repo_id"] == "owner/nora-pairs"
    assert marker["resolved_ref"] == "full-sha"


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
