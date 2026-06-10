from __future__ import annotations

import json
import os
import runpy
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


def _write_dataset_snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    (snapshot / "dataset").mkdir(parents=True)
    (snapshot / "dataset/dataset_dict.json").write_text("{}", encoding="utf-8")
    (snapshot / "build_complete.json").write_text("{}", encoding="utf-8")
    (snapshot / "build_metadata.json").write_text("{}", encoding="utf-8")
    (snapshot / "reports/previews").mkdir(parents=True)
    (snapshot / "reports/previews/sample.png").write_bytes(b"preview")
    return snapshot


def test_push_dataset_snapshot_uses_resumable_train_only_upload(tmp_path: Path) -> None:
    from edge_lipsync.hub import DATASET_TRAIN_TRANSFER_PATTERNS, push_dataset_snapshot

    snapshot = _write_dataset_snapshot(tmp_path)
    api = _FakeApi(dataset_sha="full-commit-sha")

    artifact = push_dataset_snapshot(
        snapshot,
        "owner/nora-pairs",
        workers=8,
        api=api,
    )

    assert artifact.resolved_ref == "full-commit-sha"
    assert api.created[-1]["repo_type"] == "dataset"
    assert api.upload_calls == []
    assert api.large_uploads == [
        {
            "folder_path": str(snapshot),
            "repo_id": "owner/nora-pairs",
            "repo_type": "dataset",
            "allow_patterns": DATASET_TRAIN_TRANSFER_PATTERNS,
            "num_workers": 8,
        }
    ]


def test_push_dataset_snapshot_can_include_reports(tmp_path: Path) -> None:
    from edge_lipsync.hub import push_dataset_snapshot

    snapshot = _write_dataset_snapshot(tmp_path)
    api = _FakeApi(dataset_sha="full-commit-sha")

    push_dataset_snapshot(
        snapshot,
        "owner/nora-pairs",
        include_reports=True,
        workers=3,
        api=api,
    )

    assert api.large_uploads == [
        {
            "folder_path": str(snapshot),
            "repo_id": "owner/nora-pairs",
            "repo_type": "dataset",
            "num_workers": 3,
        }
    ]


def test_push_dataset_snapshot_can_retry_same_folder_after_interruption(
    tmp_path: Path,
) -> None:
    from edge_lipsync.hub import push_dataset_snapshot

    class InterruptOnceApi(_FakeApi):
        def __init__(self) -> None:
            super().__init__(dataset_sha="full-commit-sha")
            self.attempts = 0

        def upload_large_folder(self, **kwargs: Any) -> None:
            self.large_uploads.append(kwargs)
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("upload interrupted")

    snapshot = _write_dataset_snapshot(tmp_path)
    api = InterruptOnceApi()

    with pytest.raises(ConnectionError, match="upload interrupted"):
        push_dataset_snapshot(snapshot, "owner/nora-pairs", api=api)

    artifact = push_dataset_snapshot(snapshot, "owner/nora-pairs", api=api)

    assert artifact.resolved_ref == "full-commit-sha"
    assert len(api.large_uploads) == 2
    assert snapshot.is_dir()


@pytest.mark.parametrize("workers", [0, -1])
def test_dataset_snapshot_transfer_rejects_invalid_worker_count(
    tmp_path: Path,
    workers: int,
) -> None:
    from edge_lipsync.hub import pull_dataset_snapshot, push_dataset_snapshot

    snapshot = _write_dataset_snapshot(tmp_path)
    with pytest.raises(ValueError, match="workers"):
        push_dataset_snapshot(snapshot, "owner/nora-pairs", workers=workers, api=_FakeApi())
    with pytest.raises(ValueError, match="workers"):
        pull_dataset_snapshot(
            "owner/nora-pairs",
            ref="full-sha",
            local_dir=str(tmp_path / "downloaded"),
            workers=workers,
            api=_FakeApi(dataset_sha="full-sha"),
            verify=lambda _path: {},
        )


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
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)

    artifact = hub.pull_dataset_snapshot(
        "owner/nora-pairs",
        ref="full-sha",
        local_dir=str(downloaded),
        workers=16,
        api=_FakeApi(dataset_sha="full-sha"),
        verify=lambda _path: {"train": "a", "val": "b"},
    )

    marker = json.loads((downloaded / ".snapshot_complete.json").read_text())
    assert artifact.path == downloaded
    assert marker["repo_id"] == "owner/nora-pairs"
    assert marker["resolved_ref"] == "full-sha"
    assert marker["download_profile"] == "train-only"
    assert calls == [
        {
            "repo_id": "owner/nora-pairs",
            "repo_type": "dataset",
            "revision": "full-sha",
            "local_dir": str(downloaded),
            "allow_patterns": hub.DATASET_TRAIN_TRANSFER_PATTERNS,
            "max_workers": 16,
        }
    ]


def test_pull_dataset_snapshot_full_profile_downloads_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.hub as hub

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)

    hub.pull_dataset_snapshot(
        "owner/nora-pairs",
        ref="full-sha",
        local_dir=str(downloaded),
        include_reports=True,
        workers=6,
        api=_FakeApi(dataset_sha="full-sha"),
        verify=lambda _path: {"train": "a", "val": "b"},
    )

    marker = json.loads((downloaded / ".snapshot_complete.json").read_text())
    assert marker["download_profile"] == "full"
    assert calls == [
        {
            "repo_id": "owner/nora-pairs",
            "repo_type": "dataset",
            "revision": "full-sha",
            "local_dir": str(downloaded),
            "max_workers": 6,
        }
    ]


def test_pull_dataset_snapshot_train_only_marker_does_not_satisfy_full_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.hub as hub

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / ".snapshot_complete.json").write_text(
        json.dumps(
            {
                "repo_id": "owner/nora-pairs",
                "resolved_ref": "full-sha",
                "dataset_fingerprints": {"train": "a", "val": "b"},
                "download_profile": "train-only",
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)

    hub.pull_dataset_snapshot(
        "owner/nora-pairs",
        ref="full-sha",
        local_dir=str(downloaded),
        include_reports=True,
        api=_FakeApi(dataset_sha="full-sha"),
        verify=lambda _path: {"train": "a", "val": "b"},
    )

    assert len(calls) == 1


def test_pull_dataset_snapshot_full_marker_satisfies_train_only_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.hub as hub

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / ".snapshot_complete.json").write_text(
        json.dumps(
            {
                "repo_id": "owner/nora-pairs",
                "resolved_ref": "full-sha",
                "dataset_fingerprints": {"train": "a", "val": "b"},
                "download_profile": "full",
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **kwargs: calls.append(kwargs) or str(downloaded),
    )

    artifact = hub.pull_dataset_snapshot(
        "owner/nora-pairs",
        ref="full-sha",
        local_dir=str(downloaded),
        api=_FakeApi(dataset_sha="full-sha"),
        verify=lambda _path: {"train": "a", "val": "b"},
    )

    assert artifact.path == downloaded
    assert calls == []


def test_hf_dataset_cli_enables_xet_high_performance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)

    runpy.run_path("tools/hf_dataset.py", run_name="hf_dataset_test")

    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"


@pytest.mark.parametrize("command", ["push-snapshot", "pull-snapshot"])
def test_hf_dataset_snapshot_cli_exposes_transfer_options(command: str) -> None:
    result = subprocess.run(
        [sys.executable, "tools/hf_dataset.py", command, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--workers" in result.stdout
    assert "--include-reports" in result.stdout


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
