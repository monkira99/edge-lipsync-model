from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch


class _FaceAudioModel(torch.nn.Module):
    def forward(self, face: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        assert tuple(audio.shape[1:]) == (20, 256)
        return face[:, :3] + self.bias

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))


def test_training_step_updates_parameters() -> None:
    from edge_lipsync.losses import combined_reconstruction_loss
    from edge_lipsync.training import run_train_step

    model = torch.nn.Conv2d(6, 3, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "face": torch.zeros(2, 6, 8, 8),
        "target": torch.ones(2, 3, 8, 8),
    }
    before = model.weight.detach().clone()

    loss = run_train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_fn=combined_reconstruction_loss,
        audio_optional=True,
    )

    assert loss > 0
    assert not torch.equal(before, model.weight.detach())


def test_training_step_rejects_non_finite_loss() -> None:
    from edge_lipsync.training import run_train_step

    model = torch.nn.Conv2d(6, 3, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "face": torch.zeros(2, 6, 8, 8),
        "target": torch.ones(2, 3, 8, 8),
    }

    with pytest.raises(FloatingPointError, match="Non-finite"):
        run_train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=torch.device("cpu"),
            loss_fn=lambda pred, target: pred.mean() * torch.tensor(float("nan")),
            audio_optional=True,
        )


def test_validate_batch_shapes_rejects_invalid_audio() -> None:
    from edge_lipsync.training import validate_batch_shapes

    batch = {
        "face": torch.zeros(2, 6, 160, 160),
        "audio": torch.zeros(2, 19, 256),
        "target": torch.zeros(2, 3, 160, 160),
    }

    with pytest.raises(ValueError, match="audio"):
        validate_batch_shapes(batch)


def test_run_validation_reports_reconstruction_mouth_and_temporal_metrics() -> None:
    from edge_lipsync.training import run_validation

    model = _FaceAudioModel()
    batches = [
        {
            "face": torch.zeros(1, 6, 160, 160),
            "audio": torch.zeros(1, 20, 256),
            "target": torch.ones(1, 3, 160, 160),
        },
        {
            "face": torch.ones(1, 6, 160, 160),
            "audio": torch.zeros(1, 20, 256),
            "target": torch.ones(1, 3, 160, 160),
        },
    ]

    metrics = run_validation(model, batches, torch.device("cpu"))

    assert metrics["val_reconstruction_loss"] > 0
    assert metrics["val_mouth_loss"] > 0
    assert metrics["val_temporal_delta"] > 0


def test_phase_for_step_covers_warmup_main_and_stabilization() -> None:
    from edge_lipsync.training import phase_for_step

    kwargs = {"max_steps": 10, "warmup_steps": 2, "stabilization_steps": 2}

    assert phase_for_step(1, **kwargs) == "warmup"
    assert phase_for_step(3, **kwargs) == "main"
    assert phase_for_step(9, **kwargs) == "stabilization"


def test_mixed_precision_rejects_cpu_device() -> None:
    from edge_lipsync.training import TrainConfig, use_mixed_precision

    config = TrainConfig(
        dataset_root="dataset",
        manifest="manifest",
        run_dir="run",
        precision="mixed",
    )

    with pytest.raises(ValueError, match="CUDA"):
        use_mixed_precision(config, torch.device("cpu"))


def test_build_model_rejects_missing_init_checkpoint(tmp_path: Path) -> None:
    from edge_lipsync.training import TrainConfig, build_model

    config = TrainConfig(
        dataset_root="dataset",
        manifest="manifest",
        run_dir="run",
        init_ckpt=str(tmp_path / "missing.pt"),
    )

    with pytest.raises(FileNotFoundError):
        build_model(config, torch.device("cpu"))


def test_build_model_loads_pinned_hub_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.training as training
    from edge_lipsync.sources import ResolvedSource

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    expected_model = _FaceAudioModel()

    def fake_resolve_model_source(**kwargs: str) -> ResolvedSource:
        assert kwargs == {
            "checkpoint": "",
            "hf_repo": "owner/avatar-model",
            "hf_revision": "model-v1",
            "hf_filename": "best.pt",
            "cache_dir": "/cache",
        }
        return ResolvedSource(
            path=checkpoint,
            provenance={
                "source": "huggingface",
                "repo_id": "owner/avatar-model",
                "requested_revision": "model-v1",
                "resolved_revision": "model-sha",
            },
        )

    monkeypatch.setattr(training, "resolve_model_source", fake_resolve_model_source)
    monkeypatch.setattr(training, "load_ckpt", lambda path, map_location: expected_model)
    config = training.TrainConfig(
        dataset_root="dataset",
        manifest="manifest",
        run_dir="run",
        hf_init_model_repo="owner/avatar-model",
        hf_init_model_revision="model-v1",
        hf_cache_dir="/cache",
    )

    model, source = training.build_model(config, torch.device("cpu"))

    assert model is expected_model
    assert source["kind"] == "huggingface_checkpoint"
    assert source["resolved_revision"] == "model-sha"


def test_write_run_metadata_records_provenance(tmp_path: Path) -> None:
    from edge_lipsync.training import write_run_metadata

    best = tmp_path / "best.pt"
    final = tmp_path / "final.pt"

    out = write_run_metadata(
        tmp_path,
        provenance={"dataset": {"source": "local"}},
        best_checkpoint=best,
        final_checkpoint=final,
    )

    assert json.loads(out.read_text(encoding="utf-8")) == {
        "provenance": {"dataset": {"source": "local"}},
        "artifacts": {
            "best_checkpoint": str(best.resolve()),
            "final_checkpoint": str(final.resolve()),
        },
    }


def test_write_model_card_links_dataset_and_wandb(tmp_path: Path) -> None:
    from edge_lipsync.training import write_model_card

    out = write_model_card(
        tmp_path,
        provenance={
            "dataset": {
                "repo_id": "owner/avatar-data",
                "resolved_revision": "data-sha",
            },
            "wandb": {
                "run_url": "https://wandb.ai/owner/project/runs/run-id",
            },
        },
    )

    text = out.read_text(encoding="utf-8")
    assert "owner/avatar-data" in text
    assert "data-sha" in text
    assert "https://wandb.ai/owner/project/runs/run-id" in text


def test_train_logs_writes_final_artifacts_and_publishes_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.training as training
    from edge_lipsync.hub import HubArtifact
    from edge_lipsync.sources import ResolvedSource

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest = dataset_root / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    class TinyDataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
            return {
                "face": torch.zeros(6, 160, 160),
                "audio": torch.zeros(20, 256),
                "target": torch.ones(3, 160, 160),
            }

    class FakeTracker:
        def __init__(self) -> None:
            self.logged: list[tuple[dict[str, Any], int]] = []
            self.summary: dict[str, Any] = {}
            self.exit_codes: list[int] = []

        @property
        def provenance(self) -> dict[str, str]:
            return {
                "mode": "offline",
                "run_id": "run-id",
                "run_url": "https://wandb.ai/owner/project/runs/run-id",
            }

        def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
            self.logged.append((metrics, step))

        def update_summary(self, values: dict[str, Any]) -> None:
            self.summary.update(values)

        def finish(self, *, exit_code: int = 0) -> None:
            self.exit_codes.append(exit_code)

    tracker = FakeTracker()
    uploads: list[tuple[Path, str, bool]] = []

    monkeypatch.setattr(training, "DuixManifestDataset", TinyDataset)
    monkeypatch.setattr(
        training,
        "resolve_dataset_source",
        lambda **_kwargs: ResolvedSource(
            path=dataset_root,
            provenance={
                "source": "huggingface",
                "repo_id": "owner/avatar-data",
                "requested_revision": "data-v1",
                "resolved_revision": "data-sha",
            },
        ),
    )
    monkeypatch.setattr(
        training,
        "build_model",
        lambda _config, _device: (
            _FaceAudioModel(),
            {"kind": "ncnn_bin", "path": "/tmp/dh_model.bin"},
        ),
    )
    monkeypatch.setattr(training, "create_tracker", lambda *_args, **_kwargs: tracker)

    def fake_push(run_path: Path, repo_id: str, *, private: bool) -> HubArtifact:
        uploads.append((run_path, repo_id, private))
        assert (run_path / "best.pt").exists()
        assert (run_path / "final.pt").exists()
        assert (run_path / "run_metadata.json").exists()
        assert (run_path / "README.md").exists()
        return HubArtifact(
            repo_id=repo_id,
            requested_revision="model-sha",
            resolved_revision="model-sha",
            url="https://huggingface.co/owner/avatar-model/tree/model-sha",
        )

    monkeypatch.setattr(training, "push_model_artifacts", fake_push)
    config = training.TrainConfig(
        dataset_root="",
        manifest="manifest.jsonl",
        run_dir=str(run_dir),
        init_bin="/tmp/dh_model.bin",
        hf_dataset_repo="owner/avatar-data",
        hf_dataset_revision="data-v1",
        hf_model_repo="owner/avatar-model",
        wandb_mode="offline",
        device="cpu",
        max_steps=1,
        warmup_steps=0,
        stabilization_steps=0,
        validation_interval=1,
        checkpoint_interval=1,
    )

    best = training.train(config)

    assert best == run_dir / "best.pt"
    assert (run_dir / "final.pt").exists()
    assert uploads == [(run_dir, "owner/avatar-model", True)]
    assert len(tracker.logged) == 1
    assert tracker.logged[0][1] == 1
    assert tracker.summary["hf_model_revision"] == "model-sha"
    assert tracker.exit_codes == [0]
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["provenance"]["dataset"]["resolved_revision"] == "data-sha"
    assert metadata["provenance"]["model"]["resolved_revision"] == "model-sha"


def test_train_finishes_tracker_with_error_code_when_step_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.training as training
    from edge_lipsync.sources import ResolvedSource

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text("{}\n", encoding="utf-8")

    class TinyDataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
            return {
                "face": torch.zeros(6, 160, 160),
                "audio": torch.zeros(20, 256),
                "target": torch.ones(3, 160, 160),
            }

    class FakeTracker:
        provenance = {"mode": "offline", "run_id": "run-id", "run_url": "run-url"}

        def __init__(self) -> None:
            self.exit_codes: list[int] = []

        def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
            pass

        def update_summary(self, values: dict[str, Any]) -> None:
            pass

        def finish(self, *, exit_code: int = 0) -> None:
            self.exit_codes.append(exit_code)

    tracker = FakeTracker()
    monkeypatch.setattr(training, "DuixManifestDataset", TinyDataset)
    monkeypatch.setattr(
        training,
        "resolve_dataset_source",
        lambda **_kwargs: ResolvedSource(path=dataset_root, provenance={"source": "local"}),
    )
    monkeypatch.setattr(
        training,
        "build_model",
        lambda _config, _device: (
            _FaceAudioModel(),
            {"kind": "ncnn_bin", "path": "/tmp/dh_model.bin"},
        ),
    )
    monkeypatch.setattr(training, "create_tracker", lambda *_args, **_kwargs: tracker)
    monkeypatch.setattr(
        training,
        "run_train_step",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated step failure")),
    )
    config = training.TrainConfig(
        dataset_root=str(dataset_root),
        manifest="manifest.jsonl",
        run_dir=str(tmp_path / "run"),
        init_bin="/tmp/dh_model.bin",
        device="cpu",
        max_steps=1,
        warmup_steps=0,
        stabilization_steps=0,
    )

    with pytest.raises(RuntimeError, match="simulated"):
        training.train(config)

    assert tracker.exit_codes == [1]


def test_train_finishes_tracker_with_error_code_when_dataset_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.training as training
    from edge_lipsync.sources import ResolvedSource

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text("{}\n", encoding="utf-8")

    class FakeTracker:
        provenance = {"mode": "offline", "run_id": "run-id", "run_url": "run-url"}

        def __init__(self) -> None:
            self.exit_codes: list[int] = []

        def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
            pass

        def update_summary(self, values: dict[str, Any]) -> None:
            pass

        def finish(self, *, exit_code: int = 0) -> None:
            self.exit_codes.append(exit_code)

    tracker = FakeTracker()
    monkeypatch.setattr(
        training,
        "resolve_dataset_source",
        lambda **_kwargs: ResolvedSource(path=dataset_root, provenance={"source": "local"}),
    )
    monkeypatch.setattr(
        training,
        "build_model",
        lambda _config, _device: (
            _FaceAudioModel(),
            {"kind": "ncnn_bin", "path": "/tmp/dh_model.bin"},
        ),
    )
    monkeypatch.setattr(training, "create_tracker", lambda *_args, **_kwargs: tracker)
    monkeypatch.setattr(
        training,
        "DuixManifestDataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated dataset failure")),
    )
    config = training.TrainConfig(
        dataset_root=str(dataset_root),
        manifest="manifest.jsonl",
        run_dir=str(tmp_path / "run"),
        init_bin="/tmp/dh_model.bin",
        device="cpu",
        max_steps=1,
    )

    with pytest.raises(RuntimeError, match="simulated dataset failure"):
        training.train(config)

    assert tracker.exit_codes == [1]


def test_tiny_overfit_loss_decreases() -> None:
    from edge_lipsync.losses import combined_reconstruction_loss
    from edge_lipsync.training import run_train_step

    torch.manual_seed(0)
    model = torch.nn.Conv2d(6, 3, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    batch = {
        "face": torch.ones(32, 6, 8, 8),
        "target": torch.full((32, 3, 8, 8), 0.25),
    }

    losses = [
        run_train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=torch.device("cpu"),
            loss_fn=combined_reconstruction_loss,
            audio_optional=True,
        )
        for _ in range(25)
    ]

    assert losses[-1] < losses[0]


def test_train_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/train.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Fine-tune DuixUNet" in result.stdout
