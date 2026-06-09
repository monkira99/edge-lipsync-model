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


def test_collate_training_batch_keeps_variable_metadata_as_records() -> None:
    import edge_lipsync.training as training

    batch = [
        {
            "face": torch.zeros(6, 160, 160),
            "audio": torch.zeros(20, 256),
            "target": torch.zeros(3, 160, 160),
            "meta": {"clip_id": "clip-a", "flags": ()},
        },
        {
            "face": torch.ones(6, 160, 160),
            "audio": torch.ones(20, 256),
            "target": torch.ones(3, 160, 160),
            "meta": {"clip_id": "clip-b", "flags": ("interpolated_bbox",)},
        },
    ]

    collated = training.collate_training_batch(batch)

    assert tuple(collated["face"].shape) == (2, 6, 160, 160)
    assert tuple(collated["audio"].shape) == (2, 20, 256)
    assert tuple(collated["target"].shape) == (2, 3, 160, 160)
    assert collated["meta"] == [
        {"clip_id": "clip-a", "flags": ()},
        {"clip_id": "clip-b", "flags": ("interpolated_bbox",)},
    ]


def test_run_validation_reports_image_mouth_temporal_and_audio_metrics() -> None:
    from edge_lipsync.losses import combined_reconstruction_loss
    from edge_lipsync.training import run_validation

    class AudioAwareModel(torch.nn.Module):
        def forward(self, face: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
            value = audio[:, 0, 0].view(-1, 1, 1, 1)
            return face[:, :3] + value

    model = AudioAwareModel()
    first_audio = torch.zeros(1, 20, 256)
    first_audio[:, 5, :] = 0.5
    second_audio = torch.zeros(1, 20, 256)
    second_audio[:, 0, :] = 0.25
    second_audio[:, 5, :] = 0.75
    batches = [
        {
            "face": torch.zeros(1, 6, 160, 160),
            "audio": first_audio,
            "target": torch.zeros(1, 3, 160, 160),
            "meta": [{"clip_id": "clip-a", "frame_idx": 1}],
        },
        {
            "face": torch.ones(1, 6, 160, 160),
            "audio": second_audio,
            "target": torch.ones(1, 3, 160, 160),
            "meta": [{"clip_id": "clip-a", "frame_idx": 2}],
        },
    ]

    class MeanDistance(torch.nn.Module):
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean(torch.abs(pred - target), dim=(1, 2, 3), keepdim=True)

    metrics = run_validation(
        model,
        batches,
        torch.device("cpu"),
        lpips_evaluator=MeanDistance(),
    )
    with torch.no_grad():
        expected_val_loss = sum(
            float(
                combined_reconstruction_loss(
                    model(batch["face"], batch["audio"]),
                    batch["target"],
                )
            )
            for batch in batches
        ) / len(batches)

    assert metrics["val_loss"] == pytest.approx(expected_val_loss)
    assert metrics["val_reconstruction_loss"] > 0
    assert metrics["val_mouth_loss"] > 0
    assert metrics["val_temporal_delta"] > 0
    assert metrics["val_mae"] > 0
    assert metrics["val_psnr"] > 0
    assert -1.0 <= metrics["val_ssim"] <= 1.0
    assert metrics["val_mouth_mae"] > 0
    assert metrics["val_mouth_psnr"] > 0
    assert -1.0 <= metrics["val_mouth_ssim"] <= 1.0
    assert metrics["val_mouth_temporal_error"] > 0
    assert metrics["val_temporal_pair_count"] == 1
    assert metrics["val_audio_sensitivity"] > 0
    assert metrics["val_audio_shift_mouth_mae_delta"] != 0
    assert metrics["val_lpips_face"] > 0
    assert metrics["val_lpips_mouth"] > 0


def test_run_validation_does_not_compare_temporal_motion_across_clips() -> None:
    from edge_lipsync.training import run_validation

    model = _FaceAudioModel()
    batches = [
        {
            "face": torch.zeros(1, 6, 160, 160),
            "audio": torch.zeros(1, 20, 256),
            "target": torch.zeros(1, 3, 160, 160),
            "meta": [{"clip_id": "clip-a", "frame_idx": 1}],
        },
        {
            "face": torch.ones(1, 6, 160, 160),
            "audio": torch.zeros(1, 20, 256),
            "target": torch.ones(1, 3, 160, 160),
            "meta": [{"clip_id": "clip-b", "frame_idx": 1}],
        },
    ]

    metrics = run_validation(model, batches, torch.device("cpu"))

    assert metrics["val_temporal_delta"] == 0.0
    assert metrics["val_mouth_temporal_error"] == 0.0
    assert metrics["val_temporal_pair_count"] == 0


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


def test_build_model_loads_hub_checkpoint_without_revision(
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
            "hf_filename": "best.pt",
            "cache_dir": "/cache",
        }
        return ResolvedSource(
            path=checkpoint,
            provenance={
                "source": "huggingface",
                "repo_id": "owner/avatar-model",
                "resolved_ref": "model-sha",
            },
        )

    monkeypatch.setattr(training, "resolve_model_source", fake_resolve_model_source)
    monkeypatch.setattr(training, "load_ckpt", lambda path, map_location: expected_model)
    config = training.TrainConfig(
        dataset_root="dataset",
        manifest="manifest",
        run_dir="run",
        hf_init_model_repo="owner/avatar-model",
        hf_cache_dir="/cache",
    )

    model, source = training.build_model(config, torch.device("cpu"))

    assert model is expected_model
    assert source["kind"] == "huggingface_checkpoint"
    assert source["resolved_ref"] == "model-sha"


def test_prepare_training_datasets_pulls_revision_then_loads_from_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.training as training

    snapshot = tmp_path / "snapshot"
    (snapshot / "dataset").mkdir(parents=True)
    loaded_dataset = {"train": [1], "val": [2]}
    calls: list[object] = []

    class TinyHFDataset:
        def __init__(self, dataset: object, split: str) -> None:
            self.dataset = dataset
            self.split = split

        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
            return {
                "face": torch.zeros(6, 160, 160),
                "audio": torch.zeros(20, 256),
                "target": torch.ones(3, 160, 160),
            }

    monkeypatch.setattr(
        training,
        "pull_dataset_snapshot",
        lambda *args, **kwargs: training.HubArtifact(
            repo_id="owner/nora-pairs",
            requested_ref="sha",
            resolved_ref="sha",
            path=snapshot,
        ),
    )
    monkeypatch.setattr(
        training,
        "load_from_disk",
        lambda path: calls.append(path) or loaded_dataset,
    )
    monkeypatch.setattr(training, "DuixHFDataset", TinyHFDataset)

    prepared = training.prepare_training_datasets(
        training.TrainConfig(
            run_dir=str(tmp_path / "run"),
            init_bin="/tmp/dh_model.bin",
            hf_dataset_repo="owner/nora-pairs",
            hf_dataset_revision="sha",
            hf_dataset_local_dir=str(snapshot),
        )
    )

    assert calls == [snapshot / "dataset"]
    assert prepared.provenance["resolved_ref"] == "sha"
    assert len(prepared.train_dataset) == 1
    assert len(prepared.val_dataset) == 1


def test_hf_snapshot_training_requires_revision_and_local_dir() -> None:
    import edge_lipsync.training as training

    with pytest.raises(ValueError, match="revision"):
        training.prepare_training_datasets(
            training.TrainConfig(
                run_dir="run",
                init_bin="/tmp/dh_model.bin",
                hf_dataset_repo="owner/nora-pairs",
            )
        )


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
                "resolved_ref": "data-sha",
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


def test_build_media_eval_selection_uses_first_val_clips_with_frame_limit() -> None:
    import edge_lipsync.training as training

    class ClipDataset:
        clip_ids = ("clip-a", "clip-a", "clip-a", "clip-b", "clip-b", "clip-c")

        def __len__(self) -> int:
            return len(self.clip_ids)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {"meta": {"clip_id": self.clip_ids[index]}}

    selection = training.build_media_eval_selection(
        ClipDataset(),
        clip_count=2,
        clip_ids=(),
        max_frames_per_clip=2,
    )

    assert selection.clip_ids == ("clip-a", "clip-b")
    assert selection.indices == (0, 1, 3, 4)
    assert len(selection.dataset) == 4


def test_train_renders_and_logs_media_eval_when_best_checkpoint_improves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.training as training
    from edge_lipsync.sources import ResolvedSource

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    rendered: list[dict[str, Any]] = []

    class TinyDataset:
        clip_ids = ("clip-a", "clip-a", "clip-a", "clip-b", "clip-b", "clip-c")

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __len__(self) -> int:
            return len(self.clip_ids)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "face": torch.zeros(6, 160, 160),
                "audio": torch.zeros(20, 256),
                "target": torch.ones(3, 160, 160),
                "meta": {"clip_id": self.clip_ids[index]},
            }

    class FakeTracker:
        def __init__(self) -> None:
            self.videos: list[dict[str, Any]] = []
            self.metrics: list[tuple[dict[str, Any], int]] = []
            self.summary: dict[str, Any] = {}
            self.exit_codes: list[int] = []

        @property
        def provenance(self) -> dict[str, str]:
            return {"mode": "offline", "run_id": "run-id", "run_url": "run-url"}

        def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
            self.metrics.append((metrics, step))

        def log_video(self, name: str, path: str | Path, *, step: int, caption: str = "") -> None:
            self.videos.append(
                {
                    "name": name,
                    "path": str(Path(path)),
                    "step": step,
                    "caption": caption,
                }
            )

        def update_summary(self, values: dict[str, Any]) -> None:
            self.summary.update(values)

        def finish(self, *, exit_code: int = 0) -> None:
            self.exit_codes.append(exit_code)

    tracker = FakeTracker()

    def fake_render_validation_artifacts(**kwargs: Any) -> dict[str, Any]:
        out_dir = Path(kwargs["out_dir"])
        rendered.append(
            {
                "indices": tuple(kwargs["dataset"].indices),
                "out_dir": out_dir,
                "checkpoint_path": Path(kwargs["checkpoint_path"]),
                "logged_steps_before_render": [step for _metrics, step in tracker.metrics],
            }
        )
        video_path = out_dir / "validation_grids.mp4"
        metadata_path = out_dir / "validation_grids.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text("{}", encoding="utf-8")
        return {
            "video_path": str(video_path),
            "metadata_path": str(metadata_path),
            "grid_paths": [],
            "metrics": {"val_reconstruction_loss": 0.5},
        }

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
    monkeypatch.setattr(training, "render_validation_artifacts", fake_render_validation_artifacts)

    config = training.TrainConfig(
        dataset_root=str(dataset_root),
        manifest="manifest.jsonl",
        run_dir=str(run_dir),
        init_bin="/tmp/dh_model.bin",
        device="cpu",
        max_steps=1,
        warmup_steps=0,
        stabilization_steps=0,
        validation_interval=1,
        checkpoint_interval=1,
        media_eval_on_best=True,
        media_eval_clip_count=2,
        media_eval_max_frames_per_clip=2,
        media_eval_log_to_wandb=True,
    )

    training.train(config)

    assert rendered == [
        {
            "indices": (0, 1, 3, 4),
            "out_dir": run_dir / "media_eval" / "step_0000001_best",
            "checkpoint_path": run_dir / "best.pt",
            "logged_steps_before_render": [1],
        }
    ]
    assert tracker.videos == [
        {
            "name": "media_eval/best_validation_grids",
            "path": str(run_dir / "media_eval" / "step_0000001_best" / "validation_grids.mp4"),
            "step": 1,
            "caption": "best.pt step=1 clips=clip-a,clip-b",
        }
    ]


def test_render_best_media_eval_keeps_training_nonfatal_when_wandb_video_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.training as training

    class FakeTracker:
        def log_video(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("wandb upload failed")

    class FakeModel(torch.nn.Module):
        pass

    def fake_render_validation_artifacts(**kwargs: Any) -> dict[str, Any]:
        out_dir = Path(kwargs["out_dir"])
        metadata_path = out_dir / "validation_grids.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text("{}", encoding="utf-8")
        return {
            "video_path": str(out_dir / "validation_grids.mp4"),
            "metadata_path": str(metadata_path),
            "grid_paths": [],
            "metrics": {},
        }

    monkeypatch.setattr(training, "render_validation_artifacts", fake_render_validation_artifacts)
    selection = training.MediaEvalSelection(
        dataset=object(),
        clip_ids=("clip-a",),
        indices=(0,),
    )

    artifacts = training._render_best_media_eval(
        model=FakeModel(),
        selection=selection,
        run_dir=tmp_path,
        best_path=tmp_path / "best.pt",
        device=torch.device("cpu"),
        step=500,
        fps=25.0,
        tracker=FakeTracker(),
        log_to_wandb=True,
    )

    assert artifacts["wandb_video_logged"] is False
    assert artifacts["wandb_video_error"] == "wandb upload failed"


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
                "source": "local",
                "path": str(dataset_root),
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
            requested_ref="model-sha",
            resolved_ref="model-sha",
            url="https://huggingface.co/owner/avatar-model/tree/model-sha",
        )

    monkeypatch.setattr(training, "push_model_artifacts", fake_push)
    config = training.TrainConfig(
        dataset_root=str(dataset_root),
        manifest="manifest.jsonl",
        run_dir=str(run_dir),
        init_bin="/tmp/dh_model.bin",
        hf_model_repo="owner/avatar-model",
        wandb_mode="offline",
        device="cpu",
        max_steps=1,
        warmup_steps=0,
        stabilization_steps=0,
        validation_interval=1,
        checkpoint_interval=1,
        media_eval_on_best=False,
    )

    best = training.train(config)

    assert best == run_dir / "best.pt"
    assert (run_dir / "final.pt").exists()
    assert uploads == [(run_dir, "owner/avatar-model", True)]
    assert len(tracker.logged) == 1
    assert tracker.logged[0][1] == 1
    assert tracker.summary["hf_model_ref"] == "model-sha"
    assert tracker.exit_codes == [0]
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["provenance"]["dataset"]["source"] == "local"
    assert metadata["provenance"]["model"]["resolved_ref"] == "model-sha"


def test_train_prints_progress_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    config = training.TrainConfig(
        dataset_root=str(dataset_root),
        manifest="manifest.jsonl",
        run_dir=str(tmp_path / "run"),
        init_bin="/tmp/dh_model.bin",
        device="cpu",
        max_steps=1,
        warmup_steps=0,
        stabilization_steps=0,
        validation_interval=1,
        checkpoint_interval=1,
        log_interval=1,
        media_eval_on_best=False,
    )

    training.train(config)

    out = capsys.readouterr().out
    assert "[train] step=1/1" in out
    assert "train_loss=" in out
    assert "val_loss=" in out
    assert "val_reconstruction_loss=" in out


def test_train_stops_early_when_val_loss_does_not_improve_by_min_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.training as training
    from edge_lipsync.sources import ResolvedSource

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
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
        provenance = {"mode": "offline", "run_id": "run-id", "run_url": "run-url"}

        def __init__(self) -> None:
            self.logged: list[tuple[dict[str, Any], int]] = []
            self.summary: dict[str, Any] = {}
            self.exit_codes: list[int] = []

        def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
            self.logged.append((metrics, step))

        def update_summary(self, values: dict[str, Any]) -> None:
            self.summary.update(values)

        def finish(self, *, exit_code: int = 0) -> None:
            self.exit_codes.append(exit_code)

    tracker = FakeTracker()
    validation_losses = iter((1.0, 0.95, 0.94))

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
    monkeypatch.setattr(training, "run_train_step", lambda **_kwargs: 0.25)
    monkeypatch.setattr(
        training,
        "run_validation",
        lambda *_args, **_kwargs: {
            "val_loss": next(validation_losses),
            "val_reconstruction_loss": 1.0,
            "val_mouth_loss": 1.0,
            "val_temporal_delta": 0.0,
        },
    )

    training.train(
        training.TrainConfig(
            dataset_root=str(dataset_root),
            manifest="manifest.jsonl",
            run_dir=str(run_dir),
            init_bin="/tmp/dh_model.bin",
            device="cpu",
            max_steps=5,
            warmup_steps=0,
            stabilization_steps=0,
            validation_interval=1,
            checkpoint_interval=10,
            early_stopping_patience=1,
            early_stopping_min_delta=0.1,
            media_eval_on_best=False,
        )
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    best_checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    final_checkpoint = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)

    assert [row["step"] for row in metrics] == [1, 2]
    assert metrics[-1]["early_stop_reason"] == "val_loss_patience"
    assert best_checkpoint["step"] == 2
    assert final_checkpoint["step"] == 2
    assert tracker.summary["early_stop_reason"] == "val_loss_patience"
    assert tracker.summary["early_stop_step"] == 2
    assert tracker.exit_codes == [0]


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
        media_eval_on_best=False,
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

    assert tracker.exit_codes == []


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
