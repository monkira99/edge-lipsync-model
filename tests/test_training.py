from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
