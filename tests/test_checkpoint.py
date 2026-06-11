from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def test_atomic_torch_save_roundtrip(tmp_path: Path) -> None:
    from edge_lipsync.checkpoint import atomic_torch_save

    out = tmp_path / "payload.pt"
    payload = {"value": torch.tensor([1, 2, 3])}
    atomic_torch_save(payload, out)

    loaded = torch.load(out, map_location="cpu")
    assert loaded["value"].tolist() == [1, 2, 3]
    assert not (tmp_path / "payload.pt.tmp").exists()


def test_atomic_torch_save_removes_temp_file_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edge_lipsync.checkpoint import atomic_torch_save

    out = tmp_path / "payload.pt"
    out.write_bytes(b"existing checkpoint")

    def fail_save(payload: object, path: Path) -> None:
        path.write_bytes(b"partial checkpoint")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.raises(RuntimeError, match="simulated"):
        atomic_torch_save({"value": 1}, out)

    assert out.read_bytes() == b"existing checkpoint"
    assert not (tmp_path / "payload.pt.tmp").exists()


def test_make_training_checkpoint_includes_reproducibility_metadata(tmp_path: Path) -> None:
    from edge_lipsync.checkpoint import TRAIN_CHECKPOINT_FORMAT, make_training_checkpoint
    from edge_lipsync.dataset import manifest_sha256

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"clip_id":"clip_001"}\n', encoding="utf-8")
    model = torch.nn.Conv2d(6, 3, kernel_size=1)

    payload = make_training_checkpoint(
        model=model,
        training_config={"batch_size": 2},
        dataset_root=tmp_path,
        manifest_path=manifest,
        step=5,
        epoch=2,
        metrics={"val_loss": 0.25},
        init_weight_source={"kind": "ncnn_bin", "path": "/tmp/dh_model.bin"},
    )

    assert payload["format"] == TRAIN_CHECKPOINT_FORMAT
    assert payload["training_config"] == {"batch_size": 2}
    assert payload["dataset_root"] == str(tmp_path.resolve())
    assert payload["manifest_path"] == str(manifest.resolve())
    assert payload["manifest_sha256"] == manifest_sha256(manifest)
    assert payload["step"] == 5
    assert payload["epoch"] == 2
    assert payload["metrics"] == {"val_loss": 0.25}
    assert payload["init_weight_source"]["kind"] == "ncnn_bin"


def test_make_training_checkpoint_includes_provenance(tmp_path: Path) -> None:
    from edge_lipsync.checkpoint import make_training_checkpoint

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"clip_id":"clip_001"}\n', encoding="utf-8")
    model = torch.nn.Conv2d(6, 3, kernel_size=1)

    payload = make_training_checkpoint(
        model=model,
        training_config={},
        dataset_root=tmp_path,
        manifest_path=manifest,
        step=1,
        epoch=1,
        metrics={},
        init_weight_source={"kind": "ncnn_bin", "path": "/tmp/dh_model.bin"},
        provenance={"dataset": {"source": "local"}},
    )

    assert payload["provenance"] == {"dataset": {"source": "local"}}


def test_random_state_roundtrip_restores_python_numpy_and_torch() -> None:
    from edge_lipsync.checkpoint import capture_random_state, restore_random_state

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_random_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_random_state(state)

    assert random.random() == expected[0]
    assert float(np.random.random()) == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_clone_state_dict_to_cpu_detaches_storage() -> None:
    from edge_lipsync.checkpoint import clone_state_dict_to_cpu

    original = {"weight": torch.tensor([1.0], requires_grad=True)}
    cloned = clone_state_dict_to_cpu(original)
    original["weight"].data.fill_(2.0)

    assert cloned["weight"].device.type == "cpu"
    assert cloned["weight"].requires_grad is False
    assert cloned["weight"].tolist() == [1.0]


def test_make_training_resume_checkpoint_includes_full_state(tmp_path: Path) -> None:
    from edge_lipsync.checkpoint import (
        RESUME_CHECKPOINT_FORMAT,
        capture_random_state,
        make_training_resume_checkpoint,
    )

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    generator = torch.Generator().manual_seed(123)

    payload = make_training_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        training_config={"max_steps": 10},
        dataset_identity={"kind": "manifest", "manifest_sha256": "abc"},
        dataset_root=tmp_path,
        manifest_path=tmp_path / "manifest.jsonl",
        runtime={"device_type": "cpu", "mixed_precision": False},
        step=4,
        epoch=2,
        next_batch_index=3,
        epoch_sample_indices=[2, 0, 1],
        data_order_generator_state=generator.get_state(),
        best_val_loss=0.2,
        early_stopping_best_val_loss=0.25,
        validations_without_improvement=1,
        best_metrics={"step": 3, "val_loss": 0.2},
        best_model_state_dict=best_state,
        metrics_history=[{"step": 4, "train_loss": 0.5}],
        random_state=capture_random_state(),
        init_weight_source={"kind": "ncnn_bin"},
        provenance={"dataset": {"source": "local"}},
    )

    assert payload["format"] == RESUME_CHECKPOINT_FORMAT
    assert payload["step"] == 4
    assert payload["epoch"] == 2
    assert payload["next_batch_index"] == 3
    assert payload["epoch_sample_indices"] == [2, 0, 1]
    assert payload["dataset_identity"]["manifest_sha256"] == "abc"
    assert payload["optimizer_state_dict"] == optimizer.state_dict()
    assert payload["scaler_state_dict"] == scaler.state_dict()
    assert payload["best_metrics"]["val_loss"] == 0.2
    assert payload["metrics_history"][0]["step"] == 4
    assert payload["runtime"] == {"device_type": "cpu", "mixed_precision": False}


def test_load_training_resume_checkpoint_validates_format_and_fields(tmp_path: Path) -> None:
    from edge_lipsync.checkpoint import load_training_resume_checkpoint

    invalid_format = tmp_path / "invalid_format.pt"
    torch.save({"format": "wrong"}, invalid_format)
    with pytest.raises(ValueError, match="Unsupported resume checkpoint format"):
        load_training_resume_checkpoint(invalid_format)

    missing_field = tmp_path / "missing_field.pt"
    torch.save({"format": "edge_lipsync_duix_unet_resume_v1"}, missing_field)
    with pytest.raises(ValueError, match="missing required fields"):
        load_training_resume_checkpoint(missing_field)


def test_make_model_checkpoint_from_state_dict_clones_weights() -> None:
    from edge_lipsync.checkpoint import make_model_checkpoint_from_state_dict

    state = {"weight": torch.tensor([1.0])}
    payload = make_model_checkpoint_from_state_dict(state, extra={"step": 3})
    state["weight"].fill_(2.0)

    assert payload["format"] == "duix_unet_hardcoded_blocks_ckpt_v2"
    assert payload["state_dict"]["weight"].tolist() == [1.0]
    assert payload["extra"] == {"step": 3}


def test_make_training_checkpoint_from_state_dict_preserves_metadata(tmp_path: Path) -> None:
    from edge_lipsync.checkpoint import make_training_checkpoint_from_state_dict

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"clip_id":"clip"}\n', encoding="utf-8")
    state = {"weight": torch.tensor([1.0])}

    payload = make_training_checkpoint_from_state_dict(
        state_dict=state,
        training_config={"max_steps": 10},
        dataset_root=tmp_path,
        manifest_path=manifest,
        step=3,
        epoch=2,
        metrics={"val_loss": 0.1},
        init_weight_source={"kind": "ncnn_bin"},
        provenance={"dataset": {"source": "local"}},
    )
    state["weight"].fill_(2.0)

    assert payload["format"] == "edge_lipsync_duix_unet_train_v1"
    assert payload["state_dict"]["weight"].tolist() == [1.0]
    assert payload["step"] == 3
    assert payload["epoch"] == 2
    assert payload["metrics"] == {"val_loss": 0.1}


def test_export_checkpoint_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/export_checkpoint.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Export Duix NCNN" in result.stdout
