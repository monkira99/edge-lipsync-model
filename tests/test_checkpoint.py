from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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


def test_export_checkpoint_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/export_checkpoint.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Export Duix NCNN" in result.stdout
