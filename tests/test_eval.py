from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest


def test_chw_norm_to_rgb_u8_shape() -> None:
    from edge_lipsync.eval import chw_norm_to_rgb_u8

    x = np.zeros((3, 160, 160), dtype=np.float32)
    rgb = chw_norm_to_rgb_u8(x)

    assert rgb.shape == (160, 160, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb[0, 0, 0]) == 127


def test_temporal_delta_metric_measures_consecutive_predictions() -> None:
    from edge_lipsync.eval import temporal_delta_metric

    frames = [
        np.zeros((3, 2, 2), dtype=np.float32),
        np.ones((3, 2, 2), dtype=np.float32),
        np.full((3, 2, 2), 0.5, dtype=np.float32),
    ]

    assert temporal_delta_metric(frames) == 0.75


def test_write_prediction_grid_writes_four_columns(tmp_path: Path) -> None:
    from edge_lipsync.eval import write_prediction_grid

    chw = np.zeros((3, 160, 160), dtype=np.float32)
    out = tmp_path / "grid.png"

    write_prediction_grid(chw, chw, chw, out)

    grid = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert grid is not None
    assert grid.shape == (160, 640, 3)


def test_write_rgb_video_writes_metadata_next_to_render(tmp_path: Path) -> None:
    from edge_lipsync.eval import write_rgb_video

    frames = [
        np.zeros((16, 32, 3), dtype=np.uint8),
        np.full((16, 32, 3), 255, dtype=np.uint8),
    ]
    out = tmp_path / "validation.mp4"

    metadata_path = write_rgb_video(frames, out, fps=25.0, metadata={"kind": "validation"})

    assert out.exists()
    assert out.stat().st_size > 0
    assert metadata_path == tmp_path / "validation.json"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["kind"] == "validation"


def test_render_eval_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/render_eval.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Render validation" in result.stdout
    assert "--config" in result.stdout
    assert "--hf-dataset-repo" in result.stdout
    assert "--hf-model-repo" in result.stdout


def test_resolve_eval_inputs_uses_hf_dataset_without_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.eval as evaluation
    from edge_lipsync.sources import ResolvedSource

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls: list[tuple[str, dict[str, Any]]] = []
    loaded_dataset = {"train": [1], "val": [2]}
    eval_dataset = object()

    def fake_load_dataset(repo_id: str, *, cache_dir: str = "") -> object:
        calls.append(("dataset", {"repo_id": repo_id, "cache_dir": cache_dir}))
        return loaded_dataset

    def fake_hf_dataset(dataset: object, split: str) -> object:
        calls.append(("hf_dataset", {"dataset": dataset, "split": split}))
        return eval_dataset

    def fake_model(**kwargs: Any) -> ResolvedSource:
        calls.append(("model", kwargs))
        return ResolvedSource(path=checkpoint, provenance={"source": "huggingface"})

    monkeypatch.setattr(evaluation, "load_processed_dataset", fake_load_dataset)
    monkeypatch.setattr(evaluation, "DuixHFDataset", fake_hf_dataset)
    monkeypatch.setattr(evaluation, "resolve_model_source", fake_model)
    config = evaluation.RenderEvalConfig(
        dataset_root="",
        ckpt="",
        out_dir=str(tmp_path / "eval"),
        hf_dataset_repo="owner/avatar-data",
        hf_model_repo="owner/avatar-model",
        hf_model_filename="final.pt",
        hf_cache_dir="/cache",
    )

    resolved = evaluation.resolve_eval_inputs(config)

    assert resolved.dataset is eval_dataset
    assert resolved.checkpoint == checkpoint
    assert resolved.provenance == {
        "dataset": {
            "source": "huggingface_datasets",
            "repo_id": "owner/avatar-data",
        },
        "model": {"source": "huggingface"},
    }
    assert calls == [
        (
            "dataset",
            {
                "repo_id": "owner/avatar-data",
                "cache_dir": "/cache",
            },
        ),
        ("hf_dataset", {"dataset": loaded_dataset, "split": "val"}),
        (
            "model",
            {
                "checkpoint": "",
                "hf_repo": "owner/avatar-model",
                "hf_filename": "final.pt",
                "cache_dir": "/cache",
            },
        ),
    ]
