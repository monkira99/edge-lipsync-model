from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


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
