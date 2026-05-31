from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_image_metrics_reports_exact_match() -> None:
    from edge_lipsync.parity import image_metrics

    image = np.full((24, 24, 3), 80, dtype=np.uint8)

    metrics = image_metrics(image, image.copy())

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["max_abs"] == 0
    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] == 1.0


def test_image_metrics_reports_pixel_difference() -> None:
    from edge_lipsync.parity import image_metrics

    reference = np.zeros((24, 24, 3), dtype=np.uint8)
    candidate = np.ones((24, 24, 3), dtype=np.uint8)

    metrics = image_metrics(reference, candidate)

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["max_abs"] == 1
    assert metrics["psnr"] > 40.0
    assert metrics["ssim"] < 1.0


def test_bbox_metrics_reports_iou_and_center_drift() -> None:
    from edge_lipsync.parity import bbox_metrics

    metrics = bbox_metrics((0, 0, 10, 10), (1, 0, 11, 10))

    assert metrics["iou"] == 90 / 110
    assert metrics["center_drift_px"] == 1.0


def test_write_diff_grids_replaces_stale_files(tmp_path: Path) -> None:
    from edge_lipsync.parity import _write_diff_grids

    diffs_dir = tmp_path / "diffs"
    diffs_dir.mkdir()
    stale_path = diffs_dir / "stale.png"
    stale_path.write_bytes(b"stale")
    frame = np.zeros((24, 24, 3), dtype=np.uint8)

    paths = _write_diff_grids(
        tmp_path,
        [frame],
        [frame],
        [{"mae": 0.0}],
        [(0, 0, 24, 24)],
        [{"bbox_xyxy": [0, 0, 24, 24]}],
    )

    assert not stale_path.exists()
    assert len(paths) == 3
    assert len(list(diffs_dir.glob("*.png"))) == 3


def test_compare_video_parity_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/compare_video_parity.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Compare restored video parity" in result.stdout
    assert "--oracle-bbox-json" in result.stdout
    assert "--pipeline-metadata" in result.stdout


def test_run_emma_parity_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/run_emma_parity.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run reproducible Emma parity harness" in result.stdout
    assert "--duix-repo" in result.stdout
    assert "--reuse-original" in result.stdout
    assert "--pipeline-backend" in result.stdout
    assert "--ncnn-param" in result.stdout
    assert "--diagnostic-scrfd-param" in result.stdout
    assert "--skip-historical-detector-diagnostic" in result.stdout


def test_historical_detector_diagnostic_reports_missing_optional_assets(tmp_path: Path) -> None:
    from tools.run_emma_parity import _historical_detector_diagnostic

    commands: list[dict[str, object]] = []
    args = argparse.Namespace(
        skip_historical_detector_diagnostic=False,
        diagnostic_scrfd_param=str(tmp_path / "scrfd.param"),
        diagnostic_scrfd_bin=str(tmp_path / "scrfd.bin"),
        diagnostic_pfpld_onnx=str(tmp_path / "pfpld.onnx"),
    )

    payload = _historical_detector_diagnostic(
        args,
        raw_dir=tmp_path / "frames",
        bbox_json=tmp_path / "bbox.json",
        diagnostics_dir=tmp_path / "diagnostics",
        commands=commands,  # type: ignore[arg-type]
    )

    assert payload["status"] == "unavailable_missing_public_compatible_assets"
    assert payload["missing"] == [
        str((tmp_path / "scrfd.param").resolve()),
        str((tmp_path / "scrfd.bin").resolve()),
        str((tmp_path / "pfpld.onnx").resolve()),
    ]
    assert commands == []


def test_completion_audit_distinguishes_verified_blocker_from_missing_deliverable(
    tmp_path: Path,
) -> None:
    from tools.run_emma_parity import _build_completion_audit

    config_path = tmp_path / "dataset_config.yaml"
    config_path.write_text(
        yaml.safe_dump({"bbox_detector": "mediapipe_face_landmarker"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"frame_idx": 1, "audio_idx": 0, "bbox_xyxy": [1, 2, 3, 4]}) + "\n",
        encoding="utf-8",
    )
    diff_paths = [tmp_path / f"diff{index}.png" for index in range(3)]
    for diff_path in diff_paths:
        diff_path.write_bytes(b"png")
    image_metrics = {"mae": 0.0, "rmse": 0.0, "max_abs": 0, "psnr": 100.0, "ssim": 1.0}
    report = {
        "passed": False,
        "failed_gates": ["bbox_iou_gte_0_995_for_95pct_frames"],
        "gates": {
            "frame_count_equal": True,
            "fps_equal": True,
            "audio_duration_delta_lte_20ms": True,
            "bbox_iou_gte_0_995_for_95pct_frames": False,
        },
        "frame_audio_sync": {
            "original": {},
            "pipeline": {},
            "audio_duration_delta_seconds": 0.0,
            "frame_audio_mapping": [{"output_index": 1, "audio_idx": 0}],
        },
        "geometry": {
            "bbox_iou": {},
            "bbox_center_drift_px": {},
            "crop_roi_shapes": [],
            "restored_paste_matches_bbox": True,
            "frames": [],
        },
        "image_parity": {
            "decoded_restored_video": {
                "full_frame": image_metrics,
                "roi": image_metrics,
                "mouth": image_metrics,
            }
        },
        "temporal_parity": {
            "temporal_delta_diff": {"per_frame_mae": {}, "per_frame_ssim": {}}
        },
        "model_input_parity": {"face": {}, "audio_bnf_window": {}, "prediction": {}},
        "representative_diff_grids": [str(path) for path in diff_paths],
        "harness": {
            "canonical_oracle_command": {"argv": ["python", "render_character_video.py"]},
            "oracle": {},
            "original_contract": {
                "model_runtime": {},
                "audio_features": {},
                "frame_indexing": {},
                "bbox": {},
                "crop_resize": {},
                "color_normalization": {},
                "mask": {},
                "native_blend": {},
                "paste_back": {},
            },
            "commands": [["python", "tools/build_dataset.py"]],
            "bbox_policy": "comparison diagnostics only",
        },
        "technical_blockers": [{"name": "detector_geometry"}],
    }

    audit = _build_completion_audit(
        report,
        dataset_config=tmp_path / "missing.yaml",
        dataset_manifest=manifest_path,
        tracked_model_artifacts=[],
        model_architecture_changes=[],
        captured_training_commands=[],
    )
    assert audit["terminal_status"] == "incomplete"

    audit = _build_completion_audit(
        report,
        dataset_config=config_path,
        dataset_manifest=manifest_path,
        tracked_model_artifacts=[],
        model_architecture_changes=[],
        captured_training_commands=[],
    )
    assert audit["terminal_status"] == "blocked"
    statuses = {row["id"]: row["status"] for row in audit["checklist"]}
    assert statuses["manifest_source_policy"] == "passed"
    assert statuses["target_gates"] == "blocked"


def test_terminal_exit_code_accepts_documented_blocker_but_not_incomplete_audit() -> None:
    from tools.run_emma_parity import _terminal_exit_code

    assert _terminal_exit_code({"terminal_status": "target_met"}) == 0
    assert _terminal_exit_code({"terminal_status": "blocked"}) == 0
    assert _terminal_exit_code({"terminal_status": "incomplete"}) == 1


def test_sweep_emma_roi_calibration_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/sweep_emma_roi_calibration.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Sweep landmark-only Emma ROI calibration" in result.stdout
    assert "--frames-dir" in result.stdout
    assert "--landmark-model" in result.stdout
    assert "--oracle-bbox-json" in result.stdout
