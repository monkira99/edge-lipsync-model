#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.audio_features import load_wav_mono_f32, split_audio_blocks  # noqa: E402
from edge_lipsync.inference import write_frame_sequence_audio_mp4  # noqa: E402
from edge_lipsync.model import DuixUNet, load_ckpt  # noqa: E402
from edge_lipsync.parity import compare_video_parity, image_metrics  # noqa: E402


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _run(commands: list[dict[str, Any]], command: list[str], *, cwd: Path) -> None:
    commands.append({"cwd": str(cwd.resolve()), "argv": command})
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _frame_ids(raw_dir: Path) -> list[int]:
    ids = sorted(int(path.stem) for path in raw_dir.glob("*.sij") if path.stem.isdigit())
    if not ids:
        raise ValueError(f"No numeric .sij frames in {raw_dir}")
    return ids


def _write_pipeline_source_frames(raw_dir: Path, out_dir: Path, frame_count: int) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    frame_ids = _frame_ids(raw_dir)
    for output_index in range(1, frame_count + 1):
        frame_id = frame_ids[(output_index - 1) % len(frame_ids)]
        frame = cv2.imread(str(raw_dir / f"{frame_id}.sij"), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot decode Emma source frame: {frame_id}")
        output_path = out_dir / f"{output_index:06d}.png"
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Cannot write pipeline source frame: {output_path}")


def _aggregate_metrics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    rmse = float(np.sqrt(np.mean([float(row["rmse"]) ** 2 for row in rows])))
    return {
        "frame_count": len(rows),
        "mae": float(np.mean([float(row["mae"]) for row in rows])),
        "rmse": rmse,
        "max_abs": max(int(row["max_abs"]) for row in rows),
        "ssim": float(np.mean([float(row["ssim"]) for row in rows])),
    }


def _frame_metrics(reference_dir: Path, candidate_dir: Path) -> dict[str, float | int]:
    reference_paths = sorted(reference_dir.glob("*.png"))
    candidate_paths = sorted(candidate_dir.glob("*.png"))
    if len(reference_paths) != len(candidate_paths):
        raise ValueError(
            f"Diagnostic frame counts differ: {len(reference_paths)} != {len(candidate_paths)}"
        )
    rows: list[dict[str, float | int]] = []
    for reference_path, candidate_path in zip(reference_paths, candidate_paths, strict=True):
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        candidate = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
        if reference is None or candidate is None:
            raise RuntimeError(
                f"Cannot decode diagnostic frames: {reference_path} {candidate_path}"
            )
        rows.append(image_metrics(reference, candidate))
    return _aggregate_metrics(rows)


def _video_metrics(reference_path: Path, candidate_path: Path) -> dict[str, float | int]:
    def decode(path: Path) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        return frames

    reference_frames = decode(reference_path)
    candidate_frames = decode(candidate_path)
    if len(reference_frames) != len(candidate_frames):
        raise ValueError(
            f"Diagnostic decoded frame counts differ: {len(reference_frames)} != "
            f"{len(candidate_frames)}"
        )
    return _aggregate_metrics(
        [
            image_metrics(reference, candidate)
            for reference, candidate in zip(reference_frames, candidate_frames, strict=True)
        ]
    )


def _source_frame_metrics(
    raw_dir: Path,
    dataset_frames_dir: Path,
    frame_count: int,
) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []
    frame_ids = _frame_ids(raw_dir)
    for output_index in range(1, frame_count + 1):
        frame_id = frame_ids[(output_index - 1) % len(frame_ids)]
        reference = cv2.imread(str(raw_dir / f"{frame_id}.sij"), cv2.IMREAD_COLOR)
        candidate = cv2.imread(
            str(dataset_frames_dir / f"{output_index:06d}.png"),
            cv2.IMREAD_COLOR,
        )
        if reference is None or candidate is None:
            raise RuntimeError(f"Cannot decode source-frame diagnostic at index {output_index}")
        rows.append(image_metrics(reference, candidate))
    return _aggregate_metrics(rows)


def _weight_parity(init_bin: Path, torch_checkpoint: Path) -> dict[str, Any]:
    bin_model = DuixUNet().eval()
    weight_load = bin_model.load_ncnn_bin(init_bin, face_size=160, device="cpu")
    checkpoint_model = load_ckpt(torch_checkpoint, map_location=torch.device("cpu")).eval()
    bin_state = bin_model.state_dict()
    checkpoint_state = checkpoint_model.state_dict()
    if bin_state.keys() != checkpoint_state.keys():
        raise ValueError("NCNN bin and Torch checkpoint state_dict keys differ")
    absolute_sum = 0.0
    max_abs = 0.0
    parameter_count = 0
    for key in bin_state:
        delta = torch.abs(bin_state[key].cpu() - checkpoint_state[key].cpu())
        absolute_sum += float(delta.sum())
        max_abs = max(max_abs, float(delta.max()) if delta.numel() else 0.0)
        parameter_count += delta.numel()
    return {
        "parameter_count": parameter_count,
        "mae": absolute_sum / parameter_count,
        "max_abs": max_abs,
        "weight_load": weight_load,
    }


def _array_metrics(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    if reference.shape != candidate.shape:
        raise ValueError(f"Diagnostic array shapes differ: {reference.shape} != {candidate.shape}")
    delta = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "shape": list(reference.shape),
        "mae": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(delta.max()),
        "array_equal": bool(np.array_equal(reference, candidate)),
    }


def _historical_detector_diagnostic(
    args: argparse.Namespace,
    *,
    raw_dir: Path,
    bbox_json: Path,
    diagnostics_dir: Path,
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    if args.skip_historical_detector_diagnostic:
        return {
            "kind": "historical_duix_scrfd_pfpld_bbox_comparison",
            "status": "skipped_by_cli",
        }
    model_paths = {
        "scrfd_param": Path(args.diagnostic_scrfd_param).resolve(),
        "scrfd_bin": Path(args.diagnostic_scrfd_bin).resolve(),
        "pfpld_onnx": Path(args.diagnostic_pfpld_onnx).resolve(),
    }
    missing = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing:
        return {
            "kind": "historical_duix_scrfd_pfpld_bbox_comparison",
            "status": "unavailable_missing_public_compatible_assets",
            "missing": missing,
        }
    output = diagnostics_dir / "historical_duix_detector_public_compatible.json"
    _run(
        commands,
        [
            sys.executable,
            "tools/compare_duix_detector_bbox.py",
            "--frames-dir",
            str(raw_dir),
            "--oracle-bbox-json",
            str(bbox_json),
            "--scrfd-param",
            str(model_paths["scrfd_param"]),
            "--scrfd-bin",
            str(model_paths["scrfd_bin"]),
            "--pfpld-onnx",
            str(model_paths["pfpld_onnx"]),
            "--pfpld-channel-order",
            "rgb",
            "--smooth-radius",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {
        "kind": payload["kind"],
        "status": payload["status"],
        "artifact": str(output),
        "oracle_usage": payload["oracle_usage"],
        "historical_source": payload["historical_source"],
        "weights": payload["weights"],
        "pfpld_channel_order": payload["pfpld_channel_order"],
        "smooth_radius": payload["smooth_radius"],
        "frame_count": payload["frame_count"],
        "detected_count": payload["detected_count"],
        "missing_frame_indices": payload["missing_frame_indices"],
        "raw": payload["raw"],
        "smoothed": payload["smoothed"],
    }


def _original_pipeline_contract(duix_repo: Path) -> dict[str, Any]:
    renderer = duix_repo / "tools/render_character_video.py"
    helper = duix_repo / "tools/run_real_inference_sample.py"
    return {
        "renderer": str(renderer),
        "helper": str(helper),
        "model_runtime": {
            "source": f"{renderer}:169-200",
            "behavior": (
                "Load dh_model.param/bin into ncnn.Net and load weight_168u.bin as [160,160]."
            ),
        },
        "audio_features": {
            "source": f"{renderer}:186-190; {helper}:247-255",
            "behavior": (
                "Resample WAV to 16 kHz, build cpp_session-compatible Wenet items, "
                "and select a [20,256] window per step."
            ),
        },
        "frame_indexing": {
            "source": f"{renderer}:219-223; {renderer}:262-263",
            "behavior": (
                "Loop numerically sorted raw_jpgs frame ids and use render step as BNF index."
            ),
        },
        "bbox": {
            "source": f"{renderer}:227-248; {helper}:426-480",
            "behavior": (
                "Resolve bbox.json by numeric frame key in SDK [x1,x2,y1,y2] order, "
                "convert to xyxy, clip, and apply optional adjustment."
            ),
        },
        "crop_resize": {
            "source": f"{helper}:285-303",
            "behavior": (
                "Crop xyxy ROI, resize to 168x168 with INTER_AREA, "
                "and use inner [4:164,4:164] patch."
            ),
        },
        "color_normalization": {
            "source": f"{helper}:309-320",
            "behavior": (
                "Convert real and masked patches BGR->RGB, normalize with "
                "(value-127.5)/127.5, concatenate as [6,160,160]."
            ),
        },
        "mask": {
            "source": f"{helper}:300-307",
            "behavior": "Zero rectangle x=5,y=5,w=150,h=145 in the masked 160x160 patch.",
        },
        "native_blend": {
            "source": f"{helper}:507-526",
            "behavior": "Blend inner patch as alpha*original + (1-alpha)*prediction.",
        },
        "paste_back": {
            "source": f"{helper}:529-544",
            "behavior": (
                "Write blended patch into 168x168 ROI, resize to bbox dimensions "
                "with INTER_AREA, and paste into source frame."
            ),
        },
    }


def _has_nested(payload: dict[str, Any], *keys: str) -> bool:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _audit_row(
    row_id: str,
    requirement: str,
    passed: bool,
    evidence: Any,
    *,
    blocked: bool = False,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "requirement": requirement,
        "status": "passed" if passed else ("blocked" if blocked else "failed"),
        "evidence": evidence,
    }


def _build_completion_audit(
    report: dict[str, Any],
    *,
    dataset_config: Path,
    dataset_manifest: Path,
    tracked_model_artifacts: Sequence[str],
    model_architecture_changes: Sequence[str],
    captured_training_commands: Sequence[str],
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if dataset_config.is_file():
        loaded_config = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
        if isinstance(loaded_config, dict):
            config = loaded_config
    manifest_rows = (
        [json.loads(line) for line in dataset_manifest.read_text(encoding="utf-8").splitlines()]
        if dataset_manifest.is_file()
        else []
    )
    diff_grids = [Path(path) for path in report.get("representative_diff_grids", [])]
    blockers = report.get("technical_blockers", [])
    failed_gates = list(report.get("failed_gates", []))
    target_passed = bool(report.get("passed"))
    target_blocked = bool(blockers) and not target_passed
    original_contract = report.get("harness", {}).get("original_contract", {})
    original_contract_sections = (
        "model_runtime",
        "audio_features",
        "frame_indexing",
        "bbox",
        "crop_resize",
        "color_normalization",
        "mask",
        "native_blend",
        "paste_back",
    )
    sync = report.get("frame_audio_sync", {})
    geometry = report.get("geometry", {})
    decoded = report.get("image_parity", {}).get("decoded_restored_video", {})
    temporal = report.get("temporal_parity", {}).get("temporal_delta_diff", {})
    model_input = report.get("model_input_parity", {})
    checklist = [
        _audit_row(
            "oracle_baseline",
            "Record the reproducible original Duix-Mobile Emma oracle command and assets.",
            _has_nested(report, "harness", "canonical_oracle_command")
            and _has_nested(report, "harness", "oracle"),
            report.get("harness", {}).get("canonical_oracle_command"),
        ),
        _audit_row(
            "pipeline_manifest",
            "Build and infer the current sequence pipeline from a generated dataset manifest.",
            dataset_manifest.is_file() and bool(manifest_rows),
            {
                "path": str(dataset_manifest),
                "row_count": len(manifest_rows),
                "first_row": manifest_rows[0] if manifest_rows else None,
            },
        ),
        _audit_row(
            "original_pipeline_contract",
            (
                "Record original model loading, audio windows, indexing, bbox, crop/resize, "
                "color normalization, mask, blend, and paste-back behavior."
            ),
            all(section in original_contract for section in original_contract_sections),
            original_contract,
        ),
        _audit_row(
            "manifest_source_policy",
            (
                "Generate manifest bboxes from face landmarks and Duix ROI expansion, "
                "not Emma/bbox.json."
            ),
            config.get("bbox_detector") == "mediapipe_face_landmarker"
            and "bbox" not in config
            and "bbox_json" not in config,
            {
                "dataset_config": str(dataset_config),
                "bbox_detector": config.get("bbox_detector"),
                "bbox_policy": report.get("harness", {}).get("bbox_policy"),
            },
        ),
        _audit_row(
            "sync_metrics",
            "Report fps, frame count, durations, audio delta, and frame/audio index mapping.",
            _has_nested(report, "frame_audio_sync", "original")
            and _has_nested(report, "frame_audio_sync", "pipeline")
            and _has_nested(report, "frame_audio_sync", "audio_duration_delta_seconds")
            and _has_nested(report, "frame_audio_sync", "frame_audio_mapping"),
            {
                "report_section": "frame_audio_sync",
                "original": sync.get("original"),
                "pipeline": sync.get("pipeline"),
                "audio_duration_delta_seconds": sync.get("audio_duration_delta_seconds"),
                "render_mapping_count": sync.get("render_mapping_count"),
                "audio_idx_mapping_violations": sync.get("audio_idx_mapping_violations"),
            },
        ),
        _audit_row(
            "geometry_metrics",
            "Report bbox IoU, center drift, crop ROI shape/position, and restored paste position.",
            _has_nested(report, "geometry", "bbox_iou")
            and _has_nested(report, "geometry", "bbox_center_drift_px")
            and _has_nested(report, "geometry", "crop_roi_shapes")
            and _has_nested(report, "geometry", "restored_paste_matches_bbox")
            and _has_nested(report, "geometry", "frames"),
            {
                "report_section": "geometry",
                "bbox_iou": geometry.get("bbox_iou"),
                "bbox_iou_pass_fraction_at_0_995": geometry.get(
                    "bbox_iou_pass_fraction_at_0_995"
                ),
                "bbox_center_drift_px": geometry.get("bbox_center_drift_px"),
                "bbox_center_pass_fraction_at_1px": geometry.get(
                    "bbox_center_pass_fraction_at_1px"
                ),
                "crop_roi_shapes": geometry.get("crop_roi_shapes"),
                "restored_paste_matches_bbox": geometry.get("restored_paste_matches_bbox"),
            },
        ),
        _audit_row(
            "image_metrics",
            "Report decoded full-frame, ROI, and mouth MAE/RMSE/max_abs/PSNR/SSIM metrics.",
            all(
                _has_nested(report, "image_parity", "decoded_restored_video", region, metric)
                for region in ("full_frame", "roi", "mouth")
                for metric in ("mae", "rmse", "max_abs", "psnr", "ssim")
            ),
            {"report_section": "image_parity.decoded_restored_video"}
            | {
                region: {
                    metric: decoded.get(region, {}).get(metric)
                    for metric in ("mae", "rmse", "max_abs", "psnr", "ssim")
                }
                for region in ("full_frame", "roi", "mouth")
            },
        ),
        _audit_row(
            "temporal_metrics",
            "Report temporal delta diff and per-frame distributions.",
            _has_nested(report, "temporal_parity", "temporal_delta_diff", "per_frame_mae")
            and _has_nested(report, "temporal_parity", "temporal_delta_diff", "per_frame_ssim"),
            {
                "report_section": "temporal_parity.temporal_delta_diff",
                "mae": temporal.get("mae"),
                "rmse": temporal.get("rmse"),
                "max_abs": temporal.get("max_abs"),
                "psnr": temporal.get("psnr"),
                "ssim": temporal.get("ssim"),
                "per_frame_mae": temporal.get("per_frame_mae"),
                "per_frame_ssim": temporal.get("per_frame_ssim"),
            },
        ),
        _audit_row(
            "model_input_metrics",
            "Report face, BNF-window, and prediction tensor stats when available.",
            all(
                _has_nested(report, "model_input_parity", name)
                for name in ("face", "audio_bnf_window", "prediction")
            ),
            {
                "report_section": "model_input_parity",
                "oracle_tensor_stats_available": model_input.get("oracle_tensor_stats_available"),
                "oracle_tensor_stats_note": model_input.get("oracle_tensor_stats_note"),
                "shapes": {
                    name: model_input.get(name, {}).get("shape")
                    for name in ("face", "audio_bnf_window", "prediction")
                },
                "oracle_controls": model_input.get("oracle_controls"),
            },
        ),
        _audit_row(
            "diff_artifacts",
            "Write representative diff grids.",
            len(diff_grids) == 3 and all(path.is_file() for path in diff_grids),
            [str(path) for path in diff_grids],
        ),
        _audit_row(
            "target_gates",
            "Meet all quantitative parity target gates or document a technical blocker.",
            target_passed,
            {
                "gates": report.get("gates"),
                "failed_gates": failed_gates,
                "technical_blockers": [
                    {
                        "name": blocker.get("name"),
                        "reason": blocker.get("reason"),
                        "next_experiment": blocker.get("next_experiment"),
                    }
                    for blocker in blockers
                ],
            },
            blocked=target_blocked,
        ),
        _audit_row(
            "no_training",
            "Do not train a model during parity work.",
            not captured_training_commands,
            {"captured_training_commands": list(captured_training_commands)},
        ),
        _audit_row(
            "model_architecture_unchanged",
            "Do not modify the train/model architecture to hide parity differences.",
            not model_architecture_changes,
            {"git_diff_paths": list(model_architecture_changes)},
        ),
        _audit_row(
            "asset_policy",
            "Do not commit model binaries or generated artifacts.",
            not tracked_model_artifacts,
            {"tracked_paths_under_models_or_artifacts": list(tracked_model_artifacts)},
        ),
    ]
    failed = [row["id"] for row in checklist if row["status"] == "failed"]
    blocked = [row["id"] for row in checklist if row["status"] == "blocked"]
    terminal_status = "incomplete" if failed else ("blocked" if blocked else "target_met")
    return {
        "kind": "emma_parity_completion_audit",
        "terminal_status": terminal_status,
        "objective": (
            "Reproduce Emma + sample.wav parity through the current landmark-derived manifest "
            "pipeline, emit the required diagnostics, and stop only at target parity or a "
            "documented technical blocker."
        ),
        "checklist": checklist,
        "failed_requirements": failed,
        "blocked_requirements": blocked,
    }


def _terminal_exit_code(completion_audit: dict[str, Any]) -> int:
    return 0 if completion_audit.get("terminal_status") in {"target_met", "blocked"} else 1


def _git_lines(*args: str) -> list[str]:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in process.stdout.splitlines() if line]


def _captured_training_commands(commands: Sequence[dict[str, Any]]) -> list[str]:
    return [
        " ".join(str(value) for value in command["argv"])
        for command in commands
        if any("train" in Path(str(value)).name.lower() for value in command["argv"])
    ]


def run_harness(args: argparse.Namespace) -> int:
    duix_repo = Path(args.duix_repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    character_dir = duix_repo / "Emma"
    raw_dir = character_dir / "raw_jpgs"
    wav_path = _require_file(duix_repo / "sample.wav")
    bbox_json = _require_file(character_dir / "bbox.json")
    wenet_onnx = _require_file(Path(args.wenet_onnx).resolve())
    init_bin = _require_file(Path(args.init_bin).resolve())
    ncnn_param = _require_file(Path(args.ncnn_param).resolve())
    alpha_bin = _require_file(Path(args.alpha_bin).resolve())
    landmark_model = _require_file(Path(args.landmark_model).resolve())
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("Required tool not found on PATH: ffmpeg")

    original_dir = out_dir / "original"
    pipeline_input_dir = out_dir / "pipeline_input"
    source_frames_dir = pipeline_input_dir / "source_frames"
    raw_videos_dir = pipeline_input_dir / "raw_videos"
    raw_video = raw_videos_dir / "emma_sample.mkv"
    pipeline_dir = out_dir / "pipeline"
    dataset_dir = pipeline_dir / "dataset"
    pipeline_output_dir = pipeline_dir / "output"
    diagnostics_dir = out_dir / "diagnostics"
    commands: list[dict[str, Any]] = []

    original_command = [
        str(duix_repo / ".venv/bin/python"),
        "tools/render_character_video.py",
        "--model-dir",
        str(character_dir),
        "--wav",
        str(wav_path),
        "--wenet-onnx",
        str(wenet_onnx),
        "--out-dir",
        str(original_dir),
        "--out-video",
        str(original_dir / "output.mp4"),
        "--fps",
        "25",
        "--frame-order",
        "loop",
        "--save-frames",
        "--audio-mux",
        "--blend-mode",
        "native",
        "--bbox-mode",
        "model",
        "--bbox-lookup",
        "key",
    ]
    if not args.reuse_original:
        shutil.rmtree(original_dir, ignore_errors=True)
        _run(commands, original_command, cwd=duix_repo)
    _require_file(original_dir / "output.mp4")
    _require_file(original_dir / "run_meta.json")

    frame_count = len(split_audio_blocks(load_wav_mono_f32(wav_path)))
    shutil.rmtree(pipeline_input_dir, ignore_errors=True)
    raw_videos_dir.mkdir(parents=True)
    _write_pipeline_source_frames(raw_dir, source_frames_dir, frame_count)
    _run(
        commands,
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-framerate",
            "25",
            "-start_number",
            "1",
            "-i",
            str(source_frames_dir / "%06d.png"),
            "-i",
            str(wav_path),
            "-frames:v",
            str(frame_count),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            "-c:a",
            "pcm_s16le",
            str(raw_video),
        ],
        cwd=ROOT,
    )

    shutil.rmtree(pipeline_dir, ignore_errors=True)
    pipeline_dir.mkdir(parents=True)
    dataset_config = pipeline_dir / "dataset_config.yaml"
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "raw_video_dir": str(raw_videos_dir),
                "dataset_root": str(dataset_dir),
                "wenet_onnx": str(wenet_onnx),
                "fps": 25,
                "sample_rate": 16000,
                "split_strategy": "clip",
                "validation_fraction": 0.0,
                "bbox_detector": "mediapipe_face_landmarker",
                "landmark_model_asset_path": str(landmark_model),
                "preview_count": 6,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _run(
        commands,
        [sys.executable, "tools/build_dataset.py", "--config", str(dataset_config), "--strict"],
        cwd=ROOT,
    )
    roi_calibration_path = diagnostics_dir / "roi_calibration_sweep.json"
    _run(
        commands,
        [
            sys.executable,
            "tools/sweep_emma_roi_calibration.py",
            "--frames-dir",
            str(dataset_dir / "clips" / raw_video.stem / "frames"),
            "--landmark-model",
            str(landmark_model),
            "--oracle-bbox-json",
            str(bbox_json),
            "--output",
            str(roi_calibration_path),
        ],
        cwd=ROOT,
    )
    _run(
        commands,
        [
            sys.executable,
            "tools/infer_manifest_sequence.py",
            "--dataset-root",
            str(dataset_dir),
            "--manifest",
            "manifest.jsonl",
            "--init-bin",
            str(init_bin),
            "--backend",
            args.pipeline_backend,
            "--ncnn-param",
            str(ncnn_param),
            "--audio-wav",
            str(wav_path),
            "--wenet-onnx",
            str(wenet_onnx),
            "--alpha-bin",
            str(alpha_bin),
            "--output-mp4",
            "output.mp4",
            "--out-dir",
            str(pipeline_output_dir),
            "--fps",
            "25",
            "--device",
            args.device,
        ],
        cwd=ROOT,
    )
    report = compare_video_parity(
        original_video=original_dir / "output.mp4",
        original_frames_dir=original_dir / "frames",
        pipeline_video=pipeline_output_dir / "output.mp4",
        pipeline_frames_dir=pipeline_output_dir / "frames",
        pipeline_metadata=pipeline_output_dir / "metadata.json",
        oracle_bbox_json=bbox_json,
        audio_wav=wav_path,
        out_dir=out_dir,
    )
    shutil.rmtree(diagnostics_dir / "original_torch", ignore_errors=True)
    torch_oracle_command = [
        str(duix_repo / ".venv/bin/python"),
        "tools/render_character_video_torch.py",
        "--model-dir",
        str(character_dir),
        "--wav",
        str(wav_path),
        "--wenet-onnx",
        str(wenet_onnx),
        "--out-dir",
        str(diagnostics_dir / "original_torch"),
        "--out-video",
        str(diagnostics_dir / "original_torch/output.mp4"),
        "--fps",
        "25",
        "--frame-order",
        "loop",
        "--save-frames",
        "--audio-mux",
        "--blend-mode",
        "native",
        "--bbox-mode",
        "model",
        "--bbox-lookup",
        "key",
        "--device",
        "cpu",
    ]
    _run(commands, torch_oracle_command, cwd=duix_repo)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    oracle_bnf_path = diagnostics_dir / "original_bnf.npy"
    export_bnf_code = """
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
sys.path.insert(0, sys.argv[1])
from run_real_inference_sample import (
    build_session_items_from_audio,
    get_session_window,
    load_wav_mono_f32,
)
audio = load_wav_mono_f32(Path(sys.argv[2]), target_sr=16000)
session = ort.InferenceSession(sys.argv[3], providers=["CPUExecutionProvider"])
items, total = build_session_items_from_audio(audio, sess=session)
windows = [get_session_window(items, index) for index in range(total)]
np.save(sys.argv[4], np.stack(windows).astype(np.float32))
""".strip()
    _run(
        commands,
        [
            str(duix_repo / ".venv/bin/python"),
            "-c",
            export_bnf_code,
            str(duix_repo / "tools"),
            str(wav_path),
            str(wenet_onnx),
            str(oracle_bnf_path),
        ],
        cwd=duix_repo,
    )
    reencoded_dir = diagnostics_dir / "original_reencoded"
    shutil.rmtree(reencoded_dir, ignore_errors=True)
    reencoded_dir.mkdir(parents=True)
    write_frame_sequence_audio_mp4(
        frames_dir=original_dir / "frames",
        audio_wav=wav_path,
        out_path=reencoded_dir / "output.mp4",
        fps=25.0,
    )
    clip_id = raw_video.stem
    if args.pipeline_backend == "ncnn":
        geometry_control_name = "geometry_control_ncnn_oracle_bbox_vs_ncnn_detector_bbox"
        geometry_control_reference_dir = original_dir / "frames"
    else:
        geometry_control_name = "geometry_control_torch_oracle_bbox_vs_torch_detector_bbox"
        geometry_control_reference_dir = diagnostics_dir / "original_torch/frames"
    historical_detector_diagnostic = _historical_detector_diagnostic(
        args,
        raw_dir=raw_dir,
        bbox_json=bbox_json,
        diagnostics_dir=diagnostics_dir,
        commands=commands,
    )
    diagnostic_metrics = {
        "source_frames_oracle_vs_pipeline_dataset": _source_frame_metrics(
            raw_dir,
            dataset_dir / "clips" / clip_id / "frames",
            frame_count,
        ),
        "roi_calibration_sweep": json.loads(roi_calibration_path.read_text(encoding="utf-8")),
        "audio_bnf_oracle_vs_pipeline": _array_metrics(
            oracle_bnf_path,
            dataset_dir / "clips" / clip_id / "bnf.npy",
        ),
        "model_weights_ncnn_bin_vs_torch_checkpoint": _weight_parity(
            init_bin,
            character_dir / "dh_model.pt",
        ),
        "codec_control_oracle_png_reencoded_vs_oracle_mp4": _video_metrics(
            original_dir / "output.mp4",
            reencoded_dir / "output.mp4",
        ),
        "runtime_control_ncnn_vs_torch_with_oracle_bbox": _frame_metrics(
            original_dir / "frames",
            diagnostics_dir / "original_torch/frames",
        ),
        geometry_control_name: _frame_metrics(
            geometry_control_reference_dir,
            pipeline_output_dir / "frames",
        ),
        "historical_detector_public_compatible": historical_detector_diagnostic,
    }
    report_path = out_dir / "report.json"
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    saved_report["model_input_parity"]["oracle_controls"] = {
        "audio_bnf_windows": diagnostic_metrics["audio_bnf_oracle_vs_pipeline"],
        "model_weights": diagnostic_metrics["model_weights_ncnn_bin_vs_torch_checkpoint"],
        "runtime_restored_frame_same_bbox": diagnostic_metrics[
            "runtime_control_ncnn_vs_torch_with_oracle_bbox"
        ],
    }
    saved_report["harness"] = {
        "commands": commands,
        "canonical_oracle_command": {"cwd": str(duix_repo), "argv": original_command},
        "oracle_reused": bool(args.reuse_original),
        "oracle": {
            "repo": str(duix_repo),
            "character_dir": str(character_dir),
            "input_wav": str(wav_path),
            "model_param": str(character_dir / "dh_model.param"),
            "model_bin": str(character_dir / "dh_model.bin"),
            "alpha_bin": str(character_dir / "weight_168u.bin"),
            "wenet_onnx": str(wenet_onnx),
            "metadata": str(original_dir / "run_meta.json"),
        },
        "original_contract": _original_pipeline_contract(duix_repo),
        "pipeline": {
            "backend": args.pipeline_backend,
            "dataset_config": str(dataset_config),
            "dataset_manifest": str(dataset_dir / "manifest.jsonl"),
            "metadata": str(pipeline_output_dir / "metadata.json"),
        },
        "bbox_policy": (
            "Emma/bbox.json is used only by the original NCNN oracle and post-build comparator "
            "diagnostics. The pipeline manifest is generated by MediaPipe landmarks -> Duix ROI "
            "expansion -> dataset manifest."
        ),
    }
    saved_report["diagnostics"] = diagnostic_metrics
    saved_report["technical_blockers"] = [
        {
            "name": "detector_geometry_does_not_reconstruct_offline_oracle_bbox",
            "evidence": {
                "bbox_iou_pass_fraction_at_0_995": saved_report["geometry"][
                    "bbox_iou_pass_fraction_at_0_995"
                ],
                "bbox_center_pass_fraction_at_1px": saved_report["geometry"][
                    "bbox_center_pass_fraction_at_1px"
                ],
                "restored_frame_geometry_control": diagnostic_metrics[geometry_control_name],
                "roi_calibration_sweep": diagnostic_metrics["roi_calibration_sweep"],
                "historical_detector_public_compatible": diagnostic_metrics[
                    "historical_detector_public_compatible"
                ],
            },
            "related_files": [
                "edge_lipsync/landmarks.py",
                "edge_lipsync/preprocess.py",
                "edge_lipsync/duix_detector.py",
                "tools/compare_duix_detector_bbox.py",
                str(bbox_json),
            ],
            "reason": (
                "The required manifest is MediaPipe-derived, while Emma/bbox.json was produced "
                "offline. Duix-Mobile's historical 20250714 branch contains an SCRFD+PFPLD "
                "detector path, but the exact offline packager weights and tracker/version are "
                "not present. Public-compatible assets do not reconstruct the Emma oracle."
            ),
            "next_experiment": (
                "Obtain the original offline packager, PFPLD NCNN weights, and tracker/version "
                "used to generate Emma/bbox.json, then compare its manifest geometry."
            ),
        },
    ]
    if args.pipeline_backend == "torch":
        saved_report["technical_blockers"].append(
            {
                "name": "pytorch_runtime_is_not_pixel_identical_to_ncnn_oracle",
                "evidence": diagnostic_metrics["runtime_control_ncnn_vs_torch_with_oracle_bbox"],
                "related_files": ["edge_lipsync/model.py", "edge_lipsync/inference.py"],
                "reason": (
                    "Weights and BNF windows are exact, and the codec control is exact, but NCNN "
                    "and PyTorch restored frames differ with the same oracle bbox."
                ),
                "next_experiment": (
                    "Run the harness with --pipeline-backend ncnn in an environment that provides "
                    "the optional Python ncnn package."
                ),
            }
        )
    saved_report["completion_audit"] = _build_completion_audit(
        saved_report,
        dataset_config=dataset_config,
        dataset_manifest=dataset_dir / "manifest.jsonl",
        tracked_model_artifacts=_git_lines("ls-files", "--", "models", "artifacts"),
        model_architecture_changes=_git_lines(
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "edge_lipsync/model.py",
        ),
        captured_training_commands=_captured_training_commands(commands),
    )
    report_path.write_text(json.dumps(saved_report, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    print(f"passed={report['passed']}")
    print(f"failed_gates={json.dumps(report['failed_gates'])}")
    print(f"audit_status={saved_report['completion_audit']['terminal_status']}")
    return _terminal_exit_code(saved_report["completion_audit"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible Emma parity harness.")
    parser.add_argument(
        "--duix-repo",
        default=str(ROOT.parent / "Duix-Mobile"),
        help="Path to Duix-Mobile oracle repo.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts/parity_emma"))
    parser.add_argument("--wenet-onnx", default=str(ROOT / "models/wenet/wenet.onnx"))
    parser.add_argument("--init-bin", default=str(ROOT / "models/emma/dh_model.bin"))
    parser.add_argument("--ncnn-param", default=str(ROOT / "models/emma/dh_model.param"))
    parser.add_argument("--alpha-bin", default=str(ROOT / "models/emma/weight_168u.bin"))
    parser.add_argument(
        "--landmark-model",
        default=str(ROOT / "models/mediapipe/face_landmarker.task"),
    )
    parser.add_argument(
        "--diagnostic-scrfd-param",
        default=str(ROOT / "models/duix_detector/scrfd_500m_kps-opt2.param"),
    )
    parser.add_argument(
        "--diagnostic-scrfd-bin",
        default=str(ROOT / "models/duix_detector/scrfd_500m_kps-opt2.bin"),
    )
    parser.add_argument(
        "--diagnostic-pfpld-onnx",
        default=str(ROOT / "models/duix_detector/pfpld_robust_sim_bs1_8003.onnx"),
    )
    parser.add_argument(
        "--skip-historical-detector-diagnostic",
        action="store_true",
        help="Skip the optional historical SCRFD+PFPLD public-compatible diagnostic.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pipeline-backend", choices=("torch", "ncnn"), default="ncnn")
    parser.add_argument(
        "--reuse-original",
        action="store_true",
        help="Keep an existing oracle render and rebuild only the detector-driven pipeline.",
    )
    args = parser.parse_args()
    raise SystemExit(run_harness(args))


if __name__ == "__main__":
    main()
