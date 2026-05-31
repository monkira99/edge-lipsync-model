from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _psnr_from_rmse(rmse: float) -> float:
    if rmse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / rmse)


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_f32 = reference.astype(np.float32)
    candidate_f32 = candidate.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_reference = cv2.GaussianBlur(reference_f32, (11, 11), 1.5)
    mu_candidate = cv2.GaussianBlur(candidate_f32, (11, 11), 1.5)
    sigma_reference = cv2.GaussianBlur(reference_f32 * reference_f32, (11, 11), 1.5)
    sigma_reference -= mu_reference * mu_reference
    sigma_candidate = cv2.GaussianBlur(candidate_f32 * candidate_f32, (11, 11), 1.5)
    sigma_candidate -= mu_candidate * mu_candidate
    sigma_cross = cv2.GaussianBlur(reference_f32 * candidate_f32, (11, 11), 1.5)
    sigma_cross -= mu_reference * mu_candidate
    numerator = (2.0 * mu_reference * mu_candidate + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_reference * mu_reference + mu_candidate * mu_candidate + c1) * (
        sigma_reference + sigma_candidate + c2
    )
    return float(np.mean(numerator / denominator))


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Image shapes differ: reference={reference.shape} candidate={candidate.shape}"
        )
    delta = reference.astype(np.float32) - candidate.astype(np.float32)
    rmse = float(np.sqrt(np.mean(delta * delta)))
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": rmse,
        "max_abs": int(np.max(np.abs(delta))),
        "psnr": _psnr_from_rmse(rmse),
        "ssim": _ssim(reference, candidate),
    }


def bbox_metrics(reference: BBox, candidate: BBox) -> dict[str, float]:
    x1 = max(reference[0], candidate[0])
    y1 = max(reference[1], candidate[1])
    x2 = min(reference[2], candidate[2])
    y2 = min(reference[3], candidate[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    reference_area = (reference[2] - reference[0]) * (reference[3] - reference[1])
    candidate_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    union = reference_area + candidate_area - intersection
    reference_center = ((reference[0] + reference[2]) / 2.0, (reference[1] + reference[3]) / 2.0)
    candidate_center = ((candidate[0] + candidate[2]) / 2.0, (candidate[1] + candidate[3]) / 2.0)
    return {
        "iou": float(intersection / union),
        "center_drift_px": float(math.dist(reference_center, candidate_center)),
    }


def _require_tool(name: str) -> str:
    tool = shutil.which(name)
    if tool is None:
        raise FileNotFoundError(f"Required tool not found on PATH: {name}")
    return tool


def probe_media(path: str | Path) -> dict[str, Any]:
    media_path = Path(path)
    if not media_path.exists():
        raise FileNotFoundError(media_path)
    process = subprocess.run(
        [
            _require_tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration:"
                "stream=index,codec_type,codec_name,r_frame_rate,avg_frame_rate,"
                "nb_frames,duration,sample_rate,channels"
            ),
            "-of",
            "json",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(process.stdout)


def _stream(payload: dict[str, Any], codec_type: str) -> dict[str, Any]:
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == codec_type:
            return stream
    return {}


def _fraction(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def _duration(stream: dict[str, Any], payload: dict[str, Any]) -> float:
    value = stream.get("duration", payload.get("format", {}).get("duration", 0.0))
    return float(value)


def _load_png_frames(directory: str | Path) -> list[np.ndarray]:
    paths = sorted(Path(directory).glob("*.png"))
    if not paths:
        raise ValueError(f"No PNG frames in {directory}")
    frames: list[np.ndarray] = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot decode image: {path}")
        frames.append(frame)
    return frames


def _decode_video_frames(path: str | Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No decoded video frames: {path}")
    return frames


def _mouth_bbox(roi: BBox) -> BBox:
    x1, y1, x2, y2 = roi
    width = x2 - x1
    height = y2 - y1
    return (
        int(round(x1 + width * 0.2)),
        int(round(y1 + height * 0.45)),
        int(round(x1 + width * 0.8)),
        int(round(y1 + height * 0.95)),
    )


def _crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def _aggregate_metric_rows(rows: Sequence[dict[str, float | int]]) -> dict[str, Any]:
    if not rows:
        return {"frame_count": 0}
    rmse = math.sqrt(sum(float(row["rmse"]) ** 2 for row in rows) / len(rows))
    return {
        "frame_count": len(rows),
        "mae": float(np.mean([float(row["mae"]) for row in rows])),
        "rmse": rmse,
        "max_abs": max(int(row["max_abs"]) for row in rows),
        "psnr": _psnr_from_rmse(rmse),
        "ssim": float(np.mean([float(row["ssim"]) for row in rows])),
        "per_frame_mae": _summary([float(row["mae"]) for row in rows]),
        "per_frame_ssim": _summary([float(row["ssim"]) for row in rows]),
    }


def _compare_images(
    reference_frames: Sequence[np.ndarray],
    candidate_frames: Sequence[np.ndarray],
    oracle_boxes: Sequence[BBox],
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    count = min(len(reference_frames), len(candidate_frames), len(oracle_boxes))
    full_rows: list[dict[str, float | int]] = []
    roi_rows: list[dict[str, float | int]] = []
    mouth_rows: list[dict[str, float | int]] = []
    for index in range(count):
        reference = reference_frames[index]
        candidate = candidate_frames[index]
        box = oracle_boxes[index]
        full_rows.append(image_metrics(reference, candidate))
        roi_rows.append(image_metrics(_crop(reference, box), _crop(candidate, box)))
        mouth = _mouth_bbox(box)
        mouth_rows.append(image_metrics(_crop(reference, mouth), _crop(candidate, mouth)))
    return {
        "reference_frame_count": len(reference_frames),
        "pipeline_frame_count": len(candidate_frames),
        "compared_frame_count": count,
        "full_frame": _aggregate_metric_rows(full_rows),
        "roi": _aggregate_metric_rows(roi_rows),
        "mouth": _aggregate_metric_rows(mouth_rows),
    }, full_rows


def _compare_temporal(
    reference_frames: Sequence[np.ndarray],
    candidate_frames: Sequence[np.ndarray],
) -> dict[str, Any]:
    count = min(len(reference_frames), len(candidate_frames))
    rows: list[dict[str, float | int]] = []
    for index in range(1, count):
        reference_delta = reference_frames[index].astype(np.int16) - reference_frames[
            index - 1
        ].astype(np.int16)
        candidate_delta = candidate_frames[index].astype(np.int16) - candidate_frames[
            index - 1
        ].astype(np.int16)
        delta_diff = np.abs(reference_delta - candidate_delta)
        rmse = float(np.sqrt(np.mean(delta_diff.astype(np.float32) ** 2)))
        rows.append(
            {
                "mae": float(delta_diff.mean()),
                "rmse": rmse,
                "max_abs": int(delta_diff.max()),
                "psnr": _psnr_from_rmse(rmse),
                "ssim": _ssim(
                    np.clip(reference_delta + 128, 0, 255).astype(np.uint8),
                    np.clip(candidate_delta + 128, 0, 255).astype(np.uint8),
                ),
            }
        )
    return {"temporal_delta_diff": _aggregate_metric_rows(rows)}


def _oracle_bbox(payload: dict[str, Any], frame_idx: int) -> BBox:
    raw = payload[str(frame_idx)]
    if len(raw) != 4:
        raise ValueError(f"Invalid oracle bbox for frame {frame_idx}: {raw}")
    x1, x2, y1, y2 = (int(round(float(value))) for value in raw)
    return x1, y1, x2, y2


def _geometry_metrics(
    pipeline_frames: Sequence[dict[str, Any]],
    oracle_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[BBox]]:
    rows: list[dict[str, Any]] = []
    oracle_boxes: list[BBox] = []
    for frame in pipeline_frames:
        oracle_box = _oracle_bbox(oracle_payload, int(frame["frame_idx"]))
        pipeline_box = tuple(int(value) for value in frame["bbox_xyxy"])
        metrics = bbox_metrics(oracle_box, pipeline_box)  # type: ignore[arg-type]
        oracle_boxes.append(oracle_box)
        rows.append(
            {
                "output_index": int(frame["output_index"]),
                "frame_idx": int(frame["frame_idx"]),
                "oracle_bbox_xyxy": list(oracle_box),
                "pipeline_bbox_xyxy": list(pipeline_box),
                "crop_roi_shape": frame.get("crop_roi_shape", []),
                "restored_paste_xyxy": frame.get("restored_paste_xyxy", list(pipeline_box)),
                **metrics,
            }
        )
    ious = [float(row["iou"]) for row in rows]
    center_drifts = [float(row["center_drift_px"]) for row in rows]
    return {
        "bbox_iou": _summary(ious),
        "bbox_iou_pass_fraction_at_0_995": float(np.mean(np.asarray(ious) >= 0.995)),
        "bbox_center_drift_px": _summary(center_drifts),
        "bbox_center_pass_fraction_at_1px": float(np.mean(np.asarray(center_drifts) <= 1.0)),
        "crop_roi_shapes": sorted({str(row["crop_roi_shape"]) for row in rows}),
        "restored_paste_matches_bbox": all(
            row["restored_paste_xyxy"] == row["pipeline_bbox_xyxy"] for row in rows
        ),
        "frames": rows,
    }, oracle_boxes


def _model_input_metrics(pipeline_frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = ("face", "audio_bnf_window", "prediction")
    output: dict[str, Any] = {
        "oracle_tensor_stats_available": False,
        "oracle_tensor_stats_note": "Original NCNN renderer does not emit tensor dumps.",
    }
    for name in names:
        entries = [
            frame.get("tensor_stats", {}).get(name)
            for frame in pipeline_frames
            if frame.get("tensor_stats", {}).get(name)
        ]
        output[name] = {
            "shape": entries[0]["shape"] if entries else [],
            "min": _summary([float(entry["min"]) for entry in entries]),
            "max": _summary([float(entry["max"]) for entry in entries]),
            "mean": _summary([float(entry["mean"]) for entry in entries]),
            "std": _summary([float(entry["std"]) for entry in entries]),
        }
    return output


def _sync_metrics(
    original_probe: dict[str, Any],
    pipeline_probe: dict[str, Any],
    wav_probe: dict[str, Any],
    pipeline_frames: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    original_video = _stream(original_probe, "video")
    pipeline_video = _stream(pipeline_probe, "video")
    original_audio = _stream(original_probe, "audio")
    pipeline_audio = _stream(pipeline_probe, "audio")
    wav_audio = _stream(wav_probe, "audio")
    mappings = [
        {
            "output_index": int(frame["output_index"]),
            "frame_idx": int(frame["frame_idx"]),
            "audio_idx": int(frame["audio_idx"]),
        }
        for frame in pipeline_frames
    ]
    mapping_violations = [
        row for row in mappings if row["audio_idx"] != row["output_index"] - 1
    ]
    original_audio_duration = _duration(original_audio, original_probe)
    pipeline_audio_duration = _duration(pipeline_audio, pipeline_probe)
    wav_duration = _duration(wav_audio, wav_probe)
    return {
        "original": {
            "fps": _fraction(str(original_video["avg_frame_rate"])),
            "frame_count": int(original_video["nb_frames"]),
            "video_duration_seconds": _duration(original_video, original_probe),
            "audio_duration_seconds": original_audio_duration,
        },
        "pipeline": {
            "fps": _fraction(str(pipeline_video["avg_frame_rate"])),
            "frame_count": int(pipeline_video["nb_frames"]),
            "video_duration_seconds": _duration(pipeline_video, pipeline_probe),
            "audio_duration_seconds": pipeline_audio_duration,
        },
        "wav_duration_seconds": wav_duration,
        "audio_duration_delta_seconds": abs(original_audio_duration - pipeline_audio_duration),
        "pipeline_audio_vs_wav_delta_seconds": abs(pipeline_audio_duration - wav_duration),
        "render_mapping_count": len(mappings),
        "audio_idx_mapping_rule": "audio_idx == output_index - 1",
        "audio_idx_mapping_violations": mapping_violations,
        "frame_audio_mapping": mappings,
    }


def _write_diff_grids(
    out_dir: Path,
    reference_frames: Sequence[np.ndarray],
    candidate_frames: Sequence[np.ndarray],
    full_rows: Sequence[dict[str, float | int]],
    oracle_boxes: Sequence[BBox],
    pipeline_frames: Sequence[dict[str, Any]],
) -> list[str]:
    diffs_dir = out_dir / "diffs"
    shutil.rmtree(diffs_dir, ignore_errors=True)
    diffs_dir.mkdir(parents=True, exist_ok=True)
    ranking = sorted(range(len(full_rows)), key=lambda index: float(full_rows[index]["mae"]))
    selected = [ranking[0], ranking[len(ranking) // 2], ranking[-1]]
    names = ("best", "median", "worst")
    paths: list[str] = []
    for name, index in zip(names, selected, strict=True):
        reference = reference_frames[index]
        candidate = candidate_frames[index]
        delta = np.abs(reference.astype(np.int16) - candidate.astype(np.int16)).astype(np.uint8)
        amplified = np.clip(delta.astype(np.int16) * 4, 0, 255).astype(np.uint8)
        overlay = candidate.copy()
        ox1, oy1, ox2, oy2 = oracle_boxes[index]
        px1, py1, px2, py2 = (int(value) for value in pipeline_frames[index]["bbox_xyxy"])
        cv2.rectangle(overlay, (ox1, oy1), (ox2 - 1, oy2 - 1), (0, 0, 255), 2)
        cv2.rectangle(overlay, (px1, py1), (px2 - 1, py2 - 1), (0, 255, 0), 2)
        grid = np.concatenate([reference, candidate, amplified, overlay], axis=1)
        path = diffs_dir / f"{name}_frame_{index:06d}.png"
        if not cv2.imwrite(str(path), grid):
            raise RuntimeError(f"Cannot write diff grid: {path}")
        paths.append(str(path.resolve()))
    return paths


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def compare_video_parity(
    *,
    original_video: str | Path,
    original_frames_dir: str | Path,
    pipeline_video: str | Path,
    pipeline_frames_dir: str | Path,
    pipeline_metadata: str | Path,
    oracle_bbox_json: str | Path,
    audio_wav: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(pipeline_metadata).read_text(encoding="utf-8"))
    pipeline_frame_rows = metadata["frames"]
    oracle_payload = json.loads(Path(oracle_bbox_json).read_text(encoding="utf-8"))
    geometry, oracle_boxes = _geometry_metrics(pipeline_frame_rows, oracle_payload)
    original_pngs = _load_png_frames(original_frames_dir)
    pipeline_pngs = _load_png_frames(pipeline_frames_dir)
    original_decoded = _decode_video_frames(original_video)
    pipeline_decoded = _decode_video_frames(pipeline_video)
    png_metrics, png_full_rows = _compare_images(original_pngs, pipeline_pngs, oracle_boxes)
    decoded_boxes = oracle_boxes[: min(len(original_decoded), len(pipeline_decoded))]
    decoded_metrics, _decoded_rows = _compare_images(
        original_decoded,
        pipeline_decoded,
        decoded_boxes,
    )
    sync = _sync_metrics(
        probe_media(original_video),
        probe_media(pipeline_video),
        probe_media(audio_wav),
        pipeline_frame_rows,
    )
    temporal = _compare_temporal(original_pngs, pipeline_pngs)
    diff_grids = _write_diff_grids(
        output,
        original_pngs,
        pipeline_pngs,
        png_full_rows,
        oracle_boxes,
        pipeline_frame_rows,
    )
    gates = {
        "frame_count_equal": sync["original"]["frame_count"] == sync["pipeline"]["frame_count"],
        "fps_equal": sync["original"]["fps"] == sync["pipeline"]["fps"],
        "audio_duration_delta_lte_20ms": sync["audio_duration_delta_seconds"] <= 0.020,
        "bbox_iou_gte_0_995_for_95pct_frames": geometry["bbox_iou_pass_fraction_at_0_995"]
        >= 0.95,
        "bbox_center_drift_lte_1px_for_95pct_frames": geometry[
            "bbox_center_pass_fraction_at_1px"
        ]
        >= 0.95,
        "decoded_full_frame_ssim_gte_0_999": decoded_metrics["full_frame"]["ssim"] >= 0.999,
        "decoded_full_frame_mae_lte_1": decoded_metrics["full_frame"]["mae"] <= 1.0,
        "decoded_roi_ssim_gte_0_995": decoded_metrics["roi"]["ssim"] >= 0.995,
        "decoded_roi_mae_lte_2": decoded_metrics["roi"]["mae"] <= 2.0,
        "decoded_mouth_ssim_gte_0_995": decoded_metrics["mouth"]["ssim"] >= 0.995,
        "decoded_mouth_mae_lte_2": decoded_metrics["mouth"]["mae"] <= 2.0,
    }
    report = {
        "kind": "emma_video_parity_report",
        "inputs": {
            "original_video": str(Path(original_video).resolve()),
            "original_frames_dir": str(Path(original_frames_dir).resolve()),
            "pipeline_video": str(Path(pipeline_video).resolve()),
            "pipeline_frames_dir": str(Path(pipeline_frames_dir).resolve()),
            "pipeline_metadata": str(Path(pipeline_metadata).resolve()),
            "oracle_bbox_json": {
                "path": str(Path(oracle_bbox_json).resolve()),
                "usage": "comparison_only_not_manifest_source",
            },
            "audio_wav": str(Path(audio_wav).resolve()),
        },
        "frame_audio_sync": sync,
        "geometry": geometry,
        "image_parity": {
            "restored_png_before_codec": png_metrics,
            "decoded_restored_video": decoded_metrics,
        },
        "temporal_parity": temporal,
        "model_input_parity": _model_input_metrics(pipeline_frame_rows),
        "representative_diff_grids": diff_grids,
        "gates": gates,
        "passed": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    return report
