#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.build_dataset import smooth_bboxes  # noqa: E402
from edge_lipsync.duix_detector import HistoricalDuixScrfdPfpldDetector  # noqa: E402
from edge_lipsync.parity import bbox_metrics  # noqa: E402
from edge_lipsync.preprocess import BBox  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(path: Path, *, compatibility: str) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "compatibility": compatibility,
    }


def _load_oracle(path: Path) -> dict[int, BBox]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected bbox oracle object, got {type(payload).__name__}")
    return {
        int(key): (int(values[0]), int(values[2]), int(values[1]), int(values[3]))
        for key, values in payload.items()
    }


def _aggregate(reference: dict[int, BBox], candidate: dict[int, BBox]) -> dict[str, Any]:
    frame_ids = sorted(set(reference) & set(candidate))
    rows = [bbox_metrics(reference[frame_id], candidate[frame_id]) for frame_id in frame_ids]
    if not rows:
        return {"count": 0}
    iou = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    center = np.asarray([row["center_drift_px"] for row in rows], dtype=np.float64)
    edges = np.asarray(
        [
            np.abs(np.asarray(reference[frame_id]) - np.asarray(candidate[frame_id])).mean()
            for frame_id in frame_ids
        ],
        dtype=np.float64,
    )
    return {
        "count": len(rows),
        "iou_mean": float(iou.mean()),
        "iou_p95": float(np.percentile(iou, 95)),
        "iou_pass_fraction_at_0_995": float((iou >= 0.995).mean()),
        "center_drift_mean": float(center.mean()),
        "center_drift_p95": float(np.percentile(center, 95)),
        "center_pass_fraction_at_1px": float((center <= 1.0).mean()),
        "edge_mae": float(edges.mean()),
    }


def compare_detector(
    *,
    frames_dir: Path,
    oracle_bbox_json: Path,
    scrfd_param: Path,
    scrfd_bin: Path,
    pfpld_onnx: Path,
    pfpld_channel_order: str,
    smooth_radius: int,
    output: Path,
) -> dict[str, Any]:
    oracle = _load_oracle(oracle_bbox_json)
    frame_paths = sorted(
        (path for path in frames_dir.glob("*.sij") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not frame_paths:
        raise ValueError(f"No numeric .sij frames in {frames_dir}")
    detector = HistoricalDuixScrfdPfpldDetector(
        scrfd_param=scrfd_param,
        scrfd_bin=scrfd_bin,
        pfpld_onnx=pfpld_onnx,
        pfpld_channel_order=pfpld_channel_order,  # type: ignore[arg-type]
    )
    detected: dict[int, BBox] = {}
    frames: list[dict[str, Any]] = []
    try:
        for frame_path in frame_paths:
            frame_id = int(frame_path.stem)
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot decode frame: {frame_path}")
            bbox, debug = detector.detect_bbox_with_debug(frame)
            if bbox is not None:
                detected[frame_id] = bbox
            frames.append(
                {
                    "frame_idx": frame_id,
                    "oracle_bbox_xyxy": list(oracle[frame_id]),
                    "detected_bbox_xyxy": list(bbox) if bbox is not None else None,
                    **debug,
                }
            )
    finally:
        detector.close()
    smoothed = smooth_bboxes(detected, radius=smooth_radius)
    missing = sorted(set(oracle) - set(detected))
    payload = {
        "kind": "historical_duix_scrfd_pfpld_bbox_comparison",
        "status": "completed",
        "oracle_usage": "diagnostics_only_not_manifest_source",
        "historical_source": {
            "repo": "https://github.com/duixcom/Duix-Mobile",
            "branch": "20250714",
            "scrfd": "duix-android/dh_aigc_android/duix-sdk/src/main/cpp/aisdk/scrfd.cpp",
            "pfpld": "duix-android/dh_aigc_android/duix-sdk/src/main/cpp/aisdk/pfpld.cpp",
        },
        "weights": {
            "scrfd_param": _provenance(
                scrfd_param,
                compatibility="public_exact_model_id_not_packager_provenance",
            ),
            "scrfd_bin": _provenance(
                scrfd_bin,
                compatibility="public_exact_model_id_not_packager_provenance",
            ),
            "pfpld_onnx": _provenance(
                pfpld_onnx,
                compatibility="public_compatible_mirror_not_original_duix_ncnn_weight",
            ),
        },
        "pfpld_channel_order": pfpld_channel_order,
        "smooth_radius": smooth_radius,
        "frame_count": len(frame_paths),
        "detected_count": len(detected),
        "missing_frame_indices": missing,
        "raw": _aggregate(oracle, detected),
        "smoothed": _aggregate(oracle, smoothed),
        "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the historical Duix SCRFD+PFPLD detector against an oracle."
    )
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--oracle-bbox-json", type=Path, required=True)
    parser.add_argument("--scrfd-param", type=Path, required=True)
    parser.add_argument("--scrfd-bin", type=Path, required=True)
    parser.add_argument("--pfpld-onnx", type=Path, required=True)
    parser.add_argument("--pfpld-channel-order", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument("--smooth-radius", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare_detector(
        frames_dir=args.frames_dir,
        oracle_bbox_json=args.oracle_bbox_json,
        scrfd_param=args.scrfd_param,
        scrfd_bin=args.scrfd_bin,
        pfpld_onnx=args.pfpld_onnx,
        pfpld_channel_order=args.pfpld_channel_order,
        smooth_radius=args.smooth_radius,
        output=args.output,
    )
    print(f"output={args.output.resolve()}")
    print(f"detected={payload['detected_count']}/{payload['frame_count']}")
    print(f"smoothed={json.dumps(payload['smoothed'])}")


if __name__ == "__main__":
    main()
