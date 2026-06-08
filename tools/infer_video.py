#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.inference import run_video_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference from a silent input video and driving audio."
    )
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-mp4", default="output.mp4")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--init-bin", default="")
    parser.add_argument("--hf-model-repo", default="")
    parser.add_argument("--hf-model-filename", default="best.pt")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--backend", choices=("torch", "ncnn"), default="torch")
    parser.add_argument("--ncnn-param", default="")
    parser.add_argument("--wenet-onnx", required=True)
    parser.add_argument("--alpha-bin", default="")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--bbox-detector",
        choices=("mediapipe_face_landmarker", "mediapipe_face_mesh", "haar"),
        default="mediapipe_face_landmarker",
    )
    parser.add_argument("--landmark-model-asset-path", default="")
    parser.add_argument("--landmark-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--landmark-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--no-landmark-refine", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    artifacts = run_video_inference(
        input_video=args.input_video,
        audio=args.audio,
        out_dir=args.out_dir,
        checkpoint=args.ckpt,
        init_bin=args.init_bin,
        hf_model_repo=args.hf_model_repo,
        hf_model_filename=args.hf_model_filename,
        hf_cache_dir=args.hf_cache_dir,
        backend=args.backend,
        ncnn_param=args.ncnn_param,
        wenet_onnx=args.wenet_onnx,
        alpha_bin=args.alpha_bin,
        output_mp4=args.output_mp4,
        fps=args.fps,
        sample_rate=args.sample_rate,
        bbox_detector=args.bbox_detector,
        landmark_model_asset_path=args.landmark_model_asset_path or None,
        landmark_min_detection_confidence=args.landmark_min_detection_confidence,
        landmark_min_tracking_confidence=args.landmark_min_tracking_confidence,
        landmark_refine_landmarks=not args.no_landmark_refine,
        device=torch.device(args.device),
    )
    print(f"frames_dir={artifacts['frames_dir']}")
    print(f"metadata={artifacts['metadata_path']}")
    if "output_mp4_path" in artifacts:
        print(f"output_mp4={artifacts['output_mp4_path']}")


if __name__ == "__main__":
    main()
