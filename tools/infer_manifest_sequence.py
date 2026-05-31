#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.inference import run_manifest_sequence_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference for a manifest sequence and write an end-to-end MP4."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--init-bin", default="")
    parser.add_argument("--hf-model-repo", default="")
    parser.add_argument("--hf-model-revision", default="")
    parser.add_argument("--hf-model-filename", default="best.pt")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--backend", choices=("torch", "ncnn"), default="torch")
    parser.add_argument("--ncnn-param", default="")
    parser.add_argument("--audio-wav", required=True)
    parser.add_argument("--wenet-onnx", required=True)
    parser.add_argument("--alpha-bin", default="")
    parser.add_argument("--output-mp4", default="output.mp4")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    artifacts = run_manifest_sequence_inference(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        out_dir=args.out_dir,
        split=None if args.split == "all" else args.split,
        max_frames=args.max_frames,
        checkpoint=args.ckpt,
        init_bin=args.init_bin,
        hf_model_repo=args.hf_model_repo,
        hf_model_revision=args.hf_model_revision,
        hf_model_filename=args.hf_model_filename,
        hf_cache_dir=args.hf_cache_dir,
        backend=args.backend,
        ncnn_param=args.ncnn_param,
        audio_wav=args.audio_wav,
        wenet_onnx=args.wenet_onnx,
        alpha_bin=args.alpha_bin,
        output_mp4=args.output_mp4,
        fps=args.fps,
        device=torch.device(args.device),
    )
    print(f"frames_dir={artifacts['frames_dir']}")
    print(f"output_mp4={artifacts['output_mp4_path']}")
    print(f"metadata={artifacts['metadata_path']}")


if __name__ == "__main__":
    main()
