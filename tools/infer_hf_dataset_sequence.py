#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.inference import run_hf_dataset_sequence_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference for a Hugging Face dataset sequence."
    )
    parser.add_argument("--hf-dataset-repo", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--clip-id", default="auto")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--init-bin", default="")
    parser.add_argument("--hf-model-repo", default="")
    parser.add_argument("--hf-model-filename", default="best.pt")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--backend", choices=("torch", "ncnn"), default="torch")
    parser.add_argument("--ncnn-param", default="")
    parser.add_argument("--audio-wav", default="")
    parser.add_argument("--alpha-bin", default="")
    parser.add_argument("--output-mp4", default="output.mp4")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    artifacts = run_hf_dataset_sequence_inference(
        hf_dataset_repo=args.hf_dataset_repo,
        split=args.split,
        clip_id=args.clip_id,
        max_frames=args.max_frames,
        checkpoint=args.ckpt,
        init_bin=args.init_bin,
        hf_model_repo=args.hf_model_repo,
        hf_model_filename=args.hf_model_filename,
        hf_cache_dir=args.hf_cache_dir,
        backend=args.backend,
        ncnn_param=args.ncnn_param,
        audio_wav=args.audio_wav,
        alpha_bin=args.alpha_bin,
        output_mp4=args.output_mp4,
        out_dir=args.out_dir,
        fps=args.fps,
        device=torch.device(args.device),
    )
    print(f"frames_dir={artifacts['frames_dir']}")
    print(f"metadata={artifacts['metadata_path']}")
    if "output_mp4_path" in artifacts:
        print(f"output_mp4={artifacts['output_mp4_path']}")


if __name__ == "__main__":
    main()
