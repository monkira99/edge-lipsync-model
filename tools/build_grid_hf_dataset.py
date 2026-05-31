#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.grid_dataset import GridBuildConfig, build_grid_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a processed Duix dataset from GRID and optionally push it to Hugging Face."
        )
    )
    parser.add_argument("--grid-root", required=True, help="Path to the extracted GRID corpus root")
    parser.add_argument("--dataset-root", required=True, help="Output processed dataset directory")
    parser.add_argument(
        "--work-dir",
        default="",
        help="Intermediate working directory. Defaults beside dataset-root.",
    )
    parser.add_argument("--wenet-onnx", required=True, help="Path to Wenet ONNX model")
    parser.add_argument(
        "--landmark-model-asset-path",
        default=None,
        help="MediaPipe FaceLandmarker .task path",
    )
    parser.add_argument("--speaker", default="", help="GRID speaker id to include, e.g. s1")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Limit selected videos. Use 0 for all discovered videos.",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--bbox-detector", default="mediapipe_face_landmarker")
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately when any clip fails during dataset build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover GRID samples and print the plan without muxing, building, or pushing.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the built dataset to Hugging Face",
    )
    parser.add_argument("--hf-repo-id", default="", help="Hugging Face dataset repo id")
    privacy = parser.add_mutually_exclusive_group()
    privacy.add_argument("--private", action="store_true", help="Create/use a private dataset repo")
    privacy.add_argument("--public", action="store_true", help="Create/use a public dataset repo")
    parser.add_argument(
        "--commit-message",
        default="Upload GRID processed dataset snapshot",
        help="Commit message for Hugging Face upload",
    )
    args = parser.parse_args()

    if args.push and not args.dry_run and not args.hf_repo_id:
        parser.error("--hf-repo-id is required when --push is set")

    result = build_grid_dataset(
        GridBuildConfig(
            grid_root=args.grid_root,
            dataset_root=args.dataset_root,
            work_dir=args.work_dir,
            wenet_onnx=args.wenet_onnx,
            landmark_model_asset_path=args.landmark_model_asset_path,
            speaker=args.speaker,
            max_videos=args.max_videos,
            fps=args.fps,
            sample_rate=args.sample_rate,
            validation_fraction=args.validation_fraction,
            bbox_detector=args.bbox_detector,
            preview_count=args.preview_count,
            dry_run=args.dry_run,
            progress=not args.no_progress,
            push=args.push,
            hf_repo_id=args.hf_repo_id,
            private=not args.public,
            commit_message=args.commit_message,
            strict=args.strict,
        )
    )

    print(f"dry_run={result.dry_run}")
    print(f"grid_root={result.grid_root.resolve()}")
    print(f"dataset_root={result.dataset_root.resolve()}")
    print(f"work_dir={result.work_dir.resolve()}")
    print(f"raw_video_dir={result.raw_video_dir.resolve()}")
    print(f"sample_count={result.sample_count}")
    print(f"raw_video_count={result.raw_video_count}")
    if result.pushed_revision is not None:
        print(f"revision={result.pushed_revision}")
    if result.hub_url is not None:
        print(f"url={result.hub_url}")


if __name__ == "__main__":
    main()
