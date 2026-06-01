#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.hf_video_dataset import (  # noqa: E402
    DEFAULT_METADATA_MANIFEST,
    DEFAULT_VIDEO_PREFIX,
    HfVideoDatasetBuildConfig,
    build_hf_video_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Duix dataset from videos stored in a Hugging Face dataset."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id")
    parser.add_argument("--dataset-root", required=True, help="Output processed dataset directory")
    parser.add_argument("--work-dir", default="", help="Intermediate work directory")
    parser.add_argument("--cache-dir", default="", help="Hugging Face cache directory")
    parser.add_argument(
        "--download-max-workers",
        type=int,
        default=1,
        help="Maximum concurrent datasets file downloads. Keep at 1 to avoid rate limits.",
    )
    parser.add_argument("--video-prefix", default=DEFAULT_VIDEO_PREFIX)
    parser.add_argument("--metadata-manifest", default=DEFAULT_METADATA_MANIFEST)
    parser.add_argument("--metadata-prefix", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--speaker-id",
        default="",
        help="Only build clips whose metadata src_speaker and alt_speaker match this speaker id.",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List speaker ids and available clip counts without downloading or building.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Limit selected videos. Use 0 for every video under video-prefix.",
    )
    parser.add_argument("--wenet-onnx", required=True, help="Path to Wenet ONNX model")
    parser.add_argument(
        "--landmark-model-asset-path",
        default=None,
        help="MediaPipe FaceLandmarker .task path",
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
        help="List selected videos and print the plan without downloading, building, or pushing.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument("--push", action="store_true", help="Push processed dataset to HF")
    parser.add_argument(
        "--hf-output-repo-id",
        default="",
        help="HF dataset repo for processed output",
    )
    privacy = parser.add_mutually_exclusive_group()
    privacy.add_argument("--private", action="store_true", help="Create/use a private output repo")
    privacy.add_argument("--public", action="store_true", help="Create/use a public output repo")
    args = parser.parse_args()
    if args.push and not args.dry_run and not args.list_speakers and not args.hf_output_repo_id:
        parser.error("--hf-output-repo-id is required when --push is set")

    result = build_hf_video_dataset(
        HfVideoDatasetBuildConfig(
            repo_id=args.repo_id,
            dataset_root=args.dataset_root,
            work_dir=args.work_dir,
            cache_dir=args.cache_dir,
            download_max_workers=args.download_max_workers,
            video_prefix=args.video_prefix,
            metadata_manifest=args.metadata_manifest,
            speaker_id=args.speaker_id,
            list_speakers=args.list_speakers,
            max_videos=args.max_videos,
            wenet_onnx=args.wenet_onnx,
            landmark_model_asset_path=args.landmark_model_asset_path,
            fps=args.fps,
            sample_rate=args.sample_rate,
            validation_fraction=args.validation_fraction,
            bbox_detector=args.bbox_detector,
            preview_count=args.preview_count,
            dry_run=args.dry_run,
            progress=not args.no_progress,
            push=args.push,
            hf_output_repo_id=args.hf_output_repo_id,
            private=not args.public,
            strict=args.strict,
        )
    )

    print(f"dry_run={result.dry_run}")
    print(f"repo_id={result.repo_id}")
    print(f"dataset_root={result.dataset_root.resolve()}")
    print(f"work_dir={result.work_dir.resolve()}")
    print(f"raw_video_dir={result.raw_video_dir.resolve()}")
    print(f"selected_video_count={result.selected_video_count}")
    print(f"raw_video_count={result.raw_video_count}")
    if getattr(result, "speaker_id", ""):
        print(f"speaker_id={result.speaker_id}")
    speaker_counts = getattr(result, "speaker_counts", None) or {}
    if speaker_counts:
        print(f"speaker_count={len(speaker_counts)}")
    if args.list_speakers:
        speaker_rows = sorted(
            speaker_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        for speaker_id, count in speaker_rows:
            print(f"{speaker_id}\t{count}")
    if result.hub_url is not None:
        print(f"url={result.hub_url}")


if __name__ == "__main__":
    main()
