#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import cast

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from datasets import DatasetDict, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.hf_datasets import load_processed_dataset, push_processed_dataset  # noqa: E402
from edge_lipsync.hub import pull_dataset_snapshot, push_dataset_snapshot  # noqa: E402


def _snapshot_fingerprints(root: Path) -> dict[str, str]:
    dataset = cast(DatasetDict, load_from_disk(root / "dataset"))
    if set(dataset) != {"train", "val"}:
        raise ValueError("Dataset snapshot must contain train and val splits")
    return {str(split): str(split_dataset._fingerprint) for split, split_dataset in dataset.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage processed datasets on Hugging Face Hub.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    push = subparsers.add_parser("push", help="Upload a processed dataset")
    push.add_argument("--dataset-root", required=True)
    push.add_argument("--repo-id", required=True)
    push.add_argument("--public", action="store_true")
    push.add_argument("--manifest", default="manifest.jsonl")

    pull = subparsers.add_parser("pull", help="Load a processed dataset with datasets")
    pull.add_argument("--repo-id", required=True)
    pull.add_argument("--cache-dir", default="")
    pull.add_argument("--save-to-disk", default="")

    push_snapshot = subparsers.add_parser("push-snapshot", help="Upload a complete snapshot")
    push_snapshot.add_argument("--snapshot-root", required=True)
    push_snapshot.add_argument("--repo-id", required=True)
    push_snapshot.add_argument("--public", action="store_true")
    push_snapshot.add_argument("--workers", type=int, default=8)
    push_snapshot.add_argument("--include-reports", action="store_true")

    pull_snapshot = subparsers.add_parser(
        "pull-snapshot",
        help="Download and verify a revision-pinned snapshot",
    )
    pull_snapshot.add_argument("--repo-id", required=True)
    pull_snapshot.add_argument("--revision", required=True)
    pull_snapshot.add_argument("--local-dir", required=True)
    pull_snapshot.add_argument("--cache-dir", default="")
    pull_snapshot.add_argument("--workers", type=int, default=16)
    pull_snapshot.add_argument("--include-reports", action="store_true")

    args = parser.parse_args()
    if args.command == "push":
        artifact = push_processed_dataset(
            args.dataset_root,
            args.repo_id,
            private=not args.public,
            manifest=args.manifest,
        )
        print(f"repo_id={artifact.repo_id}")
        print(f"url={artifact.url}")
    elif args.command == "pull":
        dataset = load_processed_dataset(
            args.repo_id,
            cache_dir=args.cache_dir,
        )
        print(f"repo_id={args.repo_id}")
        print(f"splits={list(dataset.keys())}")
        for split, split_dataset in dataset.items():
            print(f"{split}_rows={split_dataset.num_rows}")
        if args.save_to_disk:
            dataset.save_to_disk(args.save_to_disk)
            print(f"path={Path(args.save_to_disk).resolve()}")
    elif args.command == "push-snapshot":
        artifact = push_dataset_snapshot(
            args.snapshot_root,
            args.repo_id,
            private=not args.public,
            include_reports=args.include_reports,
            workers=args.workers,
        )
        print(f"repo_id={artifact.repo_id}")
        print(f"revision={artifact.resolved_ref}")
        print(f"url={artifact.url}")
    elif args.command == "pull-snapshot":
        artifact = pull_dataset_snapshot(
            args.repo_id,
            ref=args.revision,
            local_dir=args.local_dir,
            cache_dir=args.cache_dir,
            include_reports=args.include_reports,
            workers=args.workers,
            verify=_snapshot_fingerprints,
        )
        print(f"repo_id={artifact.repo_id}")
        print(f"revision={artifact.resolved_ref}")
        if artifact.path is not None:
            print(f"path={artifact.path.resolve()}")
        print(f"url={artifact.url}")


if __name__ == "__main__":
    main()
