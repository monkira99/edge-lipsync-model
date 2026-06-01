#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.hf_datasets import load_processed_dataset, push_processed_dataset  # noqa: E402


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
    else:
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


if __name__ == "__main__":
    main()
