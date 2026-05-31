#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.hub import pull_model_assets, push_model_assets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage reusable model assets on Hugging Face Hub."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    push = subparsers.add_parser("push", help="Upload the local models directory")
    push.add_argument("--models-root", default="models")
    push.add_argument("--repo-id", required=True)
    push.add_argument("--public", action="store_true")
    push.add_argument("--commit-message", default="Upload model assets")

    pull = subparsers.add_parser("pull", help="Download a pinned model asset snapshot")
    pull.add_argument("--repo-id", required=True)
    pull.add_argument("--revision", required=True)
    pull.add_argument("--local-dir", default="models")
    pull.add_argument("--cache-dir", default="")

    args = parser.parse_args()
    if args.command == "push":
        artifact = push_model_assets(
            args.models_root,
            args.repo_id,
            private=not args.public,
            commit_message=args.commit_message,
        )
    else:
        artifact = pull_model_assets(
            args.repo_id,
            revision=args.revision,
            local_dir=args.local_dir,
            cache_dir=args.cache_dir,
        )
    print(f"repo_id={artifact.repo_id}")
    print(f"revision={artifact.resolved_revision}")
    print(f"url={artifact.url}")
    if artifact.path is not None:
        print(f"path={artifact.path.resolve()}")


if __name__ == "__main__":
    main()
