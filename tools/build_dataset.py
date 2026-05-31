#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.build_dataset import DatasetBuildConfig, build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Duix training dataset from synchronized videos."
    )
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail the build immediately when any clip fails.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset config must be a YAML mapping")
    if args.no_progress:
        payload["progress"] = False
    build_dataset(DatasetBuildConfig(**payload), strict=args.strict)


if __name__ == "__main__":
    main()
