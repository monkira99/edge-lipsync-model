#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.silent_talking_dataset import (  # noqa: E402
    build_config_from_mapping,
    build_silent_talking_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train-ready silent/talking pose-paired dataset snapshot."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset config must be a YAML mapping")
    if args.strict:
        payload["strict"] = True
    if args.no_progress:
        payload["progress"] = False
    result = build_silent_talking_dataset(build_config_from_mapping(payload))
    print(f"snapshot_root={result.snapshot_root.resolve()}")
    print(f"train_rows={result.train_rows}")
    print(f"val_rows={result.val_rows}")
    print(f"talking_clips={result.talking_clips}")
    print(f"failed_clips={len(result.failed_clips)}")
    print(f"config_sha256={result.config_sha256}")


if __name__ == "__main__":
    main()
