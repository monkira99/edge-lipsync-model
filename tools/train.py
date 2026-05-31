#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.training import TrainConfig, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DuixUNet from a manifest dataset.")
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Train config must be a YAML mapping")
    config = TrainConfig(**payload)
    best = train(config)
    print(f"best_checkpoint={best.resolve()}")


if __name__ == "__main__":
    main()
