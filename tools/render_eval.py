#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.dataset import DuixManifestDataset  # noqa: E402
from edge_lipsync.eval import RenderEvalConfig, render_validation_artifacts  # noqa: E402
from edge_lipsync.model import load_ckpt  # noqa: E402


def _load_config(args: argparse.Namespace) -> RenderEvalConfig:
    payload: dict[str, Any] = {}
    if args.config:
        loaded = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Eval config must be a YAML mapping")
        payload.update(loaded)
    for field in ("dataset_root", "manifest", "ckpt", "out_dir", "max_batches", "device", "fps"):
        value = getattr(args, field)
        if value is not None:
            payload[field] = value
    return RenderEvalConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render validation grids and MP4 for a Duix checkpoint."
    )
    parser.add_argument("--config", help="Optional path to eval YAML config")
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest")
    parser.add_argument("--ckpt")
    parser.add_argument("--out-dir")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device")
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()
    config = _load_config(args)

    dataset = DuixManifestDataset(config.dataset_root, config.manifest, split="val")
    model = load_ckpt(config.ckpt, map_location=config.device)
    artifacts = render_validation_artifacts(
        model=model,
        dataset=dataset,
        out_dir=config.out_dir,
        checkpoint_path=config.ckpt,
        device=torch.device(config.device),
        max_batches=config.max_batches,
        fps=config.fps,
    )
    print(f"video={artifacts['video_path']}")
    print(f"metadata={artifacts['metadata_path']}")
    print(f"grids={len(artifacts['grid_paths'])}")
    print(f"metrics={artifacts['metrics']}")


if __name__ == "__main__":
    main()
