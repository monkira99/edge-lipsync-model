#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.checkpoint import atomic_torch_save, make_export_checkpoint  # noqa: E402
from edge_lipsync.model import DuixUNet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Duix NCNN dh_model.bin to PyTorch checkpoint."
    )
    parser.add_argument("--init-bin", required=True, help="Path to decrypted dh_model.bin")
    parser.add_argument("--out", required=True, help="Output PyTorch checkpoint")
    parser.add_argument("--face-size", type=int, default=160)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    init_bin = Path(args.init_bin)
    if not init_bin.exists():
        raise FileNotFoundError(init_bin)

    model = DuixUNet().to(args.device).eval()
    stats = model.load_ncnn_bin(init_bin, face_size=args.face_size, device=args.device)
    if int(stats["remaining_bytes"]) != 0:
        raise ValueError(f"NCNN bin had remaining bytes after load: {stats}")
    atomic_torch_save(
        make_export_checkpoint(
            model=model,
            face_size=args.face_size,
            init_bin=init_bin,
            weight_load=stats,
        ),
        args.out,
    )

    with torch.no_grad():
        face = torch.zeros(1, 6, args.face_size, args.face_size, device=args.device)
        audio = torch.zeros(1, 20, 256, device=args.device)
        pred = model(face, audio)

    print(f"saved={Path(args.out).resolve()}")
    print(f"weight_load={stats}")
    print(f"sanity_shape={tuple(pred.shape)}")


if __name__ == "__main__":
    main()
