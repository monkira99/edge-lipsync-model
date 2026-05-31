from __future__ import annotations

from pathlib import Path

import torch


def test_duix_unet_forward_shape() -> None:
    from edge_lipsync.model import DuixUNet

    model = DuixUNet().eval()
    face = torch.zeros(1, 6, 160, 160)
    audio = torch.zeros(1, 20, 256)

    with torch.no_grad():
        out = model(face, audio)

    assert tuple(out.shape) == (1, 3, 160, 160)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= -1.0
    assert float(out.max()) <= 1.0


def test_duix_unet_accepts_four_dim_audio() -> None:
    from edge_lipsync.model import DuixUNet

    model = DuixUNet().eval()
    face = torch.zeros(1, 6, 160, 160)
    audio = torch.zeros(1, 1, 20, 256)

    with torch.no_grad():
        out = model(face, audio)

    assert tuple(out.shape) == (1, 3, 160, 160)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    from edge_lipsync.model import DuixUNet, load_ckpt, save_ckpt

    ckpt_path = tmp_path / "model.pt"
    model = DuixUNet().eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 6, 160, 160), torch.zeros(1, 20, 256))

    save_ckpt(model, ckpt_path, face_size=160, extra={"test": True})
    loaded = load_ckpt(ckpt_path)

    assert isinstance(loaded, DuixUNet)
    assert ckpt_path.exists()
