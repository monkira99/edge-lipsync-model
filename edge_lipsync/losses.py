from __future__ import annotations

import torch
import torch.nn.functional as F

from edge_lipsync.metrics import lpips_face_and_mouth
from edge_lipsync.preprocess import MASK_H, MASK_W, MASK_X, MASK_Y


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def mouth_weight_mask(
    device: torch.device,
    dtype: torch.dtype,
    height: int = 160,
    width: int = 160,
    mouth_weight: float = 4.0,
) -> torch.Tensor:
    mask = torch.ones(1, 1, height, width, device=device, dtype=dtype)
    mask[:, :, MASK_Y : MASK_Y + MASK_H, MASK_X : MASK_X + MASK_W] = float(mouth_weight)
    return mask


def mouth_weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mouth_weight: float = 4.0,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    mask = mouth_weight_mask(
        pred.device,
        pred.dtype,
        height=pred.shape[-2],
        width=pred.shape[-1],
        mouth_weight=mouth_weight,
    )
    return (F.l1_loss(pred, target, reduction="none") * mask).mean()


def combined_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mouth_weight: float = 4.0,
    mouth_loss_scale: float = 0.5,
) -> torch.Tensor:
    return charbonnier_loss(pred, target) + mouth_loss_scale * mouth_weighted_l1(
        pred,
        target,
        mouth_weight=mouth_weight,
    )


def combined_training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lpips_evaluator: torch.nn.Module | None = None,
    lpips_face_weight: float = 0.0,
    lpips_mouth_weight: float = 0.0,
) -> torch.Tensor:
    if lpips_face_weight < 0 or lpips_mouth_weight < 0:
        raise ValueError("LPIPS loss weights must be non-negative")
    reconstruction = combined_reconstruction_loss(pred, target)
    if lpips_face_weight == 0 and lpips_mouth_weight == 0:
        return reconstruction
    if lpips_evaluator is None:
        raise ValueError("LPIPS evaluator is required when LPIPS loss weights are positive")
    with torch.autocast(device_type=pred.device.type, enabled=False):
        face_lpips, mouth_lpips = lpips_face_and_mouth(
            lpips_evaluator,
            pred.float(),
            target.float(),
        )
        perceptual = lpips_face_weight * face_lpips.mean() + lpips_mouth_weight * mouth_lpips.mean()
    return reconstruction + perceptual.to(dtype=reconstruction.dtype)
