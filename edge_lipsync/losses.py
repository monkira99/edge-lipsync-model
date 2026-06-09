from __future__ import annotations

import torch
import torch.nn.functional as F

from edge_lipsync.metrics import lpips_face_and_mouth
from edge_lipsync.preprocess import MASK_H, MASK_W, MASK_X, MASK_Y


def _per_sample_mean(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0:
        raise ValueError("Loss tensor must include a batch dimension")
    return values.flatten(start_dim=1).mean(dim=1)


def _weighted_mean(
    per_sample: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if per_sample.ndim != 1:
        raise ValueError(f"Expected per-sample loss [B], got {tuple(per_sample.shape)}")
    if sample_weight is None:
        return per_sample.mean()
    weights = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype)
    if weights.ndim != 1 or weights.shape[0] != per_sample.shape[0]:
        raise ValueError(
            f"sample_weight shape={tuple(weights.shape)} must match batch={per_sample.shape[0]}"
        )
    if not torch.all(torch.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values")
    if torch.any(weights < 0):
        raise ValueError("sample_weight must be non-negative")
    total_weight = weights.sum()
    if bool((total_weight <= 0).detach().cpu()):
        raise ValueError("sample_weight sum must be positive")
    return (per_sample * weights).sum() / total_weight


def _charbonnier_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    diff = pred - target
    return _per_sample_mean(torch.sqrt(diff * diff + eps * eps))


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return _weighted_mean(_charbonnier_per_sample(pred, target, eps), sample_weight)


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
    *,
    sample_weight: torch.Tensor | None = None,
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
    per_sample = _per_sample_mean(F.l1_loss(pred, target, reduction="none") * mask)
    return _weighted_mean(per_sample, sample_weight)


def _combined_reconstruction_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mouth_weight: float = 4.0,
    mouth_loss_scale: float = 0.5,
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
    mouth = _per_sample_mean(F.l1_loss(pred, target, reduction="none") * mask)
    return _charbonnier_per_sample(pred, target) + mouth_loss_scale * mouth


def combined_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mouth_weight: float = 4.0,
    mouth_loss_scale: float = 0.5,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return _weighted_mean(
        _combined_reconstruction_per_sample(
            pred,
            target,
            mouth_weight=mouth_weight,
            mouth_loss_scale=mouth_loss_scale,
        ),
        sample_weight,
    )


def combined_training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lpips_evaluator: torch.nn.Module | None = None,
    lpips_face_weight: float = 0.0,
    lpips_mouth_weight: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if lpips_face_weight < 0 or lpips_mouth_weight < 0:
        raise ValueError("LPIPS loss weights must be non-negative")
    reconstruction = _combined_reconstruction_per_sample(pred, target)
    if lpips_face_weight == 0 and lpips_mouth_weight == 0:
        return _weighted_mean(reconstruction, sample_weight)
    if lpips_evaluator is None:
        raise ValueError("LPIPS evaluator is required when LPIPS loss weights are positive")
    with torch.autocast(device_type=pred.device.type, enabled=False):
        face_lpips, mouth_lpips = lpips_face_and_mouth(
            lpips_evaluator,
            pred.float(),
            target.float(),
        )
        perceptual = lpips_face_weight * face_lpips + lpips_mouth_weight * mouth_lpips
    return _weighted_mean(reconstruction + perceptual.to(dtype=reconstruction.dtype), sample_weight)
