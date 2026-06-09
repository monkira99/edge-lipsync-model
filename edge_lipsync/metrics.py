from __future__ import annotations

import importlib
import warnings

import torch
import torch.nn.functional as F

# Fixed lower-face crop calibrated for the Duix 160x160 face patch.
MOUTH_ROI = (32, 72, 96, 72)
NORMALIZED_RGB_DATA_RANGE = 2.0


class LPIPSEvaluator(torch.nn.Module):
    def __init__(self, device: torch.device, *, net: str = "alex") -> None:
        super().__init__()
        if net not in {"alex", "vgg", "squeeze"}:
            raise ValueError(f"Unsupported LPIPS network={net!r}")
        lpips = importlib.import_module("lpips")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The parameter 'pretrained' is deprecated.*",
                category=UserWarning,
                module=r"torchvision\.models\._utils",
            )
            warnings.filterwarnings(
                "ignore",
                message="Arguments other than a weight enum.*",
                category=UserWarning,
                module=r"torchvision\.models\._utils",
            )
            self.metric = lpips.LPIPS(net=net, version="0.1", verbose=False).to(device).eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _validate_pair(pred, target)
        distance = self.metric(
            pred.float().clamp(-1.0, 1.0),
            target.float().clamp(-1.0, 1.0),
        )
        return distance.reshape(pred.shape[0])


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.ndim != 4 or pred.shape[1] != 3:
        raise ValueError(f"Expected RGB tensors [B,3,H,W], got {tuple(pred.shape)}")


def crop_mouth(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"Expected RGB tensor [B,3,H,W], got {tuple(value.shape)}")
    x, y, width, height = MOUTH_ROI
    if value.shape[-2] < y + height or value.shape[-1] < x + width:
        raise ValueError(f"Tensor is too small for mouth ROI {MOUTH_ROI}: {tuple(value.shape)}")
    return value[:, :, y : y + height, x : x + width]


def image_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_pair(pred, target)
    return torch.mean(torch.abs(pred.float() - target.float()), dim=(1, 2, 3))


def image_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = NORMALIZED_RGB_DATA_RANGE,
) -> torch.Tensor:
    _validate_pair(pred, target)
    mse = torch.mean((pred.float() - target.float()) ** 2, dim=(1, 2, 3))
    peak = torch.tensor(data_range * data_range, device=mse.device, dtype=mse.dtype)
    return 10.0 * torch.log10(peak / mse.clamp_min(1e-12))


def image_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = NORMALIZED_RGB_DATA_RANGE,
    window_size: int = 11,
) -> torch.Tensor:
    _validate_pair(pred, target)
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    pred_float = pred.float()
    target_float = target.float()
    padding = window_size // 2
    mu_pred = F.avg_pool2d(pred_float, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target_float, window_size, stride=1, padding=padding)
    pred_var = F.avg_pool2d(pred_float * pred_float, window_size, 1, padding) - mu_pred**2
    target_var = F.avg_pool2d(target_float * target_float, window_size, 1, padding) - mu_target**2
    covariance = (
        F.avg_pool2d(pred_float * target_float, window_size, 1, padding) - mu_pred * mu_target
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_pred * mu_target + c1) * (2.0 * covariance + c2)
    denominator = (mu_pred**2 + mu_target**2 + c1) * (pred_var + target_var + c2)
    return torch.mean(numerator / denominator.clamp_min(1e-12), dim=(1, 2, 3))


def mouth_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return image_mae(crop_mouth(pred), crop_mouth(target))


def mouth_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return image_psnr(crop_mouth(pred), crop_mouth(target))


def mouth_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return image_ssim(crop_mouth(pred), crop_mouth(target))


def lpips_face_and_mouth(
    evaluator: torch.nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_pair(pred, target)
    face_distance = evaluator(pred, target).reshape(pred.shape[0])
    mouth_distance = evaluator(crop_mouth(pred), crop_mouth(target)).reshape(pred.shape[0])
    return face_distance, mouth_distance


def mouth_temporal_error(
    current_pred: torch.Tensor,
    current_target: torch.Tensor,
    previous_pred: torch.Tensor,
    previous_target: torch.Tensor,
) -> torch.Tensor:
    pred_motion = crop_mouth(current_pred) - crop_mouth(previous_pred)
    target_motion = crop_mouth(current_target) - crop_mouth(previous_target)
    return image_mae(pred_motion, target_motion)


def shift_audio_window(audio: torch.Tensor, frames: int = 5) -> torch.Tensor:
    if audio.ndim not in {3, 4}:
        raise ValueError(f"Expected audio [B,T,F] or [B,1,T,F], got {tuple(audio.shape)}")
    time_dim = 1 if audio.ndim == 3 else 2
    if audio.shape[time_dim] <= frames:
        raise ValueError(f"Audio window is too short for shift={frames}: {tuple(audio.shape)}")
    indices = torch.arange(audio.shape[time_dim], device=audio.device)
    indices = torch.clamp(indices + frames, max=audio.shape[time_dim] - 1)
    return torch.index_select(audio, time_dim, indices)
