from __future__ import annotations

import math

import pytest
import torch


class _MeanDistance(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target), dim=(1, 2, 3), keepdim=True)


def test_image_metrics_use_normalized_rgb_data_range() -> None:
    from edge_lipsync.metrics import image_mae, image_psnr, image_ssim

    pred = torch.zeros(2, 3, 32, 32)
    target = torch.ones(2, 3, 32, 32)

    assert image_mae(pred, target).tolist() == pytest.approx([1.0, 1.0])
    assert image_psnr(pred, target).tolist() == pytest.approx([10.0 * math.log10(4.0)] * 2)
    assert image_ssim(target, target).tolist() == pytest.approx([1.0, 1.0])


def test_mouth_metrics_only_measure_the_configured_roi() -> None:
    from edge_lipsync.metrics import MOUTH_ROI, mouth_mae

    pred = torch.zeros(1, 3, 160, 160)
    target = torch.zeros_like(pred)
    x, y, width, height = MOUTH_ROI
    target[:, :, y : y + height, x : x + width] = 1.0

    assert mouth_mae(pred, target).item() == pytest.approx(1.0)


def test_mouth_temporal_error_compares_prediction_and_target_motion() -> None:
    from edge_lipsync.metrics import MOUTH_ROI, mouth_temporal_error

    previous_pred = torch.zeros(1, 3, 160, 160)
    previous_target = torch.zeros_like(previous_pred)
    current_pred = torch.zeros_like(previous_pred)
    current_target = torch.zeros_like(previous_pred)
    x, y, width, height = MOUTH_ROI
    current_pred[:, :, y : y + height, x : x + width] = 0.5
    current_target[:, :, y : y + height, x : x + width] = 1.0

    error = mouth_temporal_error(
        current_pred,
        current_target,
        previous_pred,
        previous_target,
    )

    assert error.item() == pytest.approx(0.5)


def test_shift_audio_window_uses_edge_padding_instead_of_wrapping() -> None:
    from edge_lipsync.metrics import shift_audio_window

    audio = torch.arange(5, dtype=torch.float32).view(1, 5, 1)

    shifted = shift_audio_window(audio, frames=2)

    assert shifted.flatten().tolist() == [2.0, 3.0, 4.0, 4.0, 4.0]


def test_lpips_evaluator_freezes_model_and_returns_one_distance_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.metrics as metrics

    created: dict[str, object] = {}

    class FakeLPIPS(torch.nn.Module):
        def __init__(self, *, net: str, version: str, verbose: bool) -> None:
            super().__init__()
            created["net"] = net
            created["version"] = version
            created["verbose"] = verbose
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self.weight * torch.mean(
                torch.abs(pred - target),
                dim=(1, 2, 3),
                keepdim=True,
            )

    class FakeModule:
        LPIPS = FakeLPIPS

    monkeypatch.setattr(metrics.importlib, "import_module", lambda _name: FakeModule)

    evaluator = metrics.LPIPSEvaluator(torch.device("cpu"), net="alex")
    distances = evaluator(
        torch.zeros(2, 3, 32, 32),
        torch.ones(2, 3, 32, 32),
    )

    assert created == {"net": "alex", "version": "0.1", "verbose": False}
    assert distances.tolist() == pytest.approx([1.0, 1.0])
    assert not evaluator.training
    assert all(not parameter.requires_grad for parameter in evaluator.parameters())


def test_lpips_face_and_mouth_metrics_use_the_same_frozen_evaluator() -> None:
    from edge_lipsync.metrics import MOUTH_ROI, lpips_face_and_mouth

    pred = torch.zeros(1, 3, 160, 160)
    target = torch.zeros_like(pred)
    x, y, width, height = MOUTH_ROI
    target[:, :, y : y + height, x : x + width] = 1.0

    face, mouth = lpips_face_and_mouth(_MeanDistance(), pred, target)

    assert 0.0 < face.item() < 1.0
    assert mouth.item() == pytest.approx(1.0)
