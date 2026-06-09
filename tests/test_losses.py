from __future__ import annotations

import pytest
import torch


class _MeanDistance(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target), dim=(1, 2, 3), keepdim=True)


def test_charbonnier_loss_backward() -> None:
    from edge_lipsync.losses import charbonnier_loss

    pred = torch.zeros(2, 3, 160, 160, requires_grad=True)
    target = torch.ones(2, 3, 160, 160)
    loss = charbonnier_loss(pred, target)
    loss.backward()

    assert loss.item() > 0
    assert pred.grad is not None


def test_mouth_weighted_loss_is_larger_for_mouth_error() -> None:
    from edge_lipsync.losses import mouth_weighted_l1

    pred = torch.zeros(1, 3, 160, 160)
    target = torch.zeros(1, 3, 160, 160)
    target[:, :, 20:120, 20:120] = 1.0

    weighted = mouth_weighted_l1(pred, target, mouth_weight=4.0)
    plain = torch.nn.functional.l1_loss(pred, target)

    assert weighted.item() > plain.item()


def test_mouth_weight_mask_matches_duix_mask_rectangle() -> None:
    from edge_lipsync.losses import mouth_weight_mask

    mask = mouth_weight_mask(torch.device("cpu"), torch.float32, mouth_weight=4.0)

    assert tuple(mask.shape) == (1, 1, 160, 160)
    assert mask[0, 0, 4, 4] == 1.0
    assert mask[0, 0, 5, 5] == 4.0
    assert mask[0, 0, 149, 154] == 4.0
    assert mask[0, 0, 150, 155] == 1.0


def test_combined_training_loss_adds_weighted_face_and_mouth_lpips() -> None:
    from edge_lipsync.losses import combined_reconstruction_loss, combined_training_loss

    pred = torch.zeros(1, 3, 160, 160, requires_grad=True)
    target = torch.ones_like(pred)
    reconstruction = combined_reconstruction_loss(pred, target)

    loss = combined_training_loss(
        pred,
        target,
        lpips_evaluator=_MeanDistance(),
        lpips_face_weight=0.01,
        lpips_mouth_weight=0.05,
    )
    loss.backward()

    assert loss.item() == pytest.approx(reconstruction.item() + 0.06)
    assert pred.grad is not None
    assert torch.count_nonzero(pred.grad).item() > 0


def test_combined_training_loss_requires_evaluator_for_positive_lpips_weight() -> None:
    from edge_lipsync.losses import combined_training_loss

    pred = torch.zeros(1, 3, 160, 160)
    target = torch.ones_like(pred)

    with pytest.raises(ValueError, match="LPIPS evaluator"):
        combined_training_loss(
            pred,
            target,
            lpips_mouth_weight=0.05,
        )
