from __future__ import annotations

import numpy as np
import pytest


def test_make_face_training_sample_shapes_and_ranges() -> None:
    from edge_lipsync.preprocess import make_face_training_sample

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 100
    frame[:, :, 2] = 220

    sample = make_face_training_sample(frame, (80, 40, 240, 200))

    assert sample.face.shape == (6, 160, 160)
    assert sample.target.shape == (3, 160, 160)
    assert sample.roi_168_bgr.shape == (168, 168, 3)
    assert sample.face.dtype == np.float32
    assert sample.target.dtype == np.float32
    assert float(sample.face.min()) >= -1.0
    assert float(sample.face.max()) <= 1.0
    assert float(sample.target.min()) >= -1.0
    assert float(sample.target.max()) <= 1.0


def test_make_face_training_sample_uses_rgb_then_masked_rgb_channels() -> None:
    from edge_lipsync.preprocess import MASK_X, MASK_Y, make_face_training_sample

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 100
    frame[:, :, 2] = 220

    sample = make_face_training_sample(frame, (80, 40, 240, 200))

    assert sample.face[0, 0, 0] == pytest.approx((220.0 - 127.5) / 127.5)
    assert sample.face[1, 0, 0] == pytest.approx((100.0 - 127.5) / 127.5)
    assert sample.face[2, 0, 0] == pytest.approx((10.0 - 127.5) / 127.5)
    assert np.array_equal(sample.face[:3], sample.target)
    assert np.all(sample.face[3:, MASK_Y, MASK_X] == -1.0)


def test_make_face_training_sample_rejects_invalid_bbox() -> None:
    from edge_lipsync.preprocess import make_face_training_sample

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid bbox"):
        make_face_training_sample(frame, (50, 50, 50, 80))


def test_adjust_bbox_clips_to_frame() -> None:
    from edge_lipsync.preprocess import adjust_bbox

    box = adjust_bbox((10, 20, 110, 120), (100, 100, 3), dx=-20, dy=-30, scale=2.0)

    assert box == (0, 0, 100, 100)
