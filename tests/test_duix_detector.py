from __future__ import annotations

import subprocess
import sys

import numpy as np


def test_pfpld_98_to_68_matches_historical_duix_index_mapping() -> None:
    from edge_lipsync.duix_detector import pfpld_98_to_68

    points = np.asarray([(float(index), float(100 + index)) for index in range(98)])

    converted = pfpld_98_to_68(points)

    assert converted.shape == (68, 2)
    assert np.array_equal(converted[0], points[0])
    assert np.array_equal(converted[16], points[32])
    assert np.array_equal(converted[17], points[33])
    assert np.array_equal(converted[27], points[51])
    assert np.array_equal(converted[37], (points[60] + points[62]) / 2.0)
    assert np.array_equal(converted[41], (points[60] + points[66]) / 2.0)
    assert np.array_equal(converted[47], (points[68] + points[74]) / 2.0)
    assert np.array_equal(converted[48], points[76])
    assert np.array_equal(converted[67], points[95])


def test_historical_duix_roi_from_pfpld_landmarks_matches_integer_crop_math() -> None:
    from edge_lipsync.duix_detector import historical_duix_roi_from_pfpld_landmarks

    points = np.full((68, 2), (120.0, 140.0), dtype=np.float32)
    points[0] = (100.0, 100.0)
    points[15] = (248.0, 260.0)
    points[67] = (260.0, 300.0)

    roi = historical_duix_roi_from_pfpld_landmarks(points, (960, 540, 3))

    assert roi == (104, 157, 264, 317)


def test_expand_historical_scrfd_bbox_matches_asymmetric_source_math() -> None:
    from edge_lipsync.duix_detector import expand_historical_scrfd_bbox

    expanded = expand_historical_scrfd_bbox((100, 200, 300, 500), (960, 540, 3))

    assert expanded == (80, 200, 322, 530)


def test_compare_duix_detector_bbox_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/compare_duix_detector_bbox.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Compare the historical Duix SCRFD+PFPLD detector" in result.stdout
    assert "--oracle-bbox-json" in result.stdout
    assert "--pfpld-channel-order" in result.stdout
