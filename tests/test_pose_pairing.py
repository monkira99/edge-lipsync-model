from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest


def _landmarks() -> dict[int, tuple[float, float]]:
    return {
        1: (100.0, 90.0),
        10: (100.0, 45.0),
        33: (70.0, 70.0),
        61: (82.0, 118.0),
        152: (100.0, 150.0),
        234: (55.0, 100.0),
        263: (130.0, 70.0),
        291: (118.0, 118.0),
        454: (145.0, 100.0),
    }


def test_head_pose_landmark_subset_excludes_lips() -> None:
    from edge_lipsync.pose_pairing import HEAD_POSE_LANDMARK_INDICES

    assert 61 not in HEAD_POSE_LANDMARK_INDICES
    assert 291 not in HEAD_POSE_LANDMARK_INDICES


def test_analyze_landmark_geometry_normalizes_center_and_size() -> None:
    from edge_lipsync.pose_pairing import normalized_bbox_geometry

    geometry = normalized_bbox_geometry((50, 40, 150, 160), (200, 200, 3))

    assert geometry.center_x == pytest.approx(0.5)
    assert geometry.center_y == pytest.approx(0.5)
    assert geometry.width == pytest.approx(0.5)
    assert geometry.height == pytest.approx(0.6)


def test_mouth_openness_is_normalized_by_mouth_width() -> None:
    from edge_lipsync.pose_pairing import mouth_openness

    landmarks = {
        **_landmarks(),
        13: (100.0, 112.0),
        14: (100.0, 120.0),
    }

    assert mouth_openness(landmarks) == pytest.approx(8.0 / 36.0)


def test_rotation_matrix_to_euler_returns_yaw_pitch_roll_degrees() -> None:
    from edge_lipsync.pose_pairing import rotation_matrix_to_euler

    rotation = np.eye(3, dtype=np.float64)

    pose = rotation_matrix_to_euler(rotation)

    assert pose.yaw == pytest.approx(0.0)
    assert pose.pitch == pytest.approx(0.0)
    assert pose.roll == pytest.approx(0.0)


def test_laplacian_variance_separates_sharp_and_blurred_images() -> None:
    from edge_lipsync.pose_pairing import laplacian_variance

    sharp = np.zeros((64, 64, 3), dtype=np.uint8)
    sharp[:, ::2] = 255
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

    assert laplacian_variance(sharp) > laplacian_variance(blurred)


def test_mark_bbox_continuity_rejects_jump_in_sequence() -> None:
    from edge_lipsync.pose_pairing import FrameObservation, mark_bbox_continuity

    observations = [
        FrameObservation(
            frame_idx=1,
            bbox_xyxy=(50, 50, 150, 150),
            frame_width=200,
            frame_height=200,
            landmarks={},
            pose=None,
            face_blur=100.0,
            mouth_blur=100.0,
            mouth_open=0.0,
            landmark_valid=True,
        ),
        FrameObservation(
            frame_idx=2,
            bbox_xyxy=(120, 50, 200, 130),
            frame_width=200,
            frame_height=200,
            landmarks={},
            pose=None,
            face_blur=100.0,
            mouth_blur=100.0,
            mouth_open=0.0,
            landmark_valid=True,
        ),
    ]

    result = mark_bbox_continuity(observations)

    assert result[0].bbox_continuity_valid is True
    assert result[1].bbox_continuity_valid is False
    assert result[1].reject_reason == "bbox_discontinuity"


def test_sync_search_finds_visual_lag() -> None:
    from edge_lipsync.pose_pairing import best_sync_lag

    audio = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    mouth = np.asarray([0, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)

    lag, correlation = best_sync_lag(audio, mouth, max_lag_frames=3)

    assert lag == 1
    assert correlation == pytest.approx(1.0)


def test_low_correlation_speech_window_does_not_reject_by_lag() -> None:
    from edge_lipsync.pose_pairing import SyncWindow, sync_reject_reason

    window = SyncWindow(
        window_id=0,
        start_frame=0,
        end_frame=50,
        center_frame=25.0,
        has_speech=True,
        best_lag_frames=3,
        best_correlation=0.10,
        confidence="low",
    )

    assert sync_reject_reason(window, min_correlation=0.20, max_abs_lag=2) is None


def test_nearest_sync_window_assigns_each_frame_once() -> None:
    from edge_lipsync.pose_pairing import SyncWindow, assign_sync_windows

    windows = [
        SyncWindow(0, 0, 50, 25.0, True, 0, 0.8, "high"),
        SyncWindow(1, 25, 75, 50.0, True, 1, 0.8, "high"),
    ]

    assignments = assign_sync_windows(frame_count=75, windows=windows)

    assert assignments[30].window_id == 0
    assert assignments[45].window_id == 1


def test_fill_missing_signal_interpolates_only_for_sync_analysis() -> None:
    from edge_lipsync.pose_pairing import fill_missing_signal

    values = np.asarray([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=np.float32)

    filled = fill_missing_signal(values)

    assert np.allclose(filled, [1.0, 1.0, 2.0, 3.0, 3.0])


def _observation(
    *,
    frame_idx: int,
    mouth_shift: float = 0.0,
    yaw_shift: float = 0.0,
) -> Any:
    from edge_lipsync.pose_pairing import FrameObservation, HeadPose

    landmarks = dict(_landmarks())
    for index in (61, 291):
        x, y = landmarks[index]
        landmarks[index] = (x + mouth_shift, y)
    return FrameObservation(
        frame_idx=frame_idx,
        bbox_xyxy=(50, 40, 150, 160),
        frame_width=200,
        frame_height=200,
        landmarks=landmarks,
        pose=HeadPose(yaw=yaw_shift, pitch=0.0, roll=0.0),
        face_blur=100.0,
        mouth_blur=100.0,
        mouth_open=0.2,
        landmark_valid=True,
    )


def test_post_crop_alignment_compares_normalized_roi_landmarks() -> None:
    from edge_lipsync.pose_pairing import post_crop_alignment

    source = _landmarks()
    target = {index: (x + 10.0, y + 20.0) for index, (x, y) in source.items()}

    result = post_crop_alignment(
        source,
        (50, 40, 150, 160),
        target,
        (60, 60, 160, 180),
    )

    assert result.stable_landmark_rmse == pytest.approx(0.0)
    assert result.mouth_center_delta == pytest.approx(0.0)


def test_matching_filters_alignment_before_scoring() -> None:
    from edge_lipsync.pose_pairing import MatchConfig, match_silent_observation

    target = _observation(frame_idx=10, mouth_shift=0.0)
    bad_best_pose = _observation(frame_idx=1, mouth_shift=20.0)
    valid_second = _observation(frame_idx=2, yaw_shift=1.0)

    result = match_silent_observation(
        target,
        [bad_best_pose, valid_second],
        MatchConfig(),
    )

    assert result.selected.frame_idx == 2
    assert result.valid_candidate_count == 1
    assert result.second_best_score is None


def test_matching_tie_breaks_by_silent_frame_index_and_reports_margin() -> None:
    from edge_lipsync.pose_pairing import MatchConfig, match_silent_observation

    target = _observation(frame_idx=10)
    earlier = _observation(frame_idx=1)
    later = _observation(frame_idx=2)

    result = match_silent_observation(target, [later, earlier], MatchConfig())

    assert result.selected.frame_idx == 1
    assert result.valid_candidate_count == 2
    assert result.second_best_score == pytest.approx(result.matching_score)
    assert result.matching_score_margin == pytest.approx(0.0)


def test_matching_reports_pose_geometry_no_match_before_alignment() -> None:
    from edge_lipsync.pose_pairing import MatchConfig, match_silent_observation

    target = _observation(frame_idx=10)
    source = _observation(frame_idx=1, yaw_shift=20.0)

    with pytest.raises(ValueError, match="pose_geometry_no_match"):
        match_silent_observation(target, [source], MatchConfig())


def test_matching_uses_shortest_angle_delta_for_roll_wraparound() -> None:
    from edge_lipsync.pose_pairing import (
        FrameObservation,
        HeadPose,
        MatchConfig,
        match_silent_observation,
    )

    def observation(frame_idx: int, roll: float) -> FrameObservation:
        return FrameObservation(
            frame_idx=frame_idx,
            bbox_xyxy=(50, 40, 150, 160),
            frame_width=200,
            frame_height=200,
            landmarks=_landmarks(),
            pose=HeadPose(yaw=0.0, pitch=0.0, roll=roll),
            face_blur=100.0,
            mouth_blur=100.0,
            mouth_open=0.2,
            landmark_valid=True,
        )

    result = match_silent_observation(
        observation(10, -179.0),
        [observation(1, 179.0)],
        MatchConfig(),
    )

    assert result.pose_delta.roll == pytest.approx(2.0)


def test_assign_video_splits_keeps_each_clip_in_one_split() -> None:
    from edge_lipsync.pose_pairing import assign_video_splits

    splits = assign_video_splits(
        "nora",
        ["a", "b", "c"],
        split_salt="edge-lipsync-v1",
        validation_fraction=0.2,
    )

    assert set(splits) == {"a", "b", "c"}
    assert set(splits.values()) == {"train", "val"}


def test_select_idle_frames_applies_strict_ten_percent_cap() -> None:
    from edge_lipsync.pose_pairing import select_idle_frame_indices

    selected = select_idle_frame_indices(
        idle_frame_indices=list(range(100)),
        speech_frame_indices=list(range(100, 200)),
        max_ratio=0.10,
    )

    assert len(selected) == 10
    assert selected == sorted(selected)
