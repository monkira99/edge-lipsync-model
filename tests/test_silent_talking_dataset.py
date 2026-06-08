from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest


def test_build_config_resolves_persona_layout(tmp_path: Path) -> None:
    from edge_lipsync.silent_talking_dataset import SilentTalkingBuildConfig

    config = SilentTalkingBuildConfig(
        data_root=str(tmp_path),
        persona_id="nora",
        snapshot_root=str(tmp_path / "snapshot"),
        work_root=str(tmp_path / "work"),
        wenet_onnx=str(tmp_path / "wenet.onnx"),
    )

    assert config.silent_video_path == tmp_path / "nora/silent/defaultvideo.mp4"
    assert config.talking_video_dir == tmp_path / "nora/talking"


def test_discover_talking_videos_is_sorted(tmp_path: Path) -> None:
    from edge_lipsync.silent_talking_dataset import discover_talking_videos

    talking = tmp_path / "nora/talking"
    talking.mkdir(parents=True)
    (talking / "b.mp4").write_bytes(b"b")
    (talking / "a.mp4").write_bytes(b"a")

    assert [path.name for path in discover_talking_videos(talking)] == ["a.mp4", "b.mp4"]


def test_analyze_frames_writes_reusable_jsonl_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "000001.png"), np.full((200, 200, 3), 120, dtype=np.uint8))

    class FakeDetector:
        def detect_landmarks(self, frame_bgr: np.ndarray) -> dict[int, tuple[float, float]]:
            del frame_bgr
            return _builder_landmarks()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        builder,
        "estimate_head_pose",
        lambda _landmarks, _shape: builder.HeadPose(0.0, 0.0, 0.0),
    )
    cache = tmp_path / "analysis.jsonl"

    observations = builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        cache_path=cache,
        cache_metadata={"config_sha256": "abc", "input_sha256": "def"},
        is_target=True,
        show_progress=False,
    )

    assert len(observations) == 1
    assert observations[0].landmark_valid is True
    assert cache.is_file()
    assert (tmp_path / "analysis.meta.json").is_file()


def test_analyze_frames_rebuilds_when_cache_metadata_changes(
    tmp_path: Path,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "000001.png"), np.full((200, 200, 3), 120, dtype=np.uint8))
    calls = 0

    class FakeDetector:
        def detect_landmarks(
            self,
            frame_bgr: np.ndarray,
        ) -> dict[int, tuple[float, float]] | None:
            del frame_bgr
            nonlocal calls
            calls += 1
            return None

        def close(self) -> None:
            pass

    cache = tmp_path / "analysis.jsonl"
    first = {"config_sha256": "a", "input_sha256": "x"}
    second = {"config_sha256": "b", "input_sha256": "x"}
    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        cache_path=cache,
        cache_metadata=first,
        is_target=True,
        show_progress=False,
    )
    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        cache_path=cache,
        cache_metadata=first,
        is_target=True,
        show_progress=False,
    )
    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        cache_path=cache,
        cache_metadata=second,
        is_target=True,
        show_progress=False,
    )

    assert calls == 2


def _builder_landmarks() -> dict[int, tuple[float, float]]:
    return {
        1: (100.0, 90.0),
        10: (100.0, 45.0),
        13: (100.0, 112.0),
        14: (100.0, 120.0),
        33: (70.0, 70.0),
        61: (82.0, 118.0),
        152: (100.0, 150.0),
        234: (55.0, 100.0),
        263: (130.0, 70.0),
        291: (118.0, 118.0),
        454: (145.0, 100.0),
    }


def _valid_observation(frame_idx: int):
    from edge_lipsync.pose_pairing import FrameObservation, HeadPose

    return FrameObservation(
        frame_idx=frame_idx,
        bbox_xyxy=(50, 40, 150, 160),
        frame_width=200,
        frame_height=200,
        landmarks=_builder_landmarks(),
        pose=HeadPose(0.0, 0.0, 0.0),
        face_blur=100.0,
        mouth_blur=100.0,
        mouth_open=0.2,
        landmark_valid=True,
    )


def _invalid_observation(frame_idx: int, reason: str):
    from dataclasses import replace

    return replace(
        _valid_observation(frame_idx),
        bbox_xyxy=None,
        pose=None,
        landmarks={},
        landmark_valid=False,
        reject_reason=reason,
    )


def _test_config():
    from edge_lipsync.silent_talking_dataset import SilentTalkingBuildConfig

    return SilentTalkingBuildConfig(
        data_root="data",
        persona_id="nora",
        snapshot_root="snapshot",
        work_root="work",
        wenet_onnx="wenet.onnx",
        progress=False,
    )


def test_build_pair_decisions_uses_exact_precomputed_bnf_row() -> None:
    from edge_lipsync.silent_talking_dataset import build_pair_decisions

    bnf = np.arange(3 * 20 * 256, dtype=np.float32).reshape(3, 20, 256)
    result = build_pair_decisions(
        talking_observations=[_valid_observation(1)],
        silent_observations=[_valid_observation(7)],
        bnf_windows=bnf,
        audio_rms=np.ones(3, dtype=np.float32),
        config=_test_config(),
        split_for_frame=lambda _frame_idx: "train",
    )

    assert len(result.rows) == 1
    assert result.rows[0]["audio_idx"] == 0
    assert np.array_equal(result.rows[0]["audio"], bnf[0])


def test_build_pair_decisions_records_every_talking_frame() -> None:
    from edge_lipsync.silent_talking_dataset import build_pair_decisions

    result = build_pair_decisions(
        talking_observations=[
            _invalid_observation(1, "landmark_missing"),
            _valid_observation(2),
        ],
        silent_observations=[_valid_observation(7)],
        bnf_windows=np.zeros((2, 20, 256), dtype=np.float32),
        audio_rms=np.ones(2, dtype=np.float32),
        config=_test_config(),
        split_for_frame=lambda _frame_idx: "train",
    )

    assert len(result.decisions) == 2
    assert result.decisions[0]["reject_reason"] == "landmark_missing"
    assert result.decisions[1]["status"] == "retained"


def test_post_crop_mismatch_report_counts_stable_and_mouth_failures() -> None:
    from dataclasses import replace

    from edge_lipsync.silent_talking_dataset import build_pair_decisions, quality_report

    stable_landmarks = dict(_builder_landmarks())
    x, y = stable_landmarks[1]
    stable_landmarks[1] = (x + 20.0, y)
    mouth_landmarks = dict(_builder_landmarks())
    for index in (61, 291):
        x, y = mouth_landmarks[index]
        mouth_landmarks[index] = (x + 20.0, y)

    result = build_pair_decisions(
        talking_observations=[_valid_observation(1)],
        silent_observations=[
            replace(_valid_observation(7), landmarks=stable_landmarks),
            replace(_valid_observation(8), landmarks=mouth_landmarks),
        ],
        bnf_windows=np.zeros((1, 20, 256), dtype=np.float32),
        audio_rms=np.ones(1, dtype=np.float32),
        config=_test_config(),
        split_for_frame=lambda _frame_idx: "train",
    )
    report = quality_report(
        clip_id="talk",
        talking_observations=[_valid_observation(1)],
        decisions=result.decisions,
        rows=result.rows,
    )

    assert result.decisions[0]["reject_reason"] == "post_crop_alignment_mismatch"
    assert report["post_crop_alignment_mismatch_stable_landmark"] == 1
    assert report["post_crop_alignment_mismatch_mouth_center"] == 1


def _complete_row(split: str) -> dict[str, object]:
    image = np.full((168, 168, 3), 120, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return {
        "schema_version": "edge_lipsync_silent_talking_pair_v1",
        "persona_id": "nora",
        "pair_id": f"{split}-pair",
        "talking_clip_id": f"{split}-clip",
        "source_frame_idx": 1,
        "target_frame_idx": 1,
        "audio_idx": 0,
        "source_roi": {"bytes": encoded.tobytes(), "path": None},
        "target_roi": {"bytes": encoded.tobytes(), "path": None},
        "audio": np.zeros((20, 256), dtype=np.float32),
        "source_bbox_xyxy": [50, 40, 150, 160],
        "target_bbox_xyxy": [50, 40, 150, 160],
        "source_frame_width": 200,
        "source_frame_height": 200,
        "target_frame_width": 200,
        "target_frame_height": 200,
        "sample_weight": 1.0,
        "is_idle": False,
        "sync_best_lag_frames": 0,
        "sync_correlation": 0.8,
        "sync_confidence": "high",
        "pose_delta_yaw": 0.0,
        "pose_delta_pitch": 0.0,
        "pose_delta_roll": 0.0,
        "center_delta_x": 0.0,
        "center_delta_y": 0.0,
        "width_ratio": 1.0,
        "height_ratio": 1.0,
        "stable_landmark_alignment_rmse": 0.0,
        "mouth_center_delta_after_crop": 0.0,
        "matching_score": 0.0,
        "valid_silent_candidate_count": 1,
        "second_best_matching_score": None,
        "matching_score_margin": None,
        "source_face_blur": 100.0,
        "target_face_blur": 100.0,
        "target_mouth_blur": 100.0,
        "flags": [],
        "split": split,
    }


def test_dataset_snapshot_roundtrip_keeps_embedded_png_bytes(tmp_path: Path) -> None:
    from datasets import DatasetDict, Image, load_from_disk

    from edge_lipsync.silent_talking_dataset import build_dataset_dict

    rows = [_complete_row("train"), _complete_row("val")]
    dataset = build_dataset_dict(rows)
    path = tmp_path / "dataset"
    dataset.save_to_disk(path)

    loaded = cast(DatasetDict, load_from_disk(path))
    physical = cast(
        dict[str, Any],
        loaded["train"].cast_column("source_roi", Image(decode=False))[0]["source_roi"],
    )

    assert physical["path"] is None
    assert physical["bytes"].startswith(b"\x89PNG")
    assert np.asarray(loaded["train"][0]["source_roi"]).shape == (168, 168, 3)


def test_build_silent_talking_dataset_writes_complete_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    data_root = tmp_path / "data"
    silent = data_root / "nora/silent/defaultvideo.mp4"
    talking = data_root / "nora/talking/talk.mp4"
    silent.parent.mkdir(parents=True)
    talking.parent.mkdir(parents=True)
    silent.write_bytes(b"silent")
    talking.write_bytes(b"talk")
    wenet = tmp_path / "wenet.onnx"
    wenet.write_bytes(b"wenet")

    def fake_build_contents(
        _config: object,
        temporary_root: Path,
    ) -> dict[str, object]:
        dataset = builder.build_dataset_dict([_complete_row("train"), _complete_row("val")])
        dataset.save_to_disk(temporary_root / "dataset")
        builder.write_frame_decisions(
            temporary_root / "reports/quality/talk_frame_decisions.parquet",
            [{"frame_idx": 1, "status": "retained", "reject_reason": None}],
        )
        return {
            "train_rows": 1,
            "val_rows": 1,
            "talking_clips": 1,
            "failed_clips": [],
            "dataset_fingerprints": {
                split: str(value._fingerprint) for split, value in dataset.items()
            },
        }

    monkeypatch.setattr(builder, "_build_snapshot_contents", fake_build_contents)

    snapshot = tmp_path / "snapshot"
    result = builder.build_silent_talking_dataset(
        builder.SilentTalkingBuildConfig(
            data_root=str(data_root),
            persona_id="nora",
            snapshot_root=str(snapshot),
            work_root=str(tmp_path / "work"),
            wenet_onnx=str(wenet),
            progress=False,
        )
    )

    assert result.snapshot_root == snapshot
    assert (snapshot / "dataset/dataset_dict.json").is_file()
    assert (snapshot / "reports/quality/talk_frame_decisions.parquet").is_file()
    assert (snapshot / "build_metadata.json").is_file()
    assert (snapshot / "build_complete.json").is_file()


def test_build_config_from_mapping_maps_nested_thresholds() -> None:
    from edge_lipsync.silent_talking_dataset import build_config_from_mapping

    config = build_config_from_mapping(
        {
            "data_root": "data",
            "persona_id": "nora",
            "snapshot_root": "snapshot",
            "work_root": "work",
            "wenet_onnx": "wenet.onnx",
            "matching": {"max_yaw_delta": 7.0},
            "post_crop_alignment": {"max_mouth_center_delta": 0.02},
            "blur": {"min_target_mouth_laplacian_variance": 55.0},
            "sync": {"max_reject_lag_frames": 1},
        }
    )

    assert config.match.max_yaw_delta == pytest.approx(7.0)
    assert config.match.max_mouth_center_delta == pytest.approx(0.02)
    assert config.blur.min_target_mouth_laplacian_variance == pytest.approx(55.0)
    assert config.sync.max_reject_lag_frames == 1


def test_build_silent_talking_dataset_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_silent_talking_dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "pose-paired" in result.stdout
