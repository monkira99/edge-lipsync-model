from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
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

    class FakeIdentityRuntime:
        def embed(
            self,
            frame_bgr: np.ndarray,
            landmarks: Mapping[int, tuple[float, float]],
        ) -> np.ndarray:
            del frame_bgr, landmarks
            return np.eye(1, 512, 0, dtype=np.float32)[0]

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
        identity_runtime=FakeIdentityRuntime(),
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

    class FakeIdentityRuntime:
        def embed(
            self,
            frame_bgr: np.ndarray,
            landmarks: Mapping[int, tuple[float, float]],
        ) -> np.ndarray:
            del frame_bgr, landmarks
            raise AssertionError("Invalid frames must not run ArcFace")

    cache = tmp_path / "analysis.jsonl"
    first = {"config_sha256": "a", "input_sha256": "x"}
    second = {"config_sha256": "b", "input_sha256": "x"}
    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        identity_runtime=FakeIdentityRuntime(),
        cache_path=cache,
        cache_metadata=first,
        is_target=True,
        show_progress=False,
    )
    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        identity_runtime=FakeIdentityRuntime(),
        cache_path=cache,
        cache_metadata=first,
        is_target=True,
        show_progress=False,
    )
    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        identity_runtime=FakeIdentityRuntime(),
        cache_path=cache,
        cache_metadata=second,
        is_target=True,
        show_progress=False,
    )

    assert calls == 2


def test_analyze_frames_reports_progress_on_cache_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "000001.png"), np.full((20, 20, 3), 120, dtype=np.uint8))
    progress_calls: list[dict[str, Any]] = []

    class FakeDetector:
        def detect_landmarks(self, frame_bgr: np.ndarray) -> None:
            del frame_bgr
            return None

        def close(self) -> None:
            pass

    class FakeIdentityRuntime:
        def embed(
            self,
            frame_bgr: np.ndarray,
            landmarks: Mapping[int, tuple[float, float]],
        ) -> np.ndarray:
            del frame_bgr, landmarks
            raise AssertionError("Invalid frames must not run ArcFace")

    def fake_progress(iterable: object, **kwargs: Any) -> object:
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(builder, "progress", fake_progress)

    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        identity_runtime=FakeIdentityRuntime(),
        cache_path=tmp_path / "analysis.jsonl",
        cache_metadata={"input": "fixture"},
        is_target=False,
        show_progress=True,
    )

    assert progress_calls == [
        {
            "enabled": True,
            "desc": "analyze silent",
            "total": 1,
            "unit": "frame",
        }
    ]


def _builder_landmarks() -> dict[int, tuple[float, float]]:
    return {
        1: (100.0, 90.0),
        10: (100.0, 45.0),
        13: (100.0, 112.0),
        14: (100.0, 120.0),
        33: (70.0, 70.0),
        133: (90.0, 70.0),
        61: (82.0, 118.0),
        152: (100.0, 150.0),
        234: (55.0, 100.0),
        362: (110.0, 70.0),
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
        identity_embedding=np.eye(1, 512, 0, dtype=np.float32)[0],
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


def test_observation_cache_roundtrips_identity_embedding() -> None:
    from edge_lipsync.silent_talking_dataset import observation_from_json, observation_to_json

    original = _valid_observation(3)

    restored = observation_from_json(observation_to_json(original))

    assert restored.identity_embedding is not None
    assert original.identity_embedding is not None
    assert restored.identity_embedding.dtype == np.float32
    assert np.array_equal(restored.identity_embedding, original.identity_embedding)


def test_build_pair_decisions_reports_identity_mismatch() -> None:
    from dataclasses import replace

    from edge_lipsync.silent_talking_dataset import build_pair_decisions, quality_report

    wrong_identity = np.eye(1, 512, 1, dtype=np.float32)[0]
    result = build_pair_decisions(
        talking_observations=[_valid_observation(1)],
        silent_observations=[
            replace(_valid_observation(7), identity_embedding=wrong_identity),
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

    assert result.rows == []
    assert result.decisions[0]["reject_reason"] == "identity_mismatch"
    assert result.decisions[0]["identity_mismatch_candidate_count"] == 1
    assert result.decisions[0]["identity_max_rejected_similarity"] == pytest.approx(0.0)
    assert report["identity_mismatch_count"] == 1


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
        "schema_version": "edge_lipsync_silent_talking_pair_v2",
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
        "identity_similarity": 0.9,
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


def test_dataset_shards_assemble_without_collecting_all_rows(tmp_path: Path) -> None:
    from edge_lipsync.silent_talking_dataset import (
        ClipBuildResult,
        build_dataset_dict_from_clip_results,
        write_dataset_shards,
    )

    train_row = {**_complete_row("train"), "pair_id": "clip-a:1"}
    val_row = {**_complete_row("val"), "pair_id": "clip-b:1"}
    train_shards, train_counts = write_dataset_shards(
        tmp_path / "shards/clip-a",
        [train_row],
    )
    val_shards, val_counts = write_dataset_shards(
        tmp_path / "shards/clip-b",
        [val_row],
    )

    dataset = build_dataset_dict_from_clip_results(
        [
            ClipBuildResult(
                clip_id="clip-b",
                dataset_shards=val_shards,
                row_counts=val_counts,
            ),
            ClipBuildResult(
                clip_id="clip-a",
                dataset_shards=train_shards,
                row_counts=train_counts,
            ),
        ]
    )

    assert dataset["train"]["pair_id"] == ["clip-a:1"]
    assert dataset["val"]["pair_id"] == ["clip-b:1"]
    assert np.asarray(dataset["train"][0]["source_roi"]).shape == (168, 168, 3)


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
            "identity": {
                "source": "local",
                "sha256": "arcface-sha",
                "license": "insightface-non-commercial-research",
                "min_cosine_similarity": 0.35,
            },
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
    metadata = json.loads((snapshot / "build_metadata.json").read_text(encoding="utf-8"))
    assert metadata["identity"]["sha256"] == "arcface-sha"
    assert metadata["identity"]["license"] == "insightface-non-commercial-research"


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
            "identity": {
                "arcface_onnx": "/models/arcface.onnx",
                "min_cosine_similarity": 0.42,
            },
            "blur": {"min_target_mouth_laplacian_variance": 55.0},
            "sync": {"max_reject_lag_frames": 1},
        }
    )

    assert config.match.max_yaw_delta == pytest.approx(7.0)
    assert config.match.max_mouth_center_delta == pytest.approx(0.02)
    assert config.identity.arcface_onnx == "/models/arcface.onnx"
    assert config.identity.min_cosine_similarity == pytest.approx(0.42)
    assert config.blur.min_target_mouth_laplacian_variance == pytest.approx(55.0)
    assert config.sync.max_reject_lag_frames == 1


def test_preview_rows_include_near_identity_threshold() -> None:
    from edge_lipsync.silent_talking_dataset import _preview_rows

    rows = [
        {**_complete_row("train"), "pair_id": "high", "identity_similarity": 0.9},
        {**_complete_row("train"), "pair_id": "near", "identity_similarity": 0.36},
    ]

    previews = _preview_rows(rows, count=1)

    assert previews["near_identity_threshold"][0]["pair_id"] == "near"


def test_identity_runtime_errors_are_always_fatal() -> None:
    from edge_lipsync.identity import IdentityRuntimeError
    from edge_lipsync.silent_talking_dataset import _clip_failure_is_fatal

    assert _clip_failure_is_fatal(IdentityRuntimeError("onnx failed")) is True
    assert _clip_failure_is_fatal(ValueError("bad clip")) is False


def test_cached_normalize_talking_video_reuses_valid_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    source = tmp_path / "talk.mp4"
    source.write_bytes(b"source")
    video_out = tmp_path / "normalized/video.mkv"
    audio_out = tmp_path / "normalized/audio.wav"
    calls = 0

    def fake_normalize(
        _source: Path,
        video_path: Path,
        audio_path: Path,
        *,
        fps: int,
    ) -> tuple[Path, Path]:
        nonlocal calls
        calls += 1
        assert fps == 25
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"video")
        audio_path.write_bytes(b"audio")
        return video_path, audio_path

    monkeypatch.setattr(builder, "normalize_talking_video", fake_normalize)

    first = builder.cached_normalize_talking_video(
        source,
        video_out,
        audio_out,
        fps=25,
        sample_rate=16000,
    )
    second = builder.cached_normalize_talking_video(
        source,
        video_out,
        audio_out,
        fps=25,
        sample_rate=16000,
    )

    assert first.hit is False
    assert second.hit is True
    assert second.value == (video_out, audio_out)
    assert calls == 1


def test_cached_extract_frames_rebuilds_incomplete_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    video = tmp_path / "video.mkv"
    video.write_bytes(b"video")
    frames = tmp_path / "frames"
    calls = 0

    def fake_extract(
        _video: Path,
        output: Path,
        **_kwargs: object,
    ) -> int:
        nonlocal calls
        calls += 1
        output.mkdir(parents=True, exist_ok=True)
        for frame_idx in (1, 2):
            cv2.imwrite(
                str(output / f"{frame_idx:06d}.png"),
                np.full((8, 8, 3), frame_idx, dtype=np.uint8),
            )
        return 2

    monkeypatch.setattr(builder, "extract_frames", fake_extract)

    first = builder.cached_extract_frames(video, frames, show_progress=False)
    second = builder.cached_extract_frames(video, frames, show_progress=False)
    (frames / "000002.png").unlink()
    third = builder.cached_extract_frames(video, frames, show_progress=False)

    assert (first.hit, second.hit, third.hit) == (False, True, False)
    assert calls == 2


def test_cached_bnf_windows_reuses_valid_array_and_invalidates_model(
    tmp_path: Path,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    cache = tmp_path / "bnf.npy"

    class FakeWenet:
        def __init__(self) -> None:
            self.calls = 0

        def extract_wav(self, wav_path: str | Path) -> np.ndarray:
            del wav_path
            self.calls += 1
            return np.full((3, 20, 256), self.calls, dtype=np.float32)

    wenet = FakeWenet()
    first = builder.cached_bnf_windows(
        audio,
        cache,
        wenet,
        wenet_sha256="model-a",
        sample_rate=16000,
    )
    second = builder.cached_bnf_windows(
        audio,
        cache,
        wenet,
        wenet_sha256="model-a",
        sample_rate=16000,
    )
    third = builder.cached_bnf_windows(
        audio,
        cache,
        wenet,
        wenet_sha256="model-b",
        sample_rate=16000,
    )

    assert (first.hit, second.hit, third.hit) == (False, True, False)
    assert wenet.calls == 2
    assert np.all(second.value == 1.0)
    assert np.all(third.value == 2.0)


def test_analysis_cache_metadata_uses_only_analysis_inputs(tmp_path: Path) -> None:
    from edge_lipsync.silent_talking_dataset import analysis_cache_metadata

    video = tmp_path / "video.mp4"
    landmark = tmp_path / "face_landmarker.task"
    video.write_bytes(b"video")
    landmark.write_bytes(b"landmark")

    metadata = analysis_cache_metadata(
        video,
        frame_count=25,
        landmark_model_path=landmark,
        identity_sha256="arcface-sha",
    )

    assert metadata["input"]["sha256"]
    assert metadata["landmark_model"]["sha256"]
    assert metadata["identity_sha256"] == "arcface-sha"
    assert "snapshot_root" not in metadata
    assert "preview_count_per_group" not in metadata


def test_cache_summary_does_not_count_missing_stage_as_miss() -> None:
    from edge_lipsync.silent_talking_dataset import _cache_summary

    summary = _cache_summary(
        [
            {"normalized_media": True, "frames": True, "analysis": True},
            {
                "normalized_media": False,
                "frames": False,
                "analysis": False,
                "bnf": False,
            },
        ]
    )

    assert summary["bnf"] == {"hits": 0, "misses": 1}


def test_build_config_from_mapping_maps_runtime_options() -> None:
    from edge_lipsync.silent_talking_dataset import build_config_from_mapping

    config = build_config_from_mapping(
        {
            "data_root": "data",
            "persona_id": "nora",
            "snapshot_root": "snapshot",
            "work_root": "work",
            "wenet_onnx": "wenet.onnx",
            "runtime": {
                "device": "cuda",
                "clip_workers": 6,
                "cuda_max_inflight": 3,
                "warn_on_cpu_fallback": False,
            },
        }
    )

    assert config.runtime.device == "cuda"
    assert config.runtime.clip_workers == 6
    assert config.runtime.cuda_max_inflight == 3
    assert config.runtime.warn_on_cpu_fallback is False


def test_thread_local_detector_pool_reuses_per_thread_and_closes_all() -> None:
    from edge_lipsync.silent_talking_dataset import ThreadLocalDetectorPool

    created: list[Any] = []
    barrier = threading.Barrier(2)

    class FakeDetector:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def detect_landmarks(
            self,
            frame_bgr: np.ndarray,
            /,
        ) -> Mapping[int, tuple[float, float]] | None:
            del frame_bgr
            return None

    def factory() -> FakeDetector:
        detector = FakeDetector()
        created.append(detector)
        return detector

    pool = ThreadLocalDetectorPool(factory)

    def use_detector_twice() -> tuple[int, int]:
        first = pool.get()
        barrier.wait()
        second = pool.get()
        return id(first), id(second)

    with ThreadPoolExecutor(max_workers=2) as executor:
        identities = list(executor.map(lambda _index: use_detector_twice(), range(2)))
    pool.close_all()

    assert len(created) == 2
    assert all(first == second for first, second in identities)
    assert len({first for first, _second in identities}) == 2
    assert all(detector.closed for detector in created)


def test_run_clip_workers_processes_concurrently_and_sorts_results(
    tmp_path: Path,
) -> None:
    from edge_lipsync.silent_talking_dataset import ClipBuildResult, run_clip_workers

    videos = [tmp_path / name for name in ("c.mp4", "a.mp4", "b.mp4")]
    active = 0
    peak = 0
    lock = threading.Lock()

    def worker(video: Path) -> ClipBuildResult:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return ClipBuildResult(clip_id=video.stem)

    results, failures = run_clip_workers(
        videos,
        worker,
        max_workers=2,
        strict=True,
    )

    assert peak == 2
    assert [result.clip_id for result in results] == ["a", "b", "c"]
    assert failures == []


def test_run_clip_workers_releases_heavy_result_payloads(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from edge_lipsync.silent_talking_dataset import ClipBuildResult, run_clip_workers

    handled: list[str] = []

    def persist_result(result: ClipBuildResult) -> ClipBuildResult:
        handled.append(result.clip_id)
        assert result.rows
        return replace(result, rows=[])

    results, failures = run_clip_workers(
        [tmp_path / "b.mp4", tmp_path / "a.mp4"],
        lambda video: ClipBuildResult(
            clip_id=video.stem,
            rows=[{"payload": b"x" * 1024}],
        ),
        max_workers=1,
        strict=True,
        result_handler=persist_result,
    )

    assert handled == ["a", "b"]
    assert [result.clip_id for result in results] == ["a", "b"]
    assert all(result.rows == [] for result in results)
    assert failures == []


def test_run_clip_workers_reports_completed_clip_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    progress_calls: list[dict[str, Any]] = []

    def fake_progress(iterable: object, **kwargs: Any) -> object:
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(builder, "progress", fake_progress)

    builder.run_clip_workers(
        [tmp_path / "b.mp4", tmp_path / "a.mp4"],
        lambda video: builder.ClipBuildResult(clip_id=video.stem),
        max_workers=2,
        strict=True,
        show_progress=True,
    )

    assert progress_calls == [
        {
            "enabled": True,
            "desc": "build talking clips",
            "total": 2,
            "unit": "clip",
        }
    ]


def test_run_clip_workers_keeps_nonfatal_failures_when_not_strict(
    tmp_path: Path,
) -> None:
    from edge_lipsync.silent_talking_dataset import ClipBuildResult, run_clip_workers

    videos = [tmp_path / "good.mp4", tmp_path / "bad.mp4"]

    def worker(video: Path) -> ClipBuildResult:
        if video.stem == "bad":
            raise ValueError("bad clip")
        return ClipBuildResult(clip_id=video.stem)

    results, failures = run_clip_workers(
        videos,
        worker,
        max_workers=2,
        strict=False,
    )

    assert [result.clip_id for result in results] == ["good"]
    assert [failure.clip_id for failure in failures] == ["bad"]
    assert isinstance(failures[0].error, ValueError)


def test_run_clip_workers_re_raises_fatal_runtime_error(tmp_path: Path) -> None:
    from edge_lipsync.identity import IdentityRuntimeError
    from edge_lipsync.silent_talking_dataset import ClipBuildResult, run_clip_workers

    def worker(video: Path) -> ClipBuildResult:
        raise IdentityRuntimeError(video.stem)

    with pytest.raises(IdentityRuntimeError):
        run_clip_workers(
            [tmp_path / "bad.mp4"],
            worker,
            max_workers=1,
            strict=False,
        )


def test_build_silent_talking_dataset_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_silent_talking_dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "pose-paired" in result.stdout
