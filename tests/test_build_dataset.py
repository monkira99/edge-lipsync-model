from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


def test_validate_stream_payload_requires_audio_and_video() -> None:
    from edge_lipsync.build_dataset import validate_stream_payload

    with pytest.raises(ValueError, match="audio"):
        validate_stream_payload({"streams": [{"codec_type": "video"}]})


def test_dataset_build_defaults_to_duix_roi_smoothing_radius() -> None:
    from edge_lipsync.build_dataset import DatasetBuildConfig

    config = DatasetBuildConfig(raw_video_dir="raw", dataset_root="dataset", wenet_onnx="wenet")

    assert config.bbox_smooth_radius == 1


def test_extract_frames_writes_lossless_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.build_dataset as builder

    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self._frames = [frame]

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self._frames:
                return True, self._frames.pop()
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr(builder.cv2, "VideoCapture", FakeCapture)

    count = builder.extract_frames(tmp_path / "video.mkv", tmp_path / "frames")

    extracted = cv2.imread(str(tmp_path / "frames/000001.png"), cv2.IMREAD_COLOR)
    assert count == 1
    assert extracted is not None
    assert np.array_equal(extracted, frame)


def test_normalize_clip_preserves_source_sample_rate_for_python_resampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.build_dataset as builder

    commands: list[list[str]] = []
    monkeypatch.setattr(builder, "require_tool", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        builder,
        "run",
        lambda command: commands.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )

    builder.normalize_clip(
        tmp_path / "input.mkv",
        tmp_path / "normalized",
        fps=25,
        sample_rate=16000,
    )

    audio_command = commands[1]
    assert "-ar" not in audio_command
    assert audio_command[audio_command.index("-acodec") + 1] == "pcm_s16le"


def test_interpolate_short_bbox_gaps_preserves_long_gaps() -> None:
    from edge_lipsync.build_dataset import interpolate_short_bbox_gaps

    boxes = {
        1: (10, 10, 110, 110),
        2: None,
        3: (14, 10, 114, 110),
        4: None,
        5: None,
        6: (20, 10, 120, 110),
    }

    interpolated, flags = interpolate_short_bbox_gaps(boxes, max_gap=1)

    assert interpolated[2] == (12, 10, 112, 110)
    assert interpolated[4] is None
    assert interpolated[5] is None
    assert flags == {2: ["interpolated_bbox"]}


def test_smooth_bboxes_uses_neighboring_frames() -> None:
    from edge_lipsync.build_dataset import smooth_bboxes

    smoothed = smooth_bboxes(
        {
            1: (10, 10, 100, 100),
            2: (20, 20, 120, 120),
        },
        radius=1,
    )

    assert smoothed[1] == (15, 15, 110, 110)
    assert smoothed[2] == (15, 15, 110, 110)


def test_clean_bboxes_drops_discontinuous_tracking_jump() -> None:
    from edge_lipsync.build_dataset import BBoxGates, clean_bboxes

    boxes = {
        1: (10, 10, 50, 50),
        2: (150, 150, 190, 190),
    }
    frame_shapes = {1: (200, 200, 3), 2: (200, 200, 3)}

    cleaned, _flags, drops = clean_bboxes(
        boxes,
        frame_shapes,
        gates=BBoxGates(min_size=32, max_frame_fraction=0.9, max_jump_fraction=0.25),
        max_missing_gap=1,
        smooth_radius=0,
    )

    assert cleaned == {1: (10, 10, 50, 50)}
    assert drops["discontinuous_jump"] == 1


@pytest.mark.parametrize(
    ("bbox", "reason"),
    [
        ((10, 10, 10, 20), "invalid"),
        ((-1, 10, 50, 50), "outside_frame"),
        ((10, 10, 20, 20), "too_small"),
        ((0, 0, 100, 100), "too_large"),
    ],
)
def test_bbox_quality_reason_applies_gates(
    bbox: tuple[int, int, int, int],
    reason: str,
) -> None:
    from edge_lipsync.build_dataset import BBoxGates, bbox_quality_reason

    gates = BBoxGates(min_size=32, max_frame_fraction=0.9, max_jump_fraction=0.5)

    assert bbox_quality_reason(bbox, (100, 100, 3), gates) == reason


def test_limit_silence_keeps_all_voice_and_caps_silence() -> None:
    from edge_lipsync.build_dataset import limit_silence

    frame_indices = list(range(1, 11))
    silent_audio_indices = {0, 1, 2, 3, 4, 5, 6, 7}

    kept, dropped = limit_silence(
        frame_indices,
        silent_audio_indices=silent_audio_indices,
        max_silence_fraction=0.5,
    )

    assert {9, 10}.issubset(kept)
    assert len(kept) == 4
    assert dropped == 6


def test_write_manifest_creates_relative_records_and_splits(tmp_path: Path) -> None:
    from edge_lipsync.build_dataset import write_manifest

    clips = [
        {
            "clip_id": "a",
            "valid_frames": [1, 2],
            "bboxes": {1: (10, 10, 100, 100), 2: (12, 10, 102, 100)},
            "flags": {2: ["interpolated_bbox"]},
        },
        {
            "clip_id": "b",
            "valid_frames": [1, 2],
            "bboxes": {1: (20, 20, 120, 120), 2: (22, 20, 122, 120)},
            "flags": {},
        },
    ]

    split_counts = write_manifest(tmp_path, clips, validation_fraction=0.5)
    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 4
    assert rows[0]["frame_path"] == "clips/a/frames/000001.png"
    assert rows[1]["flags"] == ["interpolated_bbox"]
    assert {row["split"] for row in rows} == {"train", "val"}
    assert split_counts == {"train": 2, "val": 2}


def test_write_preview_outputs_overlay_real_masked_and_target(tmp_path: Path) -> None:
    from edge_lipsync.build_dataset import write_preview

    frame = np.full((240, 320, 3), 120, dtype=np.uint8)

    write_preview(frame, (80, 40, 240, 200), tmp_path, frame_idx=1)

    assert sorted(path.name for path in tmp_path.glob("*.jpg")) == [
        "000001_masked.jpg",
        "000001_overlay.jpg",
        "000001_real.jpg",
        "000001_target.jpg",
    ]


def test_process_clip_uses_landmark_roi_not_raw_face_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.build_dataset as builder

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    video = raw_dir / "clip.mp4"
    video.write_bytes(b"fixture")
    wenet = tmp_path / "wenet.onnx"
    wenet.write_bytes(b"fixture")

    def fake_extract_frames(_video_path: Path, frames_dir: Path) -> int:
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame = np.full((960, 540, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "000001.png"), frame)
        return 1

    def fake_normalize_clip(
        _src: Path,
        out_dir: Path,
        _fps: int,
        _sample_rate: int,
    ) -> tuple[Path, Path]:
        return out_dir / "video.mp4", out_dir / "audio.wav"

    class FakeLandmarkDetector:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def detect_bbox(self, _frame: np.ndarray) -> tuple[int, int, int, int]:
            return (120, 320, 440, 640)

        def close(self) -> None:
            pass

    monkeypatch.setattr(builder, "probe_clip", lambda _path: {"streams": []})
    monkeypatch.setattr(builder, "normalize_clip", fake_normalize_clip)
    monkeypatch.setattr(builder, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        builder,
        "extract_bnf_windows_from_wav",
        lambda _audio_path, _wenet: np.zeros((2, 256), dtype=np.float32),
    )
    monkeypatch.setattr(builder, "load_wav_mono_f32", lambda _path: np.ones(640, dtype=np.float32))
    monkeypatch.setattr(builder, "_silent_audio_indices", lambda _audio, _threshold: set())
    monkeypatch.setattr(builder, "detect_largest_face", lambda _frame: (86, 136, 475, 525))
    monkeypatch.setattr(
        builder,
        "MediaPipeFaceLandmarkerDetector",
        FakeLandmarkDetector,
        raising=False,
    )

    config = builder.DatasetBuildConfig(
        raw_video_dir=str(raw_dir),
        dataset_root=str(tmp_path / "dataset"),
        wenet_onnx=str(wenet),
        bbox_detector="mediapipe_face_landmarker",
        preview_count=0,
    )

    clip = builder.process_clip(video, config)

    assert clip["bboxes"][1] == (120, 320, 440, 640)
    bboxes = json.loads(
        (tmp_path / "dataset/clips/clip/bboxes.json").read_text(encoding="utf-8")
    )
    assert bboxes["1"] == [120, 320, 440, 640]


def test_build_dataset_records_clip_failure_unless_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.build_dataset as builder

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "broken.mp4").write_bytes(b"not a real video")
    wenet = tmp_path / "wenet.onnx"
    wenet.write_bytes(b"fixture")
    config = builder.DatasetBuildConfig(
        raw_video_dir=str(raw_dir),
        dataset_root=str(tmp_path / "dataset"),
        wenet_onnx=str(wenet),
    )

    def fail_clip(video: Path, config: builder.DatasetBuildConfig) -> dict[str, object]:
        raise RuntimeError(f"cannot process {video.name}")

    monkeypatch.setattr(builder, "process_clip", fail_clip)

    summary = builder.build_dataset(config)

    assert summary["processed_clips"] == 0
    assert len(summary["failed_clips"]) == 1
    quality = json.loads(
        (tmp_path / "dataset/clips/broken/quality.json").read_text(encoding="utf-8")
    )
    assert quality["status"] == "failed"
    with pytest.raises(RuntimeError, match="cannot process"):
        builder.build_dataset(config, strict=True)


def test_build_dataset_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Build Duix training dataset" in result.stdout
    assert "--strict" in result.stdout
