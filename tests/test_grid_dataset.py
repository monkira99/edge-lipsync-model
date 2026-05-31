from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _write_grid_sample(root: Path, speaker: str, stem: str, *, with_audio: bool = True) -> None:
    video_dir = root / speaker / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / f"{stem}.mpg").write_bytes(b"video")
    if with_audio:
        audio_dir = root / speaker / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / f"{stem}.wav").write_bytes(b"audio")


def _write_dataset_artifacts(root: Path) -> None:
    clip = root / "clips" / "clip_000001"
    frames = clip / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    (root / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "splits.json").write_text("{}\n", encoding="utf-8")
    (root / "build_summary.json").write_text("{}\n", encoding="utf-8")
    (frames / "000001.png").write_bytes(b"png")
    (clip / "bnf.npy").write_bytes(b"npy")
    (clip / "bboxes.json").write_text("{}\n", encoding="utf-8")
    (clip / "quality.json").write_text("{}\n", encoding="utf-8")


def test_discover_grid_samples_filters_speaker_and_limit(tmp_path: Path) -> None:
    from edge_lipsync.grid_dataset import discover_grid_samples

    grid_root = tmp_path / "grid"
    _write_grid_sample(grid_root, "s1", "bbaf2n")
    _write_grid_sample(grid_root, "s1", "bbal6n")
    _write_grid_sample(grid_root, "s2", "bbaf2n")

    samples = discover_grid_samples(grid_root, speaker="s1", max_videos=1)

    assert [(sample.speaker, sample.stem, sample.video_path.name) for sample in samples] == [
        ("s1", "bbaf2n", "bbaf2n.mpg")
    ]
    assert samples[0].audio_path == grid_root / "s1" / "audio" / "bbaf2n.wav"


def test_build_grid_dataset_dry_run_skips_build_and_push(tmp_path: Path) -> None:
    from edge_lipsync.grid_dataset import GridBuildConfig, build_grid_dataset

    grid_root = tmp_path / "grid"
    _write_grid_sample(grid_root, "s1", "bbaf2n")
    _write_grid_sample(grid_root, "s1", "bbal6n")

    result = build_grid_dataset(
        GridBuildConfig(
            grid_root=str(grid_root),
            dataset_root=str(tmp_path / "dataset"),
            work_dir=str(tmp_path / "work"),
            wenet_onnx="models/wenet.onnx",
            landmark_model_asset_path="models/face_landmarker.task",
            speaker="s1",
            dry_run=True,
            push=True,
            hf_repo_id="owner/grid-duix",
        )
    )

    assert result.dry_run is True
    assert result.sample_count == 2
    assert result.raw_video_count == 0
    assert result.pushed_revision is None
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "dataset").exists()


def test_prepare_grid_raw_videos_muxes_paired_audio(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.grid_dataset as grid_dataset

    grid_root = tmp_path / "grid"
    _write_grid_sample(grid_root, "s1", "bbaf2n")
    samples = grid_dataset.discover_grid_samples(grid_root)
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(grid_dataset, "require_tool", lambda name: name)
    monkeypatch.setattr(grid_dataset, "run", fake_run)

    prepared = grid_dataset.prepare_grid_raw_videos(
        samples,
        tmp_path / "work" / "raw_videos",
        fps=25,
        sample_rate=16000,
    )

    assert prepared.raw_video_count == 1
    assert prepared.raw_video_paths == [tmp_path / "work" / "raw_videos" / "s1_bbaf2n.mp4"]
    assert commands == [
        [
            "ffmpeg",
            "-y",
            "-i",
            str(grid_root / "s1" / "video" / "bbaf2n.mpg"),
            "-i",
            str(grid_root / "s1" / "audio" / "bbaf2n.wav"),
            "-r",
            "25",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(tmp_path / "work" / "raw_videos" / "s1_bbaf2n.mp4"),
        ]
    ]


def test_prepare_grid_raw_videos_reports_progress(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.grid_dataset as grid_dataset

    grid_root = tmp_path / "grid"
    _write_grid_sample(grid_root, "s1", "bbaf2n")
    samples = grid_dataset.discover_grid_samples(grid_root)
    calls: list[dict[str, object]] = []

    def fake_progress(iterable: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(grid_dataset, "progress", fake_progress)
    monkeypatch.setattr(grid_dataset, "require_tool", lambda name: name)
    monkeypatch.setattr(
        grid_dataset,
        "run",
        lambda command: Path(command[-1]).write_bytes(b"mp4")
        or subprocess.CompletedProcess(command, 0, "", ""),
    )

    grid_dataset.prepare_grid_raw_videos(samples, tmp_path / "work" / "raw_videos")

    assert calls == [
        {
            "enabled": True,
            "desc": "prepare GRID videos",
            "total": 1,
            "unit": "clip",
        }
    ]


def test_build_grid_dataset_invokes_builder_and_pushes_snapshot(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.grid_dataset as grid_dataset

    @dataclass(frozen=True)
    class _HubArtifact:
        resolved_revision: str
        url: str

    grid_root = tmp_path / "grid"
    dataset_root = tmp_path / "dataset"
    work_dir = tmp_path / "work"
    _write_grid_sample(grid_root, "s1", "bbaf2n")
    build_calls: list[tuple[Any, bool]] = []
    push_calls: list[dict[str, Any]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_build_dataset(config: Any, *, strict: bool = False) -> dict[str, int]:
        build_calls.append((config, strict))
        _write_dataset_artifacts(Path(config.dataset_root))
        return {"processed": 1}

    def fake_push_dataset_snapshot(
        dataset_root_arg: str | Path,
        repo_id: str,
        *,
        private: bool,
        commit_message: str,
    ) -> _HubArtifact:
        push_calls.append(
            {
                "dataset_root": Path(dataset_root_arg),
                "repo_id": repo_id,
                "private": private,
                "commit_message": commit_message,
            }
        )
        return _HubArtifact("abc123", "https://huggingface.co/datasets/owner/grid-duix/tree/abc123")

    monkeypatch.setattr(grid_dataset, "require_tool", lambda name: name)
    monkeypatch.setattr(grid_dataset, "run", fake_run)
    monkeypatch.setattr(grid_dataset, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(grid_dataset, "push_dataset_snapshot", fake_push_dataset_snapshot)

    result = grid_dataset.build_grid_dataset(
        grid_dataset.GridBuildConfig(
            grid_root=str(grid_root),
            dataset_root=str(dataset_root),
            work_dir=str(work_dir),
            wenet_onnx="models/wenet.onnx",
            landmark_model_asset_path="models/face_landmarker.task",
            speaker="s1",
            max_videos=1,
            push=True,
            hf_repo_id="owner/grid-duix",
            private=False,
            strict=True,
        )
    )

    assert result.sample_count == 1
    assert result.raw_video_count == 1
    assert result.pushed_revision == "abc123"
    assert result.hub_url == "https://huggingface.co/datasets/owner/grid-duix/tree/abc123"
    assert Path(build_calls[0][0].raw_video_dir) == work_dir / "raw_videos"
    assert Path(build_calls[0][0].dataset_root) == dataset_root
    assert build_calls[0][0].wenet_onnx == "models/wenet.onnx"
    assert build_calls[0][0].landmark_model_asset_path == "models/face_landmarker.task"
    assert build_calls[0][1] is True
    assert push_calls == [
        {
            "dataset_root": dataset_root,
            "repo_id": "owner/grid-duix",
            "private": False,
            "commit_message": "Upload GRID processed dataset snapshot",
        }
    ]


def test_build_grid_hf_dataset_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_grid_hf_dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Build a processed Duix dataset from GRID" in result.stdout
    assert "--grid-root" in result.stdout
    assert "--dataset-root" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--push" in result.stdout
    assert "--no-progress" in result.stdout
