from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _HubArtifact:
    resolved_revision: str
    url: str


class _FakeApi:
    def __init__(self, files: list[str]) -> None:
        self.files = files
        self.calls: list[dict[str, Any]] = []

    def list_repo_files(self, **kwargs: Any) -> list[str]:
        self.calls.append(kwargs)
        return self.files


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


def test_select_hf_video_dataset_files_limits_videos_and_matching_metadata() -> None:
    from edge_lipsync.hf_video_dataset import select_hf_video_dataset_files

    selection = select_hf_video_dataset_files(
        [
            "README.md",
            "xdub_teacher_pairs_manifest.json",
            "xdub_teacher_pairs/videos/b.mp4",
            "xdub_teacher_pairs/videos/a.mp4",
            "xdub_teacher_pairs/videos/c.mp4",
            "xdub_teacher_pairs/meta/a.json",
            "xdub_teacher_pairs/meta/b.json",
            "xdub_teacher_pairs/meta/c.json",
            "hdtf_landmarks.tar.gz",
        ],
        video_prefix="xdub_teacher_pairs/videos",
        metadata_prefix="xdub_teacher_pairs/meta",
        max_videos=2,
    )

    assert selection.video_files == [
        "xdub_teacher_pairs/videos/a.mp4",
        "xdub_teacher_pairs/videos/b.mp4",
    ]
    assert selection.allow_patterns == [
        "README.md",
        "xdub_teacher_pairs_manifest.json",
        "xdub_teacher_pairs/videos/a.mp4",
        "xdub_teacher_pairs/videos/b.mp4",
        "xdub_teacher_pairs/meta/a.json",
        "xdub_teacher_pairs/meta/b.json",
    ]


def test_build_hf_video_dataset_dry_run_skips_download_build_and_push(tmp_path: Path) -> None:
    from edge_lipsync.hf_video_dataset import HfVideoDatasetBuildConfig, build_hf_video_dataset

    api = _FakeApi(
        [
            "README.md",
            "xdub_teacher_pairs/videos/a.mp4",
            "xdub_teacher_pairs/videos/b.mp4",
            "xdub_teacher_pairs/meta/a.json",
            "xdub_teacher_pairs/meta/b.json",
        ]
    )

    result = build_hf_video_dataset(
        HfVideoDatasetBuildConfig(
            repo_id="Pinch-Research/lipsync-hdtf-training-data",
            revision="dataset-sha",
            dataset_root=str(tmp_path / "dataset"),
            work_dir=str(tmp_path / "work"),
            wenet_onnx="models/wenet/wenet.onnx",
            landmark_model_asset_path="models/mediapipe/face_landmarker.task",
            max_videos=1,
            dry_run=True,
            push=True,
            hf_output_repo_id="owner/hdtf-duix",
        ),
        api=api,
    )

    assert result.dry_run is True
    assert result.selected_video_count == 1
    assert result.raw_video_count == 0
    assert result.pushed_revision is None
    assert result.snapshot_path is None
    assert api.calls == [
        {
            "repo_id": "Pinch-Research/lipsync-hdtf-training-data",
            "repo_type": "dataset",
            "revision": "dataset-sha",
        }
    ]
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "dataset").exists()


def test_build_hf_video_dataset_downloads_subset_builds_and_pushes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.hf_video_dataset as hf_video_dataset

    snapshot = tmp_path / "snapshot"
    for relative in (
        "xdub_teacher_pairs/videos/a.mp4",
        "xdub_teacher_pairs/videos/b.mp4",
        "xdub_teacher_pairs/meta/a.json",
        "xdub_teacher_pairs/meta/b.json",
    ):
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())

    api = _FakeApi(
        [
            "README.md",
            "xdub_teacher_pairs/videos/a.mp4",
            "xdub_teacher_pairs/videos/b.mp4",
            "xdub_teacher_pairs/meta/a.json",
            "xdub_teacher_pairs/meta/b.json",
        ]
    )
    snapshot_calls: list[dict[str, Any]] = []
    build_calls: list[tuple[Any, bool]] = []
    push_calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        snapshot_calls.append(kwargs)
        return str(snapshot)

    def fake_build_dataset(config: Any, *, strict: bool = False) -> dict[str, int]:
        build_calls.append((config, strict))
        _write_dataset_artifacts(Path(config.dataset_root))
        return {"processed_clips": 2}

    def fake_push_dataset_snapshot(
        dataset_root: str | Path,
        repo_id: str,
        *,
        private: bool,
        commit_message: str,
    ) -> _HubArtifact:
        push_calls.append(
            {
                "dataset_root": Path(dataset_root),
                "repo_id": repo_id,
                "private": private,
                "commit_message": commit_message,
            }
        )
        return _HubArtifact("processed-sha", "https://huggingface.co/datasets/owner/hdtf-duix")

    monkeypatch.setattr(hf_video_dataset, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hf_video_dataset, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(hf_video_dataset, "push_dataset_snapshot", fake_push_dataset_snapshot)

    result = hf_video_dataset.build_hf_video_dataset(
        hf_video_dataset.HfVideoDatasetBuildConfig(
            repo_id="Pinch-Research/lipsync-hdtf-training-data",
            revision="dataset-sha",
            dataset_root=str(tmp_path / "dataset"),
            work_dir=str(tmp_path / "work"),
            wenet_onnx="models/wenet/wenet.onnx",
            landmark_model_asset_path="models/mediapipe/face_landmarker.task",
            max_videos=2,
            push=True,
            hf_output_repo_id="owner/hdtf-duix",
            private=False,
            strict=True,
        ),
        api=api,
    )

    assert result.dry_run is False
    assert result.selected_video_count == 2
    assert result.raw_video_count == 2
    assert result.pushed_revision == "processed-sha"
    assert snapshot_calls == [
        {
            "repo_id": "Pinch-Research/lipsync-hdtf-training-data",
            "repo_type": "dataset",
            "revision": "dataset-sha",
            "allow_patterns": [
                "README.md",
                "xdub_teacher_pairs/videos/a.mp4",
                "xdub_teacher_pairs/videos/b.mp4",
                "xdub_teacher_pairs/meta/a.json",
                "xdub_teacher_pairs/meta/b.json",
            ],
        }
    ]
    raw_video_dir = tmp_path / "work" / "raw_videos"
    assert sorted(path.name for path in raw_video_dir.iterdir()) == ["a.mp4", "b.mp4"]
    assert Path(build_calls[0][0].raw_video_dir) == raw_video_dir
    assert Path(build_calls[0][0].dataset_root) == tmp_path / "dataset"
    assert build_calls[0][0].wenet_onnx == "models/wenet/wenet.onnx"
    assert build_calls[0][1] is True
    assert push_calls == [
        {
            "dataset_root": tmp_path / "dataset",
            "repo_id": "owner/hdtf-duix",
            "private": False,
            "commit_message": "Upload processed HF video dataset snapshot",
        }
    ]


def test_build_hf_video_dataset_requires_pinned_revision(tmp_path: Path) -> None:
    from edge_lipsync.hf_video_dataset import HfVideoDatasetBuildConfig, build_hf_video_dataset

    api = _FakeApi([])

    try:
        build_hf_video_dataset(
            HfVideoDatasetBuildConfig(
                repo_id="Pinch-Research/lipsync-hdtf-training-data",
                revision="",
                dataset_root=str(tmp_path / "dataset"),
                wenet_onnx="models/wenet/wenet.onnx",
            ),
            api=api,
        )
    except ValueError as exc:
        assert "revision" in str(exc)
    else:
        raise AssertionError("expected pinned revision error")


def test_build_hf_video_dataset_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_hf_video_dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Build a Duix dataset from videos stored in a Hugging Face dataset" in result.stdout
    assert "--repo-id" in result.stdout
    assert "--video-prefix" in result.stdout
    assert "--max-videos" in result.stdout
    assert "--dry-run" in result.stdout
