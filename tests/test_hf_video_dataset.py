from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class _HubArtifact:
    repo_id: str
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


def test_select_hf_video_dataset_files_limits_videos_without_snapshot_patterns() -> None:
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
        max_videos=2,
    )

    assert selection.video_files == [
        "xdub_teacher_pairs/videos/a.mp4",
        "xdub_teacher_pairs/videos/b.mp4",
    ]
    assert not hasattr(selection, "allow_patterns")


def test_select_hf_video_dataset_files_filters_by_speaker_metadata() -> None:
    from edge_lipsync.hf_video_dataset import select_hf_video_dataset_files

    selection = select_hf_video_dataset_files(
        [
            "xdub_teacher_pairs/videos/Alice_shot_001__x__Alice_shot_002.mp4",
            "xdub_teacher_pairs/videos/Alice_shot_003__x__Alice_shot_004.mp4",
            "xdub_teacher_pairs/videos/Bob_shot_001__x__Bob_shot_002.mp4",
            "xdub_teacher_pairs/videos/NotInManifest.mp4",
        ],
        video_prefix="xdub_teacher_pairs/videos",
        max_videos=1,
        speaker_id="Alice",
        metadata_entries=[
            {
                "id": "Alice_shot_001__x__Alice_shot_002",
                "src_speaker": "Alice",
                "alt_speaker": "Alice",
            },
            {
                "id": "Alice_shot_003__x__Alice_shot_004",
                "src_speaker": "Alice",
                "alt_speaker": "Alice",
            },
            {
                "id": "Bob_shot_001__x__Bob_shot_002",
                "src_speaker": "Bob",
                "alt_speaker": "Bob",
            },
        ],
    )

    assert selection.video_files == [
        "xdub_teacher_pairs/videos/Alice_shot_001__x__Alice_shot_002.mp4"
    ]
    assert selection.speaker_counts == {"Alice": 2, "Bob": 1}


def test_build_hf_video_dataset_dry_run_filters_by_speaker(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.hf_video_dataset as hf_video_dataset

    api = _FakeApi(
        [
            "xdub_teacher_pairs_manifest.json",
            "xdub_teacher_pairs/videos/Alice_shot_001__x__Alice_shot_002.mp4",
            "xdub_teacher_pairs/videos/Bob_shot_001__x__Bob_shot_002.mp4",
        ]
    )
    monkeypatch.setattr(
        hf_video_dataset,
        "load_hf_video_metadata_manifest",
        lambda **_kwargs: [
            {
                "id": "Alice_shot_001__x__Alice_shot_002",
                "src_speaker": "Alice",
                "alt_speaker": "Alice",
            },
            {
                "id": "Bob_shot_001__x__Bob_shot_002",
                "src_speaker": "Bob",
                "alt_speaker": "Bob",
            },
        ],
    )

    result = hf_video_dataset.build_hf_video_dataset(
        hf_video_dataset.HfVideoDatasetBuildConfig(
            repo_id="Pinch-Research/lipsync-hdtf-training-data",
            dataset_root=str(tmp_path / "dataset"),
            work_dir=str(tmp_path / "work"),
            wenet_onnx="models/wenet/wenet.onnx",
            speaker_id="Alice",
            dry_run=True,
        ),
        api=api,
    )

    assert result.dry_run is True
    assert result.speaker_id == "Alice"
    assert result.speaker_counts == {"Alice": 1, "Bob": 1}
    assert result.selected_video_count == 1
    assert result.selected_video_files == [
        "xdub_teacher_pairs/videos/Alice_shot_001__x__Alice_shot_002.mp4"
    ]


def test_build_hf_video_dataset_list_speakers_skips_push_requirement(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.hf_video_dataset as hf_video_dataset

    api = _FakeApi(
        [
            "xdub_teacher_pairs_manifest.json",
            "xdub_teacher_pairs/videos/Alice_shot_001__x__Alice_shot_002.mp4",
        ]
    )
    monkeypatch.setattr(
        hf_video_dataset,
        "load_hf_video_metadata_manifest",
        lambda **_kwargs: [
            {
                "id": "Alice_shot_001__x__Alice_shot_002",
                "src_speaker": "Alice",
                "alt_speaker": "Alice",
            }
        ],
    )

    result = hf_video_dataset.build_hf_video_dataset(
        hf_video_dataset.HfVideoDatasetBuildConfig(
            repo_id="Pinch-Research/lipsync-hdtf-training-data",
            dataset_root=str(tmp_path / "dataset"),
            work_dir=str(tmp_path / "work"),
            wenet_onnx="models/wenet/wenet.onnx",
            list_speakers=True,
            push=True,
        ),
        api=api,
    )

    assert result.dry_run is True
    assert result.speaker_counts == {"Alice": 1}


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
    assert not hasattr(result, "snapshot_path")
    assert api.calls == [
        {
            "repo_id": "Pinch-Research/lipsync-hdtf-training-data",
            "repo_type": "dataset",
        }
    ]
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "dataset").exists()


def test_build_hf_video_dataset_loads_subset_builds_and_pushes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.hf_video_dataset as hf_video_dataset

    api = _FakeApi(
        [
            "README.md",
            "xdub_teacher_pairs/videos/a.mp4",
            "xdub_teacher_pairs/videos/b.mp4",
            "xdub_teacher_pairs/meta/a.json",
            "xdub_teacher_pairs/meta/b.json",
        ]
    )
    download_calls: list[dict[str, Any]] = []
    build_calls: list[tuple[Any, bool]] = []
    push_calls: list[dict[str, Any]] = []

    def fake_download_hf_video_files(
        repo_id: str,
        video_files: list[str],
        raw_video_dir: str | Path,
        *,
        cache_dir: str,
        max_workers: int,
        show_progress: bool,
    ) -> list[Path]:
        download_calls.append(
            {
                "repo_id": repo_id,
                "video_files": video_files,
                "raw_video_dir": Path(raw_video_dir),
                "cache_dir": cache_dir,
                "max_workers": max_workers,
                "show_progress": show_progress,
            }
        )
        paths: list[Path] = []
        for relative in video_files:
            path = Path(raw_video_dir) / Path(relative).name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
            paths.append(path)
        return paths

    def fake_build_dataset(config: Any, *, strict: bool = False) -> dict[str, int]:
        build_calls.append((config, strict))
        _write_dataset_artifacts(Path(config.dataset_root))
        return {"processed_clips": 2}

    def fake_push_processed_dataset(
        dataset_root: str | Path,
        repo_id: str,
        *,
        private: bool,
    ) -> _HubArtifact:
        push_calls.append(
            {
                "dataset_root": Path(dataset_root),
                "repo_id": repo_id,
                "private": private,
            }
        )
        return _HubArtifact("owner/hdtf-duix", "https://huggingface.co/datasets/owner/hdtf-duix")

    monkeypatch.setattr(hf_video_dataset, "download_hf_video_files", fake_download_hf_video_files)
    monkeypatch.setattr(hf_video_dataset, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(hf_video_dataset, "push_processed_dataset", fake_push_processed_dataset)

    result = hf_video_dataset.build_hf_video_dataset(
        hf_video_dataset.HfVideoDatasetBuildConfig(
            repo_id="Pinch-Research/lipsync-hdtf-training-data",
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
    assert result.hub_url == "https://huggingface.co/datasets/owner/hdtf-duix"
    assert download_calls == [
        {
            "repo_id": "Pinch-Research/lipsync-hdtf-training-data",
            "video_files": [
                "xdub_teacher_pairs/videos/a.mp4",
                "xdub_teacher_pairs/videos/b.mp4",
            ],
            "raw_video_dir": tmp_path / "work" / "raw_videos",
            "cache_dir": "",
            "max_workers": 1,
            "show_progress": True,
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
        }
    ]


def test_download_hf_video_files_uses_datasets_loader_and_links_local_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.hf_video_dataset as hf_video_dataset

    cached = tmp_path / "cache"
    source_dir = cached / "source"
    source_dir.mkdir(parents=True)
    for name in ("a.mp4", "b.mp4"):
        (source_dir / name).write_bytes(name.encode())
    load_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    progress_calls: list[dict[str, object]] = []

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, dict[str, str | None]]]:
        load_calls.append((args, kwargs))
        assert hf_video_dataset.datasets_config.HF_DATASETS_MULTITHREADING_MAX_WORKERS == 1
        return [
            {"video": {"bytes": None, "path": str(source_dir / "a.mp4")}},
            {"video": {"bytes": None, "path": str(source_dir / "b.mp4")}},
        ]

    def fake_progress(iterable: object, **kwargs: object) -> object:
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(hf_video_dataset, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(hf_video_dataset, "progress", fake_progress)
    monkeypatch.setattr(
        hf_video_dataset.datasets_config,
        "HF_DATASETS_MULTITHREADING_MAX_WORKERS",
        8,
    )

    paths = hf_video_dataset.download_hf_video_files(
        "owner/source-dataset",
        ["videos/a.mp4", "videos/b.mp4"],
        tmp_path / "raw",
        cache_dir=str(cached),
        max_workers=1,
    )

    assert [path.name for path in paths] == ["a.mp4", "b.mp4"]
    assert sorted(path.name for path in (tmp_path / "raw").iterdir()) == ["a.mp4", "b.mp4"]
    assert len(load_calls) == 1
    args, kwargs = load_calls[0]
    assert args == ("owner/source-dataset",)
    assert kwargs["data_files"] == {"train": ["videos/a.mp4", "videos/b.mp4"]}
    assert kwargs["split"] == "train"
    assert "revision" not in kwargs
    assert kwargs["cache_dir"] == str(cached)
    assert kwargs["drop_labels"] is True
    assert kwargs["drop_metadata"] is True
    assert kwargs["features"]["video"].decode is False
    assert kwargs["download_config"].cache_dir == str(cached)
    assert kwargs["download_config"].resume_download is True
    assert kwargs["download_config"].max_retries == 5
    assert kwargs["download_config"].num_proc == 1
    assert hf_video_dataset.datasets_config.HF_DATASETS_MULTITHREADING_MAX_WORKERS == 8
    assert progress_calls == [
        {
            "enabled": True,
            "desc": "prepare HF videos",
            "total": 2,
            "unit": "clip",
        }
    ]


def test_download_hf_video_files_restores_worker_limit_after_loader_error(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import edge_lipsync.hf_video_dataset as hf_video_dataset

    def fake_load_dataset(*_args: Any, **_kwargs: Any) -> object:
        assert hf_video_dataset.datasets_config.HF_DATASETS_MULTITHREADING_MAX_WORKERS == 1
        raise RuntimeError("loader failed")

    monkeypatch.setattr(hf_video_dataset, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        hf_video_dataset.datasets_config,
        "HF_DATASETS_MULTITHREADING_MAX_WORKERS",
        8,
    )

    try:
        hf_video_dataset.download_hf_video_files(
            "owner/source-dataset",
            ["videos/a.mp4"],
            tmp_path / "raw",
        )
    except RuntimeError as exc:
        assert str(exc) == "loader failed"
    else:
        raise AssertionError("expected loader failure")

    assert hf_video_dataset.datasets_config.HF_DATASETS_MULTITHREADING_MAX_WORKERS == 8


def test_build_hf_video_dataset_rejects_non_positive_download_workers(tmp_path: Path) -> None:
    from edge_lipsync.hf_video_dataset import HfVideoDatasetBuildConfig, build_hf_video_dataset

    api = _FakeApi([])

    try:
        build_hf_video_dataset(
            HfVideoDatasetBuildConfig(
                repo_id="Pinch-Research/lipsync-hdtf-training-data",
                dataset_root=str(tmp_path / "dataset"),
                wenet_onnx="models/wenet/wenet.onnx",
                download_max_workers=0,
            ),
            api=api,
        )
    except ValueError as exc:
        assert "download_max_workers" in str(exc)
    else:
        raise AssertionError("expected download_max_workers error")


def test_build_hf_video_dataset_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_hf_video_dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Build a Duix dataset from videos stored in a Hugging Face dataset" in result.stdout
    assert "--repo-id" in result.stdout
    assert "--revision" not in result.stdout
    assert "--video-prefix" in result.stdout
    assert "--max-videos" in result.stdout
    assert "--download-max-workers" in result.stdout
    assert "--download-request-interval-seconds" not in result.stdout
    assert "--speaker-id" in result.stdout
    assert "--list-speakers" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--no-progress" in result.stdout


def test_build_hf_video_dataset_cli_passes_download_max_workers(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import tools.build_hf_video_dataset as cli

    configs: list[Any] = []

    def fake_build_hf_video_dataset(config: Any) -> SimpleNamespace:
        configs.append(config)
        return SimpleNamespace(
            dry_run=True,
            repo_id=config.repo_id,
            dataset_root=Path(config.dataset_root),
            work_dir=tmp_path / "work",
            raw_video_dir=tmp_path / "work" / "raw_videos",
            selected_video_count=1,
            raw_video_count=0,
            hub_url=None,
        )

    monkeypatch.setattr(cli, "build_hf_video_dataset", fake_build_hf_video_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hf_video_dataset.py",
            "--repo-id",
            "Pinch-Research/lipsync-hdtf-training-data",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--wenet-onnx",
            "models/wenet/wenet.onnx",
            "--download-max-workers",
            "3",
            "--dry-run",
        ],
    )

    cli.main()

    assert configs[0].download_max_workers == 3
