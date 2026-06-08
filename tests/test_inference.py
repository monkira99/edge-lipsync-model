from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


def _write_fixture_dataset(root: Path) -> Path:
    clip = root / "clips" / "clip_001"
    frames = clip / "frames"
    frames.mkdir(parents=True)
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(frames / "000001.jpg"), frame)
    np.save(clip / "bnf.npy", np.zeros((30, 256), dtype=np.float32))
    manifest = root / "manifest.jsonl"
    record = {
        "clip_id": "clip_001",
        "frame_idx": 1,
        "audio_idx": 1,
        "frame_path": "clips/clip_001/frames/000001.jpg",
        "bbox_xyxy": [80, 40, 240, 200],
        "bnf_path": "clips/clip_001/bnf.npy",
        "split": "train",
        "flags": [],
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


def _write_sequence_fixture_dataset(root: Path, frame_count: int = 3) -> Path:
    clip = root / "clips" / "clip_001"
    frames = clip / "frames"
    frames.mkdir(parents=True)
    rows = []
    for frame_idx in range(1, frame_count + 1):
        frame = np.full((240, 320, 3), 90 + frame_idx * 20, dtype=np.uint8)
        cv2.imwrite(str(frames / f"{frame_idx:06d}.jpg"), frame)
        rows.append(
            {
                "clip_id": "clip_001",
                "frame_idx": frame_idx,
                "audio_idx": frame_idx - 1,
                "frame_path": f"clips/clip_001/frames/{frame_idx:06d}.jpg",
                "bbox_xyxy": [80, 40, 240, 200],
                "bnf_path": "clips/clip_001/bnf.npy",
                "split": "train",
                "flags": [],
            }
        )
    np.save(clip / "bnf.npy", np.zeros((30, 256), dtype=np.float32))
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _write_checkpoint(path: Path) -> None:
    from edge_lipsync.model import DuixUNet, save_ckpt

    model = DuixUNet().eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 6, 160, 160), torch.zeros(1, 20, 256))
    save_ckpt(model, path)


def test_run_manifest_sample_inference_writes_artifacts(tmp_path: Path) -> None:
    from edge_lipsync.inference import run_manifest_sample_inference

    dataset_root = tmp_path / "dataset"
    manifest = _write_fixture_dataset(dataset_root)
    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint)

    artifacts = run_manifest_sample_inference(
        dataset_root=dataset_root,
        manifest=manifest,
        out_dir=tmp_path / "infer",
        sample_index=0,
        split=None,
        checkpoint=checkpoint,
        init_bin="",
        hf_model_repo="",
        hf_model_filename="best.pt",
        hf_cache_dir="",
        device=torch.device("cpu"),
    )

    assert Path(artifacts["prediction_path"]).exists()
    assert Path(artifacts["grid_path"]).exists()
    assert Path(artifacts["restored_frame_path"]).exists()
    assert Path(artifacts["metadata_path"]).exists()
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["kind"] == "manifest_sample_inference"
    assert metadata["model"]["source"] == "local"
    assert metadata["sample"]["clip_id"] == "clip_001"
    grid = cv2.imread(artifacts["grid_path"], cv2.IMREAD_COLOR)
    assert grid is not None
    assert grid.shape == (160, 640, 3)
    restored = cv2.imread(artifacts["restored_frame_path"], cv2.IMREAD_COLOR)
    assert restored is not None
    assert restored.shape == (240, 320, 3)
    original = cv2.imread(str(dataset_root / "clips/clip_001/frames/000001.jpg"), cv2.IMREAD_COLOR)
    assert original is not None
    assert np.array_equal(restored[:40], original[:40])


def test_restore_prediction_to_frame_applies_native_alpha_blend() -> None:
    from edge_lipsync.inference import restore_prediction_to_frame

    frame = np.zeros((168, 168, 3), dtype=np.uint8)
    roi = np.full((168, 168, 3), 50, dtype=np.uint8)
    prediction_rgb = np.full((160, 160, 3), 200, dtype=np.uint8)

    preserved = restore_prediction_to_frame(
        frame,
        (0, 0, 168, 168),
        roi,
        prediction_rgb,
        alpha_u8=np.full((160, 160), 255, dtype=np.uint8),
    )
    replaced = restore_prediction_to_frame(
        frame,
        (0, 0, 168, 168),
        roi,
        prediction_rgb,
        alpha_u8=np.zeros((160, 160), dtype=np.uint8),
    )

    assert np.all(preserved[4:164, 4:164] == 50)
    assert np.all(replaced[4:164, 4:164] == 200)


def test_ncnn_prediction_runtime_loads_assets_and_predicts_chw(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import edge_lipsync.inference as inference

    param_path = tmp_path / "dh_model.param"
    bin_path = tmp_path / "dh_model.bin"
    param_path.write_text("param", encoding="utf-8")
    bin_path.write_bytes(b"bin")
    calls: list[tuple[str, object]] = []

    class FakeMat:
        def __init__(self, array: np.ndarray) -> None:
            self.array = np.asarray(array)

        def numpy(self) -> np.ndarray:
            return self.array

    class FakeExtractor:
        def input(self, name: str, value: FakeMat) -> int:
            calls.append((f"input:{name}", value.array.copy()))
            return 0

        def extract(self, name: str) -> tuple[int, FakeMat]:
            calls.append(("extract", name))
            return 0, FakeMat(np.full((3, 160, 160), 0.25, dtype=np.float32))

    class FakeNet:
        def load_param(self, path: str) -> int:
            calls.append(("load_param", path))
            return 0

        def load_model(self, path: str) -> int:
            calls.append(("load_model", path))
            return 0

        def create_extractor(self) -> FakeExtractor:
            return FakeExtractor()

    fake_ncnn = SimpleNamespace(Net=FakeNet, Mat=FakeMat)
    monkeypatch.setattr(inference.importlib, "import_module", lambda _name: fake_ncnn)

    runtime, provenance = inference._load_ncnn_runtime(param_path, bin_path)
    prediction = runtime.predict(
        np.zeros((6, 160, 160), dtype=np.float32),
        np.zeros((20, 256), dtype=np.float32),
    )

    assert provenance == {
        "source": "local_ncnn_runtime",
        "param_path": str(param_path.resolve()),
        "bin_path": str(bin_path.resolve()),
    }
    assert prediction.shape == (3, 160, 160)
    assert np.asarray(calls[2][1]).shape == (6, 160, 160)
    assert np.asarray(calls[3][1]).shape == (1, 20, 256)
    assert calls[4] == ("extract", "output")


def test_run_manifest_sample_inference_can_use_wav_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import edge_lipsync.inference as inference
    from edge_lipsync.inference import run_manifest_sample_inference

    dataset_root = tmp_path / "dataset"
    manifest = _write_fixture_dataset(dataset_root)
    checkpoint = tmp_path / "model.pt"
    wav_path = tmp_path / "sample.wav"
    wenet_path = tmp_path / "wenet.onnx"
    _write_checkpoint(checkpoint)
    wav_path.write_bytes(b"wav")
    wenet_path.write_bytes(b"onnx")
    calls: list[tuple[Path, Path]] = []

    def fake_extract_bnf_windows_from_wav(wav: str | Path, wenet: str | Path) -> np.ndarray:
        calls.append((Path(wav), Path(wenet)))
        return np.full((30, 256), 0.25, dtype=np.float32)

    monkeypatch.setattr(
        inference,
        "extract_bnf_windows_from_wav",
        fake_extract_bnf_windows_from_wav,
        raising=False,
    )

    artifacts = run_manifest_sample_inference(
        dataset_root=dataset_root,
        manifest=manifest,
        out_dir=tmp_path / "infer",
        sample_index=0,
        split=None,
        checkpoint=checkpoint,
        init_bin="",
        hf_model_repo="",
        hf_model_filename="best.pt",
        hf_cache_dir="",
        audio_wav=wav_path,
        wenet_onnx=wenet_path,
        device=torch.device("cpu"),
    )

    assert calls == [(wav_path, wenet_path)]
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["audio_source"] == {
        "source": "wav",
        "path": str(wav_path.resolve()),
        "wenet_onnx": str(wenet_path.resolve()),
    }
    assert metadata["shapes"]["audio"] == [20, 256]


def test_run_manifest_sample_inference_can_write_audio_mp4(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import edge_lipsync.inference as inference
    from edge_lipsync.inference import run_manifest_sample_inference

    dataset_root = tmp_path / "dataset"
    manifest = _write_fixture_dataset(dataset_root)
    checkpoint = tmp_path / "model.pt"
    wav_path = tmp_path / "sample.wav"
    wenet_path = tmp_path / "wenet.onnx"
    _write_checkpoint(checkpoint)
    wav_path.write_bytes(b"wav")
    wenet_path.write_bytes(b"onnx")
    commands: list[list[str]] = []

    def fake_extract_bnf_windows_from_wav(_wav: str | Path, _wenet: str | Path) -> np.ndarray:
        return np.full((30, 256), 0.25, dtype=np.float32)

    def fake_require_tool(name: str) -> str:
        assert name == "ffmpeg"
        return "/usr/bin/ffmpeg"

    def fake_run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        inference,
        "extract_bnf_windows_from_wav",
        fake_extract_bnf_windows_from_wav,
        raising=False,
    )
    monkeypatch.setattr(inference, "_require_tool", fake_require_tool, raising=False)
    monkeypatch.setattr(inference, "_run_command", fake_run_command, raising=False)

    artifacts = run_manifest_sample_inference(
        dataset_root=dataset_root,
        manifest=manifest,
        out_dir=tmp_path / "infer",
        sample_index=0,
        split=None,
        checkpoint=checkpoint,
        init_bin="",
        hf_model_repo="",
        hf_model_filename="best.pt",
        hf_cache_dir="",
        audio_wav=wav_path,
        wenet_onnx=wenet_path,
        output_mp4="sample_output.mp4",
        device=torch.device("cpu"),
    )

    output_video = tmp_path / "infer" / "sample_output.mp4"
    assert artifacts["output_mp4_path"] == str(output_video.resolve())
    assert output_video.exists()
    assert len(commands) == 1
    assert commands[0][0] == "/usr/bin/ffmpeg"
    assert "-shortest" in commands[0]
    assert str(wav_path) in commands[0]
    assert commands[0][-1] == str(output_video)
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["artifacts"]["output_mp4"] == str(output_video.resolve())


def test_run_manifest_sequence_inference_writes_frame_sequence_mp4(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import edge_lipsync.inference as inference
    from edge_lipsync.inference import run_manifest_sequence_inference

    dataset_root = tmp_path / "dataset"
    manifest = _write_sequence_fixture_dataset(dataset_root, frame_count=3)
    checkpoint = tmp_path / "model.pt"
    wav_path = tmp_path / "sample.wav"
    wenet_path = tmp_path / "wenet.onnx"
    _write_checkpoint(checkpoint)
    wav_path.write_bytes(b"wav")
    wenet_path.write_bytes(b"onnx")
    commands: list[list[str]] = []

    def fake_extract_bnf_windows_from_wav(_wav: str | Path, _wenet: str | Path) -> np.ndarray:
        return np.full((30, 20, 256), 0.25, dtype=np.float32)

    def fake_require_tool(name: str) -> str:
        assert name == "ffmpeg"
        return "/usr/bin/ffmpeg"

    def fake_run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        inference,
        "extract_bnf_windows_from_wav",
        fake_extract_bnf_windows_from_wav,
        raising=False,
    )
    monkeypatch.setattr(inference, "_require_tool", fake_require_tool, raising=False)
    monkeypatch.setattr(inference, "_run_command", fake_run_command, raising=False)

    artifacts = run_manifest_sequence_inference(
        dataset_root=dataset_root,
        manifest=manifest,
        out_dir=tmp_path / "infer_sequence",
        split=None,
        checkpoint=checkpoint,
        init_bin="",
        hf_model_repo="",
        hf_model_filename="best.pt",
        hf_cache_dir="",
        audio_wav=wav_path,
        wenet_onnx=wenet_path,
        output_mp4="sequence.mp4",
        device=torch.device("cpu"),
    )

    frames_dir = tmp_path / "infer_sequence" / "frames"
    output_video = tmp_path / "infer_sequence" / "sequence.mp4"
    assert [path.name for path in sorted(frames_dir.glob("*.png"))] == [
        "000001.png",
        "000002.png",
        "000003.png",
    ]
    assert artifacts["output_mp4_path"] == str(output_video.resolve())
    assert output_video.exists()
    assert len(commands) == 1
    assert "-loop" not in commands[0]
    assert str(tmp_path / "infer_sequence" / "_video_no_audio.mp4") in commands[0]
    assert str(wav_path) in commands[0]
    assert commands[0][commands[0].index("-preset") + 1] == "medium"
    assert commands[0][commands[0].index("-crf") + 1] == "21"
    assert commands[0][commands[0].index("-ar") + 1] == "48000"
    assert commands[0][commands[0].index("-ac") + 1] == "2"
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["kind"] == "manifest_sequence_inference"
    assert metadata["frame_count"] == 3
    assert metadata["artifacts"]["output_mp4"] == str(output_video.resolve())


def test_run_video_inference_accepts_silent_video_and_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import edge_lipsync.inference as inference
    from edge_lipsync.inference import run_video_inference

    input_video = tmp_path / "silent.mp4"
    input_audio = tmp_path / "speech.wav"
    wenet_path = tmp_path / "wenet.onnx"
    input_video.write_bytes(b"video")
    input_audio.write_bytes(b"audio")
    wenet_path.write_bytes(b"onnx")
    frame_1 = np.full((240, 320, 3), 80, dtype=np.uint8)
    frame_2 = np.full((240, 320, 3), 120, dtype=np.uint8)
    commands: list[tuple[str, str]] = []

    class FakeRuntime:
        def predict(self, face: np.ndarray, audio: np.ndarray) -> np.ndarray:
            assert face.shape == (6, 160, 160)
            assert audio.shape == (20, 256)
            return np.zeros((3, 160, 160), dtype=np.float32)

    class FakeDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect_bbox(self, _frame_bgr: np.ndarray):
            self.calls += 1
            if self.calls == 1:
                return (80, 40, 240, 200)
            return None

        def close(self) -> None:
            pass

    def fake_extract_video_frames(
        _input_video: str | Path,
        frames_dir: str | Path,
        *,
        fps: float,
    ) -> int:
        assert fps == 25.0
        frame_dir = Path(frames_dir)
        frame_dir.mkdir(parents=True)
        cv2.imwrite(str(frame_dir / "000001.png"), frame_1)
        cv2.imwrite(str(frame_dir / "000002.png"), frame_2)
        return 2

    def fake_normalize_audio(
        audio_path: str | Path,
        out_path: str | Path,
        *,
        sample_rate: int,
    ) -> Path:
        assert Path(audio_path) == input_audio
        assert sample_rate == 16000
        normalized = Path(out_path)
        normalized.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_bytes(b"wav")
        return normalized

    def fake_write_mp4(
        *,
        frames_dir: str | Path,
        audio_wav: str | Path,
        out_path: str | Path,
        fps: float,
    ) -> Path:
        assert Path(frames_dir).is_dir()
        assert Path(audio_wav).name == "audio_16k.wav"
        assert fps == 25.0
        output = Path(out_path)
        output.write_bytes(b"mp4")
        commands.append((str(frames_dir), str(audio_wav)))
        return output

    monkeypatch.setattr(
        inference,
        "_extract_video_frames",
        fake_extract_video_frames,
        raising=False,
    )
    monkeypatch.setattr(
        inference,
        "_normalize_audio_for_inference",
        fake_normalize_audio,
        raising=False,
    )
    monkeypatch.setattr(
        inference,
        "extract_bnf_windows_from_wav",
        lambda _wav, _wenet: np.zeros((30, 20, 256), dtype=np.float32),
    )
    monkeypatch.setattr(
        inference,
        "_load_prediction_runtime",
        lambda **_kwargs: (FakeRuntime(), {"source": "fake", "backend": "torch"}),
    )
    monkeypatch.setattr(
        inference,
        "_create_video_bbox_detector",
        lambda **_kwargs: FakeDetector(),
        raising=False,
    )
    monkeypatch.setattr(inference, "write_frame_sequence_audio_mp4", fake_write_mp4)

    artifacts = run_video_inference(
        input_video=input_video,
        audio=input_audio,
        out_dir=tmp_path / "infer_video",
        checkpoint=tmp_path / "best.pt",
        init_bin="",
        hf_model_repo="",
        hf_model_filename="best.pt",
        hf_cache_dir="",
        wenet_onnx=wenet_path,
        output_mp4="output.mp4",
        fps=25.0,
        device=torch.device("cpu"),
    )

    frames_dir = tmp_path / "infer_video" / "frames"
    output_video = tmp_path / "infer_video" / "output.mp4"
    assert artifacts["output_mp4_path"] == str(output_video.resolve())
    assert output_video.exists()
    assert commands == [(str(frames_dir), str(tmp_path / "infer_video" / "audio_16k.wav"))]
    restored_1 = cv2.imread(str(frames_dir / "000001.png"), cv2.IMREAD_COLOR)
    restored_2 = cv2.imread(str(frames_dir / "000002.png"), cv2.IMREAD_COLOR)
    assert restored_1 is not None
    assert restored_2 is not None
    assert not np.array_equal(restored_1, frame_1)
    assert np.array_equal(restored_2, frame_2)
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["kind"] == "video_inference"
    assert metadata["frame_count"] == 2
    assert metadata["processed_frame_count"] == 1
    assert metadata["skipped_frame_count"] == 1
    assert metadata["frames"][1]["status"] == "skipped"
    assert metadata["frames"][1]["reason"] == "face_detection_failed"


def test_run_hf_dataset_sequence_inference_writes_video_without_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import edge_lipsync.inference as inference
    from edge_lipsync.inference import run_hf_dataset_sequence_inference

    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint)
    frame_1 = np.full((240, 320, 3), 80, dtype=np.uint8)
    frame_2 = np.full((240, 320, 3), 120, dtype=np.uint8)
    frame_other = np.full((240, 320, 3), 200, dtype=np.uint8)
    loaded_dataset = {
        "val": [
            {
                "clip_id": "clip_001",
                "frame_idx": 2,
                "audio_idx": 2,
                "frame": frame_2,
                "bbox_xyxy": [80, 40, 240, 200],
                "audio": np.zeros((20, 256), dtype=np.float32),
                "flags": [],
            },
            {
                "clip_id": "clip_001",
                "frame_idx": 1,
                "audio_idx": 1,
                "frame": frame_1,
                "bbox_xyxy": [80, 40, 240, 200],
                "audio": np.zeros((20, 256), dtype=np.float32),
                "flags": ["interpolated_bbox"],
            },
            {
                "clip_id": "clip_002",
                "frame_idx": 1,
                "audio_idx": 1,
                "frame": frame_other,
                "bbox_xyxy": [80, 40, 240, 200],
                "audio": np.zeros((20, 256), dtype=np.float32),
                "flags": [],
            },
        ]
    }
    calls: list[tuple[str, str]] = []

    def fake_load_processed_dataset(repo_id: str, *, cache_dir: str = "") -> object:
        calls.append((repo_id, cache_dir))
        return loaded_dataset

    monkeypatch.setattr(inference, "load_processed_dataset", fake_load_processed_dataset)

    artifacts = run_hf_dataset_sequence_inference(
        hf_dataset_repo="owner/avatar-data",
        split="val",
        clip_id="auto",
        out_dir=tmp_path / "hf_sequence",
        max_frames=0,
        checkpoint=checkpoint,
        init_bin="",
        hf_model_repo="",
        hf_model_filename="best.pt",
        hf_cache_dir="/cache",
        output_mp4="sequence.mp4",
        device=torch.device("cpu"),
    )

    frames_dir = tmp_path / "hf_sequence" / "frames"
    output_video = tmp_path / "hf_sequence" / "sequence.mp4"
    assert calls == [("owner/avatar-data", "/cache")]
    assert [path.name for path in sorted(frames_dir.glob("*.png"))] == [
        "000001.png",
        "000002.png",
    ]
    assert output_video.exists()
    metadata = json.loads(Path(artifacts["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["kind"] == "hf_dataset_sequence_inference"
    assert metadata["hf_dataset_repo"] == "owner/avatar-data"
    assert metadata["split"] == "val"
    assert metadata["clip_id"] == "clip_001"
    assert [frame["frame_idx"] for frame in metadata["frames"]] == [1, 2]
    assert metadata["frames"][0]["flags"] == ["interpolated_bbox"]
    assert metadata["artifacts"]["output_mp4"] == str(output_video.resolve())


def test_infer_manifest_sample_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/infer_manifest_sample.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run inference for one manifest sample" in result.stdout
    assert "--init-bin" in result.stdout
    assert "--hf-model-repo" in result.stdout
    assert "--audio-wav" in result.stdout
    assert "--wenet-onnx" in result.stdout
    assert "--alpha-bin" in result.stdout
    assert "--backend" in result.stdout
    assert "--ncnn-param" in result.stdout
    assert "--output-mp4" in result.stdout


def test_infer_hf_dataset_sequence_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/infer_hf_dataset_sequence.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run inference for a Hugging Face dataset sequence" in result.stdout
    assert "--hf-dataset-repo" in result.stdout
    assert "--clip-id" in result.stdout
    assert "--max-frames" in result.stdout
    assert "--output-mp4" in result.stdout
    assert "--audio-wav" in result.stdout


def test_infer_manifest_sequence_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/infer_manifest_sequence.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run inference for a manifest sequence" in result.stdout
    assert "--output-mp4" in result.stdout
    assert "--max-frames" in result.stdout
    assert "--alpha-bin" in result.stdout
    assert "--backend" in result.stdout
    assert "--ncnn-param" in result.stdout


def test_infer_video_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/infer_video.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run inference from a silent input video and driving audio" in result.stdout
    assert "--input-video" in result.stdout
    assert "--audio" in result.stdout
    assert "--output-mp4" in result.stdout
    assert "--landmark-model-asset-path" in result.stdout
