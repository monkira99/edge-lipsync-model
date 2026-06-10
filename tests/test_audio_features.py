from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


class _FakeWenetSession:
    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        assert output_names == ["encoder_out"]
        mel = inputs["speech"]
        assert mel.ndim == 3
        rows = int(mel.shape[1] * 0.25 - 0.75)
        out = np.arange(rows * 256, dtype=np.float32).reshape(1, rows, 256)
        return [out]


def test_get_bnf_window_clamps_to_valid_range() -> None:
    from edge_lipsync.audio_features import get_bnf_window

    bnf = np.arange(30 * 256, dtype=np.float32).reshape(30, 256)

    start = get_bnf_window(bnf, 0)
    end = get_bnf_window(bnf, 29)

    assert start.shape == (20, 256)
    assert end.shape == (20, 256)
    assert np.array_equal(start, bnf[:20])
    assert np.array_equal(end, bnf[10:30])


def test_get_bnf_window_reads_precomputed_session_window() -> None:
    from edge_lipsync.audio_features import get_bnf_window

    windows = np.arange(3 * 20 * 256, dtype=np.float32).reshape(3, 20, 256)

    assert np.array_equal(get_bnf_window(windows, 1), windows[1])


def test_get_bnf_window_rejects_short_bnf() -> None:
    from edge_lipsync.audio_features import get_bnf_window

    with pytest.raises(ValueError, match="at least 20"):
        get_bnf_window(np.zeros((19, 256), dtype=np.float32), 0)


def test_split_audio_blocks_pads_to_640_samples() -> None:
    from edge_lipsync.audio_features import split_audio_blocks

    blocks = split_audio_blocks(np.ones(641, dtype=np.float32))

    assert blocks.shape == (2, 640)
    assert blocks.dtype == np.float32
    assert blocks[1, 1] == 0.0


def test_wav_to_mel80_returns_80_bins() -> None:
    from edge_lipsync.audio_features import wav_to_mel80

    audio = np.zeros(16000, dtype=np.float32)
    mel = wav_to_mel80(audio)

    assert mel.ndim == 2
    assert mel.shape[1] == 80
    assert mel.dtype == np.float32


def test_split_blocks_like_session_matches_runtime_chunking() -> None:
    from edge_lipsync.audio_features import split_blocks_like_session

    assert split_blocks_like_session(121) == [(0, 20), (20, 50), (70, 50), (120, 1)]


def test_build_bnf_windows_from_audio_matches_audio_block_count() -> None:
    from edge_lipsync.audio_features import build_bnf_windows_from_audio

    audio = np.zeros(25 * 640, dtype=np.float32)

    windows = build_bnf_windows_from_audio(audio, _FakeWenetSession())

    assert windows.shape == (25, 20, 256)
    assert windows.dtype == np.float32
    assert np.array_equal(windows[0], windows[1])


def _write_silent_wav(path: Path, samples: int = 25 * 640) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.zeros(samples, dtype=np.int16).tobytes())


def test_wenet_runtime_uses_selected_providers_and_reuses_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.audio_features as audio_features
    from edge_lipsync.onnx_runtime import resolve_onnx_providers

    model = tmp_path / "wenet.onnx"
    model.write_bytes(b"onnx")
    wav = tmp_path / "audio.wav"
    _write_silent_wav(wav)
    providers_seen: list[list[str]] = []

    class FakeSession(_FakeWenetSession):
        pass

    def fake_session(_path: str, *, providers: list[str]) -> FakeSession:
        providers_seen.append(providers)
        return FakeSession()

    monkeypatch.setattr(audio_features.ort, "InferenceSession", fake_session)
    selection = resolve_onnx_providers(
        "cuda",
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    runtime = audio_features.WenetRuntime(model, provider_selection=selection)

    first = runtime.extract_wav(wav)
    second = runtime.extract_wav(wav)

    assert first.shape == (25, 20, 256)
    assert np.array_equal(first, second)
    assert providers_seen == [["CUDAExecutionProvider", "CPUExecutionProvider"]]


def test_wenet_runtime_uses_shared_run_limiter(tmp_path: Path) -> None:
    from edge_lipsync.audio_features import WenetRuntime

    model = tmp_path / "wenet.onnx"
    model.write_bytes(b"onnx")
    wav = tmp_path / "audio.wav"
    _write_silent_wav(wav)

    class FakeSession:
        def run(self, _output_names, _inputs):
            raise AssertionError("session.run must be delegated to the limiter")

    class FakeLimiter:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, session, output_names, inputs):
            del session, output_names
            self.calls += 1
            mel = inputs["speech"]
            rows = int(mel.shape[1] * 0.25 - 0.75)
            return [np.zeros((1, rows, 256), dtype=np.float32)]

    limiter = FakeLimiter()
    runtime = WenetRuntime(model, session=FakeSession(), run_limiter=limiter)

    runtime.extract_wav(wav)

    assert limiter.calls > 0
