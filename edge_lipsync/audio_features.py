from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

BLOCK_SAMPLES = 640
TARGET_SAMPLE_RATE = 16000
BNF_WINDOW = 20
SESSION_CONTEXT_BLOCKS = 10
SESSION_FIRST_BLOCKS = 20
SESSION_MAX_BLOCKS = 50


class WenetSession(Protocol):
    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class SessionItem:
    start_block: int
    num_blocks: int
    mmfcc: np.ndarray

    @property
    def end_block(self) -> int:
        return self.start_block + self.num_blocks


def load_wav_mono_f32(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"Only PCM16 wav is supported, got sample_width={sample_width}")
    if channels <= 0 or sample_rate <= 0:
        raise ValueError(f"Invalid wav metadata: channels={channels} sample_rate={sample_rate}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = audio / 32768.0
    if len(audio) == 0:
        raise ValueError("Audio is empty")

    if sample_rate != target_sr:
        old_idx = np.arange(len(audio), dtype=np.float32)
        new_len = int(round(len(audio) * target_sr / sample_rate))
        new_idx = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
        audio = np.interp(new_idx, old_idx, audio).astype(np.float32)
    return np.ascontiguousarray(audio.astype(np.float32))


def split_audio_blocks(audio: np.ndarray, block_samples: int = BLOCK_SAMPLES) -> np.ndarray:
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio [N], got shape={audio.shape}")
    total_blocks = int(math.ceil(len(audio) / block_samples))
    if total_blocks <= 0:
        raise ValueError("Audio is empty")
    padded = np.zeros(total_blocks * block_samples, dtype=np.float32)
    padded[: len(audio)] = audio.astype(np.float32)
    return np.ascontiguousarray(padded.reshape(total_blocks, block_samples))


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if np.isscalar(freq):
        mel = (freq - f_min) / f_sp
        if freq >= min_log_hz:
            mel = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mel
    freqs = np.asarray(freq, dtype=np.float64)
    mels = (freqs - f_min) / f_sp
    mask = freqs >= min_log_hz
    mels[mask] = min_log_mel + np.log(freqs[mask] / min_log_hz) / logstep
    return mels


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    freqs = mels * f_sp + f_min
    mask = mels >= min_log_mel
    freqs[mask] = np.exp((mels[mask] - min_log_mel) * logstep) * min_log_hz
    return freqs


def make_mel_basis(
    sr: int = TARGET_SAMPLE_RATE,
    n_fft: int = 1024,
    n_mels: int = 80,
) -> np.ndarray:
    f_max = sr / 2.0
    fftfreqs = np.linspace(0.0, f_max, 1 + n_fft // 2, dtype=np.float64)
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(f_max), n_mels + 2, dtype=np.float64)
    mel_f = mel_to_hz(mels)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float64)
    for index in range(n_mels):
        lower = -ramps[index] / fdiff[index]
        upper = ramps[index + 2] / fdiff[index + 1]
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float64)


def wav_to_mel80(audio: np.ndarray) -> np.ndarray:
    n_fft = 1024
    hop_length = 160
    win_length = 800
    preemphasis = 0.97
    ref_db = 20.0
    n_mels = 80
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio [N], got {audio.shape}")
    if len(audio) == 0:
        raise ValueError("Audio is empty")

    x = np.zeros_like(audio, dtype=np.float32)
    if len(audio) > 1:
        x[1:] = audio[1:] - preemphasis * audio[:-1]
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)), mode="constant")
    pad_mode = "reflect" if len(x) > 1 else "constant"
    x = np.pad(x, (n_fft // 2, n_fft // 2), mode=pad_mode)
    hann = np.zeros(n_fft, dtype=np.float64)
    insert = (n_fft - win_length) // 2
    k = np.arange(1, win_length + 1, dtype=np.float64)
    hann[insert : insert + win_length] = 0.5 * (
        1.0 - np.cos(2.0 * np.pi * k / (win_length + 1.0))
    )
    n_frames = (len(x) - n_fft) // hop_length + 1
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for index in range(n_frames):
        start = index * hop_length
        frame = x[start : start + n_fft].astype(np.float64) * hann
        fftv = np.fft.rfft(frame, n=n_fft)
        spec[:, index] = np.abs(fftv) ** 2
    mel = make_mel_basis(TARGET_SAMPLE_RATE, n_fft, n_mels) @ spec
    mel = 10.0 * np.log10(mel + 1e-5) - ref_db
    return np.ascontiguousarray(mel.T.astype(np.float32))


def get_bnf_window(bnf: np.ndarray, audio_idx: int, window: int = BNF_WINDOW) -> np.ndarray:
    if bnf.ndim == 3:
        if bnf.shape[1:] != (window, 256) or bnf.shape[0] <= 0:
            raise ValueError(f"Expected BNF windows [T,{window},256], got {bnf.shape}")
        row = min(max(0, int(audio_idx)), bnf.shape[0] - 1)
        return np.ascontiguousarray(bnf[row].astype(np.float32))
    if bnf.ndim != 2 or bnf.shape[1] != 256:
        raise ValueError(f"Expected BNF shape [T,256], got {bnf.shape}")
    if bnf.shape[0] < window:
        raise ValueError(f"BNF must contain at least {window} rows, got {bnf.shape[0]}")
    row = max(0, int(audio_idx))
    if row + window > bnf.shape[0]:
        row = bnf.shape[0] - window
    return np.ascontiguousarray(bnf[row : row + window].astype(np.float32))


def split_blocks_like_session(
    total_blocks: int,
    min_count: int = SESSION_FIRST_BLOCKS,
    max_count: int = SESSION_MAX_BLOCKS,
) -> list[tuple[int, int]]:
    if total_blocks <= 0:
        return []
    items: list[tuple[int, int]] = []
    current = 0
    if total_blocks >= min_count:
        items.append((current, min_count))
        current += min_count
    while total_blocks - current >= max_count:
        items.append((current, max_count))
        current += max_count
    if current < total_blocks:
        items.append((current, total_blocks - current))
    return items


def _run_wenet_item(
    blocks: np.ndarray,
    previous_blocks: np.ndarray | None,
    min_offset: int,
    session: WenetSession,
) -> np.ndarray:
    pcm_block = int(blocks.shape[0])
    previous_count = 0 if previous_blocks is None else int(previous_blocks.shape[0])
    all_count = min_offset + pcm_block + 2 * SESSION_CONTEXT_BLOCKS
    pcm_all = np.zeros((all_count, BLOCK_SAMPLES), dtype=np.float32)
    context_end = min_offset + SESSION_CONTEXT_BLOCKS
    if previous_count > 0:
        pcm_all[context_end - previous_count : context_end] = previous_blocks
    pcm_all[context_end : context_end + pcm_block] = blocks

    mel = wav_to_mel80(pcm_all.reshape(-1))
    encoder_out = session.run(
        ["encoder_out"],
        {
            "speech": mel[None, :, :].astype(np.float32),
            "speech_lengths": np.asarray([mel.shape[0]], dtype=np.int32),
        },
    )[0][0]
    bnf_all_count = int(mel.shape[0] * 0.25 - 0.75)
    encoder_out = encoder_out[:bnf_all_count]
    mmfcc = encoder_out[min_offset : min_offset + pcm_block + BNF_WINDOW - 1]
    if mmfcc.shape != (pcm_block + BNF_WINDOW - 1, 256):
        raise ValueError(
            "Unexpected Wenet output shape after session slicing: "
            f"expected={(pcm_block + BNF_WINDOW - 1, 256)} actual={mmfcc.shape}"
        )
    return np.ascontiguousarray(mmfcc.astype(np.float32))


def build_session_items_from_audio(
    audio: np.ndarray,
    session: WenetSession,
) -> tuple[list[SessionItem], int]:
    blocks = split_audio_blocks(audio)
    chunks = split_blocks_like_session(len(blocks))
    items: list[SessionItem] = []
    previous_tail: np.ndarray | None = None
    for index, (start, count) in enumerate(chunks):
        current = blocks[start : start + count]
        min_offset = 0 if index == 0 else SESSION_CONTEXT_BLOCKS
        mmfcc = _run_wenet_item(current, previous_tail, min_offset, session)
        items.append(SessionItem(start_block=start, num_blocks=count, mmfcc=mmfcc))
        previous_tail = current[-min(SESSION_CONTEXT_BLOCKS, count) :]
    return items, len(blocks)


def get_session_window(items: list[SessionItem], audio_idx: int) -> np.ndarray:
    for item in items:
        if item.start_block <= audio_idx < item.end_block:
            local_index = audio_idx - item.start_block
            row = local_index - 1 if local_index > 0 else 0
            if row + BNF_WINDOW > item.mmfcc.shape[0]:
                row = max(0, item.mmfcc.shape[0] - BNF_WINDOW)
            return np.ascontiguousarray(item.mmfcc[row : row + BNF_WINDOW].astype(np.float32))
    raise ValueError(f"audio_idx={audio_idx} not covered by session items")


def build_bnf_windows_from_audio(audio: np.ndarray, session: WenetSession) -> np.ndarray:
    items, total_blocks = build_session_items_from_audio(audio, session)
    return np.ascontiguousarray(
        np.stack([get_session_window(items, index) for index in range(total_blocks)]).astype(
            np.float32
        )
    )


def extract_bnf_windows_from_wav(wav_path: str | Path, wenet_onnx: str | Path) -> np.ndarray:
    audio = load_wav_mono_f32(wav_path)
    session = ort.InferenceSession(str(wenet_onnx), providers=["CPUExecutionProvider"])
    return build_bnf_windows_from_audio(audio, session)
