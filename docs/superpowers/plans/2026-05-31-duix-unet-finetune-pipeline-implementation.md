# Duix UNet Fine-Tune Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean Python package and CLI pipeline that ports the current Duix UNet, builds supervised datasets from synchronized videos, fine-tunes from existing weights, and renders validation clips.

**Architecture:** Keep `DuixUNet` unchanged for phase 1 and isolate all reusable behavior in the `edge_lipsync` package. CLIs in `tools/` remain thin wrappers around package modules. Large assets stay outside git and are referenced through YAML config paths.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, OpenCV, ONNX Runtime, PyYAML, pytest, ffmpeg/ffprobe.

---

## File Structure

Create this structure under `/Users/monkira/Workspace/RnD/edge-lipsync-model`:

```text
edge_lipsync/
  __init__.py              # package exports
  audio_features.py        # wav loading/resampling, mel, Wenet BNF, BNF windows
  checkpoint.py            # checkpoint payload helpers
  dataset.py               # manifest schema and PyTorch dataset
  eval.py                  # eval grids and render helpers
  losses.py                # Charbonnier and mouth-weighted losses
  model.py                 # unchanged DuixUNet port from Duix-Mobile/model.py
  preprocess.py            # bbox validation, face crop/mask/target tensors
  training.py              # train loop, validation, checkpointing
tools/
  build_dataset.py         # dataset builder CLI
  export_checkpoint.py     # NCNN bin to PyTorch checkpoint CLI
  render_eval.py           # render/eval CLI
  train.py                 # training CLI
configs/
  dataset.example.yaml     # dataset build config
  train.example.yaml       # training config
tests/
  conftest.py
  test_audio_features.py
  test_checkpoint.py
  test_dataset.py
  test_losses.py
  test_model.py
  test_preprocess.py
```

---

### Task 1: Repository Hygiene And Package Skeleton

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/.gitignore`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/README.md`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/pyproject.toml`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/__init__.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/conftest.py`

- [ ] **Step 1: Write the failing package import test**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_package.py`:

```python
from __future__ import annotations


def test_package_imports() -> None:
    import edge_lipsync

    assert edge_lipsync.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_package.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'edge_lipsync'`.

- [ ] **Step 3: Add project metadata and package skeleton**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "edge-lipsync-model"
version = "0.1.0"
description = "Edge-oriented Duix UNet fine-tuning and evaluation pipeline"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "opencv-python>=4.8",
  "onnxruntime>=1.17",
  "pyyaml>=6.0",
  "torch>=2.2",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "ruff>=0.5",
]

[tool.setuptools.packages.find]
include = ["edge_lipsync*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/.gitignore`:

```gitignore
.DS_Store
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/

data/
datasets/
artifacts/
runs/
checkpoints/
models/
*.pt
*.pth
*.ckpt
*.onnx
*.mp4
*.mov
*.wav
*.npy
*.npz
```

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/README.md`:

```markdown
# edge-lipsync-model

Clean training and evaluation pipeline for an edge-oriented Duix UNet lip-sync model.

Phase 1 keeps the current Duix UNet architecture unchanged, initializes from existing Duix weights, and fine-tunes on supervised synchronized avatar videos.

Large assets are not committed. Keep raw videos, datasets, Wenet ONNX files, checkpoints, and renders outside git and reference them through config files.
```

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/__init__.py`:

```python
from __future__ import annotations

__version__ = "0.1.0"
```

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/conftest.py`:

```python
from __future__ import annotations
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_package.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md pyproject.toml edge_lipsync/__init__.py tests/conftest.py tests/test_package.py
git commit -m "chore: scaffold python package"
```

---

### Task 2: Port The Current Duix UNet And Checkpoint API

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/model.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_model.py`

- [ ] **Step 1: Write failing model tests**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_model.py`:

```python
from __future__ import annotations

from pathlib import Path

import torch


def test_duix_unet_forward_shape() -> None:
    from edge_lipsync.model import DuixUNet

    model = DuixUNet().eval()
    face = torch.zeros(1, 6, 160, 160)
    audio = torch.zeros(1, 20, 256)

    with torch.no_grad():
        out = model(face, audio)

    assert tuple(out.shape) == (1, 3, 160, 160)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_duix_unet_accepts_four_dim_audio() -> None:
    from edge_lipsync.model import DuixUNet

    model = DuixUNet().eval()
    face = torch.zeros(1, 6, 160, 160)
    audio = torch.zeros(1, 1, 20, 256)

    with torch.no_grad():
        out = model(face, audio)

    assert tuple(out.shape) == (1, 3, 160, 160)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    from edge_lipsync.model import DuixUNet, load_ckpt, save_ckpt

    ckpt_path = tmp_path / "model.pt"
    model = DuixUNet().eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 6, 160, 160), torch.zeros(1, 20, 256))

    save_ckpt(model, ckpt_path, face_size=160, extra={"test": True})
    loaded = load_ckpt(ckpt_path)

    assert isinstance(loaded, DuixUNet)
    assert ckpt_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_model.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'edge_lipsync.model'`.

- [ ] **Step 3: Port the verified model source**

Run this exact copy command, then inspect the copied file:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
cp /Users/monkira/Workspace/RnD/Duix-Mobile/model.py edge_lipsync/model.py
python - <<'PY'
from pathlib import Path
p = Path("edge_lipsync/model.py")
text = p.read_text()
assert "class DuixUNet" in text
assert "def load_ncnn_bin" in text
assert "def save_ckpt" in text
assert "def load_ckpt" in text
assert "Duix-Mobile" not in text
PY
```

Keep the architecture and weight-loading logic unchanged. If the copied file has a command-line demo under `if __name__ == "__main__":`, leave it in place because it is harmless and useful for manual smoke checks.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_model.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/model.py tests/test_model.py
git commit -m "feat(model): port duix unet"
```

---

### Task 3: Add Shared Face Preprocessing

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/preprocess.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_preprocess.py`

- [ ] **Step 1: Write failing preprocessing tests**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_preprocess.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


def test_make_face_training_sample_shapes_and_ranges() -> None:
    from edge_lipsync.preprocess import make_face_training_sample

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 100
    frame[:, :, 2] = 220

    sample = make_face_training_sample(frame, (80, 40, 240, 200))

    assert sample.face.shape == (6, 160, 160)
    assert sample.target.shape == (3, 160, 160)
    assert sample.roi_168_bgr.shape == (168, 168, 3)
    assert sample.face.dtype == np.float32
    assert sample.target.dtype == np.float32
    assert float(sample.face.min()) >= -1.0
    assert float(sample.face.max()) <= 1.0
    assert float(sample.target.min()) >= -1.0
    assert float(sample.target.max()) <= 1.0


def test_make_face_training_sample_rejects_invalid_bbox() -> None:
    from edge_lipsync.preprocess import make_face_training_sample

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid bbox"):
        make_face_training_sample(frame, (50, 50, 50, 80))


def test_adjust_bbox_clips_to_frame() -> None:
    from edge_lipsync.preprocess import adjust_bbox

    box = adjust_bbox((10, 20, 110, 120), (100, 100, 3), dx=-20, dy=-30, scale=2.0)

    assert box == (0, 0, 100, 100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_preprocess.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'edge_lipsync.preprocess'`.

- [ ] **Step 3: Implement preprocessing**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/preprocess.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

BBox = tuple[int, int, int, int]

ROI_SOURCE_SIZE = 168
ROI_EDGE = 4
FACE_SIZE = 160
MASK_X = 5
MASK_Y = 5
MASK_W = 150
MASK_H = 145


@dataclass(frozen=True)
class FaceTrainingSample:
    face: np.ndarray
    target: np.ndarray
    roi_168_bgr: np.ndarray
    bbox_xyxy: BBox


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    return ((rgb.astype(np.float32) - 127.5) / 127.5).astype(np.float32)


def validate_bbox(bbox: Sequence[int], frame_shape: tuple[int, int, int]) -> BBox:
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox length: {bbox}")
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox with non-positive area: {(x1, y1, x2, y2)}")
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(f"Invalid bbox outside frame: {(x1, y1, x2, y2)} frame={(w, h)}")
    return x1, y1, x2, y2


def adjust_bbox(
    bbox: BBox,
    frame_shape: tuple[int, int, int],
    dx: int = 0,
    dy: int = 0,
    scale: float = 1.0,
) -> BBox:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    nbw = max(32, int(round(bw * scale)))
    nbh = max(32, int(round(bh * scale)))
    nx1 = int(round(cx - nbw / 2.0)) + dx
    ny1 = int(round(cy - nbh / 2.0)) + dy
    nx2 = nx1 + nbw
    ny2 = ny1 + nbh
    nx1 = max(0, min(nx1, w - 2))
    ny1 = max(0, min(ny1, h - 2))
    nx2 = max(nx1 + 1, min(nx2, w))
    ny2 = max(ny1 + 1, min(ny2, h))
    return nx1, ny1, nx2, ny2


def make_face_training_sample(frame_bgr: np.ndarray, bbox_xyxy: Sequence[int]) -> FaceTrainingSample:
    bbox = validate_bbox(bbox_xyxy, frame_bgr.shape)
    x1, y1, x2, y2 = bbox
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError(f"Invalid bbox produced empty ROI: {bbox}")

    roi_168_bgr = cv2.resize(roi, (ROI_SOURCE_SIZE, ROI_SOURCE_SIZE), interpolation=cv2.INTER_AREA)
    target_bgr = roi_168_bgr[ROI_EDGE : ROI_EDGE + FACE_SIZE, ROI_EDGE : ROI_EDGE + FACE_SIZE].copy()
    masked_bgr = target_bgr.copy()
    cv2.rectangle(
        masked_bgr,
        (MASK_X, MASK_Y),
        (MASK_X + MASK_W - 1, MASK_Y + MASK_H - 1),
        (0, 0, 0),
        -1,
    )

    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

    target_norm = _normalize_rgb(target_rgb)
    masked_norm = _normalize_rgb(masked_rgb)

    face = np.concatenate([target_norm, masked_norm], axis=2).transpose(2, 0, 1)
    target = target_norm.transpose(2, 0, 1)
    return FaceTrainingSample(
        face=np.ascontiguousarray(face.astype(np.float32)),
        target=np.ascontiguousarray(target.astype(np.float32)),
        roi_168_bgr=roi_168_bgr,
        bbox_xyxy=bbox,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_preprocess.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): add duix face sample builder"
```

---

### Task 4: Add Audio Feature Extraction And BNF Windowing

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/audio_features.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_audio_features.py`

- [ ] **Step 1: Write failing audio feature tests**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_audio_features.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


def test_get_bnf_window_clamps_to_valid_range() -> None:
    from edge_lipsync.audio_features import get_bnf_window

    bnf = np.arange(30 * 256, dtype=np.float32).reshape(30, 256)

    start = get_bnf_window(bnf, 0)
    end = get_bnf_window(bnf, 29)

    assert start.shape == (20, 256)
    assert end.shape == (20, 256)
    assert np.array_equal(start, bnf[:20])
    assert np.array_equal(end, bnf[10:30])


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_audio_features.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'edge_lipsync.audio_features'`.

- [ ] **Step 3: Implement audio feature utilities**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/audio_features.py`:

```python
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

BLOCK_SAMPLES = 640
TARGET_SAMPLE_RATE = 16000
BNF_WINDOW = 20


def load_wav_mono_f32(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"Only PCM16 wav is supported, got sample_width={sample_width}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = audio / 32768.0

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
        m = (freq - f_min) / f_sp
        if freq >= min_log_hz:
            m = min_log_mel + math.log(freq / min_log_hz) / logstep
        return m
    freq = np.asarray(freq, dtype=np.float64)
    m = (freq - f_min) / f_sp
    mask = freq >= min_log_hz
    m[mask] = min_log_mel + np.log(freq[mask] / min_log_hz) / logstep
    return m


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


def make_mel_basis(sr: int = TARGET_SAMPLE_RATE, n_fft: int = 1024, n_mels: int = 80) -> np.ndarray:
    f_max = sr / 2.0
    fftfreqs = np.linspace(0.0, f_max, 1 + n_fft // 2, dtype=np.float64)
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(f_max), n_mels + 2, dtype=np.float64)
    mel_f = mel_to_hz(mels)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
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
    x = np.zeros_like(audio, dtype=np.float32)
    if len(audio) > 1:
        x[1:] = audio[1:] - preemphasis * audio[:-1]
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)), mode="constant")
    x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    hann = np.zeros(n_fft, dtype=np.float64)
    insert = (n_fft - win_length) // 2
    k = np.arange(1, win_length + 1, dtype=np.float64)
    hann[insert : insert + win_length] = 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (win_length + 1.0)))
    n_frames = (len(x) - n_fft) // hop_length + 1
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        frame = x[start : start + n_fft].astype(np.float64) * hann
        fftv = np.fft.rfft(frame, n=n_fft)
        spec[:, i] = np.abs(fftv) ** 2
    mel = make_mel_basis(TARGET_SAMPLE_RATE, n_fft, n_mels) @ spec
    mel = 10.0 * np.log10(mel + 1e-5) - ref_db
    return np.ascontiguousarray(mel.T.astype(np.float32))


def get_bnf_window(bnf: np.ndarray, audio_idx: int, window: int = BNF_WINDOW) -> np.ndarray:
    if bnf.ndim != 2 or bnf.shape[1] != 256:
        raise ValueError(f"Expected BNF shape [T,256], got {bnf.shape}")
    if bnf.shape[0] < window:
        raise ValueError(f"BNF must contain at least {window} rows, got {bnf.shape[0]}")
    row = max(0, int(audio_idx))
    if row + window > bnf.shape[0]:
        row = bnf.shape[0] - window
    return np.ascontiguousarray(bnf[row : row + window].astype(np.float32))


def _run_wenet_session(mel: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    out = session.run(
        ["encoder_out"],
        {
            "speech": mel[None, :, :].astype(np.float32),
            "speech_lengths": np.asarray([mel.shape[0]], dtype=np.int32),
        },
    )[0][0]
    return np.ascontiguousarray(out.astype(np.float32))


def extract_bnf_from_mel(mel: np.ndarray, wenet_onnx: str | Path) -> np.ndarray:
    if mel.ndim != 2 or mel.shape[1] != 80:
        raise ValueError(f"Expected mel [T,80], got {mel.shape}")
    session = ort.InferenceSession(str(wenet_onnx), providers=["CPUExecutionProvider"])
    return _run_wenet_session(mel, session)


def extract_bnf_from_wav(wav_path: str | Path, wenet_onnx: str | Path) -> np.ndarray:
    audio = load_wav_mono_f32(wav_path)
    mel = wav_to_mel80(audio)
    return extract_bnf_from_mel(mel, wenet_onnx)
```

This file includes the deterministic mel extraction path used by the dataset builder, so train and inference data use the same 16 kHz, 40 ms alignment assumptions.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_audio_features.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/audio_features.py tests/test_audio_features.py
git commit -m "feat(audio): add bnf window utilities"
```

---

### Task 5: Add Manifest Dataset Loader

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/dataset.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_dataset.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


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


def test_duix_manifest_dataset_loads_sample(tmp_path: Path) -> None:
    from edge_lipsync.dataset import DuixManifestDataset

    manifest = _write_fixture_dataset(tmp_path)
    ds = DuixManifestDataset(tmp_path, manifest, split="train")
    sample = ds[0]

    assert len(ds) == 1
    assert tuple(sample["face"].shape) == (6, 160, 160)
    assert tuple(sample["audio"].shape) == (20, 256)
    assert tuple(sample["target"].shape) == (3, 160, 160)
    assert sample["meta"]["clip_id"] == "clip_001"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'edge_lipsync.dataset'`.

- [ ] **Step 3: Implement manifest dataset**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/dataset.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from edge_lipsync.audio_features import get_bnf_window
from edge_lipsync.preprocess import make_face_training_sample


@dataclass(frozen=True)
class ManifestRecord:
    clip_id: str
    frame_idx: int
    audio_idx: int
    frame_path: str
    bbox_xyxy: tuple[int, int, int, int]
    bnf_path: str
    split: str
    flags: tuple[str, ...]

    @staticmethod
    def from_json(payload: dict[str, Any]) -> "ManifestRecord":
        bbox = payload["bbox_xyxy"]
        if len(bbox) != 4:
            raise ValueError(f"bbox_xyxy must have 4 values: {bbox}")
        return ManifestRecord(
            clip_id=str(payload["clip_id"]),
            frame_idx=int(payload["frame_idx"]),
            audio_idx=int(payload["audio_idx"]),
            frame_path=str(payload["frame_path"]),
            bbox_xyxy=tuple(int(v) for v in bbox),  # type: ignore[arg-type]
            bnf_path=str(payload["bnf_path"]),
            split=str(payload["split"]),
            flags=tuple(str(v) for v in payload.get("flags", [])),
        )


def load_manifest(path: str | Path, split: str | None = None) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = ManifestRecord.from_json(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid manifest line {line_no} in {path}: {exc}") from exc
            if split is None or record.split == split:
                records.append(record)
    return records


class DuixManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset_root: str | Path, manifest_path: str | Path, split: str) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = self.dataset_root / self.manifest_path
        self.records = load_manifest(self.manifest_path, split=split)
        if not self.records:
            raise ValueError(f"No records for split={split!r} in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frame_path = self.dataset_root / record.frame_path
        bnf_path = self.dataset_root / record.bnf_path
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_path)
        if not bnf_path.exists():
            raise FileNotFoundError(bnf_path)
        bnf = np.load(bnf_path)
        face_sample = make_face_training_sample(frame, record.bbox_xyxy)
        audio = get_bnf_window(bnf, record.audio_idx)
        return {
            "face": torch.from_numpy(face_sample.face),
            "audio": torch.from_numpy(audio),
            "target": torch.from_numpy(face_sample.target),
            "meta": {
                "clip_id": record.clip_id,
                "frame_idx": record.frame_idx,
                "audio_idx": record.audio_idx,
                "bbox_xyxy": record.bbox_xyxy,
            },
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_dataset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): load duix manifest samples"
```

---

### Task 6: Add Reconstruction Losses

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/losses.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_losses.py`

- [ ] **Step 1: Write failing loss tests**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_losses.py`:

```python
from __future__ import annotations

import torch


def test_charbonnier_loss_backward() -> None:
    from edge_lipsync.losses import charbonnier_loss

    pred = torch.zeros(2, 3, 160, 160, requires_grad=True)
    target = torch.ones(2, 3, 160, 160)
    loss = charbonnier_loss(pred, target)
    loss.backward()

    assert loss.item() > 0
    assert pred.grad is not None


def test_mouth_weighted_loss_is_larger_for_mouth_error() -> None:
    from edge_lipsync.losses import mouth_weighted_l1

    pred = torch.zeros(1, 3, 160, 160)
    target = torch.zeros(1, 3, 160, 160)
    target[:, :, 20:120, 20:120] = 1.0

    weighted = mouth_weighted_l1(pred, target, mouth_weight=4.0)
    plain = torch.nn.functional.l1_loss(pred, target)

    assert weighted.item() > plain.item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_losses.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'edge_lipsync.losses'`.

- [ ] **Step 3: Implement losses**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/losses.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def mouth_weight_mask(
    device: torch.device,
    dtype: torch.dtype,
    height: int = 160,
    width: int = 160,
    mouth_weight: float = 4.0,
) -> torch.Tensor:
    mask = torch.ones(1, 1, height, width, device=device, dtype=dtype)
    mask[:, :, 5 : 5 + 145, 5 : 5 + 150] = float(mouth_weight)
    return mask


def mouth_weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mouth_weight: float = 4.0,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    mask = mouth_weight_mask(
        pred.device,
        pred.dtype,
        height=pred.shape[-2],
        width=pred.shape[-1],
        mouth_weight=mouth_weight,
    )
    return (F.l1_loss(pred, target, reduction="none") * mask).mean()


def combined_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mouth_weight: float = 4.0,
    mouth_loss_scale: float = 0.5,
) -> torch.Tensor:
    return charbonnier_loss(pred, target) + mouth_loss_scale * mouth_weighted_l1(
        pred,
        target,
        mouth_weight=mouth_weight,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_losses.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/losses.py tests/test_losses.py
git commit -m "feat(training): add reconstruction losses"
```

---

### Task 7: Add Checkpoint Helpers And Export CLI

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/checkpoint.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/export_checkpoint.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint helper tests**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_checkpoint.py`:

```python
from __future__ import annotations

from pathlib import Path

import torch


def test_atomic_torch_save_roundtrip(tmp_path: Path) -> None:
    from edge_lipsync.checkpoint import atomic_torch_save

    out = tmp_path / "payload.pt"
    payload = {"value": torch.tensor([1, 2, 3])}
    atomic_torch_save(payload, out)

    loaded = torch.load(out, map_location="cpu")
    assert loaded["value"].tolist() == [1, 2, 3]
    assert not (tmp_path / "payload.pt.tmp").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_checkpoint.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'edge_lipsync.checkpoint'`.

- [ ] **Step 3: Implement checkpoint helper and export CLI**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/checkpoint.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
```

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/export_checkpoint.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from edge_lipsync.model import DuixUNet, save_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Duix NCNN dh_model.bin to PyTorch checkpoint.")
    parser.add_argument("--init-bin", required=True, help="Path to decrypted dh_model.bin")
    parser.add_argument("--out", required=True, help="Output PyTorch checkpoint")
    parser.add_argument("--face-size", type=int, default=160)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    init_bin = Path(args.init_bin)
    if not init_bin.exists():
        raise FileNotFoundError(init_bin)

    model = DuixUNet().to(args.device).eval()
    stats = model.load_ncnn_bin(init_bin, face_size=args.face_size, device=args.device)
    save_ckpt(
        model,
        args.out,
        face_size=args.face_size,
        extra={"init_bin": str(init_bin.resolve()), "weight_load": stats},
    )

    with torch.no_grad():
        face = torch.zeros(1, 6, args.face_size, args.face_size, device=args.device)
        audio = torch.zeros(1, 20, 256, device=args.device)
        pred = model(face, audio)

    print(f"saved={Path(args.out).resolve()}")
    print(f"weight_load={stats}")
    print(f"sanity_shape={tuple(pred.shape)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_checkpoint.py tests/test_model.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/checkpoint.py tools/export_checkpoint.py tests/test_checkpoint.py
git commit -m "feat(checkpoint): add atomic save and export cli"
```

---

### Task 8: Add Training Loop And Train CLI

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/training.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/train.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/configs/train.example.yaml`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_training.py`

- [ ] **Step 1: Write failing tiny training test**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_training.py`:

```python
from __future__ import annotations

import torch


def test_training_step_updates_parameters() -> None:
    from edge_lipsync.losses import combined_reconstruction_loss
    from edge_lipsync.training import run_train_step

    model = torch.nn.Conv2d(6, 3, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "face": torch.zeros(2, 6, 8, 8),
        "target": torch.ones(2, 3, 8, 8),
    }
    before = model.weight.detach().clone()

    loss = run_train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_fn=combined_reconstruction_loss,
        audio_optional=True,
    )

    assert loss > 0
    assert not torch.equal(before, model.weight.detach())
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_training.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'edge_lipsync.training'`.

- [ ] **Step 3: Implement training loop**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/training.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from edge_lipsync.checkpoint import atomic_torch_save
from edge_lipsync.dataset import DuixManifestDataset
from edge_lipsync.losses import combined_reconstruction_loss
from edge_lipsync.model import DuixUNet, load_ckpt


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    manifest: str
    run_dir: str
    init_bin: str = ""
    init_ckpt: str = ""
    device: str = "cpu"
    batch_size: int = 2
    num_workers: int = 0
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_steps: int = 1000
    validation_interval: int = 100
    checkpoint_interval: int = 100


def _forward_model(model: torch.nn.Module, face: torch.Tensor, audio: torch.Tensor | None) -> torch.Tensor:
    if audio is None:
        return model(face)
    return model(face, audio)


def run_train_step(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    audio_optional: bool = False,
) -> float:
    model.train()
    face = batch["face"].to(device=device, dtype=torch.float32)
    target = batch["target"].to(device=device, dtype=torch.float32)
    audio = batch.get("audio")
    audio_t = None if audio is None else audio.to(device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    pred = _forward_model(model, face, audio_t if not audio_optional else None)
    loss = loss_fn(pred, target)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {float(loss.detach().cpu())}")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        face = batch["face"].to(device=device, dtype=torch.float32)
        audio = batch["audio"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        pred = model(face, audio)
        loss = combined_reconstruction_loss(pred, target)
        losses.append(float(loss.cpu()))
    if not losses:
        raise ValueError("Validation loader produced no batches")
    return sum(losses) / len(losses)


def build_model(config: TrainConfig) -> DuixUNet:
    if bool(config.init_bin) == bool(config.init_ckpt):
        raise ValueError("Set exactly one of init_bin or init_ckpt")
    device = config.device
    if config.init_ckpt:
        return load_ckpt(config.init_ckpt, map_location=device).to(device)
    model = DuixUNet().to(device)
    stats = model.load_ncnn_bin(config.init_bin, face_size=160, device=device)
    if int(stats.get("remaining_bytes", 0)) != 0:
        raise ValueError(f"NCNN bin had remaining bytes after load: {stats}")
    return model


def train(config: TrainConfig) -> Path:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    model = build_model(config).to(device)
    train_ds = DuixManifestDataset(config.dataset_root, config.manifest, split="train")
    val_ds = DuixManifestDataset(config.dataset_root, config.manifest, split="val")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    metrics: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_path = run_dir / "best.pt"
    step = 0
    while step < config.max_steps:
        for batch in train_loader:
            step += 1
            loss = run_train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                loss_fn=combined_reconstruction_loss,
            )
            row: dict[str, float | int] = {"step": step, "train_loss": loss}
            if step % config.validation_interval == 0:
                val_loss = run_validation(model, val_loader, device)
                row["val_loss"] = val_loss
                if val_loss < best_val:
                    best_val = val_loss
                    atomic_torch_save(
                        {
                            "format": "edge_lipsync_duix_unet_train_v1",
                            "state_dict": model.state_dict(),
                            "config": asdict(config),
                            "step": step,
                            "metrics": row,
                        },
                        best_path,
                    )
            if step % config.checkpoint_interval == 0:
                atomic_torch_save(
                    {
                        "format": "edge_lipsync_duix_unet_train_v1",
                        "state_dict": model.state_dict(),
                        "config": asdict(config),
                        "step": step,
                        "metrics": row,
                    },
                    run_dir / f"step_{step:07d}.pt",
                )
            metrics.append(row)
            if step >= config.max_steps:
                break

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not best_path.exists():
        atomic_torch_save(
            {
                "format": "edge_lipsync_duix_unet_train_v1",
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "step": step,
                "metrics": metrics[-1] if metrics else {},
            },
            best_path,
        )
    return best_path
```

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/train.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from edge_lipsync.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DuixUNet from a manifest dataset.")
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = TrainConfig(**payload)
    best = train(config)
    print(f"best_checkpoint={best.resolve()}")


if __name__ == "__main__":
    main()
```

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/configs/train.example.yaml`:

```yaml
dataset_root: /absolute/path/to/data/duix_datasets/avatar_name
manifest: manifest.jsonl
run_dir: /absolute/path/to/runs/avatar_name
init_bin: /absolute/path/to/dh_model.bin
init_ckpt: ""
device: cpu
batch_size: 2
num_workers: 0
learning_rate: 0.00001
weight_decay: 0.0001
max_steps: 1000
validation_interval: 100
checkpoint_interval: 100
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_training.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/training.py tools/train.py configs/train.example.yaml tests/test_training.py
git commit -m "feat(training): add fine-tuning loop"
```

---

### Task 9: Add Dataset Builder CLI

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/build_dataset.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/configs/dataset.example.yaml`

- [ ] **Step 1: Write failing CLI smoke test**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_build_dataset_cli.py`:

```python
from __future__ import annotations

import subprocess
import sys


def test_build_dataset_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Build Duix training dataset" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_build_dataset_cli.py -v
```

Expected: FAIL because `tools/build_dataset.py` does not exist.

- [ ] **Step 3: Implement dataset builder CLI shell**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/build_dataset.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml


@dataclass(frozen=True)
class DatasetBuildConfig:
    raw_video_dir: str
    dataset_root: str
    wenet_onnx: str
    fps: int = 25
    sample_rate: int = 16000
    preview_count: int = 8


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"Required tool not found on PATH: {name}")
    return path


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}")


def normalize_clip(src: Path, out_dir: Path, fps: int, sample_rate: int) -> tuple[Path, Path]:
    require_tool("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out = out_dir / "video_25fps.mp4"
    audio_out = out_dir / "audio.wav"
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"fps={fps}",
        "-an",
        str(video_out),
    ])
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(audio_out),
    ])
    return video_out, audio_out


def extract_frames(video_path: Path, frames_dir: Path) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open normalized video: {video_path}")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        cv2.imwrite(str(frames_dir / f"{count:06d}.jpg"), frame)
    cap.release()
    if count == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return count


def write_empty_quality(path: Path, clip_id: str, frame_count: int) -> None:
    payload = {
        "clip_id": clip_id,
        "frame_count": frame_count,
        "status": "frames_audio_ready",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_dataset(config: DatasetBuildConfig) -> None:
    raw_dir = Path(config.raw_video_dir)
    dataset_root = Path(config.dataset_root)
    clips_dir = dataset_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv"}])
    if not videos:
        raise ValueError(f"No videos found in {raw_dir}")
    summary = []
    for video in videos:
        clip_id = video.stem
        clip_dir = clips_dir / clip_id
        normalized_video, _audio = normalize_clip(video, clip_dir, config.fps, config.sample_rate)
        frame_count = extract_frames(normalized_video, clip_dir / "frames")
        write_empty_quality(clip_dir / "quality.json", clip_id, frame_count)
        summary.append({"clip_id": clip_id, "frame_count": frame_count})
    (dataset_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"processed_clips={len(summary)}")
    print(f"dataset_root={dataset_root.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Duix training dataset from synchronized videos.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    build_dataset(DatasetBuildConfig(**payload))


if __name__ == "__main__":
    main()
```

This task intentionally creates a working normalization/frame extraction builder first. BBox detection and BNF generation are added in the next task so this CLI has a testable checkpoint.

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/configs/dataset.example.yaml`:

```yaml
raw_video_dir: /absolute/path/to/data/raw_videos/avatar_name
dataset_root: /absolute/path/to/data/duix_datasets/avatar_name
wenet_onnx: /absolute/path/to/wenet.onnx
fps: 25
sample_rate: 16000
preview_count: 8
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_build_dataset_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/build_dataset.py configs/dataset.example.yaml tests/test_build_dataset_cli.py
git commit -m "feat(dataset): add dataset builder cli"
```

---

### Task 10: Add BBox Detection, BNF Generation, And Manifest Completion

**Files:**
- Modify: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/build_dataset.py`

- [ ] **Step 1: Write failing manifest and bbox tests**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_manifest_writer.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def test_manifest_writer_creates_train_and_val_records(tmp_path: Path) -> None:
    from tools.build_dataset import smooth_bboxes, write_manifest

    clips = [
        {
            "clip_id": "a",
            "frame_count": 4,
            "valid_frames": [1, 2, 3, 4],
            "bboxes": {"1": [10, 10, 100, 100], "2": [12, 10, 102, 100], "3": [14, 10, 104, 100], "4": [16, 10, 106, 100]},
        },
        {
            "clip_id": "b",
            "frame_count": 4,
            "valid_frames": [1, 2, 3, 4],
            "bboxes": {"1": [20, 20, 120, 120], "2": [22, 20, 122, 120], "3": [24, 20, 124, 120], "4": [26, 20, 126, 120]},
        },
    ]
    write_manifest(tmp_path, clips)

    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    splits = {row["split"] for row in rows}

    assert len(rows) == 8
    assert splits == {"train", "val"}
    assert rows[0]["bbox_xyxy"] == [10, 10, 100, 100]
    assert smooth_bboxes([(10, 10, 100, 100), (20, 20, 120, 120)], radius=1)[0] == (15, 15, 110, 110)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_manifest_writer.py -v
```

Expected: FAIL with `ImportError` or `AttributeError` for `write_manifest`.

- [ ] **Step 3: Add bbox detection, BNF generation, and manifest writer**

Modify `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/build_dataset.py` imports to include NumPy and the audio feature extractor:

```python
import numpy as np

from edge_lipsync.audio_features import extract_bnf_from_wav
```

Then add these functions above `build_dataset`:

```python
BBox = tuple[int, int, int, int]


def detect_largest_face(frame_bgr: np.ndarray) -> BBox | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda rect: int(rect[2]) * int(rect[3]))
    return int(x), int(y), int(x + w), int(y + h)


def smooth_bboxes(bboxes: list[BBox], radius: int = 2) -> list[BBox]:
    if not bboxes:
        return []
    out: list[BBox] = []
    for i in range(len(bboxes)):
        lo = max(0, i - radius)
        hi = min(len(bboxes), i + radius + 1)
        vals = np.asarray(bboxes[lo:hi], dtype=np.float32)
        out.append(tuple(int(round(v)) for v in vals.mean(axis=0)))  # type: ignore[return-value]
    return out


def detect_bboxes_for_frames(frames_dir: Path) -> dict[str, list[int]]:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    detected: list[tuple[int, BBox]] = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        bbox = detect_largest_face(frame)
        if bbox is not None:
            detected.append((int(frame_path.stem), bbox))
    smoothed = smooth_bboxes([bbox for _frame_idx, bbox in detected])
    return {str(frame_idx): list(bbox) for (frame_idx, _), bbox in zip(detected, smoothed)}


def write_manifest(dataset_root: Path, clips: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    total_clips = len(clips)
    for clip_pos, clip in enumerate(clips):
        clip_id = str(clip["clip_id"])
        split = "val" if clip_pos == total_clips - 1 else "train"
        bboxes = dict(clip["bboxes"])  # keys are string frame ids
        valid_frames = [int(v) for v in clip["valid_frames"]]
        for frame_idx in valid_frames:
            bbox = bboxes.get(str(frame_idx))
            if bbox is None:
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "audio_idx": frame_idx - 1,
                    "frame_path": f"clips/{clip_id}/frames/{frame_idx:06d}.jpg",
                    "bbox_xyxy": [int(v) for v in bbox],
                    "bnf_path": f"clips/{clip_id}/bnf.npy",
                    "split": split,
                    "flags": [],
                }
            )

    manifest = dataset_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    splits = {
        "train": sum(1 for row in rows if row["split"] == "train"),
        "val": sum(1 for row in rows if row["split"] == "val"),
    }
    (dataset_root / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
```

Then modify `build_dataset` so each clip generates `bnf.npy`, `bboxes.json`, and valid frame metadata:

```python
    wenet_onnx = Path(config.wenet_onnx)
    if not wenet_onnx.exists():
        raise FileNotFoundError(wenet_onnx)
    summary = []
    for video in videos:
        clip_id = video.stem
        clip_dir = clips_dir / clip_id
        normalized_video, audio_path = normalize_clip(video, clip_dir, config.fps, config.sample_rate)
        frames_dir = clip_dir / "frames"
        frame_count = extract_frames(normalized_video, frames_dir)
        bnf = extract_bnf_from_wav(audio_path, wenet_onnx)
        np.save(clip_dir / "bnf.npy", bnf.astype(np.float32))
        bboxes = detect_bboxes_for_frames(frames_dir)
        (clip_dir / "bboxes.json").write_text(json.dumps(bboxes, indent=2), encoding="utf-8")
        valid_frames = [
            frame_idx
            for frame_idx in range(1, frame_count + 1)
            if str(frame_idx) in bboxes and (frame_idx - 1 + 20) <= int(bnf.shape[0])
        ]
        quality = {
            "clip_id": clip_id,
            "frame_count": frame_count,
            "bnf_shape": list(bnf.shape),
            "detected_bboxes": len(bboxes),
            "valid_samples": len(valid_frames),
            "status": "ready" if valid_frames else "no_valid_samples",
        }
        (clip_dir / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
        summary.append(
            {
                "clip_id": clip_id,
                "frame_count": frame_count,
                "valid_frames": valid_frames,
                "bboxes": bboxes,
            }
        )
    write_manifest(dataset_root, summary)
    (dataset_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
```

Remove the earlier loop body in `build_dataset` that only wrote frame/audio readiness. The final `build_dataset` body must contain one loop: normalize clip, extract frames, compute BNF, detect/smooth bboxes, write quality, append summary, then write manifest.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_manifest_writer.py tests/test_build_dataset_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/build_dataset.py tests/test_manifest_writer.py
git commit -m "feat(dataset): write manifest splits"
```

---

### Task 11: Add Evaluation Grid And Render CLI

**Files:**
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/eval.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/render_eval.py`
- Create: `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_eval.py`

- [ ] **Step 1: Write failing eval test**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tests/test_eval.py`:

```python
from __future__ import annotations

import numpy as np


def test_tensor_chw_to_rgb_u8_shape() -> None:
    from edge_lipsync.eval import chw_norm_to_rgb_u8

    x = np.zeros((3, 160, 160), dtype=np.float32)
    rgb = chw_norm_to_rgb_u8(x)

    assert rgb.shape == (160, 160, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb[0, 0, 0]) == 127
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_eval.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'edge_lipsync.eval'`.

- [ ] **Step 3: Implement eval helpers and CLI**

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/edge_lipsync/eval.py`:

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def chw_norm_to_rgb_u8(chw: np.ndarray) -> np.ndarray:
    if chw.shape[0] != 3:
        raise ValueError(f"Expected CHW with 3 channels, got {chw.shape}")
    hwc = np.transpose(chw, (1, 2, 0))
    return np.clip((hwc + 1.0) * 127.5, 0, 255).astype(np.uint8)


def write_prediction_grid(
    masked_chw: np.ndarray,
    pred_chw: np.ndarray,
    target_chw: np.ndarray,
    out_path: str | Path,
) -> None:
    masked = chw_norm_to_rgb_u8(masked_chw)
    pred = chw_norm_to_rgb_u8(pred_chw)
    target = chw_norm_to_rgb_u8(target_chw)
    diff = np.clip(np.abs(pred.astype(np.int16) - target.astype(np.int16)), 0, 255).astype(np.uint8)
    grid = np.concatenate([masked, pred, target, diff], axis=1)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
```

Create `/Users/monkira/Workspace/RnD/edge-lipsync-model/tools/render_eval.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from edge_lipsync.dataset import DuixManifestDataset
from edge_lipsync.eval import write_prediction_grid
from edge_lipsync.model import load_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Render validation grids for a Duix checkpoint.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = DuixManifestDataset(args.dataset_root, args.manifest, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = load_ckpt(args.ckpt, map_location=args.device).to(args.device).eval()
    out_dir = Path(args.out_dir)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.max_batches:
                break
            face = batch["face"].to(args.device, dtype=torch.float32)
            audio = batch["audio"].to(args.device, dtype=torch.float32)
            pred = model(face, audio).cpu().numpy()[0]
            target = batch["target"].numpy()[0]
            masked = batch["face"].numpy()[0][3:6]
            write_prediction_grid(masked, pred, target, out_dir / f"grid_{idx:04d}.png")

    print(f"out_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest tests/test_eval.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/eval.py tools/render_eval.py tests/test_eval.py
git commit -m "feat(eval): add validation grid rendering"
```

---

### Task 12: Final Verification And Documentation Update

**Files:**
- Modify: `/Users/monkira/Workspace/RnD/edge-lipsync-model/README.md`

- [ ] **Step 1: Add usage documentation**

Replace `/Users/monkira/Workspace/RnD/edge-lipsync-model/README.md` with:

```markdown
# edge-lipsync-model

Clean training and evaluation pipeline for an edge-oriented Duix UNet lip-sync model.

## Phase 1

- Keep the current Duix UNet architecture unchanged.
- Initialize from an existing Duix `dh_model.bin` or exported PyTorch checkpoint.
- Build supervised datasets from synchronized talking-head videos.
- Fine-tune one avatar/persona.
- Evaluate with validation losses and prediction grids.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Export Initial Checkpoint

```bash
python tools/export_checkpoint.py \
  --init-bin /absolute/path/to/dh_model.bin \
  --out /absolute/path/to/checkpoints/init.pt
```

## Build Dataset

```bash
python tools/build_dataset.py --config configs/dataset.example.yaml
```

## Train

```bash
python tools/train.py --config configs/train.example.yaml
```

## Render Validation Grids

```bash
python tools/render_eval.py \
  --dataset-root /absolute/path/to/data/duix_datasets/avatar_name \
  --manifest manifest.jsonl \
  --ckpt /absolute/path/to/runs/avatar_name/best.pt \
  --out-dir /absolute/path/to/runs/avatar_name/eval_grids
```

## Asset Policy

Do not commit raw videos, generated datasets, Wenet ONNX files, checkpoints, rendered videos, or debug artifacts. Keep them outside git and reference them through config files.
```

- [ ] **Step 2: Run all unit tests**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m pytest -v
```

Expected: PASS.

- [ ] **Step 3: Run lint**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
python -m ruff check .
```

Expected: PASS.

- [ ] **Step 4: Inspect git status**

Run:

```bash
cd /Users/monkira/Workspace/RnD/edge-lipsync-model
git status --short
```

Expected: only `README.md` is modified.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add pipeline usage"
```

---

## Self-Review Notes

Spec coverage:

- Clean repo boundary: Task 1 and Task 12.
- Model unchanged/init from Duix weights: Task 2 and Task 7.
- Dataset from synchronized videos: Task 9 and Task 10.
- Supervised dataset loader: Task 5.
- Fine-tuning loop: Task 6 and Task 8.
- Evaluation artifacts: Task 11.
- Asset exclusion: Task 1 `.gitignore` and Task 12 docs.

Implementation notes:

- Task 10 chooses OpenCV Haar face detection as the first bbox baseline because it is already available through OpenCV and keeps the initial repo dependency surface small.
- Wenet BNF generation is wired through `extract_bnf_from_wav`, so generated manifests point at real `bnf.npy` files rather than synthetic features.
