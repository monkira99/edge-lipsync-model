# edge-lipsync-model

Clean training and evaluation pipeline for an edge-oriented Duix UNet lip-sync model.

## Phase 1

- Keep the current Duix UNet architecture unchanged.
- Initialize from an existing Duix `dh_model.bin` or exported PyTorch checkpoint.
- Build supervised datasets from synchronized talking-head videos.
- Fine-tune one avatar/persona.
- Evaluate with validation losses, prediction grids, and a validation MP4.

## Install

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

For landmark-based dataset building, install MediaPipe in the same environment:

```bash
.venv/bin/python -m pip install mediapipe
```

Current MediaPipe Tasks also needs a FaceLandmarker model asset. Download
`face_landmarker.task` from Google's model bucket and set `landmark_model_asset_path` in the
dataset config:

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Verify

```bash
.venv/bin/pyright
.venv/bin/ruff check .
.venv/bin/pytest -q
```

## Export Initial Checkpoint

```bash
.venv/bin/python tools/export_checkpoint.py \
  --init-bin /absolute/path/to/dh_model.bin \
  --out /absolute/path/to/checkpoints/init.pt
```

## Build Dataset

```bash
.venv/bin/python tools/build_dataset.py --config configs/dataset.example.yaml
```

Add `--strict` to fail immediately when any clip fails. Without it, clip-level failures are
recorded in the clip `quality.json` and summarized after the remaining clips are processed.
The default bbox detector is `mediapipe_face_landmarker`, which derives the Duix lower-face ROI
from face landmarks. `haar` remains available only as a debug fallback and produces full-face
boxes, not production Duix ROI boxes.

## Train

```bash
.venv/bin/python tools/train.py --config configs/train.example.yaml
```

Training writes atomic checkpoints plus `metrics.json` and `metrics.csv` curves. Checkpoints
include the dataset manifest hash, training config, step, epoch, metrics, and initialization
source.

## Render Validation Artifacts

```bash
.venv/bin/python tools/render_eval.py --config configs/eval.example.yaml
```

The render command writes prediction grids, `validation_grids.mp4`, numeric validation metrics,
and JSON metadata next to the video.

## Asset Policy

Do not commit raw videos, generated datasets, Wenet ONNX files, Duix character folders,
checkpoints, rendered videos, or debug artifacts. Keep them outside git and reference them
through config files.
