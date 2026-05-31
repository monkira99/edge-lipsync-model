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
