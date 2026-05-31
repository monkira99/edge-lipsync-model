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

## Authenticate External Services

Hugging Face and W&B credentials stay in their SDK login state or environment variables. Do not
write tokens into YAML files, checkpoints, or run metadata.

```bash
export HF_TOKEN=hf_write_token_from_hugging_face
hf auth login --token "$HF_TOKEN"
wandb login
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

## Version Processed Datasets On Hugging Face

Upload a built dataset after inspecting its `build_summary.json` and previews:

```bash
.venv/bin/python tools/hf_dataset.py push \
  --dataset-root /absolute/path/to/data/duix_datasets/avatar_name \
  --repo-id username/avatar-name-dataset
```

Uploads create private dataset repositories by default. Pass `--public` only when the dataset is
intended to be public. Dataset uploads include the manifest, splits, quality metadata, previews,
frames, bboxes, and BNF arrays. They exclude raw source videos and the normalized intermediate
`audio.wav` and `video_25fps.mp4` files.

Pull a pinned revision into the Hugging Face local cache:

```bash
.venv/bin/python tools/hf_dataset.py pull \
  --repo-id username/avatar-name-dataset \
  --revision dataset-v1
```

## Train

```bash
.venv/bin/python tools/train.py --config configs/train.example.yaml
```

Training writes atomic checkpoints, `best.pt`, `final.pt`, `metrics.json`, `metrics.csv`,
`run_metadata.json`, and a model card. Checkpoints include the dataset manifest hash, training
config, step, epoch, metrics, initialization source, dataset revision, and W&B run provenance.

The example config uses local paths. To train from a Hugging Face dataset revision, clear
`dataset_root` and set:

```yaml
dataset_root: ""
hf_dataset_repo: username/avatar-name-dataset
hf_dataset_revision: dataset-v1
```

Hub dataset inputs require a non-empty pinned revision. Training will not silently consume a
moving default branch.

To initialize from a Hugging Face model revision instead of `init_bin` or `init_ckpt`, set:

```yaml
init_bin: ""
init_ckpt: ""
hf_init_model_repo: username/avatar-name-model
hf_init_model_revision: baseline-v1
hf_init_model_filename: best.pt
```

To track a run and publish its final selected artifacts, set:

```yaml
wandb_mode: online
wandb_project: edge-lipsync-model
wandb_run_name: avatar-name-baseline
hf_model_repo: username/avatar-name-model
hf_model_private: true
```

Supported W&B modes are `disabled`, `offline`, and `online`. Local metrics remain available in
all modes. W&B records configuration, per-step losses, validation metrics, phase, learning rate,
Hub provenance, console output, and native system telemetry.

Model publication is optional. When `hf_model_repo` is set, successful training uploads
`best.pt`, `final.pt`, metric curves, run metadata, and the generated model card. Uploads create
private model repositories by default. A failed publication leaves local artifacts intact and
can be retried:

```bash
.venv/bin/python tools/hf_model.py push \
  --run-dir /absolute/path/to/runs/avatar_name \
  --repo-id username/avatar-name-model
```

Pull a historical model checkpoint by revision:

```bash
.venv/bin/python tools/hf_model.py pull \
  --repo-id username/avatar-name-model \
  --revision model-v1 \
  --filename best.pt
```

## Render Validation Artifacts

```bash
.venv/bin/python tools/render_eval.py --config configs/eval.example.yaml
```

The render command writes prediction grids, `validation_grids.mp4`, numeric validation metrics,
and JSON metadata next to the video.

The eval config also supports pinned Hub dataset and model inputs. Clear `dataset_root` and
`ckpt`, then set:

```yaml
dataset_root: ""
ckpt: ""
hf_dataset_repo: username/avatar-name-dataset
hf_dataset_revision: dataset-v1
hf_model_repo: username/avatar-name-model
hf_model_revision: model-v1
hf_model_filename: best.pt
```

## Asset Policy

Do not commit raw videos, generated datasets, Wenet ONNX files, Duix character folders,
checkpoints, rendered videos, or debug artifacts. Keep them outside git and reference them
through config files. Hugging Face stores versioned processed datasets and selected trained model
artifacts; W&B stores experiment history and debugging telemetry.
