# Duix UNet Fine-Tune Pipeline Design

Date: 2026-05-31

## Context

This project will live in a new repository:

- Local path: `/Users/monkira/Workspace/RnD/edge-lipsync-model`
- Remote: `https://github.com/monkira99/edge-lipsync-model.git`

The existing `/Users/monkira/Workspace/RnD/Duix-Mobile` repository remains a reference source only. It contains useful reverse-engineering work, a PyTorch `DuixUNet` implementation, NCNN weight loading, Wenet BNF extraction, and render/debug scripts, but it also contains many experiments, generated artifacts, model folders, and local caches. The new repo must start clean and avoid copying experimental clutter.

## Goal

Build a clean training and evaluation pipeline for an edge-oriented lip-sync model that can:

1. Keep the current Duix UNet architecture unchanged in phase 1.
2. Initialize model weights from the existing Duix NCNN/PyTorch weights.
3. Build a supervised training dataset from synchronized avatar videos.
4. Fine-tune the model for one specific avatar/persona.
5. Produce PyTorch checkpoints that can be evaluated through a Python inference/render pipeline.
6. Preserve a path toward Android NCNN/Vulkan deployment later.

The first target platform direction is Android mid/high-end devices. The phase 1 implementation will not export to NCNN yet, but it must avoid choices that make NCNN export unnecessarily difficult later.

## Non-Goals

Phase 1 does not include:

- Changing the model architecture.
- Exporting trained weights back to NCNN.
- Adding diffusion, VAE, DiT, GAN, or large audio/video foundation models.
- Building cloud training orchestration.
- Automatic hyperparameter search.
- Training a multi-identity general model.
- Committing datasets, videos, checkpoints, rendered artifacts, or downloaded third-party model weights.

## Research Summary

Recent high-quality talking-head systems split into two categories:

- Quality-first models such as VASA-1, Sonic, Hallo3, and LatentSync use large latent diffusion, video diffusion, or heavy transformer components. They can produce impressive results, but their runtime and memory profiles are not a good fit for Android edge deployment.
- Edge/mobile-oriented systems such as MobilePortrait show that lightweight U-Net-style renderers, precomputed appearance knowledge, and compact motion representations are better aligned with mobile deployment. MobilePortrait reports mobile real-time performance with low-FLOP U-Net variants, but its architecture is different from Duix and cannot directly reuse the current Duix weights.

For this repo, the practical phase 1 choice is to keep the existing Duix patch-UNet and improve the data/training pipeline. This preserves weight initialization and keeps later Android deployment realistic. Phase 2 can add small compatibility-preserving improvements once phase 1 is measurable.

Reference links:

- VASA-1: https://www.microsoft.com/en-us/research/publication/vasa-1-lifelike-audio-driven-talking-faces-generated-in-real-time/
- MobilePortrait: https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_MobilePortrait_Real-Time_One-Shot_Neural_Head_Avatars_on_Mobile_Devices_CVPR_2025_paper.pdf
- MuseTalk: https://github.com/TMElyralab/MuseTalk
- LatentSync: https://github.com/bytedance/LatentSync
- Sonic: https://github.com/jixiaozhong/Sonic
- Hallo3: https://github.com/fudan-generative-vision/hallo3

## Repository Boundary

The new repository will contain reusable source code, configuration examples, tests, and documentation.

It will not contain:

- Raw videos.
- Generated datasets.
- Wenet ONNX files.
- Duix character folders.
- PyTorch or NCNN model weights.
- Rendered MP4s.
- Debug image dumps.

All large or generated files will be referenced by config paths and ignored by git.

Initial planned layout:

```text
edge-lipsync-model/
  edge_lipsync/
    __init__.py
    model.py
    audio_features.py
    preprocess.py
    dataset.py
    losses.py
    training.py
    eval.py
  tools/
    build_dataset.py
    train.py
    render_eval.py
    export_checkpoint.py
  configs/
    dataset.example.yaml
    train.example.yaml
  docs/
    superpowers/specs/
  tests/
  pyproject.toml
  README.md
```

`tools/*` should remain thin CLI wrappers. Core behavior belongs in `edge_lipsync/*` so it can be tested and reused.

## Model Contract

Phase 1 uses the current Duix UNet behavior:

- Input face tensor: `[B, 6, 160, 160]`
- Input audio tensor: `[B, 20, 256]` or `[B, 1, 20, 256]`
- Output patch tensor: `[B, 3, 160, 160]`
- Output activation: `tanh`, range `[-1, 1]`

The six face channels are:

1. The normalized real RGB face crop.
2. The normalized masked RGB face crop.

Both are concatenated channel-wise. Normalization must match existing Duix inference preprocessing:

```text
norm = (rgb - 127.5) / 127.5
```

The model can initialize from:

- A decrypted Duix NCNN `dh_model.bin`.
- A PyTorch checkpoint exported from `dh_model.bin`.

The implementation should port only the required stable model code from the old repo. The model file should avoid importing from `Duix-Mobile` at runtime.

## Dataset Contract

Input raw data is synchronized talking-head video for one avatar/persona:

```text
data/raw_videos/avatar_name/
  clip_001.mp4
  clip_002.mov
```

Generated dataset layout:

```text
data/duix_datasets/avatar_name/
  manifest.jsonl
  splits.json
  clips/
    clip_001/
      audio.wav
      bnf.npy
      frames/
        000001.jpg
        000002.jpg
      bboxes.json
      quality.json
```

Each manifest line represents one supervised sample:

```json
{
  "clip_id": "clip_001",
  "frame_idx": 123,
  "audio_idx": 123,
  "frame_path": "clips/clip_001/frames/000123.jpg",
  "bbox_xyxy": [149, 370, 540, 960],
  "bnf_path": "clips/clip_001/bnf.npy",
  "split": "train",
  "flags": []
}
```

Paths in the manifest are relative to the dataset root. This makes the dataset movable and keeps configs clean.

The dataset loader creates tensors on demand:

- `face`: from the frame and bbox using the shared preprocessing function.
- `target`: the unmasked RGB target patch from the same ROI/frame.
- `audio`: a 20-step BNF window aligned with the frame/audio index.

## Preprocessing Pipeline

The dataset builder runs these steps per input clip:

1. Validate that the clip has readable video and audio streams.
2. Normalize video to 25 FPS.
3. Normalize audio to mono 16 kHz PCM WAV.
4. Extract frames with deterministic file names.
5. Detect and track face bbox per frame.
6. Smooth bbox trajectories lightly to reduce jitter.
7. Generate Wenet BNF features at 40 ms steps.
8. Build sample records where frame, bbox, target patch, and BNF window are all valid.
9. Write per-clip quality stats and previews.
10. Write dataset-level manifest and split files.

Frame/audio alignment follows the current Duix timing assumption:

- 25 FPS video.
- 16 kHz audio.
- 640 samples per audio block.
- 40 ms per block.
- One render frame maps to one audio block.
- UNet consumes a sliding BNF window of 20 steps.

If a clip has a different source FPS, preprocessing converts it to 25 FPS before sample indexing.

## BBox And ROI Handling

The phase 1 ROI behavior should match the current Duix model:

- Use `xyxy` bbox internally.
- Resize face ROI to `168 x 168`.
- Crop inner area with edge size `4` to produce `160 x 160`.
- Mask the rectangle used by the current Duix preprocessing.

BBox quality gates:

- Drop boxes that are too small, too large, invalid, or outside the frame.
- Interpolate only short missing gaps.
- Drop long missing gaps.
- Drop segments with large discontinuous tracking jumps.
- Keep a small amount of silence/closed-mouth data, but avoid letting silence dominate training.

The builder should output debug previews for a small sample of frames per clip:

- Original frame with bbox overlay.
- Real crop.
- Masked crop.
- Target patch.

## Training Design

Training is supervised fine-tuning:

1. Load `DuixUNet`.
2. Initialize from `--init-bin` or `--init-ckpt`.
3. Read `manifest.jsonl`.
4. Train on batches of `(face, audio, target)`.
5. Validate on held-out clips or held-out time ranges.
6. Save checkpoints and metrics.

Recommended optimizer:

- `AdamW`
- Small learning rate for fine-tuning.
- Mixed precision on CUDA when available.
- CPU/MPS fallback for smoke tests, not full training.

Recommended schedule:

1. Warmup: train decoder/output-heavy parts while keeping the rest conservative.
2. Main fine-tune: unfreeze most or all model parameters with a lower learning rate.
3. Stabilization: reduce learning rate and keep the best validation checkpoint.

The exact epoch/step counts will be implementation-plan decisions because they depend on dataset size and hardware.

## Losses

Phase 1 starts with stable, easy-to-debug losses:

- Patch reconstruction loss: Charbonnier or L1 over the full `160 x 160` patch.
- Mouth-weighted reconstruction loss: higher weight inside the masked/mouth region.
- Optional lightweight perceptual loss if the dependency can be added cleanly.

Phase 1 should not add GAN loss. GAN can sharpen output but introduces instability and makes debugging harder.

Temporal consistency should be measured during validation first. It can become a training loss later if frame-to-frame flicker remains a concrete issue.

## Evaluation

Evaluation must include numeric and visual checks.

Numeric metrics:

- Train reconstruction loss.
- Validation reconstruction loss.
- Mouth-region loss.
- Temporal delta metric on validation sequences.

Visual artifacts:

- Validation MP4 render.
- Grid images with masked input, prediction, target, and absolute diff.
- Dataset previews.
- Training curves as JSON/CSV.

Regression checks:

- Model forward smoke test.
- Dataset one-sample read smoke test.
- Loss backward smoke test.
- Tiny overfit test on 32-128 samples; the loss must decrease.
- Short validation render from a saved checkpoint.

## Checkpoints

Checkpoint payload should include:

- Format/version string.
- Model state dict.
- Training config.
- Dataset root or manifest path.
- Dataset fingerprint or manifest hash.
- Step/epoch.
- Metrics.
- Init weight source metadata.

Checkpoint writes should be atomic:

```text
checkpoint.tmp -> checkpoint.pt
```

This prevents corrupt checkpoints after interrupted training.

## CLI Surface

Planned CLIs:

```bash
python tools/build_dataset.py --config configs/dataset.yaml
python tools/train.py --config configs/train.yaml
python tools/render_eval.py --config configs/eval.yaml --ckpt runs/avatar/best.pt
python tools/export_checkpoint.py --init-bin /path/to/dh_model.bin --out /path/to/model.pt
```

The CLIs should print clear summaries:

- Number of clips processed.
- Number of valid samples.
- Drop counts by reason.
- Split counts.
- Checkpoint path.
- Evaluation artifact paths.

## Configuration

Config files should be YAML examples committed to git. Real configs can live outside git or be copied locally.

Example dataset config fields:

- `raw_video_dir`
- `dataset_root`
- `wenet_onnx`
- `fps`
- `sample_rate`
- `split_strategy`
- `bbox_detector`
- `preview_count`

Example train config fields:

- `dataset_root`
- `manifest`
- `init_bin` or `init_ckpt`
- `run_dir`
- `batch_size`
- `num_workers`
- `learning_rate`
- `weight_decay`
- `max_steps`
- `validation_interval`
- `checkpoint_interval`
- `device`
- `precision`

## Error Handling

Dataset builder:

- A failed clip should not fail the whole build unless `--strict` is enabled.
- Clip-level failures are recorded in `quality.json`.
- Dataset-level summary reports all failures and drop reasons.

Training:

- Fail early if init weight does not exist.
- Fail early if train or validation split is empty.
- Validate sample tensor shapes before the first optimization step.
- Detect NaN/Inf loss and stop with a clear error.

Evaluation:

- Fail clearly if render assets are missing.
- Save metadata next to every render.

## Testing

Tests should favor small deterministic fixtures.

Required phase 1 tests:

- `DuixUNet` forward shape test.
- Weight init/export smoke test if a small local fixture can be provided without committing weights.
- Preprocess shape/range test for face and target tensors.
- Manifest parser test.
- Dataset sample load test using generated tiny fixture data.
- Loss backward test.

Integration tests can be run manually or behind a marker because they depend on local video/model assets.

## Milestones

1. Repository hygiene:
   - `.gitignore`
   - README
   - package metadata
   - design docs

2. Model port:
   - Port `DuixUNet`
   - Port NCNN bin weight loading
   - Add checkpoint save/load
   - Add forward smoke tests

3. Shared preprocessing:
   - Audio loading/resampling
   - Wenet BNF extraction
   - BNF window selection
   - Face crop/mask/target generation

4. Dataset builder:
   - Raw video normalization
   - Frame extraction
   - BBox tracking
   - Manifest/splits
   - Previews and quality stats

5. Training:
   - Dataset loader
   - Losses
   - Train loop
   - Checkpoints
   - Tiny overfit verification

6. Evaluation:
   - Render validation clips
   - Export grids
   - Compare against init baseline

## Open Decisions For Implementation Plan

These are intentionally left for the implementation plan because they depend on local machine constraints and available assets, not product direction:

- Exact bbox detector/tracker choice.
- Exact train batch size and mixed precision policy.
- Whether to include perceptual loss in the first implementation pass.
- Exact train/validation split ratios.
- Whether to vendor or reference the Wenet feature extraction code.

The design direction is fixed: a clean repo, current Duix UNet unchanged in phase 1, supervised fine-tuning from synchronized avatar videos, and no architecture expansion until the baseline is measurable.
