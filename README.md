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

Landmark-based dataset building uses MediaPipe, which is installed as a project dependency.
Current MediaPipe Tasks also needs a FaceLandmarker model asset. Download
`face_landmarker.task` from Google's model bucket and set `landmark_model_asset_path` in the
dataset config:

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Download Model Assets

Large reusable assets are stored on Hugging Face instead of GitHub:

```text
tiennguyenbnbk/edge-lipsync-model-assets
```

The repository is private. Authenticate with Hugging Face before downloading:

```bash
hf auth login
.venv/bin/python tools/hf_model_assets.py pull \
  --repo-id tiennguyenbnbk/edge-lipsync-model-assets \
  --local-dir models
```

To publish a refreshed local `models/` snapshot:

```bash
.venv/bin/python tools/hf_model_assets.py push \
  --models-root models \
  --repo-id tiennguyenbnbk/edge-lipsync-model-assets
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
Normalized video intermediates use lossless `FFV1` and extracted training frames use PNG so the
dataset build does not add avoidable image degradation before preprocessing.
The default bbox detector is `mediapipe_face_landmarker`, which derives the Duix lower-face ROI
from face landmarks. `haar` remains available only as a debug fallback and produces full-face
boxes, not production Duix ROI boxes.
Dataset processing commands use `tqdm.auto` progress bars for terminal and notebook contexts.
Pass `--no-progress` on the data build CLIs, or set `progress: false` in YAML configs, to disable
progress bars in CI logs.

## Build Silent/Talking Pose-Paired Dataset

Use this builder when one persona has a silent source video and one or more talking target videos:

```text
data/<persona>/silent/defaultvideo.mp4
data/<persona>/talking/*.mp4
```

The builder analyzes every normalized 25 FPS talking frame, rejects invalid candidates with
frame-level reasons, pairs retained frames to the best pose/geometry-matched silent frame, and
saves a portable Hugging Face `DatasetDict` under `<snapshot_root>/dataset`. Rows embed `source_roi`
and `target_roi` as PNG bytes plus the exact precomputed BNF window, so training does not need the
original MP4 files.

```bash
.venv/bin/python tools/build_silent_talking_dataset.py \
  --config configs/silent_talking_dataset.example.yaml
```

The builder processes talking clips concurrently, reuses one ArcFace session and one Wenet
session, and caches normalized media, extracted frames, frame analysis, and BNF windows under
`work_root`. Configure `runtime.clip_workers` for CPU-side clip concurrency and
`runtime.cuda_max_inflight` to bound calls into one NVIDIA GPU.

For CUDA builds, install the GPU ONNX Runtime wheel instead of keeping both CPU and GPU wheels:

```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu>=1.17"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The provider list must contain `CUDAExecutionProvider`. With `runtime.device: auto` or `cuda`, the
builder prefers CUDA for ArcFace and Wenet. If CUDA is unavailable, it emits a warning, records the
fallback in `build_complete.json`, and continues on CPU. MediaPipe FaceLandmarker remains on CPU.

Inspect `reports/quality/*_frame_decisions.parquet` for one decision per normalized talking frame.
Preview groups live under `reports/previews/<clip_id>/` and include best matches, near-threshold
matches, low-sync-confidence retained rows, retained idle rows when available, and frequent
rejection reasons. `sample_weight` is propagated through training metadata and applied by the
training loss, so retained idle rows can contribute less than speech rows.

Hub is transport only for these snapshots. Upload the complete saved directory, record the full
commit SHA, then download that immutable revision once on the training machine:

```bash
HF_XET_HIGH_PERFORMANCE=1 uv run python tools/hf_dataset.py push-snapshot \
  --snapshot-root /absolute/path/to/datasets/nora_pose_pairs \
  --repo-id username/nora-pose-pairs \
  --workers 8

HF_XET_HIGH_PERFORMANCE=1 uv run python tools/hf_dataset.py pull-snapshot \
  --repo-id username/nora-pose-pairs \
  --revision <full-commit-sha> \
  --local-dir /persistent/datasets/nora/<full-commit-sha> \
  --cache-dir /persistent/hf-cache \
  --workers 16
```

Snapshot upload/download defaults to the train-ready package only:
`dataset/**`, `build_complete.json`, and `build_metadata.json`. This avoids transferring thousands
of quality-report and preview files. Add `--include-reports` to either command when the complete
review artifact is required. Large-folder uploads are resumable; rerun the same command with the
same snapshot root after an interruption.

Training uses `datasets.load_from_disk(<local_snapshot>/dataset)`. The local
`.snapshot_complete.json` marker prevents repeat Hub access after the requested repo, full commit
SHA, dataset fingerprints, and download profile have been verified.

## Build Hugging Face Video Dataset

Use `tools/build_hf_video_dataset.py` for HF datasets that already contain synced MP4 clips. This
fits `Pinch-Research/lipsync-hdtf-training-data` because the teacher videos live under
`xdub_teacher_pairs/videos/` and already include muxed audio. Start with `--dry-run` plus
`--max-videos`. The adapter downloads only selected videos, one file at a time through Hugging Face
`datasets.load_dataset()`. Video decoding stays disabled while the adapter links the cached local
MP4 paths into its work directory.
The datasets downloader is limited to one worker by default; adjust `--download-max-workers` only
when the Hub endpoint can handle additional concurrency. Authenticate with `HF_TOKEN` before large
downloads so the Hub does not apply the lower anonymous resolver quota.
For one-person training, list speakers first and then build with `--speaker-id`; the filter uses
`src_speaker == alt_speaker == speaker_id` from `xdub_teacher_pairs_manifest.json` before any video
download starts.

```bash
.venv/bin/python tools/build_hf_video_dataset.py \
  --repo-id Pinch-Research/lipsync-hdtf-training-data \
  --dataset-root /absolute/path/to/data/hdtf_xdub_duix_dataset \
  --work-dir /absolute/path/to/work/hdtf_xdub \
  --wenet-onnx /absolute/path/to/models/wenet.onnx \
  --landmark-model-asset-path /absolute/path/to/models/face_landmarker.task \
  --video-prefix xdub_teacher_pairs/videos \
  --download-max-workers 1 \
  --max-videos 20 \
  --dry-run
```

List available speakers and clip counts:

```bash
.venv/bin/python tools/build_hf_video_dataset.py \
  --repo-id Pinch-Research/lipsync-hdtf-training-data \
  --dataset-root /absolute/path/to/data/hdtf_xdub_duix_dataset \
  --work-dir /absolute/path/to/work/hdtf_xdub \
  --wenet-onnx /absolute/path/to/models/wenet.onnx \
  --landmark-model-asset-path /absolute/path/to/models/face_landmarker.task \
  --video-prefix xdub_teacher_pairs/videos \
  --list-speakers
```

Build all available clips for one speaker by setting `--max-videos 0`:

```bash
.venv/bin/python tools/build_hf_video_dataset.py \
  --repo-id Pinch-Research/lipsync-hdtf-training-data \
  --dataset-root /absolute/path/to/data/hdtf_xdub_duix_dataset \
  --work-dir /absolute/path/to/work/hdtf_xdub \
  --wenet-onnx /absolute/path/to/models/wenet.onnx \
  --landmark-model-asset-path /absolute/path/to/models/face_landmarker.task \
  --video-prefix xdub_teacher_pairs/videos \
  --speaker-id AdamSchiff \
  --download-max-workers 1 \
  --max-videos 0
```

When the selected subset looks correct, remove `--dry-run`. Add `--push` to publish the processed
dataset as a Hugging Face `DatasetDict` with `push_to_hub()`.

```bash
.venv/bin/python tools/build_hf_video_dataset.py \
  --repo-id Pinch-Research/lipsync-hdtf-training-data \
  --dataset-root /absolute/path/to/data/hdtf_xdub_duix_dataset \
  --work-dir /absolute/path/to/work/hdtf_xdub \
  --wenet-onnx /absolute/path/to/models/wenet.onnx \
  --landmark-model-asset-path /absolute/path/to/models/face_landmarker.task \
  --download-max-workers 1 \
  --max-videos 20 \
  --push \
  --hf-output-repo-id username/hdtf-xdub-duix-dataset
```

## Build GRID Baseline Dataset

Use the GRID adapter to create a small baseline dataset from an extracted GRID corpus, then push
the processed Duix dataset to Hugging Face after inspecting the previews. Start with `--dry-run`
or a small `--max-videos` value; do not run the full corpus locally until the small build is
validated.

```bash
.venv/bin/python tools/build_grid_hf_dataset.py \
  --grid-root /absolute/path/to/grid \
  --dataset-root /absolute/path/to/data/grid_duix_dataset \
  --work-dir /absolute/path/to/work/grid_duix \
  --wenet-onnx /absolute/path/to/models/wenet.onnx \
  --landmark-model-asset-path /absolute/path/to/models/face_landmarker.task \
  --speaker s1 \
  --max-videos 20 \
  --dry-run
```

When the dry run and a small local build look correct, run without `--dry-run` and add `--push`:

```bash
.venv/bin/python tools/build_grid_hf_dataset.py \
  --grid-root /absolute/path/to/grid \
  --dataset-root /absolute/path/to/data/grid_duix_dataset \
  --work-dir /absolute/path/to/work/grid_duix \
  --wenet-onnx /absolute/path/to/models/wenet.onnx \
  --landmark-model-asset-path /absolute/path/to/models/face_landmarker.task \
  --speaker s1 \
  --max-videos 20 \
  --push \
  --hf-repo-id username/grid-duix-baseline
```

## Version Processed Datasets On Hugging Face

Upload a built dataset after inspecting its `build_summary.json` and previews:

```bash
.venv/bin/python tools/hf_dataset.py push \
  --dataset-root /absolute/path/to/data/duix_datasets/avatar_name \
  --repo-id username/avatar-name-dataset
```

Uploads create private dataset repositories by default. Pass `--public` only when the dataset is
intended to be public. Dataset uploads store train/val splits as native Hugging Face datasets with
frame images, bbox metadata, BNF audio windows, and flags.

Load a processed dataset through Hugging Face `datasets`:

```bash
.venv/bin/python tools/hf_dataset.py pull \
  --repo-id username/avatar-name-dataset
```

## Train

```bash
.venv/bin/python tools/train.py --config configs/train.example.yaml
```

Training writes atomic checkpoints, `best.pt`, `final.pt`, `metrics.json`, `metrics.csv`,
`run_metadata.json`, and a model card. Checkpoints include the dataset manifest hash, training
config, step, epoch, metrics, initialization source, dataset provenance, and W&B run provenance.
The train loop also prints concise progress rows to stdout every `log_interval` steps and whenever
validation runs, which is useful in Colab notebooks. Validation rows include `val_loss`, the same
configured reconstruction-plus-perceptual objective used for training, plus reconstruction, mouth,
and temporal metrics. `best.pt` is selected by `val_loss`.

Additional validation metrics cover three failure modes that reconstruction loss alone misses:

- `val_mae`, `val_psnr`, and `val_ssim` measure full-frame reconstruction quality.
- `val_mouth_mae`, `val_mouth_psnr`, and `val_mouth_ssim` measure a fixed lower-face mouth ROI.
- `val_lpips_face` and `val_lpips_mouth` measure learned perceptual distance with a frozen LPIPS
  backbone. Lower values indicate closer perceptual similarity.
- `val_mouth_temporal_error` compares predicted and target mouth motion for consecutive frames in
  the same clip. `val_temporal_pair_count` reports how many valid frame pairs were measured.
- `val_audio_sensitivity` measures how much the mouth prediction changes when the BNF window is
  shifted by five frames. `val_audio_shift_mouth_mae_delta` measures shifted-audio mouth MAE minus
  aligned-audio mouth MAE; positive values indicate that aligned audio produces the better result.

Lower is better for MAE and temporal error; higher is better for PSNR and SSIM. Near-zero audio
sensitivity suggests that the model may be ignoring audio, while a positive audio-shift MAE delta
indicates that aligned audio produces the better reconstruction. Audio-shift metrics require one
additional model forward pass during validation. They are diagnostics rather than the current
checkpoint-selection objective.

Set `lpips_enabled: true` to compute face and mouth LPIPS. `lpips_net` supports `alex`, `vgg`, and
`squeeze`; `alex` is the default evaluation backbone. The first AlexNet run downloads approximately
233 MB of pretrained weights into the PyTorch cache. LPIPS is disabled by the dataclass default so
existing or offline runs do not download weights unexpectedly, while the example train and eval
configs enable it explicitly.

LPIPS can also participate in the training objective:

```text
train_loss = combined_reconstruction_loss
           + lpips_face_loss_weight * face_lpips
           + lpips_mouth_loss_weight * mouth_lpips
```

The example train config uses `lpips_face_loss_weight: 0.01` and
`lpips_mouth_loss_weight: 0.05`, prioritizing perceptual quality in the mouth crop. Set both weights
to `0` to recover the previous reconstruction-only objective. Positive weights automatically load
the frozen LPIPS evaluator even when `lpips_enabled` is false. Gradients pass through LPIPS to the
Duix prediction, but LPIPS parameters remain frozen. `alex` is the practical low-cost default;
`vgg` is more expensive but is the LPIPS authors' recommended backbone for optimization.

Early stopping is disabled by default. Set `early_stopping_patience` to the number of validation
runs allowed without a `val_loss` improvement, and optionally set `early_stopping_min_delta` for the
minimum improvement threshold.

When `media_eval_on_best` is enabled, every new `best.pt` renders a validation grid MP4 from the
first `media_eval_clip_count` unique clips in the validation split. The selected clip IDs and frame
indices are recorded in run provenance, and local videos are written under `media_eval/`. W&B video
upload is disabled by default to keep Colab training from blocking on large MP4 uploads; set
`media_eval_log_to_wandb: true` only when you explicitly want best-checkpoint videos in W&B. Set
`media_eval_clip_ids` to pin exact validation clips instead of using the first clips.

The example config uses local manifest paths. To train from a revision-pinned pose-paired snapshot,
leave `dataset_root` empty and set:

```yaml
dataset_root: ""
hf_dataset_repo: username/nora-pose-pairs
hf_dataset_revision: <full-commit-sha>
hf_dataset_local_dir: /persistent/datasets/nora/<full-commit-sha>
hf_cache_dir: /persistent/huggingface-cache
```

To initialize from a Hugging Face model instead of `init_bin` or `init_ckpt`, set:

```yaml
init_bin: ""
init_ckpt: ""
hf_init_model_repo: username/avatar-name-model
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
Hub provenance, console output, and native system telemetry. Best-checkpoint MP4s are logged to W&B
only when `media_eval_log_to_wandb` is enabled.

Model publication is optional. When `hf_model_repo` is set, successful training uploads
`best.pt`, `final.pt`, metric curves, run metadata, and the generated model card. Uploads create
private model repositories by default. A failed publication leaves local artifacts intact and
can be retried:

```bash
.venv/bin/python tools/hf_model.py push \
  --run-dir /absolute/path/to/runs/avatar_name \
  --repo-id username/avatar-name-model
```

Pull a model checkpoint:

```bash
.venv/bin/python tools/hf_model.py pull \
  --repo-id username/avatar-name-model \
  --filename best.pt
```

## Render Validation Artifacts

```bash
.venv/bin/python tools/render_eval.py --config configs/eval.example.yaml
```

The render command writes prediction grids, `validation_grids.mp4`, numeric validation metrics,
and JSON metadata next to the video.

The eval config also supports Hub dataset and model inputs. Set:

```yaml
ckpt: ""
hf_dataset_repo: username/avatar-name-dataset
hf_model_repo: username/avatar-name-model
hf_model_filename: best.pt
```

## Smoke Inference On One Manifest Sample

Before training, run one processed manifest sample through the current model path to verify
`manifest -> frame/bbox/bnf -> preprocess -> model -> image artifacts`:

```bash
.venv/bin/python tools/infer_manifest_sample.py \
  --dataset-root /absolute/path/to/data/duix_datasets/avatar_name \
  --manifest manifest.jsonl \
  --sample-index 0 \
  --init-bin /absolute/path/to/dh_model.bin \
  --audio-wav /absolute/path/to/sample.wav \
  --wenet-onnx /absolute/path/to/wenet.onnx \
  --alpha-bin /absolute/path/to/weight_168u.bin \
  --output-mp4 output.mp4 \
  --out-dir /absolute/path/to/runs/smoke_infer
```

Use `--ckpt /path/to/best.pt` instead of `--init-bin` after exporting or training a PyTorch
checkpoint. Set exactly one model source: `--init-bin`, `--ckpt`, or `--hf-model-repo`.
If `--audio-wav` is omitted, the command uses the manifest `bnf_path` instead. The command writes
`prediction.png`, `grid.png`, `restored_frame.png`, and `metadata.json`. `--output-mp4` on this
single-sample command is only a still-frame smoke output with audio.

For a real end-to-end MP4, run the sequence CLI on a manifest generated by the dataset builder:

```bash
.venv/bin/python tools/infer_manifest_sequence.py \
  --dataset-root /absolute/path/to/data/duix_datasets/avatar_name \
  --manifest manifest.jsonl \
  --init-bin /absolute/path/to/dh_model.bin \
  --backend ncnn \
  --ncnn-param /absolute/path/to/dh_model.param \
  --audio-wav /absolute/path/to/sample.wav \
  --wenet-onnx /absolute/path/to/wenet.onnx \
  --alpha-bin /absolute/path/to/weight_168u.bin \
  --output-mp4 output.mp4 \
  --out-dir /absolute/path/to/runs/sequence_infer
```

This renders every selected manifest record, pastes each prediction back into the source frame,
and muxes the restored frame sequence with `--audio-wav`. Install the optional NCNN runtime with
`uv sync --extra ncnn` for native-runtime parity. Omit `--backend ncnn --ncnn-param ...` to keep
the default PyTorch runtime.

If you only have a processed Hugging Face `datasets` repo and no local manifest tree, run sequence
inference directly from the validation split:

```bash
.venv/bin/python tools/infer_hf_dataset_sequence.py \
  --hf-dataset-repo username/avatar-name-dataset \
  --split val \
  --clip-id auto \
  --ckpt /absolute/path/to/runs/avatar_name/best.pt \
  --alpha-bin /absolute/path/to/weight_168u.bin \
  --output-mp4 output.mp4 \
  --out-dir /absolute/path/to/runs/hf_sequence_infer \
  --max-frames 250 \
  --device cuda
```

This selects one clip from the HF split, sorts rows by `frame_idx`, pastes predictions back into
the stored full frames, and writes an MP4. Add `--audio-wav /path/to/audio.wav` only when you want
the output MP4 muxed with an external audio file; otherwise the MP4 is video-only.

## Run Emma Oracle Parity Harness

Run the reproducible comparison against the original Duix-Mobile Emma renderer:

```bash
.venv/bin/python tools/run_emma_parity.py
```

During iteration, reuse the previously rendered NCNN oracle:

```bash
.venv/bin/python tools/run_emma_parity.py --reuse-original
```

The harness writes `artifacts/parity_emma/report.json`, representative diff grids, and an ROI
calibration sweep that documents the remaining landmark geometry gap. The new pipeline manifest
is always generated from MediaPipe landmarks and Duix ROI expansion. `Emma/bbox.json` is read only
by the original oracle and post-build parity diagnostics. The report embeds a `completion_audit`
checklist whose terminal status distinguishes target parity, a documented blocker, and an
incomplete run. The harness exits successfully for `target_met` and `blocked`; inspect `passed`
and `failed_gates` to distinguish exact target parity from an externally blocked run.

The historical `20250714` Duix-Mobile branch also contains an SCRFD+PFPLD detector path. When the
following untracked diagnostic assets exist, the harness runs a source-aligned comparison and
records it under `diagnostics.historical_detector_public_compatible`:

```text
models/duix_detector/scrfd_500m_kps-opt2.param
models/duix_detector/scrfd_500m_kps-opt2.bin
models/duix_detector/pfpld_robust_sim_bs1_8003.onnx
```

The exact-name SCRFD files are available from
[`nihui/ncnn-android-scrfd`](https://github.com/nihui/ncnn-android-scrfd/tree/master/app/src/main/assets).
The PFPLD ONNX file is a public-compatible
[`next-social/faceswap-ai-fly` mirror](https://huggingface.co/next-social/faceswap-ai-fly/blob/main/pfpld_robust_sim_bs1_8003.onnx),
not the original Duix NCNN weight. These files are diagnostic controls only: do not commit them and
do not use their output as the dataset manifest source. The harness records an unavailable status
when they are absent. Use `--skip-historical-detector-diagnostic` to skip the optional comparison
explicitly.

## Asset Policy

Do not commit raw videos, generated datasets, Wenet ONNX files, Duix character folders,
checkpoints, rendered videos, or debug artifacts. Keep them outside git and reference them
through config files. Hugging Face stores versioned processed datasets and selected trained model
artifacts; W&B stores experiment history and debugging telemetry.
