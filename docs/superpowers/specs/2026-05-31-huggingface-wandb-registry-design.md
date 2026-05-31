# Hugging Face Registry And W&B Tracking Design

Date: 2026-05-31

## Context

The current pipeline builds datasets, trains Duix UNet checkpoints, and renders validation
artifacts on the local filesystem. This works for one machine, but dataset revisions, model
revisions, and experiment history are difficult to correlate after multiple builds and runs.

The integration must preserve the current local workflow while adding:

1. Hugging Face dataset repositories for versioned preprocessed training datasets.
2. Hugging Face model repositories for versioned trained checkpoints and run metadata.
3. Weights & Biases runs for training configuration, metrics, system telemetry, and debugging.
4. Explicit provenance links from each checkpoint and model upload back to the dataset revision
   and W&B run that produced it.

## Decisions

Use Hugging Face Hub snapshots with a local cache. The existing `DuixManifestDataset` remains the
runtime loader because it already understands the custom frame, bbox, and BNF layout. A Hub
dataset revision is downloaded to the local Hugging Face cache before training starts, then the
loader reads that immutable snapshot exactly as it reads a local dataset directory.

Use private Hugging Face repositories by default. Raw source videos are not uploaded. Dataset
uploads contain only the processed artifacts required for training and debugging:

```text
manifest.jsonl
splits.json
build_summary.json
clips/*/frames/*.jpg
clips/*/bnf.npy
clips/*/bboxes.json
clips/*/quality.json
clips/*/previews/*.jpg
```

The normalized intermediate `audio.wav` and `video_25fps.mp4` files are excluded because the
trainer does not consume them and they increase storage costs.

Use W&B as an optional experiment tracker, not as the source of model or dataset truth. Hugging
Face owns artifact versioning. W&B links runs to Hugging Face revisions and retains its native
system telemetry, console capture, configuration, metrics, and summary views. Local JSON and CSV
metrics remain available when W&B is disabled or offline.

## Architecture

Add `edge_lipsync/hub.py` as the only module that imports `huggingface_hub`. It provides:

- Dataset snapshot upload with an allowlist for processed training artifacts.
- Dataset snapshot download by repository ID and revision.
- Model artifact upload with an allowlist for checkpoints, metrics, metadata, and model card.
- Model checkpoint download by repository ID, filename, and revision.
- Resolved commit SHA metadata for every download or upload.

Add `edge_lipsync/tracking.py` as the only module that imports `wandb`. It exposes a small tracker
interface with enabled and disabled implementations. The training loop logs rows through this
interface without depending on W&B internals.

Keep `tools/*` wrappers thin:

```text
tools/hf_dataset.py push --dataset-root /absolute/path/to/dataset --repo-id username/avatar-dataset
tools/hf_dataset.py pull --repo-id username/avatar-dataset --revision dataset-v1
tools/hf_model.py push --run-dir /absolute/path/to/run --repo-id username/avatar-model
tools/hf_model.py pull --repo-id username/avatar-model --revision model-v1 --filename best.pt
```

The `push` commands create private repositories unless the caller passes `--public`. The `pull`
commands print the resolved commit SHA and local cached path.

## Training Inputs

`TrainConfig` continues to accept local inputs and gains optional Hub inputs.

Dataset selection:

```yaml
dataset_root: /absolute/path/to/local/dataset
hf_dataset_repo: ""
hf_dataset_revision: ""
hf_cache_dir: ""
```

Exactly one dataset source is configured:

- Local mode: set `dataset_root`.
- Hub mode: set `hf_dataset_repo` and pin `hf_dataset_revision`.

Hub mode rejects an empty revision. Training must not silently use the moving default branch
because reproducibility depends on a pinned dataset version.

Initial checkpoint selection:

```yaml
init_bin: /absolute/path/to/dh_model.bin
init_ckpt: ""
hf_init_model_repo: ""
hf_init_model_revision: ""
hf_init_model_filename: best.pt
```

Exactly one initialization source is configured:

- Local decrypted NCNN bin.
- Local PyTorch checkpoint.
- Hugging Face model checkpoint pinned to a revision.

Optional model publication after successful training:

```yaml
hf_model_repo: username/avatar-name
hf_model_private: true
```

If `hf_model_repo` is empty, checkpoints remain local. If it is set, training uploads `best.pt`,
`final.pt`, local metric curves, run metadata, and a generated model card after the final
checkpoint is written.

## Evaluation Inputs

`RenderEvalConfig` continues to accept a local dataset root and local checkpoint. It gains the
same optional pinned dataset and model references used by training. Evaluation downloads Hub
artifacts to the local cache before constructing the existing dataset loader and model.

This keeps render behavior unchanged while allowing a historical model revision to be evaluated
without manually locating local files.

## W&B Tracking

Training configuration gains:

```yaml
wandb_mode: disabled
wandb_project: edge-lipsync-model
wandb_entity: ""
wandb_run_name: ""
wandb_group: ""
wandb_tags: []
wandb_notes: ""
wandb_dir: ""
```

Supported modes:

- `disabled`: no network access and no W&B dependency usage at runtime.
- `offline`: write a local W&B run that can be synchronized later.
- `online`: stream the run to W&B.

The tracker records:

- The complete training config, excluding secrets.
- Dataset source type, repository ID, requested revision, resolved commit SHA, and manifest hash.
- Initialization source type and model revision when initialization comes from Hub.
- Per-step epoch, phase, learning rate, and training loss.
- Validation reconstruction, mouth-region, and temporal metrics when available.
- Best validation metric, best checkpoint path, final checkpoint path, Hub model revision, and
  Hub model URL in the run summary.

W&B native telemetry and console capture provide machine-level debugging context. The training
loop still raises non-finite loss and other pipeline errors normally so failed runs remain
visible as failed W&B runs.

## Checkpoint And Run Metadata

Training checkpoints retain the existing payload and gain a `provenance` object:

```json
{
  "dataset": {
    "source": "huggingface",
    "repo_id": "username/avatar-dataset",
    "requested_revision": "v1",
    "resolved_revision": "0123456789abcdef",
    "manifest_sha256": "a4b96fb0d1f53f4afeb78e8fb38b195a4de3fc991b12a3c5ef42c1c67c58b18d"
  },
  "init_model": {
    "source": "huggingface",
    "repo_id": "username/avatar-model",
    "requested_revision": "baseline-v1",
    "resolved_revision": "fedcba9876543210"
  },
  "wandb": {
    "mode": "online",
    "run_id": "9x2k1m7q",
    "run_url": "https://wandb.ai/username/edge-lipsync-model/runs/9x2k1m7q"
  }
}
```

Local sources use `source: local` and resolved absolute paths. `run_metadata.json` repeats the
same provenance next to local output artifacts so a run can be inspected without opening a
checkpoint.

## Error Handling

- Hub dataset uploads fail before network access if required artifacts are missing.
- Hub dataset and model pulls require non-empty revisions.
- Training and evaluation reject ambiguous local-plus-Hub source configuration.
- Upload failures propagate as errors. Local checkpoints remain intact, so publication can be
  retried with `tools/hf_model.py push`.
- `disabled` W&B mode does not import or initialize W&B.
- `online` and `offline` W&B modes report a clear installation error if W&B is unavailable.
- Secrets are read from normal SDK environment configuration such as `HF_TOKEN` and W&B login
  state. Tokens are never stored in YAML, checkpoints, metadata, or W&B config.

## Dependencies

Add:

- `huggingface-hub` for repository upload, download, cache management, and revision metadata.
- `wandb` for optional online and offline experiment tracking.

Do not add `datasets` in this phase. Hugging Face dataset repositories provide the required
storage and versioning, while the existing PyTorch loader remains the correct runtime API for the
custom BNF-backed sample format. Converting frames and BNF arrays into Arrow or Parquet would add
cost without improving the training path.

## Testing

Unit tests use fake Hub and W&B clients; they do not require credentials or network access.

Required coverage:

1. Dataset upload validates required files and uses the processed-artifact allowlist.
2. Dataset download requires a pinned revision and reports the resolved commit SHA.
3. Model upload includes only intended run artifacts.
4. Model download requires a pinned revision and resolves a checkpoint file.
5. Disabled W&B mode avoids importing W&B.
6. Enabled W&B mode initializes with expected config, logs metrics, writes summaries, and
   finishes the run.
7. Training input resolution rejects ambiguous sources and resolves Hub datasets and init models.
8. Evaluation input resolution rejects ambiguous sources and resolves Hub datasets and models.
9. Checkpoint payloads preserve provenance.
10. CLI help, lint, type checking, and the complete test suite remain green.

## Documentation

Update the README and example YAML files with:

- Hugging Face login and W&B login environment setup.
- Dataset push/pull commands.
- Local and Hub training examples.
- Model push/pull commands.
- Local and Hub evaluation examples.
- Explicit private-repository default and raw-video exclusion policy.
