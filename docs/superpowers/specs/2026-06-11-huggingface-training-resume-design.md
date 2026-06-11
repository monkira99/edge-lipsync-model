# Hugging Face Training Resume Design

Date: 2026-06-11

## Context

The trainer currently writes model checkpoints with configuration and progress metadata, but it
cannot resume a run. Loading `init_ckpt` restores model weights only. The optimizer, mixed
precision scaler, step, epoch, data order, best validation state, early-stopping state, and random
number generators all start over.

The required behavior is:

1. Write a complete local resume checkpoint at a configurable step interval.
2. Upload that checkpoint to `resume/latest.pt` in the configured Hugging Face model repository.
3. Keep only that path in the current repository tree. Historical versions remain available
   through Hugging Face commit history.
4. Continue training when an upload fails and try again at the next interval.
5. Resume explicitly from `resume_hf_model_repo`, restoring the training trajectory rather than
   warm-starting model weights.

## Decisions

Use a synchronous upload after the triggering training step. The next step waits for the upload
attempt to finish, which avoids background upload races and guarantees that successful commits
refer to the most recently completed checkpoint.

Upload failures are non-fatal. The complete checkpoint remains on local disk, training prints a
structured warning, and the next configured interval writes a newer local checkpoint and attempts
another upload. Hugging Face Hub's normal HTTP retry behavior remains active; this feature does
not add a second retry framework.

Keep inference and resume checkpoints separate:

- `best.pt`, `final.pt`, and optional `step_*.pt` retain the current model-oriented payload.
- `resume_latest.pt` is the complete local resume checkpoint.
- `resume/latest.pt` is the single current resume checkpoint path on Hugging Face.

The complete resume payload includes the best model weights as well as the current model weights.
This is required because the best validation step may predate the latest training step. Without
those weights, a resumed run that never improves validation would produce a `best.pt` whose model
does not match its recorded metric.

## Configuration

Extend `TrainConfig` with:

```yaml
# Destination for periodic and final model publication.
hf_model_repo: username/avatar-name-model
hf_model_private: true

# Zero disables periodic resume uploads.
hf_resume_upload_interval: 1000

# Explicit source used only when resuming.
resume_hf_model_repo: ""
resume_hf_model_revision: ""
```

`resume_hf_model_revision` is optional. An empty value resolves the repository's current default
branch because the intended operation is to retrieve the latest recoverable state. Supplying a
commit SHA or tag allows recovery from an older Hub commit.

When `resume_hf_model_repo` is non-empty:

- `init_bin`, `init_ckpt`, and `hf_init_model_repo` must all be empty.
- The trainer downloads `resume/latest.pt`.
- `max_steps` remains the total target step, not the number of additional steps.
- The current training configuration and dataset must pass compatibility validation before any
  optimizer step runs.

When `resume_hf_model_repo` is empty, exactly one existing initialization source remains required.

`hf_resume_upload_interval` must be non-negative. A positive value requires `hf_model_repo`.

## Resume Checkpoint

Add a separate versioned format such as:

```text
edge_lipsync_duix_unet_resume_v1
```

The payload contains:

```text
format
model_state_dict
optimizer_state_dict
scaler_state_dict
training_config
dataset_root
manifest_path
manifest_sha256
step
epoch
next_batch_index
epoch_sample_indices
data_order_generator_state
best_val_loss
early_stopping_best_val_loss
validations_without_improvement
best_metrics
best_model_state_dict
metrics_history
random_state
init_weight_source
provenance
```

`random_state` contains Python, NumPy, Torch CPU, and all available CUDA RNG states. CUDA state is
optional when the checkpoint was produced without CUDA.

`scaler_state_dict` is present even when mixed precision is disabled; an empty scaler state is
valid.

`metrics_history` allows a resumed run in a new local directory to continue writing coherent
`metrics.json` and `metrics.csv` files. Media render files and W&B run identity are not embedded.
A resumed invocation starts a new W&B run whose provenance identifies the source resume
repository and resolved commit.

## Exact Data Continuation

The current `DataLoader(shuffle=True)` does not expose enough state for exact mid-epoch recovery,
especially when worker prefetch advances the sampler ahead of the last completed optimizer step.

Replace implicit shuffle state with an explicit epoch permutation:

1. A dedicated Torch generator creates the sample index permutation at epoch start.
2. The training loop owns the permutation and completed batch index.
3. The resume checkpoint records the permutation, the next batch index, and the generator state
   after the permutation was generated.
4. On resume, the loader starts from the recorded batch boundary in the saved permutation.
5. At the next epoch, the restored generator creates the same next permutation an uninterrupted
   run would have created.

The datasets and collate function are deterministic today, so restoring the explicit sample order
and process RNG states is sufficient for the existing pipeline. Batch size is therefore a
trajectory-critical configuration field and cannot change during transparent resume.

## Compatibility Validation

Resume fails before training if any trajectory-critical input differs:

- Resume checkpoint format is unsupported.
- Dataset manifest SHA-256 differs.
- Model state cannot load strictly.
- Batch size, learning rate, weight decay, warmup steps, stabilization steps, stabilization scale,
  precision mode, loss configuration, validation interval, early-stopping configuration, or
  `max_steps` differs.
- The checkpoint step is greater than or equal to `max_steps`.

Operational fields may differ:

- `run_dir`
- `num_workers`
- logging interval
- Hugging Face cache and publication settings
- W&B naming, directory, and mode
- media logging destination settings that do not affect the optimization loss

Requiring the same `max_steps` is intentional. Stabilization phase selection depends on
`max_steps`; changing it would not reproduce the trajectory of an uninterrupted run. Extending a
completed schedule is a separate fine-tuning operation and should use `init_ckpt`.

## Training Flow

Fresh training:

1. Resolve the dataset and initialization source.
2. Initialize model, optimizer, scaler, data-order generator, metrics, best state, and early-stop
   state.
3. Train normally.

Resume training:

1. Resolve the dataset.
2. Download `resume/latest.pt` from `resume_hf_model_repo` and optional revision.
3. Validate format, manifest hash, and trajectory-critical configuration.
4. Construct model, optimizer, and scaler, then restore their state dictionaries.
5. Restore progress, explicit sample order, metrics, best state, early-stopping state, and RNG
   state.
6. Continue from the recorded next batch and step.

At each completed step divisible by `hf_resume_upload_interval`:

1. Build the complete resume payload after validation and best-state updates for that step.
2. Atomically replace local `run_dir/resume_latest.pt`.
3. Upload the file to `resume/latest.pt` with a commit message containing the step.
4. Record the returned commit SHA in logs and tracker summary.
5. If upload fails, print the exception and continue training.

At normal completion or early stopping, write and attempt one final resume upload even when the
last step is not an interval boundary. Existing final artifact publication remains unchanged.

## Hugging Face Integration

Keep all SDK usage in `edge_lipsync/hub.py`.

Add a small upload function based on `HfApi.upload_file`:

```text
local path: run_dir/resume_latest.pt
path_in_repo: resume/latest.pt
commit message: Update resume checkpoint at step <step>
```

Each successful upload is one atomic Hub commit replacing the current path. The function returns
the resolved commit SHA and repository URL using the existing `HubArtifact` type.

Replacing the path does not guarantee constant total repository storage. Previous large-file
objects can remain reachable through commit history. Automatically squashing repository history
is intentionally excluded because the same repository also contains published model artifacts,
and destructive history rewriting would remove their version history. Repository history cleanup,
if ever required, must be a separate explicit maintenance operation.

Download uses the existing checkpoint download path with:

```text
filename: resume/latest.pt
revision: resume_hf_model_revision or default branch
```

The resolved commit SHA is stored in checkpoint/run provenance and logged at startup.

## Error Handling

- Local resume checkpoint write failures remain fatal because there is no valid recovery artifact
  to upload.
- Hub upload failures are caught only around periodic/final resume publication, logged with step
  and repository, and do not stop training.
- Hub download, authentication, missing-file, malformed-checkpoint, and compatibility failures are
  fatal before training begins.
- A failed final resume upload does not change successful local training completion or the
  existing final artifact publication behavior.
- Tokens continue to come from Hugging Face SDK configuration and are never written to config,
  checkpoint, metadata, or logs.

## Observability

Use concise structured console rows:

```text
[hf_resume] step=1000 status=start repo=username/avatar-name-model
[hf_resume] step=1000 status=uploaded ref=<commit-sha>
[hf_resume] step=2000 status=failed repo=username/avatar-name-model error=<message>
[resume] step=1000 epoch=4 next_batch=17 ref=<commit-sha>
```

Tracker summary records the latest successful resume upload step, commit SHA, and URL. Failed
attempts are logged as metrics/events where the active tracker supports them, but do not alter the
latest successful reference.

## Testing

Checkpoint unit tests cover:

- Complete optimizer, scaler, progress, best-state, metrics, data-order, and RNG fields.
- Format validation.
- Round-trip restoration of state.

Hub unit tests cover:

- Uploading exactly one file to `resume/latest.pt`.
- Step-specific commit messages.
- Downloading the latest file and honoring an optional revision.

Training tests cover:

- Resume continues at the next step rather than step 1.
- Optimizer and scaler state are restored.
- Mid-epoch sample order matches an uninterrupted run.
- Best validation state and best model weights survive resume.
- Early-stopping counters survive resume.
- Metrics history continues without duplicate steps.
- Incompatible dataset/configuration fails before an optimizer step.
- Periodic upload occurs at the configured interval.
- Upload failure is logged, training continues, and a later interval retries.
- Final/early-stop state receives a final upload attempt.
- Resume mode rejects all initialization sources.

An integration test compares a short uninterrupted CPU run with a split run that resumes from its
checkpoint. Final model parameters, optimizer state, processed sample order, step, epoch, and
metrics must match.

## Non-Goals

- Background or asynchronous checkpoint uploads.
- Keeping multiple named checkpoint files in the current Hub tree.
- Automatically rewriting or squashing Hugging Face repository history.
- Automatically resuming merely because `hf_model_repo` contains a checkpoint.
- Resuming the same W&B run identity.
- Changing the optimization schedule while claiming transparent resume.
