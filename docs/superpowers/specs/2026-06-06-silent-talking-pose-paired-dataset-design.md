# Silent-Talking Pose-Paired Dataset Design

Date: 2026-06-06

## Goal

Build a train-ready lip-sync dataset for one persona from:

```text
data/<persona>/silent/defaultvideo.mp4
data/<persona>/talking/*.mp4
```

Each sample uses:

- A source frame from the persona's silent video.
- The audio feature window at a talking-frame timestamp.
- The talking frame at that timestamp as the supervised target.

The selected silent and talking faces must have sufficiently similar head pose, normalized face
position, and face size. Every talking frame at 25 FPS starts as a candidate. Invalid candidates
are rejected rather than paired with a poor source frame. A silent frame may be reused by any
number of talking frames.

The output must be directly consumable by the existing Duix training pipeline without changing the
model tensor contract:

```text
face:   [6, 160, 160]
audio:  [20, 256]
target: [3, 160, 160]
```

Dataset construction and training may run on different machines. Hugging Face Hub transports an
immutable dataset snapshot, while training downloads that snapshot once to persistent storage and
then uses `datasets.load_from_disk()`.

## Scope

This design includes:

- Video normalization and frame extraction.
- MediaPipe face landmarks and head-pose estimation.
- Per-frame geometry, bbox continuity, and blur analysis.
- Coarse audio-video synchronization analysis.
- Silent-frame matching for every eligible talking frame.
- Speech and idle sample selection.
- Local Hugging Face `DatasetDict` creation and validation.
- Optional publication as a revision-pinned private Hub dataset snapshot.
- One-time snapshot download and local training on another machine.
- Backward-compatible loading through the current training entry point.

This design does not:

- Change the Duix UNet architecture or tensor shapes.
- Apply `sample_weight` to the training loss.
- Add SyncNet or another learned synchronization model.
- Use streaming datasets during training.
- Attempt to repair frames that fail pose, geometry, blur, tracking, or speech-sync hard gates.

## Existing System Fit

The repository already provides:

- MediaPipe face landmark detection and Duix ROI extraction.
- Video normalization, frame extraction, BNF extraction, bbox cleanup, and quality summaries.
- `DuixHFDataset`, which adapts Hugging Face rows to the current tensors.
- Training from a Hugging Face repository through `load_dataset()`.
- PyTorch `DataLoader` construction and batch-shape validation.

The new builder should reuse these boundaries where their behavior matches. It must extend the
dataset row contract because the source and target now come from different frames and require
different bounding boxes.

## Considered Matching Approaches

### Head pose plus explicit geometry gates

Estimate yaw, pitch, and roll from MediaPipe face landmarks, then gate and score candidates using
pose, normalized face center, and separate ROI width and height ratios.

This is the selected approach. Its behavior is explicit, testable, configurable, and relatively
insensitive to mouth expression.

### Normalized full-landmark distance

Normalize the face mesh and select the smallest landmark distance. This is compact but mouth and
eye expression can dominate the metric and bias the source selection toward matching expression
instead of rigid pose.

### MediaPipe facial transformation matrices

Use transformation matrices returned by MediaPipe Tasks. This may provide useful rigid-pose
information but would bind the builder more tightly to a specific Tasks model and API path. The
current repository supports both FaceMesh-style and Tasks-style detection, so this is not the
initial contract.

## Architecture

The pipeline has five stages:

1. Normalize videos and extract media.
2. Analyze silent and talking frames.
3. Analyze talking audio-video synchronization.
4. Match valid talking frames to valid silent frames.
5. Build, validate, save, and optionally publish the dataset snapshot.

Silent analysis is shared across all talking videos for a persona. Each talking video is processed
independently so failures, resume state, split assignment, and quality reports remain clip-scoped.

## Stage 1: Video Normalization

Normalize the silent video and every talking video to 25 FPS using the repository's lossless video
intermediate and PNG frame conventions.

Talking audio is normalized for the existing BNF pipeline:

- Mono waveform.
- 16 kHz effective sample rate for feature extraction.
- BNF windows compatible with the existing `(20, 256)` training input.

The silent audio is not used for training. The silent video may contain an audio stream, but only
its visual frames participate in matching.

The builder records source path, file identity, normalized frame count, duration, FPS, and config
hash in per-video metadata.

## Stage 2: Frame Analysis

### Face landmarks and ROI

Use `MediaPipeFaceLandmarkerDetector` to obtain one face mesh per frame. Frames with no face or
missing required landmarks are rejected.

For each valid frame, record:

- Duix-compatible bbox.
- Normalized face center `(center_x / frame_width, center_y / frame_height)`.
- Normalized ROI width and height.
- Yaw, pitch, and roll in degrees.
- Face blur score.
- Mouth blur score.
- Mouth openness signal for synchronization analysis.

Source and target bboxes remain independent. The silent bbox creates the source tensor, while the
talking bbox creates the target tensor.

### Head-pose estimation

Estimate a rigid head rotation from stable MediaPipe landmarks with OpenCV `solvePnP`. The selected
landmark subset must emphasize the eyes, nose, cheeks, and chin and must not depend on lip opening.
The implementation defines one fixed generic 3D face model, camera approximation, coordinate
convention, and Euler-angle conversion. Unit tests lock the sign and ordering of yaw, pitch, and
roll.

Pose is a relative matching signal, not a calibrated biometric measurement.

### Bbox continuity

Check bbox movement along the original frame sequence before any cross-video matching. Silent and
talking tracks are cleaned separately.

The continuity check covers normalized center movement and width/height changes between adjacent
valid frames. Short landmark gaps may follow the existing interpolation policy, but frames involved
in a discontinuous jump are rejected. Matching cannot make an invalid track frame valid.

Initial sequence gates compare a frame with the previous accepted frame:

```text
normalized_center_distance <= 0.05
adjacent_width_ratio  in [1 / 1.15, 1.15]
adjacent_height_ratio in [1 / 1.15, 1.15]
```

The ratios use bbox dimensions normalized by their corresponding frame dimensions. All sequence
thresholds remain configurable.

### Blur analysis

Face blur is measured over the face ROI. Mouth blur is measured over a landmark-derived mouth
region inside the talking target.

The initial implementation uses deterministic image sharpness metrics such as Laplacian variance.
Thresholds are configuration values. The builder publishes metric distributions and previews so
the defaults can be calibrated against real persona footage. Because the ROIs are normalized to a
fixed size before measurement, the initial defaults are:

```text
minimum_face_laplacian_variance = 60.0
minimum_target_mouth_laplacian_variance = 40.0
```

These are operational defaults, not universal quality claims. Every build records the configured
values and metric distributions.

Hard behavior:

- Heavy face blur rejects either a silent source candidate or talking target candidate.
- Heavy target-mouth blur rejects the talking candidate.
- The initial version does not lower a training weight for blur; it rejects the pair.

## Stage 3: Audio-Video Synchronization

Synchronization is estimated independently for each talking video.

### Signals

Build two 25 FPS signals:

- Audio activity or energy aligned to video-frame timestamps.
- Mouth openness derived from MediaPipe lip landmarks.

The estimator operates only as a coarse quality gate. It does not claim frame-accurate phoneme
alignment. Within each window, evaluate Pearson correlation after centering both signals. Use
`correlation(audio[t], mouth[t + lag])`, so a positive lag means visual mouth motion follows audio.
Zero-variance signals receive correlation `0.0`. Lag ties prefer the smallest absolute lag, then
the smaller signed lag for deterministic output.

### Window policy

- Window length: 2 seconds.
- Stride: 1 second.
- Lag search: `[-3, +3]` video frames.
- Each frame receives metadata from the eligible window whose center is closest to that frame.
- Overlapping bad windows do not reject frames outside the nearest-window assignment.

For each window, record:

- Whether sufficient speech activity is present.
- Best lag in frames.
- Best correlation.
- Confidence label.

Speech activity uses the existing frame-aligned RMS blocks. A frame is voiced when its RMS exceeds
the configured `silence_rms_threshold`, initially `0.001`. A window is a speech window when at
least 25% of its frames are voiced. Correlation below `0.20` is labeled low confidence; it remains
metadata rather than a hard gate.

### Rejection policy

For a speech window:

- Reject assigned frames when `abs(best_lag_frames) > 2`.
- Low correlation alone does not reject the frame.
- Low correlation sets `sync_confidence = "low"` and a quality flag.

For an idle or silence window:

- Do not reject by lag because the signals do not contain enough information to estimate sync.
- Mark assigned frames as idle candidates.

The estimated lag is never used to shift training audio. `audio_idx` remains the talking frame's
original frame-aligned audio index; lag is quality metadata and a rejection signal only.

If most windows in a continuous speech region violate the lag gate, report that region as
suspected desynchronization. The corresponding frames have already been rejected by nearest-window
assignment. The clip remains processable unless no usable samples remain or strict mode is active.

## Stage 4: Pose And Geometry Matching

Every talking frame at 25 FPS is initially a candidate. It must first pass:

- Face detection.
- Required-landmark availability.
- Talking-track bbox continuity.
- Face blur.
- Target-mouth blur.
- Audio and BNF index availability.
- Speech sync lag gate when applicable.

For each remaining talking candidate, compare it against all valid silent frames. The silent video
is small enough for exhaustive NumPy matching; no nearest-neighbor dependency is required.

### Hard gates

Default gates:

```text
abs(delta_yaw)   <= 5 degrees
abs(delta_pitch) <= 5 degrees
abs(delta_roll)  <= 4 degrees
abs(delta_center_x) <= 0.05
abs(delta_center_y) <= 0.05
width_ratio  in [0.9, 1.1]
height_ratio in [0.9, 1.1]
```

All thresholds are configurable. Center deltas use frame-normalized coordinates rather than
absolute pixels.

The ratios are:

```text
source_width_normalized  = source_roi_width / source_frame_width
target_width_normalized  = target_roi_width / target_frame_width
source_height_normalized = source_roi_height / source_frame_height
target_height_normalized = target_roi_height / target_frame_height

width_ratio  = target_width_normalized / source_width_normalized
height_ratio = target_height_normalized / source_height_normalized
```

If no silent frame passes every gate, reject the talking candidate with
`pose_geometry_no_match`.

### Matching score

Only candidates that pass all hard gates are scored.

```text
pose =
  abs(delta_yaw) / yaw_threshold +
  abs(delta_pitch) / pitch_threshold +
  abs(delta_roll) / roll_threshold

position =
  abs(delta_center_x) / center_x_threshold +
  abs(delta_center_y) / center_y_threshold

scale =
  abs(log(width_ratio)) / width_log_gate_for_ratio_direction +
  abs(log(height_ratio)) / height_log_gate_for_ratio_direction

where:
  width_log_gate_for_ratio_direction =
    log(max_width_ratio) when width_ratio >= 1
    abs(log(min_width_ratio)) otherwise

  height_log_gate_for_ratio_direction =
    log(max_height_ratio) when height_ratio >= 1
    abs(log(min_height_ratio)) otherwise

score =
  pose_weight * pose +
  position_weight * position +
  scale_weight * scale
```

Default weights are `1.0`. Threshold normalization keeps the components interpretable. Ties are
resolved deterministically by silent frame index.

A silent frame may be selected by multiple talking frames.

## Speech And Idle Sampling

Every eligible speech frame produces one pair.

Idle and silence pairs are retained only after speech matching is complete for the talking video:

- Maximum retained idle count is 10% of the video's good speech-pair count.
- Selection is distributed across the timeline.
- Candidates close to speech boundaries receive priority.
- Ties and spacing are deterministic.
- Retained idle rows receive `sample_weight = 0.25`; speech rows use `1.0`.

`sample_weight` is stored in rows and metadata but is not applied by the initial training loss.
This preserves the future contract without changing current optimization behavior.

If a video has no good speech pairs, it contributes no idle rows by default because the cap is
defined relative to good speech pairs. The cap uses
`floor(good_speech_pair_count * 0.10)`, so videos with fewer than ten good speech pairs retain no
idle row.

## Train And Validation Splits

The primary split unit is the talking video. All rows from one talking video belong to one split.
For two or more talking videos, hash each `persona_id:split_salt:talking_clip_id`, sort by hash,
and assign the final `max(1, round(video_count * validation_fraction))` videos to validation. Cap
the validation count at `video_count - 1`. The initial validation fraction is `0.20`.

When at least two talking videos exist, both `train` and `val` must be non-empty before publication.

When only one talking video exists, a true video-level holdout is impossible. The fallback uses
contiguous time segments:

- Training receives the earlier 80% region.
- Validation receives the final 20% region.
- No random frame split is allowed.
- The quality summary records `split_mode = "single_video_contiguous_fallback"`.

Idle selection runs within the final assigned split boundaries so samples do not cross a temporal
split boundary.

## Dataset Artifact

### Primary representation

The builder creates a Hugging Face `DatasetDict` with `train` and `val` splits and saves it locally
with `DatasetDict.save_to_disk()`.

Each row is self-contained for training and does not require access to the original MP4 files.
To reduce transfer size and repeated preprocessing, the row stores already extracted source and
target ROIs rather than full frames:

- `source_roi`: silent ROI, 168 by 168, lossless image.
- `target_roi`: talking ROI, 168 by 168, lossless image.
- `audio`: float32 BNF window with shape `(20, 256)`.

ROI images are inserted into the `Image` features as encoded PNG bytes, not as references to local
paths. This ensures `save_to_disk()` produces a snapshot that remains self-contained after moving
to another machine.

The outer four-pixel crop remains part of the existing preprocessing contract: the loader converts
each 168 ROI to the 160 training patch.

### Canonical row schema

```text
schema_version: string
persona_id: string
pair_id: string
talking_clip_id: string
source_frame_idx: int32
target_frame_idx: int32
audio_idx: int32
source_roi: Image
target_roi: Image
audio: Array2D(shape=(20, 256), dtype=float32)
source_bbox_xyxy: Sequence(int32, length=4)
target_bbox_xyxy: Sequence(int32, length=4)
source_frame_width: int32
source_frame_height: int32
target_frame_width: int32
target_frame_height: int32
sample_weight: float32
is_idle: bool
sync_best_lag_frames: int32
sync_correlation: float32
sync_confidence: string
pose_delta_yaw: float32
pose_delta_pitch: float32
pose_delta_roll: float32
center_delta_x: float32
center_delta_y: float32
width_ratio: float32
height_ratio: float32
matching_score: float32
source_face_blur: float32
target_face_blur: float32
target_mouth_blur: float32
flags: Sequence(string)
```

`pair_id` is stable and contains the talking clip/frame and selected silent frame identity.

Full-frame paths are not required for training and must not be absolute paths in the portable
artifact. The build report retains portable source identifiers and original frame indices for
traceability. Optional local-only previews may include full-frame overlays outside the published
dataset snapshot.

## Training Adapter

Extend `DuixHFDataset` to recognize the new schema while keeping existing Hugging Face dataset rows
readable.

For a new row:

1. Decode `source_roi` and `target_roi`.
2. Build the source six-channel tensor from the source ROI:
   - Channels `0:3`: normalized source real patch.
   - Channels `3:6`: normalized masked source patch.
3. Build `target` from the target ROI.
4. Return the precomputed BNF `audio`.
5. Return quality metadata, including `sample_weight`, without applying it to loss.

The adapter returns:

```text
face:   torch.float32 [6, 160, 160]
audio:  torch.float32 [20, 256]
target: torch.float32 [3, 160, 160]
meta:   dictionary
```

`tools/train.py` continues to build the existing `DataLoader` and train loop. The model, loss
functions, optimizer, and batch-shape contract do not change.

## Cross-Machine Transport

### Build machine

1. Save the validated `DatasetDict` to a versioned local directory.
2. Write build metadata, quality reports, config, and `build_complete.json` inside the snapshot.
3. Upload the complete saved directory with `repo_type="dataset"` to a private Hugging Face
   dataset repository.
4. Record the returned full commit SHA.

The uploaded artifact is an immutable `save_to_disk()` snapshot, not only a logical dataset rebuilt
from a moving manifest.

### Training machine

The training config accepts:

```yaml
hf_dataset_repo: username/nora-lipsync
hf_dataset_revision: <full-commit-sha>
hf_dataset_local_dir: /persistent/datasets/nora/<full-commit-sha>
hf_cache_dir: /persistent/huggingface-cache
```

The repository and revision are required together for remote transport. The local directory is
stable persistent storage.

Training preparation:

1. If the local directory contains `.snapshot_complete.json` for the requested repo and revision,
   do not access the network.
2. Otherwise use `snapshot_download(repo_type="dataset", revision=<sha>)`.
3. Download to persistent local storage with Hub cache/resume support.
4. Verify `build_complete.json`, required splits, features, row counts, and dataset fingerprints.
5. Resolve and compare the full downloaded commit SHA with the requested revision.
6. Atomically write the local-only `.snapshot_complete.json` sidecar.
7. Call `datasets.load_from_disk()` on the local snapshot.
8. Wrap its splits with `DuixHFDataset`.

All training epochs read local Arrow/image data. There is no per-epoch Hub access and no streaming
mode.

Temporary download state must not be treated as a valid local snapshot. `build_complete.json`
proves the builder finished the artifact; `.snapshot_complete.json` proves the training machine
finished and verified a specific repository revision.

## Build Layout And Resume

A local working layout may be:

```text
<work_root>/<persona>/
  normalized/
    silent/
    talking/<clip_id>/
  analysis/
    silent/
    talking/<clip_id>/
  previews/
    <clip_id>/
  quality/
    silent.json
    <clip_id>.json
  dataset_snapshot/
    dataset_dict.json
    train/
    val/
    build_metadata.json
    build_complete.json
```

Silent analysis is cached once with its input identity and config hash. Each talking clip has its
own cache and quality report.

A cache is reusable only when:

- Input file identity matches.
- Relevant configuration hash matches.
- Expected outputs exist.
- Completion metadata is valid.

Stale or incomplete clip outputs are rebuilt atomically. One clip failure does not stop other clips
unless `--strict` is set.

## Quality Reports

Per-video reports include:

- Frame count and valid analysis count.
- Rejection counts by reason.
- Bbox jump count.
- Blur metric distributions.
- Speech, idle, low-confidence, and rejected-sync window counts.
- Best-lag and correlation distributions.
- Pose and geometry no-match count.
- Matching-score distribution.
- Speech and retained-idle pair counts.

The dataset summary includes:

- Configuration and config hash.
- Persona and portable source identities.
- Split policy and row counts.
- Aggregate rejection counts.
- Dataset feature schema.
- Snapshot file identity and optional Hub commit SHA.

Previews show source ROI, target ROI, pose deltas, geometry deltas, score, blur metrics, and sync
metadata for representative pairs and near-threshold pairs.

## Failure Handling

Candidate-level failures are recorded and skipped:

- Face detection failure.
- Required-landmark failure.
- Bbox discontinuity.
- Heavy face blur.
- Heavy target-mouth blur.
- Invalid BNF/audio index.
- Speech sync lag violation.
- No silent frame inside pose and geometry gates.

Clip-level failures produce a failed quality report. The build can continue unless strict mode is
active.

The final snapshot is not publishable when:

- No train or validation rows exist.
- Required features or files are missing.
- Any sampled row violates tensor shapes.
- Any row contains a non-finite numeric value.
- A pair violates a hard gate according to its recorded metrics.
- Snapshot verification or completion metadata fails.

## Testing

### Unit tests

- Head-pose coordinate convention and Euler-angle ordering.
- Pose gate boundaries.
- Normalized center deltas.
- Separate width and height ratio gates.
- Log-ratio scale distance.
- Threshold-normalized matching score.
- Deterministic tie breaking.
- Silent-frame reuse.
- Sequence-based bbox jump rejection.
- Sync window generation and lag search.
- Nearest-center window assignment.
- Speech-only lag hard rejection.
- Low-correlation flag behavior.
- Idle cap, timeline distribution, and speech-boundary priority.
- Deterministic video split and single-video contiguous fallback.

### Dataset tests

- Source and target use distinct ROI assets.
- Independent source and target bbox metadata survive serialization.
- Loader tensors have exact expected shape and dtype.
- Source tensor and target tensor are constructed from different images.
- BNF windows remain `(20, 256)`.
- `sample_weight` is returned only in metadata.
- Legacy Hugging Face rows remain readable.

### Snapshot tests

- `DatasetDict -> save_to_disk -> load_from_disk` round trip.
- Feature schema and split row counts remain stable.
- Missing or incomplete snapshots are rejected.
- A revision-pinned snapshot download resolves to the requested commit.
- A verified local snapshot skips network access.

### Integration test

Use:

```text
data/nora/silent/defaultvideo.mp4
data/nora/talking/mejnes4l-46ae1925-e823f674-321e-4745-8d0c-70a812.mp4
```

Verify:

- The builder produces non-empty train and validation splits using the single-video contiguous
  fallback.
- No retained row violates pose, center, scale, blur, tracking, BNF, or applicable speech-sync
  gates.
- Quality reports contain rejection counts and required distributions.
- Preview pairs visibly preserve pose and geometry.
- The saved snapshot reloads locally.
- `tools/train.py` reads the local snapshot and completes at least one training step with the
  existing model contract.

## Acceptance Criteria

The feature is complete when:

1. Every 25 FPS talking frame is analyzed as an initial candidate.
2. Every retained speech candidate maps to the best valid silent frame under the approved gates and
   normalized score.
3. Invalid candidates are rejected with an explicit reason.
4. Idle rows follow the 10% cap and deterministic selection policy.
5. The dataset is saved as a self-contained Hugging Face `DatasetDict`.
6. The dataset can be uploaded and identified by a full Hub commit SHA.
7. Another machine can resume-download that exact snapshot once, verify it, and train from
   `load_from_disk()`.
8. `tools/train.py` consumes the new dataset without changing `face`, `audio`, or `target` tensor
   shapes.
9. Tests cover matching, sync, quality gates, serialization, transport, and a one-step training
   integration.
