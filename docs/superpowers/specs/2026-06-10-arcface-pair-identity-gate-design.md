# ArcFace Pair Identity Gate Design

## Goal

Add one required hard gate to the silent-talking dataset builder so a source
silent frame and target talking frame are paired only when ArcFace considers
them the same identity.

This gate supplements the existing pose, bbox geometry, post-crop landmark,
blur, continuity, and sync gates. It does not replace any existing gate.

## Scope

V1 changes only the silent-talking dataset builder.

- Use ONNX Runtime, which is already a project dependency.
- Use the InsightFace ArcFace `w600k_r50` recognition model.
- Support a configured local ONNX file.
- Automatically download and cache the default model when no local file is
  configured.
- Fail the build if the required model cannot be resolved, verified, loaded, or
  executed.
- Apply identity similarity to every candidate pair before final matching.
- Persist the selected pair's identity similarity in dataset rows and reports.

The pretrained weights are restricted to non-commercial research use. The
builder records this restriction in build metadata.

## Configuration

Add an `IdentityConfig` nested under `SilentTalkingBuildConfig`:

```yaml
identity:
  arcface_onnx: ""
  hf_repo: facefusion/models-3.0.0
  hf_filename: arcface_w600k_r50.onnx
  hf_revision: main
  cache_dir: ""
  expected_sha256: f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1
  min_cosine_similarity: 0.35
```

Behavior:

- When `arcface_onnx` is set, use that file and verify its SHA-256.
- Otherwise call `hf_hub_download()` with the configured repository, filename,
  revision, and optional cache directory.
- Hugging Face Hub handles local caching. Repeated builds reuse the cached file.
- The downloaded or local file must match `expected_sha256`.
- Download, checksum, ONNX session, or incompatible model I/O failures stop the
  entire build.
- There is no silent fallback or configuration switch that disables the gate.

## ArcFace Runtime

Add a focused `edge_lipsync.identity` module containing:

- model resolution and checksum verification;
- five-point face alignment;
- ArcFace ONNX preprocessing and inference;
- L2 normalization and cosine similarity.

The runtime creates one `onnxruntime.InferenceSession` and reuses it for all
frames in the build.

### Five-point alignment

MediaPipe landmarks provide:

- left eye center from the two eye-corner landmarks;
- right eye center from the two eye-corner landmarks;
- nose tip;
- left mouth corner;
- right mouth corner.

Estimate a similarity transform from these points to the standard ArcFace
112x112 template. Warp the BGR frame to 112x112, convert to RGB, and normalize
pixels to approximately `[-1, 1]` using:

```text
(pixel - 127.5) / 127.5
```

The ONNX input is contiguous float32 NCHW with shape `[1, 3, 112, 112]`.
The output must contain one embedding vector. Normalize it to unit L2 length
before caching or comparison.

An invalid five-point alignment rejects only that frame with
`identity_alignment_failed`. An ONNX execution or output-contract failure stops
the build because identity quality can no longer be guaranteed.

## Analysis Cache

Extend `FrameObservation` with an optional identity embedding.

Each valid frame is embedded exactly once during `analyze_frames()`. The
embedding is serialized into the existing analysis JSONL cache, so repeated
pair matching does not rerun ArcFace.

Changing identity configuration changes the builder config hash and invalidates
the existing analysis cache.

Silent observations without an embedding are excluded from the valid source
pool. Talking observations without an embedding are rejected before matching.

## Pair Matching

The candidate order is:

```text
target frame validity
-> pose and bbox geometry gate
-> ArcFace identity gate
-> post-crop landmark alignment gate
-> normalized pose/geometry score
-> deterministic best candidate selection
```

For each pose/geometry candidate:

```python
identity_similarity = dot(source_embedding, target_embedding)
```

Both embeddings are L2-normalized, so the dot product is cosine similarity.

- Keep candidates where `identity_similarity >= min_cosine_similarity`.
- Drop candidates below the threshold.
- Continue evaluating other silent candidates.
- If pose/geometry candidates exist but none pass identity, reject the target
  frame with `identity_mismatch`.
- If identity candidates exist but none pass post-crop alignment, retain the
  existing `post_crop_alignment_mismatch` reason.
- `valid_silent_candidate_count`, second-best score, and score margin are
  calculated after every hard gate, including identity.

Identity similarity is a hard gate only. It is not added to matching score in
V1, avoiding a second optimization objective after the same-person condition
has already been satisfied.

## Dataset And Reports

Add `identity_similarity: float32` to retained dataset rows.

Frame decisions include:

- selected `identity_similarity`, when retained;
- `identity_mismatch_candidate_count`;
- `reject_reason = identity_mismatch` when applicable.

Per-video quality reports include:

- identity mismatch frame count;
- distribution of retained identity similarities;
- distribution of identity-rejected candidate similarities where available;
- ArcFace model source, SHA-256, threshold, and license restriction.

Preview output adds a `near_identity_threshold` group so borderline retained
pairs can be inspected visually.

Build metadata records:

```text
identity.model_source
identity.hf_repo
identity.hf_filename
identity.hf_revision
identity.resolved_path
identity.sha256
identity.min_cosine_similarity
identity.license = insightface-non-commercial-research
```

The cached filesystem path is provenance only and is not stored inside Hugging
Face dataset rows.

## Error Handling

Hard build failures:

- ArcFace download fails and no cached model exists;
- configured local model is missing;
- SHA-256 differs from the configured value;
- ONNX model input/output contract is unsupported;
- ONNX Runtime inference fails.

Frame-level rejection:

- required MediaPipe landmarks are unavailable;
- five-point alignment cannot produce a valid transform;
- identity embedding is non-finite or has zero norm;
- every pose-compatible silent candidate is below the identity threshold.

No fallback silently disables identity checking.

## Tests

Unit tests:

- local model resolution verifies checksum;
- missing model invokes `hf_hub_download`;
- failed download propagates and fails the build;
- checksum mismatch fails before ONNX session creation;
- five-point alignment produces 112x112 input;
- ArcFace preprocessing uses float32 NCHW and expected normalization;
- embedding output is L2-normalized;
- cosine similarity accepts a same-face pair and rejects a dissimilar pair;
- matching tries the next candidate after an identity mismatch;
- matching reports `identity_mismatch` when all pose-compatible candidates fail;
- post-crop alignment still runs after identity;
- analysis cache round-trips embeddings;
- nested identity configuration is parsed correctly.

Dataset/report tests:

- retained rows include finite `identity_similarity`;
- candidate counts are computed after the identity and alignment gates;
- frame decisions and quality reports count identity mismatches;
- preview selection includes near-threshold retained pairs;
- build metadata contains model checksum and non-commercial license notice.

Integration testing may use a small synthetic ONNX model or monkeypatched
session to avoid downloading the 174 MB production model during normal tests.

## Acceptance Criteria

- A pair below cosine similarity `0.35` is never retained.
- A target can still pair with another silent frame when the first candidate
  fails identity.
- ArcFace runs once per analyzed frame, not once per pair.
- A missing model is downloaded once and reused from the Hugging Face cache.
- Model download, checksum, load, or execution failure stops the build.
- Existing pose, geometry, alignment, blur, continuity, sync, idle sampling,
  split, and tensor contracts remain unchanged.
