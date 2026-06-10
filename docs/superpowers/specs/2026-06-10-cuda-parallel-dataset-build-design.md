# CUDA Parallel Dataset Build Design

## Goal

Speed up the silent-talking dataset builder for one NVIDIA GPU and many short
talking videos, while keeping model loading bounded, CPU fallback visible, and
dataset output deterministic.

The implementation should remain simple:

- parallelize independent talking clips with a thread pool;
- create ArcFace and Wenet ONNX sessions once per build;
- prefer CUDA for ONNX inference and fall back to CPU with a warning;
- keep MediaPipe FaceLandmarker on CPU;
- reuse expensive normalized video, frame, analysis, and BNF artifacts when
  their inputs and relevant configuration have not changed.

## Scope

This design changes the silent-talking dataset builder and shared ONNX runtime
helpers used by it.

It does not:

- add TensorRT;
- run one ONNX session per clip or worker;
- use a process pool;
- enable the MediaPipe GPU delegate;
- change pairing gates, split behavior, row schema, or training tensors;
- require notebook changes.

## Runtime Configuration

Add a nested runtime configuration:

```yaml
runtime:
  device: auto
  clip_workers: 4
  cuda_max_inflight: 2
  warn_on_cpu_fallback: true
```

Behavior:

- `device: auto` selects `CUDAExecutionProvider` when available, otherwise
  `CPUExecutionProvider`.
- `device: cuda` requests CUDA but still falls back to CPU when CUDA is
  unavailable, as required for portable builds.
- `device: cpu` always uses CPU.
- When `auto` or `cuda` cannot select CUDA, emit a Python warning and a normal
  log line containing the requested device, selected provider, and available
  providers.
- `clip_workers` controls talking clips processed concurrently. Values below
  one are invalid.
- `cuda_max_inflight` bounds concurrent calls into shared CUDA sessions.
  Values below one are invalid.

The default remains portable. Existing configuration files without `runtime`
continue to work.

## ONNX Runtime Selection

Add a small shared provider resolver used by ArcFace and Wenet.

It returns:

- requested device;
- selected provider list;
- available providers;
- whether CPU fallback occurred;
- fallback reason.

Provider order:

```text
CUDAExecutionProvider, CPUExecutionProvider
```

when CUDA is selected, and:

```text
CPUExecutionProvider
```

otherwise.

The project continues to depend on `onnxruntime` by default. CUDA build
environments install the matching `onnxruntime-gpu` package instead of keeping
both CPU and GPU wheels installed.

Build metadata records provider selection separately for ArcFace and Wenet.

## Model Lifecycle

The build creates these objects once:

1. One ArcFace `InferenceSession`.
2. One Wenet `InferenceSession`.
3. One shared semaphore for bounded CUDA inference.

Both model wrappers are reused for the silent video and all talking clips.
No clip worker creates or reloads an ONNX model.

ArcFace and Wenet calls acquire the CUDA semaphore only when CUDA is selected.
CPU execution does not use that semaphore.

MediaPipe FaceLandmarker remains CPU-only. Each thread lazily creates one
detector and reuses it for every clip handled by that thread. All thread-local
detectors are closed after the pool is shut down.

The silent video is analyzed before starting talking-clip workers. It uses the
same shared ArcFace runtime.

## Parallel Clip Pipeline

Talking videos are sorted by path before submission.

Each worker performs the full independent clip pipeline:

```text
normalize video and audio
-> extract frames
-> extract BNF with shared Wenet session
-> analyze frames with thread-local MediaPipe and shared ArcFace
-> pair against immutable silent observations
-> encode retained ROI images
-> produce rows, decisions, report data, and preview data
```

Workers do not write final snapshot reports directly. Each returns a structured
clip result. The main thread sorts results by `clip_id`, then writes reports,
previews, and dataset rows in deterministic order.

This avoids concurrent mutation of global row lists and keeps output stable
regardless of worker completion order.

Per-clip error handling keeps current behavior:

- fatal identity or ONNX runtime errors stop the build;
- other errors follow `strict`;
- failed clips receive the existing failed-clip report.

## Cache Reuse

Cache expensive pipeline stages under the existing work root.

### Normalized media and frames

Reuse normalized video, extracted audio, and frame directories when a stage
metadata file matches:

- source video SHA-256;
- target FPS and sample rate;
- stage version;
- expected frame count and required output files.

Incomplete or mismatched outputs are removed and rebuilt atomically.

### Frame analysis

Keep the existing JSONL analysis cache. Its cache key must include:

- input identity;
- frame count;
- landmark model identity;
- ArcFace model SHA-256;
- analysis thresholds or version that affect observations.

Snapshot output paths and unrelated report settings must not invalidate frame
analysis.

### BNF

Add a BNF cache containing the NumPy windows and metadata:

- normalized audio SHA-256;
- Wenet model SHA-256;
- sample rate;
- BNF algorithm version.

Cache hits load windows without rerunning Wenet. Corrupt, incomplete, or
non-finite arrays are rebuilt.

Cache writes use a temporary file followed by atomic replacement.

## Thread Safety

Shared ONNX sessions are read-only after creation. Calls are protected only by
the configured CUDA semaphore, avoiding a broad lock around CPU preprocessing.

MediaPipe detectors are never shared across threads.

Each worker owns its clip filesystem directory. The silent directory and silent
observations are read-only once worker execution begins.

## Observability

At build start, log one runtime summary:

```text
runtime requested=cuda arcface=CUDAExecutionProvider
wenet=CUDAExecutionProvider clip_workers=4 cuda_max_inflight=2
```

When CUDA is unavailable:

```text
warning: CUDAExecutionProvider requested but unavailable;
falling back to CPUExecutionProvider
```

Build metadata includes:

- requested device;
- available providers;
- selected ArcFace and Wenet providers;
- fallback status and reason;
- clip worker count;
- CUDA inflight limit;
- cache hit/miss counts by stage;
- elapsed time by stage and total build.

No per-frame log output is added.

## Testing

Unit tests cover:

- provider resolver selects CUDA before CPU;
- unavailable CUDA falls back and emits a warning;
- explicit CPU never selects CUDA;
- ArcFace and Wenet receive the resolved provider list;
- Wenet session is created once and reused across clips;
- ArcFace session is created once per build;
- thread-local MediaPipe detector is reused within a worker and not shared
  across workers;
- CUDA semaphore limits concurrent model calls;
- clip results are sorted deterministically;
- strict and non-strict clip failures preserve current behavior;
- normalized media, frame analysis, and BNF cache hits skip computation;
- changed input or model checksum invalidates the appropriate cache;
- partial cache output is rebuilt.

Integration tests use fake sessions and short fixtures. A CUDA-marked test may
run only when `CUDAExecutionProvider` is available and must compare CPU/CUDA
output shapes and finite values, not require bit-identical floating-point
outputs.

## Acceptance Criteria

- On a CUDA machine, ArcFace and Wenet select `CUDAExecutionProvider`.
- If CUDA is unavailable, the build completes on CPU and emits a visible
  warning.
- ArcFace and Wenet models are each loaded once per build.
- Talking clips run concurrently up to `clip_workers`.
- A worker reuses its MediaPipe detector across clips.
- CUDA calls never exceed `cuda_max_inflight`.
- Repeating an unchanged build reuses normalized media, frame analysis, and BNF
  caches.
- Parallel and single-worker builds produce the same pair IDs, split counts,
  rejection counts, and deterministic row order.
- Existing notebooks and training tensor contracts continue to work unchanged.
