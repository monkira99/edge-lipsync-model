# Silent-Talking Pose-Paired Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Hugging Face `DatasetDict` from one persona's silent video and talking videos, pair frames by pose and geometry quality, transport the snapshot by immutable Hub revision, and train through the existing Duix tensor contract.

**Architecture:** Put deterministic pose, blur, sync, matching, split, and idle-selection algorithms in `edge_lipsync/pose_pairing.py`. Put video orchestration, caching, reports, ROI encoding, and `DatasetDict.save_to_disk()` in `edge_lipsync/silent_talking_dataset.py`. Extend the existing preprocessing, Hugging Face loader, Hub transport, and training source resolution only at their current compatibility boundaries.

**Tech Stack:** Python 3.11+, OpenCV, MediaPipe, NumPy, Hugging Face `datasets`/Hub, ONNX Runtime, PyTorch, pytest, Ruff, Pyright.

---

## Scope Boundary

This plan implements
`docs/superpowers/specs/2026-06-06-silent-talking-pose-paired-dataset-design.md`.

It does not:

- Change the Duix UNet architecture.
- Apply `sample_weight` to loss.
- Add affine target warping.
- Add SyncNet or another learned sync model.
- Stream rows from Hub during training.
- Replace the legacy manifest and processed-dataset paths.

## File Structure

```text
edge_lipsync/
  preprocess.py                 # construct tensors from an already-cropped 168x168 ROI
  dataset.py                    # recognize silent-talking HF rows in DuixHFDataset
  pose_pairing.py               # pure analysis, sync, matching, split, and idle algorithms
  silent_talking_dataset.py     # build orchestration, caches, reports, snapshot serialization
  hub.py                        # push/pull immutable dataset snapshot packages
  training.py                   # resolve pinned snapshot and load <snapshot>/dataset
tools/
  build_silent_talking_dataset.py
  hf_dataset.py                 # add snapshot push/pull commands
configs/
  silent_talking_dataset.example.yaml
  train.example.yaml
tests/
  test_preprocess.py
  test_dataset.py
  test_pose_pairing.py
  test_silent_talking_dataset.py
  test_hub.py
  test_training.py
  test_silent_talking_integration.py
README.md
pyproject.toml
```

`pose_pairing.py` must not read files or create MediaPipe/ONNX sessions. It accepts NumPy arrays and
dataclasses so its behavior is fast and deterministic under unit tests.

`silent_talking_dataset.py` owns all I/O and calls existing helpers from `build_dataset.py`,
`audio_features.py`, `landmarks.py`, and `progress.py`.

---

### Task 1: Train From Pre-Cropped Source And Target ROIs

**Files:**
- Modify: `edge_lipsync/preprocess.py`
- Modify: `edge_lipsync/dataset.py`
- Modify: `tests/test_preprocess.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write failing ROI preprocessing tests**

Add to `tests/test_preprocess.py`:

```python
def test_make_face_training_sample_from_roi_uses_distinct_source_and_target() -> None:
    from edge_lipsync.preprocess import make_face_training_sample_from_rois

    source = np.full((168, 168, 3), (10, 20, 30), dtype=np.uint8)
    target = np.full((168, 168, 3), (200, 210, 220), dtype=np.uint8)

    sample = make_face_training_sample_from_rois(source, target)

    assert sample.face.shape == (6, 160, 160)
    assert sample.target.shape == (3, 160, 160)
    assert np.allclose(sample.face[0], (30.0 - 127.5) / 127.5)
    assert np.allclose(sample.target[0], (220.0 - 127.5) / 127.5)
    assert not np.array_equal(sample.face[:3], sample.target)


def test_make_face_training_sample_from_rois_requires_168_square() -> None:
    from edge_lipsync.preprocess import make_face_training_sample_from_rois

    with pytest.raises(ValueError, match="168"):
        make_face_training_sample_from_rois(
            np.zeros((160, 160, 3), dtype=np.uint8),
            np.zeros((168, 168, 3), dtype=np.uint8),
        )
```

- [ ] **Step 2: Run the preprocessing tests to verify they fail**

Run:

```bash
rtk .venv/bin/pytest tests/test_preprocess.py -q
```

Expected: FAIL because `make_face_training_sample_from_rois` does not exist.

- [ ] **Step 3: Extract the shared ROI-to-tensor implementation**

In `edge_lipsync/preprocess.py`, add:

```python
def _validate_roi_168(value: np.ndarray, field: str) -> np.ndarray:
    if value.shape != (ROI_SOURCE_SIZE, ROI_SOURCE_SIZE, 3):
        raise ValueError(
            f"{field} must be BGR [{ROI_SOURCE_SIZE},{ROI_SOURCE_SIZE},3], got {value.shape}"
        )
    if value.dtype != np.uint8:
        raise ValueError(f"{field} must use uint8 pixels, got {value.dtype}")
    return value


def make_face_training_sample_from_rois(
    source_roi_168_bgr: np.ndarray,
    target_roi_168_bgr: np.ndarray,
    *,
    source_bbox_xyxy: BBox = (0, 0, ROI_SOURCE_SIZE, ROI_SOURCE_SIZE),
) -> FaceTrainingSample:
    source_roi = _validate_roi_168(source_roi_168_bgr, "source_roi_168_bgr")
    target_roi = _validate_roi_168(target_roi_168_bgr, "target_roi_168_bgr")
    source_patch = source_roi[
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
    ].copy()
    target_patch = target_roi[
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
        ROI_EDGE : ROI_EDGE + FACE_SIZE,
    ].copy()
    masked_patch = source_patch.copy()
    cv2.rectangle(
        masked_patch,
        (MASK_X, MASK_Y),
        (MASK_X + MASK_W - 1, MASK_Y + MASK_H - 1),
        (0, 0, 0),
        -1,
    )
    source_norm = _normalize_rgb(cv2.cvtColor(source_patch, cv2.COLOR_BGR2RGB))
    masked_norm = _normalize_rgb(cv2.cvtColor(masked_patch, cv2.COLOR_BGR2RGB))
    target_norm = _normalize_rgb(cv2.cvtColor(target_patch, cv2.COLOR_BGR2RGB))
    face = np.concatenate([source_norm, masked_norm], axis=2).transpose(2, 0, 1)
    return FaceTrainingSample(
        face=np.ascontiguousarray(face.astype(np.float32)),
        target=np.ascontiguousarray(target_norm.transpose(2, 0, 1).astype(np.float32)),
        roi_168_bgr=source_roi.copy(),
        real_patch_bgr=source_patch,
        masked_patch_bgr=masked_patch,
        bbox_xyxy=source_bbox_xyxy,
    )
```

Refactor `make_face_training_sample()` to crop and resize once, then call
`make_face_training_sample_from_rois(roi_168_bgr, roi_168_bgr, source_bbox_xyxy=bbox)`.

- [ ] **Step 4: Run preprocessing tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_preprocess.py -q
```

Expected: PASS.

- [ ] **Step 5: Write a failing silent-talking HF loader test**

Add to `tests/test_dataset.py`:

```python
def test_duix_hf_dataset_loads_silent_talking_roi_row() -> None:
    from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
    from edge_lipsync.dataset import DuixHFDataset

    source = np.full((168, 168, 3), (10, 20, 30), dtype=np.uint8)
    target = np.full((168, 168, 3), (200, 210, 220), dtype=np.uint8)
    _, source_png = cv2.imencode(".png", source)
    _, target_png = cv2.imencode(".png", target)
    features = Features(
        {
            "schema_version": Value("string"),
            "persona_id": Value("string"),
            "pair_id": Value("string"),
            "talking_clip_id": Value("string"),
            "source_frame_idx": Value("int32"),
            "target_frame_idx": Value("int32"),
            "audio_idx": Value("int32"),
            "source_roi": Image(),
            "target_roi": Image(),
            "audio": Sequence(Sequence(Value("float32"), length=256), length=20),
            "source_bbox_xyxy": Sequence(Value("int32"), length=4),
            "target_bbox_xyxy": Sequence(Value("int32"), length=4),
            "sample_weight": Value("float32"),
            "flags": Sequence(Value("string")),
        }
    )
    rows = [
        {
            "schema_version": "edge_lipsync_silent_talking_pair_v1",
            "persona_id": "nora",
            "pair_id": "talk__000001__silent__000002",
            "talking_clip_id": "talk",
            "source_frame_idx": 2,
            "target_frame_idx": 1,
            "audio_idx": 0,
            "source_roi": {"bytes": source_png.tobytes(), "path": None},
            "target_roi": {"bytes": target_png.tobytes(), "path": None},
            "audio": np.zeros((20, 256), dtype=np.float32),
            "source_bbox_xyxy": [10, 20, 110, 120],
            "target_bbox_xyxy": [12, 22, 112, 122],
            "sample_weight": 1.0,
            "flags": [],
        }
    ]
    dataset = DatasetDict({"train": Dataset.from_list(rows, features=features)})

    sample = DuixHFDataset(dataset, split="train")[0]

    assert tuple(sample["face"].shape) == (6, 160, 160)
    assert tuple(sample["audio"].shape) == (20, 256)
    assert tuple(sample["target"].shape) == (3, 160, 160)
    assert sample["meta"]["pair_id"] == rows[0]["pair_id"]
    assert sample["meta"]["sample_weight"] == pytest.approx(1.0)
    assert not np.array_equal(sample["face"][:3].numpy(), sample["target"].numpy())
```

- [ ] **Step 6: Run the loader test to verify it fails**

Run:

```bash
rtk .venv/bin/pytest tests/test_dataset.py::test_duix_hf_dataset_loads_silent_talking_roi_row -q
```

Expected: FAIL because `DuixHFDataset` still requires `frame` and `bbox_xyxy`.

- [ ] **Step 7: Add a schema branch to `DuixHFDataset`**

In `edge_lipsync/dataset.py`, import `make_face_training_sample_from_rois` and add:

```python
SILENT_TALKING_SCHEMA_VERSION = "edge_lipsync_silent_talking_pair_v1"


def _silent_talking_hf_sample(row: dict[str, Any]) -> dict[str, Any]:
    source_roi = _hf_frame_to_bgr(row["source_roi"])
    target_roi = _hf_frame_to_bgr(row["target_roi"])
    audio = np.asarray(row["audio"], dtype=np.float32)
    if audio.shape != (20, 256):
        raise ValueError(f"Invalid audio shape={audio.shape}, expected=(20, 256)")
    sample = make_face_training_sample_from_rois(source_roi, target_roi)
    meta = {
        "schema_version": SILENT_TALKING_SCHEMA_VERSION,
        "persona_id": str(row["persona_id"]),
        "pair_id": str(row["pair_id"]),
        "clip_id": str(row["talking_clip_id"]),
        "talking_clip_id": str(row["talking_clip_id"]),
        "source_frame_idx": int(row["source_frame_idx"]),
        "target_frame_idx": int(row["target_frame_idx"]),
        "frame_idx": int(row["target_frame_idx"]),
        "audio_idx": int(row["audio_idx"]),
        "source_bbox_xyxy": tuple(int(v) for v in row["source_bbox_xyxy"]),
        "target_bbox_xyxy": tuple(int(v) for v in row["target_bbox_xyxy"]),
        "sample_weight": float(row["sample_weight"]),
        "flags": tuple(str(value) for value in row.get("flags", [])),
    }
    for key in (
        "is_idle",
        "sync_best_lag_frames",
        "sync_correlation",
        "sync_confidence",
        "pose_delta_yaw",
        "pose_delta_pitch",
        "pose_delta_roll",
        "center_delta_x",
        "center_delta_y",
        "width_ratio",
        "height_ratio",
        "stable_landmark_alignment_rmse",
        "mouth_center_delta_after_crop",
        "matching_score",
        "valid_silent_candidate_count",
        "second_best_matching_score",
        "matching_score_margin",
        "source_face_blur",
        "target_face_blur",
        "target_mouth_blur",
    ):
        if key in row:
            meta[key] = row[key]
    return {
        "face": torch.from_numpy(sample.face),
        "audio": torch.from_numpy(audio),
        "target": torch.from_numpy(sample.target),
        "meta": meta,
    }
```

At the start of `DuixHFDataset.__getitem__`, branch when
`row.get("schema_version") == SILENT_TALKING_SCHEMA_VERSION`. Preserve the existing legacy branch
unchanged.

- [ ] **Step 8: Add an early Hugging Face Image portability spike**

Before the full builder work, add this portability spike to `tests/test_dataset.py`:

```python
def test_hf_image_feature_roundtrips_embedded_png_without_path(tmp_path: Path) -> None:
    from datasets import Dataset, Features, Image, load_from_disk

    image = np.full((168, 168, 3), 120, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    dataset = Dataset.from_list(
        [{"image": {"bytes": encoded.tobytes(), "path": None}}],
        features=Features({"image": Image()}),
    )
    path = tmp_path / "image_dataset"
    dataset.save_to_disk(path)

    loaded = load_from_disk(path)
    physical = loaded.cast_column("image", Image(decode=False))[0]["image"]

    assert physical["path"] is None
    assert physical["bytes"].startswith(b"\x89PNG")
    assert np.asarray(loaded[0]["image"]).shape == (168, 168, 3)
```

If this spike fails in the installed `datasets` version, stop this task and replace the planned
`Image` fields with `Value("binary")` plus explicit OpenCV decoding before proceeding. Do not build
the remaining snapshot pipeline on path-backed images.

- [ ] **Step 9: Run focused and legacy dataset tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_preprocess.py tests/test_dataset.py tests/test_hf_datasets.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
rtk git add edge_lipsync/preprocess.py edge_lipsync/dataset.py tests/test_preprocess.py tests/test_dataset.py
rtk git commit -m "feat(dataset): load paired roi training rows"
```

---

### Task 2: Frame Pose, Geometry, Blur, And Continuity Primitives

**Files:**
- Create: `edge_lipsync/pose_pairing.py`
- Create: `tests/test_pose_pairing.py`

- [ ] **Step 1: Write failing observation and pose tests**

Create `tests/test_pose_pairing.py` with:

```python
from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _landmarks() -> dict[int, tuple[float, float]]:
    return {
        1: (100.0, 90.0),
        10: (100.0, 45.0),
        33: (70.0, 70.0),
        61: (82.0, 118.0),
        152: (100.0, 150.0),
        234: (55.0, 100.0),
        263: (130.0, 70.0),
        291: (118.0, 118.0),
        454: (145.0, 100.0),
    }


def test_head_pose_landmark_subset_excludes_lips() -> None:
    from edge_lipsync.pose_pairing import HEAD_POSE_LANDMARK_INDICES

    assert 61 not in HEAD_POSE_LANDMARK_INDICES
    assert 291 not in HEAD_POSE_LANDMARK_INDICES


def test_analyze_landmark_geometry_normalizes_center_and_size() -> None:
    from edge_lipsync.pose_pairing import normalized_bbox_geometry

    geometry = normalized_bbox_geometry((50, 40, 150, 160), (200, 200, 3))

    assert geometry.center_x == pytest.approx(0.5)
    assert geometry.center_y == pytest.approx(0.5)
    assert geometry.width == pytest.approx(0.5)
    assert geometry.height == pytest.approx(0.6)


def test_mouth_openness_is_normalized_by_mouth_width() -> None:
    from edge_lipsync.pose_pairing import mouth_openness

    landmarks = {
        **_landmarks(),
        13: (100.0, 112.0),
        14: (100.0, 120.0),
    }

    assert mouth_openness(landmarks) == pytest.approx(8.0 / 36.0)


def test_rotation_matrix_to_euler_returns_yaw_pitch_roll_degrees() -> None:
    from edge_lipsync.pose_pairing import rotation_matrix_to_euler

    rotation = np.eye(3, dtype=np.float64)

    pose = rotation_matrix_to_euler(rotation)

    assert pose.yaw == pytest.approx(0.0)
    assert pose.pitch == pytest.approx(0.0)
    assert pose.roll == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify the module is missing**

Run:

```bash
rtk .venv/bin/pytest tests/test_pose_pairing.py -q
```

Expected: FAIL because `edge_lipsync.pose_pairing` does not exist.

- [ ] **Step 3: Implement core dataclasses and geometry**

Create `edge_lipsync/pose_pairing.py` with:

```python
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Mapping

import cv2
import numpy as np

from edge_lipsync.preprocess import BBox, Point

HEAD_POSE_LANDMARK_INDICES = (1, 10, 33, 234, 263, 454)
STABLE_ALIGNMENT_LANDMARK_INDICES = (1, 33, 152, 234, 263, 454)
MOUTH_CORNER_INDICES = (61, 291)
MOUTH_VERTICAL_INDICES = (13, 14)
TRACKED_LANDMARK_INDICES = tuple(
    sorted(
        set(HEAD_POSE_LANDMARK_INDICES)
        | set(STABLE_ALIGNMENT_LANDMARK_INDICES)
        | set(MOUTH_CORNER_INDICES)
        | set(MOUTH_VERTICAL_INDICES)
    )
)


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True)
class NormalizedGeometry:
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class FrameObservation:
    frame_idx: int
    bbox_xyxy: BBox | None
    frame_width: int
    frame_height: int
    landmarks: dict[int, Point]
    pose: HeadPose | None
    face_blur: float
    mouth_blur: float
    mouth_open: float
    landmark_valid: bool
    bbox_continuity_valid: bool = True
    reject_reason: str = ""


def normalized_bbox_geometry(
    bbox: BBox,
    frame_shape: tuple[int, ...],
) -> NormalizedGeometry:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return NormalizedGeometry(
        center_x=(x1 + x2) / (2.0 * width),
        center_y=(y1 + y2) / (2.0 * height),
        width=(x2 - x1) / width,
        height=(y2 - y1) / height,
    )


def mouth_openness(landmarks: Mapping[int, Point]) -> float:
    left = landmarks[61]
    right = landmarks[291]
    upper = landmarks[13]
    lower = landmarks[14]
    width = max(math.dist(left, right), 1e-6)
    return math.dist(upper, lower) / width


def rotation_matrix_to_euler(rotation: np.ndarray) -> HeadPose:
    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        pitch = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = 0.0
    return HeadPose(
        yaw=math.degrees(yaw),
        pitch=math.degrees(pitch),
        roll=math.degrees(roll),
    )
```

- [ ] **Step 4: Implement `solvePnP`, blur, mouth crop, and continuity**

Continue in `pose_pairing.py`:

```python
HEAD_MODEL_POINTS = np.asarray(
    [
        (0.0, 0.0, 0.0),
        (0.0, -55.0, -10.0),
        (-32.0, -28.0, -20.0),
        (-48.0, 5.0, -15.0),
        (32.0, -28.0, -20.0),
        (48.0, 5.0, -15.0),
    ],
    dtype=np.float64,
)


def estimate_head_pose(
    landmarks: Mapping[int, Point],
    frame_shape: tuple[int, ...],
) -> HeadPose:
    height, width = frame_shape[:2]
    image_points = np.asarray(
        [landmarks[index] for index in HEAD_POSE_LANDMARK_INDICES],
        dtype=np.float64,
    )
    focal = float(max(width, height))
    camera = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rotation_vector, _translation = cv2.solvePnP(
        HEAD_MODEL_POINTS,
        image_points,
        camera,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP failed")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return rotation_matrix_to_euler(rotation)


def laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mouth_bbox(
    landmarks: Mapping[int, Point],
    frame_shape: tuple[int, ...],
    *,
    padding_fraction: float = 0.35,
) -> BBox:
    height, width = frame_shape[:2]
    points = np.asarray(
        [landmarks[index] for index in (*MOUTH_CORNER_INDICES, *MOUTH_VERTICAL_INDICES)],
        dtype=np.float32,
    )
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    pad = max(float(x2 - x1), float(y2 - y1), 1.0) * padding_fraction
    return (
        max(0, int(math.floor(x1 - pad))),
        max(0, int(math.floor(y1 - pad))),
        min(width, int(math.ceil(x2 + pad))),
        min(height, int(math.ceil(y2 + pad))),
    )


def mark_bbox_continuity(
    observations: list[FrameObservation],
    *,
    max_center_distance: float = 0.05,
    max_scale_ratio: float = 1.15,
) -> list[FrameObservation]:
    out: list[FrameObservation] = []
    previous: FrameObservation | None = None
    for observation in observations:
        if observation.bbox_xyxy is None:
            out.append(observation)
            continue
        valid = True
        if previous is not None and previous.bbox_xyxy is not None:
            current_geometry = normalized_bbox_geometry(
                observation.bbox_xyxy,
                (observation.frame_height, observation.frame_width, 3),
            )
            previous_geometry = normalized_bbox_geometry(
                previous.bbox_xyxy,
                (previous.frame_height, previous.frame_width, 3),
            )
            center_distance = math.hypot(
                current_geometry.center_x - previous_geometry.center_x,
                current_geometry.center_y - previous_geometry.center_y,
            )
            width_ratio = current_geometry.width / previous_geometry.width
            height_ratio = current_geometry.height / previous_geometry.height
            valid = (
                center_distance <= max_center_distance
                and 1.0 / max_scale_ratio <= width_ratio <= max_scale_ratio
                and 1.0 / max_scale_ratio <= height_ratio <= max_scale_ratio
            )
        updated = replace(
            observation,
            bbox_continuity_valid=valid,
            reject_reason="" if valid else "bbox_discontinuity",
        )
        out.append(updated)
        if valid:
            previous = updated
    return out
```

- [ ] **Step 5: Add tests for blur and sequence continuity**

Append:

```python
def test_laplacian_variance_separates_sharp_and_blurred_images() -> None:
    from edge_lipsync.pose_pairing import laplacian_variance

    sharp = np.zeros((64, 64, 3), dtype=np.uint8)
    sharp[:, ::2] = 255
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

    assert laplacian_variance(sharp) > laplacian_variance(blurred)


def test_mark_bbox_continuity_rejects_jump_in_sequence() -> None:
    from edge_lipsync.pose_pairing import FrameObservation, mark_bbox_continuity

    base = dict(
        frame_width=200,
        frame_height=200,
        landmarks={},
        pose=None,
        face_blur=100.0,
        mouth_blur=100.0,
        mouth_open=0.0,
        landmark_valid=True,
    )
    observations = [
        FrameObservation(frame_idx=1, bbox_xyxy=(50, 50, 150, 150), **base),
        FrameObservation(frame_idx=2, bbox_xyxy=(120, 50, 200, 130), **base),
    ]

    result = mark_bbox_continuity(observations)

    assert result[0].bbox_continuity_valid is True
    assert result[1].bbox_continuity_valid is False
    assert result[1].reject_reason == "bbox_discontinuity"
```

- [ ] **Step 6: Run pose primitive tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_pose_pairing.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add edge_lipsync/pose_pairing.py tests/test_pose_pairing.py
rtk git commit -m "feat(dataset): add pose and frame quality analysis"
```

---

### Task 3: Audio-Video Sync Windows

**Files:**
- Modify: `edge_lipsync/pose_pairing.py`
- Modify: `tests/test_pose_pairing.py`

- [ ] **Step 1: Write failing lag-search and assignment tests**

Append:

```python
def test_sync_search_finds_visual_lag() -> None:
    from edge_lipsync.pose_pairing import best_sync_lag

    audio = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    mouth = np.asarray([0, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)

    lag, correlation = best_sync_lag(audio, mouth, max_lag_frames=3)

    assert lag == 1
    assert correlation == pytest.approx(1.0)


def test_low_correlation_speech_window_does_not_reject_by_lag() -> None:
    from edge_lipsync.pose_pairing import SyncWindow, sync_reject_reason

    window = SyncWindow(
        window_id=0,
        start_frame=0,
        end_frame=50,
        center_frame=25.0,
        has_speech=True,
        best_lag_frames=3,
        best_correlation=0.10,
        confidence="low",
    )

    assert sync_reject_reason(window, min_correlation=0.20, max_abs_lag=2) is None


def test_nearest_sync_window_assigns_each_frame_once() -> None:
    from edge_lipsync.pose_pairing import SyncWindow, assign_sync_windows

    windows = [
        SyncWindow(0, 0, 50, 25.0, True, 0, 0.8, "high"),
        SyncWindow(1, 25, 75, 50.0, True, 1, 0.8, "high"),
    ]

    assignments = assign_sync_windows(frame_count=75, windows=windows)

    assert assignments[30].window_id == 0
    assert assignments[45].window_id == 1


def test_fill_missing_signal_interpolates_only_for_sync_analysis() -> None:
    from edge_lipsync.pose_pairing import fill_missing_signal

    values = np.asarray([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=np.float32)

    filled = fill_missing_signal(values)

    assert np.allclose(filled, [1.0, 1.0, 2.0, 3.0, 3.0])
```

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
rtk .venv/bin/pytest tests/test_pose_pairing.py -q
```

Expected: FAIL because sync functions are not defined.

- [ ] **Step 3: Implement sync dataclasses and correlation**

Add:

```python
@dataclass(frozen=True)
class SyncWindow:
    window_id: int
    start_frame: int
    end_frame: int
    center_frame: float
    has_speech: bool
    best_lag_frames: int
    best_correlation: float
    confidence: str


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left.astype(np.float64) - float(left.mean())
    right_centered = right.astype(np.float64) - float(right.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def fill_missing_signal(values: np.ndarray) -> np.ndarray:
    indices = np.arange(len(values), dtype=np.float64)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float32)
    return np.interp(indices, indices[valid], values[valid]).astype(np.float32)


def best_sync_lag(
    audio: np.ndarray,
    mouth: np.ndarray,
    *,
    max_lag_frames: int,
) -> tuple[int, float]:
    candidates: list[tuple[float, int]] = []
    effective_max_lag = min(max_lag_frames, max(0, len(audio) - 2))
    for lag in range(-effective_max_lag, effective_max_lag + 1):
        if lag >= 0:
            left = audio[: len(audio) - lag or None]
            right = mouth[lag:]
        else:
            left = audio[-lag:]
            right = mouth[: len(mouth) + lag]
        correlation = _pearson(left, right)
        candidates.append((correlation, lag))
    correlation, lag = max(
        candidates,
        key=lambda item: (item[0], -abs(item[1]), -item[1]),
    )
    return lag, correlation


def build_sync_windows(
    audio_rms: np.ndarray,
    mouth_open: np.ndarray,
    *,
    fps: int = 25,
    window_seconds: float = 2.0,
    stride_seconds: float = 1.0,
    max_lag_frames: int = 3,
    silence_rms_threshold: float = 0.001,
    speech_fraction_threshold: float = 0.25,
    min_correlation: float = 0.20,
) -> list[SyncWindow]:
    if audio_rms.shape != mouth_open.shape:
        raise ValueError("audio_rms and mouth_open must have matching frame counts")
    window = int(round(window_seconds * fps))
    stride = int(round(stride_seconds * fps))
    starts = list(range(0, max(1, len(audio_rms) - window + 1), stride))
    if not starts or starts[-1] + window < len(audio_rms):
        starts.append(max(0, len(audio_rms) - window))
    windows: list[SyncWindow] = []
    for window_id, start in enumerate(sorted(set(starts))):
        end = min(len(audio_rms), start + window)
        audio_slice = audio_rms[start:end]
        mouth_slice = mouth_open[start:end]
        voiced_fraction = float(np.mean(audio_slice > silence_rms_threshold))
        has_speech = voiced_fraction >= speech_fraction_threshold
        lag, correlation = best_sync_lag(
            audio_slice,
            mouth_slice,
            max_lag_frames=max_lag_frames,
        )
        windows.append(
            SyncWindow(
                window_id=window_id,
                start_frame=start,
                end_frame=end,
                center_frame=(start + end - 1) / 2.0,
                has_speech=has_speech,
                best_lag_frames=lag,
                best_correlation=correlation,
                confidence="high" if correlation >= min_correlation else "low",
            )
        )
    return windows


def assign_sync_windows(
    *,
    frame_count: int,
    windows: list[SyncWindow],
) -> dict[int, SyncWindow]:
    if not windows:
        raise ValueError("At least one sync window is required")
    return {
        frame_idx: min(
            windows,
            key=lambda window: (abs(window.center_frame - frame_idx), window.window_id),
        )
        for frame_idx in range(frame_count)
    }


def sync_reject_reason(
    window: SyncWindow,
    *,
    min_correlation: float,
    max_abs_lag: int,
) -> str | None:
    if not window.has_speech or window.best_correlation < min_correlation:
        return None
    return "sync_lag" if abs(window.best_lag_frames) > max_abs_lag else None
```

- [ ] **Step 4: Run sync tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_pose_pairing.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add edge_lipsync/pose_pairing.py tests/test_pose_pairing.py
rtk git commit -m "feat(dataset): add coarse audio video sync gates"
```

---

### Task 4: Full Hard-Gate Matching, Splits, And Idle Selection

**Files:**
- Modify: `edge_lipsync/pose_pairing.py`
- Modify: `tests/test_pose_pairing.py`

- [ ] **Step 1: Write failing post-crop alignment tests**

Append:

```python
def _observation(
    *,
    frame_idx: int,
    mouth_shift: float = 0.0,
    yaw_shift: float = 0.0,
) -> Any:
    from edge_lipsync.pose_pairing import FrameObservation, HeadPose

    landmarks = dict(_landmarks())
    for index in (61, 291):
        x, y = landmarks[index]
        landmarks[index] = (x + mouth_shift, y)
    return FrameObservation(
        frame_idx=frame_idx,
        bbox_xyxy=(50, 40, 150, 160),
        frame_width=200,
        frame_height=200,
        landmarks=landmarks,
        pose=HeadPose(yaw=yaw_shift, pitch=0.0, roll=0.0),
        face_blur=100.0,
        mouth_blur=100.0,
        mouth_open=0.2,
        landmark_valid=True,
    )


def test_post_crop_alignment_compares_normalized_roi_landmarks() -> None:
    from edge_lipsync.pose_pairing import post_crop_alignment

    source = _landmarks()
    target = {index: (x + 10.0, y + 20.0) for index, (x, y) in source.items()}

    result = post_crop_alignment(
        source,
        (50, 40, 150, 160),
        target,
        (60, 60, 160, 180),
    )

    assert result.stable_landmark_rmse == pytest.approx(0.0)
    assert result.mouth_center_delta == pytest.approx(0.0)


def test_matching_filters_alignment_before_scoring() -> None:
    from edge_lipsync.pose_pairing import (
        FrameObservation,
        HeadPose,
        MatchConfig,
        match_silent_observation,
    )

    target = _observation(frame_idx=10, mouth_shift=0.0)
    bad_best_pose = _observation(frame_idx=1, mouth_shift=20.0)
    valid_second = _observation(frame_idx=2, yaw_shift=1.0)

    result = match_silent_observation(
        target,
        [bad_best_pose, valid_second],
        MatchConfig(),
    )

    assert result.selected.frame_idx == 2
    assert result.valid_candidate_count == 1
    assert result.second_best_score is None
```

- [ ] **Step 2: Write failing split and idle tests**

Append:

```python
def test_assign_video_splits_keeps_each_clip_in_one_split() -> None:
    from edge_lipsync.pose_pairing import assign_video_splits

    splits = assign_video_splits(
        "nora",
        ["a", "b", "c"],
        split_salt="edge-lipsync-v1",
        validation_fraction=0.2,
    )

    assert set(splits) == {"a", "b", "c"}
    assert set(splits.values()) == {"train", "val"}


def test_select_idle_frames_applies_strict_ten_percent_cap() -> None:
    from edge_lipsync.pose_pairing import select_idle_frame_indices

    selected = select_idle_frame_indices(
        idle_frame_indices=list(range(100)),
        speech_frame_indices=list(range(100, 200)),
        max_ratio=0.10,
    )

    assert len(selected) == 10
    assert selected == sorted(selected)
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
rtk .venv/bin/pytest tests/test_pose_pairing.py -q
```

Expected: FAIL because matching and selection functions are missing.

- [ ] **Step 4: Implement matching dataclasses and gates**

Add:

```python
@dataclass(frozen=True)
class MatchConfig:
    max_yaw_delta: float = 5.0
    max_pitch_delta: float = 5.0
    max_roll_delta: float = 4.0
    max_center_x_delta: float = 0.05
    max_center_y_delta: float = 0.05
    min_width_ratio: float = 0.9
    max_width_ratio: float = 1.1
    min_height_ratio: float = 0.9
    max_height_ratio: float = 1.1
    max_stable_landmark_rmse: float = 0.04
    max_mouth_center_delta: float = 0.04
    pose_weight: float = 1.0
    position_weight: float = 1.0
    scale_weight: float = 1.0


@dataclass(frozen=True)
class AlignmentMetrics:
    stable_landmark_rmse: float
    mouth_center_delta: float


@dataclass(frozen=True)
class MatchResult:
    selected: FrameObservation
    matching_score: float
    valid_candidate_count: int
    second_best_score: float | None
    matching_score_margin: float | None
    pose_delta: HeadPose
    center_delta_x: float
    center_delta_y: float
    width_ratio: float
    height_ratio: float
    alignment: AlignmentMetrics


def _roi_point(point: Point, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.asarray(
        [(point[0] - x1) / (x2 - x1), (point[1] - y1) / (y2 - y1)],
        dtype=np.float64,
    )


def post_crop_alignment(
    source_landmarks: Mapping[int, Point],
    source_bbox: BBox,
    target_landmarks: Mapping[int, Point],
    target_bbox: BBox,
) -> AlignmentMetrics:
    source_stable = np.stack(
        [_roi_point(source_landmarks[index], source_bbox) for index in STABLE_ALIGNMENT_LANDMARK_INDICES]
    )
    target_stable = np.stack(
        [_roi_point(target_landmarks[index], target_bbox) for index in STABLE_ALIGNMENT_LANDMARK_INDICES]
    )
    stable_rmse = float(np.sqrt(np.mean((source_stable - target_stable) ** 2)))
    source_mouth = np.mean(
        [_roi_point(source_landmarks[index], source_bbox) for index in MOUTH_CORNER_INDICES],
        axis=0,
    )
    target_mouth = np.mean(
        [_roi_point(target_landmarks[index], target_bbox) for index in MOUTH_CORNER_INDICES],
        axis=0,
    )
    return AlignmentMetrics(
        stable_landmark_rmse=stable_rmse,
        mouth_center_delta=float(np.linalg.norm(source_mouth - target_mouth)),
    )
```

Implement `match_silent_observation()` with this exact order:

1. Reject invalid source observations.
2. Apply pose, normalized center, normalized width, and normalized height gates.
3. Apply `post_crop_alignment()` gates.
4. Score only remaining candidates with threshold-normalized pose, position, and directional
   log-ratio scale terms.
5. Sort by `(score, source.frame_idx)`.
6. Return candidate count, best, second-best, and `second - best` margin.
7. Raise `ValueError("pose_geometry_no_match")` when step 2 leaves none.
8. Raise `ValueError("post_crop_alignment_mismatch")` when step 2 has candidates but step 3 leaves
   none.

- [ ] **Step 5: Implement deterministic splits and idle sampling**

Add:

```python
def assign_video_splits(
    persona_id: str,
    talking_clip_ids: list[str],
    *,
    split_salt: str,
    validation_fraction: float,
) -> dict[str, str]:
    clip_ids = sorted(set(talking_clip_ids))
    if len(clip_ids) < 2:
        raise ValueError("Video-level split requires at least two talking clips")
    ranked = sorted(
        clip_ids,
        key=lambda clip_id: hashlib.sha256(
            f"{persona_id}:{split_salt}:{clip_id}".encode()
        ).digest(),
    )
    val_count = min(len(ranked) - 1, max(1, round(len(ranked) * validation_fraction)))
    val_ids = set(ranked[-val_count:])
    return {clip_id: ("val" if clip_id in val_ids else "train") for clip_id in clip_ids}


def single_video_split(
    frame_idx: int,
    *,
    frame_count: int,
    validation_fraction: float,
) -> str:
    split_at = max(1, int(math.floor(frame_count * (1.0 - validation_fraction))))
    return "val" if frame_idx >= split_at else "train"


def select_idle_frame_indices(
    *,
    idle_frame_indices: list[int],
    speech_frame_indices: list[int],
    max_ratio: float,
) -> list[int]:
    limit = int(math.floor(len(speech_frame_indices) * max_ratio))
    if limit <= 0 or not idle_frame_indices:
        return []
    speech = np.asarray(sorted(speech_frame_indices), dtype=np.int64)
    ranked = sorted(
        set(idle_frame_indices),
        key=lambda frame_idx: (
            int(np.min(np.abs(speech - frame_idx))),
            frame_idx,
        ),
    )
    pool = sorted(ranked[: max(limit * 4, limit)])
    positions = np.linspace(0, len(pool) - 1, limit, dtype=int)
    return sorted({pool[position] for position in positions})
```

The idle function prioritizes candidates near speech first, then distributes the retained subset
across that priority pool.

- [ ] **Step 6: Run all pure algorithm tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_pose_pairing.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add edge_lipsync/pose_pairing.py tests/test_pose_pairing.py
rtk git commit -m "feat(dataset): match silent frames to talking poses"
```

---

### Task 5: Builder Configuration, Discovery, Normalization, And Analysis Cache

**Files:**
- Create: `edge_lipsync/silent_talking_dataset.py`
- Create: `tests/test_silent_talking_dataset.py`

- [ ] **Step 1: Write failing config and discovery tests**

Create `tests/test_silent_talking_dataset.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def test_build_config_resolves_persona_layout(tmp_path: Path) -> None:
    from edge_lipsync.silent_talking_dataset import SilentTalkingBuildConfig

    config = SilentTalkingBuildConfig(
        data_root=str(tmp_path),
        persona_id="nora",
        snapshot_root=str(tmp_path / "snapshot"),
        work_root=str(tmp_path / "work"),
        wenet_onnx=str(tmp_path / "wenet.onnx"),
    )

    assert config.silent_video_path == tmp_path / "nora/silent/defaultvideo.mp4"
    assert config.talking_video_dir == tmp_path / "nora/talking"


def test_discover_talking_videos_is_sorted(tmp_path: Path) -> None:
    from edge_lipsync.silent_talking_dataset import discover_talking_videos

    talking = tmp_path / "nora/talking"
    talking.mkdir(parents=True)
    (talking / "b.mp4").write_bytes(b"b")
    (talking / "a.mp4").write_bytes(b"a")

    assert [path.name for path in discover_talking_videos(talking)] == ["a.mp4", "b.mp4"]
```

- [ ] **Step 2: Run tests to verify the builder module is missing**

Run:

```bash
rtk .venv/bin/pytest tests/test_silent_talking_dataset.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Add config and portable input identities**

Create `edge_lipsync/silent_talking_dataset.py`:

```python
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edge_lipsync.audio_features import (
    extract_bnf_windows_from_wav,
    load_wav_mono_f32,
    split_audio_blocks,
)
from edge_lipsync.build_dataset import extract_frames, require_tool, run
from edge_lipsync.landmarks import MediaPipeFaceLandmarkerDetector
from edge_lipsync.pose_pairing import (
    TRACKED_LANDMARK_INDICES,
    FrameObservation,
    HeadPose,
    MatchConfig,
    estimate_head_pose,
    laplacian_variance,
    mark_bbox_continuity,
    mouth_bbox,
    mouth_openness,
)
from edge_lipsync.preprocess import landmarks_to_duix_roi

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}
SCHEMA_VERSION = "edge_lipsync_silent_talking_pair_v1"


@dataclass(frozen=True)
class BlurConfig:
    min_source_face_laplacian_variance: float = 60.0
    min_target_face_laplacian_variance: float = 60.0
    min_target_mouth_laplacian_variance: float = 40.0


@dataclass(frozen=True)
class SyncConfig:
    window_seconds: float = 2.0
    stride_seconds: float = 1.0
    max_lag_frames: int = 3
    max_reject_lag_frames: int = 2
    min_correlation: float = 0.20
    silence_rms_threshold: float = 0.001
    speech_fraction_threshold: float = 0.25


@dataclass(frozen=True)
class SilentTalkingBuildConfig:
    data_root: str
    persona_id: str
    snapshot_root: str
    work_root: str
    wenet_onnx: str
    landmark_model_asset_path: str | None = None
    fps: int = 25
    sample_rate: int = 16000
    validation_fraction: float = 0.20
    split_salt: str = "edge-lipsync-silent-talking-v1"
    idle_max_ratio: float = 0.10
    idle_sample_weight: float = 0.25
    preview_count_per_group: int = 8
    progress: bool = True
    strict: bool = False
    match: MatchConfig = field(default_factory=MatchConfig)
    blur: BlurConfig = field(default_factory=BlurConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)

    @property
    def persona_root(self) -> Path:
        return Path(self.data_root) / self.persona_id

    @property
    def silent_video_path(self) -> Path:
        return self.persona_root / "silent" / "defaultvideo.mp4"

    @property
    def talking_video_dir(self) -> Path:
        return self.persona_root / "talking"


def discover_talking_videos(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    videos = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"No talking videos found in {root}")
    return videos


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def config_sha256(config: SilentTalkingBuildConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
```

- [ ] **Step 4: Add visual-only and talking normalization**

Add:

```python
def normalize_visual_video(src: Path, out: Path, *, fps: int) -> Path:
    ffmpeg = require_tool("ffmpeg")
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-vf",
            f"fps={fps}",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            str(out),
        ]
    )
    return out


def normalize_talking_video(
    src: Path,
    video_out: Path,
    audio_out: Path,
    *,
    fps: int,
) -> tuple[Path, Path]:
    normalize_visual_video(src, video_out, fps=fps)
    ffmpeg = require_tool("ffmpeg")
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(audio_out),
        ]
    )
    return video_out, audio_out
```

The Python audio loader performs the existing resampling to 16 kHz.

- [ ] **Step 5: Write a failing analysis cache test with a fake detector**

Add:

```python
def test_analyze_frames_writes_reusable_jsonl_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "000001.png"), np.full((200, 200, 3), 120, dtype=np.uint8))

    class FakeDetector:
        def detect_landmarks(self, _frame: np.ndarray) -> dict[int, tuple[float, float]]:
            return {
                1: (100.0, 90.0),
                10: (100.0, 45.0),
                13: (100.0, 112.0),
                14: (100.0, 120.0),
                33: (70.0, 70.0),
                61: (82.0, 118.0),
                152: (100.0, 150.0),
                234: (55.0, 100.0),
                263: (130.0, 70.0),
                291: (118.0, 118.0),
                454: (145.0, 100.0),
            }

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        builder,
        "estimate_head_pose",
        lambda _landmarks, _shape: builder.HeadPose(0.0, 0.0, 0.0),
    )
    cache = tmp_path / "analysis.jsonl"

    observations = builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        cache_path=cache,
        cache_metadata={"config_sha256": "abc", "input_sha256": "def"},
        is_target=True,
        show_progress=False,
    )

    assert len(observations) == 1
    assert observations[0].landmark_valid is True
    assert cache.is_file()
    assert (tmp_path / "analysis.meta.json").is_file()


def test_analyze_frames_rebuilds_when_cache_metadata_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "000001.png"), np.full((200, 200, 3), 120, dtype=np.uint8))
    calls = 0

    class FakeDetector:
        def detect_landmarks(self, _frame: np.ndarray) -> None:
            nonlocal calls
            calls += 1
            return None

        def close(self) -> None:
            pass

    cache = tmp_path / "analysis.jsonl"
    first = {"config_sha256": "a", "input_sha256": "x"}
    second = {"config_sha256": "b", "input_sha256": "x"}
    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        cache_path=cache,
        cache_metadata=first,
        is_target=True,
        show_progress=False,
    )
    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        cache_path=cache,
        cache_metadata=first,
        is_target=True,
        show_progress=False,
    )
    builder.analyze_frames(
        frames,
        frame_count=1,
        detector=FakeDetector(),
        cache_path=cache,
        cache_metadata=second,
        is_target=True,
        show_progress=False,
    )

    assert calls == 2
```

- [ ] **Step 6: Implement frame analysis and cache serialization**

Implement `analyze_frames()` to:

1. Read PNG frames by one-based index.
2. Call `detector.detect_landmarks()`.
3. Retain only `TRACKED_LANDMARK_INDICES`.
4. Derive bbox with `landmarks_to_duix_roi()`.
5. Estimate pose, face blur, mouth blur, and mouth openness.
6. Emit invalid observations with explicit `reject_reason`.
7. Call `mark_bbox_continuity()` after all frames.
8. Write one JSON object per observation to a temporary JSONL file and atomically replace
   `analysis.jsonl`.
9. Write sibling `analysis.meta.json` atomically after the JSONL is complete.
10. Reuse the cache only when metadata matches exactly.

Use JSON helpers:

```python
def observation_to_json(observation: FrameObservation) -> dict[str, Any]:
    return {
        "frame_idx": observation.frame_idx,
        "bbox_xyxy": list(observation.bbox_xyxy) if observation.bbox_xyxy else None,
        "frame_width": observation.frame_width,
        "frame_height": observation.frame_height,
        "landmarks": {str(index): list(point) for index, point in observation.landmarks.items()},
        "pose": asdict(observation.pose) if observation.pose else None,
        "face_blur": observation.face_blur,
        "mouth_blur": observation.mouth_blur,
        "mouth_open": observation.mouth_open,
        "landmark_valid": observation.landmark_valid,
        "bbox_continuity_valid": observation.bbox_continuity_valid,
        "reject_reason": observation.reject_reason,
    }
```

- [ ] **Step 7: Run builder discovery and cache tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_silent_talking_dataset.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add edge_lipsync/silent_talking_dataset.py tests/test_silent_talking_dataset.py
rtk git commit -m "feat(dataset): analyze silent and talking video frames"
```

---

### Task 6: Build Pair Decisions, Dataset Rows, Reports, And Previews

**Files:**
- Modify: `edge_lipsync/silent_talking_dataset.py`
- Modify: `tests/test_silent_talking_dataset.py`

- [ ] **Step 1: Write failing exact-BNF and decision-order tests**

Add:

```python
def _builder_landmarks() -> dict[int, tuple[float, float]]:
    return {
        1: (100.0, 90.0),
        10: (100.0, 45.0),
        13: (100.0, 112.0),
        14: (100.0, 120.0),
        33: (70.0, 70.0),
        61: (82.0, 118.0),
        152: (100.0, 150.0),
        234: (55.0, 100.0),
        263: (130.0, 70.0),
        291: (118.0, 118.0),
        454: (145.0, 100.0),
    }


def _valid_observation(frame_idx: int):
    from edge_lipsync.pose_pairing import FrameObservation, HeadPose

    return FrameObservation(
        frame_idx=frame_idx,
        bbox_xyxy=(50, 40, 150, 160),
        frame_width=200,
        frame_height=200,
        landmarks=_builder_landmarks(),
        pose=HeadPose(0.0, 0.0, 0.0),
        face_blur=100.0,
        mouth_blur=100.0,
        mouth_open=0.2,
        landmark_valid=True,
    )


def _invalid_observation(frame_idx: int, reason: str):
    from dataclasses import replace

    return replace(
        _valid_observation(frame_idx),
        bbox_xyxy=None,
        pose=None,
        landmarks={},
        landmark_valid=False,
        reject_reason=reason,
    )


def _test_config():
    from edge_lipsync.silent_talking_dataset import SilentTalkingBuildConfig

    return SilentTalkingBuildConfig(
        data_root="data",
        persona_id="nora",
        snapshot_root="snapshot",
        work_root="work",
        wenet_onnx="wenet.onnx",
        progress=False,
    )


def test_build_pair_decisions_uses_exact_precomputed_bnf_row() -> None:
    from edge_lipsync.silent_talking_dataset import build_pair_decisions

    bnf = np.arange(3 * 20 * 256, dtype=np.float32).reshape(3, 20, 256)
    result = build_pair_decisions(
        talking_observations=[_valid_observation(1)],
        silent_observations=[_valid_observation(7)],
        bnf_windows=bnf,
        audio_rms=np.ones(3, dtype=np.float32),
        config=_test_config(),
        split_for_frame=lambda _frame_idx: "train",
    )

    assert len(result.rows) == 1
    assert result.rows[0]["audio_idx"] == 0
    assert np.array_equal(result.rows[0]["audio"], bnf[0])


def test_build_pair_decisions_records_every_talking_frame() -> None:
    from edge_lipsync.silent_talking_dataset import build_pair_decisions

    result = build_pair_decisions(
        talking_observations=[
            _invalid_observation(1, "landmark_missing"),
            _valid_observation(2),
        ],
        silent_observations=[_valid_observation(7)],
        bnf_windows=np.zeros((2, 20, 256), dtype=np.float32),
        audio_rms=np.ones(2, dtype=np.float32),
        config=_test_config(),
        split_for_frame=lambda _frame_idx: "train",
    )

    assert len(result.decisions) == 2
    assert result.decisions[0]["reject_reason"] == "landmark_missing"
    assert result.decisions[1]["status"] == "retained"
```

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
rtk .venv/bin/pytest tests/test_silent_talking_dataset.py -q
```

Expected: FAIL because row and decision builders do not exist.

- [ ] **Step 3: Define the canonical Features and row encoder**

In `silent_talking_dataset.py`, import `Callable` from `collections.abc` and import `Array2D`,
`Dataset`, `DatasetDict`, `Features`, `Image`, `Sequence`, and `Value`, then define:

```python
SILENT_TALKING_FEATURES = Features(
    {
        "schema_version": Value("string"),
        "persona_id": Value("string"),
        "pair_id": Value("string"),
        "talking_clip_id": Value("string"),
        "source_frame_idx": Value("int32"),
        "target_frame_idx": Value("int32"),
        "audio_idx": Value("int32"),
        "source_roi": Image(decode=True),
        "target_roi": Image(decode=True),
        "audio": Array2D(shape=(20, 256), dtype="float32"),
        "source_bbox_xyxy": Sequence(Value("int32"), length=4),
        "target_bbox_xyxy": Sequence(Value("int32"), length=4),
        "source_frame_width": Value("int32"),
        "source_frame_height": Value("int32"),
        "target_frame_width": Value("int32"),
        "target_frame_height": Value("int32"),
        "sample_weight": Value("float32"),
        "is_idle": Value("bool"),
        "sync_best_lag_frames": Value("int32"),
        "sync_correlation": Value("float32"),
        "sync_confidence": Value("string"),
        "pose_delta_yaw": Value("float32"),
        "pose_delta_pitch": Value("float32"),
        "pose_delta_roll": Value("float32"),
        "center_delta_x": Value("float32"),
        "center_delta_y": Value("float32"),
        "width_ratio": Value("float32"),
        "height_ratio": Value("float32"),
        "stable_landmark_alignment_rmse": Value("float32"),
        "mouth_center_delta_after_crop": Value("float32"),
        "matching_score": Value("float32"),
        "valid_silent_candidate_count": Value("int32"),
        "second_best_matching_score": Value("float32"),
        "matching_score_margin": Value("float32"),
        "source_face_blur": Value("float32"),
        "target_face_blur": Value("float32"),
        "target_mouth_blur": Value("float32"),
        "flags": Sequence(Value("string")),
    }
)


def encode_png(image_bgr: np.ndarray) -> dict[str, Any]:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Cannot encode ROI as PNG")
    return {"bytes": encoded.tobytes(), "path": None}
```

Hugging Face `Value("float32")` accepts null values in Arrow; use `None`, never `NaN`, for the two
optional score fields.

- [ ] **Step 4: Implement pair decision generation**

Add:

```python
@dataclass(frozen=True)
class PairDecisionResult:
    rows: list[dict[str, Any]]
    decisions: list[dict[str, Any]]


def build_pair_decisions(
    *,
    talking_observations: list[FrameObservation],
    silent_observations: list[FrameObservation],
    bnf_windows: np.ndarray,
    audio_rms: np.ndarray,
    config: SilentTalkingBuildConfig,
    split_for_frame: Callable[[int], str],
) -> PairDecisionResult:
```

Implement this exact pipeline:

1. Build mouth-open and audio-RMS arrays indexed by zero-based talking frame. Put `NaN` at
   landmark-invalid mouth frames, call `fill_missing_signal()` only for sync correlation, and keep
   those original frames rejected for training.
2. Build and assign sync windows.
3. For every talking observation, create a decision row before applying gates.
4. Reject invalid landmarks, invalid bbox continuity, target face blur, target mouth blur, missing
   exact BNF row, or qualified sync lag in that order.
5. Filter silent observations by landmark, continuity, and source face blur.
6. Call `match_silent_observation()`, preserving its two distinct no-match reasons.
7. Store a provisional speech or idle row with exact `bnf_windows[frame_idx - 1]`.
8. After all frames, retain all speech rows and select idle rows independently inside each split.
9. Set speech weight `1.0`, idle weight `config.idle_sample_weight`.
10. Update every frame decision to `retained`, `idle_downsampled`, or its gate rejection.

The function returns metadata without reading image files. Attach ROIs in the next helper after the
selected source and target frame indices are known.

- [ ] **Step 5: Implement ROI attachment and DatasetDict creation**

Add:

```python
def _read_frame(frames_dir: Path, frame_idx: int) -> np.ndarray:
    path = frames_dir / f"{frame_idx:06d}.png"
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(path)
    return frame


def attach_roi_images(
    row: dict[str, Any],
    *,
    silent_frames_dir: Path,
    talking_frames_dir: Path,
) -> dict[str, Any]:
    source_frame = _read_frame(silent_frames_dir, int(row["source_frame_idx"]))
    target_frame = _read_frame(talking_frames_dir, int(row["target_frame_idx"]))
    sx1, sy1, sx2, sy2 = row["source_bbox_xyxy"]
    tx1, ty1, tx2, ty2 = row["target_bbox_xyxy"]
    source_roi = cv2.resize(source_frame[sy1:sy2, sx1:sx2], (168, 168), interpolation=cv2.INTER_AREA)
    target_roi = cv2.resize(target_frame[ty1:ty2, tx1:tx2], (168, 168), interpolation=cv2.INTER_AREA)
    return {
        **row,
        "source_roi": encode_png(source_roi),
        "target_roi": encode_png(target_roi),
    }


def build_dataset_dict(rows: list[dict[str, Any]]) -> DatasetDict:
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val")
    }
    if not by_split["train"] or not by_split["val"]:
        raise ValueError("Both train and val splits must be non-empty")
    datasets = {
        split: Dataset.from_list(
            [{key: value for key, value in row.items() if key != "split"} for row in split_rows],
            features=SILENT_TALKING_FEATURES,
        )
        for split, split_rows in by_split.items()
    }
    return DatasetDict(datasets)
```

- [ ] **Step 6: Write reports, frame decisions, and previews**

Implement:

```python
def write_frame_decisions(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(decisions).to_parquet(str(path))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
```

Add deterministic preview selection functions for:

- Best matching score.
- Smallest margin.
- Nearest pose gate.
- Nearest center gate.
- Nearest width/height ratio gate.
- Nearest post-crop alignment gate.
- Low-confidence sync.
- Idle retained.
- Most frequent rejection reasons.

Each preview image is a horizontal grid with source ROI, target ROI, and a text panel. Use
`cv2.putText()` and write PNG files under `reports/previews/<clip_id>/<group>/`.

The per-clip JSON report must separately count:

```text
post_crop_alignment_mismatch_stable_landmark
post_crop_alignment_mismatch_mouth_center
```

It must also group retained observation pose values by mouth-openness quartile and report
`count`, `yaw_mean`, `yaw_std`, `pitch_mean`, `pitch_std`, `roll_mean`, and `roll_std` for each
non-empty bin.

- [ ] **Step 7: Add snapshot round-trip and physical-byte tests**

Add:

```python
def _complete_row(split: str) -> dict[str, object]:
    image = np.full((168, 168, 3), 120, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return {
        "schema_version": "edge_lipsync_silent_talking_pair_v1",
        "persona_id": "nora",
        "pair_id": f"{split}-pair",
        "talking_clip_id": f"{split}-clip",
        "source_frame_idx": 1,
        "target_frame_idx": 1,
        "audio_idx": 0,
        "source_roi": {"bytes": encoded.tobytes(), "path": None},
        "target_roi": {"bytes": encoded.tobytes(), "path": None},
        "audio": np.zeros((20, 256), dtype=np.float32),
        "source_bbox_xyxy": [50, 40, 150, 160],
        "target_bbox_xyxy": [50, 40, 150, 160],
        "source_frame_width": 200,
        "source_frame_height": 200,
        "target_frame_width": 200,
        "target_frame_height": 200,
        "sample_weight": 1.0,
        "is_idle": False,
        "sync_best_lag_frames": 0,
        "sync_correlation": 0.8,
        "sync_confidence": "high",
        "pose_delta_yaw": 0.0,
        "pose_delta_pitch": 0.0,
        "pose_delta_roll": 0.0,
        "center_delta_x": 0.0,
        "center_delta_y": 0.0,
        "width_ratio": 1.0,
        "height_ratio": 1.0,
        "stable_landmark_alignment_rmse": 0.0,
        "mouth_center_delta_after_crop": 0.0,
        "matching_score": 0.0,
        "valid_silent_candidate_count": 1,
        "second_best_matching_score": None,
        "matching_score_margin": None,
        "source_face_blur": 100.0,
        "target_face_blur": 100.0,
        "target_mouth_blur": 100.0,
        "flags": [],
        "split": split,
    }


def test_dataset_snapshot_roundtrip_keeps_embedded_png_bytes(tmp_path: Path) -> None:
    from datasets import Image, load_from_disk
    from edge_lipsync.silent_talking_dataset import build_dataset_dict

    rows = [_complete_row("train"), _complete_row("val")]
    dataset = build_dataset_dict(rows)
    path = tmp_path / "dataset"
    dataset.save_to_disk(path)

    loaded = load_from_disk(path)
    physical = loaded["train"].cast_column("source_roi", Image(decode=False))[0]["source_roi"]

    assert physical["path"] is None
    assert physical["bytes"].startswith(b"\x89PNG")
    assert np.asarray(loaded["train"][0]["source_roi"]).shape == (168, 168, 3)
```

- [ ] **Step 8: Run snapshot and decision tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_silent_talking_dataset.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
rtk git add edge_lipsync/silent_talking_dataset.py tests/test_silent_talking_dataset.py
rtk git commit -m "feat(dataset): build portable pose-paired snapshots"
```

---

### Task 7: End-To-End Persona Builder And YAML CLI

**Files:**
- Modify: `edge_lipsync/silent_talking_dataset.py`
- Create: `tools/build_silent_talking_dataset.py`
- Create: `configs/silent_talking_dataset.example.yaml`
- Modify: `tests/test_silent_talking_dataset.py`

- [ ] **Step 1: Write a failing orchestrator test with patched media stages**

Add:

```python
def test_build_silent_talking_dataset_writes_complete_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.silent_talking_dataset as builder

    data_root = tmp_path / "data"
    silent = data_root / "nora/silent/defaultvideo.mp4"
    talking = data_root / "nora/talking/talk.mp4"
    silent.parent.mkdir(parents=True)
    talking.parent.mkdir(parents=True)
    silent.write_bytes(b"silent")
    talking.write_bytes(b"talk")
    wenet = tmp_path / "wenet.onnx"
    wenet.write_bytes(b"wenet")

    def fake_build_contents(
        _config: object,
        temporary_root: Path,
    ) -> dict[str, object]:
        dataset = builder.build_dataset_dict([_complete_row("train"), _complete_row("val")])
        dataset.save_to_disk(temporary_root / "dataset")
        builder.write_frame_decisions(
            temporary_root / "reports/quality/talk_frame_decisions.parquet",
            [{"frame_idx": 1, "status": "retained", "reject_reason": None}],
        )
        return {
            "train_rows": 1,
            "val_rows": 1,
            "talking_clips": 1,
            "failed_clips": [],
            "dataset_fingerprints": {
                split: str(value._fingerprint) for split, value in dataset.items()
            },
        }

    monkeypatch.setattr(builder, "_build_snapshot_contents", fake_build_contents)

    snapshot = tmp_path / "snapshot"
    result = builder.build_silent_talking_dataset(
        builder.SilentTalkingBuildConfig(
            data_root=str(data_root),
            persona_id="nora",
            snapshot_root=str(snapshot),
            work_root=str(tmp_path / "work"),
            wenet_onnx=str(wenet),
            progress=False,
        )
    )

    assert result.snapshot_root == snapshot
    assert (snapshot / "dataset/dataset_dict.json").is_file()
    assert (snapshot / "reports/quality/talk_frame_decisions.parquet").is_file()
    assert (snapshot / "build_metadata.json").is_file()
    assert (snapshot / "build_complete.json").is_file()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk .venv/bin/pytest tests/test_silent_talking_dataset.py::test_build_silent_talking_dataset_writes_complete_snapshot -q
```

Expected: FAIL because the top-level builder is missing.

- [ ] **Step 3: Implement the top-level build**

Add:

```python
@dataclass(frozen=True)
class SilentTalkingBuildResult:
    snapshot_root: Path
    train_rows: int
    val_rows: int
    talking_clips: int
    failed_clips: tuple[str, ...]
    config_sha256: str
    hub_ref: str = ""


def build_silent_talking_dataset(
    config: SilentTalkingBuildConfig,
) -> SilentTalkingBuildResult:
```

Implement:

1. Validate `fps == 25`, `sample_rate == 16000`, thresholds, input paths, and Wenet model.
2. Create a temporary snapshot sibling such as `<snapshot_root>.building`.
3. Call a focused `_build_snapshot_contents(config, temporary_root)` helper that performs:
   - Normalize/extract/analyze silent once.
   - Assign video-level splits when multiple talking clips exist.
   - For each talking clip, normalize/extract audio and frames, extract precomputed BNF windows,
     compute frame-aligned RMS, analyze frames, build decisions, attach ROI images, write the clip
     report and decision Parquet.
   - For one talking clip, split rows from `target_frame_idx - 1` against the normalized frame
     count.
   - Continue after clip failure unless `strict=True`.
   - Build and validate `DatasetDict`.
   - Save it to `<temporary>/dataset`.
   - Validate sampled rows through `DuixHFDataset` and `validate_batch_shapes()`.
   - On a non-strict clip failure, write
     `reports/quality/<clip_id>.json` with `status="failed"`, exception type, and message before
     continuing. Do not emit partial rows for that clip.
4. Write build metadata with config hash, input identities, split mode, row counts, fingerprints,
    drop counts, and report paths.
5. Write `build_complete.json` last.
6. Publish the directory with a rollback-safe rename: move an existing final root to
   `<snapshot_root>.previous`, move the completed temporary root into place, then remove the
   previous root only after the new root is visible. Restore the previous root if the second rename
   fails.

Compute frame-aligned RMS with the existing 640-sample blocks:

```python
audio = load_wav_mono_f32(audio_path)
blocks = split_audio_blocks(audio)
block_rms = np.sqrt(np.mean(blocks * blocks, axis=1)).astype(np.float32)
audio_rms = np.zeros(frame_count, dtype=np.float32)
available = min(frame_count, len(block_rms))
audio_rms[:available] = block_rms[:available]
```

Frames beyond the exact BNF window count are still rejected as `bnf_out_of_range`; zero-filled RMS
must not make them trainable.

Do not write Hub commit information during the local build; the upload task returns that immutable
reference separately.

- [ ] **Step 4: Add nested YAML config parsing**

Add `replace` to the module's `dataclasses` import.

Add:

```python
def build_config_from_mapping(payload: dict[str, Any]) -> SilentTalkingBuildConfig:
    values = dict(payload)
    values["match"] = MatchConfig(**dict(values.pop("matching", {})))
    values["blur"] = BlurConfig(**dict(values.pop("blur", {})))
    sync_values = dict(values.pop("sync", {}))
    values["sync"] = SyncConfig(**sync_values)
    post_crop = dict(values.pop("post_crop_alignment", {}))
    if post_crop:
        values["match"] = replace(
            values["match"],
            max_stable_landmark_rmse=float(
                post_crop.get(
                    "max_stable_landmark_rmse",
                    values["match"].max_stable_landmark_rmse,
                )
            ),
            max_mouth_center_delta=float(
                post_crop.get(
                    "max_mouth_center_delta",
                    values["match"].max_mouth_center_delta,
                )
            ),
        )
    return SilentTalkingBuildConfig(**values)
```

- [ ] **Step 5: Create the build CLI**

Create `tools/build_silent_talking_dataset.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.silent_talking_dataset import (  # noqa: E402
    build_config_from_mapping,
    build_silent_talking_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train-ready silent/talking pose-paired dataset snapshot."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset config must be a YAML mapping")
    if args.strict:
        payload["strict"] = True
    if args.no_progress:
        payload["progress"] = False
    result = build_silent_talking_dataset(build_config_from_mapping(payload))
    print(f"snapshot_root={result.snapshot_root.resolve()}")
    print(f"train_rows={result.train_rows}")
    print(f"val_rows={result.val_rows}")
    print(f"talking_clips={result.talking_clips}")
    print(f"failed_clips={len(result.failed_clips)}")
    print(f"config_sha256={result.config_sha256}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add the example config**

Create `configs/silent_talking_dataset.example.yaml`:

```yaml
data_root: /absolute/path/to/data
persona_id: nora
snapshot_root: /absolute/path/to/datasets/nora_pose_pairs
work_root: /absolute/path/to/work/nora_pose_pairs
wenet_onnx: /absolute/path/to/models/wenet/wenet.onnx
landmark_model_asset_path: /absolute/path/to/models/mediapipe/face_landmarker.task
fps: 25
sample_rate: 16000
validation_fraction: 0.2
split_salt: edge-lipsync-silent-talking-v1
idle_max_ratio: 0.1
idle_sample_weight: 0.25
preview_count_per_group: 8
progress: true
strict: false
matching:
  max_yaw_delta: 5.0
  max_pitch_delta: 5.0
  max_roll_delta: 4.0
  max_center_x_delta: 0.05
  max_center_y_delta: 0.05
  min_width_ratio: 0.9
  max_width_ratio: 1.1
  min_height_ratio: 0.9
  max_height_ratio: 1.1
  pose_weight: 1.0
  position_weight: 1.0
  scale_weight: 1.0
post_crop_alignment:
  max_stable_landmark_rmse: 0.04
  max_mouth_center_delta: 0.04
blur:
  min_source_face_laplacian_variance: 60.0
  min_target_face_laplacian_variance: 60.0
  min_target_mouth_laplacian_variance: 40.0
sync:
  window_seconds: 2.0
  stride_seconds: 1.0
  max_lag_frames: 3
  max_reject_lag_frames: 2
  min_correlation: 0.2
  silence_rms_threshold: 0.001
  speech_fraction_threshold: 0.25
```

- [ ] **Step 7: Test CLI help and config parsing**

Add tests that run:

```python
result = subprocess.run(
    [sys.executable, "tools/build_silent_talking_dataset.py", "--help"],
    check=True,
    capture_output=True,
    text=True,
)
assert "pose-paired" in result.stdout
```

Also assert nested YAML fields reach `MatchConfig`, `BlurConfig`, and `SyncConfig`.

- [ ] **Step 8: Run builder and CLI tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_silent_talking_dataset.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
rtk git add edge_lipsync/silent_talking_dataset.py tools/build_silent_talking_dataset.py configs/silent_talking_dataset.example.yaml tests/test_silent_talking_dataset.py
rtk git commit -m "feat(dataset): build persona pose-paired datasets"
```

---

### Task 8: Immutable Hub Snapshot Push And Pull

**Files:**
- Modify: `edge_lipsync/hub.py`
- Modify: `tools/hf_dataset.py`
- Modify: `tests/test_hub.py`

- [ ] **Step 1: Write failing snapshot push and pull tests**

Add to `tests/test_hub.py`:

```python
def test_push_dataset_snapshot_uploads_complete_package(tmp_path: Path) -> None:
    from edge_lipsync.hub import push_dataset_snapshot

    snapshot = tmp_path / "snapshot"
    (snapshot / "dataset").mkdir(parents=True)
    (snapshot / "build_complete.json").write_text("{}", encoding="utf-8")
    api = _FakeApi()

    artifact = push_dataset_snapshot(snapshot, "owner/nora-pairs", api=api)

    assert artifact.resolved_ref == "commit-oid"
    assert api.upload_calls[-1]["repo_type"] == "dataset"


def test_pull_dataset_snapshot_writes_verified_local_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.hub as hub

    downloaded = tmp_path / "downloaded"
    (downloaded / "dataset/train").mkdir(parents=True)
    (downloaded / "dataset/val").mkdir(parents=True)
    (downloaded / "build_complete.json").write_text(
        json.dumps({"dataset_fingerprints": {"train": "a", "val": "b"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(hub, "snapshot_download", lambda **_kwargs: str(downloaded))

    artifact = hub.pull_dataset_snapshot(
        "owner/nora-pairs",
        ref="full-sha",
        local_dir=str(downloaded),
        api=_FakeApi(dataset_sha="full-sha"),
        verify=lambda _path: {"train": "a", "val": "b"},
    )

    marker = json.loads((downloaded / ".snapshot_complete.json").read_text())
    assert artifact.path == downloaded
    assert marker["repo_id"] == "owner/nora-pairs"
    assert marker["resolved_ref"] == "full-sha"
```

Extend `_FakeApi` with `dataset_info()`, dataset `create_repo()`, and upload call capture.

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
rtk .venv/bin/pytest tests/test_hub.py -q
```

Expected: FAIL because dataset snapshot functions do not exist.

- [ ] **Step 3: Implement snapshot push**

Add to `edge_lipsync/hub.py`:

```python
def push_dataset_snapshot(
    snapshot_root: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: str = "Upload pose-paired dataset snapshot",
    api: Any | None = None,
) -> HubArtifact:
    root = Path(snapshot_root)
    if not (root / "build_complete.json").is_file():
        raise FileNotFoundError(root / "build_complete.json")
    if not (root / "dataset/dataset_dict.json").is_file():
        raise FileNotFoundError(root / "dataset/dataset_dict.json")
    client = _client(api)
    client.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = client.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    ref = str(commit.oid)
    return HubArtifact(
        repo_id=repo_id,
        requested_ref=ref,
        resolved_ref=ref,
        url=_repo_url(repo_id, repo_type="dataset", ref=ref),
    )
```

- [ ] **Step 4: Implement resumable pull and local marker**

Add `json` and `Callable` imports to `edge_lipsync/hub.py`.

Add:

```python
def pull_dataset_snapshot(
    repo_id: str,
    *,
    ref: str,
    local_dir: str,
    cache_dir: str = "",
    api: Any | None = None,
    verify: Callable[[Path], dict[str, str]],
) -> HubArtifact:
    if not ref:
        raise ValueError("Dataset snapshot revision is required")
    root = Path(local_dir)
    marker_path = root / ".snapshot_complete.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        fingerprints = verify(root)
        if (
            marker.get("repo_id") == repo_id
            and marker.get("resolved_ref") == ref
            and marker.get("dataset_fingerprints") == fingerprints
        ):
            return HubArtifact(repo_id, ref, ref, path=root, url=_repo_url(repo_id, repo_type="dataset", ref=ref))
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": ref,
        "local_dir": str(root),
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    downloaded = Path(snapshot_download(**kwargs))
    info = _client(api).dataset_info(repo_id=repo_id, revision=ref)
    resolved = str(info.sha)
    if resolved != ref:
        raise ValueError(f"Resolved dataset revision {resolved} does not match requested {ref}")
    fingerprints = verify(downloaded)
    marker = {
        "repo_id": repo_id,
        "requested_ref": ref,
        "resolved_ref": resolved,
        "dataset_fingerprints": fingerprints,
    }
    temporary = marker_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    temporary.replace(marker_path)
    return HubArtifact(
        repo_id=repo_id,
        requested_ref=ref,
        resolved_ref=resolved,
        path=downloaded,
        url=_repo_url(repo_id, repo_type="dataset", ref=resolved),
    )
```

- [ ] **Step 5: Add snapshot commands to `tools/hf_dataset.py`**

Add `push-snapshot` and `pull-snapshot` subcommands without removing legacy `push` and `pull`.
`pull-snapshot` requires `--revision` and `--local-dir`. It verifies by calling
`datasets.load_from_disk(local_dir / "dataset")` and returning split fingerprints.

- [ ] **Step 6: Run Hub tests and CLI help**

Run:

```bash
rtk .venv/bin/pytest tests/test_hub.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add edge_lipsync/hub.py tools/hf_dataset.py tests/test_hub.py
rtk git commit -m "feat(hub): transport immutable dataset snapshots"
```

---

### Task 9: Train From A Revision-Pinned Local Snapshot

**Files:**
- Modify: `edge_lipsync/training.py`
- Modify: `configs/train.example.yaml`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write failing pinned-snapshot training source tests**

Replace `test_prepare_training_datasets_loads_hf_dataset_without_revision` in
`tests/test_training.py` with the pinned-snapshot test below, then add the validation test:

```python
def test_prepare_training_datasets_pulls_revision_then_loads_from_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.training as training

    snapshot = tmp_path / "snapshot"
    (snapshot / "dataset").mkdir(parents=True)
    loaded = {"train": [1], "val": [2]}
    calls: list[object] = []

    class TinyHFDataset:
        def __init__(self, dataset: object, split: str) -> None:
            self.dataset = dataset
            self.split = split

        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(
        training,
        "pull_dataset_snapshot",
        lambda *args, **kwargs: training.HubArtifact(
            repo_id="owner/nora-pairs",
            requested_ref="sha",
            resolved_ref="sha",
            path=snapshot,
        ),
    )
    monkeypatch.setattr(
        training,
        "load_from_disk",
        lambda path: calls.append(path) or loaded,
    )
    monkeypatch.setattr(training, "DuixHFDataset", TinyHFDataset)

    prepared = training.prepare_training_datasets(
        training.TrainConfig(
            run_dir=str(tmp_path / "run"),
            init_bin="/tmp/dh_model.bin",
            hf_dataset_repo="owner/nora-pairs",
            hf_dataset_revision="sha",
            hf_dataset_local_dir=str(snapshot),
        )
    )

    assert calls == [snapshot / "dataset"]
    assert prepared.provenance["resolved_ref"] == "sha"


def test_hf_snapshot_training_requires_revision_and_local_dir() -> None:
    import edge_lipsync.training as training

    with pytest.raises(ValueError, match="revision"):
        training.prepare_training_datasets(
            training.TrainConfig(
                run_dir="run",
                init_bin="/tmp/dh_model.bin",
                hf_dataset_repo="owner/nora-pairs",
            )
        )
```

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
rtk .venv/bin/pytest tests/test_training.py -q
```

Expected: FAIL because the config fields and snapshot path are missing.

- [ ] **Step 3: Extend `TrainConfig` and snapshot verification**

Add fields:

```python
hf_dataset_revision: str = ""
hf_dataset_local_dir: str = ""
```

Import `load_from_disk`, `HubArtifact`, and `pull_dataset_snapshot`.

Add:

```python
def _verify_dataset_snapshot(root: Path) -> dict[str, str]:
    complete = root / "build_complete.json"
    if not complete.is_file():
        raise FileNotFoundError(complete)
    metadata = json.loads(complete.read_text(encoding="utf-8"))
    dataset_path = root / "dataset"
    dataset = load_from_disk(dataset_path)
    if set(dataset) != {"train", "val"}:
        raise ValueError("Dataset snapshot must contain train and val splits")
    if len(dataset["train"]) == 0 or len(dataset["val"]) == 0:
        raise ValueError("Dataset snapshot splits must be non-empty")
    fingerprints = _dataset_fingerprints(dataset)
    if metadata.get("dataset_fingerprints") != fingerprints:
        raise ValueError("Dataset fingerprints do not match build_complete.json")
    return fingerprints
```

- [ ] **Step 4: Replace the Hub dataset branch with pinned snapshot loading**

In `prepare_training_datasets()`:

```python
if config.hf_dataset_repo:
    if not config.hf_dataset_revision:
        raise ValueError("hf_dataset_revision is required with hf_dataset_repo")
    if not config.hf_dataset_local_dir:
        raise ValueError("hf_dataset_local_dir is required with hf_dataset_repo")
    artifact = pull_dataset_snapshot(
        config.hf_dataset_repo,
        ref=config.hf_dataset_revision,
        local_dir=config.hf_dataset_local_dir,
        cache_dir=config.hf_cache_dir,
        verify=_verify_dataset_snapshot,
    )
    if artifact.path is None:
        raise ValueError("Dataset snapshot download returned no local path")
    dataset = load_from_disk(artifact.path / "dataset")
    provenance = {
        "source": "huggingface_snapshot",
        "repo_id": artifact.repo_id,
        "requested_ref": artifact.requested_ref,
        "resolved_ref": artifact.resolved_ref,
        "path": str(artifact.path),
        "fingerprints": _dataset_fingerprints(dataset),
    }
```

Keep local legacy `dataset_root` behavior unchanged. Continue requiring exactly one of
`dataset_root` or `hf_dataset_repo`.

- [ ] **Step 5: Update training example config**

Set:

```yaml
hf_dataset_repo: ""
hf_dataset_revision: ""
hf_dataset_local_dir: ""
hf_cache_dir: ""
```

Add comments in README rather than YAML comments to keep config loading simple.

- [ ] **Step 6: Run training and loader tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_training.py tests/test_dataset.py tests/test_hub.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add edge_lipsync/training.py configs/train.example.yaml tests/test_training.py
rtk git commit -m "feat(training): load pinned dataset snapshots locally"
```

---

### Task 10: Documentation, Integration Coverage, And Full Verification

**Files:**
- Modify: `README.md`
- Create: `tests/test_silent_talking_integration.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the integration marker**

Extend the existing `[tool.pytest.ini_options]` section in `pyproject.toml` to:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
  "integration: requires local model assets and real video fixtures",
]
```

- [ ] **Step 2: Add a one-step synthetic training integration test**

Create `tests/test_silent_talking_integration.py` with:

```python
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from datasets import Array2D, Dataset, DatasetDict, Features, Image, Sequence, Value
from torch.utils.data import DataLoader


def _encoded_roi(color: tuple[int, int, int]) -> dict[str, object]:
    image = np.full((168, 168, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return {"bytes": encoded.tobytes(), "path": None}


def _integration_dataset() -> DatasetDict:
    features = Features(
        {
            "schema_version": Value("string"),
            "persona_id": Value("string"),
            "pair_id": Value("string"),
            "talking_clip_id": Value("string"),
            "source_frame_idx": Value("int32"),
            "target_frame_idx": Value("int32"),
            "audio_idx": Value("int32"),
            "source_roi": Image(),
            "target_roi": Image(),
            "audio": Array2D(shape=(20, 256), dtype="float32"),
            "source_bbox_xyxy": Sequence(Value("int32"), length=4),
            "target_bbox_xyxy": Sequence(Value("int32"), length=4),
            "sample_weight": Value("float32"),
            "flags": Sequence(Value("string")),
        }
    )

    def row(pair_id: str) -> dict[str, object]:
        return {
            "schema_version": "edge_lipsync_silent_talking_pair_v1",
            "persona_id": "nora",
            "pair_id": pair_id,
            "talking_clip_id": pair_id,
            "source_frame_idx": 1,
            "target_frame_idx": 1,
            "audio_idx": 0,
            "source_roi": _encoded_roi((10, 20, 30)),
            "target_roi": _encoded_roi((200, 210, 220)),
            "audio": np.zeros((20, 256), dtype=np.float32),
            "source_bbox_xyxy": [10, 20, 110, 120],
            "target_bbox_xyxy": [12, 22, 112, 122],
            "sample_weight": 1.0,
            "flags": [],
        }

    return DatasetDict(
        {
            "train": Dataset.from_list([row("train")], features=features),
            "val": Dataset.from_list([row("val")], features=features),
        }
    )


def test_local_snapshot_runs_one_training_step_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.hub as hub
    from edge_lipsync.training import (
        TrainConfig,
        collate_training_batch,
        prepare_training_datasets,
        run_train_step,
        validate_batch_shapes,
    )

    snapshot = tmp_path / "snapshot"
    dataset = _integration_dataset()
    dataset.save_to_disk(snapshot / "dataset")
    fingerprints = {
        split: str(split_dataset._fingerprint)
        for split, split_dataset in dataset.items()
    }
    (snapshot / "build_complete.json").write_text(
        json.dumps({"dataset_fingerprints": fingerprints}),
        encoding="utf-8",
    )
    (snapshot / ".snapshot_complete.json").write_text(
        json.dumps(
            {
                "repo_id": "owner/nora-pairs",
                "requested_ref": "sha",
                "resolved_ref": "sha",
                "dataset_fingerprints": fingerprints,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("network access")),
    )
    prepared = prepare_training_datasets(
        TrainConfig(
            run_dir=str(tmp_path / "run"),
            init_bin="/tmp/unused.bin",
            hf_dataset_repo="owner/nora-pairs",
            hf_dataset_revision="sha",
            hf_dataset_local_dir=str(snapshot),
        )
    )
    loader = DataLoader(
        prepared.train_dataset,
        batch_size=1,
        collate_fn=collate_training_batch,
    )
    batch = next(iter(loader))
    validate_batch_shapes(batch)

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output = torch.nn.Conv2d(6, 3, kernel_size=1)

        def forward(self, face: torch.Tensor, _audio: torch.Tensor) -> torch.Tensor:
            return self.output(face)

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = run_train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_fn=lambda pred, target: torch.mean(torch.abs(pred - target)),
    )

    assert np.isfinite(loss)
```

This proves local snapshot loading and the unchanged tensor contract without requiring Wenet or
MediaPipe assets.

- [ ] **Step 3: Add the real Nora integration test**

In the same file:

```python
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("EDGE_LIPSYNC_WENET_ONNX")
    or not os.environ.get("EDGE_LIPSYNC_FACE_LANDMARKER_TASK")
    or not (Path(__file__).resolve().parents[1] / "data/nora/silent/defaultvideo.mp4").is_file(),
    reason="real Nora integration requires local videos and model assets",
)
def test_nora_sample_builds_and_loads_snapshot(tmp_path: Path) -> None:
    from edge_lipsync.dataset import DuixHFDataset
    from edge_lipsync.silent_talking_dataset import (
        SilentTalkingBuildConfig,
        build_silent_talking_dataset,
    )
    from datasets import load_from_disk

    root = Path(__file__).resolve().parents[1]
    wenet = Path(os.environ["EDGE_LIPSYNC_WENET_ONNX"])
    landmarker = Path(os.environ["EDGE_LIPSYNC_FACE_LANDMARKER_TASK"])
    result = build_silent_talking_dataset(
        SilentTalkingBuildConfig(
            data_root=str(root / "data"),
            persona_id="nora",
            snapshot_root=str(tmp_path / "snapshot"),
            work_root=str(tmp_path / "work"),
            wenet_onnx=str(wenet),
            landmark_model_asset_path=str(landmarker),
            progress=False,
            strict=True,
        )
    )
    dataset = load_from_disk(result.snapshot_root / "dataset")
    train = DuixHFDataset(dataset, "train")
    val = DuixHFDataset(dataset, "val")

    assert len(train) > 0
    assert len(val) > 0
    assert (result.snapshot_root / "reports/quality").is_dir()
```

Apply the skip only to this real-video test. Do not skip the module or the synthetic test.

- [ ] **Step 4: Document build, transport, and train commands**

Add a README section with:

```bash
.venv/bin/python tools/build_silent_talking_dataset.py \
  --config configs/silent_talking_dataset.example.yaml

.venv/bin/python tools/hf_dataset.py push-snapshot \
  --snapshot-root /absolute/path/to/datasets/nora_pose_pairs \
  --repo-id username/nora-pose-pairs

.venv/bin/python tools/hf_dataset.py pull-snapshot \
  --repo-id username/nora-pose-pairs \
  --revision <full-commit-sha> \
  --local-dir /persistent/datasets/nora/<full-commit-sha>

.venv/bin/python tools/train.py --config configs/train.example.yaml
```

Document:

- Hub is transport only.
- Training uses `load_from_disk()`.
- `hf_dataset_revision` must be the full commit SHA.
- The local snapshot marker avoids repeat network access.
- `sample_weight` is metadata only in V1.
- How to inspect frame-decision Parquet and preview groups.

- [ ] **Step 5: Run focused integration and CLI tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_silent_talking_integration.py tests/test_silent_talking_dataset.py tests/test_hub.py tests/test_training.py -q
```

Expected: synthetic tests PASS; real Nora test is SKIPPED unless model asset environment variables
are set.

- [ ] **Step 6: Run the real Nora integration when assets are available**

Run:

```bash
EDGE_LIPSYNC_WENET_ONNX="$PWD/models/wenet/wenet.onnx" \
EDGE_LIPSYNC_FACE_LANDMARKER_TASK="$PWD/models/mediapipe/face_landmarker.task" \
rtk .venv/bin/pytest tests/test_silent_talking_integration.py -m integration -q
```

Expected: PASS with non-empty train and validation splits. If assets are not installed, report this
verification as not run; do not claim the real-video build passed.

- [ ] **Step 7: Run full repository verification**

Run:

```bash
rtk .venv/bin/ruff check .
rtk .venv/bin/pyright
rtk .venv/bin/pytest -q
```

Expected: all commands exit 0.

- [ ] **Step 8: Inspect generated quality artifacts**

For the real Nora build, verify:

```bash
rtk find /absolute/path/to/snapshot/reports -maxdepth 4 -type f
```

Confirm:

- One decision row per normalized talking frame.
- No retained row violates recorded hard gates.
- Preview groups include best, near-threshold, low-confidence, idle when available, and rejection
  examples.
- `source_roi` and `target_roi` decode without access to original videos.

- [ ] **Step 9: Commit**

```bash
rtk git add README.md pyproject.toml tests/test_silent_talking_integration.py
rtk git commit -m "docs(dataset): document pose-paired dataset workflow"
```

---

## Final Verification Checklist

Before declaring implementation complete:

- [ ] `rtk .venv/bin/ruff check .` exits 0.
- [ ] `rtk .venv/bin/pyright` exits 0.
- [ ] `rtk .venv/bin/pytest -q` exits 0.
- [ ] Real Nora integration passes when model assets are available.
- [ ] Physical `Image(decode=False)` cells contain PNG bytes and no build-machine path.
- [ ] `frame_decisions.parquet` has exactly one row per normalized talking frame.
- [ ] `valid_silent_candidate_count` includes post-crop alignment.
- [ ] Second-best fields are null for one candidate and finite otherwise.
- [ ] Low-correlation speech windows are not rejected only because of lag.
- [ ] BNF arrays equal `bnf_windows[target_frame_idx - 1]`.
- [ ] Training loads `<snapshot>/dataset` locally and does not access Hub after marker validation.
