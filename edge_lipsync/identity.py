from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from edge_lipsync.hub import hf_hub_download
from edge_lipsync.onnx_runtime import (
    OnnxProviderSelection,
    OnnxRunExecutor,
    resolve_onnx_providers,
)
from edge_lipsync.preprocess import Point

ARCFACE_INPUT_SIZE = 112
ARCFACE_EMBEDDING_SIZE = 512
ARCFACE_LICENSE = "insightface-non-commercial-research"
ARCFACE_TEMPLATE = np.asarray(
    [
        (38.2946, 51.6963),
        (73.5318, 51.5014),
        (56.0252, 71.7366),
        (41.5493, 92.3655),
        (70.7299, 92.2041),
    ],
    dtype=np.float32,
)


class IdentityRuntimeError(RuntimeError):
    pass


class IdentityFrameError(ValueError):
    pass


@dataclass(frozen=True)
class IdentityConfig:
    arcface_onnx: str = ""
    hf_repo: str = "facefusion/models-3.0.0"
    hf_filename: str = "arcface_w600k_r50.onnx"
    hf_revision: str = "728b9659bd9691bf32cbf7f61af478d94b7ba81e"
    cache_dir: str = ""
    expected_sha256: str = (
        "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1"
    )
    min_cosine_similarity: float = 0.35


@dataclass(frozen=True)
class ResolvedIdentityModel:
    path: Path
    provenance: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_identity_model(config: IdentityConfig) -> ResolvedIdentityModel:
    if config.arcface_onnx:
        path = Path(config.arcface_onnx)
        source = "local"
    else:
        kwargs = {
            "repo_id": config.hf_repo,
            "filename": config.hf_filename,
            "revision": config.hf_revision,
        }
        if config.cache_dir:
            kwargs["cache_dir"] = config.cache_dir
        try:
            path = Path(hf_hub_download(**kwargs))
        except Exception as exc:
            raise IdentityRuntimeError(f"Failed to download ArcFace model: {exc}") from exc
        source = "huggingface"
    if not path.is_file():
        raise IdentityRuntimeError(f"ArcFace model not found: {path}")
    digest = _file_sha256(path)
    if digest.lower() != config.expected_sha256.lower():
        raise IdentityRuntimeError(
            "ArcFace model SHA-256 mismatch: "
            f"expected={config.expected_sha256} actual={digest} path={path}"
        )
    return ResolvedIdentityModel(
        path=path,
        provenance={
            "source": source,
            "resolved_path": str(path.resolve()),
            "hf_repo": config.hf_repo,
            "hf_filename": config.hf_filename,
            "hf_revision": config.hf_revision,
            "sha256": digest,
            "min_cosine_similarity": config.min_cosine_similarity,
            "license": ARCFACE_LICENSE,
        },
    )


def _mean_point(landmarks: Mapping[int, Point], indices: tuple[int, int]) -> np.ndarray:
    return np.mean(
        np.asarray([landmarks[index] for index in indices], dtype=np.float32),
        axis=0,
    )


def arcface_five_points(landmarks: Mapping[int, Point]) -> np.ndarray:
    try:
        eyes = sorted(
            (
                _mean_point(landmarks, (33, 133)),
                _mean_point(landmarks, (362, 263)),
            ),
            key=lambda point: float(point[0]),
        )
        mouth = sorted(
            (
                np.asarray(landmarks[61], dtype=np.float32),
                np.asarray(landmarks[291], dtype=np.float32),
            ),
            key=lambda point: float(point[0]),
        )
        points = np.stack(
            [
                eyes[0],
                eyes[1],
                np.asarray(landmarks[1], dtype=np.float32),
                mouth[0],
                mouth[1],
            ]
        )
    except (KeyError, ValueError) as exc:
        raise IdentityFrameError("identity_alignment_failed") from exc
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise IdentityFrameError("identity_alignment_failed")
    return points


def align_arcface_face(
    frame_bgr: np.ndarray,
    landmarks: Mapping[int, Point],
) -> np.ndarray:
    source = arcface_five_points(landmarks)
    transform, _inliers = cv2.estimateAffinePartial2D(
        source,
        ARCFACE_TEMPLATE,
        method=cv2.LMEDS,
    )
    if transform is None or transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise IdentityFrameError("identity_alignment_failed")
    aligned = cv2.warpAffine(
        frame_bgr,
        transform,
        (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if aligned.shape != (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE, 3):
        raise IdentityFrameError("identity_alignment_failed")
    return aligned


def preprocess_arcface(aligned_bgr: np.ndarray) -> np.ndarray:
    expected = (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE, 3)
    if aligned_bgr.shape != expected:
        raise IdentityFrameError(
            f"identity_alignment_failed: expected aligned face {expected}, got {aligned_bgr.shape}"
        )
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) - 127.5) / 127.5
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, :, :, :])


def _valid_batch_shape(shape: Any, trailing: tuple[int, ...]) -> bool:
    if not isinstance(shape, (list, tuple)) or len(shape) != len(trailing) + 1:
        return False
    for actual, expected in zip(shape[1:], trailing, strict=True):
        if actual != expected:
            return False
    return shape[0] in (None, 1, "None", "batch_size", "N")


class ArcFaceRuntime:
    def __init__(
        self,
        model_path: str | Path,
        *,
        provider_selection: OnnxProviderSelection | None = None,
        run_limiter: OnnxRunExecutor | None = None,
        session: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise IdentityRuntimeError(f"ArcFace model not found: {self.model_path}")
        selection = provider_selection or resolve_onnx_providers(
            "cpu",
            warn_on_fallback=False,
        )
        if session is None:
            try:
                session = ort.InferenceSession(
                    str(self.model_path),
                    providers=list(selection.selected_providers),
                )
            except Exception as exc:
                raise IdentityRuntimeError(f"Failed to load ArcFace ONNX model: {exc}") from exc
        assert session is not None
        self.session = session
        self.provider_selection = selection
        self.run_limiter = run_limiter
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        if len(inputs) != 1 or not _valid_batch_shape(inputs[0].shape, (3, 112, 112)):
            shape = inputs[0].shape if inputs else None
            raise IdentityRuntimeError(f"Unsupported ArcFace input shape: {shape}")
        if len(outputs) != 1 or not _valid_batch_shape(outputs[0].shape, (512,)):
            shape = outputs[0].shape if outputs else None
            raise IdentityRuntimeError(f"Unsupported ArcFace output shape: {shape}")
        self.input_name = str(inputs[0].name)
        self.output_name = str(outputs[0].name)

    def embed(
        self,
        frame_bgr: np.ndarray,
        landmarks: Mapping[int, Point],
    ) -> np.ndarray:
        aligned = align_arcface_face(frame_bgr, landmarks)
        tensor = preprocess_arcface(aligned)
        try:
            inputs = {self.input_name: tensor}
            outputs = (
                self.run_limiter.run(self.session, [self.output_name], inputs)
                if self.run_limiter is not None
                else self.session.run([self.output_name], inputs)
            )
        except Exception as exc:
            raise IdentityRuntimeError(f"ArcFace inference failed: {exc}") from exc
        if len(outputs) != 1:
            raise IdentityRuntimeError(f"ArcFace returned {len(outputs)} outputs")
        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if embedding.shape != (ARCFACE_EMBEDDING_SIZE,):
            raise IdentityRuntimeError(
                f"ArcFace returned embedding shape {embedding.shape}, expected "
                f"({ARCFACE_EMBEDDING_SIZE},)"
            )
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(embedding).all() or not np.isfinite(norm) or norm <= 1e-12:
            raise IdentityFrameError("identity_embedding_invalid")
        return np.ascontiguousarray(embedding / norm, dtype=np.float32)


def create_identity_runtime(
    config: IdentityConfig,
    *,
    provider_selection: OnnxProviderSelection | None = None,
    run_limiter: OnnxRunExecutor | None = None,
) -> tuple[ArcFaceRuntime, dict[str, Any]]:
    resolved = resolve_identity_model(config)
    runtime = ArcFaceRuntime(
        resolved.path,
        provider_selection=provider_selection,
        run_limiter=run_limiter,
    )
    provenance = {
        **resolved.provenance,
        "providers": list(runtime.provider_selection.selected_providers),
    }
    return runtime, provenance


def cosine_identity_similarity(source: np.ndarray, target: np.ndarray) -> float:
    source_embedding = np.asarray(source, dtype=np.float32).reshape(-1)
    target_embedding = np.asarray(target, dtype=np.float32).reshape(-1)
    if source_embedding.shape != (ARCFACE_EMBEDDING_SIZE,) or target_embedding.shape != (
        ARCFACE_EMBEDDING_SIZE,
    ):
        raise ValueError("Identity embeddings must have shape (512,)")
    similarity = float(np.dot(source_embedding, target_embedding))
    if not np.isfinite(similarity):
        raise ValueError("Identity similarity is not finite")
    return similarity
