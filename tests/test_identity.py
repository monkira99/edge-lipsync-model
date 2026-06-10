from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _arcface_landmarks() -> dict[int, tuple[float, float]]:
    return {
        1: (56.0252, 71.7366),
        33: (36.2946, 51.6963),
        133: (40.2946, 51.6963),
        362: (71.5318, 51.5014),
        263: (75.5318, 51.5014),
        61: (41.5493, 92.3655),
        291: (70.7299, 92.2041),
    }


def test_resolve_identity_model_uses_verified_local_file(tmp_path: Path) -> None:
    from edge_lipsync.identity import IdentityConfig, resolve_identity_model

    model = tmp_path / "arcface.onnx"
    content = b"local-arcface"
    model.write_bytes(content)

    resolved = resolve_identity_model(
        IdentityConfig(
            arcface_onnx=str(model),
            expected_sha256=_sha256(content),
        )
    )

    assert resolved.path == model
    assert resolved.provenance["source"] == "local"
    assert resolved.provenance["sha256"] == _sha256(content)


def test_resolve_identity_model_downloads_when_local_path_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_lipsync.identity as identity

    model = tmp_path / "downloaded.onnx"
    content = b"downloaded-arcface"
    model.write_bytes(content)
    calls: list[dict[str, str]] = []

    def fake_download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(model)

    monkeypatch.setattr(identity, "hf_hub_download", fake_download)

    resolved = identity.resolve_identity_model(
        identity.IdentityConfig(
            hf_repo="owner/models",
            hf_filename="arcface.onnx",
            hf_revision="commit-sha",
            cache_dir=str(tmp_path / "cache"),
            expected_sha256=_sha256(content),
        )
    )

    assert calls == [
        {
            "repo_id": "owner/models",
            "filename": "arcface.onnx",
            "revision": "commit-sha",
            "cache_dir": str(tmp_path / "cache"),
        }
    ]
    assert resolved.path == model
    assert resolved.provenance["source"] == "huggingface"


def test_resolve_identity_model_rejects_checksum_mismatch(tmp_path: Path) -> None:
    from edge_lipsync.identity import IdentityConfig, IdentityRuntimeError, resolve_identity_model

    model = tmp_path / "arcface.onnx"
    model.write_bytes(b"wrong")

    with pytest.raises(IdentityRuntimeError, match="SHA-256"):
        resolve_identity_model(
            IdentityConfig(
                arcface_onnx=str(model),
                expected_sha256=_sha256(b"expected"),
            )
        )


def test_align_arcface_face_uses_standard_112_template() -> None:
    from edge_lipsync.identity import align_arcface_face

    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    frame[40:100, 30:80] = (20, 80, 160)

    aligned = align_arcface_face(frame, _arcface_landmarks())

    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8
    assert np.mean(aligned) > 0


def test_preprocess_arcface_converts_bgr_to_normalized_float32_nchw() -> None:
    from edge_lipsync.identity import preprocess_arcface

    aligned = np.zeros((112, 112, 3), dtype=np.uint8)
    aligned[:, :] = (0, 127, 255)

    tensor = preprocess_arcface(aligned)

    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert tensor[0, 0, 0, 0] == pytest.approx(1.0)
    assert tensor[0, 1, 0, 0] == pytest.approx((127.0 - 127.5) / 127.5)
    assert tensor[0, 2, 0, 0] == pytest.approx(-1.0)


def test_arcface_runtime_returns_l2_normalized_embedding(tmp_path: Path) -> None:
    from edge_lipsync.identity import ArcFaceRuntime

    model = tmp_path / "arcface.onnx"
    model.write_bytes(b"onnx")
    captured: list[np.ndarray] = []

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input.1", shape=[None, 3, 112, 112])]

        def get_outputs(self):
            return [SimpleNamespace(name="683", shape=[None, 512])]

        def run(self, output_names, feed):
            assert output_names == ["683"]
            captured.append(feed["input.1"])
            output = np.zeros((1, 512), dtype=np.float32)
            output[0, :2] = (3.0, 4.0)
            return [output]

    runtime = ArcFaceRuntime(model, session=FakeSession())
    embedding = runtime.embed(np.zeros((112, 112, 3), dtype=np.uint8), _arcface_landmarks())

    assert captured[0].shape == (1, 3, 112, 112)
    assert embedding.shape == (512,)
    assert embedding[:2] == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(embedding) == pytest.approx(1.0)


def test_arcface_runtime_rejects_invalid_model_contract(tmp_path: Path) -> None:
    from edge_lipsync.identity import ArcFaceRuntime, IdentityRuntimeError

    model = tmp_path / "arcface.onnx"
    model.write_bytes(b"onnx")

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[None, 3, 224, 224])]

        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=[None, 512])]

    with pytest.raises(IdentityRuntimeError, match="input shape"):
        ArcFaceRuntime(model, session=FakeSession())


def test_arcface_runtime_wraps_onnx_execution_failure(tmp_path: Path) -> None:
    from edge_lipsync.identity import ArcFaceRuntime, IdentityRuntimeError

    model = tmp_path / "arcface.onnx"
    model.write_bytes(b"onnx")

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 112, 112])]

        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=[1, 512])]

        def run(self, _output_names, _feed):
            raise RuntimeError("backend failed")

    runtime = ArcFaceRuntime(model, session=FakeSession())

    with pytest.raises(IdentityRuntimeError, match="ArcFace inference failed"):
        runtime.embed(np.zeros((112, 112, 3), dtype=np.uint8), _arcface_landmarks())
