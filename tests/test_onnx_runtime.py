from __future__ import annotations

import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


def test_resolve_onnx_providers_prefers_cuda_before_cpu() -> None:
    from edge_lipsync.onnx_runtime import resolve_onnx_providers

    selection = resolve_onnx_providers(
        "auto",
        available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )

    assert selection.selected_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert selection.cpu_fallback is False
    assert selection.fallback_reason == ""


@pytest.mark.parametrize("requested_device", ["auto", "cuda"])
def test_resolve_onnx_providers_warns_when_cuda_is_unavailable(
    requested_device: str,
) -> None:
    from edge_lipsync.onnx_runtime import resolve_onnx_providers

    with pytest.warns(RuntimeWarning, match="falling back"):
        selection = resolve_onnx_providers(
            requested_device,
            available_providers=["CPUExecutionProvider"],
        )

    assert selection.selected_providers == ("CPUExecutionProvider",)
    assert selection.cpu_fallback is True
    assert "CUDAExecutionProvider" in selection.fallback_reason


def test_resolve_onnx_providers_explicit_cpu_does_not_warn() -> None:
    from edge_lipsync.onnx_runtime import resolve_onnx_providers

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selection = resolve_onnx_providers(
            "cpu",
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    assert selection.selected_providers == ("CPUExecutionProvider",)
    assert selection.cpu_fallback is False


def test_resolve_onnx_providers_rejects_unknown_device() -> None:
    from edge_lipsync.onnx_runtime import resolve_onnx_providers

    with pytest.raises(ValueError, match="auto, cuda, or cpu"):
        resolve_onnx_providers("metal", available_providers=["CPUExecutionProvider"])


def test_cuda_run_limiter_bounds_concurrent_session_calls() -> None:
    from edge_lipsync.onnx_runtime import OnnxRunLimiter, resolve_onnx_providers

    selection = resolve_onnx_providers(
        "cuda",
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    limiter = OnnxRunLimiter(selection, max_inflight=2)
    lock = threading.Lock()
    active = 0
    peak = 0

    class FakeSession:
        def run(
            self,
            output_names: list[str],
            inputs: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            del output_names, inputs
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return [np.zeros((1,), dtype=np.float32)]

    session = FakeSession()
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(
            executor.map(
                lambda _index: limiter.run(session, ["output"], {"input": np.zeros(1)}),
                range(6),
            )
        )

    assert len(results) == 6
    assert peak == 2
