from __future__ import annotations

import warnings
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class OnnxSession(Protocol):
    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class OnnxProviderSelection:
    requested_device: str
    available_providers: tuple[str, ...]
    selected_providers: tuple[str, ...]
    cpu_fallback: bool
    fallback_reason: str

    @property
    def uses_cuda(self) -> bool:
        return CUDA_PROVIDER in self.selected_providers


def resolve_onnx_providers(
    requested_device: str,
    *,
    available_providers: list[str] | tuple[str, ...] | None = None,
    warn_on_fallback: bool = True,
) -> OnnxProviderSelection:
    requested = requested_device.lower().strip()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("ONNX device must be auto, cuda, or cpu")
    available = tuple(
        str(provider)
        for provider in (
            ort.get_available_providers()
            if available_providers is None
            else available_providers
        )
    )
    if requested == "cpu":
        return OnnxProviderSelection(
            requested_device=requested,
            available_providers=available,
            selected_providers=(CPU_PROVIDER,),
            cpu_fallback=False,
            fallback_reason="",
        )
    if CUDA_PROVIDER in available:
        return OnnxProviderSelection(
            requested_device=requested,
            available_providers=available,
            selected_providers=(CUDA_PROVIDER, CPU_PROVIDER),
            cpu_fallback=False,
            fallback_reason="",
        )
    reason = (
        f"{CUDA_PROVIDER} requested through device={requested!r} but unavailable; "
        f"falling back to {CPU_PROVIDER}. Available providers: {list(available)}"
    )
    if warn_on_fallback:
        warnings.warn(reason, RuntimeWarning, stacklevel=2)
    return OnnxProviderSelection(
        requested_device=requested,
        available_providers=available,
        selected_providers=(CPU_PROVIDER,),
        cpu_fallback=True,
        fallback_reason=reason,
    )


class OnnxRunLimiter:
    def __init__(
        self,
        selection: OnnxProviderSelection,
        *,
        max_inflight: int,
    ) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")
        self.selection = selection
        self._semaphore = (
            BoundedSemaphore(max_inflight) if selection.uses_cuda else None
        )

    def run(
        self,
        session: OnnxSession,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[Any]:
        if self._semaphore is None:
            return session.run(output_names, inputs)
        with self._semaphore:
            return session.run(output_names, inputs)
