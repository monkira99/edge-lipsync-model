from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    enabled: bool = True,
    desc: str = "",
    total: int | None = None,
    unit: str = "",
) -> Iterable[T]:
    if not enabled:
        return iterable
    kwargs: dict[str, Any] = {
        "desc": desc or None,
        "leave": False,
    }
    if total is not None:
        kwargs["total"] = total
    if unit:
        kwargs["unit"] = unit
    return tqdm(iterable, **kwargs)
