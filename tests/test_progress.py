from __future__ import annotations

from typing import Any


def test_progress_wraps_iterable_with_tqdm_auto(monkeypatch: Any) -> None:
    import edge_lipsync.progress as progress_module

    calls: list[dict[str, Any]] = []

    def fake_tqdm(iterable: list[int], **kwargs: Any) -> list[int]:
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(progress_module, "tqdm", fake_tqdm)

    values = list(
        progress_module.progress(
            [1, 2],
            enabled=True,
            desc="clips",
            total=2,
            unit="clip",
        )
    )

    assert values == [1, 2]
    assert calls == [
        {
            "desc": "clips",
            "leave": False,
            "total": 2,
            "unit": "clip",
        }
    ]


def test_progress_returns_iterable_directly_when_disabled(monkeypatch: Any) -> None:
    import edge_lipsync.progress as progress_module

    def fail_tqdm(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("tqdm should not be called")

    monkeypatch.setattr(progress_module, "tqdm", fail_tqdm)

    values = list(progress_module.progress([1, 2], enabled=False, desc="clips"))

    assert values == [1, 2]
