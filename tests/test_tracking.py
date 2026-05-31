from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any

import pytest


class _FakeRun:
    def __init__(self) -> None:
        self.id = "run-id"
        self.url = "https://wandb.ai/owner/project/runs/run-id"
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.summary: dict[str, Any] = {}
        self.exit_codes: list[int] = []

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self, *, exit_code: int = 0) -> None:
        self.exit_codes.append(exit_code)


class _FakeWandb(ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.run = _FakeRun()
        self.init_calls: list[dict[str, Any]] = []

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_calls.append(kwargs)
        return self.run


def test_disabled_tracker_does_not_import_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    from edge_lipsync.tracking import WandbConfig, create_tracker

    original_import = builtins.__import__

    def reject_wandb_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb":
            raise AssertionError("disabled tracker imported wandb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_wandb_import)

    tracker = create_tracker(WandbConfig(mode="disabled"), run_config={}, provenance={})
    tracker.log_metrics({"train_loss": 0.5}, step=1)
    tracker.update_summary({"best_val": 0.25})
    tracker.finish()

    assert tracker.provenance == {"mode": "disabled"}


def test_tracker_rejects_unsupported_mode() -> None:
    from edge_lipsync.tracking import WandbConfig, create_tracker

    with pytest.raises(ValueError, match="mode"):
        create_tracker(WandbConfig(mode="invalid"), run_config={}, provenance={})


def test_wandb_tracker_initializes_logs_summary_and_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edge_lipsync.tracking import WandbConfig, create_tracker

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = WandbConfig(
        mode="offline",
        project="edge-lipsync-model",
        entity="owner",
        run_name="avatar-baseline",
        group="avatar",
        tags=("baseline", "duix"),
        notes="First tracked run",
        directory="/tmp/wandb",
    )

    tracker = create_tracker(
        config,
        run_config={"max_steps": 3},
        provenance={"dataset": {"source": "local"}},
    )
    tracker.log_metrics({"train_loss": 0.5}, step=1)
    tracker.update_summary({"best_val": 0.25})
    tracker.finish(exit_code=1)

    assert fake.init_calls == [
        {
            "project": "edge-lipsync-model",
            "mode": "offline",
            "config": {
                "training": {"max_steps": 3},
                "provenance": {"dataset": {"source": "local"}},
            },
            "entity": "owner",
            "name": "avatar-baseline",
            "group": "avatar",
            "tags": ["baseline", "duix"],
            "notes": "First tracked run",
            "dir": "/tmp/wandb",
        }
    ]
    assert tracker.provenance == {
        "mode": "offline",
        "run_id": "run-id",
        "run_url": "https://wandb.ai/owner/project/runs/run-id",
    }
    assert fake.run.logged == [({"train_loss": 0.5}, 1)]
    assert fake.run.summary == {"best_val": 0.25}
    assert fake.run.exit_codes == [1]
