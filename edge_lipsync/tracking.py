from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class WandbConfig:
    mode: str = "disabled"
    project: str = "edge-lipsync-model"
    entity: str = ""
    run_name: str = ""
    group: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""
    directory: str = ""


class Tracker(Protocol):
    @property
    def provenance(self) -> dict[str, str]: ...

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None: ...

    def log_video(self, name: str, path: str, *, step: int, caption: str = "") -> None: ...

    def update_summary(self, values: dict[str, Any]) -> None: ...

    def finish(self, *, exit_code: int = 0) -> None: ...


class DisabledTracker:
    @property
    def provenance(self) -> dict[str, str]:
        return {"mode": "disabled"}

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        pass

    def log_video(self, name: str, path: str, *, step: int, caption: str = "") -> None:
        pass

    def update_summary(self, values: dict[str, Any]) -> None:
        pass

    def finish(self, *, exit_code: int = 0) -> None:
        pass


class WandbTracker:
    def __init__(
        self,
        config: WandbConfig,
        *,
        run_config: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise ImportError("Install wandb to use W&B experiment tracking") from exc
        kwargs: dict[str, Any] = {
            "project": config.project,
            "mode": config.mode,
            "config": {
                "training": run_config,
                "provenance": provenance,
            },
        }
        optional = {
            "entity": config.entity,
            "name": config.run_name,
            "group": config.group,
            "notes": config.notes,
            "dir": config.directory,
        }
        kwargs.update({key: value for key, value in optional.items() if value})
        if config.tags:
            kwargs["tags"] = list(config.tags)
        self._wandb = wandb
        self._run = wandb.init(**kwargs)
        if self._run is None:
            raise RuntimeError("wandb.init() returned no run")
        self._provenance = {
            "mode": config.mode,
            "run_id": str(self._run.id),
            "run_url": str(self._run.url),
        }

    @property
    def provenance(self) -> dict[str, str]:
        return dict(self._provenance)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        self._run.log(dict(metrics), step=step)

    def log_video(self, name: str, path: str, *, step: int, caption: str = "") -> None:
        video = self._wandb.Video(str(path), format="mp4", caption=caption)
        self._run.log({name: video}, step=step)

    def update_summary(self, values: dict[str, Any]) -> None:
        self._run.summary.update(values)

    def finish(self, *, exit_code: int = 0) -> None:
        self._run.finish(exit_code=exit_code)


def create_tracker(
    config: WandbConfig,
    *,
    run_config: dict[str, Any],
    provenance: dict[str, Any],
) -> Tracker:
    if config.mode == "disabled":
        return DisabledTracker()
    if config.mode not in {"online", "offline"}:
        raise ValueError(f"Unsupported W&B mode={config.mode!r}")
    return WandbTracker(config, run_config=run_config, provenance=provenance)
