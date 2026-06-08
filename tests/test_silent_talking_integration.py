from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
import torch
from datasets import Array2D, Dataset, DatasetDict, Features, Image, Sequence, Value, load_from_disk
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
    loaded = cast(DatasetDict, load_from_disk(snapshot / "dataset"))
    fingerprints = {
        split: str(split_dataset._fingerprint)
        for split, split_dataset in loaded.items()
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
    dataset = cast(DatasetDict, load_from_disk(result.snapshot_root / "dataset"))
    train = DuixHFDataset(dataset, "train")
    val = DuixHFDataset(dataset, "val")

    assert len(train) > 0
    assert len(val) > 0
    assert (result.snapshot_root / "reports/quality").is_dir()
