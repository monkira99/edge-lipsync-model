from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

HfApi: Any
snapshot_download: Any
hf_hub_download: Any
try:
    from huggingface_hub import HfApi as _HfApi
    from huggingface_hub import hf_hub_download as _download_file
    from huggingface_hub import snapshot_download as _download_snapshot
except ImportError:
    HfApi = None

    def _missing_download(**_kwargs: Any) -> str:
        raise ImportError("Install huggingface-hub to use Hugging Face Hub integration")

    snapshot_download = _missing_download
    hf_hub_download = _missing_download
else:
    HfApi = _HfApi
    snapshot_download = _download_snapshot
    hf_hub_download = _download_file


DATASET_UPLOAD_PATTERNS = (
    "manifest.jsonl",
    "splits.json",
    "build_summary.json",
    "clips/*/frames/*.png",
    "clips/*/bnf.npy",
    "clips/*/bboxes.json",
    "clips/*/quality.json",
    "clips/*/previews/*.jpg",
)
MODEL_UPLOAD_PATTERNS = (
    "best.pt",
    "final.pt",
    "metrics.json",
    "metrics.csv",
    "run_metadata.json",
    "README.md",
)
MODEL_ASSET_UPLOAD_PATTERNS = (
    "duix_detector/*",
    "emma/*",
    "mediapipe/*",
    "wenet/*",
)
DATASET_REQUIRED_PATHS = ("manifest.jsonl", "splits.json", "build_summary.json", "clips")
MODEL_REQUIRED_PATHS = ("best.pt", "final.pt", "metrics.json", "metrics.csv", "run_metadata.json")
MODEL_ASSET_REQUIRED_PATHS = (
    "duix_detector/pfpld_robust_sim_bs1_8003.onnx",
    "duix_detector/scrfd_500m_kps-opt2.bin",
    "duix_detector/scrfd_500m_kps-opt2.param",
    "emma/dh_model.bin",
    "emma/dh_model.param",
    "emma/weight_168u.bin",
    "mediapipe/face_landmarker.task",
    "wenet/wenet.onnx",
)

@dataclass(frozen=True)
class HubArtifact:
    repo_id: str
    requested_revision: str
    resolved_revision: str
    path: Path | None = None
    url: str = ""


def _client(api: Any | None) -> Any:
    if api is not None:
        return api
    if HfApi is None:
        raise ImportError("Install huggingface-hub to use Hugging Face Hub integration")
    return HfApi()


def _require_revision(revision: str) -> None:
    if not revision:
        raise ValueError("Hugging Face revision must be pinned and non-empty")


def _validate_required_paths(root: Path, required: tuple[str, ...]) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    for relative in required:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")


def _repo_url(repo_id: str, *, repo_type: str, revision: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}/tree/{revision}"


def push_dataset_snapshot(
    dataset_root: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: str = "Upload processed dataset snapshot",
    api: Any | None = None,
) -> HubArtifact:
    root = Path(dataset_root)
    _validate_required_paths(root, DATASET_REQUIRED_PATHS)
    client = _client(api)
    client.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    client.upload_large_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=DATASET_UPLOAD_PATTERNS,
    )
    info = client.dataset_info(repo_id=repo_id)
    revision = str(info.sha)
    return HubArtifact(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=revision,
        url=_repo_url(repo_id, repo_type="dataset", revision=revision),
    )


def pull_dataset_snapshot(
    repo_id: str,
    *,
    revision: str,
    cache_dir: str = "",
    api: Any | None = None,
) -> HubArtifact:
    _require_revision(revision)
    kwargs = {"repo_id": repo_id, "repo_type": "dataset", "revision": revision}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    path = Path(snapshot_download(**kwargs))
    info = _client(api).dataset_info(repo_id=repo_id, revision=revision)
    resolved = str(info.sha)
    return HubArtifact(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=resolved,
        path=path,
        url=_repo_url(repo_id, repo_type="dataset", revision=resolved),
    )


def push_model_artifacts(
    run_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: str = "Upload training run artifacts",
    api: Any | None = None,
) -> HubArtifact:
    root = Path(run_dir)
    _validate_required_paths(root, MODEL_REQUIRED_PATHS)
    client = _client(api)
    client.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    commit = client.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        allow_patterns=MODEL_UPLOAD_PATTERNS,
        commit_message=commit_message,
    )
    revision = str(commit.oid)
    return HubArtifact(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=revision,
        url=_repo_url(repo_id, repo_type="model", revision=revision),
    )


def pull_model_checkpoint(
    repo_id: str,
    *,
    revision: str,
    filename: str = "best.pt",
    cache_dir: str = "",
    api: Any | None = None,
) -> HubArtifact:
    _require_revision(revision)
    kwargs = {"repo_id": repo_id, "filename": filename, "revision": revision}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    path = Path(hf_hub_download(**kwargs))
    info = _client(api).model_info(repo_id=repo_id, revision=revision)
    resolved = str(info.sha)
    return HubArtifact(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=resolved,
        path=path,
        url=_repo_url(repo_id, repo_type="model", revision=resolved),
    )


def push_model_assets(
    models_root: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: str = "Upload model assets",
    api: Any | None = None,
) -> HubArtifact:
    root = Path(models_root)
    _validate_required_paths(root, MODEL_ASSET_REQUIRED_PATHS)
    client = _client(api)
    client.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    commit = client.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        allow_patterns=MODEL_ASSET_UPLOAD_PATTERNS,
        commit_message=commit_message,
    )
    revision = str(commit.oid)
    return HubArtifact(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=revision,
        url=_repo_url(repo_id, repo_type="model", revision=revision),
    )


def pull_model_assets(
    repo_id: str,
    *,
    revision: str,
    local_dir: str = "models",
    cache_dir: str = "",
    api: Any | None = None,
) -> HubArtifact:
    _require_revision(revision)
    kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "allow_patterns": MODEL_ASSET_UPLOAD_PATTERNS,
    }
    if local_dir:
        kwargs["local_dir"] = local_dir
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    path = Path(snapshot_download(**kwargs))
    info = _client(api).model_info(repo_id=repo_id, revision=revision)
    resolved = str(info.sha)
    return HubArtifact(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=resolved,
        path=path,
        url=_repo_url(repo_id, repo_type="model", revision=resolved),
    )
