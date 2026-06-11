from __future__ import annotations

import json
from collections.abc import Callable
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
DATASET_TRAIN_TRANSFER_PATTERNS = [
    "dataset/**",
    "build_complete.json",
    "build_metadata.json",
]

@dataclass(frozen=True)
class HubArtifact:
    repo_id: str
    requested_ref: str
    resolved_ref: str
    path: Path | None = None
    url: str = ""


def _client(api: Any | None) -> Any:
    if api is not None:
        return api
    if HfApi is None:
        raise ImportError("Install huggingface-hub to use Hugging Face Hub integration")
    return HfApi()


def _validate_required_paths(root: Path, required: tuple[str, ...]) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    for relative in required:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")


def _repo_url(repo_id: str, *, repo_type: str, ref: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}/tree/{ref}"


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
    ref = str(commit.oid)
    return HubArtifact(
        repo_id=repo_id,
        requested_ref=ref,
        resolved_ref=ref,
        url=_repo_url(repo_id, repo_type="model", ref=ref),
    )


def push_resume_checkpoint(
    checkpoint_path: str | Path,
    repo_id: str,
    *,
    step: int,
    private: bool = True,
    api: Any | None = None,
) -> HubArtifact:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    client = _client(api)
    client.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    commit = client.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo="resume/latest.pt",
        repo_id=repo_id,
        commit_message=f"Update resume checkpoint at step {step}",
    )
    ref = str(commit.oid)
    return HubArtifact(
        repo_id=repo_id,
        requested_ref=ref,
        resolved_ref=ref,
        url=_repo_url(repo_id, repo_type="model", ref=ref),
    )


def pull_model_checkpoint(
    repo_id: str,
    *,
    ref: str = "",
    filename: str = "best.pt",
    cache_dir: str = "",
    api: Any | None = None,
) -> HubArtifact:
    kwargs = {"repo_id": repo_id, "filename": filename}
    if ref:
        kwargs["revision"] = ref
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    path = Path(hf_hub_download(**kwargs))
    info_kwargs = {"repo_id": repo_id}
    if ref:
        info_kwargs["revision"] = ref
    info = _client(api).model_info(**info_kwargs)
    resolved = str(info.sha)
    return HubArtifact(
        repo_id=repo_id,
        requested_ref=ref,
        resolved_ref=resolved,
        path=path,
        url=_repo_url(repo_id, repo_type="model", ref=resolved),
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
    ref = str(commit.oid)
    return HubArtifact(
        repo_id=repo_id,
        requested_ref=ref,
        resolved_ref=ref,
        url=_repo_url(repo_id, repo_type="model", ref=ref),
    )


def pull_model_assets(
    repo_id: str,
    *,
    ref: str = "",
    local_dir: str = "models",
    cache_dir: str = "",
    api: Any | None = None,
) -> HubArtifact:
    kwargs = {
        "repo_id": repo_id,
        "allow_patterns": MODEL_ASSET_UPLOAD_PATTERNS,
    }
    if ref:
        kwargs["revision"] = ref
    if local_dir:
        kwargs["local_dir"] = local_dir
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    path = Path(snapshot_download(**kwargs))
    info_kwargs = {"repo_id": repo_id}
    if ref:
        info_kwargs["revision"] = ref
    info = _client(api).model_info(**info_kwargs)
    resolved = str(info.sha)
    return HubArtifact(
        repo_id=repo_id,
        requested_ref=ref,
        resolved_ref=resolved,
        path=path,
        url=_repo_url(repo_id, repo_type="model", ref=resolved),
    )


def push_dataset_snapshot(
    snapshot_root: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: str = "Upload pose-paired dataset snapshot",
    include_reports: bool = False,
    workers: int = 8,
    api: Any | None = None,
) -> HubArtifact:
    del commit_message
    if workers < 1:
        raise ValueError("workers must be >= 1")
    root = Path(snapshot_root)
    if not (root / "build_complete.json").is_file():
        raise FileNotFoundError(root / "build_complete.json")
    if not (root / "dataset/dataset_dict.json").is_file():
        raise FileNotFoundError(root / "dataset/dataset_dict.json")
    client = _client(api)
    client.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    upload_kwargs: dict[str, Any] = {
        "folder_path": str(root),
        "repo_id": repo_id,
        "repo_type": "dataset",
        "num_workers": workers,
    }
    if not include_reports:
        upload_kwargs["allow_patterns"] = DATASET_TRAIN_TRANSFER_PATTERNS
    client.upload_large_folder(**upload_kwargs)
    ref = str(client.dataset_info(repo_id=repo_id).sha)
    return HubArtifact(
        repo_id=repo_id,
        requested_ref=ref,
        resolved_ref=ref,
        url=_repo_url(repo_id, repo_type="dataset", ref=ref),
    )


def pull_dataset_snapshot(
    repo_id: str,
    *,
    ref: str,
    local_dir: str,
    cache_dir: str = "",
    include_reports: bool = False,
    workers: int = 16,
    api: Any | None = None,
    verify: Callable[[Path], dict[str, str]],
) -> HubArtifact:
    if not ref:
        raise ValueError("Dataset snapshot revision is required")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    download_profile = "full" if include_reports else "train-only"
    root = Path(local_dir)
    marker_path = root / ".snapshot_complete.json"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            fingerprints = verify(root)
        except Exception:
            marker = {}
            fingerprints = {}
        marker_profile = marker.get("download_profile", "full")
        if (
            marker.get("repo_id") == repo_id
            and marker.get("resolved_ref") == ref
            and marker.get("dataset_fingerprints") == fingerprints
            and (
                marker_profile == "full"
                or marker_profile == download_profile
            )
        ):
            return HubArtifact(
                repo_id=repo_id,
                requested_ref=ref,
                resolved_ref=ref,
                path=root,
                url=_repo_url(repo_id, repo_type="dataset", ref=ref),
            )
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": ref,
        "local_dir": str(root),
        "max_workers": workers,
    }
    if not include_reports:
        kwargs["allow_patterns"] = DATASET_TRAIN_TRANSFER_PATTERNS
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    downloaded = Path(snapshot_download(**kwargs))
    info = _client(api).dataset_info(repo_id=repo_id, revision=ref)
    resolved = str(info.sha)
    if resolved != ref:
        raise ValueError(f"Resolved dataset revision {resolved} does not match requested {ref}")
    fingerprints = verify(downloaded)
    marker = {
        "repo_id": repo_id,
        "requested_ref": ref,
        "resolved_ref": resolved,
        "dataset_fingerprints": fingerprints,
        "download_profile": download_profile,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    temporary.replace(marker_path)
    return HubArtifact(
        repo_id=repo_id,
        requested_ref=ref,
        resolved_ref=resolved,
        path=downloaded,
        url=_repo_url(repo_id, repo_type="dataset", ref=resolved),
    )
