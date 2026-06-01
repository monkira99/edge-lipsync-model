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
