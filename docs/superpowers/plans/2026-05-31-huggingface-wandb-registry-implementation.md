# Hugging Face Registry And W&B Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Hugging Face dataset and model versioning plus optional W&B experiment tracking while preserving the existing local training and evaluation workflow.

**Architecture:** Keep network SDKs at the system boundary. `edge_lipsync/hub.py` wraps Hugging Face Hub, `edge_lipsync/sources.py` resolves local or pinned Hub inputs into local paths, and `edge_lipsync/tracking.py` wraps W&B behind a small tracker interface. Training and evaluation continue using local paths after resolution.

**Tech Stack:** Python 3.11+, PyTorch, `huggingface-hub`, `wandb`, pytest, Ruff, Pyright, uv.

---

## File Structure

```text
edge_lipsync/
  hub.py                    # Hugging Face upload/download boundary
  sources.py                # local-or-Hub source validation and path resolution
  tracking.py               # optional W&B boundary
  checkpoint.py             # checkpoint provenance payload
  training.py               # source resolution, tracking, final artifacts, model publication
  eval.py                   # eval source configuration
tools/
  hf_dataset.py             # dataset Hub push/pull CLI
  hf_model.py               # model Hub push/pull CLI
  render_eval.py            # resolve eval inputs before loading dataset and model
configs/
  train.example.yaml        # Hub and W&B example settings
  eval.example.yaml         # Hub-backed eval settings
tests/
  test_hub.py               # fake-client Hub behavior
  test_sources.py           # source validation and resolution
  test_tracking.py          # fake-module W&B behavior
  test_checkpoint.py        # provenance checkpoint regression
  test_training.py          # final artifact helpers and CLI regression
  test_eval.py              # eval input resolver regression
README.md                   # login, push/pull, train, eval documentation
pyproject.toml              # runtime dependencies
uv.lock                     # resolved dependencies
```

---

### Task 1: Hugging Face Registry Boundary

**Files:**
- Create: `edge_lipsync/hub.py`
- Create: `tests/test_hub.py`
- Create: `tools/hf_dataset.py`
- Create: `tools/hf_model.py`

- [ ] **Step 1: Write failing Hub boundary tests**

Add tests using a fake API object and monkeypatched download functions:

```python
def test_push_dataset_snapshot_uploads_only_processed_artifacts(tmp_path: Path) -> None:
    api = FakeApi()
    dataset_root = _write_dataset_root(tmp_path)
    result = push_dataset_snapshot(dataset_root, "owner/avatar-dataset", api=api)
    assert result.resolved_revision == "dataset-commit"
    assert api.created == [("owner/avatar-dataset", "dataset", True)]
    assert api.uploads[0]["allow_patterns"] == DATASET_UPLOAD_PATTERNS


def test_pull_dataset_snapshot_requires_revision() -> None:
    with pytest.raises(ValueError, match="revision"):
        pull_dataset_snapshot("owner/avatar-dataset", revision="")


def test_push_model_artifacts_uses_model_allowlist(tmp_path: Path) -> None:
    api = FakeApi()
    run_dir = _write_run_dir(tmp_path)
    result = push_model_artifacts(run_dir, "owner/avatar-model", api=api)
    assert result.resolved_revision == "model-commit"
    assert api.uploads[0]["allow_patterns"] == MODEL_UPLOAD_PATTERNS


def test_pull_model_checkpoint_resolves_file_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub, "hf_hub_download", lambda **kwargs: "/cache/best.pt")
    api = FakeApi()
    result = pull_model_checkpoint("owner/avatar-model", revision="v1", api=api)
    assert result.path == Path("/cache/best.pt")
    assert result.resolved_revision == "model-sha"
```

- [ ] **Step 2: Run Hub tests to verify missing-module failure**

Run:

```bash
.venv/bin/pytest tests/test_hub.py -q
```

Expected: FAIL because `edge_lipsync.hub` does not exist.

- [ ] **Step 3: Implement the Hub boundary**

Create these public types and functions in `edge_lipsync/hub.py`:

```python
DATASET_UPLOAD_PATTERNS = (
    "manifest.jsonl",
    "splits.json",
    "build_summary.json",
    "clips/*/frames/*.jpg",
    "clips/*/bnf.npy",
    "clips/*/bboxes.json",
    "clips/*/quality.json",
    "clips/*/previews/*.jpg",
)
MODEL_UPLOAD_PATTERNS = ("best.pt", "final.pt", "metrics.json", "metrics.csv", "run_metadata.json", "README.md")


@dataclass(frozen=True)
class HubArtifact:
    repo_id: str
    requested_revision: str
    resolved_revision: str
    path: Path | None = None
    url: str = ""


def push_dataset_snapshot(dataset_root: str | Path, repo_id: str, *, private: bool = True, commit_message: str = "Upload processed dataset snapshot", api: Any | None = None) -> HubArtifact:
    return _push_folder(dataset_root, repo_id, repo_type="dataset", private=private, commit_message=commit_message, allow_patterns=DATASET_UPLOAD_PATTERNS, required=DATASET_REQUIRED_PATHS, api=api)


def pull_dataset_snapshot(repo_id: str, *, revision: str, cache_dir: str = "", api: Any | None = None) -> HubArtifact:
    return _pull_snapshot(repo_id, repo_type="dataset", revision=revision, cache_dir=cache_dir, api=api)


def push_model_artifacts(run_dir: str | Path, repo_id: str, *, private: bool = True, commit_message: str = "Upload training run artifacts", api: Any | None = None) -> HubArtifact:
    return _push_folder(run_dir, repo_id, repo_type="model", private=private, commit_message=commit_message, allow_patterns=MODEL_UPLOAD_PATTERNS, required=MODEL_REQUIRED_PATHS, api=api)


def pull_model_checkpoint(repo_id: str, *, revision: str, filename: str = "best.pt", cache_dir: str = "", api: Any | None = None) -> HubArtifact:
    return _pull_file(repo_id, revision=revision, filename=filename, cache_dir=cache_dir, api=api)
```

Validate dataset roots before network calls. Require `manifest.jsonl`, `splits.json`,
`build_summary.json`, and `clips/`. Require model run directories to contain `best.pt`,
`final.pt`, `metrics.json`, `metrics.csv`, and `run_metadata.json`. Resolve download SHAs with
`api.dataset_info(repo_id=repo_id, revision=revision).sha` and
`api.model_info(repo_id=repo_id, revision=revision).sha`. Resolve upload SHAs from
`CommitInfo.oid`.

- [ ] **Step 4: Add thin CLI wrappers**

Implement `push` and `pull` subcommands. Require a revision for pulls. Make pushes private unless
`--public` is passed. Print `repo_id`, `revision`, `url`, and cached `path` values.

- [ ] **Step 5: Run Hub tests and CLI help tests**

Run:

```bash
.venv/bin/pytest tests/test_hub.py -q
.venv/bin/python tools/hf_dataset.py --help
.venv/bin/python tools/hf_model.py --help
```

Expected: PASS and both CLIs list `push` and `pull`.

- [ ] **Step 6: Commit**

```bash
git add edge_lipsync/hub.py tests/test_hub.py tools/hf_dataset.py tools/hf_model.py
git commit -m "feat(hub): add dataset and model registry boundary"
```

---

### Task 2: Shared Source Resolution And Checkpoint Provenance

**Files:**
- Create: `edge_lipsync/sources.py`
- Create: `tests/test_sources.py`
- Modify: `edge_lipsync/checkpoint.py`
- Modify: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing source and provenance tests**

Cover:

```python
def test_resolve_dataset_source_rejects_local_and_hub_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        resolve_dataset_source(dataset_root=str(tmp_path), hf_repo="owner/data", hf_revision="v1")


def test_resolve_dataset_source_uses_hub_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sources, "pull_dataset_snapshot", fake_pull_dataset)
    result = resolve_dataset_source(dataset_root="", hf_repo="owner/data", hf_revision="v1")
    assert result.path == Path("/cache/data")
    assert result.provenance["resolved_revision"] == "data-sha"


def test_make_training_checkpoint_includes_provenance(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"clip_id":"clip_001"}\n')
    payload = make_training_checkpoint(
        model=torch.nn.Conv2d(6, 3, kernel_size=1),
        training_config={},
        dataset_root=tmp_path,
        manifest_path=manifest,
        step=1,
        epoch=1,
        metrics={},
        init_weight_source={"kind": "ncnn_bin", "path": "/tmp/dh_model.bin"},
        provenance={"dataset": {"source": "local"}},
    )
    assert payload["provenance"]["dataset"]["source"] == "local"
```

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_sources.py tests/test_checkpoint.py -q
```

Expected: FAIL because source resolution and checkpoint provenance are missing.

- [ ] **Step 3: Implement shared source resolution**

Create:

```python
@dataclass(frozen=True)
class ResolvedSource:
    path: Path
    provenance: dict[str, Any]


def resolve_dataset_source(*, dataset_root: str, hf_repo: str = "", hf_revision: str = "", cache_dir: str = "") -> ResolvedSource:
    _require_exactly_one_source(dataset_root, hf_repo)
    if dataset_root:
        return _resolve_local(dataset_root, kind="dataset")
    return _resolve_hub_dataset(hf_repo, revision=hf_revision, cache_dir=cache_dir)


def resolve_model_source(*, checkpoint: str, hf_repo: str = "", hf_revision: str = "", hf_filename: str = "best.pt", cache_dir: str = "") -> ResolvedSource:
    _require_exactly_one_source(checkpoint, hf_repo)
    if checkpoint:
        return _resolve_local(checkpoint, kind="model")
    return _resolve_hub_model(hf_repo, revision=hf_revision, filename=hf_filename, cache_dir=cache_dir)
```

Require exactly one local or Hub source. Require pinned Hub revisions. For local datasets verify
the directory exists. For local checkpoints verify the file exists. Include source, path, repo ID,
requested revision, and resolved revision in provenance as applicable.

- [ ] **Step 4: Add checkpoint provenance**

Add `provenance: dict[str, Any] | None = None` to `make_training_checkpoint()` and store a copy as
`payload["provenance"]`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
.venv/bin/pytest tests/test_sources.py tests/test_checkpoint.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add edge_lipsync/sources.py tests/test_sources.py edge_lipsync/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(provenance): resolve pinned local and hub sources"
```

---

### Task 3: Optional W&B Tracker

**Files:**
- Create: `edge_lipsync/tracking.py`
- Create: `tests/test_tracking.py`

- [ ] **Step 1: Write failing tracker tests**

Use a fake `wandb` module:

```python
def test_disabled_tracker_does_not_import_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "__import__", reject_wandb_import)
    tracker = create_tracker(WandbConfig(mode="disabled"), run_config={}, provenance={})
    assert tracker.provenance == {"mode": "disabled"}


def test_wandb_tracker_logs_metrics_summary_and_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    tracker = create_tracker(WandbConfig(mode="offline", project="edge-lipsync-model"), run_config={"max_steps": 3}, provenance={"dataset": {"source": "local"}})
    tracker.log_metrics({"train_loss": 0.5}, step=1)
    tracker.update_summary({"best_val": 0.25})
    tracker.finish()
    assert fake.run.logged == [({"train_loss": 0.5}, 1)]
    assert fake.run.summary["best_val"] == 0.25
    assert fake.run.finished
```

- [ ] **Step 2: Run tracker tests to verify missing-module failure**

Run:

```bash
.venv/bin/pytest tests/test_tracking.py -q
```

Expected: FAIL because `edge_lipsync.tracking` does not exist.

- [ ] **Step 3: Implement tracker interface**

Create:

```python
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
    def provenance(self) -> dict[str, str]:
        raise NotImplementedError
    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        raise NotImplementedError
    def update_summary(self, values: dict[str, Any]) -> None:
        raise NotImplementedError
    def finish(self, *, exit_code: int = 0) -> None:
        raise NotImplementedError


def create_tracker(config: WandbConfig, *, run_config: dict[str, Any], provenance: dict[str, Any]) -> Tracker:
    if config.mode == "disabled":
        return DisabledTracker()
    if config.mode not in {"online", "offline"}:
        raise ValueError(f"Unsupported W&B mode={config.mode!r}")
    return WandbTracker(config, run_config=run_config, provenance=provenance)
```

Return a no-op tracker for `disabled`. Validate `online`, `offline`, and `disabled`. Import W&B
only for enabled modes. Initialize W&B with project, optional entity/name/group/tags/notes/dir,
mode, and config containing training configuration plus provenance.

- [ ] **Step 4: Run tracker tests**

Run:

```bash
.venv/bin/pytest tests/test_tracking.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/tracking.py tests/test_tracking.py
git commit -m "feat(tracking): add optional wandb experiment tracker"
```

---

### Task 4: Training Wiring And Model Publication

**Files:**
- Modify: `edge_lipsync/training.py`
- Modify: `tests/test_training.py`
- Modify: `configs/train.example.yaml`

- [ ] **Step 1: Write failing training helper tests**

Cover:

```python
def test_write_run_metadata_records_provenance(tmp_path: Path) -> None:
    out = write_run_metadata(tmp_path, provenance={"dataset": {"source": "local"}}, best_checkpoint=tmp_path / "best.pt", final_checkpoint=tmp_path / "final.pt")
    assert json.loads(out.read_text())["provenance"]["dataset"]["source"] == "local"


def test_write_model_card_links_dataset_and_wandb(tmp_path: Path) -> None:
    out = write_model_card(tmp_path, provenance={"dataset": {"repo_id": "owner/data", "resolved_revision": "data-sha"}, "wandb": {"run_url": "https://wandb.ai/owner/project/runs/run-id"}})
    text = out.read_text()
    assert "owner/data" in text
    assert "https://wandb.ai/owner/project/runs/run-id" in text
```

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_training.py -q
```

Expected: FAIL because run metadata and model card helpers are missing.

- [ ] **Step 3: Extend `TrainConfig` and training source resolution**

Add fields for Hub dataset input, Hub initialization input, cache directory, optional model repo
publication, and W&B settings. Resolve dataset and init model sources before creating loaders or
loading model weights. Preserve local `init_bin` loading. Add dataset and init model provenance to
each checkpoint.

- [ ] **Step 4: Integrate tracking and final artifacts**

Create the tracker before the optimization loop. Log each metrics row with its explicit step.
Always write `final.pt`, `run_metadata.json`, and `README.md` after training. Update W&B summary
with local artifacts. If `hf_model_repo` is set, publish model artifacts and add model revision
and URL to both local metadata and W&B summary. Finish W&B normally; finish with non-zero exit code
when training raises.

- [ ] **Step 5: Run training tests**

Run:

```bash
.venv/bin/pytest tests/test_training.py tests/test_checkpoint.py tests/test_sources.py tests/test_tracking.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add edge_lipsync/training.py tests/test_training.py configs/train.example.yaml
git commit -m "feat(training): track runs and publish model artifacts"
```

---

### Task 5: Hub-Backed Evaluation

**Files:**
- Modify: `edge_lipsync/eval.py`
- Modify: `tools/render_eval.py`
- Modify: `tests/test_eval.py`
- Modify: `configs/eval.example.yaml`

- [ ] **Step 1: Write failing eval input resolution tests**

Cover a local eval path and a Hub-backed eval path. Monkeypatch shared source resolution for the
Hub path and assert that `_resolve_eval_inputs()` returns cached dataset and checkpoint paths.

- [ ] **Step 2: Run eval tests to verify failure**

Run:

```bash
.venv/bin/pytest tests/test_eval.py -q
```

Expected: FAIL because Hub eval config and input resolution are missing.

- [ ] **Step 3: Add eval Hub fields and resolve inputs**

Extend `RenderEvalConfig` with:

```python
dataset_root: str = ""
ckpt: str = ""
hf_dataset_repo: str = ""
hf_dataset_revision: str = ""
hf_model_repo: str = ""
hf_model_revision: str = ""
hf_model_filename: str = "best.pt"
hf_cache_dir: str = ""
```

Resolve local or pinned Hub inputs before creating `DuixManifestDataset` and calling `load_ckpt`.
Keep render artifact generation unchanged.

- [ ] **Step 4: Run eval tests**

Run:

```bash
.venv/bin/pytest tests/test_eval.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edge_lipsync/eval.py tools/render_eval.py tests/test_eval.py configs/eval.example.yaml
git commit -m "feat(eval): resolve pinned hub datasets and models"
```

---

### Task 6: Dependencies, Documentation, And Full Verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`

- [ ] **Step 1: Add runtime dependencies**

Run:

```bash
uv add huggingface-hub wandb
```

Expected: `pyproject.toml` and `uv.lock` include both packages and their resolved dependencies.

- [ ] **Step 2: Document the workflow**

Update `README.md` with:

```bash
export HF_TOKEN=hf_write_token_from_hugging_face
hf auth login --token "$HF_TOKEN"
wandb login

.venv/bin/python tools/hf_dataset.py push \
  --dataset-root /absolute/path/to/data/duix_datasets/avatar_name \
  --repo-id username/avatar-name-dataset

.venv/bin/python tools/train.py --config configs/train.example.yaml

.venv/bin/python tools/hf_model.py pull \
  --repo-id username/avatar-name-model \
  --revision model-v1 \
  --filename best.pt

.venv/bin/python tools/render_eval.py --config configs/eval.example.yaml
```

State that repos are private by default, source videos and normalized video/audio intermediates
are excluded, Hub inputs require pinned revisions, and W&B supports disabled/offline/online modes.

- [ ] **Step 3: Run formatting, type checking, and tests**

Run:

```bash
.venv/bin/ruff check .
.venv/bin/pyright
.venv/bin/pytest -q
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 4: Audit requirement coverage**

Confirm:

```text
HF dataset storage/versioning    -> hub.py dataset push/pull + CLI + tests
HF dataset consumption          -> sources.py + training/eval resolution + tests
HF model storage/versioning      -> hub.py model push/pull + training publication + CLI + tests
W&B train logging/debugging      -> tracking.py + training integration + tests
Local fallback                  -> source resolver + disabled tracker + existing tests
Credential hygiene              -> README env setup + no token fields in configs or metadata
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock README.md
git commit -m "docs: document hub and wandb workflow"
```
