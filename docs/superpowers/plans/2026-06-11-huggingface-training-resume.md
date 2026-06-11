# Hugging Face Training Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full-state training recovery through `resume/latest.pt` on Hugging Face.

**Architecture:** Keep inference checkpoints unchanged and introduce a separate versioned resume
checkpoint containing optimization, progress, data-order, best-model, metrics, and RNG state.
Training owns explicit epoch permutations so a resumed run starts at the exact next batch. Periodic
Hub upload is synchronous and non-fatal, while local checkpoint writes and resume validation are
strict.

**Tech Stack:** Python 3.11+, PyTorch, Hugging Face Hub, pytest, Ruff, Pyright

---

### Task 1: Resume Checkpoint Primitives

**Files:**
- Modify: `edge_lipsync/checkpoint.py`
- Test: `tests/test_checkpoint.py`

- [ ] Write failing tests for RNG capture/restore, CPU-cloned state dictionaries, complete resume
      payloads, strict loading, and model checkpoint creation from a saved state dictionary.
- [ ] Run `pytest tests/test_checkpoint.py -q` and confirm the new tests fail for missing APIs.
- [ ] Implement the minimal versioned checkpoint helpers and validation.
- [ ] Run `pytest tests/test_checkpoint.py -q` and confirm all tests pass.
- [ ] Commit with `feat(training): add full resume checkpoint format`.

### Task 2: Hugging Face Latest Checkpoint API

**Files:**
- Modify: `edge_lipsync/hub.py`
- Test: `tests/test_hub.py`

- [ ] Write a failing test asserting one `upload_file` call to `resume/latest.pt` with a
      step-specific commit message.
- [ ] Run the targeted test and confirm it fails because the API is missing.
- [ ] Implement `push_resume_checkpoint`.
- [ ] Run `pytest tests/test_hub.py -q`.
- [ ] Commit with `feat(hub): upload latest training resume checkpoint`.

### Task 3: Configuration And Compatibility Validation

**Files:**
- Modify: `edge_lipsync/training.py`
- Test: `tests/test_training.py`

- [ ] Write failing tests for source exclusivity, upload interval validation, critical config
      mismatch, dataset identity mismatch, device/precision mismatch, and completed checkpoints.
- [ ] Run the targeted tests and confirm expected failures.
- [ ] Implement configuration and resume compatibility helpers.
- [ ] Run the targeted tests and then `pytest tests/test_training.py -q`.
- [ ] Commit with `feat(training): validate resume compatibility`.

### Task 4: Deterministic Mid-Epoch Continuation

**Files:**
- Modify: `edge_lipsync/training.py`
- Test: `tests/test_training.py`

- [ ] Write failing tests for explicit epoch permutations and resuming at the next batch.
- [ ] Run the targeted tests and confirm expected failures.
- [ ] Replace implicit shuffle with an explicit generator, permutation, and batch cursor.
- [ ] Run the targeted tests and then `pytest tests/test_training.py -q`.
- [ ] Commit with `feat(training): restore exact data iteration state`.

### Task 5: Training State Restoration

**Files:**
- Modify: `edge_lipsync/training.py`
- Test: `tests/test_training.py`

- [ ] Write failing tests for optimizer/scaler restoration, metrics continuation, best model
      reconstruction, early-stopping state, and exact split-run parity.
- [ ] Run the targeted tests and confirm expected failures.
- [ ] Implement Hub resume loading and full training-state restoration.
- [ ] Run the targeted tests and then `pytest tests/test_training.py -q`.
- [ ] Commit with `feat(training): resume interrupted training runs`.

### Task 6: Periodic And Final Publication

**Files:**
- Modify: `edge_lipsync/training.py`
- Test: `tests/test_training.py`

- [ ] Write failing tests for interval uploads, non-fatal failures, retries, final upload, and
      tracker summary values.
- [ ] Run the targeted tests and confirm expected failures.
- [ ] Implement atomic local resume writes plus periodic/final Hub publication.
- [ ] Run the targeted tests and then `pytest tests/test_training.py -q`.
- [ ] Commit with `feat(training): publish periodic resume checkpoints`.

### Task 7: Documentation And Verification

**Files:**
- Modify: `configs/train.example.yaml`
- Modify: `README.md`

- [ ] Document periodic uploads, explicit resume, total `max_steps`, W&B behavior, upload failure
      behavior, and Hub history storage.
- [ ] Run checkpoint, Hub, and training tests.
- [ ] Run Ruff, Pyright, and the complete pytest suite.
- [ ] Commit with `docs(training): document hugging face resume workflow`.
