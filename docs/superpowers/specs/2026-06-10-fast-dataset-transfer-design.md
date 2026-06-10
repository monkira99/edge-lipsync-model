# Fast Dataset Snapshot Transfer Design

## Goal

Make dataset upload and download practical for large silent/talking snapshots while
preserving the existing training contract:

```text
snapshot_download / pull-snapshot
-> datasets.load_from_disk(<snapshot>/dataset)
```

The train transport package contains only:

```text
dataset/**
build_complete.json
build_metadata.json
```

Quality reports and preview PNGs remain local and are not part of the default Hub
transport.

## Upload

Replace `HfApi.upload_folder()` in `push_dataset_snapshot()` with
`HfApi.upload_large_folder()`.

The builder snapshot root is uploaded directly with:

```python
allow_patterns=[
    "dataset/**",
    "build_complete.json",
    "build_metadata.json",
]
```

No staging copy is created. This avoids duplicating an 11 GB snapshot on Colab
storage.

`upload_large_folder()` provides:

- parallel hashing and upload workers;
- resumable state under `<snapshot_root>/.cache/.huggingface/`;
- multiple incremental commits instead of one large final commit;
- safe restart after a disconnected Colab session.

The CLI adds:

```text
--num-workers N
```

with a default of `8`. The value is passed directly to Hugging Face Hub. Users may
lower it for unstable connections.

Because `upload_large_folder()` returns no commit object, the implementation calls
`dataset_info(repo_id, revision="main")` after upload completes and returns the
resolved full commit SHA.

The upload command prints that SHA exactly as the current notebook expects.

## Download

`pull_dataset_snapshot()` uses the same allowlist:

```text
dataset/**
build_complete.json
build_metadata.json
```

This prevents old reports or preview files already committed to the repository from
being downloaded.

The existing revision pinning, fingerprint verification, and
`.snapshot_complete.json` marker remain unchanged. A repeated pull of the same
verified revision performs no Hub download.

For maximum Hub/Xet throughput, documentation recommends:

```bash
export HF_XET_HIGH_PERFORMANCE=1
```

This is not forced by Python because it can consume all available network bandwidth
and CPU.

## Existing Repositories

`upload_large_folder()` does not delete remote files. If a previous interrupted
upload already committed reports or previews, they may remain visible in the Hub
repository.

This does not affect training download speed because `pull-snapshot` filters them
out. Automatic remote deletion is intentionally excluded to avoid destructive
behavior.

## Error Handling

- Missing `build_complete.json` or `dataset/dataset_dict.json` fails before upload.
- Upload interruption leaves resumable metadata in `.cache/.huggingface/`.
- The full revision SHA is emitted only after upload returns successfully.
- Download still fails if the resolved revision differs from the requested SHA.
- Dataset fingerprints are verified after download.

## Tests

Add tests proving:

- snapshot upload calls `upload_large_folder`, not `upload_folder`;
- the train-only allowlist and worker count are forwarded;
- the returned artifact contains the post-upload dataset SHA;
- snapshot download uses the train-only allowlist;
- existing marker-based download skipping still works;
- CLI accepts `--num-workers`.

Notebook files are not modified. Existing local notebook changes remain untouched.
