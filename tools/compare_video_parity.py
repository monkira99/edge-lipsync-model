#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_lipsync.parity import compare_video_parity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare restored video parity and write metrics.")
    parser.add_argument("--original-video", required=True)
    parser.add_argument("--original-frames-dir", required=True)
    parser.add_argument("--pipeline-video", required=True)
    parser.add_argument("--pipeline-frames-dir", required=True)
    parser.add_argument("--pipeline-metadata", required=True)
    parser.add_argument("--oracle-bbox-json", required=True)
    parser.add_argument("--audio-wav", required=True)
    parser.add_argument("--out-dir", default="artifacts/parity_emma")
    args = parser.parse_args()

    report = compare_video_parity(
        original_video=args.original_video,
        original_frames_dir=args.original_frames_dir,
        pipeline_video=args.pipeline_video,
        pipeline_frames_dir=args.pipeline_frames_dir,
        pipeline_metadata=args.pipeline_metadata,
        oracle_bbox_json=args.oracle_bbox_json,
        audio_wav=args.audio_wav,
        out_dir=args.out_dir,
    )
    print(f"report={Path(args.out_dir, 'report.json').resolve()}")
    print(f"passed={report['passed']}")
    print(f"failed_gates={json.dumps(report['failed_gates'])}")


if __name__ == "__main__":
    main()
