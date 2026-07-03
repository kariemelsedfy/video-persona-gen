#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess one or more talking-head videos into clip folders.")
    parser.add_argument("--input", nargs="+", required=True, help="Input video path(s).")
    parser.add_argument("--identity-id", required=True, help="Target identity id, for example self_001.")
    parser.add_argument("--clip-id", help="Optional explicit clip id when preprocessing exactly one input.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed dataset root.",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="Target output FPS for frame extraction.")
    parser.add_argument("--audio-sample-rate", type=int, default=16000, help="Target audio sample rate.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split label.")
    parser.add_argument("--face-margin", type=float, default=0.2, help="Relative crop expansion around the face.")
    parser.add_argument(
        "--disable-center-crop-fallback",
        action="store_true",
        help="Fail if a face is not detected and no previous box is available.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing clip directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    identity_dir = args.output_root / args.identity_id
    payload = {
        "status": "skeleton",
        "script": "preprocess_dataset",
        "identity_dir": str(identity_dir),
        "inputs": args.input,
        "requested_clip_id": args.clip_id,
        "target_fps": args.fps,
        "audio_sample_rate": args.audio_sample_rate,
        "split": args.split,
        "next_step": "Implement avagen.data.preprocessing.preprocess_clip and manifest generation.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
