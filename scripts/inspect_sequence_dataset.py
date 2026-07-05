#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from avagen.data.dataset import AudioMotionSequenceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect aligned audio-motion sequences from a processed manifest.")
    parser.add_argument("--manifest-path", type=Path, required=True, help="Manifest path to inspect.")
    parser.add_argument("--split", action="append", default=[], help="Optional split filter.")
    parser.add_argument("--limit", type=int, help="Optional record limit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = AudioMotionSequenceDataset(
        manifest_path=args.manifest_path,
        splits=args.split or None,
        limit=args.limit,
    )
    payload = {
        "manifest_path": str(Path(args.manifest_path).expanduser().resolve()),
        "num_sequences": len(dataset),
        "clip_ids": dataset.clip_ids(),
        "identity_ids": dataset.identity_ids(),
    }
    if len(dataset) > 0:
        first = dataset[0]
        payload["first_sequence"] = {
            "clip_id": first.clip_id,
            "audio_shape": list(first.audio_features.shape),
            "motion_shape": list(first.motion_features.shape),
            "motion_fps": first.motion_fps,
            "audio_feature_names": list(first.audio_feature_names),
            "motion_feature_name": first.motion_feature_name,
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
