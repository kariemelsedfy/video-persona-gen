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

from avagen.inference.generate_motion import predict_motion_for_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict motion features and templates from a trained GRU checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained motion checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training config or resolved config JSON used to recover dataset feature names.",
    )
    parser.add_argument("--manifest-path", type=Path, required=True, help="Manifest containing processed clips to predict.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "predicted_motion",
        help="Root directory for predicted motion artifacts.",
    )
    parser.add_argument("--clip-id", action="append", default=[], help="Optional clip id filter. Repeat to select multiple.")
    parser.add_argument("--device", default="auto", help="Prediction device: auto, cpu, or cuda.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = predict_motion_for_manifest(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        manifest_path=args.manifest_path,
        output_root=args.output_root,
        clip_ids=tuple(args.clip_id),
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
