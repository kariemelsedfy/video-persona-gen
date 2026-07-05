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

from avagen.data.dataset import load_processed_clip_records
from avagen.data.splits import apply_split_assignments, assign_clip_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign train/val/test splits for one processed identity.")
    parser.add_argument("--identity-id", required=True, help="Identity directory under data/processed.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed dataset root.",
    )
    parser.add_argument("--manifest-path", type=Path, help="Optional explicit manifest path override.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Train split fraction.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split assignment.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    identity_dir = (args.processed_root / args.identity_id).resolve()
    manifest_path = args.manifest_path.resolve() if args.manifest_path else identity_dir / "manifest.jsonl"
    records = load_processed_clip_records(manifest_path)
    assignments = assign_clip_splits(
        [record.clip_id for record in records],
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    manifest_output, report_output = apply_split_assignments(identity_dir, assignments)
    payload = {
        "identity_dir": str(identity_dir),
        "manifest_path": str(manifest_output),
        "report_path": str(report_output),
        "assignments": [{"clip_id": item.clip_id, "split": item.split} for item in assignments],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
