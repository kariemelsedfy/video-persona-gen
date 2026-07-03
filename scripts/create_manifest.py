#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or refresh manifest files for a processed identity.")
    parser.add_argument("--identity-id", required=True, help="Identity directory under data/processed.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed dataset root.",
    )
    parser.add_argument("--manifest-path", type=Path, help="Optional explicit manifest output path.")
    parser.add_argument("--report-path", type=Path, help="Optional explicit dataset report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    identity_dir = args.processed_root / args.identity_id
    payload = {
        "status": "skeleton",
        "script": "create_manifest",
        "identity_dir": str(identity_dir),
        "manifest_path": str(args.manifest_path) if args.manifest_path else str(identity_dir / "manifest.jsonl"),
        "report_path": str(args.report_path) if args.report_path else str(identity_dir / "dataset_report.json"),
        "next_step": "Implement avagen.data.manifests.refresh_identity_manifest.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
