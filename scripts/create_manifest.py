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

from avagen.data.manifests import refresh_identity_manifest


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
    records, manifest_output, report_output = refresh_identity_manifest(
        identity_dir=identity_dir,
        manifest_path=args.manifest_path,
        report_path=args.report_path,
    )
    payload = {
        "identity_dir": str(identity_dir.resolve()),
        "manifest_path": str(manifest_output),
        "report_path": str(report_output),
        "num_records": len(records),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
