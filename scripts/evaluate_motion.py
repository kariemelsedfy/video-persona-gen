#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from avagen.evaluation.motion_metrics import evaluate_motion_predictions
from avagen.training.logging import write_json
from avagen.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted motion features against ground-truth motion bundles.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "evaluate_motion.yaml",
        help="YAML config file. Defaults to configs/evaluate_motion.yaml.",
    )
    parser.add_argument("--manifest-path", type=Path, help="Optional manifest override.")
    parser.add_argument("--predicted-root", type=Path, help="Optional predicted motion root override.")
    parser.add_argument("--output-path", type=Path, help="Optional metrics JSON output path override.")
    parser.add_argument("--clip-id", action="append", default=[], help="Optional clip id filter. Repeat to select multiple.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip clips without predicted artifacts instead of failing.")
    return parser.parse_args()


def _get_mapping(config_data: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = config_data.get(field_name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{field_name}' to be a mapping.")
    return dict(value)


def main() -> int:
    args = parse_args()
    config_data = load_config(args.config)
    if not isinstance(config_data, dict):
        raise ValueError(f"Expected top-level config mapping in {args.config}")

    manifest_path = args.manifest_path or config_data.get("manifest_path")
    predicted_root = args.predicted_root or config_data.get("predicted_root")
    output_path = args.output_path or config_data.get("output_path")
    if manifest_path is None or predicted_root is None:
        raise ValueError("Both manifest_path and predicted_root are required.")

    evaluation_config = _get_mapping(config_data, "evaluation")
    clip_ids = tuple(args.clip_id or evaluation_config.get("clip_ids", []))
    skip_missing = bool(args.skip_missing or evaluation_config.get("skip_missing", False))

    summary = evaluate_motion_predictions(
        manifest_path=manifest_path,
        predicted_root=predicted_root,
        clip_ids=clip_ids,
        skip_missing=skip_missing,
    )
    if output_path is not None:
        write_json(output_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
