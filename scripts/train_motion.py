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

from avagen.training.train_motion import train_motion_model
from avagen.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GRU audio-to-motion baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train_motion_gru.yaml",
        help="YAML config file. Defaults to configs/train_motion_gru.yaml.",
    )
    parser.add_argument("--manifest-path", type=Path, help="Optional manifest override.")
    parser.add_argument("--experiment-dir", type=Path, help="Optional experiment directory override.")
    parser.add_argument("--device", help="Optional device override: auto, cpu, or cuda.")
    parser.add_argument("--epochs", type=int, help="Optional epoch override.")
    parser.add_argument("--limit", type=int, help="Optional dataset limit override.")
    parser.add_argument("--resume-from", type=Path, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def _ensure_mapping(value: Any, field_name: str) -> dict[str, Any]:
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

    if args.manifest_path is not None:
        config_data["manifest_path"] = str(args.manifest_path.expanduser().resolve())
    if args.experiment_dir is not None:
        config_data["experiment_dir"] = str(args.experiment_dir.expanduser().resolve())

    dataset_config = _ensure_mapping(config_data.get("dataset"), "dataset")
    training_config = _ensure_mapping(config_data.get("training"), "training")

    if args.limit is not None:
        dataset_config["limit"] = args.limit
    if args.device is not None:
        training_config["device"] = args.device
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.resume_from is not None:
        training_config["resume_from"] = str(args.resume_from.expanduser().resolve())

    config_data["dataset"] = dataset_config
    config_data["training"] = training_config

    for field_name in ("manifest_path", "experiment_dir"):
        if field_name not in config_data or not config_data[field_name]:
            raise ValueError(
                f"Missing required value for '{field_name}'. Provide it in {args.config} or pass --{field_name.replace('_', '-')}."
            )

    summary = train_motion_model(config_data)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
