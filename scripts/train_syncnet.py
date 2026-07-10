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

from avagen.training.train_syncnet import train_syncnet
from avagen.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the audio<->motion SyncNet discriminator.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train_syncnet.yaml")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config_data = load_config(args.config)
    if not isinstance(config_data, dict):
        raise ValueError(f"Expected top-level config mapping in {args.config}")
    if args.manifest_path is not None:
        config_data["manifest_path"] = str(args.manifest_path.expanduser().resolve())
    if args.experiment_dir is not None:
        config_data["experiment_dir"] = str(args.experiment_dir.expanduser().resolve())
    training = dict(config_data.get("training", {}))
    if args.device is not None:
        training["device"] = args.device
    if args.epochs is not None:
        training["epochs"] = args.epochs
    config_data["training"] = training
    for field in ("manifest_path", "experiment_dir"):
        if not config_data.get(field):
            raise ValueError(f"Missing '{field}' (set in {args.config} or pass --{field.replace('_','-')}).")
    print(json.dumps(train_syncnet(config_data), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
