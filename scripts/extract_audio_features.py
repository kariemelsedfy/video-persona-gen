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

from avagen.data.audio import extract_audio_features_for_manifest
from avagen.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract framewise audio features for processed clips.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "extract_audio_features.yaml",
        help="YAML config file. Defaults to configs/extract_audio_features.yaml.",
    )
    parser.add_argument("--manifest-path", type=Path, help="Optional explicit manifest path override.")
    parser.add_argument("--clip-id", action="append", default=[], help="Restrict extraction to one or more clip IDs.")
    parser.add_argument("--frame-length-ms", type=float, help="Frame length in milliseconds.")
    parser.add_argument("--hop-length-ms", type=float, help="Hop length in milliseconds.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files.")
    return parser.parse_args()


def _get_value(args: argparse.Namespace, config: dict[str, object], name: str, default: object = None) -> object:
    value = getattr(args, name)
    if value not in (None, [], ""):
        return value
    return config.get(name, default)


def _get_required_value(args: argparse.Namespace, config: dict[str, object], name: str) -> object:
    value = _get_value(args, config, name)
    if value is None:
        raise ValueError(
            f"Missing required value for '{name}'. Provide it in {args.config} or pass --{name.replace('_', '-')}."
        )
    return value


def main() -> int:
    args = parse_args()
    config_data = load_config(args.config) if args.config else {}
    result = extract_audio_features_for_manifest(
        manifest_path=Path(_get_required_value(args, config_data, "manifest_path")),
        frame_length_ms=float(_get_value(args, config_data, "frame_length_ms", 40.0)),
        hop_length_ms=float(_get_value(args, config_data, "hop_length_ms", 10.0)),
        clip_ids=tuple(args.clip_id or config_data.get("clip_ids", []) or []),
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
