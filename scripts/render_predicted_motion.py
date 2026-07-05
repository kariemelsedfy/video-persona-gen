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

from avagen.renderers.video_renderer import (
    PredictedMotionRenderConfig,
    render_predicted_motion_for_manifest,
)
from avagen.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict motion from a checkpoint, convert it to a LivePortrait template, and render the result."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "render_predicted_motion.yaml",
        help="YAML config file. Defaults to configs/render_predicted_motion.yaml.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint override.")
    parser.add_argument("--model-config", type=Path, help="Optional training config or resolved config override.")
    parser.add_argument("--manifest-path", type=Path, help="Optional manifest override.")
    parser.add_argument("--output-root", type=Path, help="Optional output root override.")
    parser.add_argument("--predicted-output-root", type=Path, help="Optional predicted-motion output root override.")
    parser.add_argument("--liveportrait-root", type=Path, help="Optional LivePortrait checkout override.")
    parser.add_argument("--inference-script", type=Path, help="Optional LivePortrait inference script override.")
    parser.add_argument("--python-executable", help="Optional Python executable override for LivePortrait.")
    parser.add_argument("--source", type=Path, help="Optional source portrait override for all rendered clips.")
    parser.add_argument("--clip-id", action="append", default=[], help="Optional clip filter. Repeat to select multiple.")
    parser.add_argument("--device", help="Optional predictor device override: auto, cpu, or cuda.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional argument to forward to LivePortrait.")
    return parser.parse_args()


def _get_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: object = None) -> object:
    value = getattr(args, name)
    if value is not None and value != []:
        return value
    return config.get(name, default)


def _get_required_path(args: argparse.Namespace, config: dict[str, Any], name: str) -> Path:
    value = _get_value(args, config, name)
    if value is None:
        raise ValueError(
            f"Missing required value for '{name}'. Provide it in {args.config} or pass --{name.replace('_', '-')}."
        )
    return Path(str(value)).expanduser().resolve()


def main() -> int:
    args = parse_args()
    config_data = load_config(args.config) if args.config else {}
    if not isinstance(config_data, dict):
        raise ValueError(f"Expected top-level config mapping in {args.config}")

    extra_args = tuple(args.extra_arg or config_data.get("extra_args", []))
    summary = render_predicted_motion_for_manifest(
        PredictedMotionRenderConfig(
            checkpoint_path=_get_required_path(args, config_data, "checkpoint"),
            model_config_path=_get_required_path(args, config_data, "model_config"),
            manifest_path=_get_required_path(args, config_data, "manifest_path"),
            output_root=_get_required_path(args, config_data, "output_root"),
            predicted_output_root=(
                Path(str(_get_value(args, config_data, "predicted_output_root"))).expanduser().resolve()
                if _get_value(args, config_data, "predicted_output_root") is not None
                else None
            ),
            liveportrait_root=_get_required_path(args, config_data, "liveportrait_root"),
            inference_script=(
                Path(str(_get_value(args, config_data, "inference_script"))).expanduser().resolve()
                if _get_value(args, config_data, "inference_script") is not None
                else None
            ),
            python_executable=str(_get_value(args, config_data, "python_executable", sys.executable)),
            source_path=(
                Path(str(_get_value(args, config_data, "source"))).expanduser().resolve()
                if _get_value(args, config_data, "source") is not None
                else None
            ),
            clip_ids=tuple(args.clip_id or config_data.get("clip_ids", [])),
            device=str(_get_value(args, config_data, "device", "auto")),
            extra_args=extra_args,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
