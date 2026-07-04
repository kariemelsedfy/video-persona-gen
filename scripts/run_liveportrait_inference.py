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

from avagen.renderers.liveportrait_wrapper import LivePortraitRunConfig, run_liveportrait_inference
from avagen.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LivePortrait inference through a thin wrapper.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "render_liveportrait.yaml",
        help="YAML config file. Defaults to configs/render_liveportrait.yaml.",
    )
    parser.add_argument("--liveportrait-root", type=Path, help="Path to the LivePortrait checkout.")
    parser.add_argument("--inference-script", type=Path, help="Optional explicit inference script path.")
    parser.add_argument("--source", type=Path, required=True, help="Source portrait image.")
    parser.add_argument("--driving", type=Path, required=True, help="Driving video.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for generated assets.")
    parser.add_argument("--python-executable", default=None, help="Python executable for the LivePortrait run.")
    parser.add_argument("--source-flag", default=None, help="CLI flag name for the source image argument.")
    parser.add_argument("--driving-flag", default=None, help="CLI flag name for the driving video argument.")
    parser.add_argument("--output-flag", default=None, help="CLI flag name for the output directory argument.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional argument to forward.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    return parser.parse_args()


def _get_value(args: argparse.Namespace, config: dict[str, object], name: str, default: object = None) -> object:
    value = getattr(args, name)
    if value is not None:
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

    liveportrait_root = Path(_get_required_value(args, config_data, "liveportrait_root"))
    inference_script_value = _get_value(args, config_data, "inference_script")
    inference_script = Path(inference_script_value) if inference_script_value else None
    source_flag = _get_value(args, config_data, "source_flag", "-s")
    driving_flag = _get_value(args, config_data, "driving_flag", "-d")
    output_flag = _get_value(args, config_data, "output_flag", "-o")
    extra_args = tuple(args.extra_arg or config_data.get("extra_args", []))

    result = run_liveportrait_inference(
        LivePortraitRunConfig(
            source_path=args.source.resolve(),
            driving_path=args.driving.resolve(),
            output_dir=args.output_dir.resolve(),
            liveportrait_root=liveportrait_root.resolve(),
            inference_script=inference_script.resolve() if inference_script else None,
            python_executable=str(_get_value(args, config_data, "python_executable", sys.executable)),
            source_flag=str(source_flag),
            driving_flag=str(driving_flag),
            output_flag=str(output_flag),
            extra_args=extra_args,
        ),
        dry_run=args.dry_run,
    )

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
