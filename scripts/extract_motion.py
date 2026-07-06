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

from avagen.renderers.motion_template import MotionTemplateExtractionConfig, extract_motion_templates
from avagen.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract LivePortrait motion templates for processed clips.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "extract_motion.yaml",
        help="YAML config file. Defaults to configs/extract_motion.yaml.",
    )
    parser.add_argument("--manifest-path", type=Path, help="Optional explicit manifest path override.")
    parser.add_argument("--liveportrait-root", type=Path, help="Path to the LivePortrait checkout.")
    parser.add_argument("--inference-script", type=Path, help="Optional explicit inference script path.")
    parser.add_argument("--python-executable", help="Python executable for the LivePortrait run.")
    parser.add_argument("--work-root", type=Path, help="Optional working directory for staged videos and outputs.")
    parser.add_argument("--output-field", help="Metadata field to update. Defaults to motion_template_path.")
    parser.add_argument(
        "--driving-source",
        choices=["source_video", "face_crop_video"],
        help="Which clip representation to feed into LivePortrait.",
    )
    parser.add_argument(
        "--skip-render",
        dest="skip_render",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop LivePortrait once the motion template is written, skipping the discarded render (default: on).",
    )
    parser.add_argument("--clip-id", action="append", default=[], help="Restrict extraction to one or more clip IDs.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional argument to forward.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing motion templates.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
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

    manifest_path = Path(_get_required_value(args, config_data, "manifest_path"))
    liveportrait_root = Path(_get_required_value(args, config_data, "liveportrait_root"))
    inference_script_value = _get_value(args, config_data, "inference_script")
    inference_script = Path(inference_script_value) if inference_script_value else None
    work_root_value = _get_value(args, config_data, "work_root")
    work_root = Path(work_root_value) if work_root_value else None
    output_field = str(_get_value(args, config_data, "output_field", "motion_template_path"))
    driving_source = str(_get_value(args, config_data, "driving_source", "source_video"))
    skip_render = bool(_get_value(args, config_data, "skip_render", True))
    clip_ids = tuple(args.clip_id or config_data.get("clip_ids", []) or [])
    extra_args = tuple(args.extra_arg or config_data.get("extra_args", []) or [])

    result = extract_motion_templates(
        MotionTemplateExtractionConfig(
            manifest_path=manifest_path.resolve(),
            liveportrait_root=liveportrait_root.resolve(),
            python_executable=str(_get_required_value(args, config_data, "python_executable")),
            inference_script=inference_script.resolve() if inference_script else None,
            work_root=work_root.resolve() if work_root else None,
            output_field=output_field,
            driving_source=driving_source,
            skip_render=skip_render,
            extra_args=extra_args,
            clip_ids=clip_ids,
            overwrite=args.overwrite,
        ),
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
