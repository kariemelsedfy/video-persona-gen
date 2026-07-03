"""LivePortrait wrapper is intentionally stubbed during the scaffold phase."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class LivePortraitRunConfig:
    source_path: Path
    driving_path: Path
    output_dir: Path
    liveportrait_root: Path
    inference_script: Path | None = None
    python_executable: str = sys.executable
    source_flag: str = "-s"
    driving_flag: str = "-d"
    output_flag: str = "-o"
    extra_args: Sequence[str] = field(default_factory=tuple)


def run_liveportrait_inference(
    config: LivePortraitRunConfig,
    dry_run: bool = False,
) -> dict[str, object]:
    return {
        "status": "skeleton",
        "dry_run": dry_run,
        "liveportrait_root": str(config.liveportrait_root),
        "source_path": str(config.source_path),
        "driving_path": str(config.driving_path),
        "output_dir": str(config.output_dir),
        "inference_script": str(config.inference_script) if config.inference_script else None,
        "python_executable": config.python_executable,
        "source_flag": config.source_flag,
        "driving_flag": config.driving_flag,
        "output_flag": config.output_flag,
        "extra_args": list(config.extra_args),
    }
