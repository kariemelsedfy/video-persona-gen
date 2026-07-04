"""Thin wrapper around an external official LivePortrait checkout."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from avagen.utils.paths import ensure_dir


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


@dataclass
class LivePortraitRunResult:
    status: str
    command: list[str]
    cwd: str
    output_dir: str
    resolved_inference_script: str
    returncode: int | None = None
    dry_run: bool = False
    stdout: str | None = None
    stderr: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_inference_script(config: LivePortraitRunConfig) -> Path:
    if config.inference_script is not None:
        script_path = config.inference_script
        if not script_path.is_absolute():
            script_path = config.liveportrait_root / script_path
        script_path = script_path.resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"LivePortrait inference script not found: {script_path}")
        return script_path

    candidate = (config.liveportrait_root / "inference.py").resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find inference.py under the LivePortrait checkout. "
        "Pass --inference-script explicitly if your layout differs."
    )


def build_liveportrait_command(config: LivePortraitRunConfig) -> list[str]:
    resolved_script = resolve_inference_script(config)
    ensure_dir(config.output_dir.resolve())

    return [
        config.python_executable,
        str(resolved_script),
        config.source_flag,
        str(config.source_path.resolve()),
        config.driving_flag,
        str(config.driving_path.resolve()),
        config.output_flag,
        str(config.output_dir.resolve()),
        *config.extra_args,
    ]


def run_liveportrait_inference(
    config: LivePortraitRunConfig,
    dry_run: bool = False,
) -> LivePortraitRunResult:
    liveportrait_root = config.liveportrait_root.resolve()
    source_path = config.source_path.resolve()
    driving_path = config.driving_path.resolve()

    if not liveportrait_root.exists():
        raise FileNotFoundError(f"LivePortrait root not found: {liveportrait_root}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source input not found: {source_path}")
    if not driving_path.exists():
        raise FileNotFoundError(f"Driving input not found: {driving_path}")

    command = build_liveportrait_command(config)
    resolved_script = resolve_inference_script(config)
    result = LivePortraitRunResult(
        status="dry_run" if dry_run else "pending",
        command=command,
        cwd=str(liveportrait_root),
        output_dir=str(config.output_dir.resolve()),
        resolved_inference_script=str(resolved_script),
        dry_run=dry_run,
    )

    if dry_run:
        return result

    completed = subprocess.run(
        command,
        cwd=str(liveportrait_root),
        text=True,
        capture_output=True,
        check=False,
    )
    result.returncode = completed.returncode
    result.stdout = completed.stdout or None
    result.stderr = completed.stderr or None
    result.status = "completed" if completed.returncode == 0 else "failed"

    if completed.returncode != 0:
        stderr_text = completed.stderr.strip() if completed.stderr else "no stderr captured"
        raise RuntimeError(
            f"LivePortrait inference failed with exit code {completed.returncode}: {stderr_text}"
        )

    return result
