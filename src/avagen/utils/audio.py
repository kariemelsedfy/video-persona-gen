"""Audio extraction helpers built on ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_audio_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = True,
) -> Path:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is required to extract audio.")

    source_path = Path(input_path).expanduser().resolve()
    target_path = Path(output_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        return target_path

    command = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-i",
        str(source_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(target_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"ffmpeg audio extraction failed for {source_path}: {stderr}")
    return target_path
