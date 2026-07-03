"""Audio extraction helpers are intentionally stubbed during the scaffold phase."""

from __future__ import annotations

from pathlib import Path


def extract_audio_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = True,
) -> Path:
    raise NotImplementedError("Scaffold only. Audio extraction will be implemented in the preprocessing phase.")
