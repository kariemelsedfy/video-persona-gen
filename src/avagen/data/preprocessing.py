"""Dataset preprocessing helpers are intentionally stubbed during the scaffold phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreprocessConfig:
    input_path: Path
    identity_id: str
    clip_id: str
    output_root: Path
    target_fps: float | None = 25.0
    audio_sample_rate: int = 16000
    split: str = "train"
    face_margin: float = 0.2
    allow_center_crop_fallback: bool = True
    overwrite: bool = False


def preprocess_clip(config: PreprocessConfig) -> dict[str, object]:
    raise NotImplementedError("Scaffold only. Clip preprocessing will be implemented in the preprocessing phase.")
