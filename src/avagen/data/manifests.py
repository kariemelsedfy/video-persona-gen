"""Manifest helpers are intentionally stubbed during the scaffold phase."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ManifestRecord:
    clip_id: str
    identity_id: str
    audio_path: str
    face_crop_dir: str
    landmarks_path: str | None
    head_pose_path: str | None
    expression_path: str | None
    motion_template_path: str | None
    fps: float
    duration_sec: float
    num_frames: int
    face_detection_rate: float
    avg_yaw_abs: float | None
    avg_pitch_abs: float | None
    avg_roll_abs: float | None
    audio_sample_rate: int | None
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def refresh_identity_manifest(
    identity_dir: str | Path,
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> tuple[list[ManifestRecord], Path, Path]:
    raise NotImplementedError("Scaffold only. Manifest generation will be implemented in the preprocessing phase.")
