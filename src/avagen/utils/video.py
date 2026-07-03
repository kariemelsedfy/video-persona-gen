"""Video inspection helpers are intentionally stubbed during the scaffold phase."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    path: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    num_frames: int | None = None
    duration_sec: float | None = None
    has_audio: bool | None = None
    audio_sample_rate: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_video(video_path: str | Path) -> VideoInfo:
    raise NotImplementedError("Scaffold only. Video inspection will be implemented in the preprocessing phase.")
