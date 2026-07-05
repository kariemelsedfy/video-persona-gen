"""Video inspection helpers built on ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
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


def _parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _load_ffprobe_json(video_path: Path) -> dict[str, object]:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        raise RuntimeError("ffprobe is required to inspect videos.")

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed for {video_path}: {stderr}")
    return json.loads(result.stdout or "{}")


def inspect_video(video_path: str | Path) -> VideoInfo:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    payload = _load_ffprobe_json(path)
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})

    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)

    width = int(video_stream["width"]) if video_stream and video_stream.get("width") is not None else None
    height = int(video_stream["height"]) if video_stream and video_stream.get("height") is not None else None
    fps = None
    if video_stream:
        fps = _parse_fraction(video_stream.get("avg_frame_rate")) or _parse_fraction(video_stream.get("r_frame_rate"))

    duration_sec = None
    if video_stream and video_stream.get("duration") not in {None, "N/A"}:
        duration_sec = float(video_stream["duration"])
    elif format_info.get("duration") not in {None, "N/A"}:
        duration_sec = float(format_info["duration"])

    num_frames = None
    if video_stream and video_stream.get("nb_frames") not in {None, "N/A"}:
        try:
            num_frames = int(video_stream["nb_frames"])
        except ValueError:
            num_frames = None
    if num_frames is None and duration_sec is not None and fps is not None:
        num_frames = int(round(duration_sec * fps))

    audio_sample_rate = None
    if audio_stream and audio_stream.get("sample_rate") not in {None, "N/A"}:
        try:
            audio_sample_rate = int(audio_stream["sample_rate"])
        except ValueError:
            audio_sample_rate = None

    return VideoInfo(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
        duration_sec=duration_sec,
        has_audio=audio_stream is not None,
        audio_sample_rate=audio_sample_rate,
        video_codec=video_stream.get("codec_name") if video_stream else None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
    )
