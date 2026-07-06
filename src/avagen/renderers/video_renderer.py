"""Renderer orchestration helpers for model-generated motion."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from avagen.data.dataset import ProcessedClipRecord, load_processed_clip_records
from avagen.inference.generate_motion import predict_motion_for_manifest
from avagen.renderers.liveportrait_wrapper import (
    LivePortraitRunConfig,
    run_liveportrait_inference,
)


def _ffprobe_value(ffprobe: str, args: list[str]) -> str | None:
    result = subprocess.run([ffprobe, "-v", "error", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _mux_rendered_audio(output_dir: Path, audio_path: Path | None) -> list[str]:
    """Mux the driving clip's audio onto rendered videos.

    Renders come out silent (a static source image driven by a motion template),
    which makes lip-sync impossible to judge. The rendered frame count spans the
    audio duration, so we reinterpret the video's frame rate to match the audio
    exactly (correcting the template's rounded output_fps) and attach the track.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None or audio_path is None or not Path(audio_path).exists():
        return []

    duration = _ffprobe_value(
        ffprobe, ["-show_entries", "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", str(audio_path)]
    )
    if duration is None:
        return []
    audio_duration = float(duration)
    if audio_duration <= 0:
        return []

    produced: list[str] = []
    for video in sorted(output_dir.glob("*.mp4")):
        if video.stem.endswith("_with_audio") or "concat" in video.stem:
            continue
        frames = _ffprobe_value(
            ffprobe,
            [
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video),
            ],
        )
        if frames is None or not frames.isdigit() or int(frames) <= 0:
            continue
        fps = int(frames) / audio_duration
        out_path = video.with_name(f"{video.stem}_with_audio.mp4")
        result = subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-r", f"{fps:.6f}", "-i", str(video),
                "-i", str(audio_path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            produced.append(str(out_path.resolve()))
    return produced


@dataclass(frozen=True)
class PredictedMotionRenderConfig:
    checkpoint_path: Path
    model_config_path: Path
    manifest_path: Path
    output_root: Path
    liveportrait_root: Path
    inference_script: Path | None = None
    python_executable: str = "python"
    source_path: Path | None = None
    predicted_output_root: Path | None = None
    clip_ids: tuple[str, ...] = ()
    device: str = "auto"
    mux_audio: bool = True
    extra_args: Sequence[str] = field(default_factory=tuple)


def _resolve_source_image(record: ProcessedClipRecord, override_source_path: Path | None) -> Path:
    if override_source_path is not None:
        resolved = override_source_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Source image not found: {resolved}")
        return resolved

    images = sorted(record.face_crop_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No face crop images found for clip {record.clip_id} in {record.face_crop_dir}")
    return images[0].resolve()


def render_predicted_motion_for_manifest(config: PredictedMotionRenderConfig) -> dict[str, Any]:
    manifest_path = config.manifest_path.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    predicted_output_root = (
        config.predicted_output_root.expanduser().resolve()
        if config.predicted_output_root is not None
        else (output_root / "predicted_motion").resolve()
    )
    render_output_root = (output_root / "renders").resolve()

    prediction_summary = predict_motion_for_manifest(
        checkpoint_path=config.checkpoint_path,
        config_path=config.model_config_path,
        manifest_path=manifest_path,
        output_root=predicted_output_root,
        clip_ids=config.clip_ids,
        device=config.device,
    )

    records = load_processed_clip_records(manifest_path)
    record_by_clip_id = {record.clip_id: record for record in records}
    rendered_records: list[dict[str, Any]] = []
    for predicted_record in prediction_summary["predicted_records"]:
        clip_id = str(predicted_record["clip_id"])
        record = record_by_clip_id.get(clip_id)
        if record is None:
            raise KeyError(f"Predicted clip {clip_id} was not found in manifest {manifest_path}")

        source_path = _resolve_source_image(record, config.source_path)
        output_dir = render_output_root / record.identity_id / record.clip_id
        run_result = run_liveportrait_inference(
            LivePortraitRunConfig(
                source_path=source_path,
                driving_path=Path(str(predicted_record["predicted_motion_template_path"])).expanduser().resolve(),
                output_dir=output_dir,
                liveportrait_root=config.liveportrait_root.expanduser().resolve(),
                inference_script=config.inference_script if config.inference_script else None,
                python_executable=config.python_executable,
                extra_args=config.extra_args,
            )
        )
        audio_muxed_files = (
            _mux_rendered_audio(output_dir, record.audio_path) if config.mux_audio else []
        )
        rendered_records.append(
            {
                "clip_id": record.clip_id,
                "identity_id": record.identity_id,
                "source_path": str(source_path),
                "driving_template_path": str(predicted_record["predicted_motion_template_path"]),
                "output_dir": str(output_dir.resolve()),
                "rendered_files": sorted(str(path.resolve()) for path in output_dir.glob("*")),
                "audio_muxed_files": audio_muxed_files,
                "command": run_result.command,
            }
        )

    return {
        "status": "completed",
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "predicted_output_root": str(predicted_output_root),
        "render_output_root": str(render_output_root),
        "prediction_summary": prediction_summary,
        "rendered_records": rendered_records,
    }
