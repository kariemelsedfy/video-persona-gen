"""Dataset preprocessing helpers for single-clip ingestion."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from avagen.data.face_tracking import extract_face_crops
from avagen.utils.audio import extract_audio_to_wav
from avagen.utils.paths import ensure_dir, to_repo_relative
from avagen.utils.video import inspect_video


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


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def preprocess_clip(config: PreprocessConfig) -> dict[str, object]:
    input_path = config.input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")

    clip_dir = config.output_root.expanduser().resolve() / config.identity_id / config.clip_id
    if clip_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Clip directory already exists: {clip_dir}")
        shutil.rmtree(clip_dir)

    ensure_dir(clip_dir)
    face_crop_dir = ensure_dir(clip_dir / "face_crops")

    video_info = inspect_video(input_path)
    if not video_info.has_audio:
        raise RuntimeError(f"Input clip has no audio stream: {input_path}")

    audio_path = extract_audio_to_wav(
        input_path=input_path,
        output_path=clip_dir / "audio.wav",
        sample_rate=config.audio_sample_rate,
        channels=1,
        overwrite=True,
    )
    frame_records, effective_fps = extract_face_crops(
        input_path=input_path,
        output_dir=face_crop_dir,
        target_fps=config.target_fps,
        face_margin=config.face_margin,
        allow_center_crop_fallback=config.allow_center_crop_fallback,
    )
    if not frame_records:
        raise RuntimeError(f"No frames were extracted from {input_path}")

    frame_metadata_path = clip_dir / "frame_metadata.json"
    frame_payload = [record.to_dict() for record in frame_records]
    frame_metadata_path.write_text(json.dumps(frame_payload, indent=2, sort_keys=True), encoding="utf-8")

    detector_hits = sum(1 for record in frame_records if record.detector_hit)
    previous_box_fallback_frames = sum(1 for record in frame_records if record.fallback_source == "previous_box")
    center_crop_fallback_frames = sum(1 for record in frame_records if record.fallback_source == "center_crop")
    duration_sec = frame_records[-1].timestamp_sec + (1.0 / effective_fps if effective_fps > 0 else 0.0)

    metadata = {
        "identity_id": config.identity_id,
        "clip_id": config.clip_id,
        "split": config.split,
        "source_video_path": str(input_path),
        "artifacts": {
            "audio_path": to_repo_relative(audio_path),
            "face_crop_dir": to_repo_relative(face_crop_dir),
            "frame_metadata_path": to_repo_relative(frame_metadata_path),
        },
        "video_info": video_info.to_dict(),
        "preprocessing": {
            "target_fps_requested": config.target_fps,
            "target_fps_effective": effective_fps,
            "audio_sample_rate": config.audio_sample_rate,
            "face_margin": config.face_margin,
            "allow_center_crop_fallback": config.allow_center_crop_fallback,
        },
        "stats": {
            "num_frames": len(frame_records),
            "duration_sec": duration_sec,
            "face_detection_rate": detector_hits / len(frame_records),
            "detector_hits": detector_hits,
            "previous_box_fallback_frames": previous_box_fallback_frames,
            "center_crop_fallback_frames": center_crop_fallback_frames,
        },
        "optional_artifacts": {
            "landmarks_path": None,
            "head_pose_path": None,
            "expression_path": None,
            "motion_template_path": None,
        },
    }
    metadata_path = _write_json(clip_dir / "metadata.json", metadata)

    return {
        "identity_id": config.identity_id,
        "clip_id": config.clip_id,
        "clip_dir": str(clip_dir),
        "metadata_path": str(metadata_path),
        "audio_path": str(audio_path),
        "face_crop_dir": str(face_crop_dir),
        "num_frames": len(frame_records),
        "duration_sec": duration_sec,
        "face_detection_rate": detector_hits / len(frame_records),
        "effective_fps": effective_fps,
    }
