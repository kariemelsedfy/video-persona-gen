from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np

from avagen.data.audio import extract_audio_features_for_manifest, load_wav_mono
from avagen.data.dataset import load_audio_features, load_processed_clip_records


def _write_sine_wave(path: Path, sample_rate: int = 16000, duration_sec: float = 0.2, frequency_hz: float = 220.0) -> None:
    frame_count = int(sample_rate * duration_sec)
    samples = []
    for index in range(frame_count):
        value = math.sin(2.0 * math.pi * frequency_hz * (index / sample_rate))
        samples.append(int(max(-1.0, min(1.0, value)) * 32767))

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def test_extract_audio_features_for_manifest_updates_metadata_and_manifest(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed" / "demo_id"
    clip_dir = processed_root / "clip_a"
    face_crop_dir = clip_dir / "face_crops"
    face_crop_dir.mkdir(parents=True)
    (face_crop_dir / "000000.png").write_text("png", encoding="utf-8")
    (clip_dir / "frame_metadata.json").write_text(json.dumps([{"frame_index": 0}]), encoding="utf-8")

    audio_path = clip_dir / "audio.wav"
    _write_sine_wave(audio_path)
    metadata = {
        "identity_id": "demo_id",
        "clip_id": "clip_a",
        "split": "train",
        "source_video_path": str(tmp_path / "raw" / "clip.mp4"),
        "artifacts": {
            "audio_path": str(audio_path),
            "face_crop_dir": str(face_crop_dir),
            "frame_metadata_path": str(clip_dir / "frame_metadata.json"),
        },
        "video_info": {"fps": 25.0},
        "preprocessing": {"target_fps_effective": 25.0, "audio_sample_rate": 16000},
        "stats": {"num_frames": 5, "duration_sec": 0.2, "face_detection_rate": 1.0},
        "optional_artifacts": {
            "motion_template_path": None,
            "audio_features_path": None,
            "prosody_summary_path": None,
        },
    }
    (clip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    manifest_path = processed_root / "manifest.jsonl"
    manifest_record = {
        "clip_id": "clip_a",
        "identity_id": "demo_id",
        "audio_path": str(audio_path),
        "face_crop_dir": str(face_crop_dir),
        "landmarks_path": None,
        "head_pose_path": None,
        "expression_path": None,
        "motion_template_path": None,
        "fps": 25.0,
        "duration_sec": 0.2,
        "num_frames": 5,
        "face_detection_rate": 1.0,
        "avg_yaw_abs": None,
        "avg_pitch_abs": None,
        "avg_roll_abs": None,
        "audio_sample_rate": 16000,
        "split": "train",
    }
    manifest_path.write_text(json.dumps(manifest_record) + "\n", encoding="utf-8")

    waveform = load_wav_mono(audio_path)
    assert waveform.sample_rate == 16000

    result = extract_audio_features_for_manifest(manifest_path, overwrite=True)
    assert result["status"] == "completed"

    records = load_processed_clip_records(manifest_path)
    features = load_audio_features(records[0])
    assert "rms_energy" in features
    assert int(features["sample_rate"]) == 16000
    assert features["rms_energy"].shape[0] > 0

    updated_metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
    assert updated_metadata["optional_artifacts"]["audio_features_path"].endswith("audio_features.npz")
    assert updated_metadata["optional_artifacts"]["prosody_summary_path"].endswith("prosody_summary.json")
