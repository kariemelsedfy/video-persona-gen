"""Audio data helpers specific to dataset preparation."""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avagen.data.dataset import load_processed_clip_records
from avagen.data.manifests import refresh_identity_manifest
from avagen.features.audio_features import extract_audio_feature_bundle, save_audio_feature_bundle
from avagen.features.prosody import summarize_prosody_features
from avagen.utils.paths import to_repo_relative


@dataclass(frozen=True)
class AudioWaveform:
    samples: np.ndarray
    sample_rate: int


def load_wav_mono(input_path: str | Path) -> AudioWaveform:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw_bytes = handle.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV input, got sample width {sample_width} for {path}")

    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return AudioWaveform(samples=samples, sample_rate=sample_rate)


def frame_audio(
    samples: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive integers.")
    if samples.ndim != 1:
        raise ValueError("Expected mono audio samples.")

    if samples.size == 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    if samples.size < frame_length:
        padded = np.pad(samples, (0, frame_length - samples.size))
        return padded.reshape(1, frame_length).astype(np.float32)

    frame_count = 1 + int(np.ceil((samples.size - frame_length) / hop_length))
    target_length = frame_length + hop_length * (frame_count - 1)
    if target_length > samples.size:
        samples = np.pad(samples, (0, target_length - samples.size))

    frames = np.stack(
        [samples[index * hop_length : index * hop_length + frame_length] for index in range(frame_count)],
        axis=0,
    )
    return frames.astype(np.float32)


def extract_audio_features_for_manifest(
    manifest_path: str | Path,
    frame_length_ms: float = 40.0,
    hop_length_ms: float = 10.0,
    clip_ids: tuple[str, ...] = (),
    overwrite: bool = False,
) -> dict[str, object]:
    manifest = Path(manifest_path).expanduser().resolve()
    records = load_processed_clip_records(manifest)
    selected_clip_ids = set(clip_ids)
    processed_clips: list[dict[str, object]] = []

    for record in records:
        if selected_clip_ids and record.clip_id not in selected_clip_ids:
            continue

        clip_dir = record.audio_path.parent
        feature_path = clip_dir / "audio_features.npz"
        prosody_path = clip_dir / "prosody_summary.json"
        if feature_path.exists() and prosody_path.exists() and not overwrite:
            processed_clips.append(
                {
                    "clip_id": record.clip_id,
                    "identity_id": record.identity_id,
                    "status": "skipped_existing",
                    "audio_features_path": str(feature_path),
                    "prosody_summary_path": str(prosody_path),
                }
            )
            continue

        waveform = load_wav_mono(record.audio_path)
        bundle = extract_audio_feature_bundle(
            waveform.samples,
            sample_rate=waveform.sample_rate,
            frame_length_ms=frame_length_ms,
            hop_length_ms=hop_length_ms,
        )
        save_audio_feature_bundle(bundle, feature_path)

        prosody_summary = summarize_prosody_features(bundle)
        prosody_path.write_text(json.dumps(prosody_summary, indent=2, sort_keys=True), encoding="utf-8")

        metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
        optional_artifacts = metadata.setdefault("optional_artifacts", {})
        if not isinstance(optional_artifacts, dict):
            raise ValueError(f"Invalid optional_artifacts in {record.metadata_path}")
        optional_artifacts["audio_features_path"] = to_repo_relative(feature_path)
        optional_artifacts["prosody_summary_path"] = to_repo_relative(prosody_path)
        metadata["audio_features"] = {
            "frame_length_ms": frame_length_ms,
            "hop_length_ms": hop_length_ms,
            "audio_features_path": to_repo_relative(feature_path),
            "prosody_summary_path": to_repo_relative(prosody_path),
        }
        record.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        processed_clips.append(
            {
                "clip_id": record.clip_id,
                "identity_id": record.identity_id,
                "status": "completed",
                "audio_features_path": str(feature_path),
                "prosody_summary_path": str(prosody_path),
                "num_feature_frames": int(bundle["rms_energy"].shape[0]),
            }
        )

    identity_dir = manifest.parent
    refreshed_records, refreshed_manifest_path, report_path = refresh_identity_manifest(identity_dir)
    return {
        "manifest_path": str(refreshed_manifest_path),
        "report_path": str(report_path),
        "num_manifest_records": len(refreshed_records),
        "processed_clips": processed_clips,
        "status": "completed",
    }
