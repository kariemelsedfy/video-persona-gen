"""Audio feature extraction helpers built on NumPy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _frame_audio(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
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

    return np.stack(
        [samples[index * hop_length : index * hop_length + frame_length] for index in range(frame_count)],
        axis=0,
    ).astype(np.float32)


def extract_audio_feature_bundle(
    samples: np.ndarray,
    sample_rate: int,
    frame_length_ms: float = 40.0,
    hop_length_ms: float = 10.0,
) -> dict[str, Any]:
    frame_length = max(1, int(round(sample_rate * frame_length_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_length_ms / 1000.0)))
    frames = _frame_audio(samples.astype(np.float32), frame_length=frame_length, hop_length=hop_length)

    if frames.size == 0:
        spectral_centroid = np.zeros((0,), dtype=np.float32)
        rms_energy = np.zeros((0,), dtype=np.float32)
        zero_crossing_rate = np.zeros((0,), dtype=np.float32)
        peak_amplitude = np.zeros((0,), dtype=np.float32)
    else:
        rms_energy = np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)
        zero_crossing_rate = (
            np.mean(np.abs(np.diff(np.signbit(frames).astype(np.int8), axis=1)), axis=1).astype(np.float32)
        )
        peak_amplitude = np.max(np.abs(frames), axis=1).astype(np.float32)

        magnitudes = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
        frequency_bins = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate).astype(np.float32)
        weighted = magnitudes * frequency_bins[None, :]
        denominator = np.maximum(np.sum(magnitudes, axis=1), 1e-8)
        spectral_centroid = (np.sum(weighted, axis=1) / denominator).astype(np.float32)

    time_axis_sec = (
        np.arange(rms_energy.shape[0], dtype=np.float32) * (hop_length / float(sample_rate))
    ).astype(np.float32)

    return {
        "sample_rate": np.asarray(sample_rate, dtype=np.int32),
        "frame_length": np.asarray(frame_length, dtype=np.int32),
        "hop_length": np.asarray(hop_length, dtype=np.int32),
        "frame_length_ms": np.asarray(frame_length_ms, dtype=np.float32),
        "hop_length_ms": np.asarray(hop_length_ms, dtype=np.float32),
        "time_axis_sec": time_axis_sec,
        "rms_energy": rms_energy,
        "log_rms_energy": np.log(np.maximum(rms_energy, 1e-8)).astype(np.float32),
        "zero_crossing_rate": zero_crossing_rate,
        "peak_amplitude": peak_amplitude,
        "spectral_centroid_hz": spectral_centroid,
    }


def save_audio_feature_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bundle)
    return path
