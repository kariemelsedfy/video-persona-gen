"""Prosody summaries derived from extracted audio features."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_prosody_features(bundle: dict[str, Any]) -> dict[str, float | int]:
    rms_energy = np.asarray(bundle["rms_energy"], dtype=np.float32)
    log_rms_energy = np.asarray(bundle["log_rms_energy"], dtype=np.float32)
    zero_crossing_rate = np.asarray(bundle["zero_crossing_rate"], dtype=np.float32)
    spectral_centroid = np.asarray(bundle["spectral_centroid_hz"], dtype=np.float32)

    if rms_energy.size == 0:
        return {
            "num_frames": 0,
            "mean_rms_energy": 0.0,
            "std_rms_energy": 0.0,
            "mean_log_rms_energy": 0.0,
            "mean_zero_crossing_rate": 0.0,
            "mean_spectral_centroid_hz": 0.0,
            "active_frame_fraction": 0.0,
        }

    active_threshold = max(1e-4, float(rms_energy.mean()) * 0.5)
    active_frame_fraction = float(np.mean(rms_energy > active_threshold))
    return {
        "num_frames": int(rms_energy.size),
        "mean_rms_energy": float(rms_energy.mean()),
        "std_rms_energy": float(rms_energy.std()),
        "mean_log_rms_energy": float(log_rms_energy.mean()),
        "mean_zero_crossing_rate": float(zero_crossing_rate.mean()),
        "mean_spectral_centroid_hz": float(spectral_centroid.mean()),
        "active_frame_fraction": active_frame_fraction,
    }
