from __future__ import annotations

import numpy as np

from avagen.features.audio_features import (
    _mel_filterbank,
    extract_audio_feature_bundle,
)


def test_mel_filterbank_shape_and_nonnegative() -> None:
    fb = _mel_filterbank(frame_length=640, sample_rate=16000, n_mels=40)
    assert fb.shape[0] == 40
    assert fb.min() >= 0.0
    # each mel filter should have some energy
    assert np.all(fb.sum(axis=1) > 0)


def test_bundle_includes_log_mel_aligned_to_frames() -> None:
    rng = np.random.default_rng(0)
    sample_rate = 16000
    samples = rng.normal(size=sample_rate).astype(np.float32)  # 1 second
    bundle = extract_audio_feature_bundle(samples, sample_rate=sample_rate, n_mels=40)
    log_mel = bundle["log_mel"]
    assert log_mel.ndim == 2
    assert log_mel.shape[1] == 40
    # same number of frames as the scalar features (aligned time axis)
    assert log_mel.shape[0] == bundle["rms_energy"].shape[0]
    assert log_mel.shape[0] == bundle["time_axis_sec"].shape[0]
    assert np.all(np.isfinite(log_mel))


def test_empty_audio_yields_empty_log_mel() -> None:
    bundle = extract_audio_feature_bundle(np.zeros((0,), dtype=np.float32), sample_rate=16000, n_mels=40)
    assert bundle["log_mel"].shape == (0, 40)
