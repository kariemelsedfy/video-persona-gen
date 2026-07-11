from __future__ import annotations

import numpy as np

from avagen.features.wav2vec_features import resample_to_time_grid


def test_resample_maps_onto_target_grid_and_interpolates() -> None:
    # 3 source frames at t=0,1,2; single feature column = [0,10,20]
    src = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)
    src_t = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    tgt_t = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    out = resample_to_time_grid(src, src_t, tgt_t)
    assert out.shape == (5, 1)
    np.testing.assert_allclose(out[:, 0], [0.0, 5.0, 10.0, 15.0, 20.0], atol=1e-5)


def test_resample_multicolumn_shape() -> None:
    src = np.random.default_rng(0).normal(size=(50, 768)).astype(np.float32)
    src_t = np.linspace(0, 2, 50).astype(np.float32)
    tgt_t = np.linspace(0, 2, 120).astype(np.float32)
    out = resample_to_time_grid(src, src_t, tgt_t)
    assert out.shape == (120, 768)


def test_resample_single_frame_repeats() -> None:
    src = np.array([[1.0, 2.0]], dtype=np.float32)
    out = resample_to_time_grid(src, np.array([0.0], dtype=np.float32), np.array([0.0, 0.1, 0.2], dtype=np.float32))
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float32))
