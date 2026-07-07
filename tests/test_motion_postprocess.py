from __future__ import annotations

import numpy as np

from avagen.features.motion_postprocess import (
    apply_motion_postprocess,
    generate_blink_signal,
    inject_blinks,
    scale_component_deviation,
)


def test_scale_preserves_mean_and_amplifies_deviation() -> None:
    values = np.array([[1.0], [3.0], [2.0]], dtype=np.float32)  # mean 2.0
    scaled = scale_component_deviation(values, 2.0)
    np.testing.assert_allclose(scaled.mean(axis=0), values.mean(axis=0), atol=1e-6)
    # deviations doubled: 1->0, 3->4, 2->2  (mean + 2*(x-mean))
    np.testing.assert_allclose(scaled[:, 0], [0.0, 4.0, 2.0], atol=1e-5)


def test_scale_identity_when_one() -> None:
    values = np.random.default_rng(0).normal(size=(10, 5)).astype(np.float32)
    np.testing.assert_allclose(scale_component_deviation(values, 1.0), values)


def test_blink_signal_sits_open_with_dips() -> None:
    fps = 25.0
    sig = generate_blink_signal(int(10 * fps), fps, open_value=1.0, closed_value=0.0, interval_sec=3.0, seed=1)
    assert sig.shape == (250,)
    # mostly open (near 1.0)
    assert np.median(sig) > 0.95
    # at least a couple of blinks dipped well below open
    assert (sig < 0.3).sum() >= 2
    assert sig.min() < 0.1


def test_inject_blinks_uses_per_eye_open_baseline() -> None:
    eye = np.full((250, 2), 0.31, dtype=np.float32)
    out = inject_blinks(eye, fps=25.0, interval_sec=3.0, seed=1)
    assert out.shape == (250, 2)
    # resting openness preserved (open frames near 0.31)
    assert abs(np.median(out) - 0.31) < 0.02
    # blinks dip toward closed
    assert out.min() < 0.1


def test_apply_postprocess_scales_expression_and_injects_blinks() -> None:
    bundle = {
        "expression": np.random.default_rng(0).normal(size=(50, 21, 3)).astype(np.float32),
        "eye_ratio": np.full((50, 2), 0.3, dtype=np.float32),
    }
    original_exp_std = bundle["expression"].std()
    out = apply_motion_postprocess(
        {k: v.copy() for k, v in bundle.items()},
        {"expression_scale": 2.0, "add_blinks": True, "blink_interval_sec": 2.0},
        fps=25.0,
    )
    assert out["expression"].std() > original_exp_std  # amplified
    assert out["eye_ratio"].min() < 0.1  # blinks present
