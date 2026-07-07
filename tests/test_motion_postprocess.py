from __future__ import annotations

import numpy as np

from avagen.features.motion_postprocess import (
    EYE_EXPRESSION_KEYPOINTS,
    apply_motion_postprocess,
    generate_blink_signal,
    inject_expression_blinks,
    scale_component_deviation,
)


def test_scale_preserves_mean_and_amplifies_deviation() -> None:
    values = np.array([[1.0], [3.0], [2.0]], dtype=np.float32)  # mean 2.0
    scaled = scale_component_deviation(values, 2.0)
    np.testing.assert_allclose(scaled.mean(axis=0), values.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(scaled[:, 0], [0.0, 4.0, 2.0], atol=1e-5)


def test_scale_identity_when_one() -> None:
    values = np.random.default_rng(0).normal(size=(10, 5)).astype(np.float32)
    np.testing.assert_allclose(scale_component_deviation(values, 1.0), values)


def test_blink_signal_sits_open_with_dips() -> None:
    fps = 25.0
    sig = generate_blink_signal(int(10 * fps), fps, open_value=1.0, closed_value=0.0, interval_sec=3.0, seed=1)
    assert sig.shape == (250,)
    assert np.median(sig) > 0.95  # mostly open
    assert (sig < 0.3).sum() >= 2  # a few blink dips
    assert sig.min() < 0.1


def test_expression_blinks_only_touch_eye_keypoints_using_gt_pose() -> None:
    rng = np.random.default_rng(0)
    n, ng = 250, 300
    pred_exp = rng.normal(size=(n, 21, 3)).astype(np.float32)
    original = pred_exp.copy()
    gt_exp = rng.normal(size=(ng, 21, 3)).astype(np.float32)
    # ground-truth eye openness: mostly open, some closed frames
    gt_eye = np.full((ng, 2), 0.3, dtype=np.float32)
    gt_eye[:30] = 0.02  # closed frames
    # give the closed GT frames a distinctive eyelid pose we expect to see copied
    gt_exp[:30][:, list(EYE_EXPRESSION_KEYPOINTS), :] = -5.0

    out = inject_expression_blinks(pred_exp.copy(), gt_exp, gt_eye, fps=25.0, interval_sec=2.0, seed=1)

    non_eye = [i for i in range(21) if i not in EYE_EXPRESSION_KEYPOINTS]
    # non-eye keypoints are untouched
    np.testing.assert_allclose(out[:, non_eye, :], original[:, non_eye, :])
    # eye keypoints changed (blinks injected)
    assert not np.allclose(out[:, list(EYE_EXPRESSION_KEYPOINTS), :], original[:, list(EYE_EXPRESSION_KEYPOINTS), :])
    # during a blink, the eyelid pose moves toward the GT closed pose (negative here)
    assert out[:, list(EYE_EXPRESSION_KEYPOINTS), :].min() < -1.0


def test_apply_postprocess_scales_expression_and_blinks_via_reference() -> None:
    rng = np.random.default_rng(1)
    bundle = {"expression": rng.normal(size=(60, 21, 3)).astype(np.float32)}
    ref = {
        "expression": rng.normal(size=(80, 21, 3)).astype(np.float32),
        "eye_ratio": np.concatenate(
            [np.full((70, 2), 0.3, dtype=np.float32), np.full((10, 2), 0.02, dtype=np.float32)]
        ),
    }
    non_eye = [i for i in range(21) if i not in EYE_EXPRESSION_KEYPOINTS]
    original_non_eye_std = bundle["expression"][:, non_eye, :].std()
    out = apply_motion_postprocess(
        {"expression": bundle["expression"].copy()},
        {"expression_scale": 2.0, "add_blinks": True, "blink_interval_sec": 2.0},
        fps=25.0,
        reference_bundle=ref,
    )
    # non-eye expression amplified
    assert out["expression"][:, non_eye, :].std() > original_non_eye_std
