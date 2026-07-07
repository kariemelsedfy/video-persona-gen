"""Render-time post-processing of predicted motion (scaling + blinks).

MSE-trained regression damps motion amplitude, so the mouth/expression is often
correct in timing but too subtle. Per-component scaling amplifies the *deviation*
from each component's temporal mean (so the resting pose is preserved, only the
motion is boosted). Eye blinks are not predictable from audio, so they are
injected as a separate synthetic signal.
"""

from __future__ import annotations

import numpy as np


def scale_component_deviation(values: np.ndarray, scale: float) -> np.ndarray:
    """Amplify motion around the temporal mean: mean + scale * (x - mean)."""
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] == 0 or scale == 1.0:
        return array
    mean = array.mean(axis=0, keepdims=True)
    return (mean + float(scale) * (array - mean)).astype(np.float32)


def generate_blink_signal(
    n_frames: int,
    fps: float,
    open_value: float,
    *,
    closed_value: float = 0.02,
    interval_sec: float = 3.5,
    jitter_sec: float = 1.5,
    blink_ms: float = 180.0,
    seed: int = 0,
) -> np.ndarray:
    """A per-frame eye-open signal that sits at open_value with periodic blinks."""
    signal = np.full(int(n_frames), float(open_value), dtype=np.float32)
    if n_frames <= 0 or fps <= 0:
        return signal
    rng = np.random.default_rng(seed)
    half = max(1, int(round((blink_ms / 1000.0) * fps / 2.0)))
    sigma = max(half / 1.5, 1e-3)
    duration = n_frames / fps
    t = float(rng.uniform(0.5, max(interval_sec, 0.6)))
    while t < duration:
        center = int(round(t * fps))
        for k in range(-half, half + 1):
            idx = center + k
            if 0 <= idx < n_frames:
                weight = float(np.exp(-(k * k) / (2.0 * sigma * sigma)))
                dipped = open_value - (open_value - closed_value) * weight
                signal[idx] = min(float(signal[idx]), dipped)
        t += interval_sec + float(rng.uniform(-jitter_sec, jitter_sec))
    return signal


# LivePortrait expression keypoints that control the eyelids (21-keypoint layout).
EYE_EXPRESSION_KEYPOINTS = (11, 13, 15, 16, 18)


def inject_expression_blinks(
    expression: np.ndarray,
    reference_expression: np.ndarray,
    reference_eye_ratio: np.ndarray,
    fps: float,
    *,
    closed_percentile: float = 12.0,
    open_percentile: float = 70.0,
    interval_sec: float = 3.5,
    jitter_sec: float = 1.5,
    blink_ms: float = 180.0,
    seed: int = 0,
) -> np.ndarray:
    """Overwrite the eyelid expression keypoints with synthetic blinks.

    Audio can't predict blinks, so we animate the eyes on the normal expression
    path (NOT LivePortrait's eye-retargeting flag, which suppresses base motion).
    The open and closed eyelid poses are learned from the clip's own ground truth
    (frames with the highest / lowest measured eye openness), so blinks match the
    person's real eye shape.
    """
    expression = np.asarray(expression, dtype=np.float32).copy()  # (T, 21, 3)
    ref_exp = np.asarray(reference_expression, dtype=np.float32)  # (Tg, 21, 3)
    eye = np.asarray(reference_eye_ratio, dtype=np.float32)
    if expression.ndim != 3 or ref_exp.ndim != 3 or expression.shape[0] == 0:
        return expression
    openness = eye.mean(axis=1) if eye.ndim == 2 else eye
    if openness.shape[0] != ref_exp.shape[0]:
        return expression
    low = np.percentile(openness, closed_percentile)
    high = np.percentile(openness, open_percentile)
    closed_frames = openness <= low
    open_frames = openness >= high
    if not closed_frames.any() or not open_frames.any():
        return expression

    idx = list(EYE_EXPRESSION_KEYPOINTS)
    closed_pose = ref_exp[closed_frames][:, idx, :].mean(axis=0)  # (5, 3)
    open_pose = ref_exp[open_frames][:, idx, :].mean(axis=0)  # (5, 3)

    base = generate_blink_signal(
        expression.shape[0], fps, open_value=1.0, closed_value=0.0,
        interval_sec=interval_sec, jitter_sec=jitter_sec, blink_ms=blink_ms, seed=seed,
    )  # 1 = open, dips to 0 during blinks
    weight = (1.0 - base)[:, None, None]  # (T,1,1): 0 open, 1 fully closed
    expression[:, idx, :] = open_pose[None] + weight * (closed_pose - open_pose)[None]
    return expression


def apply_motion_postprocess(
    bundle: dict, config: dict, fps: float, reference_bundle: dict | None = None
) -> dict:
    """Apply per-component scaling and blink injection to a predicted bundle."""
    if not config:
        return bundle
    component_scales = {
        "expression": float(config.get("expression_scale", 1.0)),
        "rotation_matrix": float(config.get("rotation_scale", 1.0)),
        "translation": float(config.get("translation_scale", 1.0)),
        "lip_ratio": float(config.get("lip_scale", 1.0)),
    }
    for name, scale in component_scales.items():
        if scale != 1.0 and name in bundle:
            bundle[name] = scale_component_deviation(bundle[name], scale)

    if (
        config.get("add_blinks", False)
        and "expression" in bundle
        and reference_bundle is not None
        and "expression" in reference_bundle
        and "eye_ratio" in reference_bundle
    ):
        bundle["expression"] = inject_expression_blinks(
            bundle["expression"],
            reference_bundle["expression"],
            reference_bundle["eye_ratio"],
            fps=fps,
            interval_sec=float(config.get("blink_interval_sec", 3.5)),
            jitter_sec=float(config.get("blink_jitter_sec", 1.5)),
            blink_ms=float(config.get("blink_ms", 180.0)),
            seed=int(config.get("blink_seed", 0)),
        )
    return bundle
