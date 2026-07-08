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


def _smooth_signal(n_frames: int, fps: float, amplitude: float, rng, n_components: int = 3,
                   min_period: float = 3.0, max_period: float = 11.0) -> np.ndarray:
    """Slow, smooth, natural-looking 1D signal (sum of low-frequency sinusoids)."""
    if n_frames <= 0 or amplitude == 0.0:
        return np.zeros(max(n_frames, 0), dtype=np.float32)
    t = np.arange(n_frames, dtype=np.float64) / max(fps, 1e-6)
    sig = np.zeros(n_frames, dtype=np.float64)
    for _ in range(n_components):
        period = rng.uniform(min_period, max_period)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        weight = rng.uniform(0.5, 1.0)
        sig += weight * np.sin(2.0 * np.pi * t / period + phase)
    peak = np.max(np.abs(sig))
    if peak > 1e-8:
        sig = sig / peak * amplitude
    return sig.astype(np.float32)


def _euler_to_rotation(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Per-frame ZYX Euler angles (radians) -> (T, 3, 3) rotation matrices."""
    n = yaw.shape[0]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(roll), np.sin(roll)
    ry = np.zeros((n, 3, 3)); ry[:, 0, 0] = cy; ry[:, 0, 2] = sy; ry[:, 1, 1] = 1; ry[:, 2, 0] = -sy; ry[:, 2, 2] = cy
    rx = np.zeros((n, 3, 3)); rx[:, 0, 0] = 1; rx[:, 1, 1] = cx; rx[:, 1, 2] = -sx; rx[:, 2, 1] = sx; rx[:, 2, 2] = cx
    rz = np.zeros((n, 3, 3)); rz[:, 0, 0] = cz; rz[:, 0, 1] = -sz; rz[:, 1, 0] = sz; rz[:, 1, 1] = cz; rz[:, 2, 2] = 1
    return np.matmul(np.matmul(rz, ry), rx)


def inject_idle_head_motion(
    rotation: np.ndarray, fps: float, *, yaw_deg: float = 4.0, pitch_deg: float = 3.0,
    roll_deg: float = 2.0, seed: int = 0,
) -> np.ndarray:
    """Add gentle procedural head sway to predicted rotation matrices.

    Head pose is largely NOT audio-predictable, so the model outputs a near-still
    head. This composes a slow, smooth synthetic rotation (a few degrees of
    yaw/pitch/roll) onto the prediction so the avatar reads as alive rather than
    frozen. Result is re-orthonormalized downstream during template reconstruction.
    """
    mats = np.asarray(rotation, dtype=np.float32).reshape(-1, 3, 3)
    n = mats.shape[0]
    if n == 0:
        return mats
    rng = np.random.default_rng(seed)
    yaw = np.radians(_smooth_signal(n, fps, yaw_deg, rng))
    pitch = np.radians(_smooth_signal(n, fps, pitch_deg, rng))
    roll = np.radians(_smooth_signal(n, fps, roll_deg, rng))
    idle = _euler_to_rotation(yaw, pitch, roll).astype(np.float32)
    return np.matmul(idle, mats).astype(np.float32)


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

    if config.get("add_idle_motion", False) and "rotation_matrix" in bundle:
        bundle["rotation_matrix"] = inject_idle_head_motion(
            bundle["rotation_matrix"],
            fps=fps,
            yaw_deg=float(config.get("idle_yaw_deg", 4.0)),
            pitch_deg=float(config.get("idle_pitch_deg", 3.0)),
            roll_deg=float(config.get("idle_roll_deg", 2.0)),
            seed=int(config.get("idle_seed", 0)),
        )

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
