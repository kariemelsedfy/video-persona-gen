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


def inject_blinks(
    eye_ratio: np.ndarray,
    fps: float,
    *,
    closed_value: float = 0.02,
    interval_sec: float = 3.5,
    jitter_sec: float = 1.5,
    blink_ms: float = 180.0,
    seed: int = 0,
) -> np.ndarray:
    """Replace a (T, n_eyes) eye-open array with a synthetic blinking signal.

    The open baseline is each eye's own mean so the injected motion matches the
    predicted resting openness. Both eyes share blink timing (people blink
    together).
    """
    array = np.asarray(eye_ratio, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        return array
    n_frames = array.shape[0]
    open_values = array.mean(axis=0)
    base = generate_blink_signal(
        n_frames,
        fps,
        open_value=1.0,
        closed_value=0.0,
        interval_sec=interval_sec,
        jitter_sec=jitter_sec,
        blink_ms=blink_ms,
        seed=seed,
    )  # in [0, 1]; 1 = open, dips toward 0 at blinks
    out = np.empty_like(array)
    for eye in range(array.shape[1]):
        open_v = float(open_values[eye])
        out[:, eye] = closed_value + (open_v - closed_value) * base
    return out.astype(np.float32)


def apply_motion_postprocess(bundle: dict, config: dict, fps: float) -> dict:
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

    if config.get("add_blinks", False) and "eye_ratio" in bundle:
        bundle["eye_ratio"] = inject_blinks(
            bundle["eye_ratio"],
            fps=fps,
            closed_value=float(config.get("blink_closed_value", 0.02)),
            interval_sec=float(config.get("blink_interval_sec", 3.5)),
            jitter_sec=float(config.get("blink_jitter_sec", 1.5)),
            blink_ms=float(config.get("blink_ms", 180.0)),
            seed=int(config.get("blink_seed", 0)),
        )
    return bundle
