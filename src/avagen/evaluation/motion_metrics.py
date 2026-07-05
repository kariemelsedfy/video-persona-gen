"""Evaluate predicted motion features against ground-truth motion bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from avagen.data.dataset import load_motion_features, load_processed_clip_records


def compute_motion_error_metrics(
    predicted_bundle: dict[str, Any],
    target_bundle: dict[str, Any],
) -> dict[str, float | int]:
    predicted = np.asarray(predicted_bundle["motion_vector"], dtype=np.float32)
    target = np.asarray(target_bundle["motion_vector"], dtype=np.float32)
    if predicted.shape != target.shape:
        raise ValueError(f"Predicted and target motion vectors must match, got {predicted.shape} and {target.shape}.")

    delta = predicted - target
    metrics: dict[str, float | int] = {
        "num_frames": int(predicted.shape[0]),
        "feature_dim": int(predicted.shape[1]),
        "mse": float(np.mean(np.square(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "mae": float(np.mean(np.abs(delta))),
    }

    if predicted.shape[0] > 1:
        predicted_velocity = predicted[1:] - predicted[:-1]
        target_velocity = target[1:] - target[:-1]
        metrics["velocity_mse"] = float(np.mean(np.square(predicted_velocity - target_velocity)))
        metrics["velocity_mae"] = float(np.mean(np.abs(predicted_velocity - target_velocity)))
    else:
        metrics["velocity_mse"] = 0.0
        metrics["velocity_mae"] = 0.0

    if "translation" in predicted_bundle and "translation" in target_bundle:
        translation_delta = np.asarray(predicted_bundle["translation"], dtype=np.float32) - np.asarray(
            target_bundle["translation"], dtype=np.float32
        )
        metrics["translation_mae"] = float(np.mean(np.abs(translation_delta)))
    if "eye_ratio" in predicted_bundle and "eye_ratio" in target_bundle:
        eye_delta = np.asarray(predicted_bundle["eye_ratio"], dtype=np.float32) - np.asarray(
            target_bundle["eye_ratio"], dtype=np.float32
        )
        metrics["eye_ratio_mae"] = float(np.mean(np.abs(eye_delta)))
    if "lip_ratio" in predicted_bundle and "lip_ratio" in target_bundle:
        lip_delta = np.asarray(predicted_bundle["lip_ratio"], dtype=np.float32) - np.asarray(
            target_bundle["lip_ratio"], dtype=np.float32
        )
        metrics["lip_ratio_mae"] = float(np.mean(np.abs(lip_delta)))
    return metrics


def evaluate_motion_predictions(
    manifest_path: str | Path,
    predicted_root: str | Path,
    *,
    clip_ids: Sequence[str] = (),
    skip_missing: bool = False,
) -> dict[str, Any]:
    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(predicted_root).expanduser().resolve()
    selected_clip_ids = set(clip_ids)

    records = [
        record
        for record in load_processed_clip_records(manifest)
        if record.motion_features_path is not None
    ]

    per_clip: list[dict[str, Any]] = []
    aggregate_sums: dict[str, float] = {}
    total_frames = 0
    missing_clips: list[str] = []

    for record in records:
        if selected_clip_ids and record.clip_id not in selected_clip_ids:
            continue

        predicted_path = root / record.identity_id / record.clip_id / "predicted_motion_features.npz"
        if not predicted_path.exists():
            if skip_missing:
                missing_clips.append(record.clip_id)
                continue
            raise FileNotFoundError(f"Missing predicted motion features for clip {record.clip_id}: {predicted_path}")

        with np.load(predicted_path, allow_pickle=False) as payload:
            predicted_bundle = {key: payload[key] for key in payload.files}
        target_bundle = load_motion_features(record)
        metrics = compute_motion_error_metrics(predicted_bundle, target_bundle)
        per_clip.append(
            {
                "clip_id": record.clip_id,
                "identity_id": record.identity_id,
                "predicted_motion_features_path": str(predicted_path),
                **metrics,
            }
        )

        weight = int(metrics["num_frames"])
        total_frames += weight
        for key, value in metrics.items():
            if key in {"num_frames", "feature_dim"}:
                continue
            aggregate_sums[key] = aggregate_sums.get(key, 0.0) + float(value) * weight

    denominator = max(total_frames, 1)
    aggregate_metrics = {
        key: value / denominator
        for key, value in sorted(aggregate_sums.items())
    }

    return {
        "status": "completed",
        "manifest_path": str(manifest),
        "predicted_root": str(root),
        "num_evaluated_clips": len(per_clip),
        "total_frames": total_frames,
        "missing_clips": missing_clips,
        "aggregate_metrics": aggregate_metrics,
        "per_clip": per_clip,
    }
