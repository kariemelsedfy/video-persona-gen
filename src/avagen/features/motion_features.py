"""Motion feature extraction derived from LivePortrait motion templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from avagen.data.dataset import load_motion_template, load_processed_clip_records
from avagen.data.manifests import refresh_identity_manifest
from avagen.utils.paths import to_repo_relative


def _stack_motion_field(motion_items: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.concatenate([np.asarray(item[key], dtype=np.float32) for item in motion_items], axis=0)


def extract_motion_feature_bundle(template: dict[str, Any]) -> dict[str, Any]:
    motion_items = list(template.get("motion", []))
    if not motion_items:
        raise ValueError("Motion template does not contain any motion frames.")

    scale = _stack_motion_field(motion_items, "scale")
    rotation_matrix = _stack_motion_field(motion_items, "R")
    expression = _stack_motion_field(motion_items, "exp")
    translation = _stack_motion_field(motion_items, "t")
    keypoints = _stack_motion_field(motion_items, "kp")
    source_keypoints = _stack_motion_field(motion_items, "x_s")
    eye_ratio = np.concatenate([np.asarray(item, dtype=np.float32) for item in template["c_eyes_lst"]], axis=0)
    lip_ratio = np.concatenate([np.asarray(item, dtype=np.float32) for item in template["c_lip_lst"]], axis=0)

    motion_vector = np.concatenate(
        [
            scale.reshape(scale.shape[0], -1),
            rotation_matrix.reshape(rotation_matrix.shape[0], -1),
            expression.reshape(expression.shape[0], -1),
            translation.reshape(translation.shape[0], -1),
            keypoints.reshape(keypoints.shape[0], -1),
            source_keypoints.reshape(source_keypoints.shape[0], -1),
            eye_ratio.reshape(eye_ratio.shape[0], -1),
            lip_ratio.reshape(lip_ratio.shape[0], -1),
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "output_fps": np.asarray(template.get("output_fps", 0), dtype=np.int32),
        "scale": scale.astype(np.float32),
        "rotation_matrix": rotation_matrix.astype(np.float32),
        "expression": expression.astype(np.float32),
        "translation": translation.astype(np.float32),
        "keypoints": keypoints.astype(np.float32),
        "source_keypoints": source_keypoints.astype(np.float32),
        "eye_ratio": eye_ratio.astype(np.float32),
        "lip_ratio": lip_ratio.astype(np.float32),
        "motion_vector": motion_vector,
    }


def summarize_motion_features(bundle: dict[str, Any]) -> dict[str, float | int]:
    motion_vector = np.asarray(bundle["motion_vector"], dtype=np.float32)
    translation = np.asarray(bundle["translation"], dtype=np.float32)
    scale = np.asarray(bundle["scale"], dtype=np.float32)
    eye_ratio = np.asarray(bundle["eye_ratio"], dtype=np.float32)
    lip_ratio = np.asarray(bundle["lip_ratio"], dtype=np.float32)

    return {
        "num_frames": int(motion_vector.shape[0]),
        "feature_dim": int(motion_vector.shape[1]),
        "output_fps": int(np.asarray(bundle["output_fps"]).item()),
        "mean_scale": float(scale.mean()),
        "mean_translation_l2": float(np.linalg.norm(translation, axis=1).mean()),
        "mean_eye_ratio": float(eye_ratio.mean()),
        "mean_lip_ratio": float(lip_ratio.mean()),
    }


def save_motion_feature_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bundle)
    return path


def extract_motion_features_for_manifest(
    manifest_path: str | Path,
    clip_ids: tuple[str, ...] = (),
    overwrite: bool = False,
) -> dict[str, object]:
    manifest = Path(manifest_path).expanduser().resolve()
    records = load_processed_clip_records(manifest, require_motion_template=True)
    selected_clip_ids = set(clip_ids)
    processed_clips: list[dict[str, object]] = []

    for record in records:
        if selected_clip_ids and record.clip_id not in selected_clip_ids:
            continue

        clip_dir = record.metadata_path.parent
        features_path = clip_dir / "motion_features.npz"
        summary_path = clip_dir / "motion_summary.json"
        if features_path.exists() and summary_path.exists() and not overwrite:
            processed_clips.append(
                {
                    "clip_id": record.clip_id,
                    "identity_id": record.identity_id,
                    "status": "skipped_existing",
                    "motion_features_path": str(features_path),
                    "motion_summary_path": str(summary_path),
                }
            )
            continue

        bundle = extract_motion_feature_bundle(load_motion_template(record))
        save_motion_feature_bundle(bundle, features_path)
        summary = summarize_motion_features(bundle)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
        optional_artifacts = metadata.setdefault("optional_artifacts", {})
        if not isinstance(optional_artifacts, dict):
            raise ValueError(f"Invalid optional_artifacts in {record.metadata_path}")
        optional_artifacts["motion_features_path"] = to_repo_relative(features_path)
        optional_artifacts["motion_summary_path"] = to_repo_relative(summary_path)
        metadata["motion_features"] = {
            "motion_features_path": to_repo_relative(features_path),
            "motion_summary_path": to_repo_relative(summary_path),
            "feature_dim": summary["feature_dim"],
            "num_frames": summary["num_frames"],
        }
        record.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        processed_clips.append(
            {
                "clip_id": record.clip_id,
                "identity_id": record.identity_id,
                "status": "completed",
                "motion_features_path": str(features_path),
                "motion_summary_path": str(summary_path),
                "feature_dim": int(summary["feature_dim"]),
            }
        )

    identity_dir = manifest.parent
    refreshed_records, refreshed_manifest_path, report_path = refresh_identity_manifest(identity_dir)
    return {
        "manifest_path": str(refreshed_manifest_path),
        "report_path": str(report_path),
        "num_manifest_records": len(refreshed_records),
        "processed_clips": processed_clips,
        "status": "completed",
    }
