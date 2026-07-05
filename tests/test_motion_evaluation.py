from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avagen.evaluation.motion_metrics import compute_motion_error_metrics, evaluate_motion_predictions
from avagen.features.motion_features import extract_motion_feature_bundle


def _build_template(scale_offset: float = 0.0) -> dict[str, object]:
    motion_frame_a = {
        "scale": np.asarray([[1.0 + scale_offset]], dtype=np.float32),
        "R": np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "exp": np.asarray([[0.1, 0.2]], dtype=np.float32),
        "t": np.asarray([[0.0, 0.1]], dtype=np.float32),
        "kp": np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
        "x_s": np.asarray([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32),
    }
    motion_frame_b = {
        "scale": np.asarray([[1.1 + scale_offset]], dtype=np.float32),
        "R": np.asarray([[[0.9, 0.1], [-0.1, 0.9]]], dtype=np.float32),
        "exp": np.asarray([[0.3, 0.4]], dtype=np.float32),
        "t": np.asarray([[0.2, 0.3]], dtype=np.float32),
        "kp": np.asarray([[[0.2, 0.3], [0.4, 0.5]]], dtype=np.float32),
        "x_s": np.asarray([[[0.6, 0.7], [0.8, 0.9]]], dtype=np.float32),
    }
    return {
        "output_fps": 2,
        "motion": [motion_frame_a, motion_frame_b],
        "c_eyes_lst": [np.asarray([[0.2]], dtype=np.float32), np.asarray([[0.25]], dtype=np.float32)],
        "c_lip_lst": [np.asarray([[0.4]], dtype=np.float32), np.asarray([[0.45]], dtype=np.float32)],
    }


def test_compute_motion_error_metrics() -> None:
    target = extract_motion_feature_bundle(_build_template(scale_offset=0.0))
    predicted = extract_motion_feature_bundle(_build_template(scale_offset=0.2))
    metrics = compute_motion_error_metrics(predicted, target)
    assert metrics["num_frames"] == 2
    assert metrics["feature_dim"] == int(target["motion_vector"].shape[1])
    assert metrics["mse"] > 0.0
    assert metrics["velocity_mse"] >= 0.0


def test_evaluate_motion_predictions_reads_predicted_root(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed" / "demo_id"
    clip_dir = processed_root / "clip_a"
    face_crop_dir = clip_dir / "face_crops"
    face_crop_dir.mkdir(parents=True)
    (face_crop_dir / "000000.png").write_text("png", encoding="utf-8")
    (clip_dir / "audio.wav").write_text("audio", encoding="utf-8")
    (clip_dir / "frame_metadata.json").write_text(json.dumps([{"frame_index": 0}]), encoding="utf-8")
    (clip_dir / "metadata.json").write_text(
        json.dumps(
            {
                "identity_id": "demo_id",
                "clip_id": "clip_a",
                "split": "val",
                "source_video_path": str(tmp_path / "raw" / "clip.mp4"),
                "artifacts": {
                    "audio_path": str(clip_dir / "audio.wav"),
                    "face_crop_dir": str(face_crop_dir),
                    "frame_metadata_path": str(clip_dir / "frame_metadata.json"),
                },
                "video_info": {"fps": 25.0},
                "preprocessing": {"target_fps_effective": 25.0, "audio_sample_rate": 16000},
                "stats": {"num_frames": 2, "duration_sec": 0.08, "face_detection_rate": 1.0},
                "optional_artifacts": {
                    "motion_features_path": str(clip_dir / "motion_features.npz"),
                },
            }
        ),
        encoding="utf-8",
    )

    target_bundle = extract_motion_feature_bundle(_build_template(scale_offset=0.0))
    predicted_bundle = extract_motion_feature_bundle(_build_template(scale_offset=0.1))
    np.savez(clip_dir / "motion_features.npz", **target_bundle)

    manifest_path = processed_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "clip_id": "clip_a",
                "identity_id": "demo_id",
                "audio_path": str(clip_dir / "audio.wav"),
                "face_crop_dir": str(face_crop_dir),
                "landmarks_path": None,
                "head_pose_path": None,
                "expression_path": None,
                "motion_template_path": None,
                "audio_features_path": None,
                "motion_features_path": str(clip_dir / "motion_features.npz"),
                "fps": 25.0,
                "duration_sec": 0.08,
                "num_frames": 2,
                "face_detection_rate": 1.0,
                "avg_yaw_abs": None,
                "avg_pitch_abs": None,
                "avg_roll_abs": None,
                "audio_sample_rate": 16000,
                "split": "val",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    predicted_root = tmp_path / "outputs" / "predicted_motion" / "demo_id" / "clip_a"
    predicted_root.mkdir(parents=True)
    np.savez(predicted_root / "predicted_motion_features.npz", **predicted_bundle)

    summary = evaluate_motion_predictions(manifest_path, tmp_path / "outputs" / "predicted_motion")
    assert summary["status"] == "completed"
    assert summary["num_evaluated_clips"] == 1
    assert summary["aggregate_metrics"]["mse"] > 0.0
