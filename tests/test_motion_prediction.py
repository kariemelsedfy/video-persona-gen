from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from avagen.features.motion_features import (
    extract_motion_feature_bundle,
    motion_feature_bundle_to_template,
    unflatten_motion_vector,
)

torch = pytest.importorskip("torch")

from avagen.inference.generate_motion import predict_motion_for_manifest
from avagen.models.motion_gru import MotionGRU, MotionGRUConfig
from avagen.training.checkpointing import save_checkpoint


def _build_template() -> dict[str, object]:
    motion_frame_a = {
        "scale": np.asarray([[1.0]], dtype=np.float32),
        "R": np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "exp": np.asarray([[0.1, 0.2]], dtype=np.float32),
        "t": np.asarray([[0.0, 0.1]], dtype=np.float32),
        "kp": np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
        "x_s": np.asarray([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32),
    }
    motion_frame_b = {
        "scale": np.asarray([[1.1]], dtype=np.float32),
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


def test_motion_vector_roundtrip_and_template_reconstruction() -> None:
    bundle = extract_motion_feature_bundle(_build_template())
    reconstructed = unflatten_motion_vector(bundle["motion_vector"], bundle)
    np.testing.assert_allclose(reconstructed["scale"], bundle["scale"])
    np.testing.assert_allclose(reconstructed["rotation_matrix"], bundle["rotation_matrix"])
    template = motion_feature_bundle_to_template(reconstructed)
    assert template["n_frames"] == 2
    assert len(template["motion"]) == 2


def test_predict_motion_for_manifest_writes_artifacts(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed" / "demo_id"
    clip_dir = processed_root / "clip_a"
    face_crop_dir = clip_dir / "face_crops"
    face_crop_dir.mkdir(parents=True)
    (face_crop_dir / "000000.png").write_text("png", encoding="utf-8")
    (clip_dir / "audio.wav").write_text("audio", encoding="utf-8")
    (clip_dir / "frame_metadata.json").write_text(json.dumps([{"frame_index": 0}]), encoding="utf-8")

    np.savez(
        clip_dir / "audio_features.npz",
        time_axis_sec=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
        rms_energy=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        log_rms_energy=np.asarray([0.0, 0.0, 0.7, 1.0], dtype=np.float32),
        zero_crossing_rate=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        peak_amplitude=np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        spectral_centroid_hz=np.asarray([100.0, 110.0, 120.0, 130.0], dtype=np.float32),
    )

    motion_bundle = extract_motion_feature_bundle(_build_template())
    np.savez(clip_dir / "motion_features.npz", **motion_bundle)
    metadata = {
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
            "audio_features_path": str(clip_dir / "audio_features.npz"),
            "motion_features_path": str(clip_dir / "motion_features.npz"),
        },
    }
    (clip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

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
                "audio_features_path": str(clip_dir / "audio_features.npz"),
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

    model = MotionGRU(
        MotionGRUConfig(
            input_size=5,
            output_size=int(motion_bundle["motion_vector"].shape[1]),
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
        )
    )
    checkpoint_path = tmp_path / "checkpoints" / "motion.pt"
    save_checkpoint(
        {
            "epoch": 1,
            "model_config": {
                "input_size": 5,
                "output_size": int(motion_bundle["motion_vector"].shape[1]),
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    config_path = tmp_path / "train_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "audio_feature_names": [
                        "rms_energy",
                        "log_rms_energy",
                        "zero_crossing_rate",
                        "peak_amplitude",
                        "spectral_centroid_hz",
                    ],
                    "motion_feature_name": "motion_vector",
                }
            }
        ),
        encoding="utf-8",
    )

    summary = predict_motion_for_manifest(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        manifest_path=manifest_path,
        output_root=tmp_path / "outputs" / "predicted_motion",
        clip_ids=("clip_a",),
        device="cpu",
    )

    assert summary["status"] == "completed"
    assert len(summary["predicted_records"]) == 1
    artifact = summary["predicted_records"][0]
    assert Path(artifact["predicted_motion_features_path"]).exists()
    assert Path(artifact["predicted_motion_summary_path"]).exists()
    assert Path(artifact["predicted_motion_template_path"]).exists()
    with Path(artifact["predicted_motion_template_path"]).open("rb") as handle:
        template = pickle.load(handle)
    assert template["n_frames"] == 2
