from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from avagen.models.motion_gru import MotionGRU, MotionGRUConfig
from avagen.training.train_motion import train_motion_model


def _write_clip(
    clip_dir: Path,
    *,
    identity_id: str,
    clip_id: str,
    split: str,
    scale: float,
) -> dict[str, object]:
    face_crop_dir = clip_dir / "face_crops"
    face_crop_dir.mkdir(parents=True)
    (face_crop_dir / "000000.png").write_text("png", encoding="utf-8")
    (clip_dir / "audio.wav").write_text("audio", encoding="utf-8")
    (clip_dir / "frame_metadata.json").write_text(json.dumps([{"frame_index": 0}]), encoding="utf-8")

    time_axis = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    rms = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32) * scale
    log_rms = np.log(np.maximum(rms, 1e-4)).astype(np.float32)
    zcr = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32) * scale
    peak = np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32) * scale
    centroid = np.asarray([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
    motion = np.stack([rms[:2], peak[:2]], axis=1).astype(np.float32)

    np.savez(
        clip_dir / "audio_features.npz",
        time_axis_sec=time_axis,
        rms_energy=rms,
        log_rms_energy=log_rms,
        zero_crossing_rate=zcr,
        peak_amplitude=peak,
        spectral_centroid_hz=centroid,
    )
    np.savez(
        clip_dir / "motion_features.npz",
        output_fps=np.asarray(2, dtype=np.int32),
        motion_vector=motion,
    )

    metadata = {
        "identity_id": identity_id,
        "clip_id": clip_id,
        "split": split,
        "source_video_path": str(clip_dir / "source.mp4"),
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

    return {
        "clip_id": clip_id,
        "identity_id": identity_id,
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
        "split": split,
    }


def test_motion_gru_forward_shape() -> None:
    model = MotionGRU(MotionGRUConfig(input_size=5, output_size=3, hidden_size=8, num_layers=1))
    audio = torch.randn(2, 4, 5)
    lengths = torch.tensor([4, 2], dtype=torch.long)
    output = model(audio, lengths=lengths)
    assert output.shape == (2, 4, 3)


def test_train_motion_model_writes_outputs(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed" / "demo_id"
    clip_a = _write_clip(processed_root / "clip_a", identity_id="demo_id", clip_id="clip_a", split="train", scale=1.0)
    clip_b = _write_clip(processed_root / "clip_b", identity_id="demo_id", clip_id="clip_b", split="train", scale=0.5)
    clip_c = _write_clip(processed_root / "clip_c", identity_id="demo_id", clip_id="clip_c", split="val", scale=0.8)
    manifest_path = processed_root / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(item) for item in (clip_a, clip_b, clip_c)) + "\n",
        encoding="utf-8",
    )

    experiment_dir = tmp_path / "experiments" / "gru_smoke"
    summary = train_motion_model(
        {
            "manifest_path": str(manifest_path),
            "experiment_dir": str(experiment_dir),
            "seed": 3,
            "dataset": {
                "train_splits": ["train"],
                "val_splits": ["val"],
            },
            "model": {
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "training": {
                "batch_size": 2,
                "learning_rate": 1e-3,
                "epochs": 2,
                "device": "cpu",
                "precision": "fp32",
                "velocity_loss_weight": 0.05,
                "grad_clip_norm": 1.0,
                "num_workers": 0,
            },
        }
    )

    assert summary["status"] == "completed"
    assert summary["num_train_sequences"] == 2
    assert summary["num_val_sequences"] == 1
    assert Path(summary["best_checkpoint_path"]).exists()
    assert Path(summary["last_checkpoint_path"]).exists()
    assert Path(summary["resolved_config_path"]).exists()
    summary_payload = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert summary_payload["best_epoch"] >= 1
