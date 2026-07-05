from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avagen.data.dataset import AudioMotionSequenceDataset, collate_padded_sequences


def test_audio_motion_sequence_dataset_aligns_audio_to_motion(tmp_path: Path) -> None:
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
                "split": "train",
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
        ),
        encoding="utf-8",
    )

    np.savez(
        clip_dir / "audio_features.npz",
        time_axis_sec=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
        rms_energy=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        log_rms_energy=np.asarray([0.0, 0.0, 0.7, 1.0], dtype=np.float32),
        zero_crossing_rate=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        peak_amplitude=np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        spectral_centroid_hz=np.asarray([100.0, 110.0, 120.0, 130.0], dtype=np.float32),
    )
    np.savez(
        clip_dir / "motion_features.npz",
        output_fps=np.asarray(2, dtype=np.int32),
        motion_vector=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )

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
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = AudioMotionSequenceDataset(manifest_path, splits=["train"])
    assert len(dataset) == 1
    assert dataset.identity_ids() == ["demo_id"]
    sequence = dataset[0]
    assert sequence.audio_features.shape == (2, 5)
    assert sequence.motion_features.shape == (2, 2)
    np.testing.assert_allclose(
        sequence.audio_features,
        np.asarray(
            [
                [0.0, 0.0, 0.1, 0.2, 100.0],
                [3.0, 1.0, 0.4, 0.5, 130.0],
            ],
            dtype=np.float32,
        ),
    )
    batch = collate_padded_sequences([sequence])
    assert batch["audio_features"].shape == (1, 2, 5)
    assert batch["motion_features"].shape == (1, 2, 2)
    assert batch["identity_ids"] == ["demo_id"]
    assert batch["splits"] == ["train"]
    np.testing.assert_array_equal(batch["mask"], np.asarray([[True, True]]))
    np.testing.assert_array_equal(batch["lengths"], np.asarray([2], dtype=np.int32))
