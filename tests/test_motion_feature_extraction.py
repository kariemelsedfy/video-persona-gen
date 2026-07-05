from __future__ import annotations

import json
import pickle
from pathlib import Path

from avagen.data.dataset import load_motion_features, load_processed_clip_records
from avagen.features.motion_features import extract_motion_features_for_manifest


def test_extract_motion_features_for_manifest_updates_metadata_and_manifest(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed" / "demo_id"
    clip_dir = processed_root / "clip_a"
    face_crop_dir = clip_dir / "face_crops"
    face_crop_dir.mkdir(parents=True)
    (face_crop_dir / "000000.png").write_text("png", encoding="utf-8")
    (clip_dir / "audio.wav").write_text("audio", encoding="utf-8")
    (clip_dir / "frame_metadata.json").write_text(json.dumps([{"frame_index": 0}]), encoding="utf-8")

    motion_template = {
        "output_fps": 25,
        "c_eyes_lst": [[[0.1, 0.2]], [[0.2, 0.3]]],
        "c_lip_lst": [[[0.05]], [[0.15]]],
        "motion": [
            {
                "scale": [[[1.0]]],
                "R": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
                "exp": [[[0.0, 0.1, 0.2]] * 21],
                "t": [[[0.1, 0.2, 0.3]]],
                "kp": [[[0.0, 0.1, 0.2]] * 21],
                "x_s": [[[0.3, 0.2, 0.1]] * 21],
            },
            {
                "scale": [[[1.1]]],
                "R": [[[1.0, 0.0, 0.0], [0.0, 0.9, 0.1], [0.0, -0.1, 0.9]]],
                "exp": [[[0.2, 0.1, 0.0]] * 21],
                "t": [[[0.2, 0.1, 0.0]]],
                "kp": [[[0.1, 0.2, 0.3]] * 21],
                "x_s": [[[0.4, 0.3, 0.2]] * 21],
            },
        ],
    }
    motion_template_path = clip_dir / "motion_template.pkl"
    with motion_template_path.open("wb") as handle:
        pickle.dump(motion_template, handle)

    metadata = {
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
            "motion_template_path": str(motion_template_path),
            "motion_features_path": None,
            "motion_summary_path": None,
        },
    }
    (clip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    manifest_path = processed_root / "manifest.jsonl"
    manifest_record = {
        "clip_id": "clip_a",
        "identity_id": "demo_id",
        "audio_path": str(clip_dir / "audio.wav"),
        "face_crop_dir": str(face_crop_dir),
        "landmarks_path": None,
        "head_pose_path": None,
        "expression_path": None,
        "motion_template_path": str(motion_template_path),
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
    manifest_path.write_text(json.dumps(manifest_record) + "\n", encoding="utf-8")

    result = extract_motion_features_for_manifest(manifest_path, overwrite=True)
    assert result["status"] == "completed"

    records = load_processed_clip_records(manifest_path, require_motion_template=True)
    features = load_motion_features(records[0])
    assert "motion_vector" in features
    assert features["motion_vector"].shape[0] == 2

    updated_metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
    assert updated_metadata["optional_artifacts"]["motion_features_path"].endswith("motion_features.npz")
    assert updated_metadata["optional_artifacts"]["motion_summary_path"].endswith("motion_summary.json")
