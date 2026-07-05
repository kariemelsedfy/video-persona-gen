from __future__ import annotations

import json
import pickle
from pathlib import Path

from avagen.data.dataset import (
    ProcessedClipDataset,
    load_clip_metadata,
    load_frame_metadata,
    load_motion_template,
    load_processed_clip_records,
)


def test_load_processed_clip_records_and_motion_template(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    processed_root = repo_root / "data" / "processed" / "demo_id"
    clip_dir = processed_root / "clip_a"
    face_crop_dir = clip_dir / "face_crops"
    face_crop_dir.mkdir(parents=True)

    (clip_dir / "audio.wav").write_text("audio", encoding="utf-8")
    (face_crop_dir / "000000.png").write_text("png", encoding="utf-8")
    (clip_dir / "frame_metadata.json").write_text(json.dumps([{"frame_index": 0}]), encoding="utf-8")
    metadata = {
        "identity_id": "demo_id",
        "clip_id": "clip_a",
        "split": "train",
        "source_video_path": str(tmp_path / "raw" / "clip.mp4"),
        "artifacts": {
            "audio_path": "data/processed/demo_id/clip_a/audio.wav",
            "face_crop_dir": "data/processed/demo_id/clip_a/face_crops",
            "frame_metadata_path": "data/processed/demo_id/clip_a/frame_metadata.json",
        },
        "video_info": {"fps": 25.0},
        "preprocessing": {"target_fps_effective": 25.0, "audio_sample_rate": 16000},
        "stats": {"num_frames": 10, "duration_sec": 0.4, "face_detection_rate": 1.0},
        "optional_artifacts": {"motion_template_path": str(clip_dir / "motion_template.pkl")},
    }
    (clip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with (clip_dir / "motion_template.pkl").open("wb") as handle:
        pickle.dump({"n_frames": 10, "motion": []}, handle)

    manifest_path = processed_root / "manifest.jsonl"
    manifest_record = {
        "clip_id": "clip_a",
        "identity_id": "demo_id",
        "audio_path": "data/processed/demo_id/clip_a/audio.wav",
        "face_crop_dir": "data/processed/demo_id/clip_a/face_crops",
        "landmarks_path": None,
        "head_pose_path": None,
        "expression_path": None,
        "motion_template_path": str(clip_dir / "motion_template.pkl"),
        "fps": 25.0,
        "duration_sec": 0.4,
        "num_frames": 10,
        "face_detection_rate": 1.0,
        "avg_yaw_abs": None,
        "avg_pitch_abs": None,
        "avg_roll_abs": None,
        "audio_sample_rate": 16000,
        "split": "train",
    }
    manifest_path.write_text(json.dumps(manifest_record) + "\n", encoding="utf-8")

    records = load_processed_clip_records(manifest_path, require_motion_template=True)
    assert len(records) == 1
    assert records[0].clip_id == "clip_a"
    assert load_frame_metadata(records[0])[0]["frame_index"] == 0
    assert load_clip_metadata(records[0])["clip_id"] == "clip_a"
    assert load_motion_template(records[0])["n_frames"] == 10

    dataset = ProcessedClipDataset(manifest_path, splits=["train"], require_motion_template=True)
    assert len(dataset) == 1
    assert dataset.clip_ids() == ["clip_a"]
