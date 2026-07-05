from __future__ import annotations

import json
from pathlib import Path

from avagen.data.manifests import refresh_identity_manifest


def test_refresh_identity_manifest_writes_outputs(tmp_path: Path) -> None:
    identity_dir = tmp_path / "processed" / "demo_id" / "clip_a"
    identity_dir.mkdir(parents=True)
    metadata = {
        "identity_id": "demo_id",
        "clip_id": "clip_a",
        "split": "train",
        "artifacts": {
            "audio_path": "data/processed/demo_id/clip_a/audio.wav",
            "face_crop_dir": "data/processed/demo_id/clip_a/face_crops",
            "frame_metadata_path": "data/processed/demo_id/clip_a/frame_metadata.json",
        },
        "video_info": {"fps": 24.0},
        "preprocessing": {
            "target_fps_effective": 25.0,
            "audio_sample_rate": 16000,
        },
        "stats": {
            "num_frames": 40,
            "duration_sec": 1.6,
            "face_detection_rate": 0.85,
        },
        "optional_artifacts": {
            "landmarks_path": None,
            "head_pose_path": None,
            "expression_path": None,
            "motion_template_path": None,
        },
    }
    (identity_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    records, manifest_path, report_path = refresh_identity_manifest(tmp_path / "processed" / "demo_id")

    assert len(records) == 1
    assert records[0].clip_id == "clip_a"
    assert manifest_path.exists()
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8")
