from __future__ import annotations

import json
from pathlib import Path

from avagen.data.splits import apply_split_assignments, assign_clip_splits


def test_assign_clip_splits_covers_all_clips() -> None:
    assignments = assign_clip_splits(["clip_a", "clip_b", "clip_c", "clip_d"], seed=7)
    assert {item.clip_id for item in assignments} == {"clip_a", "clip_b", "clip_c", "clip_d"}
    assert {item.split for item in assignments}.issubset({"train", "val", "test"})


def test_assign_clip_splits_two_clips_produces_train_and_test() -> None:
    assignments = assign_clip_splits(["clip_a", "clip_b"], seed=3)
    assert [item.split for item in assignments] == ["train", "test"]


def test_apply_split_assignments_updates_metadata_and_manifest(tmp_path: Path) -> None:
    identity_dir = tmp_path / "processed" / "demo_id"
    clip_dir = identity_dir / "clip_a"
    clip_dir.mkdir(parents=True)
    metadata = {
        "identity_id": "demo_id",
        "clip_id": "clip_a",
        "split": "train",
        "artifacts": {
            "audio_path": "audio.wav",
            "face_crop_dir": str(clip_dir / "face_crops"),
            "frame_metadata_path": str(clip_dir / "frame_metadata.json"),
        },
        "video_info": {"fps": 25.0},
        "preprocessing": {"target_fps_effective": 25.0, "audio_sample_rate": 16000},
        "stats": {"num_frames": 10, "duration_sec": 0.4, "face_detection_rate": 1.0},
        "optional_artifacts": {"motion_template_path": None},
    }
    (clip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    manifest_path, _ = apply_split_assignments(identity_dir, assign_clip_splits(["clip_a"], seed=0))

    updated_metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
    assert updated_metadata["split"] in {"train", "val", "test"}
    assert manifest_path.exists()
