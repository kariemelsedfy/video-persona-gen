from __future__ import annotations

import json
from pathlib import Path

from avagen.renderers.liveportrait_wrapper import LivePortraitRunResult
from avagen.renderers.motion_template import MotionTemplateExtractionConfig, extract_motion_templates


def test_extract_motion_templates_updates_metadata_and_manifest(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed" / "demo_id"
    clip_dir = processed_root / "clip_a"
    face_crop_dir = clip_dir / "face_crops"
    face_crop_dir.mkdir(parents=True)

    source_video = tmp_path / "raw" / "clip.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_text("video", encoding="utf-8")
    source_image = face_crop_dir / "000000.png"
    source_image.write_text("png", encoding="utf-8")

    metadata = {
        "identity_id": "demo_id",
        "clip_id": "clip_a",
        "split": "train",
        "source_video_path": str(source_video),
        "artifacts": {
            "audio_path": "data/processed/demo_id/clip_a/audio.wav",
            "face_crop_dir": "data/processed/demo_id/clip_a/face_crops",
            "frame_metadata_path": "data/processed/demo_id/clip_a/frame_metadata.json",
        },
        "video_info": {"fps": 25.0},
        "preprocessing": {
            "target_fps_effective": 25.0,
            "audio_sample_rate": 16000,
        },
        "stats": {
            "num_frames": 20,
            "duration_sec": 0.8,
            "face_detection_rate": 1.0,
        },
        "optional_artifacts": {
            "landmarks_path": None,
            "head_pose_path": None,
            "expression_path": None,
            "motion_template_path": None,
        },
    }
    (clip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    manifest_path = processed_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "clip_id": "clip_a",
                "identity_id": "demo_id",
                "audio_path": "data/processed/demo_id/clip_a/audio.wav",
                "face_crop_dir": "data/processed/demo_id/clip_a/face_crops",
                "landmarks_path": None,
                "head_pose_path": None,
                "expression_path": None,
                "motion_template_path": None,
                "fps": 25.0,
                "duration_sec": 0.8,
                "num_frames": 20,
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

    def fake_runner(config, dry_run=False):
        template_path = config.driving_path.with_suffix(".pkl")
        template_path.write_text("template", encoding="utf-8")
        return LivePortraitRunResult(
            status="completed",
            command=["python", "inference.py"],
            cwd=str(config.liveportrait_root),
            output_dir=str(config.output_dir),
            resolved_inference_script="inference.py",
            returncode=0,
            dry_run=dry_run,
        )

    result = extract_motion_templates(
        MotionTemplateExtractionConfig(
            manifest_path=manifest_path,
            liveportrait_root=tmp_path / "external" / "LivePortrait",
            python_executable="python",
            work_root=tmp_path / "work",
            overwrite=True,
        ),
        runner=fake_runner,
    )

    template_path = clip_dir / "motion_template.pkl"
    assert template_path.exists()
    updated_metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
    assert updated_metadata["optional_artifacts"]["motion_template_path"].endswith("motion_template.pkl")
    assert result["status"] == "completed"

    refreshed_manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    refreshed_record = json.loads(refreshed_manifest_lines[0])
    assert refreshed_record["motion_template_path"].endswith("motion_template.pkl")
