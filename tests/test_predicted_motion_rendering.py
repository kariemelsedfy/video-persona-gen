from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from avagen.models.motion_gru import MotionGRU, MotionGRUConfig
from avagen.renderers.video_renderer import (
    PredictedMotionRenderConfig,
    render_predicted_motion_for_manifest,
)
from avagen.training.checkpointing import save_checkpoint


def _write_fake_inference_script(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('-s')",
                "parser.add_argument('-d')",
                "parser.add_argument('-o')",
                "args = parser.parse_args()",
                "output_dir = Path(args.o)",
                "output_dir.mkdir(parents=True, exist_ok=True)",
                "(output_dir / 'render_ok.txt').write_text(f'{args.s}\\n{args.d}\\n', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )


def _build_processed_clip(tmp_path: Path) -> tuple[Path, Path]:
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
    np.savez(
        clip_dir / "motion_features.npz",
        output_fps=np.asarray(2, dtype=np.int32),
        scale=np.asarray([[[1.0]], [[1.1]]], dtype=np.float32),
        rotation_matrix=np.asarray(
            [
                [[[1.0, 0.0], [0.0, 1.0]]],
                [[[0.9, 0.1], [-0.1, 0.9]]],
            ],
            dtype=np.float32,
        ),
        expression=np.asarray([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float32),
        translation=np.asarray([[[0.0, 0.1]], [[0.2, 0.3]]], dtype=np.float32),
        keypoints=np.asarray(
            [
                [[[0.1, 0.2], [0.3, 0.4]]],
                [[[0.2, 0.3], [0.4, 0.5]]],
            ],
            dtype=np.float32,
        ),
        source_keypoints=np.asarray(
            [
                [[[0.5, 0.6], [0.7, 0.8]]],
                [[[0.6, 0.7], [0.8, 0.9]]],
            ],
            dtype=np.float32,
        ),
        eye_ratio=np.asarray([[[0.2]], [[0.25]]], dtype=np.float32),
        lip_ratio=np.asarray([[[0.4]], [[0.45]]], dtype=np.float32),
        motion_vector=np.asarray(
            [
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.0, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.4],
                [1.1, 0.9, 0.1, -0.1, 0.9, 0.3, 0.4, 0.2, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25, 0.45],
            ],
            dtype=np.float32,
        ),
    )

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
    return manifest_path, clip_dir


def test_render_predicted_motion_for_manifest(tmp_path: Path) -> None:
    manifest_path, _ = _build_processed_clip(tmp_path)

    model = MotionGRU(MotionGRUConfig(input_size=5, output_size=19, hidden_size=8, num_layers=1, dropout=0.0))
    checkpoint_path = tmp_path / "checkpoints" / "motion.pt"
    save_checkpoint(
        {
            "epoch": 1,
            "model_config": {
                "input_size": 5,
                "output_size": 19,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    model_config_path = tmp_path / "train_config.json"
    model_config_path.write_text(
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

    liveportrait_root = tmp_path / "LivePortrait"
    liveportrait_root.mkdir()
    inference_path = liveportrait_root / "inference.py"
    _write_fake_inference_script(inference_path)

    output_root = tmp_path / "outputs" / "render_predicted_motion"
    summary = render_predicted_motion_for_manifest(
        PredictedMotionRenderConfig(
            checkpoint_path=checkpoint_path,
            model_config_path=model_config_path,
            manifest_path=manifest_path,
            output_root=output_root,
            liveportrait_root=liveportrait_root,
            inference_script=inference_path,
            python_executable="python3",
            device="cpu",
        )
    )

    assert summary["status"] == "completed"
    assert len(summary["rendered_records"]) == 1
    rendered_record = summary["rendered_records"][0]
    assert Path(rendered_record["driving_template_path"]).exists()
    assert Path(rendered_record["output_dir"]).exists()
    assert (Path(rendered_record["output_dir"]) / "render_ok.txt").exists()
