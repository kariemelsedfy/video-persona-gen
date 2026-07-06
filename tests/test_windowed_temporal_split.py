from __future__ import annotations

import json
from pathlib import Path

import pytest

from avagen.data.windowing import (
    WindowingConfig,
    plan_temporal_windows,
    temporal_split_region,
)


def test_temporal_split_regions_are_contiguous_and_disjoint() -> None:
    config = WindowingConfig(window_size=8, stride=4, train_fraction=0.8, val_fraction=0.1)
    n = 100
    train = temporal_split_region(n, "train", config)
    val = temporal_split_region(n, "val", config)
    test = temporal_split_region(n, "test", config)
    assert train == (0, 80)
    assert val == (80, 90)
    assert test == (90, 100)
    # contiguous cover of [0, n) with no overlap
    assert train[1] == val[0]
    assert val[1] == test[0]
    assert test[1] == n


def test_boundary_gap_separates_val_and_test_regions() -> None:
    config = WindowingConfig(
        window_size=8, stride=4, train_fraction=0.8, val_fraction=0.1, boundary_gap=3
    )
    n = 100
    train = temporal_split_region(n, "train", config)
    val = temporal_split_region(n, "val", config)
    test = temporal_split_region(n, "test", config)
    assert train == (0, 80)
    # val/test starts are pushed forward by the gap, leaving unused frames as a buffer
    assert val == (83, 90)
    assert test == (93, 100)
    assert val[0] - train[1] == 3
    assert test[0] - (train[1] + int(n * config.val_fraction)) == 3


def test_plan_windows_full_and_tail_coverage() -> None:
    config = WindowingConfig(window_size=8, stride=4, train_fraction=0.8, val_fraction=0.1)
    # train region [0, 80): stepped windows plus end-aligned tail if needed
    train_windows = plan_temporal_windows(100, "train", config)
    assert train_windows[0] == (0, 8)
    # every window is exactly window_size long and within the region
    assert all(end - start == 8 for start, end in train_windows)
    assert all(0 <= start and end <= 80 for start, end in train_windows)
    # non-decreasing starts, and the region tail is covered
    starts = [s for s, _ in train_windows]
    assert starts == sorted(starts)
    assert train_windows[-1][1] == 80


def test_plan_windows_appends_end_aligned_tail_window() -> None:
    # region [0, 20), window 8, stride 8 -> (0,8),(8,16) leaves [16,20) uncovered
    config = WindowingConfig(window_size=8, stride=8, train_fraction=1.0, val_fraction=0.0)
    windows = plan_temporal_windows(20, "train", config)
    assert windows == [(0, 8), (8, 16), (12, 20)]
    assert windows[-1][1] == 20


def test_plan_windows_region_shorter_than_window_yields_single_short_window() -> None:
    config = WindowingConfig(window_size=8, stride=4, train_fraction=0.8, val_fraction=0.1)
    # val region of a 20-frame clip is [16, 18): shorter than window_size
    val_windows = plan_temporal_windows(20, "val", config)
    assert val_windows == [(16, 18)]


def test_plan_windows_empty_region_yields_no_windows() -> None:
    config = WindowingConfig(window_size=8, stride=4, train_fraction=1.0, val_fraction=0.0)
    assert plan_temporal_windows(20, "val", config) == []
    assert plan_temporal_windows(0, "train", config) == []


def test_windowing_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        WindowingConfig(window_size=0, stride=4)
    with pytest.raises(ValueError):
        WindowingConfig(window_size=8, stride=0)
    with pytest.raises(ValueError):
        WindowingConfig(window_size=8, stride=4, train_fraction=0.8, val_fraction=0.5)
    with pytest.raises(ValueError):
        WindowingConfig(window_size=8, stride=4, boundary_gap=-1)


def test_unknown_split_raises() -> None:
    config = WindowingConfig(window_size=8, stride=4)
    with pytest.raises(ValueError):
        temporal_split_region(100, "holdout", config)


def _write_synthetic_clip(
    tmp_path: Path, clip_id: str, num_frames: int
) -> tuple[Path, dict]:
    """Write a minimal processed clip whose aligned sequence has num_frames frames."""
    np = pytest.importorskip("numpy")
    processed_root = tmp_path / "data" / "processed" / "demo_id"
    clip_dir = processed_root / clip_id
    face_crop_dir = clip_dir / "face_crops"
    face_crop_dir.mkdir(parents=True)
    (face_crop_dir / "000000.png").write_text("png", encoding="utf-8")
    (clip_dir / "audio.wav").write_text("audio", encoding="utf-8")

    fps = 25.0
    times = (np.arange(num_frames, dtype=np.float32) / fps)
    # identity alignment: audio time axis matches motion frame times exactly
    np.savez(
        clip_dir / "audio_features.npz",
        time_axis_sec=times,
        rms_energy=times,
        log_rms_energy=times,
        zero_crossing_rate=times,
        peak_amplitude=times,
        spectral_centroid_hz=times * 100.0,
    )
    np.savez(
        clip_dir / "motion_features.npz",
        output_fps=np.asarray(fps, dtype=np.float32),
        motion_vector=np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3),
    )

    record = {
        "clip_id": clip_id,
        "identity_id": "demo_id",
        "audio_path": str(clip_dir / "audio.wav"),
        "face_crop_dir": str(face_crop_dir),
        "landmarks_path": None,
        "head_pose_path": None,
        "expression_path": None,
        "motion_template_path": None,
        "audio_features_path": str(clip_dir / "audio_features.npz"),
        "motion_features_path": str(clip_dir / "motion_features.npz"),
        "fps": fps,
        "duration_sec": num_frames / fps,
        "num_frames": num_frames,
        "face_detection_rate": 1.0,
        "avg_yaw_abs": None,
        "avg_pitch_abs": None,
        "avg_roll_abs": None,
        "audio_sample_rate": 16000,
        "split": "train",
    }
    return processed_root, record


def test_windowed_dataset_enumerates_windows_within_split_region(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from avagen.data.dataset import WindowedAudioMotionDataset, collate_padded_sequences

    processed_root, record = _write_synthetic_clip(tmp_path, "clip_a", num_frames=20)
    manifest_path = processed_root / "manifest.jsonl"
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    config = WindowingConfig(window_size=8, stride=4, train_fraction=0.8, val_fraction=0.1)

    train_ds = WindowedAudioMotionDataset(manifest_path, "train", config)
    # train region [0, 16): starts 0,4,8 -> (0,8),(4,12),(8,16)
    assert len(train_ds) == 3
    first = train_ds[0]
    assert first.audio_features.shape == (8, 5)
    assert first.motion_features.shape == (8, 3)
    assert first.split == "train"

    val_ds = WindowedAudioMotionDataset(manifest_path, "val", config)
    # val region [16, 18): single short window
    assert len(val_ds) == 1
    assert val_ds[0].audio_features.shape[0] == 2

    test_ds = WindowedAudioMotionDataset(manifest_path, "test", config)
    # test region [18, 20): single short window
    assert len(test_ds) == 1
    assert test_ds[0].motion_features.shape[0] == 2

    # windows are batchable via the shared collate (all train windows share length 8)
    batch = collate_padded_sequences([train_ds[0], train_ds[1]])
    assert batch["audio_features"].shape == (2, 8, 5)
    assert batch["motion_features"].shape == (2, 8, 3)
