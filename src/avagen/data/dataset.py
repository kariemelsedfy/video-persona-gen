"""Manifest-backed dataset loading utilities."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from avagen.data.windowing import WindowingConfig, plan_temporal_windows


@dataclass(frozen=True)
class ProcessedClipRecord:
    clip_id: str
    identity_id: str
    split: str
    audio_path: Path
    face_crop_dir: Path
    frame_metadata_path: Path
    metadata_path: Path
    motion_template_path: Path | None
    audio_features_path: Path | None
    prosody_summary_path: Path | None
    motion_features_path: Path | None
    motion_summary_path: Path | None
    fps: float
    duration_sec: float
    num_frames: int
    face_detection_rate: float


@dataclass(frozen=True)
class AudioMotionSequence:
    clip_id: str
    identity_id: str
    split: str
    audio_features: np.ndarray
    motion_features: np.ndarray
    audio_feature_names: tuple[str, ...]
    motion_feature_name: str
    time_axis_sec: np.ndarray
    motion_fps: float


def _resolve_from_manifest(manifest_path: Path, raw_path: str | None) -> Path | None:
    if raw_path in {None, ""}:
        return None

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [manifest_path.parent, *manifest_path.parents]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (manifest_path.parent / candidate).resolve()


def load_processed_clip_records(
    manifest_path: str | Path,
    splits: Sequence[str] | None = None,
    require_motion_template: bool = False,
    limit: int | None = None,
) -> list[ProcessedClipRecord]:
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    selected_splits = set(splits or [])
    records: list[ProcessedClipRecord] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        split = str(payload.get("split", "train"))
        if selected_splits and split not in selected_splits:
            continue

        face_crop_dir = _resolve_from_manifest(manifest, str(payload["face_crop_dir"]))
        if face_crop_dir is None:
            raise FileNotFoundError(f"Missing face_crop_dir in manifest record: {payload}")
        clip_dir = face_crop_dir.parent
        frame_metadata_path = clip_dir / "frame_metadata.json"
        metadata_path = clip_dir / "metadata.json"
        motion_template_path = _resolve_from_manifest(manifest, payload.get("motion_template_path"))
        if require_motion_template and motion_template_path is None:
            continue

        records.append(
            ProcessedClipRecord(
                clip_id=str(payload["clip_id"]),
                identity_id=str(payload["identity_id"]),
                split=split,
                audio_path=_resolve_from_manifest(manifest, str(payload["audio_path"])) or Path(str(payload["audio_path"])),
                face_crop_dir=face_crop_dir,
                frame_metadata_path=frame_metadata_path.resolve(),
                metadata_path=metadata_path.resolve(),
                motion_template_path=motion_template_path.resolve() if motion_template_path else None,
                audio_features_path=_resolve_from_manifest(manifest, payload.get("audio_features_path")),
                prosody_summary_path=_resolve_from_manifest(manifest, payload.get("prosody_summary_path")),
                motion_features_path=_resolve_from_manifest(manifest, payload.get("motion_features_path")),
                motion_summary_path=_resolve_from_manifest(manifest, payload.get("motion_summary_path")),
                fps=float(payload.get("fps") or 0.0),
                duration_sec=float(payload.get("duration_sec") or 0.0),
                num_frames=int(payload.get("num_frames") or 0),
                face_detection_rate=float(payload.get("face_detection_rate") or 0.0),
            )
        )
        if limit is not None and len(records) >= limit:
            break

    return records


def load_frame_metadata(record: ProcessedClipRecord) -> list[dict[str, Any]]:
    if not record.frame_metadata_path.exists():
        raise FileNotFoundError(f"Frame metadata not found: {record.frame_metadata_path}")
    return json.loads(record.frame_metadata_path.read_text(encoding="utf-8"))


def load_clip_metadata(record: ProcessedClipRecord) -> dict[str, Any]:
    if not record.metadata_path.exists():
        raise FileNotFoundError(f"Clip metadata not found: {record.metadata_path}")
    return json.loads(record.metadata_path.read_text(encoding="utf-8"))


def load_motion_template(record: ProcessedClipRecord) -> dict[str, Any]:
    if record.motion_template_path is None:
        raise FileNotFoundError(f"No motion template path recorded for clip {record.clip_id}")
    if not record.motion_template_path.exists():
        raise FileNotFoundError(f"Motion template not found: {record.motion_template_path}")
    with record.motion_template_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected motion template payload for {record.motion_template_path}")
    return payload


def load_audio_features(record: ProcessedClipRecord) -> dict[str, Any]:
    if record.audio_features_path is None:
        raise FileNotFoundError(f"No audio features path recorded for clip {record.clip_id}")
    if not record.audio_features_path.exists():
        raise FileNotFoundError(f"Audio features not found: {record.audio_features_path}")
    with np.load(record.audio_features_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def load_motion_features(record: ProcessedClipRecord) -> dict[str, Any]:
    if record.motion_features_path is None:
        raise FileNotFoundError(f"No motion features path recorded for clip {record.clip_id}")
    if not record.motion_features_path.exists():
        raise FileNotFoundError(f"Motion features not found: {record.motion_features_path}")
    with np.load(record.motion_features_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _flatten_feature_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array[:, None]
    return array.reshape(array.shape[0], -1)


def _validate_feature_length(
    feature_name: str,
    values: np.ndarray,
    expected_length: int,
    clip_id: str,
) -> np.ndarray:
    matrix = _flatten_feature_array(values)
    if matrix.shape[0] != expected_length:
        raise ValueError(
            f"Feature '{feature_name}' for clip {clip_id} has {matrix.shape[0]} frames, expected {expected_length}."
        )
    return matrix


def _interpolate_feature_matrix(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    if source_times.ndim != 1:
        raise ValueError("Expected a one-dimensional source time axis.")
    if source_times.shape[0] != source_values.shape[0]:
        raise ValueError("Source time axis and feature matrix must have the same number of rows.")
    if source_values.shape[0] == 0:
        return np.zeros((target_times.shape[0], source_values.shape[1]), dtype=np.float32)
    if source_values.shape[0] == 1:
        return np.repeat(source_values.astype(np.float32), target_times.shape[0], axis=0)

    aligned_columns = [
        np.interp(target_times, source_times, source_values[:, column]).astype(np.float32)
        for column in range(source_values.shape[1])
    ]
    return np.stack(aligned_columns, axis=1).astype(np.float32)


def load_aligned_audio_motion_sequence(
    record: ProcessedClipRecord,
    audio_feature_names: Sequence[str] = (
        "rms_energy",
        "log_rms_energy",
        "zero_crossing_rate",
        "peak_amplitude",
        "spectral_centroid_hz",
    ),
    motion_feature_name: str = "motion_vector",
) -> AudioMotionSequence:
    if record.audio_features_path is None:
        raise FileNotFoundError(f"No audio features path recorded for clip {record.clip_id}")
    if record.motion_features_path is None:
        raise FileNotFoundError(f"No motion features path recorded for clip {record.clip_id}")
    if not audio_feature_names:
        raise ValueError("Expected at least one audio feature name.")

    audio_bundle = load_audio_features(record)
    motion_bundle = load_motion_features(record)
    if "time_axis_sec" not in audio_bundle:
        raise KeyError(f"Missing audio feature 'time_axis_sec' for clip {record.clip_id}")
    if motion_feature_name not in motion_bundle:
        raise KeyError(f"Missing motion feature '{motion_feature_name}' for clip {record.clip_id}")
    if "output_fps" not in motion_bundle:
        raise KeyError(f"Missing motion feature 'output_fps' for clip {record.clip_id}")

    source_times = np.asarray(audio_bundle["time_axis_sec"], dtype=np.float32)
    motion_values = _flatten_feature_array(np.asarray(motion_bundle[motion_feature_name], dtype=np.float32))
    motion_fps = float(np.asarray(motion_bundle["output_fps"]).item())
    # LivePortrait stores output_fps as an integer, truncating fractional rates
    # (e.g. 23.976 -> 23). Using that to place motion frames on the timeline drifts
    # audio vs motion by seconds over a clip and scrambles the audio->motion labels,
    # so prefer the clip's true (fractional) fps from the manifest when available.
    if record.fps and record.fps > 0:
        motion_fps = float(record.fps)
    target_times = (np.arange(motion_values.shape[0], dtype=np.float32) / max(motion_fps, 1e-8)).astype(np.float32)

    aligned_audio_columns = []
    for feature_name in audio_feature_names:
        if feature_name not in audio_bundle:
            raise KeyError(f"Missing audio feature '{feature_name}' for clip {record.clip_id}")
        feature_matrix = _validate_feature_length(
            feature_name,
            np.asarray(audio_bundle[feature_name], dtype=np.float32),
            expected_length=source_times.shape[0],
            clip_id=record.clip_id,
        )
        aligned_audio_columns.append(_interpolate_feature_matrix(source_times, feature_matrix, target_times))
    aligned_audio = np.concatenate(aligned_audio_columns, axis=1).astype(np.float32)

    return AudioMotionSequence(
        clip_id=record.clip_id,
        identity_id=record.identity_id,
        split=record.split,
        audio_features=aligned_audio,
        motion_features=motion_values,
        audio_feature_names=tuple(audio_feature_names),
        motion_feature_name=motion_feature_name,
        time_axis_sec=target_times,
        motion_fps=motion_fps,
    )


def collate_padded_sequences(sequences: Sequence[AudioMotionSequence]) -> dict[str, Any]:
    if not sequences:
        raise ValueError("Expected at least one sequence to collate.")

    max_length = max(sequence.audio_features.shape[0] for sequence in sequences)
    audio_dim = sequences[0].audio_features.shape[1]
    motion_dim = sequences[0].motion_features.shape[1]

    audio_batch = np.zeros((len(sequences), max_length, audio_dim), dtype=np.float32)
    motion_batch = np.zeros((len(sequences), max_length, motion_dim), dtype=np.float32)
    mask = np.zeros((len(sequences), max_length), dtype=bool)
    lengths = np.zeros((len(sequences),), dtype=np.int32)

    for index, sequence in enumerate(sequences):
        length = sequence.audio_features.shape[0]
        if sequence.motion_features.shape[0] != length:
            raise ValueError(f"Sequence {sequence.clip_id} has mismatched audio and motion lengths.")
        if sequence.audio_features.shape[1] != audio_dim:
            raise ValueError(f"Sequence {sequence.clip_id} has audio dim {sequence.audio_features.shape[1]}, expected {audio_dim}.")
        if sequence.motion_features.shape[1] != motion_dim:
            raise ValueError(
                f"Sequence {sequence.clip_id} has motion dim {sequence.motion_features.shape[1]}, expected {motion_dim}."
            )
        audio_batch[index, :length] = sequence.audio_features
        motion_batch[index, :length] = sequence.motion_features
        mask[index, :length] = True
        lengths[index] = length

    return {
        "audio_features": audio_batch,
        "motion_features": motion_batch,
        "mask": mask,
        "lengths": lengths,
        "clip_ids": [sequence.clip_id for sequence in sequences],
        "identity_ids": [sequence.identity_id for sequence in sequences],
        "splits": [sequence.split for sequence in sequences],
    }


class ProcessedClipDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        splits: Sequence[str] | None = None,
        require_motion_template: bool = False,
        limit: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.records = load_processed_clip_records(
            self.manifest_path,
            splits=splits,
            require_motion_template=require_motion_template,
            limit=limit,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ProcessedClipRecord:
        return self.records[index]

    def clip_ids(self) -> list[str]:
        return [record.clip_id for record in self.records]

    def identity_ids(self) -> list[str]:
        return sorted({record.identity_id for record in self.records})


class AudioMotionSequenceDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        splits: Sequence[str] | None = None,
        audio_feature_names: Sequence[str] = (
            "rms_energy",
            "log_rms_energy",
            "zero_crossing_rate",
            "peak_amplitude",
            "spectral_centroid_hz",
        ),
        motion_feature_name: str = "motion_vector",
        limit: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.audio_feature_names = tuple(audio_feature_names)
        self.motion_feature_name = motion_feature_name
        self.records = [
            record
            for record in load_processed_clip_records(self.manifest_path, splits=splits, limit=limit)
            if record.audio_features_path is not None and record.motion_features_path is not None
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AudioMotionSequence:
        return load_aligned_audio_motion_sequence(
            self.records[index],
            audio_feature_names=self.audio_feature_names,
            motion_feature_name=self.motion_feature_name,
        )

    def clip_ids(self) -> list[str]:
        return [record.clip_id for record in self.records]

    def identity_ids(self) -> list[str]:
        return sorted({record.identity_id for record in self.records})


class WindowedAudioMotionDataset:
    """Windowed, within-clip temporally-split view over aligned audio-motion data.

    Loads every clip's full aligned sequence once (ignoring the manifest's
    clip-level ``split`` field) and exposes fixed-length windows drawn from the
    requested split's temporal region of each clip.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        windowing: WindowingConfig,
        audio_feature_names: Sequence[str] = (
            "rms_energy",
            "log_rms_energy",
            "zero_crossing_rate",
            "peak_amplitude",
            "spectral_centroid_hz",
        ),
        motion_feature_name: str = "motion_vector",
        limit: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = split
        self.windowing = windowing
        self.audio_feature_names = tuple(audio_feature_names)
        self.motion_feature_name = motion_feature_name

        records = [
            record
            for record in load_processed_clip_records(self.manifest_path, splits=None, limit=limit)
            if record.audio_features_path is not None and record.motion_features_path is not None
        ]
        self._sequences: list[AudioMotionSequence] = []
        self._windows: list[tuple[int, int, int]] = []
        for seq_index, record in enumerate(records):
            sequence = load_aligned_audio_motion_sequence(
                record,
                audio_feature_names=self.audio_feature_names,
                motion_feature_name=self.motion_feature_name,
            )
            self._sequences.append(sequence)
            n_frames = int(sequence.audio_features.shape[0])
            for start, end in plan_temporal_windows(n_frames, split, windowing):
                self._windows.append((seq_index, start, end))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> AudioMotionSequence:
        seq_index, start, end = self._windows[index]
        sequence = self._sequences[seq_index]
        return AudioMotionSequence(
            clip_id=sequence.clip_id,
            identity_id=sequence.identity_id,
            split=self.split,
            audio_features=sequence.audio_features[start:end],
            motion_features=sequence.motion_features[start:end],
            audio_feature_names=sequence.audio_feature_names,
            motion_feature_name=sequence.motion_feature_name,
            time_axis_sec=sequence.time_axis_sec[start:end],
            motion_fps=sequence.motion_fps,
        )

    def clip_ids(self) -> list[str]:
        return [self._sequences[seq_index].clip_id for seq_index, _, _ in self._windows]

    def identity_ids(self) -> list[str]:
        return sorted({sequence.identity_id for sequence in self._sequences})
