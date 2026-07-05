"""Manifest-backed dataset loading utilities."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


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
    import numpy as np

    with np.load(record.audio_features_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def load_motion_features(record: ProcessedClipRecord) -> dict[str, Any]:
    if record.motion_features_path is None:
        raise FileNotFoundError(f"No motion features path recorded for clip {record.clip_id}")
    if not record.motion_features_path.exists():
        raise FileNotFoundError(f"Motion features not found: {record.motion_features_path}")
    import numpy as np

    with np.load(record.motion_features_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


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
