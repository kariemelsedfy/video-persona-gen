"""Manifest helpers for processed identity clip directories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ManifestRecord:
    clip_id: str
    identity_id: str
    audio_path: str
    face_crop_dir: str
    landmarks_path: str | None
    head_pose_path: str | None
    expression_path: str | None
    motion_template_path: str | None
    fps: float
    duration_sec: float
    num_frames: int
    face_detection_rate: float
    avg_yaw_abs: float | None
    avg_pitch_abs: float | None
    avg_roll_abs: float | None
    audio_sample_rate: int | None
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def refresh_identity_manifest(
    identity_dir: str | Path,
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> tuple[list[ManifestRecord], Path, Path]:
    root = Path(identity_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Identity directory not found: {root}")

    manifest_output = Path(manifest_path).expanduser().resolve() if manifest_path else root / "manifest.jsonl"
    report_output = Path(report_path).expanduser().resolve() if report_path else root / "dataset_report.json"

    records: list[ManifestRecord] = []
    split_counts: dict[str, int] = {}
    total_frames = 0
    total_duration_sec = 0.0

    for metadata_path in sorted(root.glob("*/metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts", {})
        optional_artifacts = payload.get("optional_artifacts", {})
        stats = payload.get("stats", {})
        preprocessing = payload.get("preprocessing", {})
        video_info = payload.get("video_info", {})
        split = str(payload.get("split", "train"))

        record = ManifestRecord(
            clip_id=str(payload["clip_id"]),
            identity_id=str(payload["identity_id"]),
            audio_path=str(artifacts["audio_path"]),
            face_crop_dir=str(artifacts["face_crop_dir"]),
            landmarks_path=optional_artifacts.get("landmarks_path"),
            head_pose_path=optional_artifacts.get("head_pose_path"),
            expression_path=optional_artifacts.get("expression_path"),
            motion_template_path=optional_artifacts.get("motion_template_path"),
            fps=float(preprocessing.get("target_fps_effective") or video_info.get("fps") or 0.0),
            duration_sec=float(stats.get("duration_sec") or 0.0),
            num_frames=int(stats.get("num_frames") or 0),
            face_detection_rate=float(stats.get("face_detection_rate") or 0.0),
            avg_yaw_abs=None,
            avg_pitch_abs=None,
            avg_roll_abs=None,
            audio_sample_rate=preprocessing.get("audio_sample_rate"),
            split=split,
        )
        records.append(record)
        split_counts[split] = split_counts.get(split, 0) + 1
        total_frames += record.num_frames
        total_duration_sec += record.duration_sec

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")

    report_payload = {
        "identity_id": root.name,
        "num_clips": len(records),
        "total_frames": total_frames,
        "total_duration_sec": total_duration_sec,
        "splits": split_counts,
        "manifest_path": str(manifest_output),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    return records, manifest_output, report_output
