"""LivePortrait-backed motion-template extraction for processed clips."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from avagen.data.manifests import refresh_identity_manifest
from avagen.renderers.liveportrait_wrapper import (
    LivePortraitRunConfig,
    LivePortraitRunResult,
    run_liveportrait_inference,
)
from avagen.utils.paths import ensure_dir, to_repo_relative


MotionTemplateRunner = Callable[[LivePortraitRunConfig, bool], LivePortraitRunResult]


@dataclass
class MotionTemplateExtractionConfig:
    manifest_path: Path
    liveportrait_root: Path
    python_executable: str
    inference_script: Path | None = None
    work_root: Path | None = None
    output_field: str = "motion_template_path"
    driving_source: str = "source_video"
    skip_render: bool = True
    extra_args: Sequence[str] = field(default_factory=tuple)
    clip_ids: tuple[str, ...] = ()
    overwrite: bool = False


def _load_manifest_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _load_metadata(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_manifest_path_reference(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [manifest_path.parent, *manifest_path.parents]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (manifest_path.parent / candidate).resolve()


def _resolve_clip_dir(manifest_path: Path, record: dict[str, object]) -> Path:
    face_crop_dir = _resolve_manifest_path_reference(manifest_path, str(record["face_crop_dir"]))
    return face_crop_dir.parent


def _resolve_source_image(manifest_path: Path, record: dict[str, object]) -> Path:
    face_crop_dir = _resolve_manifest_path_reference(manifest_path, str(record["face_crop_dir"]))
    images = sorted(face_crop_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No face crop images found in {face_crop_dir}")
    return images[0]


def _resolve_face_crop_dir(manifest_path: Path, record: dict[str, object]) -> Path:
    return _resolve_manifest_path_reference(manifest_path, str(record["face_crop_dir"]))


def _build_face_crop_video(
    manifest_path: Path,
    record: dict[str, object],
    metadata: dict[str, object],
    work_dir: Path,
) -> Path:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is required to build a face-crop driving video.")

    face_crop_dir = _resolve_face_crop_dir(manifest_path, record)
    input_pattern = face_crop_dir / "%06d.png"
    if not face_crop_dir.exists():
        raise FileNotFoundError(f"Face crop directory not found: {face_crop_dir}")
    if not next(face_crop_dir.glob("*.png"), None):
        raise FileNotFoundError(f"No face crop images found in {face_crop_dir}")

    preprocessing = metadata.get("preprocessing", {})
    fps_value = None
    if isinstance(preprocessing, dict):
        fps_value = preprocessing.get("target_fps_effective")
    if fps_value in {None, 0, 0.0}:
        fps_value = record.get("fps")
    fps = float(fps_value or 25.0)

    staged_path = work_dir / f"{record['clip_id']}_face_crops.mp4"
    command = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-start_number",
        "0",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(input_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(staged_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed while building face-crop video for {record['clip_id']}: {stderr}")
    return staged_path


def _stage_driving_video(source_video_path: Path, work_dir: Path) -> Path:
    staged_path = work_dir / source_video_path.name
    if staged_path.exists() or staged_path.is_symlink():
        staged_path.unlink()
    shutil.copy2(source_video_path, staged_path)
    return staged_path


def _extract_single_motion_template(
    manifest_path: Path,
    record: dict[str, object],
    config: MotionTemplateExtractionConfig,
    runner: MotionTemplateRunner,
    dry_run: bool,
) -> dict[str, object]:
    clip_dir = _resolve_clip_dir(manifest_path, record)
    metadata_path = clip_dir / "metadata.json"
    metadata = _load_metadata(metadata_path)
    source_video_path = Path(str(metadata["source_video_path"])).expanduser().resolve()
    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found for clip {record['clip_id']}: {source_video_path}")

    source_image = _resolve_source_image(manifest_path, record)
    work_root = config.work_root if config.work_root is not None else clip_dir / ".motion_template_work"
    work_dir = ensure_dir(work_root / str(record["identity_id"]) / str(record["clip_id"]))
    if config.driving_source == "source_video":
        staged_driving = _stage_driving_video(source_video_path, work_dir)
    elif config.driving_source == "face_crop_video":
        staged_driving = _build_face_crop_video(manifest_path, record, metadata, work_dir)
    else:
        raise ValueError(f"Unsupported driving_source: {config.driving_source}")
    output_dir = ensure_dir(work_dir / "output")
    template_target = clip_dir / "motion_template.pkl"
    generated_template = staged_driving.with_suffix(".pkl")

    if template_target.exists():
        if not config.overwrite:
            print(
                f"[extract_motion] clip_id={record['clip_id']} status=skipped_existing template_path={template_target}",
                flush=True,
            )
            return {
                "clip_id": record["clip_id"],
                "identity_id": record["identity_id"],
                "template_path": str(template_target),
                "status": "skipped_existing",
            }
        template_target.unlink()
    if generated_template.exists() or generated_template.is_symlink():
        generated_template.unlink()

    run_result = runner(
        LivePortraitRunConfig(
            source_path=source_image,
            driving_path=staged_driving,
            output_dir=output_dir,
            liveportrait_root=config.liveportrait_root,
            inference_script=config.inference_script,
            python_executable=config.python_executable,
            extra_args=config.extra_args,
            stop_when_file=generated_template if config.skip_render else None,
        ),
        dry_run=dry_run,
    )
    if dry_run:
        return {
            "clip_id": record["clip_id"],
            "identity_id": record["identity_id"],
            "status": "dry_run",
            "template_path": str(template_target),
            "command": run_result.command,
        }

    if not generated_template.exists():
        raise FileNotFoundError(f"Expected LivePortrait motion template not found: {generated_template}")

    shutil.copy2(generated_template, template_target)

    optional_artifacts = metadata.setdefault("optional_artifacts", {})
    if not isinstance(optional_artifacts, dict):
        raise ValueError(f"Invalid optional_artifacts in {metadata_path}")
    optional_artifacts[config.output_field] = to_repo_relative(template_target)
    metadata.setdefault("motion_template", {})
    if not isinstance(metadata["motion_template"], dict):
        metadata["motion_template"] = {}
    metadata["motion_template"] = {
        "template_path": to_repo_relative(template_target),
        "source_image": to_repo_relative(source_image),
        "driving_source": config.driving_source,
        "staged_driving_path": str(staged_driving),
        "work_dir": str(work_dir),
        "output_dir": str(output_dir),
    }
    _write_metadata(metadata_path, metadata)

    print(
        f"[extract_motion] clip_id={record['clip_id']} status=completed template_path={template_target}",
        flush=True,
    )

    return {
        "clip_id": record["clip_id"],
        "identity_id": record["identity_id"],
        "status": "completed",
        "template_path": str(template_target),
        "work_dir": str(work_dir),
        "output_dir": str(output_dir),
        "command": run_result.command,
    }


def extract_motion_templates(
    config: MotionTemplateExtractionConfig,
    dry_run: bool = False,
    runner: MotionTemplateRunner = run_liveportrait_inference,
) -> dict[str, object]:
    manifest_path = config.manifest_path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = _load_manifest_records(manifest_path)
    selected_clip_ids = set(config.clip_ids)
    selected_records = []
    for record in records:
        clip_id = str(record["clip_id"])
        if selected_clip_ids and clip_id not in selected_clip_ids:
            continue
        selected_records.append(record)

    clip_results = []
    total_clips = len(selected_records)
    for index, record in enumerate(selected_records, start=1):
        print(
            f"[extract_motion] clip_id={record['clip_id']} status=starting index={index}/{total_clips}",
            flush=True,
        )
        clip_results.append(_extract_single_motion_template(manifest_path, record, config, runner, dry_run))

    if dry_run:
        return {
            "manifest_path": str(manifest_path),
            "status": "dry_run",
            "processed_clips": clip_results,
        }

    identity_dir = manifest_path.parent
    refreshed_records, refreshed_manifest_path, report_path = refresh_identity_manifest(identity_dir)
    return {
        "manifest_path": str(refreshed_manifest_path),
        "report_path": str(report_path),
        "processed_clips": clip_results,
        "num_manifest_records": len(refreshed_records),
        "status": "completed",
    }
