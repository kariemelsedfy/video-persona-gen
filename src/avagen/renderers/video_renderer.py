"""Renderer orchestration helpers for model-generated motion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from avagen.data.dataset import ProcessedClipRecord, load_processed_clip_records
from avagen.inference.generate_motion import predict_motion_for_manifest
from avagen.renderers.liveportrait_wrapper import (
    LivePortraitRunConfig,
    run_liveportrait_inference,
)


@dataclass(frozen=True)
class PredictedMotionRenderConfig:
    checkpoint_path: Path
    model_config_path: Path
    manifest_path: Path
    output_root: Path
    liveportrait_root: Path
    inference_script: Path | None = None
    python_executable: str = "python"
    source_path: Path | None = None
    predicted_output_root: Path | None = None
    clip_ids: tuple[str, ...] = ()
    device: str = "auto"
    extra_args: Sequence[str] = field(default_factory=tuple)


def _resolve_source_image(record: ProcessedClipRecord, override_source_path: Path | None) -> Path:
    if override_source_path is not None:
        resolved = override_source_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Source image not found: {resolved}")
        return resolved

    images = sorted(record.face_crop_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No face crop images found for clip {record.clip_id} in {record.face_crop_dir}")
    return images[0].resolve()


def render_predicted_motion_for_manifest(config: PredictedMotionRenderConfig) -> dict[str, Any]:
    manifest_path = config.manifest_path.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    predicted_output_root = (
        config.predicted_output_root.expanduser().resolve()
        if config.predicted_output_root is not None
        else (output_root / "predicted_motion").resolve()
    )
    render_output_root = (output_root / "renders").resolve()

    prediction_summary = predict_motion_for_manifest(
        checkpoint_path=config.checkpoint_path,
        config_path=config.model_config_path,
        manifest_path=manifest_path,
        output_root=predicted_output_root,
        clip_ids=config.clip_ids,
        device=config.device,
    )

    records = load_processed_clip_records(manifest_path)
    record_by_clip_id = {record.clip_id: record for record in records}
    rendered_records: list[dict[str, Any]] = []
    for predicted_record in prediction_summary["predicted_records"]:
        clip_id = str(predicted_record["clip_id"])
        record = record_by_clip_id.get(clip_id)
        if record is None:
            raise KeyError(f"Predicted clip {clip_id} was not found in manifest {manifest_path}")

        source_path = _resolve_source_image(record, config.source_path)
        output_dir = render_output_root / record.identity_id / record.clip_id
        run_result = run_liveportrait_inference(
            LivePortraitRunConfig(
                source_path=source_path,
                driving_path=Path(str(predicted_record["predicted_motion_template_path"])).expanduser().resolve(),
                output_dir=output_dir,
                liveportrait_root=config.liveportrait_root.expanduser().resolve(),
                inference_script=config.inference_script.expanduser().resolve() if config.inference_script else None,
                python_executable=config.python_executable,
                extra_args=config.extra_args,
            )
        )
        rendered_records.append(
            {
                "clip_id": record.clip_id,
                "identity_id": record.identity_id,
                "source_path": str(source_path),
                "driving_template_path": str(predicted_record["predicted_motion_template_path"]),
                "output_dir": str(output_dir.resolve()),
                "rendered_files": sorted(str(path.resolve()) for path in output_dir.glob("*")),
                "command": run_result.command,
            }
        )

    return {
        "status": "completed",
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "predicted_output_root": str(predicted_output_root),
        "render_output_root": str(render_output_root),
        "prediction_summary": prediction_summary,
        "rendered_records": rendered_records,
    }
