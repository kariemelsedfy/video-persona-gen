"""Predict motion features from audio using a trained GRU checkpoint."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from avagen.data.dataset import (
    load_aligned_audio_motion_sequence,
    load_motion_features,
    load_processed_clip_records,
)
from avagen.features.motion_features import (
    motion_feature_bundle_to_template,
    save_motion_feature_bundle,
    summarize_motion_features,
    unflatten_motion_vector,
)
from avagen.features.motion_postprocess import apply_motion_postprocess
from avagen.models.motion_gru import MotionGRU, MotionGRUConfig
from avagen.training.checkpointing import load_checkpoint
from avagen.utils.config import load_config
from avagen.utils.device import detect_torch_device


def _resolve_device(device_name: str) -> torch.device:
    selected = detect_torch_device() if device_name == "auto" else device_name
    return torch.device(selected)


def _load_audio_motion_names(config_path: str | Path) -> tuple[tuple[str, ...], str]:
    config_data = load_config(config_path)
    if not isinstance(config_data, dict):
        raise ValueError(f"Expected config mapping in {config_path}")
    dataset_config = config_data.get("dataset", {})
    if dataset_config is None:
        dataset_config = {}
    if not isinstance(dataset_config, dict):
        raise ValueError(f"Expected dataset config mapping in {config_path}")
    audio_feature_names = tuple(
        dataset_config.get(
            "audio_feature_names",
            [
                "rms_energy",
                "log_rms_energy",
                "zero_crossing_rate",
                "peak_amplitude",
                "spectral_centroid_hz",
            ],
        )
    )
    motion_feature_name = str(dataset_config.get("motion_feature_name", "motion_vector"))
    return audio_feature_names, motion_feature_name


def _load_postprocess(config_path: str | Path) -> dict[str, Any]:
    config_data = load_config(config_path)
    if isinstance(config_data, dict):
        section = config_data.get("motion_postprocess")
        if isinstance(section, dict):
            return section
    return {}


def load_motion_predictor(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
) -> tuple[MotionGRU, dict[str, Any], torch.device]:
    torch_device = _resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=torch_device)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a model_config mapping.")

    model_type = str(checkpoint.get("model_type", "gru"))
    if model_type == "flow":
        from avagen.models.motion_flow import MotionFlowConfig, MotionFlowModel

        model = MotionFlowModel(MotionFlowConfig(**model_config)).to(torch_device)
    else:
        model = MotionGRU(MotionGRUConfig(**model_config)).to(torch_device)
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a model_state_dict mapping.")
    model.load_state_dict(model_state)
    model.eval()
    return model, checkpoint, torch_device


def predict_motion_for_manifest(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    output_root: str | Path,
    clip_ids: Sequence[str] = (),
    device: str = "auto",
) -> dict[str, Any]:
    model, checkpoint, torch_device = load_motion_predictor(checkpoint_path, device=device)
    audio_feature_names, motion_feature_name = _load_audio_motion_names(config_path)
    postprocess = _load_postprocess(config_path)
    model_type = str(checkpoint.get("model_type", "gru"))
    flow_steps = int(checkpoint.get("flow_sample_steps", 20))

    normalization = checkpoint.get("motion_normalization")
    motion_mean: np.ndarray | None = None
    motion_std: np.ndarray | None = None
    if isinstance(normalization, dict):
        motion_mean = np.asarray(normalization["mean"], dtype=np.float32)
        motion_std = np.asarray(normalization["std"], dtype=np.float32)
    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    selected_clip_ids = set(clip_ids)

    records = [
        record
        for record in load_processed_clip_records(manifest)
        if record.audio_features_path is not None and record.motion_features_path is not None
    ]
    predicted_records: list[dict[str, Any]] = []

    with torch.no_grad():
        for record in records:
            if selected_clip_ids and record.clip_id not in selected_clip_ids:
                continue

            sequence = load_aligned_audio_motion_sequence(
                record,
                audio_feature_names=audio_feature_names,
                motion_feature_name=motion_feature_name,
            )
            reference_bundle = load_motion_features(record)
            inputs = torch.from_numpy(sequence.audio_features[None, ...]).to(torch_device)
            if model_type == "flow":
                from avagen.models.motion_flow import sample_motion

                predicted_vector = sample_motion(model, inputs, steps=flow_steps).detach().cpu().numpy()[0]
            else:
                lengths = torch.tensor([sequence.audio_features.shape[0]], dtype=torch.long, device=torch_device)
                predicted_vector = model(inputs, lengths=lengths).detach().cpu().numpy()[0]
            if motion_mean is not None and motion_std is not None:
                # Model predicts in standardized space; map back to raw motion units.
                predicted_vector = predicted_vector * motion_std + motion_mean
            predicted_bundle = unflatten_motion_vector(predicted_vector, reference_bundle)
            if postprocess:
                fps = float(record.fps) if record.fps else float(np.asarray(reference_bundle["output_fps"]).item())
                predicted_bundle = apply_motion_postprocess(
                    predicted_bundle, postprocess, fps, reference_bundle=reference_bundle
                )
            template = motion_feature_bundle_to_template(predicted_bundle)

            clip_output_dir = root / record.identity_id / record.clip_id
            features_path = clip_output_dir / "predicted_motion_features.npz"
            summary_path = clip_output_dir / "predicted_motion_summary.json"
            template_path = clip_output_dir / "predicted_motion_template.pkl"
            save_motion_feature_bundle(predicted_bundle, features_path)
            summary = summarize_motion_features(predicted_bundle)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            with template_path.open("wb") as handle:
                pickle.dump(template, handle)

            predicted_records.append(
                {
                    "clip_id": record.clip_id,
                    "identity_id": record.identity_id,
                    "num_frames": int(predicted_vector.shape[0]),
                    "feature_dim": int(predicted_vector.shape[1]),
                    "predicted_motion_features_path": str(features_path),
                    "predicted_motion_summary_path": str(summary_path),
                    "predicted_motion_template_path": str(template_path),
                }
            )

    return {
        "status": "completed",
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "manifest_path": str(manifest),
        "output_root": str(root),
        "device": str(torch_device),
        "model_epoch": checkpoint.get("epoch"),
        "predicted_records": predicted_records,
    }
