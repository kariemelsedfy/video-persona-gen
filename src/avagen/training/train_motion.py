"""Training loop for the first audio-to-motion GRU baseline."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from avagen.data.dataset import (
    AudioMotionSequenceDataset,
    WindowedAudioMotionDataset,
    collate_padded_sequences,
)
from avagen.data.windowing import WindowingConfig
from avagen.models.losses import masked_mse_loss, masked_velocity_mse_loss
from avagen.models.motion_gru import MotionGRU, MotionGRUConfig
from avagen.training.checkpointing import load_checkpoint, save_checkpoint
from avagen.training.logging import append_jsonl, write_json
from avagen.utils.config import dump_json
from avagen.utils.device import detect_torch_device
from avagen.utils.seed import set_seed


def _resolve_device(device_name: str) -> torch.device:
    selected = detect_torch_device() if device_name == "auto" else device_name
    return torch.device(selected)


def _resolve_precision(precision: str, device: torch.device) -> tuple[str, torch.dtype | None]:
    normalized = precision.lower()
    if device.type != "cuda":
        return "fp32", None
    if normalized == "fp16":
        return normalized, torch.float16
    if normalized == "bf16":
        return normalized, torch.bfloat16
    return "fp32", None


def _sequence_collate(batch: list[Any]) -> dict[str, Any]:
    collated = collate_padded_sequences(batch)
    return {
        "audio_features": torch.from_numpy(collated["audio_features"]),
        "motion_features": torch.from_numpy(collated["motion_features"]),
        "mask": torch.from_numpy(collated["mask"]),
        "lengths": torch.as_tensor(collated["lengths"], dtype=torch.long),
        "clip_ids": collated["clip_ids"],
        "identity_ids": collated["identity_ids"],
        "splits": collated["splits"],
    }


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("audio_features", "motion_features", "mask", "lengths"):
        moved[key] = batch[key].to(device)
    return moved


def _compute_motion_normalization(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean/std of motion features over a dataset's frames.

    Standardizing the target makes every motion component (including the tiny
    lip/eye-ratio dims) contribute equally to the loss, instead of the loss
    being dominated by the large-scale components and collapsing the small ones.
    """
    frames = [np.asarray(dataset[index].motion_features, dtype=np.float64) for index in range(len(dataset))]
    if not frames:
        raise ValueError("Cannot compute motion normalization from an empty dataset.")
    stacked = np.concatenate(frames, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # leave constant dims unscaled
    return mean.astype(np.float32), std.astype(np.float32)


def _compute_batch_losses(
    model: MotionGRU,
    batch: dict[str, Any],
    velocity_loss_weight: float,
    motion_mean: torch.Tensor | None = None,
    motion_std: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    predictions = model(batch["audio_features"], lengths=batch["lengths"])
    targets = batch["motion_features"]
    if motion_mean is not None and motion_std is not None:
        # Model learns to predict in the standardized target space.
        targets = (targets - motion_mean) / motion_std
    reconstruction_loss = masked_mse_loss(predictions, targets, batch["mask"])
    velocity_loss = masked_velocity_mse_loss(predictions, targets, batch["mask"])
    total_loss = reconstruction_loss + velocity_loss_weight * velocity_loss
    return {
        "predictions": predictions,
        "reconstruction_loss": reconstruction_loss,
        "velocity_loss": velocity_loss,
        "total_loss": total_loss,
    }


def _run_epoch(
    model: MotionGRU,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    precision_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
    velocity_loss_weight: float,
    grad_clip_norm: float | None,
    max_batches: int | None = None,
    motion_mean: torch.Tensor | None = None,
    motion_std: torch.Tensor | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss_sum = 0.0
    reconstruction_sum = 0.0
    velocity_sum = 0.0
    total_valid_frames = 0
    total_steps = 0

    for step, batch in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break

        batch = _move_batch_to_device(batch, device)
        valid_frames = int(batch["mask"].sum().item())
        if valid_frames == 0:
            continue

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with (
            torch.autocast(device_type=device.type, dtype=precision_dtype)
            if precision_dtype is not None
            else nullcontext()
        ):
            loss_bundle = _compute_batch_losses(
                model,
                batch,
                velocity_loss_weight=velocity_loss_weight,
                motion_mean=motion_mean,
                motion_std=motion_std,
            )

        total_loss = loss_bundle["total_loss"]
        if is_training and optimizer is not None:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        total_loss_sum += float(total_loss.detach().cpu()) * valid_frames
        reconstruction_sum += float(loss_bundle["reconstruction_loss"].detach().cpu()) * valid_frames
        velocity_sum += float(loss_bundle["velocity_loss"].detach().cpu()) * valid_frames
        total_valid_frames += valid_frames
        total_steps += 1

    denominator = max(total_valid_frames, 1)
    return {
        "loss": total_loss_sum / denominator,
        "reconstruction_loss": reconstruction_sum / denominator,
        "velocity_loss": velocity_sum / denominator,
        "num_valid_frames": float(total_valid_frames),
        "num_steps": float(total_steps),
    }


def _parse_windowing_config(raw: Any) -> WindowingConfig | None:
    """Build a WindowingConfig from the ``dataset.windowing`` block, or None."""
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"dataset.windowing must be a mapping, got {type(raw).__name__}")
    if not raw.get("enabled", True):
        return None
    if "window_size" not in raw or "stride" not in raw:
        raise ValueError("dataset.windowing requires 'window_size' and 'stride'.")
    return WindowingConfig(
        window_size=int(raw["window_size"]),
        stride=int(raw["stride"]),
        train_fraction=float(raw.get("train_fraction", 0.8)),
        val_fraction=float(raw.get("val_fraction", 0.1)),
        boundary_gap=int(raw.get("boundary_gap", 0)),
    )


def _build_dataset(
    manifest_path: str | Path,
    *,
    splits: list[str],
    audio_feature_names: tuple[str, ...],
    motion_feature_name: str,
    limit: int | None,
) -> AudioMotionSequenceDataset:
    return AudioMotionSequenceDataset(
        manifest_path=manifest_path,
        splits=splits,
        audio_feature_names=audio_feature_names,
        motion_feature_name=motion_feature_name,
        limit=limit,
    )


def train_motion_model(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    manifest_path = Path(str(resolved["manifest_path"])).expanduser().resolve()
    experiment_dir = Path(str(resolved["experiment_dir"])).expanduser().resolve()
    dataset_config = dict(resolved.get("dataset", {}))
    model_config = dict(resolved.get("model", {}))
    training_config = dict(resolved.get("training", {}))
    seed = int(resolved.get("seed", 7))
    set_seed(seed)

    train_splits = list(dataset_config.get("train_splits", ["train"]))
    val_splits = list(dataset_config.get("val_splits", ["val"]))
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
    normalize_motion = bool(dataset_config.get("normalize_motion", False))
    limit = dataset_config.get("limit")
    if limit is not None:
        limit = int(limit)

    windowing_config = _parse_windowing_config(dataset_config.get("windowing"))

    if windowing_config is not None:
        train_dataset = WindowedAudioMotionDataset(
            manifest_path,
            "train",
            windowing_config,
            audio_feature_names=audio_feature_names,
            motion_feature_name=motion_feature_name,
            limit=limit,
        )
        val_dataset = WindowedAudioMotionDataset(
            manifest_path,
            "val",
            windowing_config,
            audio_feature_names=audio_feature_names,
            motion_feature_name=motion_feature_name,
            limit=limit,
        )
    else:
        train_dataset = _build_dataset(
            manifest_path,
            splits=train_splits,
            audio_feature_names=audio_feature_names,
            motion_feature_name=motion_feature_name,
            limit=limit,
        )
        val_dataset = _build_dataset(
            manifest_path,
            splits=val_splits,
            audio_feature_names=audio_feature_names,
            motion_feature_name=motion_feature_name,
            limit=limit,
        )
    if len(train_dataset) == 0:
        mode = "windowed within-clip" if windowing_config is not None else f"splits {train_splits}"
        raise ValueError(f"No training sequences available in {manifest_path} ({mode}).")

    sample = train_dataset[0]
    motion_gru_config = MotionGRUConfig(
        input_size=int(sample.audio_features.shape[1]),
        output_size=int(sample.motion_features.shape[1]),
        hidden_size=int(model_config.get("hidden_size", 256)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
        bidirectional=bool(model_config.get("bidirectional", False)),
    )

    device = _resolve_device(str(training_config.get("device", "auto")))
    precision_name, precision_dtype = _resolve_precision(str(training_config.get("precision", "fp32")), device)
    model = MotionGRU(motion_gru_config).to(device)

    motion_mean_np: np.ndarray | None = None
    motion_std_np: np.ndarray | None = None
    motion_mean_t: torch.Tensor | None = None
    motion_std_t: torch.Tensor | None = None
    if normalize_motion:
        motion_mean_np, motion_std_np = _compute_motion_normalization(train_dataset)
        motion_mean_t = torch.from_numpy(motion_mean_np).to(device)
        motion_std_t = torch.from_numpy(motion_std_np).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=precision_name == "fp16" and device.type == "cuda")

    batch_size = int(training_config.get("batch_size", 8))
    num_workers = int(training_config.get("num_workers", 0))
    epochs = int(training_config.get("epochs", 50))
    velocity_loss_weight = float(training_config.get("velocity_loss_weight", 0.1))
    grad_clip_norm_raw = training_config.get("grad_clip_norm", 1.0)
    grad_clip_norm = None if grad_clip_norm_raw is None else float(grad_clip_norm_raw)
    max_train_batches = training_config.get("max_train_batches")
    max_val_batches = training_config.get("max_val_batches")
    if max_train_batches is not None:
        max_train_batches = int(max_train_batches)
    if max_val_batches is not None:
        max_val_batches = int(max_val_batches)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_sequence_collate,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_sequence_collate,
        )
        if len(val_dataset) > 0
        else None
    )

    checkpoints_dir = experiment_dir / "checkpoints"
    metrics_dir = experiment_dir / "metrics"
    history_path = metrics_dir / "train_history.jsonl"
    summary_path = metrics_dir / "summary.json"
    resolved_config_path = experiment_dir / "config.resolved.json"
    dump_json(
        {
            "manifest_path": str(manifest_path),
            "experiment_dir": str(experiment_dir),
            "seed": seed,
            "dataset": {
                "train_splits": train_splits,
                "val_splits": val_splits,
                "audio_feature_names": list(audio_feature_names),
                "motion_feature_name": motion_feature_name,
                "normalize_motion": normalize_motion,
                "limit": limit,
                "windowing": asdict(windowing_config) if windowing_config is not None else None,
                "num_train_windows": len(train_dataset),
                "num_val_windows": len(val_dataset),
            },
            "model": asdict(motion_gru_config),
            "training": {
                "batch_size": batch_size,
                "learning_rate": float(training_config.get("learning_rate", 1e-4)),
                "weight_decay": float(training_config.get("weight_decay", 0.0)),
                "epochs": epochs,
                "device": str(device),
                "precision": precision_name,
                "grad_clip_norm": grad_clip_norm,
                "velocity_loss_weight": velocity_loss_weight,
                "num_workers": num_workers,
                "max_train_batches": max_train_batches,
                "max_val_batches": max_val_batches,
            },
        },
        resolved_config_path,
    )

    resume_from = training_config.get("resume_from")
    start_epoch = 1
    best_metric = float("inf")
    best_epoch = 0
    if resume_from:
        checkpoint = load_checkpoint(resume_from, model=model, optimizer=optimizer, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))

    last_checkpoint_path = checkpoints_dir / "last.pt"
    best_checkpoint_path = checkpoints_dir / "best.pt"

    epoch_summaries: list[dict[str, Any]] = []
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            precision_dtype=precision_dtype,
            scaler=scaler if scaler.is_enabled() else None,
            velocity_loss_weight=velocity_loss_weight,
            grad_clip_norm=grad_clip_norm,
            max_batches=max_train_batches,
            motion_mean=motion_mean_t,
            motion_std=motion_std_t,
        )
        val_metrics = None
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model,
                    val_loader,
                    device=device,
                    optimizer=None,
                    precision_dtype=precision_dtype,
                    scaler=None,
                    velocity_loss_weight=velocity_loss_weight,
                    grad_clip_norm=None,
                    max_batches=max_val_batches,
                    motion_mean=motion_mean_t,
                    motion_std=motion_std_t,
                )

        target_metric = float(val_metrics["loss"] if val_metrics is not None else train_metrics["loss"])
        epoch_summary = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "target_metric": target_metric,
        }
        epoch_summaries.append(epoch_summary)
        append_jsonl(history_path, epoch_summary)

        checkpoint_state = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "manifest_path": str(manifest_path),
            "model_config": asdict(motion_gru_config),
            "training_config": {
                "batch_size": batch_size,
                "precision": precision_name,
                "velocity_loss_weight": velocity_loss_weight,
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "last_epoch_metrics": epoch_summary,
            "motion_normalization": (
                {"mean": motion_mean_np.tolist(), "std": motion_std_np.tolist()}
                if motion_mean_np is not None
                else None
            ),
        }

        if target_metric <= best_metric:
            best_metric = target_metric
            best_epoch = epoch
            checkpoint_state["best_epoch"] = best_epoch
            checkpoint_state["best_metric"] = best_metric
            save_checkpoint(checkpoint_state, best_checkpoint_path)

        save_checkpoint(checkpoint_state, last_checkpoint_path)

    summary = {
        "status": "completed",
        "manifest_path": str(manifest_path),
        "experiment_dir": str(experiment_dir),
        "resolved_config_path": str(resolved_config_path),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "num_train_sequences": len(train_dataset),
        "num_val_sequences": len(val_dataset),
        "audio_feature_dim": motion_gru_config.input_size,
        "motion_feature_dim": motion_gru_config.output_size,
        "device": str(device),
        "precision": precision_name,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "epochs_completed": len(epoch_summaries),
    }
    write_json(summary_path, summary)
    return summary
