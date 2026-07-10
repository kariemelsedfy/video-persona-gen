"""Training loop for the conditional flow-matching motion generator."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from avagen.data.dataset import WindowedAudioMotionDataset
from avagen.models.losses import masked_mse_loss
from avagen.models.motion_flow import MotionFlowConfig, MotionFlowModel, sample_motion
from avagen.training.checkpointing import save_checkpoint
from avagen.training.logging import append_jsonl, write_json
from avagen.training.train_motion import (
    _compute_motion_normalization,
    _move_batch_to_device,
    _parse_windowing_config,
    _resolve_device,
    _sequence_collate,
)
from avagen.utils.config import dump_json
from avagen.utils.seed import set_seed


def _flow_batch_loss(model, batch, mean, std, p_uncond: float = 0.0):
    x1 = (batch["motion_features"] - mean) / std        # standardized data motion
    x0 = torch.randn_like(x1)                            # gaussian prior
    t = torch.rand(x1.shape[0], device=x1.device)        # flow time in [0,1]
    t_b = t.view(-1, 1, 1)
    xt = (1.0 - t_b) * x0 + t_b * x1                     # rectified-flow interpolation
    v_target = x1 - x0                                   # constant target velocity
    # classifier-free guidance: randomly drop audio so the model also learns the
    # unconditional velocity field.
    audio_drop = None
    if p_uncond > 0.0:
        audio_drop = torch.rand(x1.shape[0], device=x1.device) < p_uncond
    v_pred = model(xt, batch["audio_features"], t, audio_drop=audio_drop)
    return masked_mse_loss(v_pred, v_target, batch["mask"])


def train_flow_model(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    manifest_path = Path(str(resolved["manifest_path"])).expanduser().resolve()
    experiment_dir = Path(str(resolved["experiment_dir"])).expanduser().resolve()
    dataset_config = dict(resolved.get("dataset", {}))
    model_config = dict(resolved.get("model", {}))
    training_config = dict(resolved.get("training", {}))
    seed = int(resolved.get("seed", 7))
    set_seed(seed)

    audio_feature_names = tuple(dataset_config.get("audio_feature_names", ["wav2vec"]))
    motion_feature_name = str(dataset_config.get("motion_feature_name", "motion_vector"))
    windowing = _parse_windowing_config(dataset_config.get("windowing"))
    if windowing is None:
        raise ValueError("Flow training expects dataset.windowing to be enabled.")
    limit = dataset_config.get("limit")
    limit = int(limit) if limit is not None else None

    def build(split: str) -> WindowedAudioMotionDataset:
        return WindowedAudioMotionDataset(
            manifest_path, split, windowing,
            audio_feature_names=audio_feature_names,
            motion_feature_name=motion_feature_name, limit=limit,
        )

    train_dataset = build("train")
    val_dataset = build("val")
    if len(train_dataset) == 0:
        raise ValueError(f"No training windows in {manifest_path}.")

    device = _resolve_device(str(training_config.get("device", "auto")))
    sample = train_dataset[0]
    flow_config = MotionFlowConfig(
        motion_size=int(sample.motion_features.shape[1]),
        audio_size=int(sample.audio_features.shape[1]),
        hidden_size=int(model_config.get("hidden_size", 512)),
        num_layers=int(model_config.get("num_layers", 3)),
        time_embed_dim=int(model_config.get("time_embed_dim", 128)),
        dropout=float(model_config.get("dropout", 0.1)),
        bidirectional=bool(model_config.get("bidirectional", True)),
    )
    model = MotionFlowModel(flow_config).to(device)

    motion_mean_np, motion_std_np = _compute_motion_normalization(train_dataset)
    mean = torch.from_numpy(motion_mean_np).to(device)
    std = torch.from_numpy(motion_std_np).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 2e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    batch_size = int(training_config.get("batch_size", 16))
    epochs = int(training_config.get("epochs", 200))
    grad_clip = training_config.get("grad_clip_norm", 1.0)
    grad_clip = None if grad_clip is None else float(grad_clip)
    sample_steps = int(training_config.get("flow_sample_steps", 20))
    p_uncond = float(training_config.get("cfg_p_uncond", 0.0))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=_sequence_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=_sequence_collate) if len(val_dataset) else None

    checkpoints_dir = experiment_dir / "checkpoints"
    metrics_dir = experiment_dir / "metrics"
    history_path = metrics_dir / "train_history.jsonl"
    dump_json(
        {"manifest_path": str(manifest_path), "seed": seed, "model_type": "flow",
         "dataset": {"audio_feature_names": list(audio_feature_names), "windowing": asdict(windowing),
                     "num_train_windows": len(train_dataset), "num_val_windows": len(val_dataset)},
         "model": asdict(flow_config),
         "training": {"batch_size": batch_size, "epochs": epochs,
                      "learning_rate": float(training_config.get("learning_rate", 2e-4))}},
        experiment_dir / "config.resolved.json",
    )

    best_metric = float("inf")
    best_epoch = 0
    last_ckpt = checkpoints_dir / "last.pt"
    best_ckpt = checkpoints_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        tr_sum = 0.0; tr_n = 0
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            frames = int(batch["mask"].sum().item())
            if frames == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = _flow_batch_loss(model, batch, mean, std, p_uncond=p_uncond)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tr_sum += float(loss.detach().cpu()) * frames; tr_n += frames
        train_loss = tr_sum / max(tr_n, 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            v_sum = 0.0; v_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = _move_batch_to_device(batch, device)
                    frames = int(batch["mask"].sum().item())
                    if frames == 0:
                        continue
                    # average flow loss over a few random t for a stable val estimate
                    losses = [float(_flow_batch_loss(model, batch, mean, std).cpu()) for _ in range(4)]
                    v_sum += (sum(losses) / len(losses)) * frames; v_n += frames
            val_loss = v_sum / max(v_n, 1)

        target = float(val_loss if val_loss is not None else train_loss)
        append_jsonl(history_path, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        state = {
            "epoch": epoch, "best_epoch": best_epoch, "best_metric": best_metric,
            "model_type": "flow", "model_config": asdict(flow_config),
            "flow_sample_steps": sample_steps, "cfg_p_uncond": p_uncond,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "motion_normalization": {"mean": motion_mean_np.tolist(), "std": motion_std_np.tolist()},
        }
        if target <= best_metric:
            best_metric = target; best_epoch = epoch
            state["best_epoch"] = best_epoch; state["best_metric"] = best_metric
            save_checkpoint(state, best_ckpt)
        save_checkpoint(state, last_ckpt)

    summary = {"status": "completed", "model_type": "flow", "best_epoch": best_epoch,
               "best_metric": best_metric, "best_checkpoint_path": str(best_ckpt),
               "num_train_windows": len(train_dataset), "num_val_windows": len(val_dataset),
               "motion_feature_dim": flow_config.motion_size, "audio_feature_dim": flow_config.audio_size}
    write_json(experiment_dir / "metrics" / "summary.json", summary)
    return summary
