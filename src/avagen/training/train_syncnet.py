"""Train the audio<->motion SyncNet and evaluate sync confidence."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from avagen.data.dataset import WindowedAudioMotionDataset
from avagen.models.sync_net import SyncNet, SyncNetConfig
from avagen.training.checkpointing import save_checkpoint
from avagen.training.logging import append_jsonl, write_json
from avagen.training.train_motion import (
    _parse_windowing_config,
    _resolve_device,
    _sequence_collate,
)
from avagen.utils.config import dump_json
from avagen.utils.seed import set_seed


def _standardize_stats(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    audio = np.concatenate([np.asarray(dataset[i].audio_features, np.float64) for i in range(len(dataset))], 0)
    motion = np.concatenate([np.asarray(dataset[i].motion_features, np.float64) for i in range(len(dataset))], 0)
    a_mean, a_std = audio.mean(0), audio.std(0)
    m_mean, m_std = motion.mean(0), motion.std(0)
    a_std = np.where(a_std < 1e-6, 1.0, a_std)
    m_std = np.where(m_std < 1e-6, 1.0, m_std)
    return (a_mean.astype(np.float32), a_std.astype(np.float32),
            m_mean.astype(np.float32), m_std.astype(np.float32))


def train_syncnet(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    manifest_path = Path(str(resolved["manifest_path"])).expanduser().resolve()
    experiment_dir = Path(str(resolved["experiment_dir"])).expanduser().resolve()
    dataset_config = dict(resolved.get("dataset", {}))
    model_config = dict(resolved.get("model", {}))
    training_config = dict(resolved.get("training", {}))
    set_seed(int(resolved.get("seed", 7)))

    audio_names = tuple(dataset_config.get("audio_feature_names", ["wav2vec"]))
    motion_name = str(dataset_config.get("motion_feature_name", "motion_vector"))
    windowing = _parse_windowing_config(dataset_config.get("windowing"))
    if windowing is None:
        raise ValueError("SyncNet training expects dataset.windowing (small window_size, e.g. 20).")

    def build(split):
        return WindowedAudioMotionDataset(manifest_path, split, windowing,
                                          audio_feature_names=audio_names, motion_feature_name=motion_name)

    train_ds = build("train")
    val_ds = build("val")
    if len(train_ds) < 2:
        raise ValueError("Need >=2 windows for contrastive training.")

    device = _resolve_device(str(training_config.get("device", "auto")))
    a_mean, a_std, m_mean, m_std = _standardize_stats(train_ds)
    am = torch.from_numpy(a_mean).to(device); as_ = torch.from_numpy(a_std).to(device)
    mm = torch.from_numpy(m_mean).to(device); ms = torch.from_numpy(m_std).to(device)

    sample = train_ds[0]
    sync_config = SyncNetConfig(
        audio_size=int(sample.audio_features.shape[1]),
        motion_size=int(sample.motion_features.shape[1]),
        hidden_size=int(model_config.get("hidden_size", 256)),
        embed_dim=int(model_config.get("embed_dim", 256)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    model = SyncNet(sync_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(training_config.get("learning_rate", 3e-4)),
                                  weight_decay=float(training_config.get("weight_decay", 1e-4)))
    batch_size = int(training_config.get("batch_size", 64))
    epochs = int(training_config.get("epochs", 100))

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                          drop_last=shuffle, collate_fn=_sequence_collate)
    train_loader = loader(train_ds, True)
    val_loader = loader(val_ds, False) if len(val_ds) >= 2 else None

    def std_batch(batch):
        a = (batch["audio_features"].to(device) - am) / as_
        m = (batch["motion_features"].to(device) - mm) / ms
        return a, m

    history_path = experiment_dir / "metrics" / "train_history.jsonl"
    dump_json({"manifest_path": str(manifest_path), "model_type": "syncnet",
               "model": asdict(sync_config), "windowing": asdict(windowing),
               "num_train_windows": len(train_ds), "num_val_windows": len(val_ds)},
              experiment_dir / "config.resolved.json")

    best = float("inf"); best_epoch = 0
    best_ckpt = experiment_dir / "checkpoints" / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train(); tr = 0.0; n = 0
        for batch in train_loader:
            a, m = std_batch(batch)
            if a.shape[0] < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = model.contrastive_loss(a, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr += float(loss.detach().cpu()); n += 1
        train_loss = tr / max(n, 1)

        val_loss = None
        pos_sim = neg_sim = None
        if val_loader is not None:
            model.eval(); vs = 0.0; vn = 0; ps = 0.0; ns = 0.0; nc = 0
            with torch.no_grad():
                for batch in val_loader:
                    a, m = std_batch(batch)
                    if a.shape[0] < 2:
                        continue
                    vs += float(model.contrastive_loss(a, m).cpu()); vn += 1
                    # positive = aligned; negative = motion rolled by 1 in batch
                    ps += float(model.sync_score(a, m).mean().cpu())
                    ns += float(model.sync_score(a, torch.roll(m, 1, 0)).mean().cpu()); nc += 1
            val_loss = vs / max(vn, 1)
            pos_sim = ps / max(nc, 1); neg_sim = ns / max(nc, 1)

        target = float(val_loss if val_loss is not None else train_loss)
        append_jsonl(history_path, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                                    "val_pos_sim": pos_sim, "val_neg_sim": neg_sim})
        state = {"epoch": epoch, "model_type": "syncnet", "model_config": asdict(sync_config),
                 "model_state_dict": model.state_dict(),
                 "audio_norm": {"mean": a_mean.tolist(), "std": a_std.tolist()},
                 "motion_norm": {"mean": m_mean.tolist(), "std": m_std.tolist()},
                 "window_size": windowing.window_size}
        if target <= best:
            best = target; best_epoch = epoch; state["best_metric"] = best
            save_checkpoint(state, best_ckpt)
        save_checkpoint(state, experiment_dir / "checkpoints" / "last.pt")

    summary = {"status": "completed", "model_type": "syncnet", "best_epoch": best_epoch,
               "best_metric": best, "best_checkpoint_path": str(best_ckpt),
               "num_train_windows": len(train_ds), "num_val_windows": len(val_ds)}
    write_json(experiment_dir / "metrics" / "summary.json", summary)
    return summary
