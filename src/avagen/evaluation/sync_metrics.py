"""Sync-confidence metric using a trained SyncNet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from avagen.models.sync_net import SyncNet, SyncNetConfig
from avagen.training.checkpointing import load_checkpoint


def load_syncnet(checkpoint_path: str | Path, device: str = "cpu") -> tuple[SyncNet, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model = SyncNet(SyncNetConfig(**ckpt["model_config"])).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def compute_sync_confidence(
    model: SyncNet,
    audio: np.ndarray,      # (T, A) aligned to motion frames
    motion: np.ndarray,     # (T, M)
    ckpt: dict[str, Any],
    stride: int | None = None,
    offset_range: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    """Mean aligned cosine sync score over windows, plus an offset sweep.

    Returns sync_confidence (aligned, offset 0) and best_offset (the lag that
    maximizes sync -- should be ~0 for well-synced motion).
    """
    W = int(ckpt["window_size"])
    stride = stride or max(W // 2, 1)
    am = np.asarray(ckpt["audio_norm"]["mean"], np.float32); as_ = np.asarray(ckpt["audio_norm"]["std"], np.float32)
    mm = np.asarray(ckpt["motion_norm"]["mean"], np.float32); ms = np.asarray(ckpt["motion_norm"]["std"], np.float32)
    a = (np.asarray(audio, np.float32) - am) / as_
    m = (np.asarray(motion, np.float32) - mm) / ms
    T = min(a.shape[0], m.shape[0])

    def windows(arr, shift):
        out = []
        for s in range(0, T - W + 1, stride):
            js = s + shift
            if js < 0 or js + W > T:
                continue
            out.append(arr[js:js + W])
        return out

    def mean_sim(shift_motion):
        aw = windows(a, 0)
        mw = windows(m, shift_motion)
        n = min(len(aw), len(mw))
        if n == 0:
            return float("nan")
        at = torch.from_numpy(np.stack(aw[:n])).to(device)
        mt = torch.from_numpy(np.stack(mw[:n])).to(device)
        with torch.no_grad():
            return float(model.sync_score(at, mt).mean().cpu())

    sweep = {k: mean_sim(k) for k in range(-offset_range, offset_range + 1)}
    valid = {k: v for k, v in sweep.items() if v == v}
    best_offset = max(valid, key=valid.get) if valid else 0
    return {
        "sync_confidence": sweep.get(0, float("nan")),
        "best_offset": best_offset,
        "best_offset_sim": valid.get(best_offset, float("nan")),
        "offset_sweep": sweep,
    }
