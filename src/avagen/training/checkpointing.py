"""Checkpoint helpers for motion-model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint payload in {checkpoint_path}")

    model_state = payload.get("model_state_dict")
    if model is not None and isinstance(model_state, dict):
        model.load_state_dict(model_state)

    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None and isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)

    return payload
