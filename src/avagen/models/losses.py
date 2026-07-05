"""Loss helpers for motion-model training."""

from __future__ import annotations

import torch


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {predictions.shape} and {targets.shape}."
        )
    if mask.shape != predictions.shape[:2]:
        raise ValueError(f"Mask shape must match batch/time dims, got {mask.shape} for {predictions.shape}.")

    weight = mask.to(dtype=predictions.dtype).unsqueeze(-1)
    squared_error = (predictions - targets).pow(2) * weight
    normalizer = torch.clamp(weight.sum() * predictions.shape[-1], min=1.0)
    return squared_error.sum() / normalizer


def masked_velocity_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predictions.shape[1] < 2:
        return predictions.new_zeros(())

    velocity_mask = mask[:, 1:] & mask[:, :-1]
    predicted_velocity = predictions[:, 1:] - predictions[:, :-1]
    target_velocity = targets[:, 1:] - targets[:, :-1]
    return masked_mse_loss(predicted_velocity, target_velocity, velocity_mask)
