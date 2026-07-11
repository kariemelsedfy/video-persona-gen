"""Conditional flow-matching model for audio-driven motion generation.

Frontier approach (FLOAT / Ditto / FlowTalk, 2024-2025): instead of regressing
motion with MSE (which averages -> damped, frozen motion), learn a generative
*flow* from Gaussian noise to the motion distribution, conditioned on audio.
At inference we sample from noise by integrating an ODE, which yields diverse,
full-amplitude, natural motion (head sway + mouth), not the conditional mean.

We use rectified flow (straight-line paths): given data motion x1 and noise x0,
x_t = (1-t) x0 + t x1 and the target velocity field is constant, v* = x1 - x0.
A bidirectional-GRU vector field predicts v(x_t, t, audio); it is O(T) in the
sequence length so whole clips can be sampled in one pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MotionFlowConfig:
    motion_size: int          # dimensionality of the motion vector (e.g. 205)
    audio_size: int           # dimensionality of the per-frame audio feature (e.g. 768)
    hidden_size: int = 512
    num_layers: int = 3
    time_embed_dim: int = 128
    dropout: float = 0.1
    bidirectional: bool = True


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of a scalar flow-time t in [0,1] -> (B, dim)."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:  # odd dim padding
        emb = torch.cat([emb, torch.zeros(emb.shape[0], dim - emb.shape[-1], device=device)], dim=-1)
    return emb


class MotionFlowModel(nn.Module):
    """Predicts the rectified-flow velocity field v(x_t, t, audio)."""

    def __init__(self, config: MotionFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.motion_proj = nn.Linear(config.motion_size, config.hidden_size)
        self.audio_proj = nn.Linear(config.audio_size, config.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        gru_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=config.bidirectional,
        )
        out_dim = config.hidden_size * (2 if config.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(out_dim, config.motion_size),
        )
        # Learned unconditional embedding for classifier-free guidance: used in
        # place of the audio projection when audio is dropped.
        self.null_cond = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(
        self,
        noisy_motion: torch.Tensor,   # (B, T, motion_size)
        audio_features: torch.Tensor, # (B, T, audio_size)
        flow_time: torch.Tensor,      # (B,) in [0,1]
        audio_drop: torch.Tensor | None = None,  # (B,) bool: True -> unconditional
    ) -> torch.Tensor:
        time_emb = sinusoidal_time_embedding(flow_time, self.config.time_embed_dim)
        time_h = self.time_mlp(time_emb).unsqueeze(1)  # (B,1,H)
        audio_h = self.audio_proj(audio_features)      # (B,T,H)
        if audio_drop is not None:
            null = self.null_cond.view(1, 1, -1)
            audio_h = torch.where(audio_drop.view(-1, 1, 1), null, audio_h)
        h = self.motion_proj(noisy_motion) + audio_h + time_h
        encoded, _ = self.gru(h)
        return self.head(encoded)


@torch.no_grad()
def sample_motion(
    model: MotionFlowModel,
    audio_features: torch.Tensor,  # (B, T, audio_size)
    steps: int = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Integrate dx/dt = v(x,t,audio) from t=0 (noise) to t=1 (data) with Euler steps."""
    device = audio_features.device
    b, seq_len, _ = audio_features.shape
    x = torch.randn(b, seq_len, model.config.motion_size, device=device, generator=generator)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=device)
        v = model(x, audio_features, t)
        x = x + v * dt
    return x


@torch.no_grad()
def sample_motion_cfg(
    model: MotionFlowModel,
    audio_features: torch.Tensor,
    steps: int = 20,
    guidance_weight: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Classifier-free guided sampling.

    v = v_uncond + w * (v_cond - v_uncond). w=1 recovers plain conditional
    sampling; w>1 pushes motion to adhere harder to the audio (tighter lip-sync).
    """
    device = audio_features.device
    b, seq_len, _ = audio_features.shape
    x = torch.randn(b, seq_len, model.config.motion_size, device=device, generator=generator)
    drop_true = torch.ones(b, dtype=torch.bool, device=device)
    drop_false = torch.zeros(b, dtype=torch.bool, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=device)
        v_cond = model(x, audio_features, t, audio_drop=drop_false)
        if guidance_weight == 1.0:
            v = v_cond
        else:
            v_uncond = model(x, audio_features, t, audio_drop=drop_true)
            v = v_uncond + guidance_weight * (v_cond - v_uncond)
        x = x + v * dt
    return x
