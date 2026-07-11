"""SyncNet-style audio<->motion synchronization discriminator.

Inspired by Wav2Lip's SyncNet: learn a joint embedding where a short window of
audio and the *temporally aligned* window of face motion are close, and
misaligned windows are far. We train it CLIP-style with in-batch negatives
(aligned window = positive; every other window in the batch = negative), which
gives many negatives and a stable objective.

Two uses:
  1. Sync-confidence metric: cosine similarity of aligned windows over a clip
     (higher = better lip-sync). Also supports an offset sweep to check that
     alignment peaks at zero lag.
  2. Test-time guidance: gradient of the sync score can nudge generated motion
     toward better synchrony (see inference.sync_guidance).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SyncNetConfig:
    audio_size: int
    motion_size: int
    hidden_size: int = 256
    embed_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


class _WindowEncoder(nn.Module):
    """Bi-GRU over a fixed window -> mean-pooled, L2-normalized embedding."""

    def __init__(self, input_size: int, hidden: int, embed: int, layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if layers > 1 else 0.0
        self.gru = nn.GRU(input_size, hidden, num_layers=layers, batch_first=True,
                          dropout=gru_dropout, bidirectional=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, embed), nn.GELU(),
            nn.Linear(embed, embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, W, input)
        h, _ = self.gru(x)
        pooled = h.mean(dim=1)
        return F.normalize(self.proj(pooled), dim=-1)


class SyncNet(nn.Module):
    def __init__(self, config: SyncNetConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder = _WindowEncoder(
            config.audio_size, config.hidden_size, config.embed_dim, config.num_layers, config.dropout
        )
        self.motion_encoder = _WindowEncoder(
            config.motion_size, config.hidden_size, config.embed_dim, config.num_layers, config.dropout
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.3026))  # ln(10), CLIP-style temperature

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio_encoder(audio)

    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return self.motion_encoder(motion)

    def sync_score(self, audio: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of aligned audio/motion windows -> (B,) in [-1, 1]."""
        a = self.encode_audio(audio)
        m = self.encode_motion(motion)
        return (a * m).sum(dim=-1)

    def contrastive_loss(self, audio: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        """Symmetric CLIP-style in-batch contrastive loss."""
        a = self.encode_audio(audio)
        m = self.encode_motion(motion)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * a @ m.t()  # (B, B)
        target = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
