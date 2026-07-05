"""GRU baseline for aligned audio-to-motion prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class MotionGRUConfig:
    input_size: int
    output_size: int
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


class MotionGRU(nn.Module):
    def __init__(self, config: MotionGRUConfig) -> None:
        super().__init__()
        self.config = config
        gru_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=config.bidirectional,
        )
        hidden_dim = config.hidden_size * (2 if config.bidirectional else 1)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.output_size),
        )

    def forward(self, audio_features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            encoded, _ = self.encoder(audio_features)
        else:
            packed = pack_padded_sequence(
                audio_features,
                lengths.detach().to(device="cpu", dtype=torch.long),
                batch_first=True,
                enforce_sorted=False,
            )
            encoded_packed, _ = self.encoder(packed)
            encoded, _ = pad_packed_sequence(
                encoded_packed,
                batch_first=True,
                total_length=audio_features.shape[1],
            )
        return self.output_head(encoded)
