"""iTransformer forecaster (Liu et al., ICLR 2024).

Reference: Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, M. Long,
"iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting," ICLR 2024. Original repo: thuml/iTransformer.

Self-contained port. The "inversion" is exactly one architectural choice:
attention is computed over the *variate* (channel) axis, not the time axis.
Each channel's full window becomes a single token of dimension d_model,
and the encoder learns cross-variate dependencies.

This is conceptually opposite to PatchTST's channel-independent backbone.
We include both because:
  - PatchTST: cheap (no cross-channel cost), channel-independent.
  - iTransformer: cross-channel attention, captures correlations between
    routed-together OD pairs / co-bursting links / institutions.
Reviewers expect both; CLAUDE.md Rule 3 mandates we ablate the loss on
>=2 Transformer backbones to show the contribution is loss-driven, not
architecture-driven.

Surface contract matches the existing forecasters:
    forward(x: (batch, window_size, num_links)) -> (batch, num_links)

Three pieces:
  1. RevIN (reused from PatchTST module).
  2. Variate embedding: nn.Linear(window_size, d_model) applied per channel,
     producing (batch, num_channels, d_model) — channels are tokens.
  3. TransformerEncoder over the channel-token sequence + per-channel
     projection head.

Param count is O(d_model**2 * n_layers + window * d_model) — flat in
num_links because the same Linear is shared across channel tokens. But
attention compute is O(num_links**2 * d_model) per encoder layer; on
CESNET-scale (top-N institutions) keep N moderate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .patchtst import RevIN  # reuse the same RevIN implementation


class ITransformerForecaster(nn.Module):
    """Inverted Transformer: each variate is a token; attention over variates.

    Args:
        input_size: number of channels (= num_links).
        window_size: input sequence length.
        horizon: forecast horizon. Default 1 to match the runner.
        d_model, n_heads, n_layers, dim_ff, dropout: encoder hyperparameters.
        revin: enable RevIN normalisation (default True).
        revin_affine: learnable affine parameters in RevIN.
    """

    def __init__(
        self,
        input_size: int,
        window_size: int,
        horizon: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.2,
        revin: bool = True,
        revin_affine: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.input_size = int(input_size)
        self.window_size = int(window_size)
        self.horizon = int(horizon)

        self.revin: RevIN | None = (
            RevIN(self.input_size, affine=revin_affine) if revin else None
        )

        # Variate embedding: a single Linear(window, d_model) shared across
        # channels. Each channel's full time series is one token.
        self.embed = nn.Linear(self.window_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Per-channel projection head — shared Linear(d_model, horizon)
        # applied independently to each channel token.
        self.head = nn.Linear(d_model, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"expected (batch, window, channels); got {tuple(x.shape)}"
            )
        if x.shape[1] != self.window_size:
            raise ValueError(
                f"window_size mismatch: model={self.window_size}, "
                f"input={x.shape[1]}"
            )
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"input_size mismatch: model={self.input_size}, "
                f"input={x.shape[2]}"
            )

        if self.revin is not None:
            x = self.revin.normalize(x)  # (batch, time, channels)

        # Variate-as-token: transpose so channels are the sequence axis.
        # (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        # Project each channel's full window to d_model.
        tokens = self.embed_dropout(self.embed(x))  # (batch, channels, d_model)
        # Self-attention over channels.
        out = self.encoder(tokens)
        out = self.norm(out)
        # Per-channel projection.
        out = self.head(out)  # (batch, channels, horizon)

        if self.revin is not None:
            # Denormalise expects (batch, time, channels).
            out = out.transpose(1, 2)  # (batch, horizon, channels)
            out = self.revin.denormalize(out)
            out = out.transpose(1, 2)  # (batch, channels, horizon)

        if self.horizon == 1:
            return out.squeeze(-1)  # (batch, channels), parity with LSTM
        return out.transpose(1, 2)  # (batch, horizon, channels)
