"""PatchTST forecaster (Nie et al., ICLR 2023).

Reference: Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam,
"A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers,"
ICLR 2023. Original repo: yuqinie98/PatchTST.

Self-contained port (no pip dep on the upstream repo). Surface matches the
existing LSTMForecaster / DLinearForecaster:
    forward(x: (batch, window_size, num_links)) -> (batch, num_links)

Three pieces, each ~30 lines:
  1. RevIN — per-channel reversible instance normalisation. The key trick
     for non-stationary series (Kim et al. ICLR 2022). We compute per-
     channel mean and std at input, normalise, run through the model, then
     de-normalise at output.
  2. PatchEmbedding — split each channel's window into overlapping
     patches of length `patch_len` and stride `stride`, each patch
     projected to `d_model` via a Linear layer.
  3. ChannelIndependent backbone — `nn.TransformerEncoder` applied to each
     channel separately (i.e. the channel axis is folded into the batch
     axis before the encoder, then unfolded). This is the "CI" variant
     from the paper, which beats the channel-mixing variant on every
     benchmark in their Table 4.

For the project's one-step-ahead use case (horizon=1) the flatten head is
a single Linear from (num_patches * d_model) to horizon=1.

Defaults are tuned for window_size=72 (our project default):
    patch_len=12, stride=6 → 11 patches per window.
On the cloud box where window_size=288 (full daily seasonality) the
defaults still work but you may want to bump `patch_len` to 24.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RevIN
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al. ICLR 2022).

    Per-channel mean/std computed at input. Has learnable affine parameters
    (gamma, beta) so the network can adjust the normalisation. Denormalises
    on the way back out.

    Implementation note: we keep the per-batch mean/std as a buffer on the
    instance, which means a single `RevIN` cannot be safely reused across
    *parallel* forward passes with different inputs. PatchTST's flow is
    `normalize → backbone → denormalize` synchronously inside one forward,
    so this is fine.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5,
                 affine: bool = True):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        # Stats stash; populated by `normalize()`, read by `denormalize()`.
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = x.std(dim=1, keepdim=True, unbiased=False).detach()
        out = (x - self._mean) / (self._std + self.eps)
        if self.affine:
            out = out * self.gamma + self.beta
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            raise RuntimeError("RevIN.denormalize called before normalize")
        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)
        return x * (self._std + self.eps) + self._mean


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class _PatchEmbedding(nn.Module):
    """Slice (batch*channels, window) into overlapping patches and project.

    Output: (batch*channels, num_patches, d_model).
    """

    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        if patch_len < 1 or stride < 1:
            raise ValueError(f"patch_len, stride must be >= 1; "
                             f"got {patch_len}, {stride}")
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch*channels, window)
        # unfold along the time axis to get (batch*channels, num_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.proj(patches)

    def num_patches(self, window_size: int) -> int:
        if window_size < self.patch_len:
            raise ValueError(
                f"window_size={window_size} < patch_len={self.patch_len}"
            )
        return (window_size - self.patch_len) // self.stride + 1


# ---------------------------------------------------------------------------
# Full forecaster
# ---------------------------------------------------------------------------

class PatchTSTForecaster(nn.Module):
    """Channel-independent PatchTST.

    Args:
        input_size: number of channels (= num_links).
        window_size: length of the input window.
        horizon: forecast horizon. We default to 1 (one-step-ahead) to match
            the rest of the runner; the model returns (batch, num_links).
        patch_len, stride: patching hyperparameters.
        d_model, n_heads, n_layers, dim_ff, dropout: Transformer encoder
            hyperparameters.
        revin: enable RevIN normalisation (default True; recommended).
        revin_affine: enable learnable affine parameters in RevIN.
    """

    def __init__(
        self,
        input_size: int,
        window_size: int,
        horizon: int = 1,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.2,
        revin: bool = True,
        revin_affine: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.revin: RevIN | None = (
            RevIN(self.input_size, affine=revin_affine) if revin else None
        )
        self.patch = _PatchEmbedding(patch_len=patch_len, stride=stride,
                                     d_model=d_model)
        self.num_patches = self.patch.num_patches(self.window_size)

        # Learnable additive positional encoding over the patch axis.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(self.num_patches * d_model, self.horizon)
        self.dropout = nn.Dropout(dropout)

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
            x = self.revin.normalize(x)

        # Channel-independent: fold channels into the batch axis.
        b, t, c = x.shape
        # (batch, time, channels) → (batch, channels, time) → flatten to
        # (batch*channels, time).
        x_ci = x.permute(0, 2, 1).reshape(b * c, t)
        # Patch + project: (batch*c, num_patches, d_model)
        emb = self.patch(x_ci) + self.pos_embed
        emb = self.dropout(emb)
        # Transformer encoder.
        out = self.encoder(emb)  # (batch*c, num_patches, d_model)
        # Flatten head per channel.
        flat = out.reshape(b * c, self.num_patches * out.shape[-1])
        head_out = self.head(flat)  # (batch*c, horizon)
        # Restore the channel axis.
        head_out = head_out.reshape(b, c, self.horizon)

        if self.revin is not None:
            # Denormalise expects (batch, time, channels), so transpose.
            head_out = head_out.permute(0, 2, 1)  # (b, horizon, c)
            head_out = self.revin.denormalize(head_out)
            head_out = head_out.permute(0, 2, 1)  # back to (b, c, horizon)

        if self.horizon == 1:
            return head_out.squeeze(-1)  # (batch, channels), parity with LSTM
        # Otherwise return (batch, horizon, channels).
        return head_out.permute(0, 2, 1)
