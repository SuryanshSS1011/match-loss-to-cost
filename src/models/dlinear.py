"""DLinear forecaster (Zeng, Chen, Zhang, Xu, AAAI 2023).

Reference: A. Zeng et al., "Are Transformers Effective for Time Series
Forecasting?", AAAI 2023. Original repo: cure-lab/LTSF-Linear.

DLinear is the embarrassingly-simple "unbeatable" baseline that put the
2021–22 long-horizon Transformer wave in question. The architecture:

  1. Decompose the input into trend (moving average) + seasonal residual.
  2. Apply an independent per-channel Linear(window → horizon) to each.
  3. Sum the two projections.

That's the whole model. ~30 lines, no attention, no normalization. We
include it as a strong forecasting baseline (per the minimum-viable-8 list
in CLAUDE.md). Same input/output shape as `LSTMForecaster`:
    input:  (batch, window_size, num_links)
    output: (batch, num_links)   for one-step ahead, or
            (batch, horizon, num_links)  for multi-step (horizon > 1).

This module defines only the model class. Training is wired through the
same `train_lstm.py` -style loop in week 3-4; the runner refactor already
threads the loss config (`lstm_loss`, `loss_alpha`, etc.) through CONFIG.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MovingAverage(nn.Module):
    """Moving-average smoother along the time axis with edge replication.

    DLinear convention: an odd kernel (default 25) with replicate-padding
    so the smoothed series is the same length as the input. We use kernel=25
    by default to match the LTSF-Linear repo; pass `kernel_size=k` to vary.
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd and >= 1; got {kernel_size}")
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels). Pool along time.
        # Move time to the last dim, replicate-pad, pool, restore shape.
        b, t, c = x.shape
        x_t = x.transpose(1, 2)  # (b, c, t)
        x_pad = nn.functional.pad(
            x_t, (self.padding, self.padding), mode="replicate"
        )
        out = self.avg(x_pad)  # (b, c, t)
        return out.transpose(1, 2)  # (b, t, c)


class _SeriesDecomposition(nn.Module):
    """Decompose into (seasonal_residual, trend) via a moving-average trend."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.ma = _MovingAverage(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.ma(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearForecaster(nn.Module):
    """DLinear: per-channel linear projection of trend + seasonal components.

    Args:
        input_size: number of channels (= num_links).
        window_size: input length.
        horizon: number of forecast steps. Default 1 to match the existing
            LSTM / SARIMA convention (one-step-ahead).
        kernel_size: moving-average kernel for the trend component. Must be odd.
        individual: if True, use a separate Linear per channel (LTSF-Linear
            default). If False, share a single Linear across channels (smaller,
            faster). Default True for parity with the published numbers.
    """

    def __init__(self, input_size: int, window_size: int, horizon: int = 1,
                 kernel_size: int = 25, individual: bool = True):
        super().__init__()
        self.input_size = int(input_size)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.individual = bool(individual)

        self.decomp = _SeriesDecomposition(kernel_size=kernel_size)

        if self.individual:
            # Per-channel Linears, packed as ModuleLists.
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(window_size, horizon) for _ in range(self.input_size)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(window_size, horizon) for _ in range(self.input_size)]
            )
        else:
            self.linear_seasonal = nn.Linear(window_size, horizon)
            self.linear_trend = nn.Linear(window_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, num_links)
        if x.dim() != 3:
            raise ValueError(f"expected (batch, window, channels); got {tuple(x.shape)}")
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

        seasonal, trend = self.decomp(x)
        # → (batch, time, channels). Transpose so we project along the time
        # axis: (batch, channels, time) → (batch, channels, horizon).
        seasonal_t = seasonal.transpose(1, 2)
        trend_t = trend.transpose(1, 2)

        if self.individual:
            outs = []
            for c in range(self.input_size):
                s_c = self.linear_seasonal[c](seasonal_t[:, c, :])
                t_c = self.linear_trend[c](trend_t[:, c, :])
                outs.append(s_c + t_c)  # (batch, horizon)
            out = torch.stack(outs, dim=-1)  # (batch, horizon, channels)
        else:
            s = self.linear_seasonal(seasonal_t)  # (batch, channels, horizon)
            t = self.linear_trend(trend_t)
            out = (s + t).transpose(1, 2)  # (batch, horizon, channels)

        if self.horizon == 1:
            return out.squeeze(1)  # (batch, channels), parity with LSTM.
        return out  # (batch, horizon, channels)
