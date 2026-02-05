"""Pinball / quantile loss for τ-quantile regression.

L_τ(y, ŷ) = max(τ · (y − ŷ),  (τ − 1) · (y − ŷ))
         = τ · max(y − ŷ, 0) + (1 − τ) · max(ŷ − y, 0)

For asymmetric capacity-aware forecasting, choose τ = α / (α + β) where α and β
are the asymmetric-MSE penalties. Minimising the pinball loss at this τ recovers
the conditional τ-quantile of y, which is the *capacity* an operator should
provision to bound the under-provisioning probability at 1 − τ (Koenker-Bassett
1978, DeepAR Salinas 2020).

Reference: Koenker & Bassett, Econometrica 1978 ("Regression Quantiles").
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pinball(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """Pinball / quantile loss at level τ; returns a scalar mean.

    Args:
        y_pred: predicted τ-quantile of y.
        y_true: ground-truth values, same shape as y_pred.
        tau:    quantile level in (0, 1). 0.5 = median (= MAE / 2).

    Returns:
        Scalar loss tensor.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"shape mismatch: y_pred {tuple(y_pred.shape)} vs y_true {tuple(y_true.shape)}"
        )
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1); got {tau}")
    err = y_true - y_pred
    return torch.maximum(tau * err, (tau - 1.0) * err).mean()


class PinballLoss(nn.Module):
    """nn.Module wrapper around `pinball`.

    `criterion = PinballLoss(tau=0.83)` for an α=5, β=1 capacity setup
    (τ = 5 / (5 + 1) ≈ 0.83 → 83rd-percentile forecast).
    """

    def __init__(self, tau: float = 0.5):
        super().__init__()
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1); got {tau}")
        self.tau = float(tau)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return pinball(y_pred, y_true, self.tau)

    def extra_repr(self) -> str:
        return f"tau={self.tau}"
