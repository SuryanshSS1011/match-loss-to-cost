"""Asymmetric squared-error loss for capacity-aware traffic forecasting.

L(y, ŷ) = α · max(y − ŷ, 0)² + β · max(ŷ − y, 0)²

Interpretation
--------------
- (y − ŷ) > 0 means we *under*-predicted → operator under-provisions → SLA risk.
  This is the *expensive* failure mode in capacity planning, so we attach
  weight α >= β.
- (ŷ − y) > 0 means we *over*-predicted → operator over-provisions → wasted
  capacity. Less harmful but not free, so β > 0.

α / β is calibrated to a target SLA-violation budget. With α = β this collapses
to plain MSE.

Reference: Eramo et al., IEEE Access 2020 ("cusp-linear cost"); Eramo et al.,
Comput. Netw. 2021. We use the squared-quadratic form rather than cusp-linear
because it is differentiable everywhere and matches the Çiftçioğlu 2023 setup
under quadratic costs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def asymmetric_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float = 5.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Asymmetric squared-error loss; returns a scalar mean.

    Args:
        y_pred: predicted values.
        y_true: ground-truth values, same shape as y_pred.
        alpha:  per-unit squared cost of under-prediction (y > ŷ). Default 5.0.
        beta:   per-unit squared cost of over-prediction  (ŷ > y). Default 1.0.

    Returns:
        Scalar loss tensor.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"shape mismatch: y_pred {tuple(y_pred.shape)} vs y_true {tuple(y_true.shape)}"
        )
    if alpha < 0 or beta < 0:
        raise ValueError(f"alpha, beta must be >= 0; got alpha={alpha}, beta={beta}")
    err = y_true - y_pred
    under = torch.clamp(err, min=0.0)   # positive when y > ŷ (under-prediction)
    over = torch.clamp(-err, min=0.0)   # positive when ŷ > y (over-prediction)
    return alpha * under.pow(2).mean() + beta * over.pow(2).mean()


class AsymmetricMSE(nn.Module):
    """nn.Module wrapper around `asymmetric_mse`.

    Lets the loss be plugged into any training loop that expects an nn.Module,
    e.g. `criterion = AsymmetricMSE(alpha=5.0, beta=1.0)`.
    """

    def __init__(self, alpha: float = 5.0, beta: float = 1.0):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha, beta must be >= 0; got alpha={alpha}, beta={beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return asymmetric_mse(y_pred, y_true, self.alpha, self.beta)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}"
