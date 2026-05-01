"""Cusp-linear (L1 piecewise) asymmetric loss.

L(y, ŷ) = α · max(y − ŷ, 0) + β · max(ŷ − y, 0)

The L1 analogue of `asymmetric.py`'s squared form. Compared to the squared
asymmetric loss this is much less sensitive to the α/β imbalance ratio
because it doesn't square many-moderate over-predictions into a dominating
sum. Equivalent up to a constant scale to the pinball loss at τ = α/(α+β),
included as a separate name so the runner can dispatch on the form
explicitly instead of going through pinball's quantile-loss interpretation.

Reference: Eramo et al. IEEE Access 2020 ("cusp-linear cost"); Eramo et al.
Comput. Netw. 2021. The original formulation in those papers is L1, not
squared.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def cusp_linear(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float = 5.0,
    beta: float = 1.0,
) -> torch.Tensor:
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"shape mismatch: y_pred {tuple(y_pred.shape)} vs y_true {tuple(y_true.shape)}"
        )
    if alpha < 0 or beta < 0:
        raise ValueError(f"alpha, beta must be >= 0; got alpha={alpha}, beta={beta}")
    err = y_true - y_pred
    under = torch.clamp(err, min=0.0)
    over = torch.clamp(-err, min=0.0)
    return alpha * under.mean() + beta * over.mean()


class CuspLinear(nn.Module):
    def __init__(self, alpha: float = 5.0, beta: float = 1.0):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha, beta must be >= 0; got alpha={alpha}, beta={beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return cusp_linear(y_pred, y_true, self.alpha, self.beta)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}"
