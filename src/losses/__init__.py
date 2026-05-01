"""Loss functions for traffic-forecast training.

Public API:
    asymmetric_mse(y_pred, y_true, alpha, beta) -> tensor
    pinball(y_pred, y_true, tau)                -> tensor
    make_loss(name, **kwargs)                   -> nn.Module wrapping a callable

Conventions:
- y_pred, y_true: torch.Tensor of identical shape, no broadcasting tricks.
- All losses return a *scalar* mean over all elements (matching nn.MSELoss).
"""

from .asymmetric import asymmetric_mse, AsymmetricMSE
from .cusp_linear import cusp_linear, CuspLinear
from .pinball import pinball, PinballLoss
from .factory import make_loss

__all__ = [
    "asymmetric_mse",
    "AsymmetricMSE",
    "cusp_linear",
    "CuspLinear",
    "pinball",
    "PinballLoss",
    "make_loss",
]
