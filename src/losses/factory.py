"""Loss factory: dispatch a string config name to an nn.Module loss.

Supported names:
    'mse'                                    -> nn.MSELoss
    'asym' | 'asymmetric_mse'                -> AsymmetricMSE(alpha, beta)
    'asym_l1' | 'cusp_linear'                -> CuspLinear(alpha, beta)
    'pinball' | 'quantile'                   -> PinballLoss(tau)

If `tau` is not given for pinball but `alpha`/`beta` are, τ is computed as
α / (α + β) so the pinball forecast targets the capacity-implied quantile.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .asymmetric import AsymmetricMSE
from .cusp_linear import CuspLinear
from .pinball import PinballLoss


def make_loss(
    name: str,
    *,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    tau: Optional[float] = None,
) -> nn.Module:
    """Construct a loss module from a string name and keyword arguments."""
    n = name.lower()
    if n == "mse":
        return nn.MSELoss()

    if n in ("asym", "asymmetric_mse"):
        if alpha is None or beta is None:
            raise ValueError(f"loss {name!r} requires alpha and beta")
        return AsymmetricMSE(alpha=alpha, beta=beta)

    if n in ("asym_l1", "cusp_linear"):
        if alpha is None or beta is None:
            raise ValueError(f"loss {name!r} requires alpha and beta")
        return CuspLinear(alpha=alpha, beta=beta)

    if n in ("pinball", "quantile"):
        if tau is None:
            if alpha is None or beta is None:
                raise ValueError(f"loss {name!r} requires tau, or both alpha and beta")
            tau = alpha / (alpha + beta)
        return PinballLoss(tau=tau)

    raise ValueError(f"unknown loss {name!r}")
