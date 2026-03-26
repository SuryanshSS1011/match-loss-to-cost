"""DCRNN training entry point.

Wraps `src.train_neural.run` with a builder that consumes the routing
matrix `R` from the dataset npz (passed in by the trainer) and converts
it to a link-link adjacency. Falls back to identity adjacency when R is
missing or `R = I` (e.g. CESNET) — which makes DCRNN reduce to a
per-link GRU.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch.nn as nn

from .config import CONFIG
from .models.dcrnn import DCRNNForecaster, adjacency_from_routing
from .train_neural import run


def _build_model(num_links: int, R: Optional[np.ndarray] = None,
                 **_) -> nn.Module:
    if R is None:
        adjacency = None  # forecaster falls back to identity
    else:
        adjacency = adjacency_from_routing(np.asarray(R))
    return DCRNNForecaster(
        input_size=num_links,
        window_size=CONFIG["window_size"],
        adjacency=adjacency,
        horizon=1,
        hidden_dim=CONFIG.get("dcrnn_hidden_dim", 64),
        num_layers=CONFIG.get("dcrnn_num_layers", 2),
        K=CONFIG.get("dcrnn_K", 2),
    )


def main():
    return run("dcrnn", _build_model)


if __name__ == "__main__":
    main()
