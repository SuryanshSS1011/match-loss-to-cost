"""iTransformer training entry point.

Mirrors `train_patchtst.py`: builds the model, hands it to
`src.train_neural.run`. Reads `itransformer_*` CONFIG keys.
"""

from __future__ import annotations

import torch.nn as nn

from .config import CONFIG
from .models.itransformer import ITransformerForecaster
from .train_neural import run


def _build_model(num_links: int, **_) -> nn.Module:
    return ITransformerForecaster(
        input_size=num_links,
        window_size=CONFIG["window_size"],
        horizon=1,
        d_model=CONFIG.get("itransformer_d_model", 128),
        n_heads=CONFIG.get("itransformer_n_heads", 8),
        n_layers=CONFIG.get("itransformer_n_layers", 3),
        dim_ff=CONFIG.get("itransformer_dim_ff", 256),
        dropout=CONFIG.get("itransformer_dropout", 0.2),
        revin=CONFIG.get("itransformer_revin", True),
        revin_affine=CONFIG.get("itransformer_revin_affine", True),
    )


def main():
    return run("itransformer", _build_model)


if __name__ == "__main__":
    main()
