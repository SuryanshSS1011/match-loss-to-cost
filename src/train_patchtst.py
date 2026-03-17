"""PatchTST training entry point.

Same shape as `train_lstm.py` / `train_dlinear.py` — supplies a model
builder to `src.train_neural.run`. Reads `patchtst_*` CONFIG keys with
sensible defaults so existing CONFIG dicts work without edits.
"""

from __future__ import annotations

import torch.nn as nn

from .config import CONFIG
from .models.patchtst import PatchTSTForecaster
from .train_neural import run


def _build_model(num_links: int, **_) -> nn.Module:
    return PatchTSTForecaster(
        input_size=num_links,
        window_size=CONFIG["window_size"],
        horizon=1,
        patch_len=CONFIG.get("patchtst_patch_len", 12),
        stride=CONFIG.get("patchtst_stride", 6),
        d_model=CONFIG.get("patchtst_d_model", 128),
        n_heads=CONFIG.get("patchtst_n_heads", 8),
        n_layers=CONFIG.get("patchtst_n_layers", 3),
        dim_ff=CONFIG.get("patchtst_dim_ff", 256),
        dropout=CONFIG.get("patchtst_dropout", 0.2),
        revin=CONFIG.get("patchtst_revin", True),
        revin_affine=CONFIG.get("patchtst_revin_affine", True),
    )


def main():
    return run("patchtst", _build_model)


if __name__ == "__main__":
    main()
