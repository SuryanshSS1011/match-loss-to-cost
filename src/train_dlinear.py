"""DLinear training entry point.

Same shape as `train_lstm.py` — supplies a model builder to
`src.train_neural.run` and re-exports a `main()` that the runner can call
via `importlib.reload(train_dlinear); train_dlinear.main()`.

Reads the same loss + window CONFIG keys as the LSTM trainer, so the
runner's loss override path is unchanged.
"""

from __future__ import annotations

import torch.nn as nn

from .config import CONFIG
from .models.dlinear import DLinearForecaster
from .train_neural import run


def _build_model(num_links: int, **_) -> nn.Module:
    return DLinearForecaster(
        input_size=num_links,
        window_size=CONFIG["window_size"],
        horizon=1,
        kernel_size=CONFIG.get("dlinear_kernel_size", 25),
        individual=CONFIG.get("dlinear_individual", True),
    )


def main():
    return run("dlinear", _build_model)


if __name__ == "__main__":
    main()
