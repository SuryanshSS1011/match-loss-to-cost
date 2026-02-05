"""LSTM training entry point.

Thin wrapper around `src.train_neural.run`: supplies the model factory
that builds the LSTM, plus a `main()` so the runner can call
`importlib.reload(train_lstm); train_lstm.main()` exactly as before.

The model class itself lives here so existing imports
(`from src.train_lstm import LSTMForecaster`) keep working. Anything new
should import from `src.models.*`.
"""

from __future__ import annotations

import torch.nn as nn

from .config import CONFIG
from .train_neural import run


class LSTMForecaster(nn.Module):
    """Multi-variate LSTM forecaster.

    Maps (batch, window_size, num_links) → (batch, num_links) via an LSTM
    stack and a dense readout on the last time step.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _build_model(num_links: int, **_) -> nn.Module:
    return LSTMForecaster(
        input_size=num_links,
        hidden_size=CONFIG["lstm_hidden_size"],
        num_layers=CONFIG["lstm_num_layers"],
    )


def main():
    return run("lstm", _build_model)


if __name__ == "__main__":
    main()
