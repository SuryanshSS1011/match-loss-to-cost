"""Unit tests for src/models/dlinear.py.

Pure tensor math. No training; we verify the model is a well-formed
`nn.Module` that respects shape contracts and produces finite output.
"""

from __future__ import annotations

import pytest
import torch

from src.models.dlinear import (
    DLinearForecaster,
    _MovingAverage,
    _SeriesDecomposition,
)


class TestMovingAverage:
    def test_preserves_shape(self):
        ma = _MovingAverage(kernel_size=5)
        x = torch.randn(2, 30, 4)
        out = ma(x)
        assert out.shape == x.shape

    def test_constant_input_unchanged(self):
        ma = _MovingAverage(kernel_size=7)
        x = torch.full((1, 20, 3), 3.14)
        out = ma(x)
        assert torch.allclose(out, x)

    def test_rejects_even_kernel(self):
        with pytest.raises(ValueError):
            _MovingAverage(kernel_size=4)


class TestSeriesDecomposition:
    def test_seasonal_plus_trend_reconstructs(self):
        decomp = _SeriesDecomposition(kernel_size=9)
        x = torch.randn(2, 50, 6)
        seasonal, trend = decomp(x)
        assert torch.allclose(x, seasonal + trend, atol=1e-5)


class TestDLinearForecaster:
    def test_shape_one_step(self):
        model = DLinearForecaster(input_size=8, window_size=72, horizon=1)
        x = torch.randn(4, 72, 8)
        out = model(x)
        assert out.shape == (4, 8)

    def test_shape_multi_step(self):
        model = DLinearForecaster(input_size=8, window_size=72, horizon=12)
        x = torch.randn(3, 72, 8)
        out = model(x)
        assert out.shape == (3, 12, 8)

    def test_shared_vs_individual(self):
        ind = DLinearForecaster(input_size=4, window_size=24, horizon=1,
                                individual=True)
        shr = DLinearForecaster(input_size=4, window_size=24, horizon=1,
                                individual=False)
        n_ind = sum(p.numel() for p in ind.parameters())
        n_shr = sum(p.numel() for p in shr.parameters())
        # Individual should have ~num_links× more parameters.
        assert n_ind > n_shr * 2

    def test_finite_output(self):
        model = DLinearForecaster(input_size=4, window_size=24, horizon=1)
        x = torch.randn(2, 24, 4)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_rejects_wrong_window(self):
        model = DLinearForecaster(input_size=4, window_size=24, horizon=1)
        with pytest.raises(ValueError, match="window_size"):
            model(torch.randn(2, 30, 4))

    def test_rejects_wrong_channels(self):
        model = DLinearForecaster(input_size=4, window_size=24, horizon=1)
        with pytest.raises(ValueError, match="input_size"):
            model(torch.randn(2, 24, 8))

    def test_backward_pass_runs(self):
        # Make sure gradients flow — this is the cheapest end-to-end check
        # we can do without a real training loop.
        model = DLinearForecaster(input_size=3, window_size=12, horizon=1)
        x = torch.randn(2, 12, 3, requires_grad=False)
        target = torch.randn(2, 3)
        out = model(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        # At least one parameter should have a non-zero gradient.
        any_grad = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in model.parameters()
        )
        assert any_grad
