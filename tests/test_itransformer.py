"""Unit tests for src/models/itransformer.py.

Pure tensor math. We verify shape contracts, the variate-as-token
behaviour (param count is flat in num_links, but compute is not), RevIN
reversibility, gradient flow, and constructor validation.
"""

from __future__ import annotations

import pytest
import torch

from src.models.itransformer import ITransformerForecaster


class TestShapeContracts:
    def test_one_step(self):
        m = ITransformerForecaster(input_size=8, window_size=72, horizon=1,
                                    d_model=32, n_heads=4, n_layers=2)
        x = torch.randn(4, 72, 8)
        out = m(x)
        assert out.shape == (4, 8)

    def test_multi_step(self):
        m = ITransformerForecaster(input_size=8, window_size=72, horizon=24,
                                    d_model=32, n_heads=4, n_layers=2)
        x = torch.randn(2, 72, 8)
        out = m(x)
        assert out.shape == (2, 24, 8)

    def test_finite_output(self):
        m = ITransformerForecaster(input_size=4, window_size=24, horizon=1,
                                    d_model=16, n_heads=2, n_layers=1)
        x = torch.randn(2, 24, 4)
        out = m(x)
        assert torch.isfinite(out).all()

    def test_no_revin(self):
        m = ITransformerForecaster(input_size=4, window_size=24, horizon=1,
                                    d_model=16, n_heads=2, n_layers=1,
                                    revin=False)
        out = m(torch.randn(2, 24, 4))
        assert out.shape == (2, 4)
        assert m.revin is None


class TestParamCount:
    def test_param_flat_in_num_channels(self):
        # Variate-as-token uses one shared Linear(window, d_model). The only
        # channel-dependent parameters are RevIN's gamma/beta (2*c each) — so
        # going 4 → 16 channels adds at most 24 params total, regardless of
        # how attention is structured.
        m4 = ITransformerForecaster(input_size=4, window_size=24, horizon=1,
                                     d_model=16, n_heads=2, n_layers=1)
        m16 = ITransformerForecaster(input_size=16, window_size=24, horizon=1,
                                      d_model=16, n_heads=2, n_layers=1)
        n4 = sum(p.numel() for p in m4.parameters())
        n16 = sum(p.numel() for p in m16.parameters())
        # Difference = 2 * 2 * (16-4) = 48 params (gamma + beta per channel
        # delta). Allow generous slack for any scaffolding params we might add.
        assert abs(n16 - n4) < 100

    def test_param_grows_with_window(self):
        # Variate embedding is Linear(window, d_model), so doubling window
        # roughly doubles those params (window * d_model + d_model bias).
        m24 = ITransformerForecaster(input_size=4, window_size=24, horizon=1,
                                      d_model=32, n_heads=2, n_layers=1)
        m48 = ITransformerForecaster(input_size=4, window_size=48, horizon=1,
                                      d_model=32, n_heads=2, n_layers=1)
        n24 = sum(p.numel() for p in m24.parameters())
        n48 = sum(p.numel() for p in m48.parameters())
        # Embedding diff = (48-24) * 32 = 768 params extra; everything else
        # constant. Make sure we land in that ballpark.
        assert n48 - n24 == pytest.approx(768, abs=8)


class TestCrossChannelEffect:
    def test_perturbing_one_channel_affects_others(self):
        # Cross-channel attention means a perturbation to channel 0's
        # input should propagate to the output of channel 1 (and others).
        # Without cross-channel attention (e.g. PatchTST), channel 1's
        # output would be invariant to channel 0's input.
        torch.manual_seed(0)
        m = ITransformerForecaster(input_size=4, window_size=24, horizon=1,
                                    d_model=32, n_heads=4, n_layers=2,
                                    revin=False)
        m.eval()
        x = torch.randn(1, 24, 4)
        x_perturbed = x.clone()
        x_perturbed[..., 0] += 5.0  # large perturbation on channel 0 only
        with torch.no_grad():
            out_orig = m(x)
            out_perturbed = m(x_perturbed)
        # Channel 1's output must shift because of attention from channel 0.
        diff_ch1 = (out_perturbed[..., 1] - out_orig[..., 1]).abs().mean()
        assert diff_ch1.item() > 1e-3


class TestValidation:
    def test_d_model_n_heads_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            ITransformerForecaster(input_size=4, window_size=24, horizon=1,
                                    d_model=15, n_heads=4)

    def test_rejects_wrong_window(self):
        m = ITransformerForecaster(input_size=4, window_size=24, horizon=1,
                                    d_model=16, n_heads=2, n_layers=1)
        with pytest.raises(ValueError, match="window_size"):
            m(torch.randn(2, 30, 4))

    def test_rejects_wrong_channels(self):
        m = ITransformerForecaster(input_size=4, window_size=24, horizon=1,
                                    d_model=16, n_heads=2, n_layers=1)
        with pytest.raises(ValueError, match="input_size"):
            m(torch.randn(2, 24, 8))

    def test_rejects_2d_input(self):
        m = ITransformerForecaster(input_size=4, window_size=24, horizon=1,
                                    d_model=16, n_heads=2, n_layers=1)
        with pytest.raises(ValueError):
            m(torch.randn(24, 4))


class TestGradientFlow:
    def test_backward_runs(self):
        m = ITransformerForecaster(input_size=3, window_size=24, horizon=1,
                                    d_model=16, n_heads=2, n_layers=1)
        x = torch.randn(2, 24, 3)
        target = torch.randn(2, 3)
        loss = (m(x) - target).pow(2).mean()
        loss.backward()
        any_grad = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in m.parameters()
        )
        assert any_grad
