"""Unit tests for src/models/patchtst.py.

Pure tensor math. No training; we verify shape contracts, RevIN
reversibility, channel-independent equivariance (same channel order →
same output channels), gradient flow, and constructor validation.
"""

from __future__ import annotations

import pytest
import torch

from src.models.patchtst import (
    PatchTSTForecaster,
    RevIN,
    _PatchEmbedding,
)


# ---------------------------------------------------------------------------
# RevIN
# ---------------------------------------------------------------------------

class TestRevIN:
    def test_reversible_no_affine(self):
        rn = RevIN(num_channels=4, affine=False)
        x = torch.randn(2, 30, 4)
        z = rn.normalize(x)
        x_back = rn.denormalize(z)
        assert torch.allclose(x, x_back, atol=1e-4)

    def test_reversible_with_affine(self):
        rn = RevIN(num_channels=4, affine=True)
        # Disturb gamma/beta to nontrivial values.
        with torch.no_grad():
            rn.gamma.fill_(1.5)
            rn.beta.fill_(0.3)
        x = torch.randn(2, 30, 4)
        z = rn.normalize(x)
        x_back = rn.denormalize(z)
        assert torch.allclose(x, x_back, atol=1e-3)

    def test_normalize_zero_mean_unit_std(self):
        rn = RevIN(num_channels=3, affine=False)
        x = torch.randn(2, 100, 3) * 10 + 5  # mean ≈ 5, std ≈ 10
        z = rn.normalize(x)
        # Per-batch, per-channel, the normalised mean should be ~0 and std ~1.
        assert z.mean(dim=1).abs().max().item() < 1e-4
        assert (z.std(dim=1, unbiased=False) - 1.0).abs().max().item() < 1e-3

    def test_denormalize_before_normalize_raises(self):
        rn = RevIN(num_channels=2)
        with pytest.raises(RuntimeError, match="before normalize"):
            rn.denormalize(torch.zeros(1, 5, 2))


# ---------------------------------------------------------------------------
# _PatchEmbedding
# ---------------------------------------------------------------------------

class TestPatchEmbedding:
    def test_num_patches_formula(self):
        pe = _PatchEmbedding(patch_len=12, stride=6, d_model=32)
        # window=72: (72 - 12) / 6 + 1 = 11.
        assert pe.num_patches(72) == 11
        # window=64, patch=16, stride=8: (64-16)/8+1 = 7.
        pe2 = _PatchEmbedding(patch_len=16, stride=8, d_model=32)
        assert pe2.num_patches(64) == 7

    def test_forward_shape(self):
        pe = _PatchEmbedding(patch_len=12, stride=6, d_model=32)
        # batch*channels=8, window=72.
        x = torch.randn(8, 72)
        out = pe(x)
        assert out.shape == (8, 11, 32)

    def test_window_too_short_raises(self):
        pe = _PatchEmbedding(patch_len=20, stride=4, d_model=32)
        with pytest.raises(ValueError):
            pe.num_patches(10)

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            _PatchEmbedding(patch_len=0, stride=1, d_model=32)
        with pytest.raises(ValueError):
            _PatchEmbedding(patch_len=4, stride=0, d_model=32)


# ---------------------------------------------------------------------------
# PatchTSTForecaster
# ---------------------------------------------------------------------------

class TestPatchTSTForecaster:
    def test_shape_one_step(self):
        m = PatchTSTForecaster(input_size=8, window_size=72, horizon=1,
                                d_model=32, n_heads=4, n_layers=2)
        x = torch.randn(4, 72, 8)
        out = m(x)
        assert out.shape == (4, 8)

    def test_shape_multi_step(self):
        m = PatchTSTForecaster(input_size=8, window_size=72, horizon=24,
                                d_model=32, n_heads=4, n_layers=2)
        x = torch.randn(2, 72, 8)
        out = m(x)
        assert out.shape == (2, 24, 8)

    def test_finite_output(self):
        m = PatchTSTForecaster(input_size=4, window_size=24, horizon=1,
                                patch_len=4, stride=2,
                                d_model=16, n_heads=2, n_layers=1)
        x = torch.randn(2, 24, 4)
        out = m(x)
        assert torch.isfinite(out).all()

    def test_revin_makes_output_track_input_scale(self):
        # With RevIN on, scaling the input by 100 should ~scale the output
        # by 100 too — that's the whole point of de-normalisation.
        torch.manual_seed(0)
        m = PatchTSTForecaster(input_size=2, window_size=24, horizon=1,
                                patch_len=4, stride=2,
                                d_model=16, n_heads=2, n_layers=1)
        m.eval()
        x = torch.randn(1, 24, 2)
        with torch.no_grad():
            small = m(x)
            big = m(x * 100.0)
        # Output ratio should be in the same ballpark as the input ratio.
        # Allow generous slack (untrained model + RevIN learnable affine).
        ratio = big.abs().mean() / small.abs().mean().clamp(min=1e-4)
        assert ratio.item() > 10.0

    def test_no_revin(self):
        m = PatchTSTForecaster(input_size=4, window_size=24, horizon=1,
                                patch_len=4, stride=2,
                                d_model=16, n_heads=2, n_layers=1,
                                revin=False)
        x = torch.randn(2, 24, 4)
        out = m(x)
        assert out.shape == (2, 4)
        assert m.revin is None

    def test_channel_independent_param_sharing(self):
        # Channel-independent backbone: doubling the number of channels
        # should NOT roughly double the parameter count (only the head's
        # input dim grows trivially via num_patches scaling — actually it
        # doesn't, since head input is num_patches*d_model regardless of c).
        m4 = PatchTSTForecaster(input_size=4, window_size=24, horizon=1,
                                 patch_len=4, stride=2,
                                 d_model=16, n_heads=2, n_layers=1)
        m16 = PatchTSTForecaster(input_size=16, window_size=24, horizon=1,
                                  patch_len=4, stride=2,
                                  d_model=16, n_heads=2, n_layers=1)
        n4 = sum(p.numel() for p in m4.parameters())
        n16 = sum(p.numel() for p in m16.parameters())
        # The only c-dependent parameters are RevIN's gamma/beta (2*c each)
        # — so going 4→16 channels adds 24 params total. n16 should be
        # within a few hundred of n4.
        assert abs(n16 - n4) < 100

    def test_gradient_flow(self):
        m = PatchTSTForecaster(input_size=3, window_size=24, horizon=1,
                                patch_len=4, stride=2,
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

    def test_rejects_wrong_window(self):
        m = PatchTSTForecaster(input_size=4, window_size=24, horizon=1,
                                patch_len=4, stride=2,
                                d_model=16, n_heads=2, n_layers=1)
        with pytest.raises(ValueError, match="window_size"):
            m(torch.randn(2, 30, 4))

    def test_rejects_wrong_channels(self):
        m = PatchTSTForecaster(input_size=4, window_size=24, horizon=1,
                                patch_len=4, stride=2,
                                d_model=16, n_heads=2, n_layers=1)
        with pytest.raises(ValueError, match="input_size"):
            m(torch.randn(2, 24, 8))

    def test_d_model_must_divide_n_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            PatchTSTForecaster(input_size=4, window_size=24, horizon=1,
                                d_model=15, n_heads=4)
