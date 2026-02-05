"""Unit tests for src/losses.

Pure tensor math. No model training, no I/O. Locks in the contract that
`train_lstm.py` relies on:
  - mse / asymmetric_mse / pinball compute the right thing,
  - α = β reduces asymmetric_mse to MSE (up to a global scale),
  - τ = 0.5 reduces pinball to MAE / 2,
  - α > β makes the under-prediction gradient larger than the over-prediction gradient,
  - the factory dispatches strings → modules without surprise.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.losses import AsymmetricMSE, PinballLoss, asymmetric_mse, make_loss, pinball


def _t(*xs: float) -> torch.Tensor:
    return torch.tensor(list(xs), dtype=torch.float32)


class TestAsymmetricMSE:
    def test_zero_when_perfect(self):
        y = _t(1.0, 2.0, 3.0, 4.0)
        loss = asymmetric_mse(y, y, alpha=5.0, beta=1.0)
        assert float(loss) == pytest.approx(0.0, abs=1e-7)

    def test_alpha_equal_beta_is_scaled_mse(self):
        torch.manual_seed(0)
        y_pred = torch.randn(64)
        y_true = torch.randn(64)
        # asymmetric_mse with α=β=w returns w * (under² + over²).mean(),
        # while MSE returns ((y - ŷ)²).mean(). The element-wise sum
        # under² + over² == (y - ŷ)², so asym(α=β=w) == w * MSE.
        w = 3.0
        asym = asymmetric_mse(y_pred, y_true, alpha=w, beta=w)
        mse = nn.functional.mse_loss(y_pred, y_true)
        assert float(asym) == pytest.approx(float(w * mse), rel=1e-5)

    def test_under_prediction_dominates_when_alpha_big(self):
        # y > ŷ everywhere => under-prediction. α=10, β=1 should give 10× a
        # symmetric (1,1) loss on the same residuals.
        y_true = _t(2.0, 3.0, 4.0)
        y_pred = _t(1.0, 1.0, 1.0)  # under by 1, 2, 3
        big_alpha = asymmetric_mse(y_pred, y_true, alpha=10.0, beta=1.0)
        symmetric = asymmetric_mse(y_pred, y_true, alpha=1.0, beta=1.0)
        # Pure under-prediction case: asym = α · MSE_under, so 10× ratio.
        assert float(big_alpha) == pytest.approx(10.0 * float(symmetric), rel=1e-5)

    def test_only_over_when_pred_above_truth(self):
        # ŷ > y everywhere => only the β term contributes. α=99 should not
        # change the loss in this case.
        y_true = _t(1.0, 1.0, 1.0)
        y_pred = _t(2.0, 3.0, 4.0)  # over by 1, 2, 3
        beta_only = asymmetric_mse(y_pred, y_true, alpha=99.0, beta=2.0)
        # Mean of (1², 2², 3²) is 14/3, scaled by β=2 → 28/3.
        assert float(beta_only) == pytest.approx(28.0 / 3.0, rel=1e-5)

    def test_gradient_sign(self):
        # If we under-predict (y > ŷ), grad wrt ŷ should be negative
        # (raising ŷ reduces the loss). Conversely, over-prediction gives
        # a positive gradient.
        y_pred = torch.tensor([0.0, 0.0], requires_grad=True)
        y_true = _t(1.0, -1.0)  # first under, second over
        loss = asymmetric_mse(y_pred, y_true, alpha=5.0, beta=1.0)
        loss.backward()
        assert y_pred.grad is not None
        assert float(y_pred.grad[0]) < 0.0  # under: grad pushes ŷ up
        assert float(y_pred.grad[1]) > 0.0  # over:  grad pushes ŷ down

    def test_module_matches_functional(self):
        torch.manual_seed(1)
        y_pred = torch.randn(32)
        y_true = torch.randn(32)
        mod = AsymmetricMSE(alpha=5.0, beta=1.0)
        assert float(mod(y_pred, y_true)) == pytest.approx(
            float(asymmetric_mse(y_pred, y_true, 5.0, 1.0)), rel=1e-6
        )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            asymmetric_mse(torch.zeros(3), torch.zeros(4))

    def test_negative_weights_rejected(self):
        with pytest.raises(ValueError):
            AsymmetricMSE(alpha=-1.0, beta=1.0)
        with pytest.raises(ValueError):
            asymmetric_mse(torch.zeros(2), torch.zeros(2), alpha=1.0, beta=-1.0)


class TestPinball:
    def test_zero_when_perfect(self):
        y = _t(1.0, 2.0, 3.0)
        assert float(pinball(y, y, tau=0.5)) == pytest.approx(0.0, abs=1e-7)
        assert float(pinball(y, y, tau=0.83)) == pytest.approx(0.0, abs=1e-7)

    def test_median_equals_half_mae(self):
        torch.manual_seed(2)
        y_pred = torch.randn(64)
        y_true = torch.randn(64)
        # At τ = 0.5, pinball loss = 0.5 * |y - ŷ|.mean() == MAE / 2.
        pin = pinball(y_pred, y_true, tau=0.5)
        mae = (y_true - y_pred).abs().mean()
        assert float(pin) == pytest.approx(float(0.5 * mae), rel=1e-5)

    def test_high_tau_penalizes_under_prediction(self):
        # Two equal-magnitude residuals: one under (y > ŷ), one over (ŷ > y).
        # At τ = 0.9, the under residual gets weight 0.9, over gets 0.1.
        y_pred = _t(0.0, 0.0)
        y_true = _t(1.0, -1.0)
        loss = pinball(y_pred, y_true, tau=0.9)
        # Mean of (0.9 * 1, 0.1 * 1) = 0.5.
        assert float(loss) == pytest.approx(0.5, rel=1e-5)

    def test_factory_derives_tau_from_alpha_beta(self):
        loss = make_loss("pinball", alpha=5.0, beta=1.0)
        assert isinstance(loss, PinballLoss)
        assert loss.tau == pytest.approx(5.0 / 6.0)

    def test_invalid_tau_rejected(self):
        with pytest.raises(ValueError):
            PinballLoss(tau=0.0)
        with pytest.raises(ValueError):
            PinballLoss(tau=1.0)
        with pytest.raises(ValueError):
            pinball(torch.zeros(2), torch.zeros(2), tau=-0.1)


class TestFactory:
    def test_mse(self):
        loss = make_loss("mse")
        assert isinstance(loss, nn.MSELoss)

    def test_asym_aliases(self):
        for name in ("asym", "asymmetric_mse", "ASYM"):
            loss = make_loss(name, alpha=5.0, beta=1.0)
            assert isinstance(loss, AsymmetricMSE)
            assert loss.alpha == 5.0 and loss.beta == 1.0

    def test_asym_requires_weights(self):
        with pytest.raises(ValueError):
            make_loss("asym")

    def test_pinball_aliases(self):
        for name in ("pinball", "quantile"):
            loss = make_loss(name, tau=0.7)
            assert isinstance(loss, PinballLoss)
            assert loss.tau == pytest.approx(0.7)

    def test_pinball_requires_tau_or_weights(self):
        with pytest.raises(ValueError):
            make_loss("pinball")

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="unknown loss"):
            make_loss("does-not-exist")

    def test_module_callable_in_training_loop(self):
        # Sanity: factory output behaves like a real nn.Module loss.
        torch.manual_seed(3)
        criterion = make_loss("asym", alpha=5.0, beta=1.0)
        y_pred = torch.randn(16, requires_grad=True)
        y_true = torch.randn(16)
        loss = criterion(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None
        assert math.isfinite(float(loss.detach()))
