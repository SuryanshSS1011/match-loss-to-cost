"""Unit tests for src/calibration/cqr.py.

Pure numpy. The point: lock the CQR calibration math against a few worked
examples + a coverage smoke test, before the cloud sweep wires it into the
runner. No model training, no external data.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    capacity_from_cqr_upper,
    cqr_calibrate,
    cqr_predict,
    cqr_split,
    empirical_coverage,
)
from src.calibration.cqr import _finite_sample_quantile


# ---------------------------------------------------------------------------
# _finite_sample_quantile
# ---------------------------------------------------------------------------

class TestFiniteSampleQuantile:
    def test_matches_ceil_formula(self):
        # n=10, alpha=0.1 → k = ⌈11 * 0.9⌉ = 10 → returns max.
        scores = np.linspace(0, 9, num=10)
        q = _finite_sample_quantile(scores, alpha=0.1)
        assert q == pytest.approx(9.0)

    def test_caps_at_max_when_k_overflows(self):
        # n=5, alpha=0.05 → k = ⌈6 * 0.95⌉ = 6 > n; should clip to n=5 → max.
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = _finite_sample_quantile(scores, alpha=0.05)
        assert q == 5.0

    def test_handles_alpha_one(self):
        # Edge: alpha→1 means we want the smallest quantile. k = max(1, ...).
        scores = np.array([5.0, 1.0, 3.0])
        q = _finite_sample_quantile(scores, alpha=0.99)
        assert q == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _finite_sample_quantile(np.array([]), alpha=0.1)


# ---------------------------------------------------------------------------
# cqr_calibrate
# ---------------------------------------------------------------------------

class TestCqrCalibrate:
    def test_perfect_quantile_yields_negative_qhat(self):
        # If q_lo and q_hi already sandwich y comfortably, the score
        # max(q_lo - y, y - q_hi) is *negative* on every cell. qhat is
        # then a negative quantile, and the calibrated band shrinks.
        rng = np.random.default_rng(0)
        T, K = 200, 3
        y = rng.uniform(0, 10, size=(T, K)).astype(np.float32)
        q_lo = y - 5.0
        q_hi = y + 5.0
        qhat = cqr_calibrate(y, q_lo, q_hi, alpha=0.1)
        assert qhat.shape == (K,)
        assert (qhat < 0).all()

    def test_too_narrow_quantile_yields_positive_qhat(self):
        rng = np.random.default_rng(1)
        T, K = 200, 4
        y = rng.uniform(0, 10, size=(T, K)).astype(np.float32)
        q_lo = y + 1.0  # always above y → systematic miscoverage
        q_hi = y - 1.0  # always below y → systematic miscoverage
        qhat = cqr_calibrate(y, q_lo, q_hi, alpha=0.1)
        assert (qhat > 0).all()

    def test_per_link_qhats_can_differ(self):
        # Build two links with different residual scales.
        T = 200
        y = np.zeros((T, 2), dtype=np.float32)
        rng = np.random.default_rng(2)
        # Link 0: tight residuals; link 1: wide residuals.
        y[:, 0] = rng.normal(0, 1, size=T)
        y[:, 1] = rng.normal(0, 10, size=T)
        q_lo = np.full_like(y, 0.0)
        q_hi = np.full_like(y, 0.0)
        qhat = cqr_calibrate(y, q_lo, q_hi, alpha=0.1)
        # Wider link should need a larger qhat.
        assert qhat[1] > qhat[0]

    def test_global_qhat_is_uniform(self):
        rng = np.random.default_rng(3)
        T, K = 200, 5
        y = rng.uniform(0, 10, size=(T, K)).astype(np.float32)
        q_lo = y - 1.0
        q_hi = y + 1.0
        qhat = cqr_calibrate(y, q_lo, q_hi, alpha=0.1, per_link=False)
        assert qhat.shape == (K,)
        # All entries identical because per_link=False.
        assert np.allclose(qhat, qhat[0])

    def test_invalid_alpha_rejected(self):
        y = np.zeros((10, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="alpha"):
            cqr_calibrate(y, y, y, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            cqr_calibrate(y, y, y, alpha=0.0)

    def test_shape_validation(self):
        y = np.zeros((10, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            cqr_calibrate(y, y[:, :1], y, alpha=0.1)


# ---------------------------------------------------------------------------
# cqr_predict
# ---------------------------------------------------------------------------

class TestCqrPredict:
    def test_basic_arithmetic(self):
        T, K = 50, 3
        q_lo = np.full((T, K), 1.0, dtype=np.float32)
        q_hi = np.full((T, K), 5.0, dtype=np.float32)
        qhat = np.array([0.5, 1.0, 2.0])
        lo, hi = cqr_predict(q_lo, q_hi, qhat)
        assert np.allclose(lo[:, 0], 0.5)
        assert np.allclose(lo[:, 1], 0.0)
        assert np.allclose(lo[:, 2], -1.0)
        assert np.allclose(hi[:, 0], 5.5)
        assert np.allclose(hi[:, 1], 6.0)
        assert np.allclose(hi[:, 2], 7.0)

    def test_scalar_qhat_broadcasts(self):
        q_lo = np.zeros((10, 4), dtype=np.float32)
        q_hi = np.ones((10, 4), dtype=np.float32)
        lo, hi = cqr_predict(q_lo, q_hi, np.float64(0.5))
        assert lo.shape == hi.shape == (10, 4)
        assert np.allclose(lo, -0.5)
        assert np.allclose(hi, 1.5)

    def test_bad_qhat_shape(self):
        q_lo = np.zeros((10, 4), dtype=np.float32)
        q_hi = np.ones((10, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="not broadcastable"):
            cqr_predict(q_lo, q_hi, np.array([0.5, 0.5]))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            cqr_predict(
                np.zeros((10, 4), dtype=np.float32),
                np.zeros((10, 5), dtype=np.float32),
                np.zeros(4),
            )


# ---------------------------------------------------------------------------
# empirical_coverage
# ---------------------------------------------------------------------------

class TestEmpiricalCoverage:
    def test_full_coverage(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        lo = y - 1.0
        hi = y + 1.0
        out = empirical_coverage(y, lo, hi)
        assert out["coverage_overall"] == 1.0
        assert all(c == 1.0 for c in out["coverage_per_link"])
        assert out["mean_width"] == 2.0

    def test_zero_coverage(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        # Bands strictly above y → coverage = 0.
        lo = y + 5.0
        hi = y + 10.0
        out = empirical_coverage(y, lo, hi)
        assert out["coverage_overall"] == 0.0

    def test_partial_coverage(self):
        y = np.array([[1.0], [10.0], [100.0]])
        lo = np.array([[0.0], [9.0], [200.0]])
        hi = np.array([[2.0], [11.0], [201.0]])
        # First two cells covered; third is below the band.
        out = empirical_coverage(y, lo, hi)
        assert out["coverage_overall"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# end-to-end via cqr_split
# ---------------------------------------------------------------------------

class TestCqrSplit:
    def test_coverage_hits_target(self):
        # Build a controlled scenario: y ~ N(0, 1) per link, q_lo and q_hi
        # come from a *too-narrow* quantile predictor (50% nominal coverage).
        # CQR should widen the band so test coverage approaches 1 - alpha.
        rng = np.random.default_rng(0)
        T, K = 4000, 5
        y = rng.standard_normal((T, K)).astype(np.float32)
        # Predict q_lo, q_hi as an interval that nominally covers ~50% — i.e.
        # the 25th and 75th percentile of a standard normal: ±0.6745.
        q_lo = np.full_like(y, -0.6745)
        q_hi = np.full_like(y, +0.6745)
        out = cqr_split(y, q_lo, q_hi, alpha=0.1, cal_frac=0.5, per_link=True)
        # With ~2000 cal points per link, finite-sample slack is small;
        # expect coverage ∈ [0.85, 0.95].
        assert 0.85 <= out["coverage"]["coverage_overall"] <= 0.97

    def test_per_link_widths_independent(self):
        # Two links with very different residual scales — calibrated widths
        # should also differ.
        rng = np.random.default_rng(1)
        T = 2000
        y = np.column_stack([
            rng.standard_normal(T),
            rng.standard_normal(T) * 10.0,
        ]).astype(np.float32)
        q_lo = np.zeros_like(y)
        q_hi = np.zeros_like(y)
        out = cqr_split(y, q_lo, q_hi, alpha=0.1, cal_frac=0.5, per_link=True)
        widths = out["coverage"]["mean_width_per_link"]
        # Wide-residual link has the wider calibrated interval.
        assert widths[1] > widths[0] * 5

    def test_invalid_split(self):
        y = np.zeros((10, 1), dtype=np.float32)
        with pytest.raises(ValueError):
            cqr_split(y, y, y, alpha=0.1, cal_frac=1.5)


# ---------------------------------------------------------------------------
# capacity_from_cqr_upper
# ---------------------------------------------------------------------------

class TestCapacityFromCqrUpper:
    def test_returns_per_link_max(self):
        upper = np.array([[1.0, 5.0], [3.0, 4.0], [2.0, 6.0]])
        cap = capacity_from_cqr_upper(upper, margin=1.0)
        assert cap.shape == (2,)
        assert cap.tolist() == [3.0, 6.0]

    def test_margin_scales(self):
        upper = np.array([[1.0, 5.0], [3.0, 4.0]])
        cap = capacity_from_cqr_upper(upper, margin=1.5)
        assert np.allclose(cap, [4.5, 7.5])

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            capacity_from_cqr_upper(np.array([1.0, 2.0, 3.0]))
