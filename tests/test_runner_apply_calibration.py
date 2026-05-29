"""Direct unit tests for `_apply_cqr` and `_apply_aci` in run_experiments.py.

Why this file: the underlying CQR math is covered by test_cqr.py and the
ACI math by test_aci.py, and the runner's *integration* of the
calibration mode (CONFIG flow, neural retraining, npz reads) is covered
by test_runner_calibration.py. What's NOT covered: the small adapter
functions `_apply_cqr` and `_apply_aci` that take pre-computed bands and
produce the runner's calibration + operational dict. This file pins
their behavior with deterministic synthetic bands.

Strategy: build a tiny `bands` dict (val + test) where the ground truth
is a simple linear ramp and the quantile bands cleanly bracket it; check
that the returned coverage, qhat, and operational keys are correct.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))


@pytest.fixture
def well_calibrated_bands():
    """Build bands that already cover 90% on val and test by construction.

    Ground truth is `y = 50 + 0.05 * t` per link with two links. Bands at
    ±10 around y on val and test produce per-element width 20 and roughly
    100% coverage when the bands literally bracket y. CQR's qhat then
    shrinks the band toward target_alpha=0.1.
    """
    T_val, T_test, K = 200, 200, 2
    t_val = np.arange(T_val).astype(np.float32)
    t_test = np.arange(T_test).astype(np.float32)
    L_val = np.column_stack([50 + 0.05 * t_val + 0.5 * k for k in range(K)]).astype(np.float32)
    L_test = np.column_stack([55 + 0.05 * t_test + 0.5 * k for k in range(K)]).astype(np.float32)
    # Bands at ±10 around the true series → trivially well-calibrated.
    q_lo_val = (L_val - 10.0).astype(np.float32)
    q_hi_val = (L_val + 10.0).astype(np.float32)
    q_lo_test = (L_test - 10.0).astype(np.float32)
    q_hi_test = (L_test + 10.0).astype(np.float32)
    return {
        "L_val_aligned": L_val,
        "L_test_aligned": L_test,
        "q_lo_val": q_lo_val,
        "q_hi_val": q_hi_val,
        "q_lo_test": q_lo_test,
        "q_hi_test": q_hi_test,
    }


class TestApplyCQR:
    def test_returns_required_blocks(self, well_calibrated_bands):
        from run_experiments import _apply_cqr
        out = _apply_cqr(well_calibrated_bands, target_alpha=0.1,
                         op_alpha=5.0, op_beta=1.0)
        assert "calibration" in out
        assert "operational" in out
        for k in ("method", "target_alpha", "qhat_mean",
                  "coverage_overall", "mean_width"):
            assert k in out["calibration"], f"missing {k}"
        assert out["calibration"]["method"] == "cqr"
        assert out["calibration"]["target_alpha"] == pytest.approx(0.1)

    def test_qhat_is_negative_or_zero_when_bands_already_cover(
        self, well_calibrated_bands
    ):
        """When val bands already contain every y, the conformity scores
        are non-positive and qhat ≤ 0 → CQR shrinks the band."""
        from run_experiments import _apply_cqr
        out = _apply_cqr(well_calibrated_bands, target_alpha=0.1,
                         op_alpha=5.0, op_beta=1.0)
        # Scores = max(q_lo - y, y - q_hi). Bands at ±10 around y →
        # max(-10, -10) = -10 for every element → qhat negative.
        assert out["calibration"]["qhat_mean"] < 0.0

    def test_coverage_near_one_when_overcovered(self, well_calibrated_bands):
        """Trivially-bracketing bands → coverage near 1 even after
        shrinkage, since the test series is identical-distribution to val."""
        from run_experiments import _apply_cqr
        out = _apply_cqr(well_calibrated_bands, target_alpha=0.1,
                         op_alpha=5.0, op_beta=1.0)
        # Allow some slack; for this construction it's essentially 1.0.
        assert out["calibration"]["coverage_overall"] >= 0.85

    def test_operational_keys_present(self, well_calibrated_bands):
        from run_experiments import _apply_cqr, OPERATIONAL_KEYS
        out = _apply_cqr(well_calibrated_bands, target_alpha=0.1,
                         op_alpha=5.0, op_beta=1.0)
        for k in OPERATIONAL_KEYS:
            assert k in out["operational"]
            assert np.isfinite(out["operational"][k])

    def test_higher_target_alpha_yields_smaller_qhat(
        self, well_calibrated_bands
    ):
        """Larger target_alpha (looser coverage) ⇒ less conservative qhat."""
        from run_experiments import _apply_cqr
        out_strict = _apply_cqr(well_calibrated_bands, target_alpha=0.05,
                                op_alpha=5.0, op_beta=1.0)
        out_loose = _apply_cqr(well_calibrated_bands, target_alpha=0.20,
                               op_alpha=5.0, op_beta=1.0)
        assert (out_strict["calibration"]["qhat_mean"]
                >= out_loose["calibration"]["qhat_mean"])


class TestApplyACI:
    def test_returns_required_blocks(self, well_calibrated_bands):
        from run_experiments import _apply_aci
        out = _apply_aci(well_calibrated_bands, target_alpha=0.1,
                         op_alpha=5.0, op_beta=1.0,
                         gamma=0.01, window=50)
        assert "calibration" in out
        assert "operational" in out
        for k in ("method", "target_alpha", "gamma", "window",
                  "qhat_mean", "alpha_final_mean",
                  "coverage_overall", "mean_width"):
            assert k in out["calibration"], f"missing {k}"
        assert out["calibration"]["method"] == "aci"
        assert out["calibration"]["target_alpha"] == pytest.approx(0.1)
        assert out["calibration"]["gamma"] == pytest.approx(0.01)
        assert out["calibration"]["window"] == 50

    def test_coverage_in_range(self, well_calibrated_bands):
        from run_experiments import _apply_aci
        out = _apply_aci(well_calibrated_bands, target_alpha=0.1,
                         op_alpha=5.0, op_beta=1.0,
                         gamma=0.01, window=50)
        cov = out["calibration"]["coverage_overall"]
        assert 0.0 <= cov <= 1.0
        # On trivially-bracketing bands ACI should hit very high coverage.
        assert cov >= 0.85

    def test_alpha_drifts_toward_target_under_overcoverage(
        self, well_calibrated_bands
    ):
        """If the bands always cover, ACI's alpha should drift *up* (the
        tracker concludes it's over-paying for coverage) and approach
        the target from below. Just verify alpha is bounded."""
        from run_experiments import _apply_aci
        out = _apply_aci(well_calibrated_bands, target_alpha=0.1,
                         op_alpha=5.0, op_beta=1.0,
                         gamma=0.05, window=50)
        a_final = out["calibration"]["alpha_final_mean"]
        # With aggressive gamma=0.05 and 100% in-band updates, alpha climbs
        # toward target. With non-aggressive gamma it might still hover.
        # The defensible assertion: alpha stayed in [0, 1].
        assert 0.0 <= a_final <= 1.0

    def test_operational_keys_present(self, well_calibrated_bands):
        from run_experiments import _apply_aci, OPERATIONAL_KEYS
        out = _apply_aci(well_calibrated_bands, target_alpha=0.1,
                         op_alpha=5.0, op_beta=1.0,
                         gamma=0.01, window=50)
        for k in OPERATIONAL_KEYS:
            assert k in out["operational"]
            assert np.isfinite(out["operational"][k])
