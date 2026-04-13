"""Tests for scripts/build_pit_diagram.py.

Pure numpy + matplotlib. Verifies the PIT bin counts on canned inputs,
the reliability anchors, and the plot writer.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# coarse_pit
# ---------------------------------------------------------------------------

class TestCoarsePIT:
    def test_full_in_band(self):
        from build_pit_diagram import coarse_pit
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        q_lo = y - 1.0
        q_hi = y + 1.0
        pit = coarse_pit(y, q_lo, q_hi)
        assert pit["in_band"] == pytest.approx(1.0)
        assert pit["below_band"] == 0.0
        assert pit["above_band"] == 0.0

    def test_all_above(self):
        from build_pit_diagram import coarse_pit
        y = np.array([[10.0, 10.0], [10.0, 10.0]])
        q_lo = np.zeros_like(y)
        q_hi = np.ones_like(y)
        pit = coarse_pit(y, q_lo, q_hi)
        assert pit["above_band"] == pytest.approx(1.0)

    def test_all_below(self):
        from build_pit_diagram import coarse_pit
        y = np.array([[-10.0, -10.0]])
        q_lo = np.zeros_like(y)
        q_hi = np.ones_like(y)
        pit = coarse_pit(y, q_lo, q_hi)
        assert pit["below_band"] == pytest.approx(1.0)

    def test_balanced_split(self):
        from build_pit_diagram import coarse_pit
        y = np.array([1.0, 5.0, 9.0, 11.0]).reshape(-1, 1)
        q_lo = np.full_like(y, 3.0)
        q_hi = np.full_like(y, 10.0)
        pit = coarse_pit(y, q_lo, q_hi)
        # 1.0 < 3.0 → below; 5,9 in band; 11 above. → 1/4, 2/4, 1/4.
        assert pit["below_band"] == pytest.approx(0.25)
        assert pit["in_band"] == pytest.approx(0.5)
        assert pit["above_band"] == pytest.approx(0.25)

    def test_shape_mismatch_raises(self):
        from build_pit_diagram import coarse_pit
        with pytest.raises(ValueError, match="shape mismatch"):
            coarse_pit(np.zeros((3, 2)), np.zeros((3, 1)), np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# reliability_anchors
# ---------------------------------------------------------------------------

class TestReliability:
    def test_perfectly_calibrated(self):
        from build_pit_diagram import reliability_anchors
        # Build y from a uniform [0,1]; set q_lo = 0.05, q_hi = 0.95.
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 1, size=(10000, 1)).astype(np.float32)
        q_lo = np.full_like(y, 0.05)
        q_hi = np.full_like(y, 0.95)
        a = reliability_anchors(y, q_lo, q_hi, tau_lo=0.05, tau_hi=0.95)
        # P(y <= 0.05) ≈ 0.05, P(y <= 0.95) ≈ 0.95.
        assert abs(a["empirical_lo"] - 0.05) < 0.02
        assert abs(a["empirical_hi"] - 0.95) < 0.02

    def test_too_narrow_band(self):
        from build_pit_diagram import reliability_anchors
        # y standard normal; band is [-0.5, +0.5]. A perfect 0.05-quantile
        # is around -1.64 of a normal, so q_lo=-0.5 over-covers — empirical
        # P(y <= -0.5) ≈ 0.31. Higher than nominal 0.05 → q_lo is too high.
        rng = np.random.default_rng(1)
        y = rng.standard_normal(size=(10000, 1)).astype(np.float32)
        q_lo = np.full_like(y, -0.5)
        q_hi = np.full_like(y, 0.5)
        a = reliability_anchors(y, q_lo, q_hi, tau_lo=0.05, tau_hi=0.95)
        assert a["empirical_lo"] > 0.20  # under-coverage of left tail
        assert a["empirical_hi"] < 0.80  # under-coverage of right tail


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

class TestPlot:
    def test_writes_png(self, tmp_path):
        from build_pit_diagram import plot_pit_and_reliability
        pit = {"below_band": 0.05, "in_band": 0.90, "above_band": 0.05}
        anchors = {
            "tau_lo": 0.05, "empirical_lo": 0.06,
            "tau_hi": 0.95, "empirical_hi": 0.94,
        }
        out = tmp_path / "pit.png"
        plot_pit_and_reliability(pit, anchors, str(out),
                                  title="LSTM @ τ=[0.05, 0.95]")
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# _read_quantile_pair
# ---------------------------------------------------------------------------

class TestReadQuantilePair:
    def test_loads_arrays(self, tmp_path):
        from build_pit_diagram import _read_quantile_pair
        # Fabricate two npz files with the runner's qlo/qhi schema.
        T, num_links = 50, 3
        L_test_aligned = np.random.RandomState(0).rand(T, num_links).astype(np.float32)
        np.savez(
            tmp_path / "lstm_qlo_predictions.npz",
            predictions=L_test_aligned - 1.0,
            L_test_aligned=L_test_aligned,
            val_predictions=np.zeros_like(L_test_aligned),
            L_val_aligned=np.zeros_like(L_test_aligned),
        )
        np.savez(
            tmp_path / "lstm_qhi_predictions.npz",
            predictions=L_test_aligned + 1.0,
            L_test_aligned=L_test_aligned,
            val_predictions=np.zeros_like(L_test_aligned),
            L_val_aligned=np.zeros_like(L_test_aligned),
        )
        q_lo, q_hi, y = _read_quantile_pair(str(tmp_path), "lstm")
        assert q_lo.shape == (T, num_links)
        assert q_hi.shape == (T, num_links)
        assert y.shape == (T, num_links)
        assert np.allclose(q_hi - q_lo, 2.0)

    def test_missing_files_raises(self, tmp_path):
        from build_pit_diagram import _read_quantile_pair
        with pytest.raises(FileNotFoundError, match="missing quantile npz"):
            _read_quantile_pair(str(tmp_path), "lstm")
