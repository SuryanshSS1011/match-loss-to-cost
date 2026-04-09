"""Tests for the Wilcoxon overlay on Pareto plots.

Two layers:
  - `_significance_overlay.significance_vs_reference` — the math.
  - `run_pareto._significance_per_ratio` + `plot_pareto(... significance=...)` —
    the integration through the runner-shaped aggregated JSONs.
"""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# significance_vs_reference
# ---------------------------------------------------------------------------

class TestSignificanceVsReference:
    def test_strict_beat_returns_true(self):
        from _significance_overlay import significance_vs_reference
        rows = {
            "MSE":  {"values": [10.0, 11.0, 9.0, 10.5, 9.5]},
            "ASYM": {"values": [3.0, 3.5, 2.5, 3.2, 2.8]},
        }
        out = significance_vs_reference(
            rows, reference="MSE", lower_is_better=True, alpha=0.05,
        )
        assert out["MSE"] is None
        assert out["ASYM"] is True  # ASYM strictly beats MSE

    def test_no_difference_returns_false(self):
        from _significance_overlay import significance_vs_reference
        rng_seed = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8]
        rows = {
            "MSE":  {"values": rng_seed},
            "TWIN": {"values": [v + 1e-9 for v in rng_seed]},
        }
        out = significance_vs_reference(rows, reference="MSE")
        assert out["TWIN"] is False  # essentially the same

    def test_unknown_reference_raises(self):
        from _significance_overlay import significance_vs_reference
        with pytest.raises(KeyError):
            significance_vs_reference(
                {"A": {"values": [1, 2, 3]}, "B": {"values": [1, 2, 3]}},
                reference="C",
            )

    def test_reference_without_values_raises(self):
        from _significance_overlay import significance_vs_reference
        rows = {
            "MSE":  {},  # no values
            "ASYM": {"values": [1.0, 2.0, 3.0]},
        }
        with pytest.raises(ValueError, match="no per-seed values"):
            significance_vs_reference(rows, reference="MSE")

    def test_only_reference_returns_all_none(self):
        from _significance_overlay import significance_vs_reference
        rows = {"MSE": {"values": [1.0, 2.0, 3.0]}}
        out = significance_vs_reference(rows, reference="MSE")
        assert out == {"MSE": None}


# ---------------------------------------------------------------------------
# _significance_per_ratio integration
# ---------------------------------------------------------------------------

class TestSignificancePerRatio:
    def test_extracts_values_per_ratio(self):
        from scripts.run_pareto import _significance_per_ratio
        by_ratio = {
            "1:1": {"models": {
                "MSE":  {"operational": {"overload_rate":
                    {"values": [0.5, 0.55, 0.45, 0.52, 0.48]}}},
                "ASYM": {"operational": {"overload_rate":
                    {"values": [0.10, 0.11, 0.09, 0.12, 0.08]}}},
            }},
            "5:1": {"models": {
                "MSE":  {"operational": {"overload_rate":
                    {"values": [0.4, 0.42, 0.38, 0.41, 0.39]}}},
                "ASYM": {"operational": {"overload_rate":
                    {"values": [0.40, 0.42, 0.38, 0.41, 0.39]}}},
            }},
        }
        out = _significance_per_ratio(
            by_ratio, reference="MSE", metric="overload_rate",
        )
        # 1:1: ASYM dominates → True
        # 5:1: ASYM matches MSE → False
        assert out["1:1"]["ASYM"] is True
        assert out["5:1"]["ASYM"] is False
        # Reference always None.
        assert out["1:1"]["MSE"] is None

    def test_missing_reference_returns_all_none(self):
        from scripts.run_pareto import _significance_per_ratio
        by_ratio = {
            "1:1": {"models": {
                "ASYM": {"operational": {"overload_rate":
                    {"values": [0.1, 0.2]}}},
            }},
        }
        out = _significance_per_ratio(by_ratio, reference="MSE")
        assert out["1:1"]["ASYM"] is None


# ---------------------------------------------------------------------------
# plot integration with significance overlay
# ---------------------------------------------------------------------------

class TestPlotWithOverlay:
    def test_plot_writes_with_significance(self, tmp_path):
        from scripts.run_pareto import plot_pareto
        points = {
            "MSE": [
                {"ratio": "1:1", "alpha": 1.0, "beta": 1.0,
                 "overload_rate": 0.5, "over_provisioning_cost": 10.0,
                 "asymmetric_op_cost": 100.0, "rmse_mean": 2.0},
            ],
            "ASYM": [
                {"ratio": "1:1", "alpha": 1.0, "beta": 1.0,
                 "overload_rate": 0.1, "over_provisioning_cost": 30.0,
                 "asymmetric_op_cost": 50.0, "rmse_mean": 2.1},
            ],
        }
        sig = {"1:1": {"MSE": None, "ASYM": True}}
        out = tmp_path / "p.png"
        plot_pareto(points, str(out), dataset="abilene",
                    significance=sig, wilcoxon_reference="MSE")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_works_without_significance(self, tmp_path):
        from scripts.run_pareto import plot_pareto
        points = {
            "MSE": [{"ratio": "1:1", "alpha": 1.0, "beta": 1.0,
                     "overload_rate": 0.5, "over_provisioning_cost": 10.0,
                     "asymmetric_op_cost": 100.0, "rmse_mean": 2.0}],
        }
        out = tmp_path / "p.png"
        plot_pareto(points, str(out), dataset="abilene")
        assert out.exists()
