"""Tests for scripts/run_pareto.py.

Stubs `main_programmatic` so we don't actually train, then verifies:
  - parse_ratio handles valid + invalid inputs.
  - cell_output_dir lays out per-ratio directories deterministically.
  - collect_points pulls the right keys, sorts by α.
  - plot_pareto writes a non-empty PNG and skips _CQR/_ACI rows.
  - --from-cache reads existing aggregated JSONs without retraining.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


def _agg(model: str, *, overload: float, over_prov: float,
         asym_op: float = 0.0, rmse: float = 1.0) -> dict:
    return {
        "models": {
            model: {
                "operational": {
                    "overload_rate": {"mean": overload, "std": 0.01},
                    "over_provisioning_cost": {"mean": over_prov, "std": 1.0},
                    "asymmetric_op_cost": {"mean": asym_op, "std": 1.0},
                },
                "forecast": {
                    "rmse_mean": {"mean": rmse, "std": 0.1},
                },
            }
        }
    }


# ---------------------------------------------------------------------------
# parse_ratio
# ---------------------------------------------------------------------------

class TestParseRatio:
    def test_valid(self):
        from scripts.run_pareto import parse_ratio
        assert parse_ratio("5:1") == (5.0, 1.0)
        assert parse_ratio("0.5:0.5") == (0.5, 0.5)

    def test_no_colon_raises(self):
        from scripts.run_pareto import parse_ratio
        with pytest.raises(ValueError, match="must be 'a:b'"):
            parse_ratio("51")

    def test_too_many_colons(self):
        from scripts.run_pareto import parse_ratio
        with pytest.raises(ValueError, match="exactly one colon"):
            parse_ratio("1:2:3")

    def test_non_numeric(self):
        from scripts.run_pareto import parse_ratio
        with pytest.raises(ValueError, match="numeric"):
            parse_ratio("a:b")

    def test_negative_rejected(self):
        from scripts.run_pareto import parse_ratio
        with pytest.raises(ValueError, match="positive"):
            parse_ratio("-1:1")
        with pytest.raises(ValueError, match="positive"):
            parse_ratio("1:0")


# ---------------------------------------------------------------------------
# cell_output_dir
# ---------------------------------------------------------------------------

class TestCellOutputDir:
    def test_layout(self, tmp_path):
        from scripts.run_pareto import cell_output_dir
        # Default loss_form is "asym" (squared); new layout includes it
        # so squared and L1 sweeps don't collide.
        d = cell_output_dir("abilene", "5:1", base_dir=str(tmp_path))
        assert d == str(tmp_path / "abilene_pareto_asym" / "ratio_5_1")

    def test_fractional(self, tmp_path):
        from scripts.run_pareto import cell_output_dir
        d = cell_output_dir("abilene", "0.5:0.5", base_dir=str(tmp_path))
        assert d.endswith("ratio_0.5_0.5")

    def test_loss_form_separates_paths(self, tmp_path):
        """Squared and cusp-linear sweeps must NEVER share a cell dir."""
        from scripts.run_pareto import cell_output_dir
        d_asym = cell_output_dir("abilene", "5:1",
                                 base_dir=str(tmp_path), loss_form="asym")
        d_l1 = cell_output_dir("abilene", "5:1",
                               base_dir=str(tmp_path), loss_form="asym_l1")
        assert d_asym != d_l1
        assert d_asym.endswith("abilene_pareto_asym/ratio_5_1")
        assert d_l1.endswith("abilene_pareto_asym_l1/ratio_5_1")


# ---------------------------------------------------------------------------
# collect_points
# ---------------------------------------------------------------------------

class TestCollectPoints:
    def test_extracts_means_per_model(self):
        from scripts.run_pareto import collect_points
        by_ratio = {
            "1:1": _agg("LSTM", overload=0.5, over_prov=10.0, rmse=2.0),
            "5:1": _agg("LSTM", overload=0.2, over_prov=20.0, rmse=2.1),
            "10:1": _agg("LSTM", overload=0.05, over_prov=35.0, rmse=2.2),
        }
        points = collect_points(by_ratio)
        assert "LSTM" in points
        assert len(points["LSTM"]) == 3
        # Sorted by α.
        alphas = [e["alpha"] for e in points["LSTM"]]
        assert alphas == sorted(alphas)
        # Values pulled correctly.
        first = points["LSTM"][0]
        assert first["alpha"] == 1.0
        assert first["overload_rate"] == 0.5
        assert first["over_provisioning_cost"] == 10.0
        assert first["rmse_mean"] == 2.0

    def test_handles_multiple_models(self):
        from scripts.run_pareto import collect_points
        agg_lstm_sarima = {
            "models": {
                "LSTM": _agg("LSTM", overload=0.1, over_prov=5.0)["models"]["LSTM"],
                "SARIMA": _agg("SARIMA", overload=0.5, over_prov=2.0)["models"]["SARIMA"],
            }
        }
        points = collect_points({"5:1": agg_lstm_sarima})
        assert set(points.keys()) == {"LSTM", "SARIMA"}

    def test_missing_keys_become_none(self):
        from scripts.run_pareto import collect_points
        agg = {
            "models": {
                "LSTM": {
                    "operational": {},  # no overload_rate
                    "forecast": {},
                }
            }
        }
        points = collect_points({"1:1": agg})
        assert points["LSTM"][0]["overload_rate"] is None
        assert points["LSTM"][0]["over_provisioning_cost"] is None


# ---------------------------------------------------------------------------
# plot_pareto
# ---------------------------------------------------------------------------

class TestPlot:
    def test_writes_png(self, tmp_path):
        from scripts.run_pareto import plot_pareto
        points = {
            "LSTM": [
                {"ratio": "1:1", "alpha": 1.0, "beta": 1.0,
                 "overload_rate": 0.5, "over_provisioning_cost": 10.0,
                 "asymmetric_op_cost": 50.0, "rmse_mean": 2.0},
                {"ratio": "5:1", "alpha": 5.0, "beta": 1.0,
                 "overload_rate": 0.2, "over_provisioning_cost": 20.0,
                 "asymmetric_op_cost": 30.0, "rmse_mean": 2.1},
            ],
        }
        out = tmp_path / "pareto.png"
        plot_pareto(points, str(out), dataset="abilene")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_skips_calibration_rows(self, tmp_path):
        # _CQR / _ACI rows must NOT appear on the headline plot.
        from scripts.run_pareto import plot_pareto
        points = {
            "LSTM_CQR": [
                {"ratio": "1:1", "alpha": 1.0, "beta": 1.0,
                 "overload_rate": 0.05, "over_provisioning_cost": 50.0,
                 "asymmetric_op_cost": 5.0, "rmse_mean": 2.0},
            ],
        }
        out = tmp_path / "pareto.png"
        plot_pareto(points, str(out))
        # Plot should still write (empty figure), not crash.
        assert out.exists()


# ---------------------------------------------------------------------------
# _load_or_run cache mode
# ---------------------------------------------------------------------------

class TestFromCache:
    def test_from_cache_reads_existing_file(self, tmp_path):
        from scripts.run_pareto import _load_or_run, cell_output_dir
        # Pre-populate the cache file.
        cell_dir = cell_output_dir("abilene", "5:1", base_dir=str(tmp_path))
        os.makedirs(cell_dir, exist_ok=True)
        canned = _agg("LSTM", overload=0.2, over_prov=20.0)
        with open(os.path.join(cell_dir, "aggregated_results.json"), "w") as f:
            json.dump(canned, f)

        out = _load_or_run("abilene", "5:1", seeds=[42], models=("lstm",),
                          from_cache=True, base_dir=str(tmp_path))
        assert out == canned

    def test_from_cache_missing_file_raises(self, tmp_path):
        from scripts.run_pareto import _load_or_run
        with pytest.raises(FileNotFoundError, match="missing"):
            _load_or_run("abilene", "5:1", seeds=[42], models=("lstm",),
                         from_cache=True, base_dir=str(tmp_path))

    def test_no_cache_calls_main_programmatic(self, tmp_path, monkeypatch):
        from scripts.run_pareto import _load_or_run
        from scripts import run_experiments

        calls = []

        def _fake(*, dataset, loss, alpha, beta, tau, seeds, models,
                  output_dir):
            calls.append({
                "dataset": dataset, "loss": loss,
                "alpha": alpha, "beta": beta,
                "models": models, "output_dir": output_dir,
            })
            return _agg("LSTM", overload=0.1, over_prov=5.0)

        monkeypatch.setattr(run_experiments, "main_programmatic", _fake)
        out = _load_or_run("abilene", "10:1", seeds=[42],
                          models=("lstm",), from_cache=False,
                          base_dir=str(tmp_path))
        assert len(calls) == 1
        assert calls[0]["loss"] == "asym"
        assert calls[0]["alpha"] == 10.0
        assert calls[0]["beta"] == 1.0
        assert "ratio_10_1" in calls[0]["output_dir"]
        assert out["models"]["LSTM"]["operational"]["overload_rate"]["mean"] == 0.1
