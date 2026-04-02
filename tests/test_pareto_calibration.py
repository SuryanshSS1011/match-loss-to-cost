"""Tests for scripts/run_pareto_calibration.py.

Stubs `scripts.run_experiments.main_programmatic` so we don't actually
train. Verifies:
  - cell_output_dir lays out per-target_alpha directories.
  - collect_calibration_points pulls only _CQR / _ACI rows, sorted by
    target_alpha; ignores plain point-forecast rows.
  - plot_calibration_pareto writes a non-empty PNG with horizontal
    target-coverage lines.
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


def _agg_with_calibration(
    *,
    point_models: dict[str, dict] | None = None,
    cqr_models: dict[str, dict] | None = None,
    aci_models: dict[str, dict] | None = None,
) -> dict:
    """Build a runner-shaped aggregated dict with the requested rows."""
    models: dict[str, dict] = {}
    for name, cal in (point_models or {}).items():
        models[name] = {
            "operational": {
                "overload_rate": {"mean": 0.1, "std": 0.01},
            },
            "forecast": {"rmse_mean": {"mean": 1.0, "std": 0.1}},
        }
    for suffix, container in (("_CQR", cqr_models or {}),
                              ("_ACI", aci_models or {})):
        for name, vals in container.items():
            row = name + suffix
            models[row] = {
                "operational": {
                    "overload_rate": {"mean": 0.05, "std": 0.01},
                },
                "forecast": {"rmse_mean": {"mean": None}},
                "calibration": {
                    "coverage_overall": {"mean": vals["coverage"], "std": 0.01},
                    "mean_width": {"mean": vals["width"], "std": 0.1},
                    "qhat_mean": {"mean": vals.get("qhat", 1.0), "std": 0.1},
                    "target_alpha": {"mean": vals.get("target", 0.1)},
                },
            }
    return {"models": models}


# ---------------------------------------------------------------------------
# cell_output_dir
# ---------------------------------------------------------------------------

class TestCellOutputDir:
    def test_layout(self, tmp_path):
        from scripts.run_pareto_calibration import cell_output_dir
        d = cell_output_dir("abilene", 0.1, base_dir=str(tmp_path))
        assert d == str(tmp_path / "abilene_calib_pareto" / "alpha_0.1")

    def test_two_decimal(self, tmp_path):
        from scripts.run_pareto_calibration import cell_output_dir
        d = cell_output_dir("abilene", 0.05, base_dir=str(tmp_path))
        assert d.endswith("alpha_0.05")


# ---------------------------------------------------------------------------
# collect_calibration_points
# ---------------------------------------------------------------------------

class TestCollect:
    def test_extracts_only_cqr_aci_rows(self):
        from scripts.run_pareto_calibration import collect_calibration_points
        by_target = {
            0.10: _agg_with_calibration(
                point_models={"LSTM": {}},
                cqr_models={"LSTM": {"coverage": 0.91, "width": 5.0}},
                aci_models={"LSTM": {"coverage": 0.89, "width": 4.7}},
            ),
        }
        points = collect_calibration_points(by_target)
        # LSTM (point) must be excluded; LSTM_CQR + LSTM_ACI included.
        assert set(points.keys()) == {"LSTM_CQR", "LSTM_ACI"}
        assert points["LSTM_CQR"][0]["coverage_overall"] == 0.91
        assert points["LSTM_ACI"][0]["mean_width"] == 4.7

    def test_sorted_by_target_alpha(self):
        from scripts.run_pareto_calibration import collect_calibration_points
        by_target = {
            0.20: _agg_with_calibration(
                cqr_models={"LSTM": {"coverage": 0.79, "width": 4.0,
                                       "target": 0.20}},
            ),
            0.05: _agg_with_calibration(
                cqr_models={"LSTM": {"coverage": 0.94, "width": 6.0,
                                       "target": 0.05}},
            ),
            0.10: _agg_with_calibration(
                cqr_models={"LSTM": {"coverage": 0.89, "width": 5.0,
                                       "target": 0.10}},
            ),
        }
        points = collect_calibration_points(by_target)
        ts = [e["target_alpha"] for e in points["LSTM_CQR"]]
        assert ts == sorted(ts)
        assert ts == [0.05, 0.10, 0.20]

    def test_target_coverage_derived(self):
        from scripts.run_pareto_calibration import collect_calibration_points
        by_target = {
            0.10: _agg_with_calibration(
                cqr_models={"LSTM": {"coverage": 0.9, "width": 5.0}},
            ),
        }
        points = collect_calibration_points(by_target)
        assert points["LSTM_CQR"][0]["target_coverage"] == pytest.approx(0.9)

    def test_handles_multiple_models(self):
        from scripts.run_pareto_calibration import collect_calibration_points
        by_target = {
            0.10: _agg_with_calibration(
                cqr_models={
                    "LSTM": {"coverage": 0.91, "width": 5.0},
                    "PatchTST": {"coverage": 0.92, "width": 4.5},
                },
            ),
        }
        points = collect_calibration_points(by_target)
        assert "LSTM_CQR" in points and "PatchTST_CQR" in points

    def test_empty_calibration_block_dropped(self):
        from scripts.run_pareto_calibration import collect_calibration_points
        agg = {
            "models": {
                "LSTM_CQR": {
                    "operational": {},
                    "forecast": {},
                    # No 'calibration' key → row should be skipped.
                }
            }
        }
        points = collect_calibration_points({0.1: agg})
        assert points == {}


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

class TestPlot:
    def test_writes_png_with_target_lines(self, tmp_path):
        from scripts.run_pareto_calibration import plot_calibration_pareto
        points = {
            "LSTM_CQR": [
                {"target_alpha": 0.05, "target_coverage": 0.95,
                 "coverage_overall": 0.94, "mean_width": 6.0,
                 "qhat_mean": 1.5},
                {"target_alpha": 0.10, "target_coverage": 0.90,
                 "coverage_overall": 0.91, "mean_width": 5.0,
                 "qhat_mean": 1.2},
                {"target_alpha": 0.20, "target_coverage": 0.80,
                 "coverage_overall": 0.79, "mean_width": 4.0,
                 "qhat_mean": 0.9},
            ],
        }
        out = tmp_path / "calib.png"
        plot_calibration_pareto(points, [0.05, 0.1, 0.2], str(out),
                                 dataset="abilene")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_calibration_rows_does_not_crash(self, tmp_path):
        from scripts.run_pareto_calibration import plot_calibration_pareto
        out = tmp_path / "calib_empty.png"
        plot_calibration_pareto({}, [0.05, 0.1], str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# _load_or_run cache mode
# ---------------------------------------------------------------------------

class TestFromCache:
    def test_from_cache_reads_existing(self, tmp_path):
        from scripts.run_pareto_calibration import (
            _load_or_run, cell_output_dir,
        )
        cell_dir = cell_output_dir("abilene", 0.1, base_dir=str(tmp_path))
        os.makedirs(cell_dir, exist_ok=True)
        canned = _agg_with_calibration(
            cqr_models={"LSTM": {"coverage": 0.9, "width": 5.0}}
        )
        with open(os.path.join(cell_dir, "aggregated_results.json"), "w") as f:
            json.dump(canned, f)

        out = _load_or_run("abilene", 0.1, seeds=[42], models=("lstm",),
                          calibration="cqr", alpha=5.0, beta=1.0,
                          from_cache=True, base_dir=str(tmp_path))
        assert out == canned

    def test_no_cache_calls_main_programmatic(self, tmp_path, monkeypatch):
        from scripts.run_pareto_calibration import _load_or_run
        from scripts import run_experiments

        calls = []

        def _fake(*, dataset, loss, alpha, beta, tau, seeds, models,
                  output_dir, calibration=None, target_alpha=None):
            calls.append({
                "dataset": dataset, "loss": loss,
                "alpha": alpha, "beta": beta,
                "calibration": calibration,
                "target_alpha": target_alpha,
                "output_dir": output_dir,
            })
            return _agg_with_calibration(
                cqr_models={"LSTM": {"coverage": 0.9, "width": 5.0}}
            )

        monkeypatch.setattr(run_experiments, "main_programmatic", _fake)
        out = _load_or_run("abilene", 0.05, seeds=[42],
                          models=("lstm",), calibration="both",
                          alpha=5.0, beta=1.0,
                          from_cache=False, base_dir=str(tmp_path))
        assert len(calls) == 1
        assert calls[0]["loss"] == "asym"
        assert calls[0]["calibration"] == "both"
        assert calls[0]["target_alpha"] == 0.05
        assert "alpha_0.05" in calls[0]["output_dir"]
        assert "LSTM_CQR" in out["models"]

    def test_missing_cache_raises(self, tmp_path):
        from scripts.run_pareto_calibration import _load_or_run
        with pytest.raises(FileNotFoundError, match="missing"):
            _load_or_run("abilene", 0.1, seeds=[42], models=("lstm",),
                         calibration="cqr", alpha=5.0, beta=1.0,
                         from_cache=True, base_dir=str(tmp_path))
