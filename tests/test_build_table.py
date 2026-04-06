"""Unit tests for scripts/build_headline_table.py.

Pure data-mungeing — no plots, no I/O on the system. Verifies the
mean±std formatting, bold-best policy, calibration-row em-dash
behaviour, two-input join, and that the LaTeX shape parses to the
right column count.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


def _agg(models: dict[str, dict]) -> dict:
    """Build a runner-shaped aggregated dict.

    Each models[name] entry can carry any subset of:
      operational: {asymmetric_op_cost, overload_rate, over_provisioning_cost,
                     u_max_mean}
      forecast:    {rmse_mean, mae_mean}
      calibration: {coverage_overall, mean_width}
    each as a {mean, std} dict.
    """
    return {"models": models}


def _stat(mean: float, std: float = 0.0) -> dict:
    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# collect_rows + column_visible
# ---------------------------------------------------------------------------

class TestCollectRows:
    def test_basic(self):
        from scripts.build_headline_table import collect_rows
        agg = _agg({
            "LSTM": {
                "operational": {"asymmetric_op_cost": _stat(50.0, 1.0)},
                "forecast": {"rmse_mean": _stat(1.5, 0.05)},
            }
        })
        rows = collect_rows([agg])
        assert len(rows) == 1
        assert rows[0]["label"] == "LSTM"
        assert rows[0]["stats"]["RMSE"] == _stat(1.5, 0.05)
        assert rows[0]["stats"]["Asym. op. cost"] == _stat(50.0, 1.0)
        assert rows[0]["stats"]["Coverage"] is None  # no calibration block

    def test_multi_input_with_names(self):
        from scripts.build_headline_table import collect_rows
        a = _agg({"LSTM": {"forecast": {"rmse_mean": _stat(1.5)}}})
        b = _agg({"LSTM": {"forecast": {"rmse_mean": _stat(2.5)}}})
        rows = collect_rows([a, b], names=["abilene", "geant"])
        labels = [r["label"] for r in rows]
        assert labels == ["abilene/LSTM", "geant/LSTM"]

    def test_calibration_row_flagged(self):
        from scripts.build_headline_table import collect_rows
        agg = _agg({
            "LSTM_CQR": {
                "operational": {"overload_rate": _stat(0.05, 0.01)},
                "calibration": {"coverage_overall": _stat(0.92, 0.01)},
            }
        })
        rows = collect_rows([agg])
        assert rows[0]["is_calibration"] is True


class TestColumnVisible:
    def test_visible_when_any_row_has_value(self):
        from scripts.build_headline_table import collect_rows, column_visible
        rows = collect_rows([_agg({
            "LSTM": {"forecast": {"rmse_mean": _stat(1.0)}},
            "DCRNN": {"forecast": {}},
        })])
        assert column_visible("RMSE", rows)

    def test_hidden_when_no_row_has_value(self):
        from scripts.build_headline_table import collect_rows, column_visible
        rows = collect_rows([_agg({
            "LSTM": {"forecast": {"rmse_mean": _stat(1.0)}},
        })])
        # Coverage is in calibration; no row has it → hidden.
        assert not column_visible("Coverage", rows)


# ---------------------------------------------------------------------------
# bold_best_indices
# ---------------------------------------------------------------------------

class TestBoldBest:
    def test_lower_is_better_for_op_cost(self):
        from scripts.build_headline_table import (
            collect_rows, bold_best_indices,
        )
        rows = collect_rows([_agg({
            "LSTM": {"operational": {"asymmetric_op_cost": _stat(50.0)}},
            "SARIMA": {"operational": {"asymmetric_op_cost": _stat(80.0)}},
            "DLinear": {"operational": {"asymmetric_op_cost": _stat(45.0)}},
        })])
        best = bold_best_indices(rows)
        # Best (lowest) is DLinear at index 2.
        assert best["Asym. op. cost"] == {2}

    def test_higher_is_better_for_coverage(self):
        from scripts.build_headline_table import (
            collect_rows, bold_best_indices,
        )
        rows = collect_rows([_agg({
            "LSTM_CQR": {"calibration": {"coverage_overall": _stat(0.91)}},
            "PatchTST_CQR": {"calibration": {"coverage_overall": _stat(0.94)}},
        })])
        best = bold_best_indices(rows)
        # Higher coverage is better → index 1 (PatchTST_CQR).
        assert best["Coverage"] == {1}

    def test_ties_both_marked(self):
        from scripts.build_headline_table import (
            collect_rows, bold_best_indices,
        )
        rows = collect_rows([_agg({
            "A": {"operational": {"asymmetric_op_cost": _stat(50.0)}},
            "B": {"operational": {"asymmetric_op_cost": _stat(50.0)}},
            "C": {"operational": {"asymmetric_op_cost": _stat(60.0)}},
        })])
        best = bold_best_indices(rows)
        assert best["Asym. op. cost"] == {0, 1}


# ---------------------------------------------------------------------------
# render_latex
# ---------------------------------------------------------------------------

class TestRenderLatex:
    def test_column_count_matches_header(self):
        from scripts.build_headline_table import collect_rows, render_latex
        rows = collect_rows([_agg({
            "LSTM": {
                "operational": {
                    "asymmetric_op_cost": _stat(50.0, 1.0),
                    "overload_rate": _stat(0.10, 0.01),
                },
                "forecast": {"rmse_mean": _stat(1.5, 0.05)},
            },
        })])
        latex = render_latex(rows)
        # Three visible columns + the "Model" column = 4 cells per body line.
        body_lines = [
            ln for ln in latex.splitlines()
            if r"\\" in ln and not ln.strip().startswith("Model")
        ]
        for ln in body_lines:
            n = ln.count("&")
            assert n == 3, f"expected 3 ampersands in body row, got {n}: {ln}"

    def test_calibration_row_dash_for_forecast(self):
        from scripts.build_headline_table import collect_rows, render_latex
        rows = collect_rows([_agg({
            "LSTM": {"forecast": {"rmse_mean": _stat(1.5, 0.05)}},
            "LSTM_CQR": {
                "calibration": {
                    "coverage_overall": _stat(0.91, 0.01),
                    "mean_width": _stat(5.0, 0.2),
                },
                "operational": {"overload_rate": _stat(0.05, 0.01)},
            },
        })])
        latex = render_latex(rows)
        # The LSTM_CQR row should have an em-dash in the RMSE column.
        cqr_lines = [ln for ln in latex.splitlines() if "LSTM\\_CQR" in ln]
        assert len(cqr_lines) == 1
        assert "--" in cqr_lines[0]  # at least one missing column rendered

    def test_bold_best_wraps_in_textbf(self):
        from scripts.build_headline_table import collect_rows, render_latex
        rows = collect_rows([_agg({
            "A": {"operational": {"asymmetric_op_cost": _stat(50.0, 1.0)}},
            "B": {"operational": {"asymmetric_op_cost": _stat(60.0, 1.0)}},
        })])
        latex = render_latex(rows, bold_best=True)
        assert r"\textbf{50.00" in latex
        assert r"\textbf{60.00" not in latex  # B is not best, not bolded

    def test_no_bold_best_disables(self):
        from scripts.build_headline_table import collect_rows, render_latex
        rows = collect_rows([_agg({
            "A": {"operational": {"asymmetric_op_cost": _stat(50.0)}},
            "B": {"operational": {"asymmetric_op_cost": _stat(60.0)}},
        })])
        latex = render_latex(rows, bold_best=False)
        assert r"\textbf" not in latex

    def test_caption_and_label(self):
        from scripts.build_headline_table import collect_rows, render_latex
        rows = collect_rows([_agg({
            "A": {"forecast": {"rmse_mean": _stat(1.0)}},
        })])
        latex = render_latex(rows, caption="Test", label="tab:t")
        assert r"\caption{Test}" in latex
        assert r"\label{tab:t}" in latex

    def test_percent_metric_formatting(self):
        from scripts.build_headline_table import collect_rows, render_latex
        rows = collect_rows([_agg({
            "A": {"operational": {"overload_rate": _stat(0.123, 0.01)}},
        })])
        latex = render_latex(rows)
        # Overload rate is a percentage column → mean·100 displayed.
        assert "12.30" in latex


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    def test_pipe_table(self):
        from scripts.build_headline_table import (
            collect_rows, render_markdown,
        )
        rows = collect_rows([_agg({
            "LSTM": {
                "operational": {"asymmetric_op_cost": _stat(50.0, 1.0)},
                "forecast": {"rmse_mean": _stat(1.5, 0.05)},
            },
        })])
        md = render_markdown(rows)
        assert md.startswith("| Model |")
        assert "± 1.00" in md or "± 1.0" in md
        # First line is header, second is the |---| separator.
        assert md.splitlines()[1].startswith("|---|")

    def test_bold_best_uses_double_star(self):
        from scripts.build_headline_table import (
            collect_rows, render_markdown,
        )
        rows = collect_rows([_agg({
            "A": {"operational": {"asymmetric_op_cost": _stat(50.0)}},
            "B": {"operational": {"asymmetric_op_cost": _stat(60.0)}},
        })])
        md = render_markdown(rows, bold_best=True)
        assert "**50.00" in md
