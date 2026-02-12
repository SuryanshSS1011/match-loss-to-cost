"""Logic tests for scripts/run_v0.py.

No real training. Just feeds canned aggregated-result dicts into
`evaluate_gate` and `plot_v0` and asserts the gate pass/fail and the
plot artefact land where expected.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _agg(rmse_mean: float, rmse_std: float,
         opc_mean: float, opc_std: float,
         model: str = "LSTM") -> dict:
    return {
        "models": {
            model: {
                "forecast": {
                    "rmse_mean": {
                        "mean": rmse_mean, "std": rmse_std,
                        "values": [rmse_mean - rmse_std, rmse_mean + rmse_std],
                    },
                },
                "operational": {
                    "asymmetric_op_cost": {
                        "mean": opc_mean, "std": opc_std,
                        "values": [opc_mean - opc_std, opc_mean + opc_std],
                    },
                },
            }
        }
    }


def test_gate_passes_when_asym_ties_rmse_and_beats_opcost():
    from scripts.run_v0 import evaluate_gate

    by_loss = {
        "mse":  _agg(rmse_mean=10.0, rmse_std=0.5, opc_mean=100.0, opc_std=5.0),
        "asym": _agg(rmse_mean=10.2, rmse_std=0.5, opc_mean=70.0,  opc_std=4.0),
        "pinball": _agg(rmse_mean=10.5, rmse_std=0.5, opc_mean=80.0, opc_std=4.0),
    }
    gate = evaluate_gate(by_loss)
    assert gate["pass_rmse_tie_or_better"]   # asym RMSE within mse_std of mse
    assert gate["pass_opcost_strictly_lower"]
    assert gate["v0_pass"]


def test_gate_fails_when_asym_blows_up_rmse():
    from scripts.run_v0 import evaluate_gate

    by_loss = {
        "mse":  _agg(rmse_mean=10.0, rmse_std=0.5, opc_mean=100.0, opc_std=5.0),
        "asym": _agg(rmse_mean=15.0, rmse_std=0.5, opc_mean=70.0,  opc_std=4.0),
        "pinball": _agg(rmse_mean=12.0, rmse_std=0.5, opc_mean=80.0, opc_std=4.0),
    }
    gate = evaluate_gate(by_loss)
    assert not gate["pass_rmse_tie_or_better"]
    assert gate["pass_opcost_strictly_lower"]
    assert not gate["v0_pass"]


def test_gate_fails_when_opcost_does_not_drop():
    from scripts.run_v0 import evaluate_gate

    by_loss = {
        "mse":  _agg(rmse_mean=10.0, rmse_std=0.5, opc_mean=100.0, opc_std=5.0),
        "asym": _agg(rmse_mean=10.0, rmse_std=0.5, opc_mean=110.0, opc_std=4.0),
        "pinball": _agg(rmse_mean=10.0, rmse_std=0.5, opc_mean=120.0, opc_std=4.0),
    }
    gate = evaluate_gate(by_loss)
    assert gate["pass_rmse_tie_or_better"]
    assert not gate["pass_opcost_strictly_lower"]
    assert not gate["v0_pass"]


def test_gate_handles_missing_loss(tmp_path):
    from scripts.run_v0 import evaluate_gate

    # Missing 'asym' entry entirely.
    by_loss = {
        "mse": _agg(rmse_mean=10.0, rmse_std=0.5, opc_mean=100.0, opc_std=5.0),
    }
    gate = evaluate_gate(by_loss)
    assert not gate["v0_pass"]
    assert gate["stats"]["asym"]["rmse_mean"] is None


def test_plot_writes_png(tmp_path):
    from scripts.run_v0 import plot_v0

    by_loss = {
        "mse":  _agg(rmse_mean=10.0, rmse_std=0.5, opc_mean=100.0, opc_std=5.0),
        "asym": _agg(rmse_mean=10.2, rmse_std=0.5, opc_mean=70.0,  opc_std=4.0),
    }
    out = tmp_path / "v0.png"
    plot_v0(by_loss, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
