"""Backfill LSTM rows into aggregated_results.json for each dataset.

The multi-architecture sweep wrote lstm_predictions.npz per seed but the
runner only logged DLinear + iTransformer in the per-seed results.json.
This script reads each seed's lstm_predictions.npz, recomputes forecast
and operational metrics with the exact same code paths the runner uses
(src.utils.compute_metrics/aggregate_metrics + src.evaluation.operational_metrics
with margin=1.1, alpha=5, beta=1), and writes a separate
aggregated_results_lstm.json that build_headline_table.py can consume.

Why a separate file: we keep the historical aggregated_results.json
untouched (no surprise edits to prior commits' artifact); the table
builder accepts multiple --inputs and merges by model name.
"""
import json
from pathlib import Path
import numpy as np

from src.utils import compute_metrics, aggregate_metrics
from src.evaluation import operational_metrics
from src.config import CONFIG

OP_KEYS = (
    "overload_rate", "sla_violation_rate", "over_provisioning_cost",
    "asymmetric_op_cost", "u_max_mean", "u_max_max",
)
FC_KEYS = ("rmse_mean", "mae_mean", "mape_mean", "smape_mean")

MARGIN = float(CONFIG.get("capacity_margin", 1.1))
ALPHA = 5.0
BETA = 1.0


def _summarize(values):
    """Match the {mean, std, min, max, values} shape used by run_experiments.py."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(v) for v in arr],
    }


def reaggregate(dataset: str) -> None:
    base = Path("results") / f"{dataset}_pareto_asym" / "ratio_5_1"
    seed_dirs = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("seed_"))
    forecast_vals = {k: [] for k in FC_KEYS}
    op_vals = {k: [] for k in OP_KEYS}
    seeds_used = []
    for sd in seed_dirs:
        npz = sd / "lstm_predictions.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        y_true = z["L_test_aligned"]; y_pred = z["predictions"]
        fc = aggregate_metrics(compute_metrics(y_true, y_pred))
        cap = MARGIN * np.nanmax(y_pred, axis=0)
        op = operational_metrics(y_true, cap, alpha=ALPHA, beta=BETA)
        for k in FC_KEYS:
            forecast_vals[k].append(fc.get(k))
        for k in OP_KEYS:
            op_vals[k].append(op[k])
        seeds_used.append(int(sd.name.split("_")[-1]))

    out = {
        "num_seeds": len(seeds_used),
        "seeds": seeds_used,
        "dataset": dataset,
        "loss": "asym",
        "per_seed": {},
        "models": {
            "LSTM": {
                "forecast": {k: _summarize(forecast_vals[k]) for k in FC_KEYS},
                "operational": {k: _summarize(op_vals[k]) for k in OP_KEYS},
            }
        },
    }
    out_path = base / "aggregated_results_lstm.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[{dataset}] wrote {out_path}: n={len(seeds_used)}, "
          f"RMSE={out['models']['LSTM']['forecast']['rmse_mean']['mean']:.3f}, "
          f"asym_op_cost={out['models']['LSTM']['operational']['asymmetric_op_cost']['mean']:.3e}")


if __name__ == "__main__":
    for ds in ("abilene", "geant", "cesnet"):
        reaggregate(ds)
