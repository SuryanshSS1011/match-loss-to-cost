"""Re-aggregate per-seed prediction npz files into a 20-seed aggregated_results.json.

Use case: when seeds were trained via multiple `run_experiments.py`
invocations (one per seed), each invocation overwrites aggregated_results.json
with a single-seed summary. This script reads all seed_*/<model>_predictions.npz
files in a directory and emits a proper multi-seed aggregated_results.json
matching the schema build_headline_table.py expects.

Uses the same code paths as the runner (`src.utils.compute_metrics` +
`src.evaluation.operational_metrics`, margin from CONFIG, default α=5,
β=1). Schema matches what the runner writes for the
{mean,std,min,max,values} summary blocks.

Generalization of scripts/backfill_lstm_rows.py — that one is dataset-
hardcoded, this one takes (dir, model_name) as args.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/aggregate_from_seed_npz.py \\
        --dir results/abilene_patchtst_cpu \\
        --model patchtst \\
        --display-name PatchTST \\
        --dataset abilene
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import CONFIG
from src.evaluation import operational_metrics
from src.utils import aggregate_metrics, compute_metrics

OP_KEYS = (
    "overload_rate", "sla_violation_rate", "over_provisioning_cost",
    "asymmetric_op_cost", "u_max_mean", "u_max_max",
)
FC_KEYS = ("rmse_mean", "mae_mean", "mape_mean", "smape_mean")


def _summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(v) for v in arr],
    }


def reaggregate(
    base_dir: Path,
    model_filename: str,
    display_name: str,
    dataset: str,
    alpha: float,
    beta: float,
    margin: float,
    output_path: Path,
) -> None:
    seed_dirs = sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("seed_")
    )

    forecast_vals = {k: [] for k in FC_KEYS}
    op_vals = {k: [] for k in OP_KEYS}
    seeds_used = []

    for sd in seed_dirs:
        npz = sd / f"{model_filename}_predictions.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        y_true = z["L_test_aligned"]
        y_pred = z["predictions"]
        fc = aggregate_metrics(compute_metrics(y_true, y_pred))
        cap = margin * np.nanmax(y_pred, axis=0)
        op = operational_metrics(y_true, cap, alpha=alpha, beta=beta)
        for k in FC_KEYS:
            forecast_vals[k].append(fc.get(k))
        for k in OP_KEYS:
            op_vals[k].append(op[k])
        seeds_used.append(int(sd.name.split("_")[-1]))

    if not seeds_used:
        raise SystemExit(
            f"No {model_filename}_predictions.npz files found under {base_dir}"
        )

    out = {
        "num_seeds": len(seeds_used),
        "seeds": seeds_used,
        "dataset": dataset,
        "loss": "asym",
        "per_seed": {},
        "models": {
            display_name: {
                "forecast": {k: _summarize(forecast_vals[k]) for k in FC_KEYS},
                "operational": {k: _summarize(op_vals[k]) for k in OP_KEYS},
            }
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(output_path, "w"), indent=2)
    print(
        f"[aggregate] wrote {output_path}: n={len(seeds_used)}, "
        f"RMSE={out['models'][display_name]['forecast']['rmse_mean']['mean']:.3f}, "
        f"asym_op_cost={out['models'][display_name]['operational']['asymmetric_op_cost']['mean']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="directory containing seed_*/ subdirs")
    parser.add_argument("--model", required=True,
                        help="model filename prefix (e.g. patchtst → reads "
                             "<model>_predictions.npz)")
    parser.add_argument("--display-name", required=True,
                        help="model display name for the aggregated JSON "
                             "(e.g. PatchTST)")
    parser.add_argument("--dataset", required=True,
                        choices=("abilene", "geant", "cesnet", "synthetic"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--margin", type=float,
                        default=float(CONFIG.get("capacity_margin", 1.1)))
    parser.add_argument("--output", default=None,
                        help="output path (default: <dir>/aggregated_results.json)")
    args = parser.parse_args()

    base = Path(args.dir)
    if not base.is_dir():
        raise SystemExit(f"--dir does not exist: {base}")
    output = Path(args.output) if args.output else base / "aggregated_results.json"

    reaggregate(
        base_dir=base,
        model_filename=args.model,
        display_name=args.display_name,
        dataset=args.dataset,
        alpha=args.alpha,
        beta=args.beta,
        margin=args.margin,
        output_path=output,
    )


if __name__ == "__main__":
    main()
