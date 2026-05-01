#!/usr/bin/env python
"""V0 gate runner — produces plots/v0_rmse_vs_opcost.png on real Abilene.

Decision criterion (CLAUDE.md, plan.md, deadline 2026-05-12):
    asymmetric-loss LSTM achieves <= MSE-LSTM RMSE while delivering lower
    operational cost on real Abilene over 5 seeds. Pass = continue, fail = pivot.

This script does *no* training itself. It:
  1. (default) invokes `scripts.run_experiments.main` once per loss in
     {mse, asym, pinball}, sharing the same seed list; OR
  2. (`--from-cache`) skips the runs and just reads the aggregated JSONs that
     a previous invocation wrote to `results/abilene_<loss>/aggregated_results.json`.

Then it plots one point per (loss, seed) on the (RMSE, asymmetric_op_cost)
plane and saves to `plots/v0_rmse_vs_opcost.png`.

Intended usage on the cloud box:
    python scripts/run_v0.py --seeds 42 123 456 789 1024
On the laptop (after cloud has populated results/):
    python scripts/run_v0.py --from-cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import PLOTS_DIR, RESULTS_DIR  # noqa: E402


V0_LOSSES = ("mse", "asym", "pinball")
V0_DATASET = "abilene"


def _aggregated_path(dataset: str, loss: str) -> str:
    return os.path.join(RESULTS_DIR, f"{dataset}_{loss}",
                        "aggregated_results.json")


def _load_or_run(loss: str, seeds: list[int],
                 alpha: float, beta: float, tau: Optional[float],
                 from_cache: bool,
                 models: Optional[tuple] = None) -> dict:
    cache = _aggregated_path(V0_DATASET, loss)
    if from_cache:
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"--from-cache set but {cache} missing; run without it first"
            )
        with open(cache) as f:
            return json.load(f)

    # Defer runner import until needed; importing it triggers heavy modules.
    from scripts import run_experiments
    kwargs = dict(
        dataset=V0_DATASET, loss=loss,
        alpha=alpha, beta=beta, tau=tau,
        seeds=seeds,
        output_dir=os.path.join(RESULTS_DIR, f"{V0_DATASET}_{loss}"),
    )
    if models is not None:
        kwargs["models"] = models
    return run_experiments.main_programmatic(**kwargs)


def _extract_points(agg: dict, model: str) -> tuple[list[float], list[float]]:
    """Return (rmse_values, op_cost_values) per seed for the given model."""
    block = agg.get("models", {}).get(model)
    if block is None:
        return [], []
    rmse = block.get("forecast", {}).get("rmse_mean", {}).get("values", [])
    opcost = (block.get("operational", {})
                   .get("asymmetric_op_cost", {})
                   .get("values", []))
    return list(rmse), list(opcost)


def plot_v0(by_loss: dict[str, dict], save_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(7.5, 5.5))

    palette = {"mse": "tab:red", "asym": "tab:green", "pinball": "tab:blue"}
    markers = {"LSTM": "o", "SARIMA": "s"}

    for loss, agg in by_loss.items():
        for model in ("LSTM", "SARIMA"):
            xs, ys = _extract_points(agg, model)
            if not xs or not ys:
                continue
            plt.scatter(
                xs, ys,
                color=palette.get(loss, "gray"),
                marker=markers.get(model, "x"),
                edgecolor="black", linewidth=0.5,
                alpha=0.85,
                label=f"{model} / {loss}",
            )

    plt.xlabel("RMSE (mean over links)")
    plt.ylabel("Asymmetric operational cost")
    plt.title("V0 gate: RMSE vs operational cost on real Abilene")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[v0] wrote {save_path}")


def evaluate_gate(by_loss: dict[str, dict]) -> dict:
    """Decide pass / fail against the CLAUDE.md criterion.

    Pass criteria (LSTM only, since the headline question is loss-vs-loss
    on the same backbone):
      1. RMSE_asym mean <= RMSE_mse mean + 1*std_mse  (allow 1σ slack).
      2. OpCost_asym mean < OpCost_mse mean.

    Returns a dict with the booleans + raw stats. We do NOT run a Wilcoxon
    test here — that lives in src/evaluation/significance.py (week 5–6).
    """
    def stats(loss: str, model: str = "LSTM") -> dict:
        agg = by_loss.get(loss) or {}
        block = agg.get("models", {}).get(model, {})
        rmse = block.get("forecast", {}).get("rmse_mean", {})
        opc = block.get("operational", {}).get("asymmetric_op_cost", {})
        return {
            "rmse_mean": rmse.get("mean"),
            "rmse_std": rmse.get("std"),
            "opcost_mean": opc.get("mean"),
            "opcost_std": opc.get("std"),
        }

    s_mse = stats("mse")
    s_asym = stats("asym")

    pass_rmse = (
        s_mse["rmse_mean"] is not None and s_asym["rmse_mean"] is not None
        and s_asym["rmse_mean"] <= s_mse["rmse_mean"] + (s_mse.get("rmse_std") or 0.0)
    )
    pass_opcost = (
        s_mse["opcost_mean"] is not None and s_asym["opcost_mean"] is not None
        and s_asym["opcost_mean"] < s_mse["opcost_mean"]
    )
    return {
        "pass_rmse_tie_or_better": bool(pass_rmse),
        "pass_opcost_strictly_lower": bool(pass_opcost),
        "v0_pass": bool(pass_rmse and pass_opcost),
        "stats": {"mse": s_mse, "asym": s_asym},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v0 gate runner")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--from-cache", action="store_true",
                        help="skip training; read aggregated JSONs only")
    parser.add_argument("--plot-path", default=None,
                        help="default: plots/v0_rmse_vs_opcost.png")
    parser.add_argument("--skip-sarima", action="store_true",
                        help="skip SARIMA, run LSTM only (memory-constrained envs)")
    args = parser.parse_args()

    models = ("lstm",) if args.skip_sarima else None

    by_loss = {}
    for loss in V0_LOSSES:
        print(f"[v0] loss={loss}  from_cache={args.from_cache}  "
              f"models={'lstm only' if args.skip_sarima else 'lstm+sarima'}")
        by_loss[loss] = _load_or_run(
            loss, args.seeds,
            alpha=args.alpha, beta=args.beta, tau=args.tau,
            from_cache=args.from_cache,
            models=models,
        )

    plot_path = args.plot_path or os.path.join(PLOTS_DIR, "v0_rmse_vs_opcost.png")
    plot_v0(by_loss, plot_path)

    gate = evaluate_gate(by_loss)
    print("\n[v0 gate]")
    for k, v in gate.items():
        if k == "stats":
            continue
        print(f"  {k}: {v}")
    print(f"  stats: {json.dumps(gate['stats'], indent=2)}")

    out = {"by_loss": by_loss, "gate": gate}
    out_path = os.path.join(RESULTS_DIR, "v0_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[v0] wrote {out_path}")


if __name__ == "__main__":
    main()
