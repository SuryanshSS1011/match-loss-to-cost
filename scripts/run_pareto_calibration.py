#!/usr/bin/env python
"""Calibration Pareto: empirical coverage vs mean interval width.

Sibling to scripts/run_pareto.py. The headline Pareto (run_pareto.py)
sweeps α/β and plots over-provisioning cost vs overload rate for
*point-forecast* models. This one sweeps the target miscoverage rate
target_alpha ∈ {0.05, 0.10, 0.20} (per CLAUDE.md headline metric list)
and plots empirical coverage vs mean band width for the *calibrated*
models — `LSTM_CQR`, `LSTM_ACI`, `PatchTST_CQR`, etc.

Per cell: one runner invocation with `--calibration both --target-alpha
v`, sharing the asym (α=5, β=1) point-loss config. Each cell emits one
or two calibration rows per neural model. The plot shows one line per
(model, method) — connecting the three target_alpha points — and one
horizontal "target coverage = 1 - target_alpha" line per swept value.

Modes mirror run_pareto.py:
  default      — train all cells from scratch.
  --from-cache — read cached aggregated JSONs and replot only.

Per-cell artefacts land at:
    results/<dataset>_calib_pareto/alpha_<v>/aggregated_results.json
The summary plot is written to plots/pareto_calibration_<dataset>.png
and a machine-readable summary at
results/<dataset>_calib_pareto/summary.json.
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


DEFAULT_TARGETS = (0.05, 0.10, 0.20)


def cell_output_dir(dataset: str, target_alpha: float,
                    base_dir: str = RESULTS_DIR) -> str:
    return os.path.join(
        base_dir, f"{dataset}_calib_pareto",
        f"alpha_{target_alpha:g}",
    )


def _aggregated_path(dataset: str, target_alpha: float,
                     base_dir: str = RESULTS_DIR) -> str:
    return os.path.join(cell_output_dir(dataset, target_alpha, base_dir),
                        "aggregated_results.json")


def _load_or_run(dataset: str, target_alpha: float, seeds: list[int],
                 models: tuple, calibration: str, alpha: float, beta: float,
                 from_cache: bool, base_dir: str = RESULTS_DIR) -> dict:
    cache = _aggregated_path(dataset, target_alpha, base_dir)
    if from_cache:
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"--from-cache: missing {cache}; run without it first"
            )
        with open(cache) as f:
            return json.load(f)

    from scripts import run_experiments
    return run_experiments.main_programmatic(
        dataset=dataset, loss="asym",
        alpha=alpha, beta=beta, tau=None,
        seeds=seeds,
        models=models,
        output_dir=cell_output_dir(dataset, target_alpha, base_dir),
        calibration=calibration,
        target_alpha=target_alpha,
    )


def collect_calibration_points(
    by_target: dict[float, dict],
) -> dict[str, list[dict]]:
    """Extract calibration entries per (model, method) from each cell.

    Only model rows whose name ends with '_CQR' or '_ACI' carry a
    `calibration` sub-block; we ignore point-forecast rows.

    Returns:
        {row_name: [{target_alpha, coverage_overall, mean_width,
                     qhat_mean}, ...]} sorted by target_alpha.
    """
    points: dict[str, list[dict]] = {}
    for target_alpha, agg in by_target.items():
        for model_row, block in agg.get("models", {}).items():
            if not (model_row.endswith("_CQR") or model_row.endswith("_ACI")):
                continue
            cal = block.get("calibration", {})
            if not cal:
                continue
            entry = {
                "target_alpha": float(target_alpha),
                "target_coverage": float(1.0 - target_alpha),
                "coverage_overall": cal.get("coverage_overall", {}).get("mean"),
                "mean_width": cal.get("mean_width", {}).get("mean"),
                "qhat_mean": cal.get("qhat_mean", {}).get("mean"),
            }
            points.setdefault(model_row, []).append(entry)
    for row in points:
        points[row].sort(key=lambda e: e["target_alpha"])
    return points


def plot_calibration_pareto(
    points: dict[str, list[dict]],
    target_alphas: list[float],
    save_path: str,
    dataset: str = "",
) -> None:
    """One line per (model, method); horizontal line per target coverage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    method_marker = {"CQR": "o", "ACI": "s"}
    base_palette = {
        "LSTM": "tab:blue", "DLinear": "tab:green",
        "PatchTST": "tab:purple", "iTransformer": "tab:red",
        "DCRNN": "tab:orange",
    }

    for row_name, entries in points.items():
        # row_name is e.g. "LSTM_CQR" or "iTransformer_ACI".
        if "_" not in row_name:
            continue
        model_base, method = row_name.rsplit("_", 1)
        xs = [e["mean_width"] for e in entries
              if e["mean_width"] is not None
              and e["coverage_overall"] is not None]
        ys = [e["coverage_overall"] for e in entries
              if e["mean_width"] is not None
              and e["coverage_overall"] is not None]
        if not xs:
            continue
        ax.plot(
            xs, ys,
            "-" + method_marker.get(method, "x"),
            color=base_palette.get(model_base, "gray"),
            label=row_name, alpha=0.9,
            linewidth=1.5, markersize=7,
        )
        # Annotate each point with its target_alpha.
        for x, y, e in zip(xs, ys, entries):
            ax.annotate(f" α={e['target_alpha']:g}", (x, y),
                         fontsize=7, alpha=0.6)

    # Horizontal target-coverage lines.
    for ta in sorted(set(target_alphas)):
        ax.axhline(1.0 - ta, color="black", linestyle="--",
                   linewidth=0.8, alpha=0.4)
        ax.text(ax.get_xlim()[1], 1.0 - ta + 0.005,
                f" target {1 - ta:.2f}", fontsize=7, alpha=0.5,
                ha="right", va="bottom")

    ax.set_xlabel("Mean interval width")
    ax.set_ylabel("Empirical coverage")
    title = "Calibration Pareto: coverage vs interval width"
    if dataset:
        title += f" ({dataset})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[calib-pareto] wrote {save_path}")


def write_summary(by_target: dict[float, dict],
                  points: dict[str, list[dict]],
                  out_path: str) -> None:
    out = {
        "target_alphas": [float(t) for t in by_target.keys()],
        "rows": list(points.keys()),
        "per_row": points,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calib-pareto] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration Pareto sweep")
    parser.add_argument("--dataset", default="abilene",
                        choices=("synthetic", "abilene", "geant", "cesnet"))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--models", nargs="+",
                        default=["lstm"],
                        help="neural models to include "
                             "(non-neural names are accepted but contribute "
                             "no calibration rows)")
    parser.add_argument("--target-alphas", type=float, nargs="+",
                        default=list(DEFAULT_TARGETS),
                        help=f"target miscoverage values "
                             f"(default: {list(DEFAULT_TARGETS)})")
    parser.add_argument("--calibration", default="both",
                        choices=("cqr", "aci", "both"))
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="asymmetric loss alpha for the underlying "
                             "neural training (default 5.0)")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--plot-path", default=None,
                        help="default: plots/pareto_calibration_<dataset>.png")
    args = parser.parse_args()

    for ta in args.target_alphas:
        if not (0.0 < ta < 1.0):
            raise SystemExit(f"target_alpha must lie in (0, 1); got {ta}")

    by_target: dict[float, dict] = {}
    for ta in args.target_alphas:
        print(f"[calib-pareto] dataset={args.dataset} target_alpha={ta} "
              f"calibration={args.calibration} from_cache={args.from_cache}")
        by_target[ta] = _load_or_run(
            args.dataset, ta, args.seeds,
            tuple(args.models), args.calibration,
            args.alpha, args.beta,
            args.from_cache,
        )

    points = collect_calibration_points(by_target)

    plot_path = args.plot_path or os.path.join(
        PLOTS_DIR, f"pareto_calibration_{args.dataset}.png"
    )
    plot_calibration_pareto(points, args.target_alphas, plot_path,
                             dataset=args.dataset)

    summary_path = os.path.join(
        RESULTS_DIR, f"{args.dataset}_calib_pareto", "summary.json"
    )
    write_summary(by_target, points, summary_path)


if __name__ == "__main__":
    main()
