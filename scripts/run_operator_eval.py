#!/usr/bin/env python
"""Fixed-operator post-hoc evaluator for Pareto-sweep results.

Re-evaluates each (training α/β) cell at fixed operator α/β cost
structures, producing the 2D heatmap that maps the paper's actual claim:
"at any realistic operator α/β, there exists a training α/β that beats
MSE on operator-relevant cost."

This script does NOT retrain. It loads each cell's persisted predictions,
recomputes operational cost at every operator α/β in a grid, and emits a
JSON + heatmap PNG.

Inputs: results/<dataset>_pareto/ratio_<a>_<b>/seed_*/lstm_predictions.npz
        results/<dataset>_pareto/baseline_mse/seed_*/lstm_predictions.npz
Output: results/<dataset>_pareto/operator_eval.json
        plots/operator_eval_<dataset>.png

Usage:
    python scripts/run_operator_eval.py --dataset abilene
    python scripts/run_operator_eval.py --dataset abilene \\
        --operator-ratios 1:1 2:1 5:1 10:1 20:1 100:1 \\
        --pareto-base-dir results/abilene_pareto

The cell axis order is read from each cell directory's name; operator
ratios default to the same set as training ratios (square heatmap).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import PLOTS_DIR, RESULTS_DIR  # noqa: E402
from src.evaluation.operational import (  # noqa: E402
    asymmetric_op_cost,
    capacity_from_predictions,
)


RATIO_DIR_RE = re.compile(r"^ratio_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$")


def parse_ratio(s: str) -> tuple[float, float]:
    if ":" not in s:
        raise ValueError(f"ratio {s!r} must be 'a:b'")
    a, b = s.split(":")
    return float(a), float(b)


def discover_cells(pareto_dir: str) -> list[tuple[str, float, float, str]]:
    """Walk the pareto base dir and return [(name, alpha, beta, path), ...].

    Each entry is one cell: ratio_a_b dirs from the sweep plus the
    baseline_mse dir if present. Sorted by (alpha, beta) for stable
    heatmap row order.
    """
    cells: list[tuple[str, float, float, str]] = []
    if not os.path.isdir(pareto_dir):
        raise FileNotFoundError(f"pareto dir not found: {pareto_dir}")
    for name in sorted(os.listdir(pareto_dir)):
        full = os.path.join(pareto_dir, name)
        if not os.path.isdir(full):
            continue
        m = RATIO_DIR_RE.match(name)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            label = f"asym {a:g}:{b:g}"
            cells.append((label, a, b, full))
            continue
        if name == "baseline_mse":
            cells.append(("MSE", 1.0, 1.0, full))
    cells.sort(key=lambda c: (c[1] / c[2], c[1]))
    return cells


def load_cell_predictions(cell_dir: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(predictions, y_true), ...] one entry per seed.

    Reads each seed_*/lstm_predictions.npz; ignores cells without seed
    subdirs. Cells with missing/corrupt npz are silently skipped to keep
    the heatmap robust to partial sweeps.
    """
    seeds = []
    for entry in sorted(os.listdir(cell_dir)):
        if not entry.startswith("seed_"):
            continue
        npz_path = os.path.join(cell_dir, entry, "lstm_predictions.npz")
        if not os.path.exists(npz_path):
            continue
        try:
            data = np.load(npz_path)
            preds = data["predictions"]
            y_true = data["L_test_aligned"]
        except (KeyError, OSError):
            continue
        if preds.shape != y_true.shape:
            continue
        seeds.append((preds, y_true))
    return seeds


def evaluate_cell(seeds: list[tuple[np.ndarray, np.ndarray]],
                  operator_grid: list[tuple[float, float]],
                  margin: float = 1.1) -> dict:
    """For one cell, compute mean asym_op_cost across seeds at each operator α/β."""
    if not seeds:
        return {f"{a}:{b}": None for a, b in operator_grid}
    out: dict[str, float] = {}
    for op_a, op_b in operator_grid:
        per_seed = []
        for preds, y_true in seeds:
            cap = capacity_from_predictions(preds, margin=margin)
            cost = asymmetric_op_cost(y_true, cap, alpha=op_a, beta=op_b)
            per_seed.append(cost)
        out[f"{op_a:g}:{op_b:g}"] = float(np.mean(per_seed))
    return out


def find_best_per_operator(matrix: dict[str, dict[str, Optional[float]]],
                           cell_labels: list[str],
                           operator_grid: list[tuple[float, float]]
                           ) -> dict[str, dict]:
    """For each operator α/β, find the training cell with the lowest cost.

    Returns: {operator_str: {best_cell, best_cost, mse_cost, win_pct}}.
    `win_pct` is the % improvement of best over MSE. Negative if MSE wins.
    `mse_cost` is None if no MSE baseline cell is present.
    """
    out: dict[str, dict] = {}
    for op_a, op_b in operator_grid:
        op_key = f"{op_a:g}:{op_b:g}"
        best_cell, best_cost = None, float("inf")
        mse_cost = None
        for cell in cell_labels:
            cost = matrix.get(cell, {}).get(op_key)
            if cost is None:
                continue
            if cell == "MSE":
                mse_cost = cost
            if cost < best_cost:
                best_cost, best_cell = cost, cell
        win_pct = None
        if mse_cost is not None and best_cost != float("inf"):
            win_pct = 100.0 * (mse_cost - best_cost) / mse_cost
        out[op_key] = {
            "best_cell": best_cell,
            "best_cost": (best_cost if best_cost != float("inf") else None),
            "mse_cost": mse_cost,
            "win_pct_vs_mse": win_pct,
        }
    return out


def plot_heatmap(matrix: dict[str, dict[str, Optional[float]]],
                 cell_labels: list[str],
                 operator_grid: list[tuple[float, float]],
                 save_path: str, dataset: str = "") -> None:
    """Heatmap: rows = training cells, cols = operator α/β.

    Cell color = (cost - MSE_cost_at_this_operator) / MSE_cost. Negative
    (= asym beats MSE) is green; positive (= MSE wins) is red. The MSE
    row is exactly zero by construction. Best training cell per column
    gets a black-outline marker.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    op_keys = [f"{a:g}:{b:g}" for a, b in operator_grid]
    n_rows = len(cell_labels)
    n_cols = len(op_keys)

    grid = np.full((n_rows, n_cols), np.nan)
    mse_idx = cell_labels.index("MSE") if "MSE" in cell_labels else None
    for i, cell in enumerate(cell_labels):
        for j, op in enumerate(op_keys):
            val = matrix.get(cell, {}).get(op)
            if val is None or mse_idx is None:
                continue
            mse_val = matrix.get("MSE", {}).get(op)
            if mse_val is None or mse_val == 0:
                continue
            grid[i, j] = (val - mse_val) / mse_val

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * n_cols), max(5, 0.6 * n_rows)))
    vmax = np.nanmax(np.abs(grid)) if np.any(~np.isnan(grid)) else 1.0
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"op {k}" for k in op_keys], rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(cell_labels)
    ax.set_xlabel("Operator cost ratio (α:β)")
    ax.set_ylabel("Training cost ratio (α:β)")
    title = "Δ asym_op_cost vs MSE (negative = asym wins)"
    if dataset:
        title += f" — {dataset}"
    ax.set_title(title)

    # Annotate each cell with the percent.
    for i in range(n_rows):
        for j in range(n_cols):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{100*v:+.1f}%", ha="center", va="center",
                    fontsize=8,
                    color=("white" if abs(v) > 0.5 * vmax else "black"))

    # Mark best training cell per operator column with a black outline.
    for j in range(n_cols):
        col = grid[:, j]
        if np.all(np.isnan(col)):
            continue
        best_i = int(np.nanargmin(col))
        rect = plt.Rectangle((j - 0.5, best_i - 0.5), 1, 1,
                             fill=False, edgecolor="black", linewidth=2.5)
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="(cost - MSE_cost) / MSE_cost")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[operator-eval] wrote {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-operator Pareto evaluator")
    parser.add_argument("--dataset", default="abilene")
    parser.add_argument("--pareto-base-dir", default=None,
                        help="default: results/<dataset>_pareto")
    parser.add_argument("--operator-ratios", nargs="+",
                        default=["1:1", "2:1", "5:1", "10:1", "20:1", "100:1"],
                        help="operator cost ratios to evaluate at")
    parser.add_argument("--margin", type=float, default=1.1,
                        help="safety margin for capacity provisioning (default 1.1)")
    parser.add_argument("--plot-path", default=None,
                        help="default: plots/operator_eval_<dataset>.png")
    parser.add_argument("--summary-path", default=None,
                        help="default: results/<dataset>_pareto/operator_eval.json")
    args = parser.parse_args()

    pareto_dir = args.pareto_base_dir or os.path.join(
        RESULTS_DIR, f"{args.dataset}_pareto"
    )
    operator_grid = [parse_ratio(r) for r in args.operator_ratios]

    cells = discover_cells(pareto_dir)
    if not cells:
        raise SystemExit(f"no cells found in {pareto_dir}")

    print(f"[operator-eval] dataset={args.dataset}  "
          f"cells={len(cells)}  operator_ratios={args.operator_ratios}")
    for label, a, b, _ in cells:
        print(f"  cell: {label} (training α/β = {a:g}/{b:g})")

    matrix: dict[str, dict[str, Optional[float]]] = {}
    cell_labels: list[str] = []
    for label, _, _, path in cells:
        seeds = load_cell_predictions(path)
        print(f"  [{label}] {len(seeds)} seeds with predictions")
        matrix[label] = evaluate_cell(seeds, operator_grid, margin=args.margin)
        cell_labels.append(label)

    best = find_best_per_operator(matrix, cell_labels, operator_grid)

    summary = {
        "dataset": args.dataset,
        "margin": args.margin,
        "operator_ratios": args.operator_ratios,
        "training_cells": cell_labels,
        "matrix": matrix,
        "best_per_operator": best,
    }
    summary_path = args.summary_path or os.path.join(
        pareto_dir, "operator_eval.json"
    )
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[operator-eval] wrote {summary_path}")

    plot_path = args.plot_path or os.path.join(
        PLOTS_DIR, f"operator_eval_{args.dataset}.png"
    )
    plot_heatmap(matrix, cell_labels, operator_grid, plot_path,
                 dataset=args.dataset)

    # Print a one-screen verdict.
    print("\n[operator-eval] verdict:")
    print(f"{'operator α:β':>14s}  {'best training':>20s}  "
          f"{'win % vs MSE':>14s}")
    for op_key, info in best.items():
        win = info.get("win_pct_vs_mse")
        win_str = f"{win:+.2f}%" if win is not None else "  N/A "
        print(f"{op_key:>14s}  {str(info['best_cell']):>20s}  {win_str:>14s}")


if __name__ == "__main__":
    main()
