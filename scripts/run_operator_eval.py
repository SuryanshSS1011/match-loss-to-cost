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
                  operator_grid: list[tuple[float, float]]) -> dict:
    """For one cell, compute per-seed and mean operator-cost at each operator α/β.

    Operator cost is the *prediction-level* asymmetric squared loss:
        cost(α, β) = α · Σ max(y − ŷ, 0)² + β · Σ max(ŷ − y, 0)²
    summed over all (t, link) cells. This is exactly the loss the runner
    minimises during training, just evaluated post-hoc with a fixed
    operator (α, β) instead of the training (α, β).

    We deliberately do NOT route through `capacity_from_predictions` +
    `asymmetric_op_cost` here. That route compares y_true against the
    model's own provisioned capacity (margin × max(pred)), which means
    each model is judged against capacity *it itself chose*. An over-
    predictor like asym 100:1 always has its own capacity above truth
    everywhere → zero underloads, large headroom → cost ≈ β × headroom.
    That number is operationally meaningless for cross-model comparison
    because the capacity scheme leaks the predictor.

    Direct prediction-level cost has no such leak: every model's
    predictions are compared to the same y_true under the same operator
    weights, and the question becomes "which forecaster's predictions
    minimise the operator's actual cost surrogate?"

    Returns: {operator_key: {"mean": float, "values": [per-seed costs in input order]}}
             or {operator_key: None} for cells with no usable seeds.
    """
    if not seeds:
        return {f"{a}:{b}": None for a, b in operator_grid}
    out: dict = {}
    for op_a, op_b in operator_grid:
        per_seed: list[float] = []
        for preds, y_true in seeds:
            err = y_true - preds
            under = np.clip(err, 0.0, None)   # y > ŷ (under-prediction)
            over = np.clip(-err, 0.0, None)   # ŷ > y (over-prediction)
            cost = op_a * (under ** 2).sum() + op_b * (over ** 2).sum()
            per_seed.append(float(cost))
        out[f"{op_a:g}:{op_b:g}"] = {
            "mean": float(np.mean(per_seed)),
            "values": per_seed,
        }
    return out


def _bootstrap_ci_winpct(cell_values: list[float],
                         mse_values: list[float],
                         n_boot: int = 2000,
                         alpha: float = 0.05,
                         rng: Optional[np.random.Generator] = None
                         ) -> tuple[float, float, float]:
    """Bootstrap 95% CI on win% = 100 * (mse_mean - cell_mean) / mse_mean.

    Paired bootstrap: at each of `n_boot` iterations, resample seed
    *indices* with replacement, then recompute the win% on (cell, mse)
    pairs at those indices. Returns (point, lo, hi) — the point estimate
    is from the original (non-resampled) means, and (lo, hi) is the
    central (1−α) interval over the bootstrap distribution.

    Pairing matters: even though seed routing diverged between MSE and
    asym cells (different model checkpoints), seed *index* still
    correlates the data realisation each model was evaluated on (same
    test set). The paired bootstrap respects that correlation; the
    independent-samples form would over-estimate the variance.

    Returns (None, None, None) if either series has < 2 valid points.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cv = np.asarray(cell_values, dtype=np.float64)
    mv = np.asarray(mse_values, dtype=np.float64)
    n = min(len(cv), len(mv))
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))
    cv, mv = cv[:n], mv[:n]
    point = 100.0 * (mv.mean() - cv.mean()) / mv.mean() if mv.mean() != 0 else 0.0
    boots = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        cv_b = cv[idx].mean()
        mv_b = mv[idx].mean()
        boots[k] = (
            100.0 * (mv_b - cv_b) / mv_b if mv_b != 0 else 0.0
        )
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return (float(point), lo, hi)


def _cell_mean(matrix: dict, cell: str, op_key: str) -> Optional[float]:
    block = matrix.get(cell, {}).get(op_key)
    if block is None:
        return None
    return block.get("mean")


def _cell_values(matrix: dict, cell: str, op_key: str) -> Optional[list[float]]:
    block = matrix.get(cell, {}).get(op_key)
    if block is None:
        return None
    return block.get("values")


def find_best_per_operator(matrix: dict,
                           cell_labels: list[str],
                           operator_grid: list[tuple[float, float]],
                           n_boot: int = 2000,
                           ) -> dict[str, dict]:
    """For each operator α/β, find the training cell with the lowest cost.

    Returns: {operator_str: {
        best_cell, best_cost, mse_cost,
        win_pct_vs_mse, win_pct_lo, win_pct_hi, significant
    }}.
    `win_pct_lo`/`win_pct_hi` is the central 95% paired-bootstrap CI on
    the win % over MSE; `significant` is True iff that CI excludes zero.
    """
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for op_a, op_b in operator_grid:
        op_key = f"{op_a:g}:{op_b:g}"
        best_cell, best_cost = None, float("inf")
        for cell in cell_labels:
            cost = _cell_mean(matrix, cell, op_key)
            if cost is None:
                continue
            if cost < best_cost:
                best_cost, best_cell = cost, cell

        mse_cost = _cell_mean(matrix, "MSE", op_key)
        win_pct, lo, hi, sig = None, None, None, None
        if (mse_cost is not None and best_cell is not None
                and best_cell != "MSE"):
            best_vals = _cell_values(matrix, best_cell, op_key) or []
            mse_vals = _cell_values(matrix, "MSE", op_key) or []
            point, lo_b, hi_b = _bootstrap_ci_winpct(
                best_vals, mse_vals, n_boot=n_boot, alpha=0.05, rng=rng,
            )
            win_pct = point
            if not (np.isnan(lo_b) or np.isnan(hi_b)):
                lo, hi = lo_b, hi_b
                sig = bool(lo > 0 or hi < 0)  # CI excludes 0 ⇒ real win/loss
        elif mse_cost is not None and best_cell == "MSE":
            win_pct, lo, hi, sig = 0.0, 0.0, 0.0, False

        out[op_key] = {
            "best_cell": best_cell,
            "best_cost": (best_cost if best_cost != float("inf") else None),
            "mse_cost": mse_cost,
            "win_pct_vs_mse": win_pct,
            "win_pct_lo": lo,
            "win_pct_hi": hi,
            "significant": sig,
        }
    return out


def _significance_grid(matrix: dict, cell_labels: list[str],
                       op_keys: list[str], n_boot: int = 2000,
                       ) -> np.ndarray:
    """For each (cell, operator) pair, paired-bootstrap CI on win% vs MSE.

    Returns a (n_cells, n_ops) bool array where True means the 95% CI on
    win% (vs MSE at the same operator) strictly excludes zero — i.e. the
    cell is significantly different from MSE on that operator's cost.
    The MSE row itself is always False (by construction it's zero).
    """
    rng = np.random.default_rng(0)
    n_rows, n_cols = len(cell_labels), len(op_keys)
    sig = np.zeros((n_rows, n_cols), dtype=bool)
    if "MSE" not in cell_labels:
        return sig
    for j, op in enumerate(op_keys):
        mse_vals = _cell_values(matrix, "MSE", op) or []
        if len(mse_vals) < 2:
            continue
        for i, cell in enumerate(cell_labels):
            if cell == "MSE":
                continue
            cell_vals = _cell_values(matrix, cell, op) or []
            if len(cell_vals) < 2:
                continue
            _, lo, hi = _bootstrap_ci_winpct(
                cell_vals, mse_vals, n_boot=n_boot, alpha=0.05, rng=rng,
            )
            if not (np.isnan(lo) or np.isnan(hi)):
                sig[i, j] = bool(lo > 0 or hi < 0)
    return sig


def plot_heatmap(matrix: dict,
                 cell_labels: list[str],
                 operator_grid: list[tuple[float, float]],
                 save_path: str, dataset: str = "",
                 n_boot: int = 2000) -> None:
    """Heatmap: rows = training cells, cols = operator α/β.

    Cell color = (cost - MSE_cost_at_this_operator) / MSE_cost. Negative
    (= asym beats MSE) is green; positive (= MSE wins) is red. The MSE
    row is exactly zero by construction. Best training cell per column
    gets a black-outline marker. Cells where the paired-bootstrap 95% CI
    on win% straddles zero (statistically indistinguishable from MSE)
    get a hatched overlay so reviewers see which numbers are real wins
    vs seed noise.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    op_keys = [f"{a:g}:{b:g}" for a, b in operator_grid]
    n_rows = len(cell_labels)
    n_cols = len(op_keys)

    grid = np.full((n_rows, n_cols), np.nan)
    for i, cell in enumerate(cell_labels):
        for j, op in enumerate(op_keys):
            val = _cell_mean(matrix, cell, op)
            mse_val = _cell_mean(matrix, "MSE", op)
            if val is None or mse_val is None or mse_val == 0:
                continue
            grid[i, j] = (val - mse_val) / mse_val

    sig = _significance_grid(matrix, cell_labels, op_keys, n_boot=n_boot)

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

    # Annotate each cell with the percent (and hatched if non-significant).
    for i in range(n_rows):
        for j in range(n_cols):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{100*v:+.1f}%", ha="center", va="center",
                    fontsize=8,
                    color=("white" if abs(v) > 0.5 * vmax else "black"))
            # MSE row is always exactly zero; never hatch.
            if cell_labels[i] != "MSE" and not sig[i, j]:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="white",
                    hatch="///", linewidth=0.0,
                    alpha=0.6, zorder=2,
                ))

    # Mark best training cell per operator column with a black outline.
    for j in range(n_cols):
        col = grid[:, j]
        if np.all(np.isnan(col)):
            continue
        best_i = int(np.nanargmin(col))
        rect = plt.Rectangle((j - 0.5, best_i - 0.5), 1, 1,
                             fill=False, edgecolor="black", linewidth=2.5,
                             zorder=3)
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="(cost - MSE_cost) / MSE_cost")
    plt.figtext(
        0.01, 0.01,
        "Hatched = 95% bootstrap CI on win% includes 0 (not significant)",
        ha="left", va="bottom", fontsize=7, alpha=0.7,
    )
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
        matrix[label] = evaluate_cell(seeds, operator_grid)
        cell_labels.append(label)

    best = find_best_per_operator(matrix, cell_labels, operator_grid)

    summary = {
        "dataset": args.dataset,
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
          f"{'win % vs MSE':>14s}  {'95% CI':>22s}  {'sig?':>5s}")
    for op_key, info in best.items():
        win = info.get("win_pct_vs_mse")
        lo = info.get("win_pct_lo")
        hi = info.get("win_pct_hi")
        sig = info.get("significant")
        win_str = f"{win:+.2f}%" if win is not None else "  N/A "
        if lo is not None and hi is not None:
            ci_str = f"[{lo:+.1f}%, {hi:+.1f}%]"
        else:
            ci_str = "      N/A      "
        sig_str = ("yes" if sig else "no") if sig is not None else "n/a"
        print(f"{op_key:>14s}  {str(info['best_cell']):>20s}  "
              f"{win_str:>14s}  {ci_str:>22s}  {sig_str:>5s}")


if __name__ == "__main__":
    main()
