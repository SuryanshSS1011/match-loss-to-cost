#!/usr/bin/env python
"""E6 — significance synthesis across the operator-eval matrix.

Consumes the per-(training_cell × operator) cost matrix that
`run_operator_eval.py` writes to `results/<dataset>_pareto_<form>/operator_eval.json`
(each cell carries `{mean, values:[per-seed]}`) and produces:

  1. A **critical-difference diagram** (Demsar 2006) ranking the training-α
     cells across the operator columns — i.e. "which training loss config is
     best, averaged over operator cost structures, and which differences are
     significant?" One diagram per (dataset, loss-form).

  2. A **paired-Wilcoxon + Holm** table per operator column, each training cell
     vs the MSE baseline, using the per-seed values. Printed and written to
     `results/<dataset>_pareto_<form>/significance.json`.

This is pure analysis — no training. Run after the sweeps + operator-eval land.

Usage:
    python scripts/build_significance.py --dataset geant --loss-form asym_l1
    python scripts/build_significance.py --dataset geant --loss-form asym_l1 \
        --cd-plot plots/cd_geant_asym_l1.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import RESULTS_DIR, PLOTS_DIR  # noqa: E402
from src.evaluation.significance import (  # noqa: E402
    critical_difference_diagram,
    pairwise_significance_table,
)


def _operator_eval_path(dataset: str, loss_form: str) -> str:
    return os.path.join(
        RESULTS_DIR, f"{dataset}_pareto_{loss_form}", "operator_eval.json"
    )


def _values_per_cell_across_operators(matrix: dict) -> dict[str, list[float]]:
    """Build {training_cell: [mean_cost_at_op1, ..., mean_cost_at_opK]}.

    Each training cell becomes a 'model'; each operator column is a 'dataset'
    in the Demsar sense. Ranking is over the operator columns.
    """
    out: dict[str, list[float]] = {}
    # Stable operator order from the first cell.
    first = next(iter(matrix.values()))
    operators = list(first.keys())
    for cell, ops in matrix.items():
        out[cell] = [
            ops[op]["mean"] if isinstance(ops[op], dict) else ops[op]
            for op in operators
        ]
    return out


def _per_operator_wilcoxon(matrix: dict, reference: str) -> dict:
    """Per operator: paired Wilcoxon + Holm of each cell vs `reference`."""
    first = next(iter(matrix.values()))
    operators = list(first.keys())
    out = {}
    for op in operators:
        values_by_cell = {}
        for cell, ops in matrix.items():
            entry = ops.get(op)
            if isinstance(entry, dict) and "values" in entry:
                values_by_cell[cell] = entry["values"]
        if reference not in values_by_cell:
            out[op] = {"error": f"reference {reference!r} missing"}
            continue
        rows = pairwise_significance_table(
            values_by_cell, lower_is_better=True, alpha=0.05, reference=reference
        )
        out[op] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E6 significance synthesis")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss-form", default="asym_l1",
                    choices=("asym", "asym_l1"))
    ap.add_argument("--reference", default="MSE",
                    help="baseline cell name to test against (default MSE)")
    ap.add_argument("--cd-plot", default=None,
                    help="path for the critical-difference PNG "
                         "(default plots/cd_<dataset>_<form>.png)")
    args = ap.parse_args()

    oe_path = _operator_eval_path(args.dataset, args.loss_form)
    if not os.path.exists(oe_path):
        sys.exit(f"missing {oe_path}; run run_operator_eval.py first")
    oe = json.load(open(oe_path))
    matrix = oe["matrix"]

    # 1. Critical-difference diagram.
    cd_plot = args.cd_plot or os.path.join(
        PLOTS_DIR, f"cd_{args.dataset}_{args.loss_form}.png"
    )
    values_by_cell = _values_per_cell_across_operators(matrix)
    cd_info = critical_difference_diagram(
        values_by_cell, cd_plot, lower_is_better=True, alpha=0.05,
        title=f"{args.dataset} ({args.loss_form}) — training-cell ranks over operators",
    )
    print(f"[sig] wrote CD diagram → {cd_plot}")
    print(f"[sig] avg ranks (lower=better): "
          + ", ".join(f"{m}={r:.2f}" for m, r in
                      sorted(cd_info['ranks'].items(), key=lambda kv: kv[1])))
    print(f"[sig] critical difference (M={cd_info['M']}, K={cd_info['K']}): "
          f"{cd_info['cd']:.3f}")

    # 2. Per-operator Wilcoxon vs reference.
    wilcoxon = _per_operator_wilcoxon(matrix, args.reference)
    print(f"\n[sig] paired Wilcoxon + Holm vs {args.reference} (per operator):")
    for op, rows in wilcoxon.items():
        if isinstance(rows, dict) and "error" in rows:
            print(f"  {op}: {rows['error']}")
            continue
        sig = [r for r in rows if r.get("reject")]
        print(f"  operator {op}: {len(sig)}/{len(rows)} cells significantly "
              f"beat {args.reference}")

    out_path = os.path.join(
        RESULTS_DIR, f"{args.dataset}_pareto_{args.loss_form}", "significance.json"
    )
    json.dump(
        {"dataset": args.dataset, "loss_form": args.loss_form,
         "reference": args.reference, "cd": cd_info, "wilcoxon": wilcoxon},
        open(out_path, "w"), indent=2, default=float,
    )
    print(f"\n[sig] wrote {out_path}")


if __name__ == "__main__":
    main()
