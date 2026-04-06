#!/usr/bin/env python
"""Build the headline IEEEtran-style results table from aggregated JSONs.

Reads one or more `aggregated_results.json` files (the runner emits one
per (dataset, loss) cell at `results/<dataset>_<loss>/aggregated_results.json`)
and produces:
  - LaTeX: `\\toprule … \\midrule … \\bottomrule` `tabular` body, ready
    to paste into the CNSM/IEEE template after a `\\begin{tabular}{l ...}`
    column spec wrapper.
  - Markdown twin: a |…|…| table that mirrors the LaTeX one for the
    repo README and progress logs.

Usage:
    python scripts/build_headline_table.py \\
        --inputs results/abilene_asym/aggregated_results.json \\
        --output report/table_abilene.tex --markdown

Multi-cell mode (e.g. one row per dataset):
    python scripts/build_headline_table.py \\
        --inputs results/abilene_asym/aggregated_results.json \\
                  results/geant_asym/aggregated_results.json \\
        --names abilene geant \\
        --output report/table_combined.tex

The headline column order follows CLAUDE.md Rule 1: operational metrics
*first*, RMSE/MAE last and only "to show methods tie on RMSE while
differing on op cost." Calibration rows (`<model>_CQR`, `<model>_ACI`)
fill em-dashes for forecast-only columns and add `coverage` / `width`
columns when those keys are present.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# Column ordering and metadata. Each entry: (display_name, JSON path,
# group, lower_is_better, percent, fmt). `group` mirrors the runner's
# block names so we can pull from the aggregated JSON without remapping.
HEADLINE_COLUMNS = [
    # operational (lower is better)
    ("Asym. op. cost", "asymmetric_op_cost", "operational", True, False, "{:.2f}"),
    ("Overload rate", "overload_rate",       "operational", True, True,  "{:.2f}"),
    ("Over-prov. cost", "over_provisioning_cost", "operational", True, False, "{:.2f}"),
    ("U_max mean", "u_max_mean",            "operational", True, False, "{:.3f}"),
    # forecast (lower is better)
    ("RMSE", "rmse_mean", "forecast", True, False, "{:.3f}"),
    ("MAE", "mae_mean",   "forecast", True, False, "{:.3f}"),
    # calibration (special handling: only shown when any row has them)
    ("Coverage", "coverage_overall", "calibration", False, True, "{:.2f}"),
    ("Mean width", "mean_width",     "calibration", True,  False, "{:.3f}"),
]


def _format_value(stat: Optional[dict], fmt: str, percent: bool) -> str:
    """Render a {mean, std} stat dict as 'mean ± std' (or '—' if missing)."""
    if stat is None or stat.get("mean") is None:
        return "--"
    mean = stat["mean"]
    std = stat.get("std", 0.0)
    if percent:
        mean *= 100.0
        std *= 100.0
        out = fmt.format(mean) + r"\% $\pm$ " + fmt.format(std) + r"\%"
    else:
        out = fmt.format(mean) + r" $\pm$ " + fmt.format(std)
    return out


def _format_value_md(stat: Optional[dict], fmt: str, percent: bool) -> str:
    if stat is None or stat.get("mean") is None:
        return "—"
    mean = stat["mean"]
    std = stat.get("std", 0.0)
    if percent:
        return f"{fmt.format(mean*100)}% ± {fmt.format(std*100)}%"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def _is_calibration_row(model_name: str) -> bool:
    return model_name.endswith("_CQR") or model_name.endswith("_ACI")


def collect_rows(
    aggregated: list[dict],
    names: Optional[list[str]] = None,
) -> list[dict]:
    """Flatten one or more aggregated JSONs into a list of row dicts.

    Each row carries its display label (with optional name prefix) and
    a per-column stat reference so the renderer can do mean±std + bold-best.
    """
    rows: list[dict] = []
    for i, agg in enumerate(aggregated):
        prefix = (names[i] + "/") if names else ""
        for model, block in agg.get("models", {}).items():
            row: dict = {
                "label": f"{prefix}{model}",
                "model": model,
                "is_calibration": _is_calibration_row(model),
                "stats": {},
            }
            for col_label, key, group, *_ in HEADLINE_COLUMNS:
                row["stats"][col_label] = (
                    block.get(group, {}).get(key)
                )
            rows.append(row)
    return rows


def column_visible(col_label: str, rows: list[dict]) -> bool:
    """A column is visible iff at least one row has a non-missing stat."""
    for row in rows:
        s = row["stats"].get(col_label)
        if s is not None and s.get("mean") is not None:
            return True
    return False


def bold_best_indices(rows: list[dict],
                      lower_is_better: bool = True) -> dict[str, set[int]]:
    """For each visible column, return the row indices whose mean is the best.

    Forecast columns are typically NaN/missing for calibration rows; those
    rows are skipped from the per-column min/max comparison.
    """
    best: dict[str, set[int]] = {}
    for col_label, _key, _group, col_lib, _pct, _fmt in HEADLINE_COLUMNS:
        if not column_visible(col_label, rows):
            continue
        candidates: list[tuple[int, float]] = []
        for i, row in enumerate(rows):
            s = row["stats"].get(col_label)
            if s is not None and s.get("mean") is not None:
                candidates.append((i, float(s["mean"])))
        if not candidates:
            continue
        target = (min(c[1] for c in candidates) if col_lib
                  else max(c[1] for c in candidates))
        best[col_label] = {i for i, v in candidates if math.isclose(
            v, target, rel_tol=0, abs_tol=1e-12
        )}
    return best


def render_latex(
    rows: list[dict],
    bold_best: bool = True,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """Emit a complete `\\begin{table}...\\end{table}` LaTeX snippet."""
    visible_cols = [
        (label_, key, group, lib, pct, fmt)
        for (label_, key, group, lib, pct, fmt) in HEADLINE_COLUMNS
        if column_visible(label_, rows)
    ]
    col_spec = "l" + "r" * len(visible_cols)
    best = bold_best_indices(rows) if bold_best else {}

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Model & " + " & ".join(c[0] for c in visible_cols) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for i, row in enumerate(rows):
        cells = [row["label"].replace("_", r"\_")]
        for col_label, _key, _group, _lib, pct, fmt in visible_cols:
            stat = row["stats"].get(col_label)
            cell = _format_value(stat, fmt, pct)
            if bold_best and i in best.get(col_label, set()):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_markdown(rows: list[dict], bold_best: bool = True) -> str:
    visible_cols = [
        (label_, key, group, lib, pct, fmt)
        for (label_, key, group, lib, pct, fmt) in HEADLINE_COLUMNS
        if column_visible(label_, rows)
    ]
    best = bold_best_indices(rows) if bold_best else {}

    lines: list[str] = []
    header = "| Model | " + " | ".join(c[0] for c in visible_cols) + " |"
    sep = "|" + "---|" * (len(visible_cols) + 1)
    lines.append(header)
    lines.append(sep)
    for i, row in enumerate(rows):
        cells = [row["label"]]
        for col_label, _key, _group, _lib, pct, fmt in visible_cols:
            cell = _format_value_md(row["stats"].get(col_label), fmt, pct)
            if bold_best and i in best.get(col_label, set()):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="headline results table")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="aggregated_results.json files to read")
    parser.add_argument("--names", nargs="+", default=None,
                        help="prefix label per input; one per --inputs")
    parser.add_argument("--output", default=None,
                        help="LaTeX output path (default: stdout)")
    parser.add_argument("--markdown", action="store_true",
                        help="also write a .md sibling next to --output")
    parser.add_argument("--no-bold-best", action="store_true")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    if args.names is not None and len(args.names) != len(args.inputs):
        raise SystemExit("--names must match --inputs count")

    aggregated = []
    for p in args.inputs:
        with open(p) as f:
            aggregated.append(json.load(f))

    rows = collect_rows(aggregated, names=args.names)
    if not rows:
        raise SystemExit("no model rows found in inputs")

    bold = not args.no_bold_best
    latex = render_latex(rows, bold_best=bold,
                          caption=args.caption, label=args.label)

    if args.output:
        with open(args.output, "w") as f:
            f.write(latex)
        print(f"[table] wrote {args.output}")
        if args.markdown:
            md_path = os.path.splitext(args.output)[0] + ".md"
            with open(md_path, "w") as f:
                f.write(render_markdown(rows, bold_best=bold))
            print(f"[table] wrote {md_path}")
    else:
        print(latex)
        if args.markdown:
            print()
            print(render_markdown(rows, bold_best=bold))


if __name__ == "__main__":
    main()
