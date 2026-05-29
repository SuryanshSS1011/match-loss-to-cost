"""Build the multi-dataset calibration table for §5.4 of the paper.

Reads:
    results/<ds>_calib_pareto/alpha_<target>/aggregated_results.json
    for ds in {abilene, geant, cesnet}, target in {0.05, 0.10, 0.20}.

Writes:
    report/table_calibration_main.tex (LaTeX, multirow per dataset)
    report/table_calibration_main.md  (Markdown preview)

Schema expected at each cell:
    models.LSTM_CQR.calibration.{coverage_overall, mean_width}.mean
    models.LSTM_ACI.calibration.{coverage_overall, mean_width}.mean

Usage:
    .venv/bin/python scripts/build_calibration_table.py
    .venv/bin/python scripts/build_calibration_table.py --datasets abilene
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DATASET_DISPLAY = {
    "abilene": "Abilene",
    "geant": r"G\'EANT",
    "cesnet": "CESNET",
}
TARGETS = (0.05, 0.10, 0.20)


def _read_cell(dataset: str, target_alpha: float, base: Path) -> dict | None:
    # Two on-disk forms exist because different launchers used different
    # formatters. The :g form drops trailing zeros (0.10 → 0.1); literal
    # bash interpolation preserves them (0.10 → 0.10). Accept either.
    canonical = f"alpha_{target_alpha:g}"
    literal = f"alpha_{target_alpha:.2f}"
    for d in (canonical, literal):
        path = base / f"{dataset}_calib_pareto" / d / "aggregated_results.json"
        if path.exists():
            break
    else:
        return None
    with open(path) as f:
        d = json.load(f)
    cqr = d.get("models", {}).get("LSTM_CQR", {}).get("calibration", {})
    aci = d.get("models", {}).get("LSTM_ACI", {}).get("calibration", {})
    if not cqr or not aci:
        return None
    return {
        "cqr_cov": cqr.get("coverage_overall", {}).get("mean"),
        "aci_cov": aci.get("coverage_overall", {}).get("mean"),
        "cqr_w":   cqr.get("mean_width", {}).get("mean"),
        "aci_w":   aci.get("mean_width", {}).get("mean"),
    }


def _format_cov(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _format_width(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def build_tex(rows: list[dict], output_tex: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Empirical coverage and mean band width across the three "
        r"datasets (LSTM, $20$ seeds). ACI attains the target coverage; "
        r"split-CQR under-covers under temporal drift.}",
        r"\label{tab:calibration}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Mean width} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Dataset & Target & CQR & ACI & CQR & ACI \\",
        r"\midrule",
    ]
    for ds_idx, ds in enumerate(rows):
        for t_idx, cell in enumerate(ds["cells"]):
            target_str = f"${1.0 - cell['target']:.2f}$"
            # ACI bold if it's closer to target than CQR (which is the paper's claim).
            target_cov = 1.0 - cell["target"]
            cqr_cov = cell["cqr_cov"]; aci_cov = cell["aci_cov"]
            cqr_str = _format_cov(cqr_cov)
            aci_str = _format_cov(aci_cov)
            if cqr_cov is not None and aci_cov is not None:
                if abs(aci_cov - target_cov) < abs(cqr_cov - target_cov):
                    aci_str = r"\textbf{" + aci_str + "}"
                else:
                    cqr_str = r"\textbf{" + cqr_str + "}"
            prefix = ""
            if t_idx == 0:
                prefix = rf"\multirow{{3}}{{*}}{{{ds['display']}}}"
            lines.append(
                rf"{prefix} & {target_str} & {cqr_str} & {aci_str} & "
                rf"{_format_width(cell['cqr_w'])} & {_format_width(cell['aci_w'])} \\"
            )
        if ds_idx < len(rows) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines) + "\n")
    print(f"[calib] wrote {output_tex}")


def build_md(rows: list[dict], output_md: Path) -> None:
    lines = [
        "| Dataset | Target | CQR coverage | ACI coverage | CQR width | ACI width |",
        "|---|---|---|---|---|---|",
    ]
    for ds in rows:
        for cell in ds["cells"]:
            target_str = f"{1.0 - cell['target']:.2f}"
            lines.append(
                f"| {ds['plain']} | {target_str} | "
                f"{_format_cov(cell['cqr_cov'])} | {_format_cov(cell['aci_cov'])} | "
                f"{_format_width(cell['cqr_w'])} | {_format_width(cell['aci_w'])} |"
            )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")
    print(f"[calib] wrote {output_md}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASET_DISPLAY.keys()))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-tex", default=str(PROJECT_ROOT / "report" / "table_calibration_main.tex"))
    parser.add_argument("--output-md", default=str(PROJECT_ROOT / "report" / "table_calibration_main.md"))
    args = parser.parse_args()

    base = Path(args.results_dir)
    rows = []
    for ds in args.datasets:
        cells = []
        for ta in TARGETS:
            cell = _read_cell(ds, ta, base)
            if cell is None:
                print(f"  [skip] {ds}@alpha_{ta}: aggregated json missing")
                cells.append({"target": ta, "cqr_cov": None, "aci_cov": None,
                              "cqr_w": None, "aci_w": None})
            else:
                cells.append({"target": ta, **cell})
        rows.append({"display": DATASET_DISPLAY[ds],
                     "plain": ds.upper() if ds == "cesnet" else ds.capitalize(),
                     "cells": cells})

    build_tex(rows, Path(args.output_tex))
    build_md(rows, Path(args.output_md))


if __name__ == "__main__":
    main()
