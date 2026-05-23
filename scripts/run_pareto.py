#!/usr/bin/env python
"""Pareto sweep over asymmetric-loss α/β ratios.

CLAUDE.md framing rule 1: "Sweep α/β ∈ {1:1, 5:1, 10:1, 100:1} and show
a Pareto frontier of overload-rate vs over-provisioning cost." This is
the headline figure for the paper.

Per ratio cell, we invoke `scripts.run_experiments.main_programmatic`
once with `--loss asym --alpha α --beta β`, then load the cell's
aggregated JSON and emit one point per model on the (over-provisioning
cost, overload rate) plane. One line per model, connecting increasing-α
points so the eye can trace each method's frontier.

Modes:
  default   — train all cells from scratch.
  --from-cache — read cached aggregated JSONs and replot only.

Per-cell artefacts land at:
    results/<dataset>_pareto/ratio_<a>_<b>/aggregated_results.json
The summary plot is written to plots/pareto_<dataset>.png and a
machine-readable summary at results/<dataset>_pareto/summary.json.
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


DEFAULT_RATIOS = ("1:1", "5:1", "10:1", "100:1")


def parse_ratio(s: str) -> tuple[float, float]:
    """Parse a 'a:b' string into a (α, β) tuple of floats."""
    if ":" not in s:
        raise ValueError(f"ratio {s!r} must be 'a:b'")
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"ratio {s!r} must have exactly one colon")
    try:
        a = float(parts[0])
        b = float(parts[1])
    except ValueError as e:
        raise ValueError(f"ratio {s!r} components must be numeric") from e
    if a <= 0 or b <= 0:
        raise ValueError(f"ratio {s!r} components must be positive")
    return a, b


def _sweep_root(dataset: str, loss_form: str = "asym",
                base_dir: str = RESULTS_DIR) -> str:
    """Per-loss-form sweep root: results/<dataset>_pareto_<loss_form>/.

    Different loss forms (squared 'asym' vs cusp-linear 'asym_l1') MUST
    cache to different roots so concurrent / sequential sweeps don't
    overwrite each other's per-cell artefacts. The MSE baseline cell
    also lives under its sweep's root, so a "mse baseline trained
    alongside an asym sweep" reads from the squared root and won't
    collide with one trained alongside the asym_l1 sweep.
    """
    return os.path.join(base_dir, f"{dataset}_pareto_{loss_form}")


def cell_output_dir(dataset: str, ratio_str: str,
                    base_dir: str = RESULTS_DIR,
                    loss_form: str = "asym") -> str:
    a, b = parse_ratio(ratio_str)
    return os.path.join(
        _sweep_root(dataset, loss_form, base_dir),
        f"ratio_{a:g}_{b:g}",
    )


def _aggregated_path(dataset: str, ratio_str: str,
                     base_dir: str = RESULTS_DIR,
                     loss_form: str = "asym") -> str:
    return os.path.join(
        cell_output_dir(dataset, ratio_str, base_dir, loss_form),
        "aggregated_results.json",
    )


def _load_or_run(dataset: str, ratio_str: str, seeds: list[int],
                 models: tuple, from_cache: bool,
                 base_dir: str = RESULTS_DIR,
                 loss_form: str = "asym") -> dict:
    """Run or load one ratio cell.

    `loss_form` selects the asymmetric variant: 'asym' (squared, default,
    matches AsymmetricMSE) or 'asym_l1' (cusp-linear, Eramo-style L1).
    Cells from different loss forms cache to different output dirs.
    """
    cache = _aggregated_path(dataset, ratio_str, base_dir, loss_form)
    if from_cache:
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"--from-cache: missing {cache}; run without it first"
            )
        with open(cache) as f:
            return json.load(f)

    a, b = parse_ratio(ratio_str)
    from scripts import run_experiments
    return run_experiments.main_programmatic(
        dataset=dataset, loss=loss_form,
        alpha=a, beta=b, tau=None,
        seeds=seeds,
        models=models,
        output_dir=cell_output_dir(dataset, ratio_str, base_dir, loss_form),
    )


def _load_or_run_mse_baseline(dataset: str, seeds: list[int],
                              models: tuple, from_cache: bool,
                              base_dir: str = RESULTS_DIR,
                              loss_form: str = "asym") -> dict:
    """Run or load the MSE-baseline cell.

    Used as a reference X marker on the Pareto plot so reviewers can see
    the asymmetric-loss frontier dominating the MSE point. Cached at
    results/<dataset>_pareto_<loss_form>/baseline_mse/aggregated_results.json.

    The MSE baseline lives under the *sweep's* root (not a shared root)
    so its per-seed predictions can't be overwritten by a concurrent
    sweep with a different loss form. Re-trains MSE once per sweep,
    which is wasteful (~22 min) but eliminates the cross-sweep
    contamination class of bug.
    """
    out_dir = os.path.join(_sweep_root(dataset, loss_form, base_dir),
                           "baseline_mse")
    cache = os.path.join(out_dir, "aggregated_results.json")
    if from_cache:
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"--from-cache: missing MSE baseline at {cache}; "
                f"run without --from-cache once to populate it"
            )
        with open(cache) as f:
            return json.load(f)
    from scripts import run_experiments
    return run_experiments.main_programmatic(
        dataset=dataset, loss="mse",
        alpha=1.0, beta=1.0, tau=None,
        seeds=seeds,
        models=models,
        output_dir=out_dir,
    )


def collect_points(by_ratio: dict[str, dict]) -> dict[str, list[dict]]:
    """Extract per-model points from each ratio's aggregated JSON.

    Returns {model_name: [{ratio, alpha, beta, overload_rate, over_prov_cost,
                            asym_op_cost, rmse_mean,
                            overload_rate_values}, ...]} sorted by α. The
    `*_values` lists carry the per-seed numbers (used for Wilcoxon overlays).
    """
    points: dict[str, list[dict]] = {}
    for ratio_str, agg in by_ratio.items():
        a, b = parse_ratio(ratio_str)
        for model, block in agg.get("models", {}).items():
            op = block.get("operational", {})
            forecast = block.get("forecast", {})
            entry = {
                "ratio": ratio_str,
                "alpha": a,
                "beta": b,
                "overload_rate": op.get("overload_rate", {}).get("mean"),
                "over_provisioning_cost": op.get(
                    "over_provisioning_cost", {}
                ).get("mean"),
                "asymmetric_op_cost": op.get(
                    "asymmetric_op_cost", {}
                ).get("mean"),
                "rmse_mean": forecast.get("rmse_mean", {}).get("mean"),
                "overload_rate_values": op.get("overload_rate", {}).get(
                    "values", []
                ),
            }
            points.setdefault(model, []).append(entry)
    for model in points:
        points[model].sort(key=lambda e: e["alpha"])
    return points


def _significance_per_ratio(
    by_ratio: dict[str, dict],
    reference: str,
    metric: str = "overload_rate",
    alpha: float = 0.05,
) -> dict[str, dict[str, Optional[bool]]]:
    """For each ratio cell, run paired Wilcoxon vs the reference model.

    Returns: {ratio_str: {model_name: True | False | None}}.
    `True` means the model strictly beats the reference on `metric`
    (lower-is-better) at FWER `alpha`, Holm-corrected.
    """
    from scripts._significance_overlay import significance_vs_reference
    out: dict[str, dict[str, Optional[bool]]] = {}
    for ratio_str, agg in by_ratio.items():
        rows: dict[str, dict] = {}
        for model, block in agg.get("models", {}).items():
            stat = block.get("operational", {}).get(metric, {})
            if "values" in stat:
                rows[model] = {"values": stat["values"]}
        if reference not in rows:
            out[ratio_str] = {model: None for model in rows}
            continue
        try:
            out[ratio_str] = significance_vs_reference(
                rows, reference=reference,
                lower_is_better=True, alpha=alpha,
            )
        except (KeyError, ValueError):
            out[ratio_str] = {model: None for model in rows}
    return out


def plot_pareto(points: dict[str, list[dict]], save_path: str,
                dataset: str = "",
                significance: Optional[dict[str, dict[str, Optional[bool]]]]
                = None,
                wilcoxon_reference: Optional[str] = None,
                mse_baseline: Optional[dict] = None) -> None:
    """One line per model on (over-prov cost, overload rate) plane.

    If `significance` is given, points where the model strictly beats the
    `wilcoxon_reference` on overload_rate (Holm-corrected) get a black
    edge ring; non-significant or reference points stay unringed. The
    legend gains a one-line annotation explaining the convention.

    If `mse_baseline` is a runner aggregated dict, each model's MSE-trained
    point is overlaid as a large black X-marker so the reader can see the
    asym frontier dominating the MSE point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    palette = {
        "SARIMA": "#7f7f7f", "LSTM": "#1f77b4",
        "DLinear": "#2ca02c", "PatchTST": "#9467bd",
        "iTransformer": "#d62728", "DCRNN": "#ff7f0e",
        "Chronos": "#8c564b",
    }

    def _engfmt(v):
        """Compact engineering-style number for axis ticks / annotations.

        Uses fixed-point (not %g) so values like 2e8 render as "200M", never
        "2e+02M". Trailing zeros are trimmed (250M, not 250.0M).
        """
        a = abs(v)
        if a >= 1e9:
            s, suf = v / 1e9, "B"
        elif a >= 1e6:
            s, suf = v / 1e6, "M"
        elif a >= 1e3:
            s, suf = v / 1e3, "k"
        else:
            s, suf = v, ""
        txt = f"{s:.1f}".rstrip("0").rstrip(".")
        return f"{txt}{suf}"

    plotted_any = False
    for model, entries in points.items():
        if model.endswith("_CQR") or model.endswith("_ACI"):
            continue
        xs = [e["over_provisioning_cost"] for e in entries
              if e["over_provisioning_cost"] is not None
              and e["overload_rate"] is not None]
        ys = [e["overload_rate"] for e in entries
              if e["over_provisioning_cost"] is not None
              and e["overload_rate"] is not None]
        es = [e for e in entries
              if e["over_provisioning_cost"] is not None
              and e["overload_rate"] is not None]
        if not xs:
            continue
        plotted_any = True
        color = palette.get(model, "#1f77b4")
        ax.plot(xs, ys, "-", color=color, alpha=0.9, linewidth=2.2,
                label=f"{model} (asym sweep)", zorder=3)
        for x, y, e in zip(xs, ys, es):
            ring = False
            if significance is not None and model != wilcoxon_reference:
                ring = bool(significance.get(e["ratio"], {}).get(model))
            ax.plot([x], [y], "o", color=color,
                    markeredgecolor="black" if ring else "white",
                    markeredgewidth=1.4 if ring else 0.8,
                    markersize=9, alpha=0.98, zorder=4)
            # Ratio label, offset above-right of the point, in the line colour.
            ax.annotate(f"{e['ratio']}", (x, y),
                        textcoords="offset points", xytext=(7, 5),
                        fontsize=9, fontweight="bold", color=color, zorder=5)

    # MSE-baseline X markers + a guide line from the frontier to show dominance.
    if mse_baseline is not None:
        for model, block in mse_baseline.get("models", {}).items():
            if model.endswith("_CQR") or model.endswith("_ACI"):
                continue
            op = block.get("operational", {})
            x = op.get("over_provisioning_cost", {}).get("mean")
            y = op.get("overload_rate", {}).get("mean")
            if x is None or y is None:
                continue
            ax.scatter([x], [y], marker="X", s=210, c="black",
                       edgecolors="white", linewidth=1.2, zorder=6,
                       label="MSE baseline")
            ax.annotate("MSE", (x, y), textcoords="offset points",
                        xytext=(10, -4), fontsize=10, fontweight="bold",
                        color="black", zorder=7)

    ax.set_xlabel("Mean over-provisioning cost  (lower = better)", fontsize=11)
    ax.set_ylabel("Mean overload rate  (lower = better)", fontsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _engfmt(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))

    ttl = "Operational trade-off: provisioning cost vs. overload rate"
    if dataset:
        ttl += f"  —  {dataset}"
    ax.set_title(ttl, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # de-dup repeated "MSE baseline" labels from multi-model overlays
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l); h2.append(h); l2.append(l)
        ax.legend(h2, l2, fontsize=9.5, loc="best", framealpha=0.92)

    note = ("Each point = one training α:β; lower-left is better. "
            "The asym frontier sits below-left of the MSE baseline "
            "→ lower overload at comparable cost.")
    if significance is not None and wilcoxon_reference is not None:
        note += (f"\nBlack ring = significantly beats {wilcoxon_reference} "
                 f"(paired Wilcoxon, Holm p<0.05).")
    fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=8.5,
             alpha=0.75, wrap=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pareto] wrote {save_path}")


def write_summary(by_ratio: dict[str, dict], points: dict[str, list[dict]],
                  out_path: str,
                  mse_baseline: Optional[dict] = None) -> None:
    out = {
        "ratios": list(by_ratio.keys()),
        "models": list(points.keys()),
        "per_model": points,
    }
    if mse_baseline is not None:
        # Just the per-model headline operational means; the full per-seed
        # values live in the MSE-baseline cell's aggregated_results.json.
        baseline_pts = {}
        for model, block in mse_baseline.get("models", {}).items():
            op = block.get("operational", {})
            forecast = block.get("forecast", {})
            baseline_pts[model] = {
                "overload_rate": op.get("overload_rate", {}).get("mean"),
                "over_provisioning_cost": op.get(
                    "over_provisioning_cost", {}
                ).get("mean"),
                "asymmetric_op_cost": op.get(
                    "asymmetric_op_cost", {}
                ).get("mean"),
                "rmse_mean": forecast.get("rmse_mean", {}).get("mean"),
            }
        out["mse_baseline"] = baseline_pts
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[pareto] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto α/β sweep")
    parser.add_argument("--dataset", default="abilene",
                        choices=("synthetic", "abilene", "geant", "cesnet"))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--models", nargs="+",
                        default=["sarima", "lstm"],
                        help="models to include in each cell's runner call")
    parser.add_argument("--ratios", nargs="+", default=list(DEFAULT_RATIOS),
                        help="α/β ratios as 'a:b' strings "
                             f"(default: {list(DEFAULT_RATIOS)})")
    parser.add_argument("--from-cache", action="store_true",
                        help="skip training; read cached cells only")
    parser.add_argument("--plot-path", default=None,
                        help="default: plots/pareto_<dataset>.png")
    parser.add_argument("--wilcoxon-vs", default=None,
                        help="display name of the reference model "
                             "for the per-point Wilcoxon overlay "
                             "(e.g. 'LSTM'). Skipped if absent.")
    parser.add_argument("--loss-form", default="asym",
                        choices=("asym", "asym_l1"),
                        help="asymmetric loss variant for the sweep: "
                             "'asym' = squared (default, AsymmetricMSE), "
                             "'asym_l1' = cusp-linear (Eramo-style L1, "
                             "less knob-sensitive at extreme α/β)")
    parser.add_argument("--include-mse-baseline", action="store_true",
                        help="train a separate MSE-loss cell and emit it as "
                             "an X-marker reference point on the Pareto "
                             "plot. Lets reviewers see the asym frontier "
                             "dominating MSE in (overload, over-prov) space.")
    args = parser.parse_args()

    # Validate ratio strings up front so we fail fast.
    for r in args.ratios:
        parse_ratio(r)

    by_ratio: dict[str, dict] = {}
    for r in args.ratios:
        print(f"[pareto] dataset={args.dataset}  ratio={r}  "
              f"loss_form={args.loss_form}  "
              f"from_cache={args.from_cache}")
        by_ratio[r] = _load_or_run(
            args.dataset, r, args.seeds,
            tuple(args.models), args.from_cache,
            loss_form=args.loss_form,
        )

    mse_baseline = None
    if args.include_mse_baseline:
        print(f"[pareto] dataset={args.dataset}  loss=mse (baseline)  "
              f"from_cache={args.from_cache}  loss_form={args.loss_form}")
        mse_baseline = _load_or_run_mse_baseline(
            args.dataset, args.seeds,
            tuple(args.models), args.from_cache,
            loss_form=args.loss_form,
        )

    points = collect_points(by_ratio)

    significance = None
    if args.wilcoxon_vs is not None:
        significance = _significance_per_ratio(
            by_ratio, reference=args.wilcoxon_vs, metric="overload_rate",
        )

    plot_path = args.plot_path or os.path.join(
        PLOTS_DIR, f"pareto_{args.dataset}.png"
    )
    plot_pareto(points, plot_path, dataset=args.dataset,
                significance=significance,
                wilcoxon_reference=args.wilcoxon_vs,
                mse_baseline=mse_baseline)

    summary_path = os.path.join(
        _sweep_root(args.dataset, args.loss_form), "summary.json"
    )
    write_summary(by_ratio, points, summary_path,
                  mse_baseline=mse_baseline)


if __name__ == "__main__":
    main()
