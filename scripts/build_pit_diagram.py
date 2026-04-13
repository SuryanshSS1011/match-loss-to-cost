#!/usr/bin/env python
"""PIT histogram + reliability diagram per (model, calibration method).

The probability-integral-transform (PIT) of `y` under predicted CDF F is
`u = F(y)`. If the predictive distribution is well-calibrated, u is
uniformly distributed on [0, 1]. The reliability diagram plots
*empirical* coverage at each nominal level α — i.e. the fraction of test
points where `y ≤ F⁻¹(α)` — vs the nominal α. A perfectly calibrated
model lies on the diagonal.

Where we get the quantiles:
  - Pinball-trained pairs (`<model>_qlo_predictions.npz`,
    `<model>_qhi_predictions.npz`): only two quantiles per cell
    (τ_lo, τ_hi). The PIT is binarised to {below band, in band, above
    band}, so the histogram has only 3 visible bins and the reliability
    diagram only 2 anchor points (at τ_lo and τ_hi). Documented; this is
    a coverage check more than a full PIT.
  - Future hook: if the runner ever persists a richer per-quantile
    prediction (e.g. Chronos-Bolt's 9 quantiles), this script would read
    those and produce a full 9-bin PIT. Stub kept for the follow-up.

Usage:
    python scripts/build_pit_diagram.py \\
        --seed-dir results/abilene_asym/seed_42 \\
        --model lstm \\
        --tau-lo 0.05 --tau-hi 0.95 \\
        --output plots/pit_lstm_seed42.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def coarse_pit(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> dict:
    """Return the coarse 3-bin PIT for a (q_lo, q_hi) pair.

    Output:
        below_band: fraction of cells with y < q_lo
        in_band:    fraction with q_lo <= y <= q_hi
        above_band: fraction with y > q_hi
    """
    if not (y_true.shape == q_lo.shape == q_hi.shape):
        raise ValueError(
            f"shape mismatch: y {y_true.shape}, q_lo {q_lo.shape}, "
            f"q_hi {q_hi.shape}"
        )
    below = float(np.mean(y_true < q_lo))
    above = float(np.mean(y_true > q_hi))
    in_band = 1.0 - below - above
    return {"below_band": below, "in_band": in_band, "above_band": above}


def reliability_anchors(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    tau_lo: float,
    tau_hi: float,
) -> dict:
    """Return the two reliability-diagram anchor points.

    A τ-quantile predictor `q_τ` is well-calibrated iff
    `P(y ≤ q_τ) = τ`. Here we estimate that empirical probability for
    τ_lo and τ_hi, then return the (nominal, empirical) pairs for plotting.
    """
    return {
        "tau_lo": float(tau_lo),
        "empirical_lo": float(np.mean(y_true <= q_lo)),
        "tau_hi": float(tau_hi),
        "empirical_hi": float(np.mean(y_true <= q_hi)),
    }


def plot_pit_and_reliability(
    pit: dict,
    anchors: dict,
    save_path: str,
    title: str = "",
) -> None:
    """Two-panel figure: 3-bin PIT histogram + reliability diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # PIT 3-bin.
    bins = ["y < q_lo", "in band", "y > q_hi"]
    heights = [pit["below_band"], pit["in_band"], pit["above_band"]]
    colors = ["tab:red", "tab:green", "tab:red"]
    ax1.bar(bins, heights, color=colors, edgecolor="black", alpha=0.85)
    target_lo = anchors["tau_lo"]
    target_hi = 1.0 - anchors["tau_hi"]  # mass above q_hi if perfectly calibrated
    target_mid = anchors["tau_hi"] - anchors["tau_lo"]
    ax1.axhline(target_lo, color="tab:red", linestyle="--", linewidth=0.8,
                alpha=0.6, label=f"target {target_lo:.2f}")
    ax1.axhline(target_mid, color="tab:green", linestyle="--", linewidth=0.8,
                alpha=0.6, label=f"target {target_mid:.2f}")
    ax1.set_ylabel("Fraction of test cells")
    ax1.set_ylim(0, 1)
    ax1.set_title("3-bin PIT")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Reliability: nominal vs empirical at the two anchors + diagonal.
    ax2.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5,
             label="perfect")
    ax2.plot(
        [anchors["tau_lo"], anchors["tau_hi"]],
        [anchors["empirical_lo"], anchors["empirical_hi"]],
        "o-", color="tab:blue", markersize=8,
        linewidth=1.5, label="model",
    )
    for tau, emp in (("tau_lo", "empirical_lo"), ("tau_hi", "empirical_hi")):
        ax2.annotate(
            f" τ={anchors[tau]:.2f}",
            (anchors[tau], anchors[emp]),
            fontsize=8, alpha=0.7,
        )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Nominal quantile τ")
    ax2.set_ylabel("Empirical P(y ≤ q_τ)")
    ax2.set_title("Reliability (2-anchor)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11)
        fig.subplots_adjust(top=0.86)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[pit] wrote {save_path}")


def _read_quantile_pair(seed_dir: str, model: str
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (q_lo, q_hi, y_true) test arrays from a seed dir."""
    qlo_path = os.path.join(seed_dir, f"{model}_qlo_predictions.npz")
    qhi_path = os.path.join(seed_dir, f"{model}_qhi_predictions.npz")
    if not os.path.exists(qlo_path) or not os.path.exists(qhi_path):
        raise FileNotFoundError(
            f"missing quantile npz under {seed_dir}: "
            f"{model}_qlo_predictions.npz / _qhi_predictions.npz"
        )
    qlo = np.load(qlo_path)
    qhi = np.load(qhi_path)
    return (
        np.asarray(qlo["predictions"], dtype=np.float32),
        np.asarray(qhi["predictions"], dtype=np.float32),
        np.asarray(qlo["L_test_aligned"], dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIT + reliability diagram for a quantile-trained model"
    )
    parser.add_argument("--seed-dir", required=True,
                        help="path to a per-seed dir, e.g. "
                             "results/abilene_asym/seed_42")
    parser.add_argument("--model", required=True,
                        help="model name (lstm, dlinear, patchtst, ...)")
    parser.add_argument("--tau-lo", type=float, default=0.05)
    parser.add_argument("--tau-hi", type=float, default=0.95)
    parser.add_argument("--output", required=True,
                        help="output PNG path")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    q_lo, q_hi, y_true = _read_quantile_pair(args.seed_dir, args.model)
    pit = coarse_pit(y_true, q_lo, q_hi)
    anchors = reliability_anchors(y_true, q_lo, q_hi,
                                    args.tau_lo, args.tau_hi)
    title = args.title or (f"{args.model} @ τ=[{args.tau_lo:.2f}, "
                            f"{args.tau_hi:.2f}]")
    plot_pit_and_reliability(pit, anchors, args.output, title=title)


if __name__ == "__main__":
    main()
