"""Statistical-significance protocol for the Provision-Aware paper.

Per Rule 4 in CLAUDE.md: every modeling claim needs >=5 seeds + a paired
test, with Holm-Bonferroni adjustment for multiple comparisons; appendix
gets a Demsar 2006 critical-difference diagram across (dataset, model)
cells.

Public API (re-exported from `src.evaluation`):
    paired_wilcoxon(a, b, alternative='two-sided')
    holm_bonferroni(pvals, alpha=0.05)
    pairwise_significance_table(values_by_model, ...)
    critical_difference_diagram(ranks_by_dataset, save_path, ...)

The first three are pure scipy.stats; the fourth uses matplotlib.

Inputs are always paired across the same set of seeds (or datasets, for
the CD diagram). The functions assume **lower-is-better** by default — the
typical case for RMSE / MAE / operational cost. Pass `lower_is_better=False`
to flip for accuracy-style metrics.

References:
- F. Wilcoxon, "Individual comparisons by ranking methods", Biometrics 1945.
- S. Holm, "A simple sequentially rejective multiple test procedure",
  Scand. J. Statist. 1979.
- J. Demsar, "Statistical comparisons of classifiers over multiple data
  sets", JMLR 2006.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. paired Wilcoxon
# ---------------------------------------------------------------------------

def paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> dict:
    """Wilcoxon signed-rank test on the paired differences a - b.

    Args:
        a, b: equal-length sequences of paired observations (e.g. seed-aligned
            metric values for two models).
        alternative: 'two-sided', 'less' (a < b), or 'greater' (a > b).

    Returns:
        {statistic, pvalue, n_pairs, mean_diff, n_zero_diff}. `n_pairs` is the
        number of *non-zero* differences actually used by the test (zero
        differences are dropped per Wilcoxon's convention).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"shape mismatch: a {a_arr.shape} vs b {b_arr.shape}"
        )
    if a_arr.ndim != 1:
        raise ValueError(f"a, b must be 1-D; got shape {a_arr.shape}")
    if a_arr.size < 2:
        raise ValueError(f"need at least 2 paired observations; got {a_arr.size}")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(f"unknown alternative {alternative!r}")

    diff = a_arr - b_arr
    n_zero = int(np.sum(diff == 0.0))
    n_pairs = int(diff.size - n_zero)

    if n_pairs == 0:
        # All differences are zero: no signal, p-value is 1.
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "n_pairs": 0,
            "n_zero_diff": n_zero,
            "mean_diff": 0.0,
        }

    # zero_method='wilcox' drops zeros (default; spelled out for clarity).
    res = stats.wilcoxon(
        a_arr, b_arr, alternative=alternative, zero_method="wilcox"
    )
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n_pairs": n_pairs,
        "n_zero_diff": n_zero,
        "mean_diff": float(diff.mean()),
    }


# ---------------------------------------------------------------------------
# 2. Holm-Bonferroni
# ---------------------------------------------------------------------------

def holm_bonferroni(
    pvals: Sequence[float],
    alpha: float = 0.05,
) -> dict:
    """Step-down Holm-Bonferroni multiple-testing correction.

    Returns the family-wise error rate adjusted p-values and a reject mask
    at level alpha. The output preserves the input ordering.

    Args:
        pvals: raw p-values from a family of paired tests.
        alpha: family-wise error rate target.

    Returns:
        {adjusted: list[float], reject: list[bool], alpha: float, m: int}.
    """
    p = np.asarray(pvals, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"pvals must be 1-D; got shape {p.shape}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must lie in [0, 1]")

    m = p.size
    if m == 0:
        return {"adjusted": [], "reject": [], "alpha": alpha, "m": 0}

    order = np.argsort(p)              # ascending
    p_sorted = p[order]
    # Holm step-down: adj_i = max_{j<=i} ((m - j) * p_(j)), capped at 1.
    factors = (m - np.arange(m)).astype(np.float64)
    raw = factors * p_sorted
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]  # running min reversed
    # We want running max forward, not running min reverse — fix:
    adj_sorted = np.maximum.accumulate(raw)
    adj_sorted = np.minimum(adj_sorted, 1.0)

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adj_sorted
    reject = adjusted < alpha
    return {
        "adjusted": [float(x) for x in adjusted],
        "reject": [bool(x) for x in reject],
        "alpha": float(alpha),
        "m": int(m),
    }


# ---------------------------------------------------------------------------
# 3. pairwise table for a (model, metric) family
# ---------------------------------------------------------------------------

def pairwise_significance_table(
    values_by_model: dict[str, Sequence[float]],
    *,
    lower_is_better: bool = True,
    alpha: float = 0.05,
    reference: Optional[str] = None,
) -> list[dict]:
    """All-pairs paired-Wilcoxon table with Holm correction.

    Args:
        values_by_model: {model_name: [val_seed0, val_seed1, ...]}. All
            value lists must have the same length and be aligned by seed.
        lower_is_better: if True (default), the alternative for each pair
            (a, b) is "a < b" — i.e. testing whether `a` beats `b`.
        alpha: family-wise error rate.
        reference: if given, only test pairs (model, reference) for each
            other model. Reduces the family from C(M, 2) to (M-1) and is
            the standard "vs MSE baseline" or "vs PatchTST" framing.

    Returns:
        A list of dicts with keys:
            model_a, model_b, mean_a, mean_b, mean_diff, statistic,
            pvalue, pvalue_holm, reject.
        Sorted by raw pvalue ascending so the most-significant pair is first.
    """
    models = list(values_by_model.keys())
    if len(models) < 2:
        raise ValueError(f"need >=2 models; got {len(models)}")
    seed_lengths = {len(v) for v in values_by_model.values()}
    if len(seed_lengths) != 1:
        raise ValueError(
            f"all models must have the same #seeds; got {seed_lengths}"
        )

    if reference is not None and reference not in models:
        raise ValueError(f"reference={reference!r} not in models {models}")

    # Build the unordered pair list.
    if reference is not None:
        pairs = [(m, reference) for m in models if m != reference]
    else:
        pairs = [
            (models[i], models[j])
            for i in range(len(models))
            for j in range(i + 1, len(models))
        ]

    # `lower_is_better=True` ⇒ test alternative "a < b" (a beats b).
    alt = "less" if lower_is_better else "greater"

    raw = []
    for a_name, b_name in pairs:
        a = values_by_model[a_name]
        b = values_by_model[b_name]
        res = paired_wilcoxon(a, b, alternative=alt)
        raw.append({
            "model_a": a_name,
            "model_b": b_name,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(res["mean_diff"]),
            "statistic": res["statistic"],
            "pvalue": res["pvalue"],
            "n_pairs": res["n_pairs"],
        })

    pvals = [r["pvalue"] for r in raw]
    holm = holm_bonferroni(pvals, alpha=alpha)
    for r, adj, rej in zip(raw, holm["adjusted"], holm["reject"]):
        r["pvalue_holm"] = adj
        r["reject"] = rej

    raw.sort(key=lambda r: r["pvalue"])
    return raw


# ---------------------------------------------------------------------------
# 4. Critical-difference diagram (Demsar 2006)
# ---------------------------------------------------------------------------

def _compute_average_ranks(
    values_by_model: dict[str, Sequence[float]],
    lower_is_better: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Return (model_names, average_ranks) over the columns of the input.

    Each column (= dataset / cell) is ranked independently 1..M with ties
    averaged; the per-model average across columns is returned.
    """
    models = list(values_by_model.keys())
    arr = np.asarray([values_by_model[m] for m in models], dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"each model's values must be 1-D; stacked shape {arr.shape}"
        )
    M, K = arr.shape  # M models, K datasets
    if K < 2:
        raise ValueError(f"need >=2 datasets/cells for ranking; got K={K}")

    if lower_is_better:
        # rankdata gives 1 to the smallest. We want 1 to the best, so this
        # already does the right thing for lower-is-better.
        ranks = np.apply_along_axis(stats.rankdata, 0, arr)
    else:
        ranks = np.apply_along_axis(lambda c: stats.rankdata(-c), 0, arr)
    avg_ranks = ranks.mean(axis=1)
    return models, avg_ranks


# Studentized-range constants for Nemenyi (Demsar 2006, Table 5 row 0.05).
# Index = number of classifiers M; q_alpha = q_{0.05, M, inf} / sqrt(2).
# Source: Demsar 2006 Table 5; covers M in 2..10 which is plenty for our
# minimum-viable + DLinear + PatchTST + iTransformer + DCRNN stack.
_NEMENYI_Q_05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def _critical_difference(M: int, K: int, alpha: float = 0.05) -> float:
    """Nemenyi CD = q_alpha * sqrt(M*(M+1) / (6*K))."""
    if alpha != 0.05:
        raise NotImplementedError("only alpha=0.05 is tabulated")
    if M not in _NEMENYI_Q_05:
        raise ValueError(
            f"M={M} not in tabulated range 2..10; extend _NEMENYI_Q_05"
        )
    q = _NEMENYI_Q_05[M]
    return q * math.sqrt(M * (M + 1) / (6.0 * K))


def critical_difference_diagram(
    values_by_model: dict[str, Sequence[float]],
    save_path: str,
    *,
    lower_is_better: bool = True,
    alpha: float = 0.05,
    title: str = "",
) -> dict:
    """Demsar-2006 critical-difference diagram.

    Args:
        values_by_model: {model_name: [metric_per_dataset_or_cell]}. Each
            list is one row; columns are the datasets/cells across which we
            rank.
        save_path: where to write the PNG.
        lower_is_better: ranking direction.
        alpha: significance level (only 0.05 supported for now).
        title: optional plot title.

    Returns:
        {ranks: {model: avg_rank}, cd: float, M: int, K: int} for tests.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os as _os

    models, avg_ranks = _compute_average_ranks(values_by_model, lower_is_better)
    M = len(models)
    K = len(next(iter(values_by_model.values())))
    cd = _critical_difference(M, K, alpha=alpha)

    # Sort by ascending rank (best first → leftmost, since the axis is inverted).
    order = np.argsort(avg_ranks)
    models_sorted = [models[i] for i in order]
    ranks_sorted = avg_ranks[order]
    n = len(ranks_sorted)

    # --- Maximal cliques (Demsar 2006): every *maximal* run of models whose
    # rank span is within CD. Unlike a greedy consecutive merge, we scan every
    # start i and take the longest run [i..j] with ranks[j]-ranks[i] <= cd,
    # then drop runs that are fully contained in another (keep only maximal,
    # length >= 2). This is the canonical "connect models that are not
    # significantly different" recipe.
    runs: list[tuple[int, int]] = []
    for i in range(n):
        j = i
        while j + 1 < n and ranks_sorted[j + 1] - ranks_sorted[i] <= cd + 1e-9:
            j += 1
        if j > i:
            runs.append((i, j))
    cliques: list[tuple[int, int]] = []
    for (a, b) in runs:
        if not any(a2 <= a and b <= b2 and (a2, b2) != (a, b) for (a2, b2) in runs):
            cliques.append((a, b))

    # --- Layout in data coords. Axis spans rank 1..M, drawn inverted so the
    # best (lowest) rank is on the RIGHT — the standard Demsar orientation where
    # the right half of models fans right and the left half fans left.
    # Vertical zones (top→bottom), no overlap:
    #   CD ruler  →  rank axis (y=0, ticks above)  →  clique bars just below
    #   →  leader lines fanning to model labels, each on its OWN row.
    #
    # Split: the best half (lowest ranks) go to the RIGHT column, the worst half
    # to the LEFT column. Each model gets a unique row so tied ranks never share
    # a label line; their vertical connectors branch at the top but land on
    # distinct rows, staying legible.
    n_right = n // 2                      # best ranks → right column
    n_left = n - n_right                  # worst ranks → left column
    rows_per_side = max(n_left, n_right)
    # Clean, non-overlapping y-bands below the axis (y=0), top→bottom:
    #   clique bars  →  tie-fork notches  →  first label row.
    n_cliques = max(len(cliques), 1)
    clique_y0 = -0.18                     # first (topmost) clique bar
    clique_dy = 0.16
    clique_band = clique_y0 - clique_dy * (n_cliques - 1)   # lowest bar
    fork_y = clique_band - 0.14           # tie forks sit just BELOW all bars
    row_top = fork_y - 0.34               # first label row, below the forks
    row_gap = 0.60
    label_bottom = row_top - row_gap * (rows_per_side - 1)
    y_min = label_bottom - 0.5
    y_max = 2.55                           # headroom for the centred title

    longest = max((len(m) for m in models_sorted), default=8)
    fig_w = max(8.5, 5.5 + 0.13 * longest + 0.35 * M)
    fig_h = 2.6 + 0.42 * rows_per_side
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    pad = 1.1
    ax.set_xlim(M + pad, 1 - pad)         # inverted: rank 1 on the right... no:
    # set_xlim(hi, lo) with hi>lo flips the axis so SMALL ranks render on the
    # RIGHT. Labels live in the pad beyond the rank-1 (right) and rank-M (left)
    # ends, outside where any leader line runs.
    ax.set_ylim(y_min, y_max)
    ax.axis("off")

    # Rank axis: clean line, ticks + numbers above it, italic caption higher up.
    ax.plot([1, M], [0, 0], color="black", linewidth=1.3, zorder=1)
    for r in range(1, M + 1):
        ax.plot([r, r], [0, 0.12], color="black", linewidth=1.0, zorder=1)
        ax.text(r, 0.24, str(r), ha="center", va="bottom", fontsize=11)
    ax.text((1 + M) / 2.0, 0.62,
            "Average rank  (lower = better)" if lower_is_better
            else "Average rank  (higher = better)",
            ha="center", va="bottom", fontsize=10.5, style="italic")

    # Leader lines + labels. Right column = the n_right best (smallest) ranks,
    # ordered so the very best sits on the TOP row nearest the axis.
    edge_left, edge_right = float(M), 1.0   # left column ends at rank M, right at 1
    # Tied-rank handling: give each model a tiny unique x-offset for the vertical
    # connector's top so coincident verticals separate visibly.
    from collections import defaultdict
    tie_groups: dict[float, list[int]] = defaultdict(list)
    for idx, rr in enumerate(ranks_sorted):
        tie_groups[round(float(rr), 6)].append(idx)
    tie_offset = {}
    for rr, idxs in tie_groups.items():
        if len(idxs) == 1:
            tie_offset[idxs[0]] = 0.0
        else:
            spread = 0.07
            for k, idx in enumerate(idxs):
                tie_offset[idx] = (k - (len(idxs) - 1) / 2.0) * spread

    for i, (m, r) in enumerate(zip(models_sorted, ranks_sorted)):
        on_right = i < n_right
        if on_right:
            row = i
            edge = edge_right
        else:
            row = i - n_right
            edge = edge_left
        y = row_top - row_gap * row
        # Tied-rank connector: the vertical leaves the axis exactly at the true
        # rank `r` (so it meets its tick honestly), then forks to a nudged x a
        # short way down before continuing to the row. Untied ranks go straight.
        off = tie_offset[i]
        if abs(off) < 1e-9:
            ax.plot([r, r], [0.0, y], color="black", linewidth=0.9, zorder=2)
            x_join = r
        else:
            # fork below the clique band so the notch never touches a bar.
            ax.plot([r, r], [0.0, fork_y], color="black", linewidth=0.9, zorder=2)
            ax.plot([r, r + off], [fork_y, fork_y], color="black",
                    linewidth=0.9, zorder=2)
            ax.plot([r + off, r + off], [fork_y, y], color="black",
                    linewidth=0.9, zorder=2)
            x_join = r + off
        # horizontal out to the column edge, then the label just beyond it.
        ax.plot([x_join, edge], [y, y], color="black", linewidth=0.9, zorder=2)
        ty = y - 0.05
        if on_right:
            ax.text(edge - 0.16, ty, f"({r:.2f})  {m}",
                    ha="left", va="top", fontsize=10.5)
        else:
            ax.text(edge + 0.16, ty, f"{m}  ({r:.2f})",
                    ha="right", va="top", fontsize=10.5)

    # Clique bars: thick, between the axis and the labels, stacked downward.
    for k_idx, (a, b) in enumerate(cliques):
        y = clique_y0 - clique_dy * k_idx
        r_lo, r_hi = ranks_sorted[a], ranks_sorted[b]
        ax.plot([r_lo, r_hi], [y, y], color="black", linewidth=5.0,
                solid_capstyle="round", zorder=3)

    # CD ruler at the top — clearly separated, with end caps. Drawn at the
    # worst-rank (left) side so it never overlaps the best models' label rows.
    cd_y = 1.50
    cd_x1 = float(M)
    cd_x0 = M - cd
    ax.plot([cd_x0, cd_x1], [cd_y, cd_y], color="black", linewidth=1.6)
    for xc in (cd_x0, cd_x1):
        ax.plot([xc, xc], [cd_y - 0.07, cd_y + 0.07], color="black", linewidth=1.6)
    ax.text((cd_x0 + cd_x1) / 2.0, cd_y + 0.12, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=10.5)

    if title:
        # Center over the rank axis (the visual centre of the figure), not the
        # axes bounding box — the asymmetric label pads otherwise shift a
        # set_title()/suptitle() off to one side.
        ax.text((1 + M) / 2.0, y_max - 0.08, title,
                ha="center", va="top", fontsize=13, fontweight="bold")

    _os.makedirs(_os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "ranks": {m: float(r) for m, r in zip(models, avg_ranks)},
        "cd": float(cd),
        "M": int(M),
        "K": int(K),
        "cliques": [(models_sorted[a], models_sorted[b]) for (a, b) in cliques],
    }
