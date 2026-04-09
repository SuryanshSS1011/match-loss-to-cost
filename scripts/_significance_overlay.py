"""Shared helper: significance overlay for Pareto plots.

Given a set of per-(ratio, model) per-seed values, run paired Wilcoxon vs
a reference model with Holm-Bonferroni adjustment and return a dict that
the plotters can use to mark significant vs non-significant points.

Inputs (`rows`) follow the runner's aggregated JSON shape, restricted to
one ratio cell:
    {model_name: {"values": [v_seed0, v_seed1, ...]}}

We return:
    {model_name: bool}   — True iff `model_name` strictly beats `reference`
    on the chosen metric at FWER `alpha` (Holm-adjusted), under
    `lower_is_better`. Reference itself maps to None.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.significance import (  # noqa: E402
    pairwise_significance_table,
)


def significance_vs_reference(
    rows: dict[str, dict],
    reference: str,
    *,
    lower_is_better: bool = True,
    alpha: float = 0.05,
) -> dict[str, Optional[bool]]:
    """Run paired Wilcoxon (Holm-corrected) vs `reference`.

    `rows[model]` must carry a `"values"` list of per-seed metric values,
    aligned across models. Models that don't have `values` (e.g. constants
    in a stub aggregated dict) are silently skipped.

    Returns: {model: True | False | None}. `None` for the reference itself.
    """
    if reference not in rows:
        raise KeyError(
            f"reference {reference!r} not in rows {list(rows.keys())}"
        )
    values_by_model: dict[str, list[float]] = {}
    for model, block in rows.items():
        vals = block.get("values")
        if isinstance(vals, list) and len(vals) >= 2:
            values_by_model[model] = list(vals)
    if reference not in values_by_model:
        raise ValueError(
            f"reference {reference!r} has no per-seed values; "
            "cannot run a paired test"
        )
    if len(values_by_model) < 2:
        # Only the reference has values — nothing to compare.
        return {model: None for model in rows}

    table = pairwise_significance_table(
        values_by_model,
        lower_is_better=lower_is_better,
        alpha=alpha,
        reference=reference,
    )

    out: dict[str, Optional[bool]] = {model: None for model in rows}
    for entry in table:
        a, b = entry["model_a"], entry["model_b"]
        # The reference is always model_b in `reference=`.
        if b != reference:
            continue
        # `reject=True` AND mean_a < mean_b (under lower_is_better) means
        # `a` strictly beats `b`. The pairwise table already encodes
        # direction via the chosen alternative.
        out[a] = bool(entry["reject"])
    return out
