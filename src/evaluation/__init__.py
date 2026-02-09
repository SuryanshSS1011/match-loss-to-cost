"""Evaluation utilities for the Provision-Aware paper.

Public API:
    --- operational metrics (operational.py) ---
    asymmetric_op_cost(y_true, capacity, alpha, beta) -> float
    sla_violation_rate(y_true, capacity)              -> float
    overload_rate(y_true, capacity)                   -> float
    over_provisioning_cost(y_true, capacity)          -> float
    capacity_from_predictions(y_pred, margin)         -> ndarray
    operational_metrics(y_true, capacity, alpha, beta) -> dict

    --- statistical significance (significance.py) ---
    paired_wilcoxon(a, b, alternative)                -> dict
    holm_bonferroni(pvals, alpha)                     -> dict
    pairwise_significance_table(values_by_model, ...) -> list[dict]
    critical_difference_diagram(values_by_model, save_path, ...) -> dict
"""

from .operational import (
    asymmetric_op_cost,
    capacity_from_predictions,
    operational_metrics,
    overload_rate,
    over_provisioning_cost,
    sla_violation_rate,
)
from .significance import (
    critical_difference_diagram,
    holm_bonferroni,
    paired_wilcoxon,
    pairwise_significance_table,
)

__all__ = [
    "asymmetric_op_cost",
    "capacity_from_predictions",
    "operational_metrics",
    "overload_rate",
    "over_provisioning_cost",
    "sla_violation_rate",
    "paired_wilcoxon",
    "holm_bonferroni",
    "pairwise_significance_table",
    "critical_difference_diagram",
]
