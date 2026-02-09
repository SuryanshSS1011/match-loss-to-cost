"""Operational metrics for capacity-planning evaluation.

These are the *headline* metrics for the Provision-Aware paper (per Rule 1 in
CLAUDE.md): operational cost, SLA-violation rate, overload rate. RMSE/MAE are
reported alongside but are not the headline.

All inputs are numpy arrays of shape (T, num_links). All outputs are floats
(scalar means over all (t, link) cells), unless documented otherwise.
"""

from __future__ import annotations

import numpy as np


def capacity_from_predictions(y_pred: np.ndarray, margin: float = 1.1) -> np.ndarray:
    """Provisioned capacity per link from predictions: c_ℓ = α · max_t ŷ_ℓ(t).

    Matches the paper definition (eval_capacity.py uses the same convention).

    Args:
        y_pred: predicted link loads, shape (T, num_links).
        margin: safety margin α. Default 1.1.

    Returns:
        Capacity per link, shape (num_links,).
    """
    if y_pred.ndim != 2:
        raise ValueError(f"expected (T, num_links); got shape {y_pred.shape}")
    return margin * np.nanmax(y_pred, axis=0)


def overload_rate(y_true: np.ndarray, capacity: np.ndarray) -> float:
    """Fraction of (t, link) cells with y_true > capacity (= utilization > 1)."""
    util = y_true / np.maximum(capacity, 1e-12)
    return float(np.mean(util > 1.0))


def sla_violation_rate(y_true: np.ndarray, capacity: np.ndarray) -> float:
    """Alias of overload_rate. SLA violation == link overload at provisioned capacity."""
    return overload_rate(y_true, capacity)


def over_provisioning_cost(y_true: np.ndarray, capacity: np.ndarray) -> float:
    """Sum of headroom: Σ_{t,ℓ} max(c_ℓ − y_ℓ(t), 0).

    Units = Mbps · time-steps. Convert externally if Gbps · hours desired.
    """
    headroom = np.clip(capacity[None, :] - y_true, 0.0, None)
    return float(headroom.sum())


def asymmetric_op_cost(
    y_true: np.ndarray,
    capacity: np.ndarray,
    alpha: float = 5.0,
    beta: float = 1.0,
) -> float:
    """Operational cost: α · Σ max(y − c, 0) + β · Σ max(c − y, 0).

    α weights under-provisioning (SLA violation), β weights over-provisioning.

    Args:
        y_true:   ground-truth link loads, shape (T, num_links).
        capacity: provisioned capacity per link, shape (num_links,).
        alpha:    under-provisioning weight. Default 5.0.
        beta:     over-provisioning weight. Default 1.0.

    Returns:
        Total operational cost as a float (sum, not mean).
    """
    if y_true.ndim != 2 or capacity.ndim != 1:
        raise ValueError(
            f"expected y_true (T, num_links) and capacity (num_links,); "
            f"got {y_true.shape} and {capacity.shape}"
        )
    if y_true.shape[1] != capacity.shape[0]:
        raise ValueError(
            f"link-axis mismatch: y_true has {y_true.shape[1]} links, "
            f"capacity has {capacity.shape[0]}"
        )
    diff = y_true - capacity[None, :]
    under = np.clip(diff, 0.0, None).sum()
    over = np.clip(-diff, 0.0, None).sum()
    return float(alpha * under + beta * over)


def operational_metrics(
    y_true: np.ndarray,
    capacity: np.ndarray,
    alpha: float = 5.0,
    beta: float = 1.0,
) -> dict:
    """Bundle of operational metrics, all scalar floats."""
    return {
        "overload_rate": overload_rate(y_true, capacity),
        "sla_violation_rate": sla_violation_rate(y_true, capacity),
        "over_provisioning_cost": over_provisioning_cost(y_true, capacity),
        "asymmetric_op_cost": asymmetric_op_cost(y_true, capacity, alpha, beta),
        "u_max_mean": float(np.mean(np.nanmax(y_true / np.maximum(capacity, 1e-12),
                                              axis=0))),
        "u_max_max": float(np.nanmax(y_true / np.maximum(capacity, 1e-12))),
    }
