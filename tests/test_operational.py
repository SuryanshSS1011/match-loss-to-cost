"""Unit tests for src/evaluation/operational.py.

Pure numpy. These metrics are the *headline* of the Provision-Aware paper
(per Rule 1 in CLAUDE.md), so the contract has to be airtight before the
cloud sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    asymmetric_op_cost,
    capacity_from_predictions,
    operational_metrics,
    overload_rate,
    over_provisioning_cost,
    sla_violation_rate,
)


class TestCapacityFromPredictions:
    def test_matches_margin_times_max(self):
        rng = np.random.default_rng(0)
        y_pred = rng.uniform(0, 100, size=(50, 7)).astype(np.float32)
        cap = capacity_from_predictions(y_pred, margin=1.1)
        assert cap.shape == (7,)
        assert np.allclose(cap, 1.1 * y_pred.max(axis=0), rtol=1e-6)

    def test_default_margin_is_paper_default(self):
        # Paper convention: α = 1.1 (10 % safety margin).
        y_pred = np.array([[1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
        cap = capacity_from_predictions(y_pred)
        assert np.allclose(cap, 1.1 * np.array([3.0, 2.0]), rtol=1e-6)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            capacity_from_predictions(np.zeros(10))


class TestOverloadRate:
    def test_zero_when_capacity_dominates(self):
        y = np.array([[1.0, 2.0], [3.0, 1.0]])
        cap = np.array([10.0, 10.0])  # dwarfs y
        assert overload_rate(y, cap) == 0.0

    def test_one_when_capacity_zero(self):
        y = np.array([[1.0, 2.0], [3.0, 1.0]])
        cap = np.array([0.0, 0.0])
        # Every (t, link) cell is over its capacity.
        assert overload_rate(y, cap) == 1.0

    def test_partial_overload(self):
        y = np.array([[1.0, 5.0], [4.0, 2.0]])
        cap = np.array([3.0, 3.0])
        # cell-by-cell overload mask:
        #   y[0,0]=1 vs 3 → no, y[0,1]=5 vs 3 → yes,
        #   y[1,0]=4 vs 3 → yes, y[1,1]=2 vs 3 → no.
        # 2 of 4 cells overloaded.
        assert overload_rate(y, cap) == pytest.approx(0.5)

    def test_sla_alias_matches(self):
        y = np.array([[1.0, 5.0], [4.0, 2.0]])
        cap = np.array([3.0, 3.0])
        assert sla_violation_rate(y, cap) == overload_rate(y, cap)


class TestOverProvisioningCost:
    def test_zero_when_y_dominates(self):
        y = np.array([[10.0, 10.0], [10.0, 10.0]])
        cap = np.array([1.0, 1.0])  # always under y, no headroom
        assert over_provisioning_cost(y, cap) == pytest.approx(0.0)

    def test_sums_headroom(self):
        y = np.array([[1.0, 2.0], [3.0, 1.0]])
        cap = np.array([4.0, 5.0])
        # headroom = max(c - y, 0):
        #   t=0: (3, 3); t=1: (1, 4)  → sum = 11.
        assert over_provisioning_cost(y, cap) == pytest.approx(11.0)


class TestAsymmetricOpCost:
    def test_zero_when_capacity_equals_truth(self):
        # If c == y at every cell, neither under nor over → cost is zero.
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        cap = y.max(axis=0)  # [3.0, 4.0]
        # At t=1 we hit c exactly (no over, no under), at t=0 we have headroom
        # (over). So this is non-zero in general; build a cleaner construction:
        y_flat = np.array([[1.0, 1.0]])
        c_flat = np.array([1.0, 1.0])
        cost = asymmetric_op_cost(y_flat, c_flat, alpha=5.0, beta=1.0)
        assert cost == pytest.approx(0.0)

    def test_decomposes_into_under_and_over(self):
        # Construct y and c so that under and over contributions are both
        # known. y has one cell above c (under-provisioned) and one below.
        y = np.array([[10.0, 1.0]])
        cap = np.array([4.0, 4.0])
        # under_sum = max(y - c, 0).sum() = max(6, -3, ...) = 6
        # over_sum  = max(c - y, 0).sum() = 0 + 3 = 3
        cost = asymmetric_op_cost(y, cap, alpha=5.0, beta=1.0)
        assert cost == pytest.approx(5.0 * 6.0 + 1.0 * 3.0)

    def test_alpha_dominates_under_provisioning(self):
        # Pure under-prediction. Doubling α should double the cost.
        y = np.array([[10.0]])
        cap = np.array([5.0])
        c1 = asymmetric_op_cost(y, cap, alpha=1.0, beta=1.0)
        c2 = asymmetric_op_cost(y, cap, alpha=2.0, beta=1.0)
        assert c2 == pytest.approx(2.0 * c1)

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            asymmetric_op_cost(np.zeros(10), np.zeros(10))
        with pytest.raises(ValueError):
            asymmetric_op_cost(np.zeros((5, 3)), np.zeros(2))


class TestOperationalMetricsBundle:
    def test_keys_present(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(0, 10, size=(20, 4)).astype(np.float32)
        cap = capacity_from_predictions(y)
        m = operational_metrics(y, cap, alpha=5.0, beta=1.0)
        for key in (
            "overload_rate",
            "sla_violation_rate",
            "over_provisioning_cost",
            "asymmetric_op_cost",
            "u_max_mean",
            "u_max_max",
        ):
            assert key in m, f"missing key: {key}"

    def test_oracle_capacity_zero_overload(self):
        # If we use the *oracle* capacity (margin · max y), no cell should
        # overload by construction.
        rng = np.random.default_rng(2)
        y = rng.uniform(0, 10, size=(50, 6)).astype(np.float32)
        cap = capacity_from_predictions(y, margin=1.1)
        m = operational_metrics(y, cap, alpha=5.0, beta=1.0)
        assert m["overload_rate"] == 0.0
        assert m["sla_violation_rate"] == 0.0
        # Max utilisation should be ≤ 1/margin (= 1/1.1 ≈ 0.909).
        assert m["u_max_max"] <= 1.0 / 1.1 + 1e-6
