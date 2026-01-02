"""Tests for Oracle capacity invariants.

These are INTEGRATION tests that require trained model artifacts.
Run the full pipeline first, or skip with: pytest -m "not integration"
"""

import os
import numpy as np
import pytest

from src.config import RESULTS_DIR

# Auto-skip if artifacts are missing
pytestmark = pytest.mark.integration
if not os.path.exists(os.path.join(RESULTS_DIR, 'lstm_predictions.npz')):
    pytest.skip("Integration artifacts not found (run pipeline first)", allow_module_level=True)


class TestOracleInvariants:
    """Oracle capacity must satisfy mathematical invariants."""

    def test_oracle_has_zero_overload(self, predictions, capacities, margin):
        """Oracle must have zero overload events (utilization > 1 never occurs)."""
        Y_true = predictions['Y_true']
        C_oracle = capacities['Oracle']

        utilization = Y_true / np.maximum(C_oracle, 1e-6)
        overload_count = int(np.nansum(utilization > 1.0))

        assert overload_count == 0, \
            f"Oracle has {overload_count} overload events (should be 0)"

    def test_oracle_u_max_bounded_by_inverse_alpha(self, predictions, capacities, margin):
        """Oracle u_max must be <= 1/alpha for all links."""
        Y_true = predictions['Y_true']
        C_oracle = capacities['Oracle']

        utilization = Y_true / np.maximum(C_oracle, 1e-6)
        u_max_per_link = np.nanmax(utilization, axis=0)

        expected = 1.0 / margin
        tol = 1e-6

        assert np.all(u_max_per_link <= expected + tol), \
            f"Oracle u_max > 1/alpha ({expected:.6f}). " \
            f"Max observed: {np.max(u_max_per_link):.6f}"

        # Informational: print actual values
        print(f"\nOracle u_max: mean={np.mean(u_max_per_link):.6f}, "
              f"max={np.max(u_max_per_link):.6f}, expected={expected:.6f}")


class TestPerLinkMetrics:
    """Verify per-link metrics have correct shapes and bounds."""

    def test_per_link_shapes(self, predictions, capacity_metrics):
        """Per-link metric arrays should have correct shape."""
        num_links = predictions['num_links']

        for model_name, m in capacity_metrics.items():
            u_max = np.array(m['u_max_per_link'])
            f_over = np.array(m['f_over_per_link'])

            assert u_max.shape == (num_links,), \
                f"{model_name} u_max shape {u_max.shape} != ({num_links},)"
            assert f_over.shape == (num_links,), \
                f"{model_name} f_over shape {f_over.shape} != ({num_links},)"

    def test_per_link_bounds(self, capacity_metrics):
        """Per-link metrics must be within valid bounds."""
        for model_name, m in capacity_metrics.items():
            u_max = np.array(m['u_max_per_link'])
            f_over = np.array(m['f_over_per_link'])

            # f_over must be in [0, 1]
            assert np.all(f_over >= 0), f"{model_name} f_over < 0"
            assert np.all(f_over <= 1), f"{model_name} f_over > 1"

            # u_max must be non-negative
            assert np.all(u_max >= 0), f"{model_name} u_max < 0"

    def test_oracle_per_link_f_over_zero(self, capacity_metrics, margin):
        """Oracle f_over must be 0 for all links."""
        oracle_f_over = np.array(capacity_metrics['Oracle']['f_over_per_link'])

        assert np.allclose(oracle_f_over, 0), \
            f"Oracle f_over_per_link != 0: {oracle_f_over[oracle_f_over > 0]}"

    def test_oracle_per_link_u_max_bounded(self, capacity_metrics, margin):
        """Oracle u_max must be <= 1/alpha for all links."""
        oracle_u_max = np.array(capacity_metrics['Oracle']['u_max_per_link'])
        expected = 1.0 / margin

        assert np.all(oracle_u_max <= expected + 1e-6), \
            f"Oracle u_max_per_link > 1/alpha for links: " \
            f"{np.where(oracle_u_max > expected + 1e-6)[0]}"
