"""Tests for identity forecast (perfect prediction) behavior.

These are UNIT tests using synthetic fixtures - no trained artifacts required.
"""

import numpy as np
import pytest

from src.config import CONFIG
from src.eval_capacity import compute_capacities, compute_utilization_metrics


class TestIdentityForecast:
    """If predictions == ground truth, metrics should match Oracle."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for identity tests (independent of saved predictions)."""
        np.random.seed(42)
        T_eff = 100
        num_links = 10
        margin = CONFIG['capacity_margin']

        # Generate simple sinusoidal traffic
        t = np.arange(T_eff)
        base = 10.0
        Y_true = np.column_stack([
            base + 5 * np.sin(2 * np.pi * t / 24 + i * np.pi / num_links)
            for i in range(num_links)
        ])
        Y_true = np.maximum(Y_true, 1.0)

        return {
            'Y_true': Y_true,
            'T_eff': T_eff,
            'num_links': num_links,
            'margin': margin
        }

    def test_identity_forecast_matches_oracle_metrics(self, synthetic_data):
        """
        If predictions == ground truth, capacity metrics should match oracle.

        This tests the FULL pipeline (compute_capacities + compute_utilization_metrics),
        not just capacity equality by construction.
        """
        Y_true = synthetic_data['Y_true'].copy()
        margin = synthetic_data['margin']

        # Create identity forecast (predictions = ground truth)
        identity_predictions = {
            'Y_true': Y_true,
            'Y_pred_sarima': Y_true.copy(),
            'Y_pred_lstm': Y_true.copy(),
            'T_eff': synthetic_data['T_eff'],
            'num_links': synthetic_data['num_links']
        }

        capacities = compute_capacities(identity_predictions, margin)
        metrics = compute_utilization_metrics(Y_true, capacities)

        # With identity forecast, all models should behave like oracle
        assert metrics['SARIMA']['f_over_mean'] == 0, \
            f"Identity SARIMA has overload: {metrics['SARIMA']['f_over_mean']}"
        assert metrics['LSTM']['f_over_mean'] == 0, \
            f"Identity LSTM has overload: {metrics['LSTM']['f_over_mean']}"
        assert metrics['Oracle']['f_over_mean'] == 0, \
            f"Oracle has overload: {metrics['Oracle']['f_over_mean']}"

    def test_identity_forecast_capacities_equal(self, synthetic_data):
        """With identity forecast, all model capacities should equal Oracle."""
        Y_true = synthetic_data['Y_true'].copy()
        margin = synthetic_data['margin']

        identity_predictions = {
            'Y_true': Y_true,
            'Y_pred_sarima': Y_true.copy(),
            'Y_pred_lstm': Y_true.copy(),
            'T_eff': synthetic_data['T_eff'],
            'num_links': synthetic_data['num_links']
        }

        capacities = compute_capacities(identity_predictions, margin)

        assert np.allclose(capacities['SARIMA'], capacities['Oracle']), \
            "Identity SARIMA capacity != Oracle capacity"
        assert np.allclose(capacities['LSTM'], capacities['Oracle']), \
            "Identity LSTM capacity != Oracle capacity"

    def test_identity_u_max_equals_expected(self, synthetic_data):
        """With identity forecast, u_max should be exactly 1/alpha for all links."""
        Y_true = synthetic_data['Y_true'].copy()
        margin = synthetic_data['margin']

        identity_predictions = {
            'Y_true': Y_true,
            'Y_pred_sarima': Y_true.copy(),
            'Y_pred_lstm': Y_true.copy(),
            'T_eff': synthetic_data['T_eff'],
            'num_links': synthetic_data['num_links']
        }

        capacities = compute_capacities(identity_predictions, margin)
        metrics = compute_utilization_metrics(Y_true, capacities)

        expected = 1.0 / margin
        tol = 1e-6

        for model in ['SARIMA', 'LSTM', 'Oracle']:
            u_max_mean = metrics[model]['u_max_mean']
            assert abs(u_max_mean - expected) < tol, \
                f"{model} u_max_mean={u_max_mean:.6f} != expected={expected:.6f}"
