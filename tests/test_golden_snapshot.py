"""Golden data snapshot test for regression testing."""

import os
import numpy as np
import pytest

from config import CONFIG
from eval_capacity import compute_capacities, compute_utilization_metrics
from utils import load_json


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


class TestGoldenData:
    """Run full pipeline on small deterministic dataset, check headline numbers."""

    @pytest.fixture
    def golden_data(self):
        """Load golden dataset."""
        golden_path = os.path.join(FIXTURES_DIR, 'golden_data.npz')
        if not os.path.exists(golden_path):
            pytest.skip("Golden data not found. Run scripts/generate_golden_data.py first.")
        return np.load(golden_path)

    @pytest.fixture
    def golden_expected(self):
        """Load expected golden metrics."""
        expected_path = os.path.join(FIXTURES_DIR, 'golden_expected.json')
        if not os.path.exists(expected_path):
            pytest.skip("Golden expected not found. Run scripts/generate_golden_data.py first.")
        return load_json(expected_path)

    def test_golden_oracle_no_overload(self, golden_data, golden_expected):
        """Golden: Oracle must have zero overload (robust float comparison)."""
        predictions = {
            'Y_true': golden_data['Y_true'],
            'Y_pred_sarima': golden_data['Y_pred_sarima'],
            'Y_pred_lstm': golden_data['Y_pred_lstm'],
            'T_eff': int(golden_data['T_eff']),
            'num_links': int(golden_data['num_links'])
        }

        margin = golden_expected['config']['capacity_margin']
        capacities = compute_capacities(predictions, margin)
        metrics = compute_utilization_metrics(predictions['Y_true'], capacities)

        # Use <= for robust float comparison (f_over is mean of booleans, but be safe)
        assert metrics['Oracle']['f_over_mean'] <= 1e-12, \
            f"Golden: Oracle f_over_mean={metrics['Oracle']['f_over_mean']} > 0"

    def test_golden_oracle_u_max(self, golden_data, golden_expected):
        """Golden: Oracle u_max_mean and u_max_max should be bounded by 1/alpha."""
        predictions = {
            'Y_true': golden_data['Y_true'],
            'Y_pred_sarima': golden_data['Y_pred_sarima'],
            'Y_pred_lstm': golden_data['Y_pred_lstm'],
            'T_eff': int(golden_data['T_eff']),
            'num_links': int(golden_data['num_links'])
        }

        margin = golden_expected['config']['capacity_margin']
        capacities = compute_capacities(predictions, margin)
        metrics = compute_utilization_metrics(predictions['Y_true'], capacities)

        expected = 1.0 / margin
        tol = 1e-6

        # Check both u_max_mean and u_max_max
        assert metrics['Oracle']['u_max_mean'] <= expected + tol, \
            f"Golden: Oracle u_max_mean={metrics['Oracle']['u_max_mean']:.6f} > {expected:.6f}"
        assert metrics['Oracle']['u_max_max'] <= expected + tol, \
            f"Golden: Oracle u_max_max={metrics['Oracle']['u_max_max']:.6f} > {expected:.6f}"

    def test_golden_headline_numbers(self, golden_data, golden_expected):
        """Golden: Headline metrics should match expected values."""
        predictions = {
            'Y_true': golden_data['Y_true'],
            'Y_pred_sarima': golden_data['Y_pred_sarima'],
            'Y_pred_lstm': golden_data['Y_pred_lstm'],
            'T_eff': int(golden_data['T_eff']),
            'num_links': int(golden_data['num_links'])
        }

        margin = golden_expected['config']['capacity_margin']
        capacities = compute_capacities(predictions, margin)
        metrics = compute_utilization_metrics(predictions['Y_true'], capacities)

        # Verify key invariants (these are integer counts, so == 0 is safe)
        assert metrics['Oracle']['links_over_100'] == 0, \
            f"Golden: Oracle links_over_100={metrics['Oracle']['links_over_100']} != 0"
        assert metrics['Oracle']['links_over_110'] == 0, \
            f"Golden: Oracle links_over_110={metrics['Oracle']['links_over_110']} != 0"
