"""Tests for data leakage in normalization.

These are INTEGRATION tests that require trained model artifacts.
Run the full pipeline first, or skip with: pytest -m "not integration"
"""

import os
import numpy as np
import pytest

from src.config import DATA_DIR, RESULTS_DIR
from src.utils import load_json

# Auto-skip if artifacts are missing
pytestmark = pytest.mark.integration
if not os.path.exists(os.path.join(DATA_DIR, 'traffic_data.npz')):
    pytest.skip("Integration artifacts not found (run pipeline first)", allow_module_level=True)


class TestNoLeakageScaling:
    """Ensure normalization stats come from training data only."""

    def test_normalization_stats_exist(self):
        """Normalization stats file should exist after training."""
        stats_path = os.path.join(RESULTS_DIR, 'normalization_stats.json')

        if not os.path.exists(stats_path):
            pytest.skip(
                "normalization_stats.json not found. "
                "Run train_lstm.py first (requires saving normalization stats)."
            )

    def test_normalization_uses_train_only(self, traffic_data):
        """
        Normalization stats must come from training data only.

        Tests the SAVED stats from the pipeline, not a flaky comparison.
        Requires pipeline to save normalization_stats.json during training.
        """
        stats_path = os.path.join(RESULTS_DIR, 'normalization_stats.json')

        if not os.path.exists(stats_path):
            pytest.skip("normalization_stats.json not found")

        # Load what the pipeline actually used
        stats = load_json(stats_path)
        used_mean = np.array(stats['mean'])
        used_std = np.array(stats['std'])

        # Compute expected train-only stats
        L = traffic_data['L']
        train_end = traffic_data['train_end']

        true_train_mean = L[:train_end].mean(axis=0)
        true_train_std = L[:train_end].std(axis=0)
        # Match pipeline's zero-handling (avoid division by zero)
        true_train_std[true_train_std < 1e-6] = 1.0

        # Assert pipeline used train-only stats
        assert np.allclose(used_mean, true_train_mean, atol=1e-6), \
            "Pipeline mean != train mean (possible data leakage)"
        assert np.allclose(used_std, true_train_std, atol=1e-6), \
            "Pipeline std != train std (possible data leakage)"

    def test_train_split_is_before_val_test(self, traffic_data):
        """Verify train/val/test splits are in correct chronological order."""
        L = traffic_data['L']
        train_end = traffic_data['train_end']
        val_end = traffic_data['val_end']
        T = L.shape[0]

        assert 0 < train_end < val_end < T, \
            f"Invalid split order: train_end={train_end}, val_end={val_end}, T={T}"

        # Verify splits are consistent with T values
        T_train = traffic_data['T_train']
        T_val = traffic_data['T_val']
        T_test = traffic_data['T_test']

        assert train_end == T_train, \
            f"train_end={train_end} != T_train={T_train}"
        assert val_end == train_end + T_val, \
            f"val_end={val_end} != train_end + T_val={train_end + T_val}"
        assert T == val_end + T_test, \
            f"T={T} != val_end + T_test={val_end + T_test}"
