"""Tests for SARIMA/LSTM prediction alignment.

These are INTEGRATION tests that require trained model artifacts.
Run the full pipeline first, or skip with: pytest -m "not integration"
"""

import os
import numpy as np
import pytest

from config import CONFIG, DATA_DIR, RESULTS_DIR

# Auto-skip if artifacts are missing
pytestmark = pytest.mark.integration
if not os.path.exists(os.path.join(RESULTS_DIR, 'lstm_predictions.npz')):
    pytest.skip("Integration artifacts not found (run pipeline first)", allow_module_level=True)


class TestPredictionAlignment:
    """Verify predictions are correctly aligned between models."""

    def test_sarima_lstm_shapes_match_ground_truth(self, predictions):
        """SARIMA and LSTM predictions must have the same shape as Y_true."""
        Y_true = predictions['Y_true']
        Y_pred_lstm = predictions['Y_pred_lstm']
        Y_pred_sarima = predictions['Y_pred_sarima']

        assert Y_true.shape == Y_pred_lstm.shape, \
            f"LSTM shape {Y_pred_lstm.shape} != Y_true shape {Y_true.shape}"
        assert Y_true.shape == Y_pred_sarima.shape, \
            f"SARIMA shape {Y_pred_sarima.shape} != Y_true shape {Y_true.shape}"

    def test_t_eff_matches_actual_length(self, predictions):
        """T_eff must match the actual length of prediction arrays."""
        Y_true = predictions['Y_true']
        T_eff = predictions['T_eff']

        assert Y_true.shape[0] == T_eff, \
            f"T_eff={T_eff} but Y_true has {Y_true.shape[0]} time steps"

    def test_y_true_matches_ground_truth(self, predictions, traffic_data, window_size):
        """Y_true must match the aligned ground truth from traffic data."""
        Y_true = predictions['Y_true']
        L = traffic_data['L']
        train_end = traffic_data['train_end']
        val_end = traffic_data['val_end']

        # Bulletproof boundary checks
        assert val_end > train_end, \
            f"Invalid boundary: val_end={val_end} must be > train_end={train_end}"
        assert val_end + window_size < L.shape[0], \
            f"Invalid offset: val_end + window_size exceeds data length"

        # Ground truth for test set, aligned to LSTM window
        L_test_aligned = L[val_end + window_size:]

        # Shape must match before content comparison
        assert L_test_aligned.shape == Y_true.shape, \
            f"Shape mismatch: L_test_aligned {L_test_aligned.shape} != Y_true {Y_true.shape}"

        assert np.allclose(Y_true, L_test_aligned), \
            "Y_true doesn't match expected aligned ground truth"

    def test_num_links_consistent(self, predictions, traffic_data):
        """Number of links must be consistent across all arrays."""
        expected_links = traffic_data['L'].shape[1]
        actual_links = predictions['num_links']

        assert actual_links == expected_links, \
            f"num_links={actual_links} but traffic data has {expected_links} links"

        # Verify all prediction arrays have correct number of links
        assert predictions['Y_true'].shape[1] == expected_links
        assert predictions['Y_pred_lstm'].shape[1] == expected_links
        assert predictions['Y_pred_sarima'].shape[1] == expected_links

    def test_no_nan_in_predictions(self, predictions):
        """Predictions should not contain NaN values."""
        assert not np.any(np.isnan(predictions['Y_true'])), \
            "Y_true contains NaN"
        assert not np.any(np.isnan(predictions['Y_pred_lstm'])), \
            "LSTM predictions contain NaN"
        assert not np.any(np.isnan(predictions['Y_pred_sarima'])), \
            "SARIMA predictions contain NaN"
