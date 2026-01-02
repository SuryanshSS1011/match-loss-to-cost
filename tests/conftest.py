"""Shared fixtures for network traffic forecasting tests."""

import os
import sys
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CONFIG, DATA_DIR, RESULTS_DIR
from src.eval_capacity import load_predictions, compute_capacities, compute_utilization_metrics


@pytest.fixture
def predictions():
    """Load aligned predictions from both models."""
    return load_predictions()


@pytest.fixture
def capacities(predictions):
    """Compute capacities for all models."""
    margin = CONFIG['capacity_margin']
    return compute_capacities(predictions, margin)


@pytest.fixture
def capacity_metrics(predictions, capacities):
    """Compute utilization metrics for all models."""
    return compute_utilization_metrics(predictions['Y_true'], capacities)


@pytest.fixture
def traffic_data():
    """Load raw traffic data."""
    data = np.load(os.path.join(DATA_DIR, 'traffic_data.npz'))
    return {
        'L': data['L'],
        'train_end': int(data['train_end']),
        'val_end': int(data['val_end']),
        'T_train': int(data['T_train']),
        'T_val': int(data['T_val']),
        'T_test': int(data['T_test'])
    }


@pytest.fixture
def margin():
    """Return the capacity margin alpha."""
    return CONFIG['capacity_margin']


@pytest.fixture
def window_size():
    """Return the LSTM window size."""
    return CONFIG['window_size']
