#!/usr/bin/env python
"""
Generate golden data fixtures for regression testing.

This creates a small, deterministic dataset that can be used to verify
the capacity planning pipeline works correctly.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import save_json

# Output paths
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests', 'fixtures')


def generate_golden_data():
    """Generate a small, deterministic test dataset."""
    np.random.seed(42)  # Reproducible

    # Small dataset: 2 links, 100 time steps
    num_links = 2
    T_eff = 100
    margin = 1.1

    # Generate simple sinusoidal traffic with noise
    t = np.arange(T_eff)
    base = 10.0

    # Link 0: simple sine wave
    Y_true_0 = base + 5 * np.sin(2 * np.pi * t / 24)
    # Link 1: phase-shifted sine wave
    Y_true_1 = base + 3 * np.sin(2 * np.pi * t / 24 + np.pi / 4)

    Y_true = np.column_stack([Y_true_0, Y_true_1])

    # Add small noise
    Y_true += np.random.randn(T_eff, num_links) * 0.5
    Y_true = np.maximum(Y_true, 1.0)  # Ensure positive

    # Create predictions with some error
    noise_sarima = np.random.randn(T_eff, num_links) * 1.0
    noise_lstm = np.random.randn(T_eff, num_links) * 0.8

    Y_pred_sarima = Y_true + noise_sarima
    Y_pred_lstm = Y_true + noise_lstm

    # Ensure predictions are positive
    Y_pred_sarima = np.maximum(Y_pred_sarima, 0.1)
    Y_pred_lstm = np.maximum(Y_pred_lstm, 0.1)

    # Save golden data
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    np.savez(
        os.path.join(FIXTURES_DIR, 'golden_data.npz'),
        Y_true=Y_true,
        Y_pred_sarima=Y_pred_sarima,
        Y_pred_lstm=Y_pred_lstm,
        T_eff=T_eff,
        num_links=num_links
    )

    # Save expected metrics
    expected = {
        'config': {
            'capacity_margin': margin
        },
        'expected_oracle': {
            'f_over_mean': 0,
            'links_over_100': 0,
            'links_over_110': 0,
            'u_max_mean': 1.0 / margin
        }
    }
    save_json(expected, os.path.join(FIXTURES_DIR, 'golden_expected.json'))

    print(f"Generated golden data:")
    print(f"  - Y_true shape: {Y_true.shape}")
    print(f"  - Y_pred_sarima shape: {Y_pred_sarima.shape}")
    print(f"  - Y_pred_lstm shape: {Y_pred_lstm.shape}")
    print(f"  - Margin: {margin}")
    print(f"  - Saved to: {FIXTURES_DIR}")


if __name__ == '__main__':
    generate_golden_data()
