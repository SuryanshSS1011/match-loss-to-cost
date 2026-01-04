"""
Global configuration for network traffic forecasting experiment.
"""

import os

# Project paths (go up one level from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

# Experiment configuration
CONFIG = {
    # Network topology
    'num_nodes': 12,
    'watts_strogatz_k': 4,
    'watts_strogatz_p': 0.2,

    # Time series parameters
    'time_step_minutes': 5,
    'days': 14,

    # Train/val/test split
    'train_frac': 0.6,
    'val_frac': 0.2,
    'test_frac': 0.2,

    # Reproducibility
    'random_seed': 42,

    # Traffic generation parameters
    'base_traffic_min': 5,
    'base_traffic_max': 20,
    'amplitude_min': 0.2,
    'amplitude_max': 1.0,
    'noise_factor': 0.5,  # noise std = noise_factor * base
    'burst_prob': 0.01,
    'burst_factor_min': 2,
    'burst_factor_max': 5,

    # Forecasting model parameters
    # LSTM window=72 (6 hours) captures partial-cycle context for computational efficiency
    # Daily structure is captured by Seasonal Naive baseline and SARIMA seasonal component
    'window_size': 72,  # 6 hours at 5-min intervals

    # SARIMA parameters
    'arima_order': (2, 1, 2),
    # Using s=72 (6 hours) to match LSTM window size for fair comparison
    'seasonal_order': (1, 0, 1, 72),

    # LSTM parameters
    'lstm_hidden_size': 64,
    'lstm_num_layers': 2,
    'lstm_epochs': 50,
    'lstm_lr': 1e-3,
    'lstm_batch_size': 64,
    'lstm_patience': 5,  # early stopping patience

    # Capacity planning
    'capacity_method': 'max',  # Use max over eval window per paper definition
    'capacity_margin': 1.1,    # α = 10% safety margin

    # MAPE calculation
    # Points where y_true < mape_threshold are excluded to avoid inflation
    'mape_threshold': 1.0,
}

# Derived constants
CONFIG['total_time_steps'] = CONFIG['days'] * 24 * 60 // CONFIG['time_step_minutes']
CONFIG['seasonal_period'] = 24 * 60 // CONFIG['time_step_minutes']  # 288 (1 day)
