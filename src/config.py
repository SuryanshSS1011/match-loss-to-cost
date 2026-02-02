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
    # 'rolling': fit once on train+val, then walk the test set one step at a time
    #           updating the state with each new observation (statsmodels apply()).
    # 'oneshot': single open-loop forecast for T_test steps. Only valid for short horizons.
    'sarima_mode': 'rolling',
    # If set, cap the SARIMA test horizon at this many steps (one rolling step per index).
    # None means full T_test. Useful on real data where T_test is in the thousands.
    # Local: 288 = 1 day. Cloud full-fidelity: 2016 = 1 week.
    'sarima_test_steps': 288,
    # If set, only use the last N training points for SARIMA fitting (None = full history).
    # Local: 4032 (14 days). Cloud full-fidelity: None.
    'sarima_train_window': 4032,
    # If set, fit SARIMA on a random subset of N links (deterministic via random_seed).
    # Local presence-check default: 8. Cloud full-fidelity: None (= all links).
    'sarima_link_subset': 8,

    # LSTM parameters
    'lstm_hidden_size': 64,
    'lstm_num_layers': 2,
    'lstm_epochs': 50,
    'lstm_lr': 1e-3,
    'lstm_batch_size': 64,
    'lstm_patience': 5,  # early stopping patience

    # Loss function for LSTM training. One of: 'mse' | 'asym' | 'pinball'.
    # asym uses (loss_alpha, loss_beta); pinball uses loss_tau (or derives
    # from alpha/beta as alpha/(alpha+beta)).
    # Despite the 'lstm_' prefix, this key is read by ALL neural trainers
    # (LSTM, DLinear, PatchTST) — kept for backwards compatibility.
    'lstm_loss': 'mse',
    'loss_alpha': 5.0,   # under-prediction penalty weight
    'loss_beta': 1.0,    # over-prediction penalty weight
    'loss_tau': None,    # if None, derived from alpha/beta for pinball

    # DLinear parameters
    'dlinear_kernel_size': 25,
    'dlinear_individual': True,

    # PatchTST parameters (defaults tuned for window_size=72).
    'patchtst_patch_len': 12,
    'patchtst_stride': 6,
    'patchtst_d_model': 128,
    'patchtst_n_heads': 8,
    'patchtst_n_layers': 3,
    'patchtst_dim_ff': 256,
    'patchtst_dropout': 0.2,
    'patchtst_revin': True,
    'patchtst_revin_affine': True,

    # iTransformer parameters (variate-as-token; cross-channel attention).
    'itransformer_d_model': 128,
    'itransformer_n_heads': 8,
    'itransformer_n_layers': 3,
    'itransformer_dim_ff': 256,
    'itransformer_dropout': 0.2,
    'itransformer_revin': True,
    'itransformer_revin_affine': True,

    # Chronos-Bolt zero-shot parameters.
    # Variants (size / VRAM): tiny (9M / ~1GB), mini (21M), small (48M),
    # base (200M), large (700M). Tiny is enough for the foundation-model
    # column; bump up only on the cloud GPU box.
    'chronos_model_name': 'amazon/chronos-bolt-tiny',
    'chronos_context_length': 72,  # match window_size by default
    'chronos_batch_size': 128,

    # DCRNN parameters (graph-aware via routing-matrix-derived adjacency).
    'dcrnn_hidden_dim': 64,
    'dcrnn_num_layers': 2,
    'dcrnn_K': 2,  # diffusion steps per direction (total spatial filter = 2K)

    # Holt-Winters parameters.
    # Like SARIMA, bounded for laptop dev; set None on the cloud for full
    # fidelity. Seasonal period is shared with seasonal-naive (= seasonal_period).
    'holtwinters_train_window': 4032,   # ~14 days at 5-min; None = full
    'holtwinters_test_steps': 288,      # 1 day rolling test; None = full

    # Dataset selection. One of: 'synthetic', 'abilene', 'geant', 'cesnet'.
    # Loader resolves to data/<DATASET_FILES[name]>.
    'dataset': 'abilene',

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

# Dataset → npz filename
DATASET_FILES = {
    'synthetic': 'traffic_data.npz',
    'abilene':   'abilene_traffic.npz',
    'geant':     'geant_traffic.npz',
    'cesnet':    'cesnet_traffic.npz',
}


def dataset_path(name: str | None = None) -> str:
    """Resolve the data .npz path for a dataset name (defaults to CONFIG['dataset'])."""
    name = name or CONFIG['dataset']
    if name not in DATASET_FILES:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASET_FILES)}")
    return os.path.join(DATA_DIR, DATASET_FILES[name])
