"""
Shared utility functions for network traffic forecasting experiment.
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_sequences(L: np.ndarray, window_size: int) -> tuple:
    """
    Create sequences for LSTM training/prediction.

    Args:
        L: Link load array of shape (T, num_links)
        window_size: Number of past time steps to use as input

    Returns:
        X: Input sequences of shape (num_samples, window_size, num_links)
        y: Target values of shape (num_samples, num_links)
    """
    T, num_links = L.shape
    num_samples = T - window_size

    X = np.zeros((num_samples, window_size, num_links), dtype=np.float32)
    y = np.zeros((num_samples, num_links), dtype=np.float32)

    for i in range(num_samples):
        X[i] = L[i:i + window_size]
        y[i] = L[i + window_size]

    return X, y


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute Symmetric MAPE (sMAPE) per link.

    sMAPE handles zeros and near-zeros better than MAPE by using
    the average of true and predicted values in the denominator.

    sMAPE = 100 * mean(|y - ŷ| / ((|y| + |ŷ|) / 2))

    Args:
        y_true: True values of shape (T, num_links)
        y_pred: Predicted values of shape (T, num_links)

    Returns:
        Per-link sMAPE array of shape (num_links,)
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.maximum(denominator, 1e-6)  # Avoid division by zero

    return 100 * np.mean(numerator / denominator, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    mape_threshold: float = None) -> dict:
    """
    Compute forecasting metrics per link.

    Args:
        y_true: True values of shape (T, num_links)
        y_pred: Predicted values of shape (T, num_links)
        mape_threshold: Minimum y_true value to include in MAPE
                        (default: from CONFIG['mape_threshold'])

    Returns:
        Dictionary with per-link RMSE, MAE, MAPE, sMAPE arrays
        and MAPE exclusion diagnostics
    """
    from .config import CONFIG

    if mape_threshold is None:
        mape_threshold = CONFIG.get('mape_threshold', 1.0)

    # Per-link RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    # Per-link MAE
    mae = np.mean(np.abs(y_true - y_pred), axis=0)

    # Per-link sMAPE (symmetric MAPE - handles zeros better)
    smape = compute_smape(y_true, y_pred)

    # Per-link MAPE with threshold masking
    # Points where y_true < threshold are excluded to avoid inflation
    # Note: Loop is acceptable for small num_links; vectorize if scaling up
    mask = y_true > mape_threshold
    num_links = y_true.shape[1]
    mape = np.zeros(num_links)
    pct_excluded = np.zeros(num_links)

    for link in range(num_links):
        link_mask = mask[:, link]
        valid_count = link_mask.sum()
        total_count = len(link_mask)

        pct_excluded[link] = 100 * (1 - valid_count / total_count)

        if valid_count > 0:
            mape[link] = 100 * np.mean(
                np.abs(y_true[link_mask, link] - y_pred[link_mask, link]) /
                y_true[link_mask, link]
            )
        else:
            mape[link] = np.nan

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'mape_pct_excluded': pct_excluded,
        'mape_threshold': mape_threshold
    }


def aggregate_metrics(per_link_metrics: dict) -> dict:
    """
    Compute aggregate statistics over all links.

    Args:
        per_link_metrics: Dictionary with per-link metric arrays

    Returns:
        Dictionary with mean, median, and 90th percentile for each metric,
        plus MAPE configuration if present
    """
    aggregated = {}

    # Metrics to aggregate (skip non-array fields)
    array_metrics = ['rmse', 'mae', 'mape', 'smape', 'mape_pct_excluded']

    for metric_name in array_metrics:
        if metric_name not in per_link_metrics:
            continue
        values = per_link_metrics[metric_name]
        if not isinstance(values, np.ndarray):
            continue
        # Use nan-aware functions to handle masked MAPE values
        aggregated[f'{metric_name}_mean'] = float(np.nanmean(values))
        aggregated[f'{metric_name}_median'] = float(np.nanmedian(values))
        aggregated[f'{metric_name}_p90'] = float(np.nanpercentile(values, 90))

    # Include MAPE threshold if present
    if 'mape_threshold' in per_link_metrics:
        aggregated['mape_threshold'] = per_link_metrics['mape_threshold']

    return aggregated


def save_json(data: dict, path: str) -> None:
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_metric_comparison(metrics_dict: dict, metric_name: str,
                           title: str, save_path: str) -> None:
    """
    Create bar chart comparing a metric across models.

    Args:
        metrics_dict: Dict mapping model name to its aggregated metrics
        metric_name: Name of the metric to plot (e.g., 'rmse_mean')
        title: Plot title
        save_path: Path to save the figure
    """
    models = list(metrics_dict.keys())
    values = [metrics_dict[m][metric_name] for m in models]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, values, color=['steelblue', 'darkorange', 'green'][:len(models)])

    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Model')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_rmse_histogram(rmse_dict: dict, save_path: str) -> None:
    """
    Plot histogram of per-link RMSE for each model.

    Args:
        rmse_dict: Dict mapping model name to per-link RMSE array
        save_path: Path to save the figure
    """
    n_models = len(rmse_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)

    if n_models == 1:
        axes = [axes]

    colors = ['steelblue', 'darkorange', 'green']

    for ax, (model_name, rmse_values), color in zip(axes, rmse_dict.items(), colors):
        ax.hist(rmse_values, bins=15, color=color, edgecolor='black', alpha=0.7)
        ax.set_xlabel('RMSE')
        ax.set_ylabel('Number of Links')
        ax.set_title(f'{model_name} Per-Link RMSE')
        ax.axvline(np.mean(rmse_values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(rmse_values):.2f}')
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_timeseries_comparison(time_indices: np.ndarray, y_true: np.ndarray,
                               predictions_dict: dict, link_idx: int,
                               capacity_dict: dict, save_path: str) -> None:
    """
    Plot time series of true vs predicted values for a single link.

    Args:
        time_indices: Time step indices
        y_true: True values of shape (T,)
        predictions_dict: Dict mapping model name to predictions array of shape (T,)
        link_idx: Index of the link being plotted
        capacity_dict: Dict mapping model name to capacity value for this link
        save_path: Path to save the figure
    """
    plt.figure(figsize=(14, 6))

    # Plot true values
    plt.plot(time_indices, y_true, 'k-', linewidth=1.5, label='True', alpha=0.8)

    # Plot predictions
    colors = {'SARIMA': 'steelblue', 'LSTM': 'darkorange', 'Oracle': 'green'}
    for model_name, preds in predictions_dict.items():
        color = colors.get(model_name, 'gray')
        plt.plot(time_indices, preds, '--', color=color, linewidth=1,
                 label=f'{model_name} Pred', alpha=0.7)

    # Plot capacity lines
    linestyles = {'SARIMA': ':', 'LSTM': '-.', 'Oracle': '--'}
    for model_name, cap in capacity_dict.items():
        color = colors.get(model_name, 'gray')
        ls = linestyles.get(model_name, ':')
        plt.axhline(cap, color=color, linestyle=ls, linewidth=1.5,
                    label=f'{model_name} Cap: {cap:.1f}', alpha=0.6)

    plt.xlabel('Time Step')
    plt.ylabel('Link Load')
    plt.title(f'Link {link_idx}: True vs Predicted Load with Capacity Lines')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_capacity_bars(capacity_metrics: dict, metric_name: str,
                       title: str, save_path: str) -> None:
    """
    Create bar chart for capacity planning metrics.

    Args:
        capacity_metrics: Dict mapping model name to capacity metrics dict
        metric_name: Name of metric to plot (e.g., 'u_max_mean')
        title: Plot title
        save_path: Path to save figure
    """
    models = list(capacity_metrics.keys())
    values = [capacity_metrics[m][metric_name] for m in models]

    plt.figure(figsize=(8, 5))
    colors = ['steelblue', 'darkorange', 'green'][:len(models)]
    bars = plt.bar(models, values, color=colors)

    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Add reference line at 1.0 for utilization metrics
    if 'u_max' in metric_name:
        plt.axhline(1.0, color='red', linestyle='--', linewidth=1.5,
                    label='100% Utilization')
        plt.legend()

    plt.xlabel('Model')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def check_periodicity(L: np.ndarray, expected_period: int = 288,
                       link_idx: int = 0, min_peak_height: float = 0.3) -> bool:
    """
    Verify time series has expected periodicity via autocorrelation.

    This is an OPTIONAL DIAGNOSTIC - call manually to verify data generation
    produces the expected daily periodicity. Not run automatically.

    Uses ACF peak detection to confirm the data has strong periodicity
    at the expected period (e.g., daily = 288 steps at 5-min intervals).

    Args:
        L: Link load array of shape (T, num_links)
        expected_period: Expected periodicity in time steps (default: 288 = 1 day)
        link_idx: Which link to analyze (default: 0)
        min_peak_height: Minimum ACF peak height to consider significant

    Returns:
        True if periodicity is confirmed, False otherwise
    """
    from scipy import signal

    # Use a representative link
    link_data = L[:, link_idx]

    # Compute autocorrelation (normalized)
    link_centered = link_data - link_data.mean()
    acf = np.correlate(link_centered, link_centered, mode='full')
    acf = acf[len(acf) // 2:]  # Keep positive lags only
    acf = acf / acf[0]  # Normalize

    # Find peaks in ACF
    peaks, properties = signal.find_peaks(acf, height=min_peak_height)

    # Check for peak near expected_period (within 5 steps tolerance)
    tolerance = 5
    near_expected = [p for p in peaks if abs(p - expected_period) <= tolerance]

    if near_expected:
        peak_height = acf[near_expected[0]]
        print(f"   ✓ Periodicity confirmed: peak at lag {near_expected[0]} "
              f"(expected {expected_period}), ACF = {peak_height:.3f}")
        return True
    else:
        nearby_peaks = peaks[:10] if len(peaks) > 0 else []
        print(f"   ⚠ No strong peak near {expected_period}. "
              f"Peaks found at lags: {list(nearby_peaks)}")
        return False


def print_summary_table(forecasting_metrics: dict, capacity_metrics: dict) -> None:
    """Print a formatted summary table of all results."""
    print("\n" + "=" * 60)
    print("       NETWORK TRAFFIC FORECASTING RESULTS")
    print("=" * 60)

    # Forecasting metrics
    print("\n--- Forecasting Metrics (Test Set) ---\n")
    print(f"{'Metric':<15} {'SARIMA':>12} {'LSTM':>12}")
    print("-" * 40)

    for metric in ['rmse_mean', 'mae_mean', 'mape_mean', 'smape_mean']:
        sarima_val = forecasting_metrics.get('SARIMA', {}).get(metric, 'N/A')
        lstm_val = forecasting_metrics.get('LSTM', {}).get(metric, 'N/A')

        if isinstance(sarima_val, (int, float)):
            sarima_str = f"{sarima_val:.4f}"
        else:
            sarima_str = str(sarima_val)

        if isinstance(lstm_val, (int, float)):
            lstm_str = f"{lstm_val:.4f}"
        else:
            lstm_str = str(lstm_val)

        metric_label = metric.replace('_mean', '').upper()
        if 'mape' in metric.lower() or 'smape' in metric.lower():
            metric_label += ' (%)'
        print(f"{metric_label:<15} {sarima_str:>12} {lstm_str:>12}")

    # Show MAPE exclusion info if available
    mape_threshold = forecasting_metrics.get('SARIMA', {}).get('mape_threshold')
    mape_excluded = forecasting_metrics.get('SARIMA', {}).get('mape_pct_excluded_mean')
    if mape_threshold is not None and mape_excluded is not None:
        print(f"\n   Note: MAPE excludes points where y < {mape_threshold} "
              f"({mape_excluded:.1f}% of data)")

    # Capacity planning metrics
    print("\n--- Capacity Planning Metrics ---\n")
    print(f"{'Metric':<15} {'SARIMA':>12} {'LSTM':>12} {'Oracle':>12}")
    print("-" * 55)

    for metric in ['u_max_mean', 'u_max_max', 'f_over_mean', 'links_over_110']:
        sarima_val = capacity_metrics.get('SARIMA', {}).get(metric, 'N/A')
        lstm_val = capacity_metrics.get('LSTM', {}).get(metric, 'N/A')
        oracle_val = capacity_metrics.get('Oracle', {}).get(metric, 'N/A')

        vals = []
        for v in [sarima_val, lstm_val, oracle_val]:
            if isinstance(v, (int, float)):
                if 'f_over' in metric:
                    vals.append(f"{v*100:.2f}%")
                elif isinstance(v, int) or v == int(v):
                    vals.append(f"{int(v)}")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))

        metric_label = metric.replace('_', ' ').title()
        print(f"{metric_label:<15} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    print("\n" + "=" * 60)
