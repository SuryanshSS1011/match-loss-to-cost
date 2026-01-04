#!/usr/bin/env python
"""
Multi-seed experiment runner for network traffic forecasting.

Runs the full pipeline across multiple random seeds and aggregates
results with uncertainty estimates (mean ± std).
"""

import os
import sys
import subprocess
import argparse
import shutil
from datetime import datetime
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import CONFIG, DATA_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_DIR
from src.utils import set_all_seeds, save_json, load_json


# Default seeds for reproducibility
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def get_git_info():
    """Get current git commit info for reproducibility logging."""
    try:
        commit = subprocess.getoutput('git rev-parse HEAD')
        dirty = subprocess.getoutput('git status --porcelain') != ''
        return {'commit': commit, 'dirty': dirty}
    except Exception:
        return {'commit': 'unknown', 'dirty': True}


def log_run_metadata(seed: int, output_dir: str, config: dict):
    """Log everything needed to reproduce this run."""
    metadata = {
        'seed': seed,
        'git': get_git_info(),
        'config': {k: str(v) for k, v in config.items()},  # Convert tuples to strings
        'timestamp': datetime.now().isoformat()
    }
    save_json(metadata, os.path.join(output_dir, 'run_metadata.json'))


def run_single_seed(seed: int, base_dir: str = None) -> dict:
    """
    Run full pipeline with given seed, return metrics.

    Args:
        seed: Random seed to use
        base_dir: Base directory for outputs (default: results/)

    Returns:
        Dictionary with headline metrics from this seed
    """
    import src.config as config_module

    # Update the CONFIG dict in the module itself
    config_module.CONFIG['random_seed'] = seed

    # Set all RNGs
    set_all_seeds(seed)

    # Create seed-specific output directory
    if base_dir is None:
        base_dir = RESULTS_DIR

    seed_dir = os.path.join(base_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)

    # Log metadata
    log_run_metadata(seed, seed_dir, config_module.CONFIG)

    print(f"\n{'='*50}")
    print(f"Running seed {seed}")
    print(f"{'='*50}")

    # Import modules fresh for each seed to reset state
    import importlib

    # Run data generation
    print(f"\n[Seed {seed}] Generating traffic data...")
    try:
        from src import simulate_data
        importlib.reload(simulate_data)
        simulate_data.main()
    except Exception as e:
        print(f"Error in simulate_data: {e}")
        import traceback
        traceback.print_exc()
        return {'seed': seed, 'error': f'simulate_data: {e}'}

    # Run SARIMA training
    print(f"\n[Seed {seed}] Training SARIMA...")
    try:
        from src import train_arima
        importlib.reload(train_arima)
        train_arima.main()
    except Exception as e:
        print(f"Error in train_arima: {e}")
        import traceback
        traceback.print_exc()
        return {'seed': seed, 'error': f'train_arima: {e}'}

    # Run LSTM training
    print(f"\n[Seed {seed}] Training LSTM...")
    try:
        from src import train_lstm
        importlib.reload(train_lstm)
        train_lstm.main()
    except Exception as e:
        print(f"Error in train_lstm: {e}")
        import traceback
        traceback.print_exc()
        return {'seed': seed, 'error': f'train_lstm: {e}'}

    # Run evaluation
    print(f"\n[Seed {seed}] Running evaluation...")
    try:
        from src import eval_capacity
        importlib.reload(eval_capacity)
        eval_capacity.main()
    except Exception as e:
        print(f"Error in eval_capacity: {e}")
        import traceback
        traceback.print_exc()
        return {'seed': seed, 'error': f'eval_capacity: {e}'}

    # Load and save results to seed-specific directory
    try:
        # Copy results to seed directory
        for fname in ['sarima_predictions.npz', 'lstm_predictions.npz',
                      'sarima_metrics.json', 'lstm_metrics.json',
                      'capacity_planning.json', 'combined_results.json']:
            src_path = os.path.join(RESULTS_DIR, fname)
            dst_path = os.path.join(seed_dir, fname)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

        # Load metrics
        combined = load_json(os.path.join(RESULTS_DIR, 'combined_results.json'))
        sarima_metrics = combined['forecasting']['SARIMA']
        lstm_metrics = combined['forecasting']['LSTM']
        capacity = combined['capacity_planning']

        return {
            'seed': seed,
            'sarima_rmse': sarima_metrics.get('rmse_mean'),
            'sarima_mae': sarima_metrics.get('mae_mean'),
            'sarima_mape': sarima_metrics.get('mape_mean'),
            'lstm_rmse': lstm_metrics.get('rmse_mean'),
            'lstm_mae': lstm_metrics.get('mae_mean'),
            'lstm_mape': lstm_metrics.get('mape_mean'),
            'sarima_u_max': capacity['SARIMA']['u_max_mean'],
            'lstm_u_max': capacity['LSTM']['u_max_mean'],
            'sarima_f_over': capacity['SARIMA']['f_over_mean'],
            'lstm_f_over': capacity['LSTM']['f_over_mean'],
        }
    except Exception as e:
        print(f"Error loading results for seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return {'seed': seed, 'error': str(e)}


def aggregate_multi_seed(results: list) -> dict:
    """
    Compute mean ± std across seeds.

    Args:
        results: List of per-seed result dictionaries

    Returns:
        Dictionary with aggregated statistics
    """
    # Filter out failed runs
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        return {'error': 'All seeds failed'}

    metrics = [
        'sarima_rmse', 'sarima_mae', 'sarima_mape',
        'lstm_rmse', 'lstm_mae', 'lstm_mape',
        'sarima_u_max', 'lstm_u_max',
        'sarima_f_over', 'lstm_f_over'
    ]

    aggregated = {
        'num_seeds': len(valid_results),
        'seeds': [r['seed'] for r in valid_results],
        'per_seed': valid_results
    }

    for metric in metrics:
        values = [r[metric] for r in valid_results if r.get(metric) is not None]
        if values:
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }

    return aggregated


def print_multi_seed_summary(aggregated: dict):
    """Print formatted summary of multi-seed results."""
    print("\n" + "=" * 70)
    print("       MULTI-SEED EXPERIMENT RESULTS")
    print("=" * 70)

    n = aggregated.get('num_seeds', 0)
    print(f"\nSeeds: {aggregated.get('seeds', [])} (n={n})")

    print("\n--- Forecasting Metrics (mean ± std) ---\n")
    print(f"{'Metric':<15} {'SARIMA':>20} {'LSTM':>20}")
    print("-" * 55)

    for metric_base in ['rmse', 'mae', 'mape']:
        sarima_key = f'sarima_{metric_base}'
        lstm_key = f'lstm_{metric_base}'

        sarima = aggregated.get(sarima_key, {})
        lstm = aggregated.get(lstm_key, {})

        sarima_str = f"{sarima.get('mean', 0):.4f} ± {sarima.get('std', 0):.4f}"
        lstm_str = f"{lstm.get('mean', 0):.4f} ± {lstm.get('std', 0):.4f}"

        label = metric_base.upper()
        if 'mape' in metric_base:
            label += ' (%)'
        print(f"{label:<15} {sarima_str:>20} {lstm_str:>20}")

    print("\n--- Capacity Planning Metrics (mean ± std) ---\n")
    print(f"{'Metric':<15} {'SARIMA':>20} {'LSTM':>20}")
    print("-" * 55)

    for metric_base in ['u_max', 'f_over']:
        sarima_key = f'sarima_{metric_base}'
        lstm_key = f'lstm_{metric_base}'

        sarima = aggregated.get(sarima_key, {})
        lstm = aggregated.get(lstm_key, {})

        sarima_str = f"{sarima.get('mean', 0):.4f} ± {sarima.get('std', 0):.4f}"
        lstm_str = f"{lstm.get('mean', 0):.4f} ± {lstm.get('std', 0):.4f}"

        label = metric_base.replace('_', ' ').title()
        if 'f_over' in metric_base:
            # Convert to percentage
            sarima_str = f"{sarima.get('mean', 0)*100:.2f}% ± {sarima.get('std', 0)*100:.2f}%"
            lstm_str = f"{lstm.get('mean', 0)*100:.2f}% ± {lstm.get('std', 0)*100:.2f}%"
        print(f"{label:<15} {sarima_str:>20} {lstm_str:>20}")

    print("\n" + "=" * 70)


def main():
    """Run multi-seed experiments."""
    parser = argparse.ArgumentParser(description='Run multi-seed experiments')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='Random seeds to use')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/)')
    args = parser.parse_args()

    seeds = args.seeds
    base_dir = args.output_dir or RESULTS_DIR

    print("=" * 70)
    print("Multi-Seed Experiment Runner")
    print("=" * 70)
    print(f"\nSeeds: {seeds}")
    print(f"Output: {base_dir}")

    # Run experiments for each seed
    results = []
    for seed in seeds:
        result = run_single_seed(seed, base_dir)
        results.append(result)

        # Print intermediate result
        if 'error' not in result:
            print(f"\n   [Seed {seed}] SARIMA RMSE: {result['sarima_rmse']:.4f}, LSTM RMSE: {result['lstm_rmse']:.4f}")
        else:
            print(f"\n   [Seed {seed}] FAILED: {result['error']}")

    # Aggregate results
    aggregated = aggregate_multi_seed(results)

    # Save aggregated results
    save_json(aggregated, os.path.join(base_dir, 'aggregated_results.json'))

    # Print summary
    print_multi_seed_summary(aggregated)

    return aggregated


if __name__ == '__main__':
    main()
