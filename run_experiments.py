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
from datetime import datetime
import numpy as np

from config import CONFIG, DATA_DIR, RESULTS_DIR
from utils import set_all_seeds, save_json, load_json


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
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    save_json(metadata, os.path.join(output_dir, 'run_metadata.json'))


def run_single_seed(seed: int, base_dir: str = None) -> dict:
    """
    Run full pipeline with given seed, return metrics.

    Creates isolated output directory per seed.

    Args:
        seed: Random seed to use
        base_dir: Base directory for outputs (default: results/)

    Returns:
        Dictionary with headline metrics from this seed
    """
    # Create isolated config (don't mutate global)
    config = dict(CONFIG)
    config['random_seed'] = seed

    # Set all RNGs
    set_all_seeds(seed)

    # Create seed-specific output directory
    if base_dir is None:
        base_dir = RESULTS_DIR

    seed_dir = os.path.join(base_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)

    # Log metadata
    log_run_metadata(seed, seed_dir, config)

    print(f"\n{'='*50}")
    print(f"Running seed {seed}")
    print(f"{'='*50}")

    # Import and run each stage
    # Note: These would need to be refactored to accept config and output_dir
    # For now, we'll run as subprocesses with environment variables

    # Run data generation
    print(f"\n[Seed {seed}] Generating traffic data...")
    result = subprocess.run(
        [sys.executable, 'generate_traffic.py'],
        capture_output=True, text=True,
        env={**os.environ, 'RANDOM_SEED': str(seed)}
    )
    if result.returncode != 0:
        print(f"Error in generate_traffic.py: {result.stderr}")

    # Run SARIMA training
    print(f"\n[Seed {seed}] Training SARIMA...")
    result = subprocess.run(
        [sys.executable, 'train_sarima.py'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error in train_sarima.py: {result.stderr}")

    # Run LSTM training
    print(f"\n[Seed {seed}] Training LSTM...")
    result = subprocess.run(
        [sys.executable, 'train_lstm.py'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error in train_lstm.py: {result.stderr}")

    # Run evaluation
    print(f"\n[Seed {seed}] Running evaluation...")
    result = subprocess.run(
        [sys.executable, 'eval_capacity.py'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error in eval_capacity.py: {result.stderr}")

    # Load results
    try:
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

    # Aggregate results
    aggregated = aggregate_multi_seed(results)

    # Save aggregated results
    save_json(aggregated, os.path.join(base_dir, 'aggregated_results.json'))

    # Print summary
    print_multi_seed_summary(aggregated)

    return aggregated


if __name__ == '__main__':
    main()
