"""
Capacity planning evaluation for network traffic forecasting.

Compares how different forecasting models affect capacity planning decisions:
- Computes capacities based on predicted 95th percentile loads
- Evaluates utilization and overload metrics against true loads
- Generates comparison plots
"""

import os
import numpy as np

from config import CONFIG, DATA_DIR, RESULTS_DIR, PLOTS_DIR
from utils import (
    load_json, save_json, print_summary_table,
    plot_rmse_histogram, plot_metric_comparison, plot_capacity_bars,
    plot_timeseries_comparison
)


def load_predictions():
    """
    Load predictions from both models and align them.

    Returns:
        Dictionary with aligned predictions and true values
    """
    # Load SARIMA predictions
    sarima_data = np.load(os.path.join(RESULTS_DIR, 'sarima_predictions.npz'))
    sarima_preds = sarima_data['predictions']
    sarima_L_test = sarima_data['L_test']

    # Load LSTM predictions
    lstm_data = np.load(os.path.join(RESULTS_DIR, 'lstm_predictions.npz'))
    lstm_preds = lstm_data['predictions']
    lstm_L_test = lstm_data['L_test_aligned']

    # SARIMA predicts all T_test steps
    # LSTM predicts T_test - window_size steps (starting at window_size offset)
    window_size = CONFIG['window_size']

    # Align SARIMA predictions to match LSTM
    sarima_preds_aligned = sarima_preds[window_size:]
    sarima_L_test_aligned = sarima_L_test[window_size:]

    # Verify alignment
    assert sarima_preds_aligned.shape == lstm_preds.shape, \
        f"Shape mismatch: SARIMA {sarima_preds_aligned.shape} vs LSTM {lstm_preds.shape}"
    assert np.allclose(sarima_L_test_aligned, lstm_L_test), \
        "True values mismatch between SARIMA and LSTM"

    return {
        'Y_true': lstm_L_test,
        'Y_pred_sarima': sarima_preds_aligned,
        'Y_pred_lstm': lstm_preds,
        'T_eff': lstm_preds.shape[0],
        'num_links': lstm_preds.shape[1]
    }


def compute_capacities(predictions: dict, margin: float = 1.1) -> dict:
    """
    Compute link capacities based on predicted loads.

    For each model, capacity = margin * max(predicted_loads)
    per paper definition: C_ℓ = α · max_{t∈eval} ŷ_ℓ(t)

    Arrays are expected to be shaped (T_eval, num_links) where axis=0 is time.
    Uses nanmax to safely handle any missing values.

    Args:
        predictions: Dictionary with aligned predictions
        margin: Safety margin multiplier α (default 1.1 = 10%)

    Returns:
        Dictionary mapping model name to capacity array
    """
    Y_true = predictions['Y_true']
    Y_pred_sarima = predictions['Y_pred_sarima']
    Y_pred_lstm = predictions['Y_pred_lstm']
    T_eff = predictions['T_eff']

    # Verify expected shape: (T_eval, num_links) where axis=0 is time
    assert Y_true.shape[0] == T_eff, \
        f"Shape mismatch: Y_true has {Y_true.shape[0]} rows but T_eff={T_eff}. " \
        f"Expected shape (T_eff, num_links)."

    capacities = {}

    # SARIMA capacities: C = α × max_t(predictions)
    capacities['SARIMA'] = margin * np.nanmax(Y_pred_sarima, axis=0)

    # LSTM capacities: C = α × max_t(predictions)
    capacities['LSTM'] = margin * np.nanmax(Y_pred_lstm, axis=0)

    # Oracle capacities: C = α × max_t(true_values)
    capacities['Oracle'] = margin * np.nanmax(Y_true, axis=0)

    return capacities


def compute_utilization_metrics(Y_true: np.ndarray, capacities: dict) -> dict:
    """
    Compute utilization and overload metrics for each model.

    Args:
        Y_true: True loads of shape (T, num_links)
        capacities: Dictionary mapping model name to capacity array

    Returns:
        Dictionary mapping model name to metrics dictionary
    """
    metrics = {}

    for model_name, cap in capacities.items():
        # Avoid division by zero
        cap_safe = np.maximum(cap, 1e-6)

        # Per-time-step utilization
        utilization = Y_true / cap_safe  # Shape: (T, num_links)

        # Max utilization per link
        u_max = utilization.max(axis=0)  # Shape: (num_links,)

        # Fraction of time in overload per link
        f_over = (utilization > 1.0).mean(axis=0)  # Shape: (num_links,)

        # Aggregate metrics
        metrics[model_name] = {
            'u_max_per_link': u_max.tolist(),
            'f_over_per_link': f_over.tolist(),
            'u_max_mean': float(u_max.mean()),
            'u_max_max': float(u_max.max()),
            'u_max_median': float(np.median(u_max)),
            'f_over_mean': float(f_over.mean()),
            'f_over_max': float(f_over.max()),
            'links_over_110': int((u_max > 1.1).sum()),  # Badly under-provisioned
            'links_over_100': int((u_max > 1.0).sum()),  # Any overload
        }

    return metrics


def verify_oracle_invariants(Y_true: np.ndarray, capacities: dict, margin: float):
    """
    Verify oracle capacity invariants per paper definition.

    With C_oracle = α × max(Y_true):
    - Overload condition: Y_true > C_oracle (equivalently utilization > 1)
    - All utilizations must be <= 1/α (no overload possible)
    - Max utilization per link should be close to 1/α
    """
    C_oracle = capacities['Oracle']
    expected = 1.0 / margin  # Should be ~0.909 for α=1.1
    tol = 1e-6

    # Per-link max utilization
    utilization = Y_true / np.maximum(C_oracle, 1e-6)
    u_max_per_link = np.nanmax(utilization, axis=0)

    # Primary invariant: no utilization can exceed 1/α + tolerance
    assert np.all(u_max_per_link <= expected + tol), \
        f"Oracle invariant violated: some u_max > 1/α ({expected:.6f}). " \
        f"Max observed: {np.max(u_max_per_link):.6f}"

    # Secondary check: no overload events (utilization > 1.0)
    # Use nansum for full NaN-safety (NaN > 1.0 is False, but explicit is better)
    overload_count = int(np.nansum(utilization > 1.0))
    assert overload_count == 0, \
        f"Oracle invariant violated: {overload_count} overload events (should be 0)"

    # Informational: report actual values
    print(f"   ✓ Oracle invariants verified:")
    print(f"     - Expected u_max = {expected:.6f}")
    print(f"     - Actual mean u_max = {np.mean(u_max_per_link):.6f}")
    print(f"     - Actual max u_max = {np.max(u_max_per_link):.6f}")
    print(f"     - Overload events = 0")


def generate_plots(predictions: dict, capacities: dict,
                   forecasting_metrics: dict, capacity_metrics: dict):
    """Generate all comparison plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    Y_true = predictions['Y_true']
    Y_pred_sarima = predictions['Y_pred_sarima']
    Y_pred_lstm = predictions['Y_pred_lstm']
    T_eff = predictions['T_eff']

    # 1. RMSE histogram comparison
    print("   - Generating RMSE histogram...")
    sarima_metrics = load_json(os.path.join(RESULTS_DIR, 'sarima_metrics.json'))
    lstm_metrics = load_json(os.path.join(RESULTS_DIR, 'lstm_metrics.json'))

    rmse_dict = {
        'SARIMA': np.array(sarima_metrics['per_link']['rmse']),
        'LSTM': np.array(lstm_metrics['per_link']['rmse'])
    }
    plot_rmse_histogram(rmse_dict, os.path.join(PLOTS_DIR, 'rmse_histogram.png'))

    # 2. Forecasting metrics comparison
    print("   - Generating forecasting metrics comparison...")
    for metric in ['rmse_mean', 'mae_mean', 'mape_mean']:
        plot_metric_comparison(
            forecasting_metrics, metric,
            f'Forecasting Comparison: {metric.replace("_", " ").title()}',
            os.path.join(PLOTS_DIR, f'forecast_{metric}.png')
        )

    # 3. Capacity planning metrics comparison
    print("   - Generating capacity planning comparison...")
    for metric in ['u_max_mean', 'u_max_max', 'f_over_mean']:
        title = f'Capacity Planning: {metric.replace("_", " ").title()}'
        plot_capacity_bars(
            capacity_metrics, metric, title,
            os.path.join(PLOTS_DIR, f'capacity_{metric}.png')
        )

    # 4. Time series examples for representative links
    print("   - Generating time series examples...")

    # Find links with different characteristics
    sarima_rmse = np.array(sarima_metrics['per_link']['rmse'])
    lstm_rmse = np.array(lstm_metrics['per_link']['rmse'])

    # Link where LSTM does better
    diff = sarima_rmse - lstm_rmse
    best_lstm_link = np.argmax(diff)

    # Link where SARIMA does better
    best_sarima_link = np.argmin(diff)

    # Median performance link
    median_idx = np.argsort(sarima_rmse)[len(sarima_rmse) // 2]

    example_links = [
        (best_lstm_link, 'lstm_better'),
        (best_sarima_link, 'sarima_better'),
        (median_idx, 'median_perf')
    ]

    time_indices = np.arange(T_eff)

    for link_idx, label in example_links:
        predictions_dict = {
            'SARIMA': Y_pred_sarima[:, link_idx],
            'LSTM': Y_pred_lstm[:, link_idx]
        }
        capacity_dict = {
            'SARIMA': capacities['SARIMA'][link_idx],
            'LSTM': capacities['LSTM'][link_idx],
            'Oracle': capacities['Oracle'][link_idx]
        }
        plot_timeseries_comparison(
            time_indices, Y_true[:, link_idx],
            predictions_dict, link_idx, capacity_dict,
            os.path.join(PLOTS_DIR, f'timeseries_link_{link_idx}_{label}.png')
        )


def sweep_alpha(predictions: dict, alphas: list) -> dict:
    """
    Compute capacity metrics for multiple alpha (safety margin) values.

    Args:
        predictions: Dictionary with aligned predictions
        alphas: List of alpha values to sweep

    Returns:
        Dictionary mapping model name to list of metrics per alpha
    """
    results = {}
    Y_true = predictions['Y_true']

    for alpha in alphas:
        capacities = compute_capacities(predictions, margin=alpha)
        metrics = compute_utilization_metrics(Y_true, capacities)

        for model in ['SARIMA', 'LSTM', 'Oracle']:
            if model not in results:
                results[model] = []

            results[model].append({
                'alpha': alpha,
                'f_over_mean': metrics[model]['f_over_mean'],
                'u_max_mean': metrics[model]['u_max_mean'],
                'mean_capacity': float(np.mean(capacities[model])),
                'sum_capacity': float(np.sum(capacities[model])),
            })

    return results


def plot_alpha_tradeoff(sweep_results: dict, save_dir: str):
    """
    Plot overload vs capacity cost and overload vs alpha as separate figures.

    Generates two separate plots for easy LaTeX/paper inclusion:
    - overload_vs_capacity.png: Pareto-style capacity-overload tradeoff
    - overload_vs_alpha.png: Direct alpha sensitivity

    Args:
        sweep_results: Output from sweep_alpha()
        save_dir: Directory to save figures
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    colors = {'SARIMA': 'steelblue', 'LSTM': 'darkorange', 'Oracle': 'green'}

    # Plot 1: Overload vs Capacity Cost (Pareto curve)
    plt.figure(figsize=(6, 5))
    for model in ['SARIMA', 'LSTM', 'Oracle']:
        data = sweep_results[model]
        overloads = [d['f_over_mean'] * 100 for d in data]
        capacities = [d['mean_capacity'] for d in data]
        plt.plot(capacities, overloads, 'o-', color=colors.get(model, 'gray'), label=model)

    plt.xlabel('Mean Capacity (provisioning cost)')
    plt.ylabel('Overload Fraction (%)')
    plt.legend()
    plt.title('Capacity-Overload Tradeoff')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overload_vs_capacity.png'), dpi=150)
    plt.close()

    # Plot 2: Overload vs Alpha
    plt.figure(figsize=(6, 5))
    for model in ['SARIMA', 'LSTM', 'Oracle']:
        data = sweep_results[model]
        overloads = [d['f_over_mean'] * 100 for d in data]
        alphas = [d['alpha'] for d in data]
        plt.plot(alphas, overloads, 'o-', color=colors.get(model, 'gray'), label=model)

    plt.xlabel('Safety Margin α')
    plt.ylabel('Overload Fraction (%)')
    plt.legend()
    plt.title('Overload vs Safety Margin')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overload_vs_alpha.png'), dpi=150)
    plt.close()


def run_alpha_sweep(predictions: dict, alphas: list = None):
    """
    Run alpha sweep and generate plots/tables.

    Sweeps α over [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.50] by default.

    Args:
        predictions: Dictionary with aligned predictions
        alphas: List of alpha values (default: standard sweep)

    Returns:
        Sweep results dictionary
    """
    if alphas is None:
        alphas = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.50]

    print("\n   Running alpha sweep...")
    sweep_results = sweep_alpha(predictions, alphas)

    # Save results to JSON
    save_json(sweep_results, os.path.join(RESULTS_DIR, 'alpha_sweep.json'))
    print(f"   Saved alpha sweep to: {RESULTS_DIR}/alpha_sweep.json")

    # Generate plots (two separate figures for paper inclusion)
    plot_alpha_tradeoff(sweep_results, PLOTS_DIR)
    print(f"   Saved plots to: {PLOTS_DIR}/overload_vs_capacity.png, overload_vs_alpha.png")

    # Print table
    print("\n   Alpha Sweep Results:")
    print(f"   {'Alpha':<8} {'SARIMA f_over':<15} {'LSTM f_over':<15} {'Oracle f_over':<15}")
    print("   " + "-" * 53)

    for i, alpha in enumerate(alphas):
        sarima_f = sweep_results['SARIMA'][i]['f_over_mean'] * 100
        lstm_f = sweep_results['LSTM'][i]['f_over_mean'] * 100
        oracle_f = sweep_results['Oracle'][i]['f_over_mean'] * 100
        print(f"   {alpha:<8.2f} {sarima_f:<15.2f}% {lstm_f:<15.2f}% {oracle_f:<15.2f}%")

    return sweep_results


def main():
    """Run capacity planning evaluation."""
    print("=" * 50)
    print("Capacity Planning Evaluation")
    print("=" * 50)

    # Load and align predictions
    print("\n1. Loading predictions...")
    predictions = load_predictions()
    print(f"   - Effective test length: {predictions['T_eff']}")
    print(f"   - Number of links: {predictions['num_links']}")

    # Compute capacities
    print("\n2. Computing capacities...")
    margin = CONFIG['capacity_margin']
    capacities = compute_capacities(predictions, margin)

    print(f"   Using max with {margin}x margin (α = {margin})")
    for model, cap in capacities.items():
        print(f"   - {model}: mean={cap.mean():.2f}, max={cap.max():.2f}")

    # Compute utilization metrics
    print("\n3. Computing utilization metrics...")
    capacity_metrics = compute_utilization_metrics(
        predictions['Y_true'], capacities
    )

    for model, metrics in capacity_metrics.items():
        print(f"\n   {model}:")
        print(f"     Mean U_max: {metrics['u_max_mean']:.4f}")
        print(f"     Max U_max:  {metrics['u_max_max']:.4f}")
        print(f"     Mean f_over: {metrics['f_over_mean']*100:.2f}%")
        print(f"     Links >110%: {metrics['links_over_110']}")

    # Verify oracle invariants
    verify_oracle_invariants(predictions['Y_true'], capacities, margin)

    # Save capacity metrics
    save_json(capacity_metrics, os.path.join(RESULTS_DIR, 'capacity_planning.json'))

    # Load forecasting metrics for comparison
    print("\n4. Loading forecasting metrics...")
    sarima_metrics = load_json(os.path.join(RESULTS_DIR, 'sarima_metrics.json'))
    lstm_metrics = load_json(os.path.join(RESULTS_DIR, 'lstm_metrics.json'))

    forecasting_metrics = {
        'SARIMA': sarima_metrics['aggregated'],
        'LSTM': lstm_metrics['aggregated']
    }

    # Generate plots
    print("\n5. Generating plots...")
    generate_plots(predictions, capacities, forecasting_metrics, capacity_metrics)
    print(f"   Saved to: {PLOTS_DIR}")

    # Print summary table
    print_summary_table(forecasting_metrics, capacity_metrics)

    # Save combined results
    combined_results = {
        'forecasting': forecasting_metrics,
        'capacity_planning': capacity_metrics,
        'config': {
            'capacity_method': 'max',
            'capacity_margin': margin,
            'window_size': CONFIG['window_size'],
            'T_eff': predictions['T_eff'],
            'num_links': predictions['num_links']
        }
    }
    save_json(combined_results, os.path.join(RESULTS_DIR, 'combined_results.json'))

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)

    return forecasting_metrics, capacity_metrics


if __name__ == '__main__':
    main()
