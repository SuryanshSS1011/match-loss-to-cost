"""
Baseline forecasting models for network traffic prediction.

Provides simple baseline models for comparison:
- Naive (persistence): y_hat[t] = y[t-1]
- Seasonal Naive: y_hat[t] = y[t-s] where s = seasonal period
- Holt-Winters: Exponential smoothing with trend and seasonality

EVALUATION REGIME:
These baselines are defined for one-step-ahead evaluation on the test window,
where predictions are aligned so each time step uses only past observations.
This matches the per-time-step aligned evaluation used by SARIMA/LSTM.
"""

import os
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import CONFIG, DATA_DIR, RESULTS_DIR
from utils import compute_metrics, aggregate_metrics, save_json


def naive_forecast(L_full: np.ndarray, test_start: int, test_end: int) -> np.ndarray:
    """
    Naive (persistence) forecast: y_hat[t] = y[t-1].

    For one-step-ahead evaluation, this predicts each time step using
    the immediately preceding observation.

    Args:
        L_full: Full traffic data array of shape (T, num_links)
        test_start: Start index of test period
        test_end: End index of test period

    Returns:
        Predictions of shape (test_end - test_start, num_links)
    """
    # Predict L[t] using L[t-1] for t in [test_start, test_end)
    return L_full[test_start - 1 : test_end - 1].copy()


def seasonal_naive_forecast(L_full: np.ndarray, test_start: int, test_end: int,
                             seasonal_period: int = 288) -> np.ndarray:
    """
    Seasonal naive forecast: y_hat[t] = y[t-s] where s = seasonal period.

    Uses the value from exactly one season ago.

    Args:
        L_full: Full traffic data array of shape (T, num_links)
        test_start: Start index of test period
        test_end: End index of test period
        seasonal_period: Seasonal period in time steps (default: 288 = 1 day)

    Returns:
        Predictions of shape (test_end - test_start, num_links)
    """
    # Predict L[t] using L[t - seasonal_period]
    return L_full[test_start - seasonal_period : test_end - seasonal_period].copy()


def holtwinters_forecast(L_train: np.ndarray, horizon: int,
                          seasonal_period: int = 288) -> np.ndarray:
    """
    Holt-Winters exponential smoothing with additive trend and seasonality.

    Args:
        L_train: Training data of shape (T_train, num_links)
        horizon: Number of steps to forecast
        seasonal_period: Seasonal period in time steps

    Returns:
        Predictions of shape (horizon, num_links)
    """
    num_links = L_train.shape[1]
    predictions = np.zeros((horizon, num_links))

    print(f"   Fitting Holt-Winters for {num_links} links...")

    for link in range(num_links):
        if (link + 1) % 10 == 0:
            print(f"   - Link {link + 1}/{num_links}")

        try:
            model = ExponentialSmoothing(
                L_train[:, link],
                seasonal_periods=seasonal_period,
                trend='add',
                seasonal='add',
                damped_trend=True  # Damped trend for more stable forecasts
            )
            fitted = model.fit(optimized=True)
            predictions[:, link] = fitted.forecast(horizon)
        except Exception as e:
            # Fallback to naive if Holt-Winters fails
            print(f"   ⚠ Link {link} HW failed, using naive: {e}")
            predictions[:, link] = L_train[-1, link]

    return predictions


def run_baselines():
    """Run all baseline models and save results."""
    print("=" * 50)
    print("Running Baseline Models")
    print("=" * 50)

    # Load data
    print("\n1. Loading data...")
    data = np.load(os.path.join(DATA_DIR, 'traffic_data.npz'))
    L = data['L']
    train_end = int(data['train_end'])
    val_end = int(data['val_end'])

    L_train = L[:train_end]
    L_test = L[val_end:]
    test_start = val_end
    test_end = L.shape[0]
    horizon = test_end - test_start

    print(f"   - Train length: {train_end}")
    print(f"   - Test length: {horizon}")
    print(f"   - Seasonal period: {CONFIG['seasonal_period']}")

    results = {}

    # 1. Naive (Persistence)
    print("\n2. Running Naive baseline...")
    naive_preds = naive_forecast(L, test_start, test_end)
    naive_metrics = compute_metrics(L_test, naive_preds)
    naive_agg = aggregate_metrics(naive_metrics)
    results['Naive'] = {
        'predictions': naive_preds,
        'per_link': {k: v.tolist() for k, v in naive_metrics.items()},
        'aggregated': naive_agg
    }
    print(f"   - Mean RMSE: {naive_agg['rmse_mean']:.4f}")
    print(f"   - Mean MAE: {naive_agg['mae_mean']:.4f}")

    # 2. Seasonal Naive
    print("\n3. Running Seasonal Naive baseline...")
    seasonal_period = CONFIG['seasonal_period']
    seasonal_preds = seasonal_naive_forecast(L, test_start, test_end, seasonal_period)
    seasonal_metrics = compute_metrics(L_test, seasonal_preds)
    seasonal_agg = aggregate_metrics(seasonal_metrics)
    results['Seasonal_Naive'] = {
        'predictions': seasonal_preds,
        'per_link': {k: v.tolist() for k, v in seasonal_metrics.items()},
        'aggregated': seasonal_agg
    }
    print(f"   - Mean RMSE: {seasonal_agg['rmse_mean']:.4f}")
    print(f"   - Mean MAE: {seasonal_agg['mae_mean']:.4f}")

    # 3. Holt-Winters
    print("\n4. Running Holt-Winters baseline...")
    hw_preds = holtwinters_forecast(L_train, horizon, seasonal_period)
    hw_metrics = compute_metrics(L_test, hw_preds)
    hw_agg = aggregate_metrics(hw_metrics)
    results['Holt_Winters'] = {
        'predictions': hw_preds,
        'per_link': {k: v.tolist() for k, v in hw_metrics.items()},
        'aggregated': hw_agg
    }
    print(f"   - Mean RMSE: {hw_agg['rmse_mean']:.4f}")
    print(f"   - Mean MAE: {hw_agg['mae_mean']:.4f}")

    # Save predictions
    print("\n5. Saving results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.savez(
        os.path.join(RESULTS_DIR, 'baseline_predictions.npz'),
        naive=naive_preds,
        seasonal_naive=seasonal_preds,
        holt_winters=hw_preds,
        L_test=L_test
    )

    # Save metrics (without predictions to keep JSON small)
    baseline_metrics = {
        'Naive': {
            'per_link': results['Naive']['per_link'],
            'aggregated': results['Naive']['aggregated']
        },
        'Seasonal_Naive': {
            'per_link': results['Seasonal_Naive']['per_link'],
            'aggregated': results['Seasonal_Naive']['aggregated']
        },
        'Holt_Winters': {
            'per_link': results['Holt_Winters']['per_link'],
            'aggregated': results['Holt_Winters']['aggregated']
        }
    }
    save_json(baseline_metrics, os.path.join(RESULTS_DIR, 'baseline_metrics.json'))

    # Print summary
    print("\n" + "=" * 60)
    print("       BASELINE COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'MAPE (%)':<10}")
    print("-" * 50)

    for name in ['Naive', 'Seasonal_Naive', 'Holt_Winters']:
        agg = results[name]['aggregated']
        print(f"{name:<20} {agg['rmse_mean']:<10.4f} {agg['mae_mean']:<10.4f} "
              f"{agg['mape_mean']:<10.2f}")

    print("\n" + "=" * 50)
    print("Baseline models complete!")
    print("=" * 50)

    return results


if __name__ == '__main__':
    run_baselines()
