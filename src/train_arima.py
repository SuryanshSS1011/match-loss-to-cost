"""
SARIMA model training and forecasting for network traffic prediction.

Fits a seasonal ARIMA model independently on each link's time series
and generates forecasts for the test period.
"""

import os
import warnings
import numpy as np
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import CONFIG, DATA_DIR, RESULTS_DIR
from .utils import set_all_seeds, compute_metrics, aggregate_metrics, save_json


def load_link_loads():
    """Load link load data from disk."""
    data = np.load(os.path.join(DATA_DIR, 'traffic_data.npz'))
    L = data['L']
    T_train = int(data['T_train'])
    T_val = int(data['T_val'])
    T_test = int(data['T_test'])
    train_end = int(data['train_end'])
    val_end = int(data['val_end'])

    return L, T_train, T_val, T_test, train_end, val_end


def fit_sarima_single_link(link_data: np.ndarray, order: tuple,
                           seasonal_order: tuple, forecast_steps: int) -> np.ndarray:
    """
    Fit SARIMA model on a single link and generate forecasts.

    Args:
        link_data: Training data for single link (1D array)
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
        forecast_steps: Number of steps to forecast

    Returns:
        Forecasted values (1D array)
    """
    try:
        # Fit SARIMA model
        model = SARIMAX(
            link_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=200)

        # Generate forecasts
        forecast = fitted.get_forecast(steps=forecast_steps)
        predictions = forecast.predicted_mean

        return predictions

    except Exception as e:
        # If fitting fails, return simple mean forecast
        print(f"    Warning: SARIMA fitting failed, using mean forecast. Error: {e}")
        return np.full(forecast_steps, link_data.mean())


def fit_sarima_all_links(L_train: np.ndarray, forecast_steps: int,
                         order: tuple, seasonal_order: tuple,
                         n_jobs: int = -1) -> np.ndarray:
    """
    Fit SARIMA models for all links in parallel.

    Args:
        L_train: Training link loads of shape (T_train, num_links)
        forecast_steps: Number of steps to forecast (T_test)
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        Predictions of shape (forecast_steps, num_links)
    """
    num_links = L_train.shape[1]

    print(f"   Fitting SARIMA models for {num_links} links...")
    print(f"   Order: {order}, Seasonal: {seasonal_order}")
    print(f"   This may take several minutes...")

    # Suppress warnings during fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Parallel fitting
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(fit_sarima_single_link)(
                L_train[:, l], order, seasonal_order, forecast_steps
            )
            for l in range(num_links)
        )

    # Stack results
    predictions = np.column_stack(results)

    return predictions


def main():
    """Train SARIMA models and generate forecasts."""
    print("=" * 50)
    print("Training SARIMA Models")
    print("=" * 50)

    # Set random seed
    set_all_seeds(CONFIG['random_seed'])

    # Load data
    print("\n1. Loading data...")
    L, T_train, T_val, T_test, train_end, val_end = load_link_loads()
    num_links = L.shape[1]
    print(f"   - Total time steps: {L.shape[0]}")
    print(f"   - Number of links: {num_links}")
    print(f"   - Train: {T_train}, Val: {T_val}, Test: {T_test}")

    # Use train + val for SARIMA fitting
    L_train_val = L[:val_end]
    L_test = L[val_end:]

    print(f"\n2. Using train+val for fitting: {len(L_train_val)} time steps")
    print(f"   Forecasting {T_test} test steps")

    # Fit SARIMA and get predictions
    print("\n3. Fitting SARIMA models...")
    order = CONFIG['arima_order']
    seasonal_order = CONFIG['seasonal_order']

    predictions = fit_sarima_all_links(
        L_train_val,
        forecast_steps=T_test,
        order=order,
        seasonal_order=seasonal_order,
        n_jobs=-1  # Use all CPUs 
    )

    print(f"\n   Predictions shape: {predictions.shape}")

    # Save predictions
    print("\n4. Saving predictions...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        os.path.join(RESULTS_DIR, 'sarima_predictions.npz'),
        predictions=predictions,
        L_test=L_test
    )

    # Compute metrics
    print("\n5. Computing metrics...")
    per_link_metrics = compute_metrics(L_test, predictions)
    aggregated = aggregate_metrics(per_link_metrics)

    # Print summary
    print("\n   SARIMA Forecasting Metrics:")
    print(f"   - Mean RMSE: {aggregated['rmse_mean']:.4f}")
    print(f"   - Mean MAE:  {aggregated['mae_mean']:.4f}")
    print(f"   - Mean MAPE: {aggregated['mape_mean']:.2f}%")

    # Save metrics
    metrics_data = {
        'per_link': {
            'rmse': per_link_metrics['rmse'].tolist(),
            'mae': per_link_metrics['mae'].tolist(),
            'mape': per_link_metrics['mape'].tolist()
        },
        'aggregated': aggregated
    }
    save_json(metrics_data, os.path.join(RESULTS_DIR, 'sarima_metrics.json'))

    print("\n" + "=" * 50)
    print("SARIMA training complete!")
    print("=" * 50)

    return predictions, L_test, per_link_metrics, aggregated


if __name__ == '__main__':
    main()
