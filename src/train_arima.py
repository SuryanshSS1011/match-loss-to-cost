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

from .config import CONFIG, DATA_DIR, RESULTS_DIR, dataset_path
from .utils import set_all_seeds, compute_metrics, aggregate_metrics, save_json


def load_link_loads():
    """Load link load data from disk."""
    data = np.load(dataset_path())
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
    Fit SARIMA on a single link and produce a single open-loop forecast.

    Use only for short horizons. For real-data test sets (>~500 steps) prefer
    `fit_sarima_rolling_single_link()` which walks the test set one step at a time.
    """
    try:
        model = SARIMAX(
            link_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=200)
        forecast = fitted.get_forecast(steps=forecast_steps)
        return forecast.predicted_mean

    except Exception as e:
        print(f"    Warning: SARIMA fitting failed, using mean forecast. Error: {e}")
        return np.full(forecast_steps, link_data.mean())


def fit_sarima_rolling_single_link(train_data: np.ndarray, test_data: np.ndarray,
                                   order: tuple, seasonal_order: tuple) -> np.ndarray:
    """
    Rolling one-step-ahead SARIMA forecast on the test set.

    Fits once on `train_data`, then for each step in `test_data`:
      (1) get the one-step-ahead forecast,
      (2) update the state with the actual observation via `apply()`,
      (3) repeat.

    `apply()` reuses the fitted parameters and only recomputes the filter on the
    extended series — it does not refit, so the cost per step is small.

    Returns predictions array of shape `(len(test_data),)`.
    """
    n_test = len(test_data)
    preds = np.empty(n_test, dtype=np.float32)
    try:
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)

        history = list(train_data)
        current = fitted
        for t in range(n_test):
            preds[t] = float(current.get_forecast(steps=1).predicted_mean[0])
            history.append(float(test_data[t]))
            # apply() reuses the fitted params on the extended series.
            current = current.apply(np.asarray(history, dtype=np.float64), refit=False)
        return preds

    except Exception as e:
        print(f"    Warning: rolling SARIMA failed, using mean forecast. Error: {e}")
        return np.full(n_test, float(np.mean(train_data)), dtype=np.float32)


def fit_sarima_rolling_all_links(L_train: np.ndarray, L_test: np.ndarray,
                                 order: tuple, seasonal_order: tuple,
                                 n_jobs: int = -1) -> np.ndarray:
    """Run rolling one-step-ahead SARIMA forecasts in parallel across links."""
    num_links = L_train.shape[1]
    n_test = L_test.shape[0]
    print(f"   Rolling SARIMA: {num_links} links, train={L_train.shape[0]}, "
          f"test={n_test}, order={order}, seasonal={seasonal_order}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(fit_sarima_rolling_single_link)(
                L_train[:, l], L_test[:, l], order, seasonal_order
            )
            for l in range(num_links)
        )
    return np.column_stack(results)


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

    # Use train + val for SARIMA fitting; optionally truncate to last N points.
    L_train_val_full = L[:val_end]
    train_window = CONFIG.get('sarima_train_window')
    if train_window is not None and train_window < L_train_val_full.shape[0]:
        L_train_val = L_train_val_full[-train_window:]
        print(f"\n2. Bounding SARIMA history to last {train_window} steps "
              f"({train_window*CONFIG['time_step_minutes']/60/24:.1f} days; "
              f"full = {L_train_val_full.shape[0]})")
    else:
        L_train_val = L_train_val_full
    L_test_full = L[val_end:]

    # Optionally truncate the SARIMA test horizon to keep wall-clock manageable.
    cap = CONFIG.get('sarima_test_steps')
    if cap is not None and cap < L_test_full.shape[0]:
        print(f"\n2. Capping SARIMA test horizon to {cap} steps "
              f"(full T_test = {L_test_full.shape[0]})")
        L_test = L_test_full[:cap]
    else:
        L_test = L_test_full
    print(f"   Fitting on {len(L_train_val)} steps")
    print(f"   Forecasting {len(L_test)} test steps")

    # Fit SARIMA and get predictions
    print("\n3. Fitting SARIMA models...")
    order = CONFIG['arima_order']
    seasonal_order = CONFIG['seasonal_order']
    mode = CONFIG.get('sarima_mode', 'oneshot')

    # Optionally fit only a deterministic subset of links; fill the rest with the
    # per-link train mean so the prediction array has uniform (T_test, num_links) shape.
    n_links_total = L_train_val.shape[1]
    subset = CONFIG.get('sarima_link_subset')
    if subset is not None and subset < n_links_total:
        rng = np.random.default_rng(CONFIG['random_seed'])
        fit_links = np.sort(rng.choice(n_links_total, size=subset, replace=False))
        print(f"   Fitting SARIMA on a subset of {subset}/{n_links_total} links: "
              f"{fit_links.tolist()}")
    else:
        fit_links = np.arange(n_links_total)

    L_train_sub = L_train_val[:, fit_links]
    L_test_sub = L_test[:, fit_links]

    n_jobs = CONFIG.get('sarima_n_jobs', -1)
    if mode == 'rolling':
        sub_preds = fit_sarima_rolling_all_links(
            L_train_sub, L_test_sub, order=order, seasonal_order=seasonal_order, n_jobs=n_jobs
        )
    elif mode == 'oneshot':
        sub_preds = fit_sarima_all_links(
            L_train_sub,
            forecast_steps=len(L_test),
            order=order,
            seasonal_order=seasonal_order,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"unknown sarima_mode {mode!r}")

    # Place SARIMA predictions for fit_links; fill remaining links with their train mean.
    predictions = np.broadcast_to(
        L_train_val.mean(axis=0), (len(L_test), n_links_total)
    ).astype(np.float32).copy()
    predictions[:, fit_links] = sub_preds.astype(np.float32)
    fit_mask = np.zeros(n_links_total, dtype=bool)
    fit_mask[fit_links] = True

    print(f"\n   Predictions shape: {predictions.shape}")

    # Save predictions
    print("\n4. Saving predictions...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        os.path.join(RESULTS_DIR, 'sarima_predictions.npz'),
        predictions=predictions,
        L_test=L_test,
        fit_mask=fit_mask,
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
