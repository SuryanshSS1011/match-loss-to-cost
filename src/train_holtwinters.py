"""Holt-Winters exponential-smoothing baseline trainer.

Single-link Holt-Winters with additive trend and additive seasonality at
`s = CONFIG['seasonal_period']`. Like SARIMA, runs with bounded training
history and a capped test horizon by default to keep wall-clock
manageable on long real-data series:

    holtwinters_train_window  — last N points of train+val for fit (None = full)
    holtwinters_test_steps    — cap forecast horizon (None = full T_test)

Uses statsmodels' `ExponentialSmoothing` per link, falls back to the
last training value if the optimiser blows up. Output schema mirrors
SARIMA (`predictions, L_test`), so the runner's existing dispatch
handles it the same way.

Reference: Holt 1957; Winters 1960. Standard seasonal smoother — should
appear in any "naive baselines" row in a CNSM/NOMS results table.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .config import CONFIG, RESULTS_DIR, dataset_path
from .utils import aggregate_metrics, compute_metrics, save_json, set_all_seeds


def _fit_one(link_train: np.ndarray, horizon: int,
             seasonal_period: int) -> np.ndarray:
    """Fit Holt-Winters on one link, return forecast of length horizon."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                link_train,
                seasonal_periods=seasonal_period,
                trend="add", seasonal="add", damped_trend=True,
            )
            fitted = model.fit(optimized=True)
            return np.asarray(fitted.forecast(horizon), dtype=np.float32)
    except Exception:
        # Most common failure: too few full seasonal cycles in the train
        # window. Fall back to repeating the last observation.
        return np.full(horizon, float(link_train[-1]), dtype=np.float32)


def main():
    print("=" * 50)
    print("Holt-Winters baseline")
    print("=" * 50)

    set_all_seeds(CONFIG["random_seed"])
    data = np.load(dataset_path())
    L = np.asarray(data["L"], dtype=np.float32)
    val_end = int(data["val_end"])
    T = L.shape[0]
    seasonal_period = int(CONFIG.get(
        "seasonal_period", CONFIG.get("window_size", 288)
    ))

    L_train_full = L[:val_end]
    train_window = CONFIG.get("holtwinters_train_window")
    if train_window is not None and train_window < L_train_full.shape[0]:
        L_train = L_train_full[-int(train_window):]
        print(f"   bounding history to last {train_window} steps "
              f"(full = {L_train_full.shape[0]})")
    else:
        L_train = L_train_full

    L_test_full = L[val_end:T]
    test_cap = CONFIG.get("holtwinters_test_steps")
    if test_cap is not None and test_cap < L_test_full.shape[0]:
        print(f"   capping test horizon to {test_cap} "
              f"(full = {L_test_full.shape[0]})")
        L_test = L_test_full[:int(test_cap)]
    else:
        L_test = L_test_full
    horizon = L_test.shape[0]
    num_links = L.shape[1]
    print(f"   train={len(L_train)}  test horizon={horizon}  "
          f"links={num_links}  s={seasonal_period}")

    print("   Fitting per-link Holt-Winters in parallel...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_fit_one)(L_train[:, ell], horizon, seasonal_period)
        for ell in range(num_links)
    )
    preds = np.column_stack(results).astype(np.float32)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        os.path.join(RESULTS_DIR, "holtwinters_predictions.npz"),
        predictions=preds,
        L_test=L_test.astype(np.float32),
    )

    per_link = compute_metrics(L_test, preds)
    agg = aggregate_metrics(per_link)
    print(f"   RMSE={agg['rmse_mean']:.4f}  MAE={agg['mae_mean']:.4f}")
    save_json(
        {
            "per_link": {
                "rmse": per_link["rmse"].tolist(),
                "mae": per_link["mae"].tolist(),
                "mape": per_link["mape"].tolist(),
            },
            "aggregated": agg,
            "seasonal_period": seasonal_period,
            "train_window": train_window,
            "test_steps": test_cap,
        },
        os.path.join(RESULTS_DIR, "holtwinters_metrics.json"),
    )
    return agg


if __name__ == "__main__":
    main()
