"""Seasonal-naive baseline trainer.

Predicts `y_hat[t] = y[t - s]` where `s = CONFIG['seasonal_period']`
(default 288 = 1 day at 5-min granularity). No fit; one numpy slice.

Output schema mirrors SARIMA so the runner's existing dispatch
(`_load_predictions_for_model('seasonal_naive', ...)`) just works:
    predictions: (T_test, num_links)
    L_test:      (T_test, num_links)
The runner drops the first `window_size` rows downstream to align with
the neural models' aligned ground truth.

Reference: every benchmarking paper since Hyndman & Athanasopoulos's
*Forecasting: Principles and Practice* recommends seasonal-naive as the
absolute floor for any seasonal series.
"""

from __future__ import annotations

import os

import numpy as np

from .baselines import seasonal_naive_forecast
from .config import CONFIG, RESULTS_DIR, dataset_path
from .utils import aggregate_metrics, compute_metrics, save_json, set_all_seeds


def main():
    print("=" * 50)
    print("Seasonal-naive baseline")
    print("=" * 50)

    set_all_seeds(CONFIG["random_seed"])
    data = np.load(dataset_path())
    L = np.asarray(data["L"], dtype=np.float32)
    val_end = int(data["val_end"])
    T = L.shape[0]
    seasonal_period = CONFIG.get(
        "seasonal_period", CONFIG.get("window_size", 288)
    )

    test_start, test_end = val_end, T
    if test_start - seasonal_period < 0:
        raise ValueError(
            f"seasonal_period={seasonal_period} > test_start={test_start}; "
            "not enough history for seasonal-naive"
        )
    preds = seasonal_naive_forecast(
        L, test_start=test_start, test_end=test_end,
        seasonal_period=seasonal_period,
    ).astype(np.float32)
    L_test = L[test_start:test_end].astype(np.float32)
    print(f"   T={T} test=[{test_start}, {test_end})  s={seasonal_period}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        os.path.join(RESULTS_DIR, "seasonal_naive_predictions.npz"),
        predictions=preds, L_test=L_test,
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
            "seasonal_period": int(seasonal_period),
        },
        os.path.join(RESULTS_DIR, "seasonal_naive_metrics.json"),
    )
    return agg


if __name__ == "__main__":
    main()
