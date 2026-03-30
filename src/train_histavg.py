"""Historical-average baseline trainer.

Predicts a per-link constant equal to the train-window mean. Cheapest
possible baseline — exists to give reviewers a "predicting the mean
beats your model" sanity floor. Often surprisingly competitive on
overload-rate at large α/β when the data is heavy-tailed.

Output schema: SARIMA-like (predictions, L_test) so the runner's
dispatch handles it.
"""

from __future__ import annotations

import os

import numpy as np

from .config import CONFIG, RESULTS_DIR, dataset_path
from .utils import aggregate_metrics, compute_metrics, save_json, set_all_seeds


def main():
    print("=" * 50)
    print("Historical-average baseline")
    print("=" * 50)

    set_all_seeds(CONFIG["random_seed"])
    data = np.load(dataset_path())
    L = np.asarray(data["L"], dtype=np.float32)
    train_end = int(data["train_end"])
    val_end = int(data["val_end"])
    T = L.shape[0]

    # Per-link mean over the *train* window only — using val or test would
    # be data leakage.
    train_mean = L[:train_end].mean(axis=0)
    L_test = L[val_end:T].astype(np.float32)
    preds = np.broadcast_to(train_mean, L_test.shape).astype(np.float32).copy()
    print(f"   T={T} test=[{val_end}, {T})  mean shape={train_mean.shape}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        os.path.join(RESULTS_DIR, "histavg_predictions.npz"),
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
        },
        os.path.join(RESULTS_DIR, "histavg_metrics.json"),
    )
    return agg


if __name__ == "__main__":
    main()
