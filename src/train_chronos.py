"""Chronos-Bolt zero-shot forecasting.

Reference: A. F. Ansari et al., "Chronos: Learning the Language of Time
Series," TMLR 2024 (arXiv:2403.07815). The "Bolt" follow-up (Amazon, Sept
2024) is a faster encoder-decoder variant trained on the same LOTSA
corpus. We use it zero-shot — no fine-tuning on our data.

Why we run it:
  - Reviewers in 2026 expect a foundation-model column (Lentini 2025;
    transportation analogue arXiv:2602.24238).
  - Single zero-shot baseline is the cheapest way to demonstrate "does
    pretraining transfer to backbone link loads?"
  - Chronos exposes per-quantile predictions natively, but we use the
    median (q=0.5) here for parity with the point-forecast pipeline.
    Pinball-band CQR/ACI is *not* applied (zero-shot, no pinball training);
    `chronos` is excluded from NEURAL_MODELS in the runner.

Schema parity. We write `chronos_predictions.npz` with the same four keys
the runner expects:
    predictions       — (T_test - window, num_links)
    L_test_aligned    — same shape, ground truth on the test slice.
    val_predictions   — (T_val - window, num_links), zero-shot rolling.
    L_val_aligned     — same shape, ground truth on the val slice.

Lazy import of chronos. The `chronos-forecasting` PyPI package pulls in
~500 MB of weights + transformers; we don't want it as a hard dep on the
laptop. `_load_pipeline()` raises a clear ImportError if missing.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np

from .config import CONFIG, RESULTS_DIR, dataset_path
from .utils import save_json, set_all_seeds


def _load_pipeline(model_name: str, device: Optional[str] = None):
    """Load a pretrained Chronos pipeline; raise a clear error if missing."""
    try:
        from chronos import BaseChronosPipeline
    except ImportError as e:
        raise ImportError(
            "chronos-forecasting is not installed. "
            "Add it on the cloud box: `pip install chronos-forecasting`. "
            "On the laptop, this baseline is intentionally skipped."
        ) from e
    import torch  # imported via chronos but referenced explicitly here

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[chronos] loading {model_name!r} on {device}")
    return BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype="auto",
    )


def _rolling_predictions(
    pipeline,
    L_full: np.ndarray,
    start_idx: int,
    end_idx: int,
    context_length: int,
    batch_size: int = 128,
) -> np.ndarray:
    """Zero-shot rolling one-step-ahead forecasts over `[start_idx, end_idx)`.

    For each prediction time t in [start_idx, end_idx), feed the
    context window `L_full[t - context_length : t, link]` through the
    pipeline and read the median forecast for step t. We loop over links
    (Chronos is univariate); inside each link we batch all `T = end_idx -
    start_idx` predictions in chunks of `batch_size` to amortise the
    transformer forward pass.

    Returns predictions of shape `(end_idx - start_idx, num_links)`.
    """
    if end_idx <= start_idx:
        raise ValueError(f"end_idx={end_idx} <= start_idx={start_idx}")
    if start_idx < context_length:
        raise ValueError(
            f"start_idx={start_idx} < context_length={context_length}; "
            "no history available"
        )

    import torch
    T = end_idx - start_idx
    num_links = L_full.shape[1]
    preds = np.zeros((T, num_links), dtype=np.float32)

    for link in range(num_links):
        link_preds: list[np.ndarray] = []
        for batch_start in range(0, T, batch_size):
            batch_end = min(batch_start + batch_size, T)
            # Build batch of context windows.
            contexts = [
                torch.from_numpy(
                    L_full[start_idx + t - context_length: start_idx + t, link]
                ).float()
                for t in range(batch_start, batch_end)
            ]
            # Chronos accepts a list of 1-D tensors as a batched call.
            forecasts = pipeline.predict(
                context=contexts,
                prediction_length=1,
            )
            # `forecasts` shape per the chronos API: (B, num_samples, 1) for
            # the original Chronos, or (B, num_quantiles, 1) for Bolt. Take
            # the median across the second axis.
            arr = forecasts.detach().cpu().numpy() if hasattr(forecasts, "detach") \
                else np.asarray(forecasts)
            # arr shape: (B, K, 1). Median over K.
            link_preds.append(np.median(arr[:, :, 0], axis=1))
        preds[:, link] = np.concatenate(link_preds)
        if (link + 1) % 10 == 0 or link == num_links - 1:
            print(f"[chronos]   link {link + 1}/{num_links} done")
    return preds


def main():
    """Zero-shot Chronos-Bolt forecast on the active dataset."""
    print("=" * 50)
    print("Zero-shot Chronos-Bolt")
    print("=" * 50)

    set_all_seeds(CONFIG["random_seed"])

    model_name = CONFIG.get("chronos_model_name", "amazon/chronos-bolt-tiny")
    context_length = CONFIG.get("chronos_context_length",
                                CONFIG.get("window_size", 72))
    batch_size = CONFIG.get("chronos_batch_size", 128)

    # Load dataset.
    data = np.load(dataset_path())
    L = np.asarray(data["L"], dtype=np.float32)
    train_end = int(data["train_end"])
    val_end = int(data["val_end"])
    T = L.shape[0]
    num_links = L.shape[1]
    print(f"[chronos] dataset T={T} num_links={num_links} "
          f"context_length={context_length}")

    # Load the pipeline once and reuse for val + test.
    pipeline = _load_pipeline(model_name)

    # Predict val: rolling windows starting at train_end + context_length.
    val_start = train_end + context_length
    if val_start >= val_end:
        raise ValueError(
            f"val window too short for context_length={context_length}: "
            f"train_end={train_end}, val_end={val_end}"
        )
    print(f"[chronos] rolling val [{val_start}, {val_end})")
    val_preds = _rolling_predictions(
        pipeline, L, val_start, val_end,
        context_length=context_length, batch_size=batch_size,
    )
    L_val_aligned = L[val_start:val_end]

    # Predict test: same protocol on the test slice.
    test_start = val_end + context_length
    if test_start >= T:
        raise ValueError(
            f"test window too short for context_length={context_length}: "
            f"val_end={val_end}, T={T}"
        )
    print(f"[chronos] rolling test [{test_start}, {T})")
    test_preds = _rolling_predictions(
        pipeline, L, test_start, T,
        context_length=context_length, batch_size=batch_size,
    )
    L_test_aligned = L[test_start:T]

    # Persist.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "chronos_predictions.npz")
    np.savez(
        out_path,
        predictions=test_preds.astype(np.float32),
        L_test_aligned=L_test_aligned.astype(np.float32),
        val_predictions=val_preds.astype(np.float32),
        L_val_aligned=L_val_aligned.astype(np.float32),
    )
    print(f"[chronos] saved → {out_path}")

    # Quick metrics for the standalone case.
    from .utils import aggregate_metrics, compute_metrics
    per_link = compute_metrics(L_test_aligned, test_preds)
    agg = aggregate_metrics(per_link)
    print(f"[chronos] RMSE={agg['rmse_mean']:.4f}  MAE={agg['mae_mean']:.4f}  "
          f"MAPE={agg['mape_mean']:.2f}%")
    save_json(
        {
            "per_link": {
                "rmse": per_link["rmse"].tolist(),
                "mae": per_link["mae"].tolist(),
                "mape": per_link["mape"].tolist(),
            },
            "aggregated": agg,
            "model_name": model_name,
            "context_length": int(context_length),
        },
        os.path.join(RESULTS_DIR, "chronos_metrics.json"),
    )
    return agg


if __name__ == "__main__":
    main()
