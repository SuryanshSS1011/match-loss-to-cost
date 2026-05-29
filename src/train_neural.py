"""Generic neural-net training loop, shared by LSTM/DLinear/PatchTST/etc.

Pulled out of `train_lstm.py` so we don't have N copy-pastes of the same
data prep / training loop / prediction / artefact-saving code. Each
specific model file (`train_lstm.py`, `train_dlinear.py`, ...) is now a
~5-line wrapper that supplies a `build_model(num_links) -> nn.Module`
closure plus a string `model_name` used for artefact filenames.

Output artefacts (per model_name `M`):
    models/M_forecaster.pt          — checkpoint with state_dict + config + history
    results/M_predictions.npz       — denormalized predictions + L_test_aligned
    results/M_metrics.json          — per-link + aggregated forecasting metrics
    results/normalization_stats.json — train mean/std (overwritten per call;
                                       all neural models normalise the same way)

The function reads loss config from `src.config.CONFIG`:
    lstm_loss, loss_alpha, loss_beta, loss_tau     — same keys the runner
                                                      already overrides.
The 'lstm_' prefix is a legacy name kept stable so the runner override path
doesn't have to change.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import CONFIG, MODELS_DIR, RESULTS_DIR, dataset_path
from .losses import make_loss
from .utils import (
    aggregate_metrics,
    compute_metrics,
    make_sequences,
    save_json,
    set_all_seeds,
)


def _pick_device() -> torch.device:
    # PROVISION_AWARE_DEVICE=cpu lets a caller force CPU for one trainer
    # without disabling MPS globally. Used for PatchTST/Abilene where the
    # MPSGraph constructor hits INT_MAX on the long-time-axis dataset.
    forced = os.environ.get("PROVISION_AWARE_DEVICE", "").lower()
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_and_prepare_data(window_size: int):
    """Load CONFIG['dataset'], normalise on train only, make sequences."""
    data = np.load(dataset_path())
    L = data["L"]
    train_end = int(data["train_end"])
    val_end = int(data["val_end"])
    # `R` is the routing matrix (num_links, num_od). Some loaders save it,
    # some (synthetic) don't — graph-aware models read it via `dataset_R()`.
    _ = data  # keep `data` alive while we pull R later
    R = data["R"] if "R" in data.files else None

    L_train = L[:train_end]
    L_val = L[train_end:val_end]
    L_test = L[val_end:]

    mean = L_train.mean(axis=0)
    std = L_train.std(axis=0)
    std[std < 1e-6] = 1.0

    L_train_norm = (L_train - mean) / std
    L_val_norm = (L_val - mean) / std
    L_test_norm = (L_test - mean) / std

    X_train, y_train = make_sequences(L_train_norm, window_size)
    X_val, y_val = make_sequences(L_val_norm, window_size)
    X_test, y_test = make_sequences(L_test_norm, window_size)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test)

    batch = CONFIG["lstm_batch_size"]
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=batch, shuffle=False
    )

    L_test_aligned = L_test[window_size:]
    L_val_aligned = L_val[window_size:]
    return (train_loader, val_loader,
            X_val_t, L_val_aligned,
            X_test_t, L_test_aligned,
            mean, std, train_end, R)


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def _validate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            loss = criterion(model(X), y)
            total += float(loss.item())
            n += 1
    return total / max(n, 1)


def _train(model, train_loader, val_loader, device, config):
    criterion = make_loss(
        config.get("lstm_loss", "mse"),
        alpha=config.get("loss_alpha"),
        beta=config.get("loss_beta"),
        tau=config.get("loss_tau"),
    )
    print(f"   Loss: {criterion}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lstm_lr"])
    epochs = config["lstm_epochs"]
    patience = config["lstm_patience"]

    best_val, best_state = float("inf"), None
    waited = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n   Training for up to {epochs} epochs (patience={patience})...")
    for epoch in range(epochs):
        tl = _train_epoch(model, train_loader, criterion, optimizer, device)
        vl = _validate(model, val_loader, criterion, device)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"   epoch {epoch+1:3d}: train={tl:.6f}  val={vl:.6f}")
        if vl < best_val:
            best_val, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
        if waited >= patience:
            print(f"   early stop @ epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"   best val: {best_val:.6f}")
    return model, history


def _predict(model, X_test, mean, std, device):
    model.eval()
    with torch.no_grad():
        out = model(X_test.to(device)).cpu().numpy()
    return out * std + mean


def run(model_name: str,
        build_model: Callable[..., nn.Module]) -> dict:
    """Train + evaluate a neural forecaster; write artefacts under model_name.

    Args:
        model_name: lowercase identifier used for artefact filenames
            (`models/<model_name>_forecaster.pt`, etc.).
        build_model: callable returning an nn.Module that maps
            `(batch, window_size, num_links)` → `(batch, num_links)`.
            Receives `num_links` positionally and `R=...` as a keyword
            argument (the routing matrix from the dataset npz, or None).
            Builders that don't need R accept it via `**_` and ignore.

    Returns the aggregated metric dict.
    """
    print("=" * 50)
    print(f"Training {model_name.upper()} model")
    print("=" * 50)

    set_all_seeds(CONFIG["random_seed"])
    device = _pick_device()
    print(f"\n   device={device}")

    window = CONFIG["window_size"]
    (train_loader, val_loader,
     X_val, L_val_aligned,
     X_test, L_test_aligned,
     mean, std, train_end, R) = _load_and_prepare_data(window)
    num_links = mean.shape[0]
    print(f"   window={window}  num_links={num_links}  "
          f"train_batches={len(train_loader)}  val_batches={len(val_loader)}  "
          f"val_samples={len(X_val)}  test_samples={len(X_test)}  "
          f"R={'present' if R is not None else 'absent'}")

    model = build_model(num_links, R=R).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   {model_name} parameters: {n_params:,}")

    model, history = _train(model, train_loader, val_loader, device, CONFIG)

    # Save checkpoint.
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_forecaster.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": dict(CONFIG),
        "mean": mean,
        "std": std,
        "history": history,
    }, ckpt_path)
    print(f"   ckpt → {ckpt_path}")

    # Save normalisation stats (overwrites; same for any neural model).
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_json(
        {"mean": mean.tolist(), "std": std.tolist(), "train_end": int(train_end)},
        os.path.join(RESULTS_DIR, "normalization_stats.json"),
    )

    # Predict on val + test, persist both. The val arrays are needed for
    # downstream conformal calibration (CQR/ACI use the val window as the
    # held-out calibration set).
    preds = _predict(model, X_test, mean, std, device)
    val_preds = _predict(model, X_val, mean, std, device)
    np.savez(
        os.path.join(RESULTS_DIR, f"{model_name}_predictions.npz"),
        predictions=preds.astype(np.float32),
        L_test_aligned=L_test_aligned.astype(np.float32),
        val_predictions=val_preds.astype(np.float32),
        L_val_aligned=L_val_aligned.astype(np.float32),
    )

    # Forecasting metrics.
    per_link = compute_metrics(L_test_aligned, preds)
    agg = aggregate_metrics(per_link)
    print(f"\n   {model_name.upper()} metrics: "
          f"RMSE={agg['rmse_mean']:.4f}  MAE={agg['mae_mean']:.4f}  "
          f"MAPE={agg['mape_mean']:.2f}%")

    save_json(
        {
            "per_link": {
                "rmse": per_link["rmse"].tolist(),
                "mae": per_link["mae"].tolist(),
                "mape": per_link["mape"].tolist(),
            },
            "aggregated": agg,
        },
        os.path.join(RESULTS_DIR, f"{model_name}_metrics.json"),
    )
    return agg
