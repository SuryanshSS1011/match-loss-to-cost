#!/usr/bin/env python
"""Multi-seed runner for network-traffic forecasting experiments.

Drives the v0 critical-path workflow:
    python scripts/run_experiments.py \
        --dataset abilene --loss asym --alpha 5 --beta 1 \
        --seeds 42 123 456 789 1024

For each (loss, seed) cell it:
  1. overrides `src.config.CONFIG` (dataset, seed, loss config),
  2. runs `simulate_data` (only when --dataset synthetic),
  3. runs `train_arima.main()` and `train_lstm.main()`,
  4. loads the saved predictions, computes forecasting *and* operational
     metrics via `src/evaluation/operational.py`,
  5. writes per-seed JSON + an aggregated JSON.

The result schema is the new "Provision-Aware" headline (per Rule 1):
    rmse, mae          — secondary, only to show ties on accuracy.
    overload_rate, sla_violation_rate, over_provisioning_cost,
    asymmetric_op_cost, u_max_mean, u_max_max — headline.

This is *the* harness everything downstream (v0 plot, cloud sweep) calls.
Schema-locked by tests/test_runner_keys.py.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration import (  # noqa: E402
    ACITracker,
    capacity_from_cqr_upper,
    cqr_calibrate,
    cqr_predict,
    empirical_coverage,
)
from src.config import CONFIG, RESULTS_DIR, dataset_path  # noqa: E402
from src.evaluation import operational_metrics  # noqa: E402
from src.utils import (  # noqa: E402
    aggregate_metrics,
    compute_metrics,
    save_json,
    set_all_seeds,
)


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_MODELS = ("sarima", "lstm")
SUPPORTED_MODELS = ("sarima", "lstm", "dlinear", "patchtst",
                    "itransformer", "chronos", "dcrnn",
                    "seasonal_naive", "holtwinters", "histavg")
CALIBRATION_MODES = ("none", "cqr", "aci", "both")
# NEURAL_MODELS = pinball-band-trainable models. Chronos is zero-shot, so
# it is in SUPPORTED_MODELS but NOT in NEURAL_MODELS — calibration mode
# skips it. seasonal_naive / holtwinters / histavg are statistical
# baselines and likewise excluded.
NEURAL_MODELS = ("lstm", "dlinear", "patchtst", "itransformer", "dcrnn")
# Models that use SARIMA's npz schema (predictions, L_test) — i.e. the
# full-test-window forecast convention rather than the neural models'
# (predictions aligned to L_test[window:], val_predictions, L_val_aligned).
SARIMA_LIKE_MODELS = ("sarima", "seasonal_naive", "holtwinters", "histavg")

# Headline keys produced per (model, seed). Tests pin this list.
OPERATIONAL_KEYS = (
    "overload_rate",
    "sla_violation_rate",
    "over_provisioning_cost",
    "asymmetric_op_cost",
    "u_max_mean",
    "u_max_max",
)
FORECAST_KEYS = ("rmse_mean", "mae_mean", "mape_mean", "smape_mean")
CALIBRATION_KEYS = (
    "target_alpha",
    "qhat_mean",
    "coverage_overall",
    "mean_width",
)


def get_git_info() -> dict:
    try:
        commit = subprocess.getoutput("git rev-parse HEAD")
        dirty = subprocess.getoutput("git status --porcelain") != ""
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": True}


def _override_config(seed: int, dataset: str, loss: str,
                     alpha: Optional[float], beta: Optional[float],
                     tau: Optional[float]) -> None:
    """Mutate the shared CONFIG dict before downstream modules import it."""
    import src.config as config_module

    config_module.CONFIG["random_seed"] = seed
    config_module.CONFIG["dataset"] = dataset
    config_module.CONFIG["lstm_loss"] = loss
    if alpha is not None:
        config_module.CONFIG["loss_alpha"] = alpha
    if beta is not None:
        config_module.CONFIG["loss_beta"] = beta
    if tau is not None:
        config_module.CONFIG["loss_tau"] = tau


def _load_predictions_for_model(model: str, window_size: int) -> dict:
    """Load one model's saved predictions. Returns {preds, L_test_aligned}.

    Neural models write `<model>_predictions.npz` with `predictions`
    already aligned to `L_test[window_size:]`. SARIMA-like models
    (sarima, seasonal_naive, holtwinters, histavg) write `predictions` +
    `L_test` covering the *full* test window starting at `val_end`; we
    drop the first `window_size` rows to align with the neural
    ground-truth slice.
    """
    if model in SARIMA_LIKE_MODELS:
        npz = np.load(os.path.join(RESULTS_DIR, f"{model}_predictions.npz"))
        preds_full = np.asarray(npz["predictions"], dtype=np.float32)
        L_full = np.asarray(npz["L_test"], dtype=np.float32)
        return {
            "preds": preds_full[window_size:],
            "L_test_aligned": L_full[window_size:],
        }
    npz = np.load(os.path.join(RESULTS_DIR, f"{model}_predictions.npz"))
    return {
        "preds": np.asarray(npz["predictions"], dtype=np.float32),
        "L_test_aligned": np.asarray(npz["L_test_aligned"], dtype=np.float32),
    }


def _align_models(model_preds: dict) -> dict:
    """Trim each model's predictions to the shortest common test horizon.

    Returns {L_test_aligned, T_eff, num_links, models: {name: preds_array}}.
    All arrays are float32 and have the first `T_eff` rows.
    """
    if not model_preds:
        raise ValueError("no model predictions to align")
    T_effs = [v["preds"].shape[0] for v in model_preds.values()]
    L_lens = [v["L_test_aligned"].shape[0] for v in model_preds.values()]
    T_eff = min(T_effs + L_lens)

    # All models should agree on L_test_aligned[:T_eff] (same dataset, same
    # window). Take it from any model's npz.
    any_name = next(iter(model_preds))
    L_aligned = model_preds[any_name]["L_test_aligned"][:T_eff]
    out_models = {name: v["preds"][:T_eff] for name, v in model_preds.items()}
    return {
        "L_test_aligned": L_aligned,
        "T_eff": T_eff,
        "num_links": int(L_aligned.shape[1]),
        "models": out_models,
    }


def _per_model_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       alpha: float, beta: float, margin: float) -> dict:
    """Compute forecasting + operational metrics for one model's predictions."""
    forecast = aggregate_metrics(compute_metrics(y_true, y_pred))
    cap = margin * np.nanmax(y_pred, axis=0)
    op = operational_metrics(y_true, cap, alpha=alpha, beta=beta)
    return {
        "forecast": {k: forecast.get(k) for k in FORECAST_KEYS},
        "operational": {k: op[k] for k in OPERATIONAL_KEYS},
    }


_MODEL_TRAINERS = {
    "sarima": "src.train_arima",
    "lstm": "src.train_lstm",
    "dlinear": "src.train_dlinear",
    "patchtst": "src.train_patchtst",
    "itransformer": "src.train_itransformer",
    "chronos": "src.train_chronos",
    "dcrnn": "src.train_dcrnn",
    "seasonal_naive": "src.train_seasonal_naive",
    "holtwinters": "src.train_holtwinters",
    "histavg": "src.train_histavg",
}


def _run_trainer(model_name: str) -> None:
    """Import the trainer module fresh and call its `main()`.

    `importlib.reload` ensures the module re-reads the (mutated) CONFIG.
    """
    module_path = _MODEL_TRAINERS[model_name]
    mod = importlib.import_module(module_path)
    importlib.reload(mod)
    mod.main()


def _model_display_name(model: str) -> str:
    return {
        "sarima": "SARIMA", "lstm": "LSTM",
        "dlinear": "DLinear", "patchtst": "PatchTST",
        "itransformer": "iTransformer", "chronos": "Chronos",
        "dcrnn": "DCRNN",
        "seasonal_naive": "SeasonalNaive",
        "holtwinters": "HoltWinters",
        "histavg": "HistAvg",
    }.get(model, model.upper())


def _train_pinball_band(model_name: str, tau_lo: float, tau_hi: float) -> None:
    """Train one neural model twice (at tau_lo and tau_hi) for CQR/ACI bands.

    Side effect: writes `<model>_qlo_predictions.npz` and
    `<model>_qhi_predictions.npz` under RESULTS_DIR. Overrides
    `lstm_loss` / `loss_tau` between calls.
    """
    import src.config as config_module
    cfg = config_module.CONFIG
    saved = (cfg.get("lstm_loss"),
             cfg.get("loss_tau"),
             cfg.get("loss_alpha"),
             cfg.get("loss_beta"))

    try:
        for suffix, tau in (("qlo", tau_lo), ("qhi", tau_hi)):
            cfg["lstm_loss"] = "pinball"
            cfg["loss_tau"] = tau
            # alpha/beta are unused for pinball but pass them anyway so the
            # factory doesn't raise.
            print(f"\n  [calibration] training {model_name} at tau={tau}")
            _run_trainer(model_name)
            # Rename the just-written predictions npz to a band-specific name.
            src_p = os.path.join(RESULTS_DIR, f"{model_name}_predictions.npz")
            dst_p = os.path.join(RESULTS_DIR, f"{model_name}_{suffix}_predictions.npz")
            if os.path.exists(src_p):
                shutil.copy2(src_p, dst_p)
    finally:
        (cfg["lstm_loss"], cfg["loss_tau"],
         cfg["loss_alpha"], cfg["loss_beta"]) = saved


def _load_band_predictions(model_name: str) -> dict:
    """Load the (q_lo, q_hi) val + test arrays written by `_train_pinball_band`."""
    qlo = np.load(os.path.join(RESULTS_DIR, f"{model_name}_qlo_predictions.npz"))
    qhi = np.load(os.path.join(RESULTS_DIR, f"{model_name}_qhi_predictions.npz"))
    return {
        "q_lo_val": np.asarray(qlo["val_predictions"], dtype=np.float32),
        "q_hi_val": np.asarray(qhi["val_predictions"], dtype=np.float32),
        "L_val_aligned": np.asarray(qlo["L_val_aligned"], dtype=np.float32),
        "q_lo_test": np.asarray(qlo["predictions"], dtype=np.float32),
        "q_hi_test": np.asarray(qhi["predictions"], dtype=np.float32),
        "L_test_aligned": np.asarray(qlo["L_test_aligned"], dtype=np.float32),
    }


def _apply_cqr(bands: dict, target_alpha: float, op_alpha: float, op_beta: float
               ) -> dict:
    """Run split CQR on the loaded bands; return calibration + operational dict.

    Capacity is taken from the CQR upper edge with `margin=1.0` since the
    upper edge already absorbs the safety margin via qhat.
    """
    qhat = cqr_calibrate(
        bands["L_val_aligned"], bands["q_lo_val"], bands["q_hi_val"],
        alpha=target_alpha, per_link=True,
    )
    lower, upper = cqr_predict(bands["q_lo_test"], bands["q_hi_test"], qhat)
    cov = empirical_coverage(bands["L_test_aligned"], lower, upper)
    cap = capacity_from_cqr_upper(upper, margin=1.0)
    op = operational_metrics(bands["L_test_aligned"], cap,
                             alpha=op_alpha, beta=op_beta)
    return {
        "calibration": {
            "method": "cqr",
            "target_alpha": float(target_alpha),
            "qhat_mean": float(np.mean(qhat)),
            "qhat_per_link": qhat.astype(np.float64).tolist(),
            "coverage_overall": cov["coverage_overall"],
            "mean_width": cov["mean_width"],
        },
        "operational": {k: op[k] for k in OPERATIONAL_KEYS},
    }


def _apply_aci(bands: dict, target_alpha: float, op_alpha: float, op_beta: float,
               gamma: float, window: int) -> dict:
    """Stream ACI through the test set; return calibration + operational dict.

    The ACI tracker is warm-started from the val-set scores so the first
    `window` test steps don't return an unbounded band — same warm-start
    suggestion noted in the cqr STEPS entry.
    """
    L_val = bands["L_val_aligned"]
    L_test = bands["L_test_aligned"]
    q_lo_v, q_hi_v = bands["q_lo_val"], bands["q_hi_val"]
    q_lo_t, q_hi_t = bands["q_lo_test"], bands["q_hi_test"]
    num_links = L_test.shape[1]

    tracker = ACITracker(num_links=num_links, target_alpha=target_alpha,
                         gamma=gamma, window=window)

    # Warm-start: feed val-set scores in order, with in_band=True so alpha
    # stays near the target while we populate the score window.
    val_scores = np.maximum(q_lo_v - L_val, L_val - q_hi_v)
    n_warm = min(window, val_scores.shape[0])
    warm_idxs = np.linspace(0, val_scores.shape[0] - 1, num=n_warm, dtype=int)
    for t in warm_idxs:
        tracker.update(val_scores[t], np.ones(num_links, dtype=bool))

    # Stream test set.
    T = L_test.shape[0]
    upper = np.zeros((T, num_links), dtype=np.float32)
    lower = np.zeros_like(upper)
    in_band_arr = np.zeros((T, num_links), dtype=bool)
    for t in range(T):
        qh = tracker.qhat()
        # Replace +inf qhat (cold tracker) with a large finite cap to avoid
        # overflowing the np.float32 storage; finite-but-huge is the right
        # operational meaning.
        qh = np.where(np.isfinite(qh), qh, 1e6)
        u = q_hi_t[t] + qh
        l = q_lo_t[t] - qh
        upper[t] = u.astype(np.float32)
        lower[t] = l.astype(np.float32)
        scores_t = np.maximum(q_lo_t[t] - L_test[t], L_test[t] - q_hi_t[t])
        in_band = (L_test[t] >= l) & (L_test[t] <= u)
        in_band_arr[t] = in_band
        tracker.update(scores_t, in_band)

    cov_overall = float(in_band_arr.mean())
    width = upper - lower
    cap = (np.nanmax(upper, axis=0)).astype(np.float64)
    op = operational_metrics(L_test, cap, alpha=op_alpha, beta=op_beta)
    return {
        "calibration": {
            "method": "aci",
            "target_alpha": float(target_alpha),
            "gamma": float(gamma),
            "window": int(window),
            "qhat_mean": float(np.mean(tracker.qhat()[np.isfinite(tracker.qhat())]
                                       if np.any(np.isfinite(tracker.qhat()))
                                       else [0.0])),
            "alpha_final_mean": float(tracker.alpha.mean()),
            "coverage_overall": cov_overall,
            "mean_width": float(width.mean()),
        },
        "operational": {k: op[k] for k in OPERATIONAL_KEYS},
    }


def run_single_seed(seed: int, *, dataset: str, loss: str,
                    alpha: float, beta: float, tau: Optional[float],
                    base_dir: str,
                    models: tuple = DEFAULT_MODELS,
                    calibration: str = "none",
                    target_alpha: float = 0.1,
                    tau_lo: float = 0.05,
                    tau_hi: float = 0.95,
                    aci_gamma: float = 0.005,
                    aci_window: int = 500) -> dict:
    """Run the pipeline for one seed and return its metrics dict.

    Side effects: writes `<base_dir>/seed_<seed>/{run_metadata,results}.json`
    plus per-seed prediction-npz copies for each requested model.

    `calibration` ∈ {none, cqr, aci, both}. When != none, each requested
    *neural* model also gets two extra pinball-loss trainings at tau_lo and
    tau_hi to produce the (q_lo, q_hi) bands that CQR/ACI consume. SARIMA
    is skipped from band-mode (no pinball-loss path).
    """
    bad = [m for m in models if m not in SUPPORTED_MODELS]
    if bad:
        return {"seed": seed,
                "error": f"unknown models {bad}; supported={SUPPORTED_MODELS}"}
    if calibration not in CALIBRATION_MODES:
        return {"seed": seed,
                "error": f"unknown calibration {calibration!r}; "
                         f"choose from {CALIBRATION_MODES}"}

    _override_config(seed, dataset, loss, alpha, beta, tau)
    set_all_seeds(seed)

    import src.config as config_module
    cfg = config_module.CONFIG
    print(f"\n[seed {seed}] dataset={cfg['dataset']} loss={cfg['lstm_loss']} "
          f"alpha={cfg.get('loss_alpha')} beta={cfg.get('loss_beta')}  "
          f"models={list(models)}")

    seed_dir = os.path.join(base_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    save_json(
        {
            "seed": seed,
            "dataset": dataset,
            "loss": loss,
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
            "models": list(models),
            "git": get_git_info(),
            "config": {k: str(v) for k, v in cfg.items()},
            "timestamp": datetime.now().isoformat(),
        },
        os.path.join(seed_dir, "run_metadata.json"),
    )

    # 1. simulate_data only for the synthetic path.
    if dataset == "synthetic":
        try:
            from src import simulate_data
            importlib.reload(simulate_data)
            simulate_data.main()
        except Exception as e:
            return {"seed": seed, "error": f"simulate_data: {e}"}
    else:
        if not os.path.exists(dataset_path()):
            return {"seed": seed,
                    "error": f"missing dataset npz at {dataset_path()} — "
                             "run the loader first"}

    # 2. Train each requested model in turn.
    for model in models:
        try:
            _run_trainer(model)
        except Exception as e:
            return {"seed": seed, "error": f"{model}: {e}"}

    # 3. Operational metrics, one entry per model.
    try:
        margin = cfg["capacity_margin"]
        window = cfg["window_size"]
        loaded = {m: _load_predictions_for_model(m, window) for m in models}
        aligned = _align_models(loaded)
        y_true = aligned["L_test_aligned"]
        per_model = {
            _model_display_name(m): _per_model_metrics(
                y_true, aligned["models"][m],
                alpha=alpha, beta=beta, margin=margin,
            )
            for m in models
        }
        result = {
            "seed": seed,
            "dataset": dataset,
            "loss": loss,
            "T_eff": aligned["T_eff"],
            "num_links": aligned["num_links"],
            "models": per_model,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"seed": seed, "error": f"operational metrics: {e}"}

    # 4. Calibration pass (optional). Trains each neural model twice at
    #    tau_lo and tau_hi, then runs CQR / ACI on the resulting bands.
    if calibration != "none":
        neural = [m for m in models if m in NEURAL_MODELS]
        if not neural:
            print("[calibration] no neural models requested; skipping calibration")
        else:
            try:
                for m in neural:
                    _train_pinball_band(m, tau_lo=tau_lo, tau_hi=tau_hi)
                    bands = _load_band_predictions(m)
                    display = _model_display_name(m)
                    if calibration in ("cqr", "both"):
                        cqr_block = _apply_cqr(
                            bands, target_alpha=target_alpha,
                            op_alpha=alpha, op_beta=beta,
                        )
                        per_model[f"{display}_CQR"] = {
                            "forecast": {k: None for k in FORECAST_KEYS},
                            "operational": cqr_block["operational"],
                            "calibration": cqr_block["calibration"],
                        }
                    if calibration in ("aci", "both"):
                        aci_block = _apply_aci(
                            bands, target_alpha=target_alpha,
                            op_alpha=alpha, op_beta=beta,
                            gamma=aci_gamma, window=aci_window,
                        )
                        per_model[f"{display}_ACI"] = {
                            "forecast": {k: None for k in FORECAST_KEYS},
                            "operational": aci_block["operational"],
                            "calibration": aci_block["calibration"],
                        }
                result["models"] = per_model
                result["calibration"] = calibration
                result["target_alpha"] = float(target_alpha)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"seed": seed, "error": f"calibration: {e}"}

    save_json(result, os.path.join(seed_dir, "results.json"))

    # 5. Copy per-seed prediction artefacts so a seed dir is self-contained.
    for m in models:
        fname = f"{m}_predictions.npz"
        src_p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(src_p):
            shutil.copy2(src_p, os.path.join(seed_dir, fname))
    if calibration != "none":
        for m in [x for x in models if x in NEURAL_MODELS]:
            for suffix in ("qlo", "qhi"):
                fname = f"{m}_{suffix}_predictions.npz"
                src_p = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(src_p):
                    shutil.copy2(src_p, os.path.join(seed_dir, fname))

    return result


def aggregate(results: list) -> dict:
    """Aggregate per-seed results: mean / std / values per (model, metric)."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "all seeds failed", "per_seed": results}

    out = {
        "num_seeds": len(valid),
        "seeds": [r["seed"] for r in valid],
        "dataset": valid[0]["dataset"],
        "loss": valid[0]["loss"],
        "per_seed": valid,
        "models": {},
    }

    # Union of model names across seeds (calibration models may only appear
    # when the user passed --calibration).
    all_models: list[str] = []
    for r in valid:
        for m in r["models"].keys():
            if m not in all_models:
                all_models.append(m)

    for model in all_models:
        model_block: dict = {"forecast": {}, "operational": {}, "calibration": {}}
        groups = (
            ("forecast", FORECAST_KEYS),
            ("operational", OPERATIONAL_KEYS),
            ("calibration", CALIBRATION_KEYS),
        )
        for group, keys in groups:
            for key in keys:
                vals = []
                for r in valid:
                    block = r["models"].get(model, {}).get(group, {})
                    v = block.get(key)
                    if v is not None and isinstance(v, (int, float)):
                        vals.append(float(v))
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                model_block[group][key] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "values": [float(v) for v in arr],
                }
        # Drop the calibration sub-dict if empty so the legacy schema (no
        # calibration) stays clean.
        if not model_block["calibration"]:
            model_block.pop("calibration")
        out["models"][model] = model_block
    return out


def print_summary(agg: dict) -> None:
    if "error" in agg:
        print(f"\nERROR: {agg['error']}")
        return
    print("\n" + "=" * 78)
    print(f"  dataset={agg['dataset']}  loss={agg['loss']}  "
          f"seeds={agg['seeds']}  n={agg['num_seeds']}")
    print("=" * 78)
    for model, block in agg["models"].items():
        print(f"\n  --- {model} ---")
        for group_name in ("forecast", "operational", "calibration"):
            if group_name not in block:
                continue
            for key, stats in block[group_name].items():
                print(f"    {key:<26} {stats['mean']:>14.4f} ± {stats['std']:.4f}")


def main_programmatic(*, dataset: str, loss: str,
                      alpha: float, beta: float, tau: Optional[float],
                      seeds: list,
                      models: tuple = DEFAULT_MODELS,
                      output_dir: Optional[str] = None,
                      calibration: str = "none",
                      target_alpha: float = 0.1,
                      tau_lo: float = 0.05,
                      tau_hi: float = 0.95,
                      aci_gamma: float = 0.005,
                      aci_window: int = 500) -> dict:
    """Same as `main()` but takes args programmatically. Used by run_v0.py."""
    base_dir = output_dir or os.path.join(RESULTS_DIR, f"{dataset}_{loss}")
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 78)
    print(f"  [programmatic] dataset={dataset}  loss={loss}  "
          f"alpha={alpha}  beta={beta}  tau={tau}  models={list(models)}  "
          f"seeds={seeds}  calibration={calibration}")
    print("=" * 78)

    results = []
    for seed in seeds:
        r = run_single_seed(
            seed,
            dataset=dataset, loss=loss,
            alpha=alpha, beta=beta, tau=tau,
            base_dir=base_dir,
            models=models,
            calibration=calibration,
            target_alpha=target_alpha,
            tau_lo=tau_lo, tau_hi=tau_hi,
            aci_gamma=aci_gamma, aci_window=aci_window,
        )
        results.append(r)
        if "error" in r:
            print(f"  [seed {seed}] FAILED: {r['error']}")

    agg = aggregate(results)
    save_json(agg, os.path.join(base_dir, "aggregated_results.json"))
    print_summary(agg)
    return agg


def main() -> dict:
    parser = argparse.ArgumentParser(description="multi-seed experiment runner")
    parser.add_argument("--dataset", default=CONFIG.get("dataset", "abilene"),
                        choices=("synthetic", "abilene", "geant", "cesnet"))
    parser.add_argument("--loss", default=CONFIG.get("lstm_loss", "mse"),
                        choices=("mse", "asym", "pinball"))
    parser.add_argument("--alpha", type=float,
                        default=CONFIG.get("loss_alpha", 5.0))
    parser.add_argument("--beta", type=float,
                        default=CONFIG.get("loss_beta", 1.0))
    parser.add_argument("--tau", type=float, default=CONFIG.get("loss_tau"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        choices=SUPPORTED_MODELS,
                        help=f"models to train (default: {list(DEFAULT_MODELS)})")
    parser.add_argument("--output-dir", default=None,
                        help="base directory for per-seed artefacts "
                             "(default: results/<dataset>_<loss>)")
    parser.add_argument("--calibration", default="none",
                        choices=CALIBRATION_MODES,
                        help="conformal calibration mode (default: none). "
                             "When != none, each neural model is also "
                             "trained twice at tau_lo and tau_hi for "
                             "(q_lo, q_hi) bands.")
    parser.add_argument("--target-alpha", type=float, default=0.1,
                        help="target miscoverage for CQR/ACI (default: 0.1, "
                             "= 90%% coverage)")
    parser.add_argument("--tau-lo", type=float, default=0.05)
    parser.add_argument("--tau-hi", type=float, default=0.95)
    parser.add_argument("--aci-gamma", type=float, default=0.005)
    parser.add_argument("--aci-window", type=int, default=500)
    args = parser.parse_args()

    return main_programmatic(
        dataset=args.dataset, loss=args.loss,
        alpha=args.alpha, beta=args.beta, tau=args.tau,
        seeds=args.seeds,
        models=tuple(args.models),
        output_dir=args.output_dir,
        calibration=args.calibration,
        target_alpha=args.target_alpha,
        tau_lo=args.tau_lo, tau_hi=args.tau_hi,
        aci_gamma=args.aci_gamma, aci_window=args.aci_window,
    )


if __name__ == "__main__":
    main()
