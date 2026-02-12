"""Tests for `--calibration cqr|aci|both` mode in run_experiments.py.

Strategy:
  - Same fake-dataset / fake-results-dir / reload-stub plumbing as
    test_runner_keys.py.
  - Stub the neural trainer's `main` to read CONFIG['loss_tau'] at call
    time and write tau-dependent predictions, so the qlo/qhi runs
    produce a non-degenerate band.
  - Skip SARIMA from the model list — it doesn't go through the band path.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


# Reuse the fake_dataset / fake_results_dir fixtures from test_runner_keys
# without re-defining them; pytest will discover them from conftest if we
# put them there. Inline copies are simpler for now.
@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    T, num_links = 800, 4
    # Build a simple sinusoidal signal so quantile-based calibration has
    # something non-trivial to fit.
    t = np.arange(T)
    L = np.column_stack([
        5 + 4 * np.sin(2 * np.pi * t / 100 + i)
        + rng.normal(0, 0.5, size=T)
        for i in range(num_links)
    ]).astype(np.float32)
    train_end = 480
    val_end = 640

    npz_path = tmp_path / "fake_traffic.npz"
    np.savez(
        npz_path,
        TM=L, L=L, T=T, num_links=num_links, num_od=num_links,
        T_train=train_end, T_val=val_end - train_end, T_test=T - val_end,
        train_end=train_end, val_end=val_end,
    )

    import src.config as config_module
    monkeypatch.setitem(
        config_module.DATASET_FILES, "abilene", str(npz_path.name)
    )
    monkeypatch.setattr(config_module, "DATA_DIR", str(tmp_path))
    return {"npz": npz_path, "L": L, "train_end": train_end, "val_end": val_end}


@pytest.fixture
def fake_results_dir(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    import src.config as config_module
    monkeypatch.setattr(config_module, "RESULTS_DIR", str(results_dir))
    import scripts.run_experiments as runner
    importlib.reload(runner)
    return results_dir


def _make_quantile_aware_stub(fake_dataset, results_dir, model_name: str):
    """Stub that reads CONFIG['loss_tau'] at call time.

    For tau != None and lstm_loss == 'pinball':
        emits y + offset(tau) where offset(0.5)=0 and offset
        scales linearly: tau=0.05 → -1.5, tau=0.95 → +1.5. This
        creates a *too-narrow* band that CQR has to widen.
    Otherwise (point forecast):
        emits y + small Gaussian noise.
    """
    L = fake_dataset["L"]
    train_end = fake_dataset["train_end"]
    val_end = fake_dataset["val_end"]
    import src.config as config_module
    window = config_module.CONFIG["window_size"]
    L_val_aligned = L[train_end + window: val_end].astype(np.float32)
    L_test_aligned = L[val_end + window:].astype(np.float32)

    def _main():
        cfg = config_module.CONFIG
        loss = cfg.get("lstm_loss")
        tau = cfg.get("loss_tau")

        if loss == "pinball" and tau is not None:
            offset = (tau - 0.5) * 3.0  # tau=0.05 → -1.35; tau=0.95 → +1.35
        else:
            offset = 0.0

        rng = np.random.default_rng(int(1000 * (tau or 0.5)))
        val_noise = rng.normal(0, 0.05, size=L_val_aligned.shape).astype(np.float32)
        test_noise = rng.normal(0, 0.05, size=L_test_aligned.shape).astype(np.float32)
        val_preds = (L_val_aligned + offset + val_noise).astype(np.float32)
        test_preds = (L_test_aligned + offset + test_noise).astype(np.float32)

        np.savez(
            os.path.join(results_dir, f"{model_name}_predictions.npz"),
            predictions=test_preds,
            L_test_aligned=L_test_aligned,
            val_predictions=val_preds,
            L_val_aligned=L_val_aligned,
        )
    return _main


def _patch_runner(monkeypatch, fake_dataset, fake_results_dir):
    import scripts.run_experiments as runner
    monkeypatch.setattr(runner.importlib, "reload", lambda mod: mod)

    import src.train_lstm as train_lstm
    monkeypatch.setattr(
        train_lstm, "main",
        _make_quantile_aware_stub(fake_dataset, str(fake_results_dir), "lstm"),
    )
    return runner


def test_calibration_cqr_emits_calibration_block(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)
    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("lstm",),
        calibration="cqr",
        target_alpha=0.1,
        tau_lo=0.05, tau_hi=0.95,
    )

    assert "error" not in result, f"runner failed: {result.get('error')}"
    assert "LSTM" in result["models"]
    assert "LSTM_CQR" in result["models"]

    cqr_block = result["models"]["LSTM_CQR"]
    assert cqr_block["calibration"]["method"] == "cqr"
    assert cqr_block["calibration"]["target_alpha"] == 0.1
    assert "qhat_mean" in cqr_block["calibration"]
    assert "coverage_overall" in cqr_block["calibration"]
    # Operational metrics must still be present.
    for key in runner.OPERATIONAL_KEYS:
        assert key in cqr_block["operational"]

    # Per-seed band-prediction artefacts should be in the seed dir.
    seed_dir = base_dir / "seed_42"
    assert (seed_dir / "lstm_qlo_predictions.npz").exists()
    assert (seed_dir / "lstm_qhi_predictions.npz").exists()


def test_calibration_aci_emits_aci_block(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)
    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("lstm",),
        calibration="aci",
        target_alpha=0.1,
        tau_lo=0.05, tau_hi=0.95,
        aci_gamma=0.01, aci_window=50,
    )

    assert "error" not in result, f"runner failed: {result.get('error')}"
    assert "LSTM_ACI" in result["models"]
    aci_block = result["models"]["LSTM_ACI"]
    assert aci_block["calibration"]["method"] == "aci"
    assert aci_block["calibration"]["gamma"] == 0.01
    assert aci_block["calibration"]["window"] == 50
    assert 0.0 < aci_block["calibration"]["coverage_overall"] <= 1.0


def test_calibration_both_emits_cqr_and_aci(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)
    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("lstm",),
        calibration="both",
        target_alpha=0.1,
    )

    assert "error" not in result
    assert "LSTM_CQR" in result["models"]
    assert "LSTM_ACI" in result["models"]


def test_calibration_skips_when_no_neural_models(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """SARIMA-only run with --calibration cqr should not crash; just no-ops."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    # Stub SARIMA too, since we'll request it.
    L = fake_dataset["L"]
    val_end = fake_dataset["val_end"]
    sarima_preds = np.broadcast_to(
        L[:val_end].mean(axis=0), L[val_end:].shape
    ).astype(np.float32).copy()

    def _sarima_main():
        np.savez(
            os.path.join(fake_results_dir, "sarima_predictions.npz"),
            predictions=sarima_preds,
            L_test=L[val_end:].astype(np.float32),
            fit_mask=np.ones(L.shape[1], dtype=bool),
        )

    import src.train_arima as train_arima
    monkeypatch.setattr(train_arima, "main", _sarima_main)

    base_dir = tmp_path / "out"
    base_dir.mkdir()
    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("sarima",),
        calibration="cqr",
    )
    assert "error" not in result
    assert "SARIMA" in result["models"]
    # No CQR block because SARIMA is excluded from band-mode.
    assert not any("CQR" in k for k in result["models"].keys())


def test_calibration_default_none_unchanged_schema(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """Without --calibration, the result schema must match the legacy shape."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    L = fake_dataset["L"]
    val_end = fake_dataset["val_end"]
    sarima_preds = np.broadcast_to(
        L[:val_end].mean(axis=0), L[val_end:].shape
    ).astype(np.float32).copy()

    def _sarima_main():
        np.savez(
            os.path.join(fake_results_dir, "sarima_predictions.npz"),
            predictions=sarima_preds,
            L_test=L[val_end:].astype(np.float32),
            fit_mask=np.ones(L.shape[1], dtype=bool),
        )
    import src.train_arima as train_arima
    monkeypatch.setattr(train_arima, "main", _sarima_main)

    base_dir = tmp_path / "out"
    base_dir.mkdir()
    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("sarima", "lstm"),
        calibration="none",
    )
    assert "error" not in result
    assert set(result["models"].keys()) == {"SARIMA", "LSTM"}
    # No calibration sub-dict since mode was 'none'.
    for m in ("SARIMA", "LSTM"):
        assert "calibration" not in result["models"][m]


def test_unknown_calibration_mode_rejected(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)
    base_dir = tmp_path / "out"
    base_dir.mkdir()
    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("lstm",),
        calibration="bogus",
    )
    assert "error" in result
    assert "calibration" in result["error"]
