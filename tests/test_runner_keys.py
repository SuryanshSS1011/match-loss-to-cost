"""Schema lock-in for scripts/run_experiments.py.

Stubs out `train_arima.main` and `train_lstm.main` so they drop a tiny
canned prediction npz into RESULTS_DIR (no real training, no GPU). Then runs
`run_single_seed` end-to-end and asserts the headline keys are present in
the per-seed and aggregated JSON.

The point: catch silent schema drift before the cloud sweep. If the runner
ever stops emitting `asymmetric_op_cost` etc., we want a red test, not a
quiet hole in the results table.
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """Create a tiny abilene-shaped npz and point `dataset_path` at it."""
    rng = np.random.default_rng(0)
    T, num_links = 600, 4
    L = rng.uniform(1.0, 5.0, size=(T, num_links)).astype(np.float32)
    # Tiny routing matrix so graph-aware builders (e.g. DCRNN) get a real R.
    # Two OD pairs sharing each link → cross-link adjacency is non-trivial.
    R = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=np.float32)
    train_end = 360
    val_end = 480

    npz_path = tmp_path / "fake_traffic.npz"
    np.savez(
        npz_path,
        TM=L,  # not used downstream; placeholder to satisfy any reader
        L=L,
        R=R,
        T=T,
        num_links=num_links,
        num_od=num_links,
        T_train=train_end,
        T_val=val_end - train_end,
        T_test=T - val_end,
        train_end=train_end,
        val_end=val_end,
    )

    import src.config as config_module
    monkeypatch.setitem(
        config_module.DATASET_FILES, "abilene", str(npz_path.name)
    )
    monkeypatch.setattr(config_module, "DATA_DIR", str(tmp_path))
    return {"npz": npz_path, "L": L, "R": R,
            "train_end": train_end, "val_end": val_end}


@pytest.fixture
def fake_results_dir(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR to a scratch path so the test doesn't pollute results/."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    import src.config as config_module
    monkeypatch.setattr(config_module, "RESULTS_DIR", str(results_dir))

    # The runner module captured RESULTS_DIR at import time; reload it so it
    # picks up the patched value.
    import scripts.run_experiments as runner
    importlib.reload(runner)
    return results_dir


def _stub_train_arima(fake_dataset, results_dir):
    """Return a no-op replacement for `train_arima.main` that drops a fake npz."""
    L = fake_dataset["L"]
    val_end = fake_dataset["val_end"]
    L_test = L[val_end:]

    # SARIMA "predicts" the per-link mean of the train window, broadcast across
    # the test length. Simple, deterministic, finite.
    train_mean = L[:val_end].mean(axis=0)
    sarima_preds = np.broadcast_to(train_mean, L_test.shape).astype(np.float32).copy()

    def _main():
        np.savez(
            os.path.join(results_dir, "sarima_predictions.npz"),
            predictions=sarima_preds,
            L_test=L_test.astype(np.float32),
            fit_mask=np.ones(L.shape[1], dtype=bool),
        )
    return _main


def _stub_neural_main(fake_dataset, results_dir, model_name: str, seed: int):
    """Replacement for a neural-trainer `main` (lstm, dlinear, ...).

    Drops `<model_name>_predictions.npz` containing both val and test
    predictions aligned to the dataset's val/test windows. Different
    `seed` values give distinguishable predictions so a multi-model run
    yields different metrics per model.
    """
    L = fake_dataset["L"]
    train_end = fake_dataset["train_end"]
    val_end = fake_dataset["val_end"]

    import src.config as config_module
    window = config_module.CONFIG["window_size"]
    L_val_aligned = L[train_end + window: val_end]
    L_test_aligned = L[val_end + window:]

    rng = np.random.default_rng(seed)
    val_noise = rng.normal(0.0, 0.1, size=L_val_aligned.shape).astype(np.float32)
    test_noise = rng.normal(0.0, 0.1, size=L_test_aligned.shape).astype(np.float32)
    val_preds = (L_val_aligned + val_noise).astype(np.float32)
    test_preds = (L_test_aligned + test_noise).astype(np.float32)

    def _main():
        np.savez(
            os.path.join(results_dir, f"{model_name}_predictions.npz"),
            predictions=test_preds,
            L_test_aligned=L_test_aligned.astype(np.float32),
            val_predictions=val_preds,
            L_val_aligned=L_val_aligned.astype(np.float32),
        )
    return _main


def _stub_train_lstm(fake_dataset, results_dir):
    return _stub_neural_main(fake_dataset, results_dir, "lstm", seed=1)


def _stub_train_dlinear(fake_dataset, results_dir):
    return _stub_neural_main(fake_dataset, results_dir, "dlinear", seed=2)


def _stub_train_patchtst(fake_dataset, results_dir):
    return _stub_neural_main(fake_dataset, results_dir, "patchtst", seed=3)


def _stub_train_itransformer(fake_dataset, results_dir):
    return _stub_neural_main(fake_dataset, results_dir, "itransformer", seed=4)


def _stub_train_chronos(fake_dataset, results_dir):
    # Chronos writes the same schema; reuse the neural stub helper.
    return _stub_neural_main(fake_dataset, results_dir, "chronos", seed=5)


def _stub_train_dcrnn(fake_dataset, results_dir):
    return _stub_neural_main(fake_dataset, results_dir, "dcrnn", seed=6)


def _stub_sarima_like_main(fake_dataset, results_dir, model_name: str,
                            seed: int):
    """Stub for SARIMA-like trainers (seasonal_naive, holtwinters, histavg).

    Writes <model_name>_predictions.npz with the SARIMA schema:
        predictions (T_test, num_links), L_test (T_test, num_links).
    """
    L = fake_dataset["L"]
    val_end = fake_dataset["val_end"]
    L_test = L[val_end:].astype(np.float32)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.1, size=L_test.shape).astype(np.float32)
    preds = (L_test + noise).astype(np.float32)

    def _main():
        np.savez(
            os.path.join(results_dir, f"{model_name}_predictions.npz"),
            predictions=preds,
            L_test=L_test,
        )
    return _main


def _stub_train_seasonal_naive(fake_dataset, results_dir):
    return _stub_sarima_like_main(fake_dataset, results_dir,
                                    "seasonal_naive", seed=7)


def _stub_train_holtwinters(fake_dataset, results_dir):
    return _stub_sarima_like_main(fake_dataset, results_dir,
                                    "holtwinters", seed=8)


def _stub_train_histavg(fake_dataset, results_dir):
    return _stub_sarima_like_main(fake_dataset, results_dir,
                                    "histavg", seed=9)


def _patch_runner(monkeypatch, fake_dataset, fake_results_dir):
    """Common runner-mock setup: neutralise reload, stub each trainer's main."""
    import scripts.run_experiments as runner

    # The runner does `importlib.reload(train_lstm)` on every seed, which would
    # wipe our monkeypatch by re-binding `<module>.main` to the real function.
    # Stub `reload` to a no-op so our patched `main`s stick.
    monkeypatch.setattr(runner.importlib, "reload", lambda mod: mod)

    import src.train_arima as train_arima
    import src.train_lstm as train_lstm
    import src.train_dlinear as train_dlinear
    import src.train_patchtst as train_patchtst
    import src.train_itransformer as train_itransformer
    import src.train_chronos as train_chronos
    import src.train_dcrnn as train_dcrnn
    import src.train_seasonal_naive as train_seasonal_naive
    import src.train_holtwinters as train_holtwinters
    import src.train_histavg as train_histavg
    monkeypatch.setattr(
        train_arima, "main", _stub_train_arima(fake_dataset, str(fake_results_dir))
    )
    monkeypatch.setattr(
        train_lstm, "main", _stub_train_lstm(fake_dataset, str(fake_results_dir))
    )
    monkeypatch.setattr(
        train_dlinear, "main",
        _stub_train_dlinear(fake_dataset, str(fake_results_dir)),
    )
    monkeypatch.setattr(
        train_patchtst, "main",
        _stub_train_patchtst(fake_dataset, str(fake_results_dir)),
    )
    monkeypatch.setattr(
        train_itransformer, "main",
        _stub_train_itransformer(fake_dataset, str(fake_results_dir)),
    )
    monkeypatch.setattr(
        train_chronos, "main",
        _stub_train_chronos(fake_dataset, str(fake_results_dir)),
    )
    monkeypatch.setattr(
        train_dcrnn, "main",
        _stub_train_dcrnn(fake_dataset, str(fake_results_dir)),
    )
    monkeypatch.setattr(
        train_seasonal_naive, "main",
        _stub_train_seasonal_naive(fake_dataset, str(fake_results_dir)),
    )
    monkeypatch.setattr(
        train_holtwinters, "main",
        _stub_train_holtwinters(fake_dataset, str(fake_results_dir)),
    )
    monkeypatch.setattr(
        train_histavg, "main",
        _stub_train_histavg(fake_dataset, str(fake_results_dir)),
    )
    return runner


def test_run_single_seed_default_models_emit_headline_keys(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """Default model list (sarima + lstm) must produce both with full keys."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="asym",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
    )

    assert "error" not in result, f"runner failed: {result.get('error')}"
    assert result["seed"] == 42
    assert result["dataset"] == "abilene"
    assert result["loss"] == "asym"

    for model in ("LSTM", "SARIMA"):
        assert model in result["models"]
        op = result["models"][model]["operational"]
        for key in runner.OPERATIONAL_KEYS:
            assert key in op, f"missing operational key {key!r} on {model}"
            assert isinstance(op[key], (int, float))
        fc = result["models"][model]["forecast"]
        for key in runner.FORECAST_KEYS:
            assert key in fc, f"missing forecast key {key!r} on {model}"

    seed_dir = base_dir / "seed_42"
    assert (seed_dir / "results.json").exists()
    assert (seed_dir / "run_metadata.json").exists()


def test_run_single_seed_with_dlinear_three_models(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """Passing models=(sarima, lstm, dlinear) must yield all three entries."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="asym",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("sarima", "lstm", "dlinear"),
    )

    assert "error" not in result, f"runner failed: {result.get('error')}"
    assert set(result["models"].keys()) == {"SARIMA", "LSTM", "DLinear"}
    for model in ("SARIMA", "LSTM", "DLinear"):
        op = result["models"][model]["operational"]
        for key in runner.OPERATIONAL_KEYS:
            assert key in op
            assert isinstance(op[key], (int, float))

    # Per-model prediction npz must have been copied into the seed dir.
    seed_dir = base_dir / "seed_42"
    assert (seed_dir / "sarima_predictions.npz").exists()
    assert (seed_dir / "lstm_predictions.npz").exists()
    assert (seed_dir / "dlinear_predictions.npz").exists()


def test_run_single_seed_with_patchtst(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """PatchTST must integrate the same way as LSTM/DLinear."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("sarima", "lstm", "patchtst"),
    )

    assert "error" not in result, f"runner failed: {result.get('error')}"
    assert set(result["models"].keys()) == {"SARIMA", "LSTM", "PatchTST"}
    op = result["models"]["PatchTST"]["operational"]
    for key in runner.OPERATIONAL_KEYS:
        assert key in op
        assert isinstance(op[key], (int, float))

    seed_dir = base_dir / "seed_42"
    assert (seed_dir / "patchtst_predictions.npz").exists()


def test_run_single_seed_with_itransformer(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """iTransformer must integrate the same way as the other neural models."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("sarima", "patchtst", "itransformer"),
    )

    assert "error" not in result, f"runner failed: {result.get('error')}"
    assert set(result["models"].keys()) == {"SARIMA", "PatchTST", "iTransformer"}
    op = result["models"]["iTransformer"]["operational"]
    for key in runner.OPERATIONAL_KEYS:
        assert key in op
        assert isinstance(op[key], (int, float))

    seed_dir = base_dir / "seed_42"
    assert (seed_dir / "itransformer_predictions.npz").exists()


def test_run_single_seed_with_chronos_zero_shot(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """Chronos must integrate as a zero-shot baseline (point forecasts)."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("lstm", "chronos"),
    )

    assert "error" not in result, f"runner failed: {result.get('error')}"
    assert set(result["models"].keys()) == {"LSTM", "Chronos"}
    op = result["models"]["Chronos"]["operational"]
    for key in runner.OPERATIONAL_KEYS:
        assert key in op
        assert isinstance(op[key], (int, float))
    seed_dir = base_dir / "seed_42"
    assert (seed_dir / "chronos_predictions.npz").exists()


def test_run_single_seed_with_dcrnn(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """DCRNN must integrate the same way as the other neural models."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("sarima", "lstm", "dcrnn"),
    )

    assert "error" not in result, f"runner failed: {result.get('error')}"
    assert set(result["models"].keys()) == {"SARIMA", "LSTM", "DCRNN"}
    op = result["models"]["DCRNN"]["operational"]
    for key in runner.OPERATIONAL_KEYS:
        assert key in op
        assert isinstance(op[key], (int, float))

    seed_dir = base_dir / "seed_42"
    assert (seed_dir / "dcrnn_predictions.npz").exists()


def test_run_single_seed_with_classical_baselines(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """seasonal_naive / holtwinters / histavg integrate as SARIMA-like models."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("seasonal_naive", "holtwinters", "histavg"),
    )
    assert "error" not in result, f"runner failed: {result.get('error')}"
    assert set(result["models"].keys()) == {"SeasonalNaive", "HoltWinters", "HistAvg"}
    for model in ("SeasonalNaive", "HoltWinters", "HistAvg"):
        op = result["models"][model]["operational"]
        for key in runner.OPERATIONAL_KEYS:
            assert key in op
            assert isinstance(op[key], (int, float))

    seed_dir = base_dir / "seed_42"
    assert (seed_dir / "seasonal_naive_predictions.npz").exists()
    assert (seed_dir / "holtwinters_predictions.npz").exists()
    assert (seed_dir / "histavg_predictions.npz").exists()


def test_classical_baselines_skipped_from_calibration(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """--calibration cqr with classical baselines must NOT emit *_CQR rows."""
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("seasonal_naive", "histavg", "lstm"),
        calibration="cqr",
    )
    assert "error" not in result
    keys = set(result["models"].keys())
    assert "SeasonalNaive" in keys
    assert "HistAvg" in keys
    assert "LSTM_CQR" in keys  # LSTM is neural → gets a CQR row.
    # Classical baselines must not get calibration rows.
    assert not any(k.startswith("SeasonalNaive_") for k in keys)
    assert not any(k.startswith("HistAvg_") for k in keys)


def test_chronos_skipped_from_calibration(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    """--calibration with chronos in --models must NOT crash and must NOT
    emit a Chronos_CQR/Chronos_ACI block (chronos is zero-shot).
    """
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)

    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("lstm", "chronos"),
        calibration="cqr",
        target_alpha=0.1,
    )

    assert "error" not in result, f"runner failed: {result.get('error')}"
    keys = set(result["models"].keys())
    # Chronos point forecast row must be present.
    assert "Chronos" in keys
    # LSTM gets a CQR row because it's in NEURAL_MODELS.
    assert "LSTM_CQR" in keys
    # Chronos must NOT have a calibration row.
    assert not any(k.startswith("Chronos_") for k in keys), (
        f"unexpected chronos calibration row: {keys}"
    )


def test_run_single_seed_rejects_unknown_model(
    fake_dataset, fake_results_dir, tmp_path, monkeypatch
):
    runner = _patch_runner(monkeypatch, fake_dataset, fake_results_dir)
    base_dir = tmp_path / "out"
    base_dir.mkdir()

    result = runner.run_single_seed(
        seed=42, dataset="abilene", loss="mse",
        alpha=5.0, beta=1.0, tau=None,
        base_dir=str(base_dir),
        models=("sarima", "informer"),  # informer not yet supported
    )
    assert "error" in result
    assert "informer" in result["error"]


def test_aggregate_handles_multi_seed_and_failures(fake_results_dir):
    """`aggregate()` must compute mean/std/values and survive partial failures."""
    import scripts.run_experiments as runner

    def _per_seed(seed: int, lstm_op: float, sarima_op: float) -> dict:
        return {
            "seed": seed,
            "dataset": "abilene",
            "loss": "asym",
            "T_eff": 100,
            "num_links": 4,
            "models": {
                "LSTM": {
                    "forecast": {k: 1.0 for k in runner.FORECAST_KEYS},
                    "operational": {
                        k: (lstm_op if k == "asymmetric_op_cost" else 0.5)
                        for k in runner.OPERATIONAL_KEYS
                    },
                },
                "SARIMA": {
                    "forecast": {k: 2.0 for k in runner.FORECAST_KEYS},
                    "operational": {
                        k: (sarima_op if k == "asymmetric_op_cost" else 0.7)
                        for k in runner.OPERATIONAL_KEYS
                    },
                },
            },
        }

    results = [
        _per_seed(42, lstm_op=10.0, sarima_op=20.0),
        _per_seed(123, lstm_op=14.0, sarima_op=24.0),
        {"seed": 999, "error": "stub failure"},  # one failed seed
    ]
    agg = runner.aggregate(results)

    assert agg["num_seeds"] == 2
    assert sorted(agg["seeds"]) == [42, 123]
    assert agg["dataset"] == "abilene"

    lstm_op = agg["models"]["LSTM"]["operational"]["asymmetric_op_cost"]
    assert lstm_op["mean"] == pytest.approx(12.0)
    assert lstm_op["std"] == pytest.approx(2.0)
    assert sorted(lstm_op["values"]) == [10.0, 14.0]

    sarima_op = agg["models"]["SARIMA"]["operational"]["asymmetric_op_cost"]
    assert sarima_op["mean"] == pytest.approx(22.0)


def test_aggregate_all_failed_returns_error(fake_results_dir):
    import scripts.run_experiments as runner
    agg = runner.aggregate([{"seed": 1, "error": "x"}, {"seed": 2, "error": "y"}])
    assert "error" in agg
    assert "per_seed" in agg
