"""Tests for the seasonal-naive / Holt-Winters / historical-average trainers.

These trainers run end-to-end on tiny canned npz files and write their
own `<name>_predictions.npz`. We assert the SARIMA-like schema
(`predictions`, `L_test`) and shape contracts.

Holt-Winters fitting with statsmodels can be slow on long series; we keep
the test data short (T=900, 4 links, seasonal_period=24) so the whole
file runs in <2 s.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """Tiny seasonal dataset with enough history for s=24 baselines."""
    rng = np.random.default_rng(0)
    T, num_links = 900, 4
    t = np.arange(T)
    # Sin signal with period 24 + noise — enough for seasonal-naive and
    # Holt-Winters to do something non-trivial.
    L = np.column_stack([
        5 + 3 * np.sin(2 * np.pi * t / 24 + i)
        + rng.normal(0, 0.3, size=T)
        for i in range(num_links)
    ]).astype(np.float32)
    train_end = 540    # 60%
    val_end = 720      # +20%

    npz_path = tmp_path / "fake_traffic.npz"
    np.savez(
        npz_path,
        TM=L, L=L, T=T, num_links=num_links, num_od=num_links,
        T_train=train_end, T_val=val_end - train_end, T_test=T - val_end,
        train_end=train_end, val_end=val_end,
    )

    import src.config as config_module
    monkeypatch.setitem(config_module.DATASET_FILES, "abilene", npz_path.name)
    monkeypatch.setattr(config_module, "DATA_DIR", str(tmp_path))
    monkeypatch.setitem(config_module.CONFIG, "dataset", "abilene")
    monkeypatch.setitem(config_module.CONFIG, "seasonal_period", 24)
    return {"L": L, "train_end": train_end, "val_end": val_end, "T": T,
            "num_links": num_links}


@pytest.fixture
def fake_results_dir(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    import src.config as config_module
    monkeypatch.setattr(config_module, "RESULTS_DIR", str(results_dir))
    return results_dir


def _reload(module_path: str):
    mod = importlib.import_module(module_path)
    return importlib.reload(mod)


# ---------------------------------------------------------------------------
# seasonal_naive
# ---------------------------------------------------------------------------

class TestSeasonalNaiveTrainer:
    def test_writes_sarima_like_schema(
        self, fake_dataset, fake_results_dir, monkeypatch
    ):
        tc = _reload("src.train_seasonal_naive")
        tc.main()
        npz_path = os.path.join(str(fake_results_dir),
                                "seasonal_naive_predictions.npz")
        assert os.path.exists(npz_path)
        npz = np.load(npz_path)
        assert "predictions" in npz.files
        assert "L_test" in npz.files

        T_test = fake_dataset["T"] - fake_dataset["val_end"]
        assert npz["predictions"].shape == (T_test, fake_dataset["num_links"])
        assert npz["L_test"].shape == (T_test, fake_dataset["num_links"])

    def test_predictions_match_lagged_values(
        self, fake_dataset, fake_results_dir
    ):
        tc = _reload("src.train_seasonal_naive")
        tc.main()
        npz = np.load(os.path.join(str(fake_results_dir),
                                   "seasonal_naive_predictions.npz"))
        # With seasonal_period=24, prediction at test_start + t should
        # equal L[test_start + t - 24].
        L = fake_dataset["L"]
        val_end = fake_dataset["val_end"]
        s = 24
        for t in (0, 5, 50, 100):
            for ell in range(fake_dataset["num_links"]):
                expected = L[val_end + t - s, ell]
                assert npz["predictions"][t, ell] == pytest.approx(
                    expected, abs=1e-5
                )

    def test_metrics_json_written(self, fake_dataset, fake_results_dir):
        tc = _reload("src.train_seasonal_naive")
        tc.main()
        path = os.path.join(str(fake_results_dir),
                            "seasonal_naive_metrics.json")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# histavg
# ---------------------------------------------------------------------------

class TestHistAvgTrainer:
    def test_writes_sarima_like_schema(self, fake_dataset, fake_results_dir):
        tc = _reload("src.train_histavg")
        tc.main()
        npz_path = os.path.join(str(fake_results_dir),
                                "histavg_predictions.npz")
        npz = np.load(npz_path)
        assert "predictions" in npz.files
        assert "L_test" in npz.files
        T_test = fake_dataset["T"] - fake_dataset["val_end"]
        assert npz["predictions"].shape == (T_test, fake_dataset["num_links"])

    def test_predictions_are_train_mean(self, fake_dataset, fake_results_dir):
        tc = _reload("src.train_histavg")
        tc.main()
        npz = np.load(os.path.join(str(fake_results_dir),
                                   "histavg_predictions.npz"))
        train_mean = fake_dataset["L"][:fake_dataset["train_end"]].mean(axis=0)
        # Every test row must equal the per-link train mean exactly.
        assert np.allclose(npz["predictions"], train_mean[None, :], atol=1e-5)

    def test_no_data_leakage(self, fake_dataset, fake_results_dir):
        # The mean must come from train only; not val + train.
        tc = _reload("src.train_histavg")
        tc.main()
        npz = np.load(os.path.join(str(fake_results_dir),
                                   "histavg_predictions.npz"))
        bad_mean = fake_dataset["L"][:fake_dataset["val_end"]].mean(axis=0)
        # If the implementation accidentally used val_end, the test mean
        # would equal `bad_mean`, not `train_mean`.
        good_mean = fake_dataset["L"][:fake_dataset["train_end"]].mean(axis=0)
        assert np.allclose(npz["predictions"][0], good_mean, atol=1e-5)
        # And not equal to the bad mean (confirming the test is meaningful).
        assert not np.allclose(npz["predictions"][0], bad_mean, atol=1e-3)


# ---------------------------------------------------------------------------
# holtwinters
# ---------------------------------------------------------------------------

class TestHoltWintersTrainer:
    def test_writes_sarima_like_schema(
        self, fake_dataset, fake_results_dir, monkeypatch
    ):
        # Cap the test horizon hard so this test stays cheap.
        import src.config as config_module
        monkeypatch.setitem(config_module.CONFIG, "holtwinters_test_steps", 24)
        monkeypatch.setitem(
            config_module.CONFIG, "holtwinters_train_window", 200
        )
        tc = _reload("src.train_holtwinters")
        tc.main()

        npz_path = os.path.join(str(fake_results_dir),
                                "holtwinters_predictions.npz")
        npz = np.load(npz_path)
        assert "predictions" in npz.files
        assert "L_test" in npz.files
        # Capped to 24.
        assert npz["predictions"].shape == (24, fake_dataset["num_links"])
        assert npz["L_test"].shape == (24, fake_dataset["num_links"])

    def test_metrics_json_written(self, fake_dataset, fake_results_dir,
                                   monkeypatch):
        import src.config as config_module
        monkeypatch.setitem(config_module.CONFIG, "holtwinters_test_steps", 24)
        monkeypatch.setitem(
            config_module.CONFIG, "holtwinters_train_window", 200
        )
        tc = _reload("src.train_holtwinters")
        tc.main()
        path = os.path.join(str(fake_results_dir),
                            "holtwinters_metrics.json")
        assert os.path.exists(path)
