"""Unit tests for src/train_chronos.py.

The chronos-forecasting package isn't a laptop dependency. Tests stub the
pipeline loader (`_load_pipeline`) so we never actually pull weights, then
call `_rolling_predictions` and `main` against canned fake outputs.

We verify:
  - Lazy import: module loads on a machine without chronos installed.
  - Missing-package error message is actionable.
  - Rolling-prediction window math: predictions[t] uses
    L_full[start_idx + t - context_length : start_idx + t, link].
  - main() writes the expected npz schema.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class FakePipeline:
    """Stand-in for `BaseChronosPipeline`. Returns the *last* context value
    as the "prediction" (a sane zero-shot baseline behaviour to test the
    plumbing without needing a real model)."""

    def __init__(self):
        self.calls = []

    def predict(self, context, prediction_length=1):
        # `context` is a list of 1-D torch tensors.
        import torch
        self.calls.append((len(context), prediction_length))
        # Return a (B, K=9, prediction_length) tensor where every quantile
        # is just the last value of the context. The trainer takes the
        # median, which is the same value.
        last_vals = torch.stack([c[-1] for c in context])  # (B,)
        # Expand to (B, 9, prediction_length).
        out = last_vals.view(-1, 1, 1).expand(-1, 9, prediction_length)
        return out.contiguous()


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace `_load_pipeline` so it returns a FakePipeline."""
    from src import train_chronos
    fake = FakePipeline()
    monkeypatch.setattr(
        train_chronos, "_load_pipeline",
        lambda *args, **kwargs: fake,
    )
    return fake


# ---------------------------------------------------------------------------
# lazy import / clear missing-dep error
# ---------------------------------------------------------------------------

class TestLazyImport:
    def test_module_imports_without_chronos(self):
        # Just importing the module shouldn't try to import chronos.
        import src.train_chronos  # noqa: F401

    def test_load_pipeline_raises_clear_error_when_chronos_missing(
        self, monkeypatch
    ):
        # Inject a fake import failure.
        import src.train_chronos as tc

        real_import = __builtins__["__import__"] if isinstance(
            __builtins__, dict
        ) else __builtins__.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "chronos":
                raise ImportError("chronos not installed (simulated)")
            return real_import(name, *args, **kwargs)

        monkeypatch.setitem(sys.modules, "chronos", None)
        # `import chronos` will now fail with ImportError because the
        # cached module is None. Verify the error message is actionable.
        with pytest.raises(ImportError, match="chronos-forecasting"):
            tc._load_pipeline("amazon/chronos-bolt-tiny")
        # Cleanup: remove the poisoned cache entry.
        sys.modules.pop("chronos", None)


# ---------------------------------------------------------------------------
# _rolling_predictions
# ---------------------------------------------------------------------------

class TestRollingPredictions:
    def test_shape_and_window_alignment(self, stub_pipeline):
        from src.train_chronos import _rolling_predictions

        rng = np.random.default_rng(0)
        T, num_links = 200, 3
        L_full = rng.uniform(1, 10, size=(T, num_links)).astype(np.float32)
        context_length = 24
        start_idx, end_idx = 50, 80

        preds = _rolling_predictions(
            stub_pipeline, L_full,
            start_idx=start_idx, end_idx=end_idx,
            context_length=context_length,
            batch_size=8,
        )

        assert preds.shape == (end_idx - start_idx, num_links)
        # FakePipeline returns the last context value, so prediction at
        # offset t == L_full[start_idx + t - 1, link].
        for t in range(end_idx - start_idx):
            for link in range(num_links):
                expected = L_full[start_idx + t - 1, link]
                assert preds[t, link] == pytest.approx(expected, abs=1e-5)

    def test_rejects_negative_history(self, stub_pipeline):
        from src.train_chronos import _rolling_predictions
        L_full = np.ones((50, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="no history"):
            _rolling_predictions(
                stub_pipeline, L_full,
                start_idx=10, end_idx=20, context_length=24,
            )

    def test_rejects_empty_window(self, stub_pipeline):
        from src.train_chronos import _rolling_predictions
        L_full = np.ones((50, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="end_idx="):
            _rolling_predictions(
                stub_pipeline, L_full,
                start_idx=20, end_idx=20, context_length=10,
            )

    def test_batch_boundary(self, stub_pipeline):
        # Make T not divisible by batch_size so we exercise the partial
        # final batch.
        from src.train_chronos import _rolling_predictions
        rng = np.random.default_rng(1)
        L_full = rng.uniform(0, 1, size=(100, 2)).astype(np.float32)
        preds = _rolling_predictions(
            stub_pipeline, L_full,
            start_idx=20, end_idx=47,    # 27 timesteps
            context_length=10,
            batch_size=10,
        )
        assert preds.shape == (27, 2)


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """Tiny abilene-shaped npz for the integration test."""
    rng = np.random.default_rng(0)
    T, num_links = 400, 3
    L = rng.uniform(1, 10, size=(T, num_links)).astype(np.float32)
    train_end = 240
    val_end = 320

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
    return {"L": L, "train_end": train_end, "val_end": val_end, "T": T}


@pytest.fixture
def fake_results_dir(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    import src.config as config_module
    monkeypatch.setattr(config_module, "RESULTS_DIR", str(results_dir))
    # train_chronos captures RESULTS_DIR at import; reload to pick up patch.
    import importlib
    import src.train_chronos as tc
    importlib.reload(tc)
    return results_dir


def test_main_writes_expected_schema(
    fake_dataset, fake_results_dir, stub_pipeline, monkeypatch
):
    import src.train_chronos as tc
    import src.config as config_module
    config_module.CONFIG["dataset"] = "abilene"
    config_module.CONFIG["chronos_context_length"] = 24
    config_module.CONFIG["chronos_batch_size"] = 16

    tc.main()

    npz_path = os.path.join(str(fake_results_dir), "chronos_predictions.npz")
    assert os.path.exists(npz_path)
    npz = np.load(npz_path)
    expected_keys = {"predictions", "L_test_aligned",
                     "val_predictions", "L_val_aligned"}
    assert expected_keys.issubset(set(npz.files))

    # Shapes: val window is (val_end - train_end) = 80, after dropping
    # context_length=24 → 56 rows. Test is (T - val_end) = 80, → 56 rows.
    assert npz["val_predictions"].shape == (56, 3)
    assert npz["predictions"].shape == (56, 3)
    assert npz["L_val_aligned"].shape == (56, 3)
    assert npz["L_test_aligned"].shape == (56, 3)

    metrics_path = os.path.join(str(fake_results_dir), "chronos_metrics.json")
    assert os.path.exists(metrics_path)
