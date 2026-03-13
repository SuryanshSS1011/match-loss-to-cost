"""Spot-check `seasonal_naive_forecast` on synthetic input.

The function is dead-simple (returns L_full[t-s : t]), but we want to lock
in the contract: shape and indexing. No real data needed.
"""

from __future__ import annotations

import numpy as np

from src.baselines import naive_forecast, seasonal_naive_forecast


def test_seasonal_naive_index_alignment():
    rng = np.random.default_rng(0)
    s = 288
    T = s * 5  # five "days"
    L = rng.uniform(0, 10, size=(T, 4)).astype(np.float32)

    test_start = 4 * s
    test_end = T

    pred = seasonal_naive_forecast(L, test_start, test_end, seasonal_period=s)
    expected = L[test_start - s : test_end - s]
    assert pred.shape == (test_end - test_start, 4)
    assert np.array_equal(pred, expected)


def test_seasonal_naive_perfect_on_periodic_signal():
    # Build a perfectly daily-periodic signal; seasonal-naive should be exact.
    s = 288
    T = s * 4
    t = np.arange(T)
    pure = np.sin(2 * np.pi * t / s).astype(np.float32)
    L = np.column_stack([pure, np.cos(2 * np.pi * t / s)]).astype(np.float32)

    test_start = 2 * s
    test_end = T

    pred = seasonal_naive_forecast(L, test_start, test_end, seasonal_period=s)
    truth = L[test_start:test_end]
    assert np.allclose(pred, truth, atol=1e-6)


def test_naive_forecast_is_lag_one():
    rng = np.random.default_rng(1)
    L = rng.uniform(0, 10, size=(50, 3)).astype(np.float32)
    pred = naive_forecast(L, test_start=10, test_end=20)
    assert pred.shape == (10, 3)
    assert np.array_equal(pred, L[9:19])
