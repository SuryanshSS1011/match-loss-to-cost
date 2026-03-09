"""Unit tests for src/calibration/aci.py.

Pure numpy. Verify the Gibbs-Candès update converges to the target
coverage on a stationary stream and recovers from a drift shock.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import ACITracker, aci_step, sliding_quantile


# ---------------------------------------------------------------------------
# aci_step
# ---------------------------------------------------------------------------

class TestAciStep:
    def test_in_band_increases_alpha(self):
        # Coverage success → alpha grows toward (and past) target.
        nxt = aci_step(in_band=True, alpha_t=0.1, target_alpha=0.1, gamma=0.005)
        assert nxt > 0.1

    def test_miss_decreases_alpha(self):
        nxt = aci_step(in_band=False, alpha_t=0.1, target_alpha=0.1, gamma=0.005)
        assert nxt < 0.1

    def test_alpha_clipped_to_open_interval(self):
        # Alpha can't reach 0 even after many in-band steps with target near 0.
        a = 0.5
        for _ in range(10000):
            a = aci_step(in_band=True, alpha_t=a, target_alpha=0.001, gamma=0.05)
        assert 0.0 < a < 1.0

    def test_invalid_target_alpha(self):
        with pytest.raises(ValueError):
            aci_step(in_band=True, alpha_t=0.1, target_alpha=1.5, gamma=0.005)

    def test_nonpositive_gamma(self):
        with pytest.raises(ValueError):
            aci_step(in_band=True, alpha_t=0.1, target_alpha=0.1, gamma=0.0)


# ---------------------------------------------------------------------------
# sliding_quantile
# ---------------------------------------------------------------------------

class TestSlidingQuantile:
    def test_empty_returns_inf(self):
        assert sliding_quantile([], alpha=0.1) == float("inf")

    def test_matches_finite_sample_formula(self):
        scores = list(range(10))  # 0..9
        # alpha=0.1, n=10 → k = ⌈11 * 0.9⌉ = 10 → max = 9.
        assert sliding_quantile(scores, alpha=0.1) == 9.0

    def test_caps_at_max(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        # alpha=0.05, n=5 → k = ⌈6 * 0.95⌉ = 6 > n → return max.
        assert sliding_quantile(scores, alpha=0.05) == 5.0


# ---------------------------------------------------------------------------
# ACITracker — long-run coverage
# ---------------------------------------------------------------------------

class TestACITracker:
    def test_init_state(self):
        t = ACITracker(num_links=4, target_alpha=0.1, gamma=0.005, window=200)
        assert t.alpha.shape == (4,)
        assert np.allclose(t.alpha, 0.1)

    def test_long_run_alpha_converges_to_target(self):
        # Stationary "in-band" stream with mean-rate (1 - target). The
        # update has fixed point at mean(alpha_t) ≈ target.
        rng = np.random.default_rng(0)
        target = 0.1
        gamma = 0.05  # moderate step size for fast convergence in unit test
        t = ACITracker(num_links=1, target_alpha=target, gamma=gamma, window=500)
        n_steps = 5000
        for _ in range(n_steps):
            in_band = rng.random() > target  # marginal coverage = 1-target
            t.update(
                scores=np.array([0.5]),  # constant score, just to populate
                in_band=np.array([in_band]),
            )
        # Last 1000 alphas should sit near target.
        # We can't read history without retaining it; check the final alpha
        # is in a reasonable band around target.
        assert abs(float(t.alpha[0]) - target) < 0.1

    def test_drift_shock_recovers(self):
        # Run with target=0.1 + always covered → alpha drifts up toward 1.
        # Then flip to never-covered → alpha decays toward 0.
        target = 0.1
        gamma = 0.05
        t = ACITracker(num_links=1, target_alpha=target, gamma=gamma, window=200)

        # Phase 1: 500 always-covered steps. alpha should grow above target.
        for _ in range(500):
            t.update(np.array([0.0]), np.array([True]))
        assert t.alpha[0] > target + 0.05

        # Phase 2: 500 always-missed steps. alpha should drop below target.
        for _ in range(500):
            t.update(np.array([0.0]), np.array([False]))
        assert t.alpha[0] < target

    def test_per_link_independence(self):
        # Two links: link 0 always covered, link 1 always missed. Their
        # alpha trajectories must diverge.
        t = ACITracker(num_links=2, target_alpha=0.1, gamma=0.05, window=100)
        for _ in range(200):
            t.update(
                scores=np.array([0.0, 0.0]),
                in_band=np.array([True, False]),
            )
        assert t.alpha[0] > 0.1
        assert t.alpha[1] < 0.1

    def test_qhat_uses_per_link_alpha(self):
        # Pre-load both links with the same scores; bias their alphas
        # differently and verify qhat differs accordingly.
        t = ACITracker(num_links=2, target_alpha=0.1, gamma=0.001, window=200)
        scores = np.linspace(0, 9, num=10)
        for s in scores:
            t.update(np.array([s, s]), np.array([True, True]))
        # Manually bias alphas: link 1 wants tighter coverage (smaller alpha).
        t.alpha[0] = 0.2
        t.alpha[1] = 0.05
        qh = t.qhat()
        # Smaller alpha → higher quantile → larger qhat.
        assert qh[1] >= qh[0]

    def test_update_shape_validation(self):
        t = ACITracker(num_links=3, target_alpha=0.1, gamma=0.005)
        with pytest.raises(ValueError):
            t.update(np.array([0.0]), np.array([True, False, True]))
        with pytest.raises(ValueError):
            t.update(np.array([0.0, 0.0, 0.0]), np.array([True]))

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            ACITracker(num_links=0)
        with pytest.raises(ValueError):
            ACITracker(num_links=1, window=0)
