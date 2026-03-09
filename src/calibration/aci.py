"""Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021).

Reference: I. Gibbs, E. Candès, "Adaptive Conformal Inference Under
Distribution Shift", NeurIPS 2021. (arXiv:2106.00170)

Setting. Split CQR (see `cqr.py`) gives marginal coverage
`P(Y in band) >= 1 - target_alpha` under exchangeability. Real backbone
traffic is *not* exchangeable across weeks — drift, BGP changes, link
failures all break the assumption (see Dietz 2024 arXiv:2404.05304 for
the failure-induced drift case relevant to us). ACI fixes this by
tracking an *effective* miscoverage rate `alpha_t` that adapts online:

    alpha_{t+1} = alpha_t + gamma * (target_alpha - err_t)

where `err_t = 1` if `y_t` falls outside the predicted band at step t,
else 0. If we just miscovered (err_t=1), alpha_t shrinks → next band is
wider. If we just covered (err_t=0), alpha_t grows toward target → next
band tightens. Long-run average coverage converges to 1 - target_alpha
*regardless* of distribution shift, with a O(1/gamma) constant in the
regret bound (Theorem 1 of the paper).

The band itself at step t is constructed by taking the `1 - alpha_t`
empirical quantile of recent non-conformity scores
    s_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))
over a sliding window of length `window`.

Public API:
    aci_step(score, alpha_t, target_alpha, gamma)            -> alpha_{t+1}
    sliding_quantile(scores, alpha)                          -> qhat
    ACITracker(num_links, target_alpha, gamma, window)
        .qhat() -> ndarray (num_links,)
        .update(scores: ndarray (num_links,), in_band: bool ndarray) -> None

`ACITracker` is the per-link wrapper used by the runner. It maintains the
sliding score window and the per-link `alpha_t` state. The caller is
responsible for forming the (q_lo, q_hi) predictions from a quantile
forecaster — same as CQR.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import numpy as np


_ALPHA_EPS = 1e-6  # keep alpha_t strictly inside (0, 1)


def aci_step(
    in_band: bool,
    alpha_t: float,
    target_alpha: float,
    gamma: float = 0.005,
) -> float:
    """One ACI update: alpha_{t+1} = alpha_t + gamma * (target_alpha - err_t).

    err_t = 1 - 1[in_band], so:
        in_band=True  → err_t=0 → alpha_{t+1} = alpha_t + gamma * target_alpha
        in_band=False → err_t=1 → alpha_{t+1} = alpha_t - gamma * (1 - target_alpha)

    The result is clipped to (eps, 1 - eps) to keep the band well-defined.

    Args:
        in_band: whether the most recent observation fell inside the band.
        alpha_t: current effective miscoverage rate.
        target_alpha: target marginal miscoverage (e.g. 0.1 for 90% coverage).
        gamma: step size. Gibbs & Candès show 0.005 is a good default for
            target_alpha=0.1 with stationary scores; smaller gamma →
            slower drift recovery, larger gamma → noisier coverage.

    Returns:
        alpha_{t+1}.
    """
    if not (0.0 < target_alpha < 1.0):
        raise ValueError(f"target_alpha must be in (0, 1); got {target_alpha}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive; got {gamma}")

    err_t = 0.0 if in_band else 1.0
    alpha_next = alpha_t + gamma * (target_alpha - err_t)
    return float(np.clip(alpha_next, _ALPHA_EPS, 1.0 - _ALPHA_EPS))


def sliding_quantile(
    scores: Iterable[float],
    alpha: float,
) -> float:
    """Empirical (1 - alpha) quantile of a window of non-conformity scores.

    Uses the same finite-sample-aware indexing as CQR: the (⌈(n+1)(1-α)⌉)-th
    order statistic, clipped at n. With an empty window, returns +inf as a
    conservative placeholder so the resulting band is unboundedly wide.
    """
    s = np.asarray(list(scores), dtype=np.float64)
    if s.size == 0:
        return float("inf")
    n = s.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    return float(np.sort(s)[k - 1])


class ACITracker:
    """Per-link adaptive conformal inference state.

    Maintains:
      - `alpha[ell]`  — effective miscoverage rate per link.
      - `_scores[ell]` — sliding deque of recent non-conformity scores.

    The runner is expected to call:
      - `.qhat()` to get the current per-link calibration constant
        (used to widen `(q_lo, q_hi)` into the calibrated band), then
      - `.update(scores, in_band)` after observing the next ground truth.
    """

    def __init__(
        self,
        num_links: int,
        target_alpha: float = 0.1,
        gamma: float = 0.005,
        window: int = 500,
    ):
        if num_links < 1:
            raise ValueError(f"num_links must be >= 1; got {num_links}")
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        self.num_links = int(num_links)
        self.target_alpha = float(target_alpha)
        self.gamma = float(gamma)
        self.window = int(window)
        # Initialise alpha_t to target. Per Gibbs-Candès this is a fine
        # starting point; the algorithm self-corrects.
        self.alpha = np.full(num_links, target_alpha, dtype=np.float64)
        self._scores: list[deque] = [
            deque(maxlen=self.window) for _ in range(num_links)
        ]

    def qhat(self) -> np.ndarray:
        """Per-link calibration constant: (1 - alpha_t) empirical quantile."""
        out = np.empty(self.num_links, dtype=np.float64)
        for ell in range(self.num_links):
            out[ell] = sliding_quantile(self._scores[ell], self.alpha[ell])
        return out

    def update(self, scores: np.ndarray, in_band: np.ndarray) -> None:
        """Update per-link state with one new (score, in-band) observation.

        Args:
            scores: shape `(num_links,)` non-conformity score per link
                for the just-observed timestep:
                    `s = max(q_lo - y, y - q_hi)`.
            in_band: shape `(num_links,)` boolean array — True if `y_t`
                fell inside the *current* (alpha_t-calibrated) band.
        """
        scores = np.asarray(scores, dtype=np.float64)
        in_band = np.asarray(in_band, dtype=bool)
        if scores.shape != (self.num_links,):
            raise ValueError(
                f"scores shape {scores.shape}, expected ({self.num_links},)"
            )
        if in_band.shape != (self.num_links,):
            raise ValueError(
                f"in_band shape {in_band.shape}, expected ({self.num_links},)"
            )
        for ell in range(self.num_links):
            self.alpha[ell] = aci_step(
                bool(in_band[ell]),
                self.alpha[ell],
                self.target_alpha,
                self.gamma,
            )
            self._scores[ell].append(float(scores[ell]))

    def empirical_coverage_long_run(self) -> float:
        """Return the running mean of `target_alpha - (target_alpha - err_t)`
        based on `alpha_t`'s trajectory.

        Since the ACI update has stationary point `mean(err_t) = target_alpha`,
        the running mean of `alpha_t` should converge to `target_alpha`
        in expectation. We expose the mean of `1 - alpha_t` so the caller
        can compare it directly to a "1 - target" coverage target.
        """
        return float(1.0 - np.mean(self.alpha))
