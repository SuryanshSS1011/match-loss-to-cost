"""Conformalized Quantile Regression (CQR) for backbone capacity planning.

Reference: Y. Romano, E. Patterson, E. Candès, "Conformalized Quantile
Regression," NeurIPS 2019.

Algorithm (split CQR):
    1. A quantile forecaster (e.g. an LSTM trained with pinball loss at
       τ_lo = α/2 and τ_hi = 1 − α/2) emits predicted lower and upper
       quantile bands `q_lo(x)`, `q_hi(x)`.
    2. On a held-out **calibration set**, compute the non-conformity score
       per example:
            s_i = max(q_lo(x_i) − y_i,  y_i − q_hi(x_i)).
       Positive when y_i falls *outside* the band; negative when y_i is
       comfortably inside.
    3. Set
            qhat = quantile_{(⌈(n + 1)(1 − α)⌉) / n}(s_1, …, s_n).
       The (n+1)/n correction is the finite-sample adjustment from
       Vovk et al. 2005 / Romano 2019; without it the coverage guarantee
       only holds asymptotically.
    4. At test time, output the calibrated band
            [q_lo(x) − qhat,  q_hi(x) + qhat].
       Theorem 1 of Romano 2019: marginal coverage ≥ 1 − α under exchangeable
       (cal, test) data, regardless of how `q_lo`, `q_hi` were trained.

For backbone capacity planning, the upper edge `q_hi(x) + qhat` is the
provisioned capacity proposal — wrapped by `capacity_from_cqr_upper` to
collapse it to per-link constants matching the existing pipeline.

Per-link vs global qhat. We default to **per-link calibration** because
residual scales differ by an order of magnitude across links in the
Abilene / GÉANT / CESNET traces. Pass `per_link=False` to fit a single
global qhat across all (link, time) cells (Romano 2019's univariate
default). The per-link guarantee is marginal *per link*; the global one
is marginal *aggregated over (link, time)*. Reviewers in our area expect
per-link.

Everything in this module is pure numpy; no training, no torch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _validate_arrays(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray,
                     name: str = "input") -> None:
    if y.ndim != 2 or q_lo.ndim != 2 or q_hi.ndim != 2:
        raise ValueError(
            f"{name}: y / q_lo / q_hi must be 2-D (T, num_links); got "
            f"{y.shape}, {q_lo.shape}, {q_hi.shape}"
        )
    if not (y.shape == q_lo.shape == q_hi.shape):
        raise ValueError(
            f"{name}: shape mismatch — y {y.shape}, q_lo {q_lo.shape}, "
            f"q_hi {q_hi.shape}"
        )


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the (⌈(n+1)(1-α)⌉)/n empirical quantile of `scores`.

    `scores` is a 1-D array. Capped at the maximum score so finite small-n
    cases (where ⌈(n+1)(1-α)⌉ > n) just return the max — the conservative
    choice and what Romano 2019 / Vovk 2005 actually prescribe.
    """
    n = scores.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    # Rank index in {1, ..., n}.
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


def cqr_calibrate(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    alpha: float = 0.1,
    per_link: bool = True,
) -> np.ndarray:
    """Compute the CQR calibration constant `qhat` from a held-out cal set.

    Args:
        y_cal: ground-truth link loads on the calibration set, shape
            `(T_cal, num_links)`.
        q_lo_cal, q_hi_cal: predicted lower and upper quantiles on the
            same calibration set, same shape.
        alpha: target miscoverage. The conformal interval will have
            marginal coverage ≥ `1 - alpha`.
        per_link: if True (default), compute one `qhat` per link from
            that link's residual scores. If False, use a single global
            qhat across all (cal, link) cells.

    Returns:
        ndarray of shape `(num_links,)` if per_link else scalar wrapped in a
        0-d array. Convention: callers always read it as `qhat.shape ==
        (num_links,)` is broadcastable against `(T, num_links)`.
    """
    _validate_arrays(y_cal, q_lo_cal, q_hi_cal, name="cqr_calibrate")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    # Non-conformity score per (t, link): max(q_lo - y, y - q_hi).
    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    num_links = y_cal.shape[1]

    if per_link:
        qhat = np.empty(num_links, dtype=np.float64)
        for ell in range(num_links):
            qhat[ell] = _finite_sample_quantile(scores[:, ell], alpha)
        return qhat
    # Global: pool all cells into one score vector.
    qhat_global = _finite_sample_quantile(scores.ravel(), alpha)
    return np.full(num_links, qhat_global, dtype=np.float64)


def cqr_predict(
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    qhat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the CQR calibration to test-set quantile predictions.

    Args:
        q_lo_test, q_hi_test: predicted lower/upper quantiles on test,
            shape `(T_test, num_links)`.
        qhat: per-link or scalar calibration constant from `cqr_calibrate`.

    Returns:
        `(lower, upper)` arrays of shape `(T_test, num_links)`.
    """
    if q_lo_test.shape != q_hi_test.shape:
        raise ValueError(
            f"shape mismatch: q_lo {q_lo_test.shape}, q_hi {q_hi_test.shape}"
        )
    if q_lo_test.ndim != 2:
        raise ValueError(f"expected 2-D test arrays; got {q_lo_test.shape}")
    qhat = np.asarray(qhat, dtype=np.float64)
    if qhat.ndim == 0:
        qhat_b = qhat
    elif qhat.ndim == 1 and qhat.shape[0] == q_lo_test.shape[1]:
        qhat_b = qhat[None, :]
    else:
        raise ValueError(
            f"qhat shape {qhat.shape} is not broadcastable to "
            f"(T, num_links={q_lo_test.shape[1]})"
        )

    lower = (q_lo_test - qhat_b).astype(np.float32)
    upper = (q_hi_test + qhat_b).astype(np.float32)
    return lower, upper


def empirical_coverage(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict:
    """Empirical marginal coverage on a (held-out) test slice.

    Returns:
        {coverage_overall, coverage_per_link, mean_width, mean_width_per_link}.
        `coverage_overall` is the fraction of (t, link) cells with
        `lower ≤ y ≤ upper`.
    """
    if not (y.shape == lower.shape == upper.shape):
        raise ValueError(
            f"shape mismatch: y {y.shape}, lower {lower.shape}, "
            f"upper {upper.shape}"
        )
    inside = (y >= lower) & (y <= upper)
    width = upper - lower
    return {
        "coverage_overall": float(inside.mean()),
        "coverage_per_link": inside.mean(axis=0).astype(np.float64).tolist(),
        "mean_width": float(width.mean()),
        "mean_width_per_link": width.mean(axis=0).astype(np.float64).tolist(),
    }


def capacity_from_cqr_upper(
    upper: np.ndarray,
    margin: float = 1.0,
) -> np.ndarray:
    """Provisioned capacity = margin × max_t upper(t), per link.

    The CQR upper band already absorbs the safety margin via `qhat`, so
    the default `margin=1.0` is the principled choice. Set `margin > 1`
    only if you want a *belt-and-braces* extra safety factor on top.
    """
    if upper.ndim != 2:
        raise ValueError(f"expected 2-D upper; got {upper.shape}")
    return (margin * np.nanmax(upper, axis=0)).astype(np.float64)


def cqr_split(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    alpha: float = 0.1,
    cal_frac: float = 0.5,
    per_link: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """One-shot helper: split y into cal/test, calibrate, return everything.

    Useful for ablations where you have predictions over a single contiguous
    segment and want to run CQR end-to-end without thinking about the split.
    For paper-grade results use the explicit train/val/test split from the
    dataset loader.

    Args:
        y, q_lo, q_hi: shape `(T, num_links)`.
        alpha: target miscoverage.
        cal_frac: fraction of timesteps used for calibration. The first
            `cal_frac * T` rows go to calibration, the rest to test —
            chronological by default to respect the temporal structure.
        per_link: passed to `cqr_calibrate`.
        rng: unused; reserved for an i.i.d. (random) split variant.

    Returns:
        {qhat, lower, upper, coverage} with `coverage` from
        `empirical_coverage` on the test slice.
    """
    _validate_arrays(y, q_lo, q_hi, name="cqr_split")
    if not (0.0 < cal_frac < 1.0):
        raise ValueError(f"cal_frac must be in (0, 1); got {cal_frac}")

    T = y.shape[0]
    n_cal = int(T * cal_frac)
    if n_cal < 2 or T - n_cal < 1:
        raise ValueError(
            f"need ≥2 cal rows and ≥1 test row; got T={T}, cal_frac={cal_frac}"
        )

    qhat = cqr_calibrate(
        y[:n_cal], q_lo[:n_cal], q_hi[:n_cal], alpha=alpha, per_link=per_link
    )
    lower, upper = cqr_predict(q_lo[n_cal:], q_hi[n_cal:], qhat)
    cov = empirical_coverage(y[n_cal:], lower, upper)
    return {"qhat": qhat, "lower": lower, "upper": upper, "coverage": cov}
