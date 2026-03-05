"""Conformal-prediction calibration utilities.

Public API:
    --- Conformalized Quantile Regression (cqr.py) ---
    cqr_calibrate(y_cal, q_lo_cal, q_hi_cal, alpha, per_link)  -> qhat
    cqr_predict(q_lo_test, q_hi_test, qhat)                    -> (lo, hi)
    cqr_split(y, q_lo, q_hi, alpha, cal_frac, per_link)        -> dict
    empirical_coverage(y, lower, upper)                        -> dict
    capacity_from_cqr_upper(upper, margin)                     -> ndarray

    --- Adaptive Conformal Inference (aci.py) ---
    aci_step(in_band, alpha_t, target_alpha, gamma)            -> alpha_{t+1}
    sliding_quantile(scores, alpha)                            -> qhat
    ACITracker(num_links, target_alpha, gamma, window)         -- online state
"""

from .aci import ACITracker, aci_step, sliding_quantile
from .cqr import (
    capacity_from_cqr_upper,
    cqr_calibrate,
    cqr_predict,
    cqr_split,
    empirical_coverage,
)

__all__ = [
    "capacity_from_cqr_upper",
    "cqr_calibrate",
    "cqr_predict",
    "cqr_split",
    "empirical_coverage",
    "aci_step",
    "sliding_quantile",
    "ACITracker",
]
