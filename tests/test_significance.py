"""Unit tests for src/evaluation/significance.py.

Pure scipy.stats + matplotlib. No real data, no models.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.evaluation.significance import (
    _critical_difference,
    critical_difference_diagram,
    holm_bonferroni,
    paired_wilcoxon,
    pairwise_significance_table,
)


# ---------------------------------------------------------------------------
# paired_wilcoxon
# ---------------------------------------------------------------------------

class TestPairedWilcoxon:
    def test_zero_diff_returns_one(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        res = paired_wilcoxon(a, a)
        assert res["pvalue"] == 1.0
        assert res["n_pairs"] == 0
        assert res["mean_diff"] == 0.0

    def test_large_gap_significant(self):
        # a is uniformly smaller than b; alternative='less' must reject.
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        res = paired_wilcoxon(a, b, alternative="less")
        assert res["pvalue"] < 0.05
        assert res["mean_diff"] < 0.0
        assert res["n_pairs"] == 7

    def test_alternative_direction(self):
        # 'greater' on the same data should NOT reject.
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        res = paired_wilcoxon(a, b, alternative="greater")
        assert res["pvalue"] > 0.5

    def test_two_sided_symmetric(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        # All differences are -1; two-sided rejects.
        res = paired_wilcoxon(a, b, alternative="two-sided")
        assert res["pvalue"] < 0.05

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            paired_wilcoxon([1, 2, 3], [1, 2])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            paired_wilcoxon([1.0], [2.0])

    def test_unknown_alternative(self):
        with pytest.raises(ValueError, match="alternative"):
            paired_wilcoxon([1, 2, 3, 4], [2, 3, 4, 5], alternative="foo")


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------

class TestHolm:
    def test_empty(self):
        out = holm_bonferroni([])
        assert out["adjusted"] == []
        assert out["reject"] == []
        assert out["m"] == 0

    def test_single_pvalue_unchanged(self):
        out = holm_bonferroni([0.03])
        assert out["adjusted"] == [pytest.approx(0.03)]
        assert out["reject"] == [True]

    def test_worked_example(self):
        # Holm step-down on p = [0.01, 0.04, 0.03, 0.005].
        # Sorted asc: 0.005, 0.01, 0.03, 0.04 → factors 4, 3, 2, 1
        # raw: 0.020, 0.030, 0.060, 0.040
        # running max: 0.020, 0.030, 0.060, 0.060
        # → adjusted (orig order):
        #    p=0.005 → 0.020
        #    p=0.01  → 0.030
        #    p=0.03  → 0.060
        #    p=0.04  → 0.060
        out = holm_bonferroni([0.01, 0.04, 0.03, 0.005], alpha=0.05)
        assert out["adjusted"][0] == pytest.approx(0.030)
        assert out["adjusted"][1] == pytest.approx(0.060)
        assert out["adjusted"][2] == pytest.approx(0.060)
        assert out["adjusted"][3] == pytest.approx(0.020)
        assert out["reject"] == [True, False, False, True]

    def test_capped_at_one(self):
        out = holm_bonferroni([0.5, 0.6, 0.7], alpha=0.05)
        assert all(p <= 1.0 for p in out["adjusted"])
        assert all(not r for r in out["reject"])

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            holm_bonferroni([0.1, 0.2], alpha=1.5)

    def test_invalid_pvalue(self):
        with pytest.raises(ValueError):
            holm_bonferroni([0.1, 1.5])


# ---------------------------------------------------------------------------
# pairwise_significance_table
# ---------------------------------------------------------------------------

class TestPairwiseTable:
    def test_basic_three_models(self):
        # Three models, 6 seeds. asym beats both, mse middle, baseline worst.
        values = {
            "asym": [10, 11, 9, 10, 10, 9],
            "mse":  [20, 21, 19, 20, 20, 19],
            "base": [30, 31, 29, 30, 30, 29],
        }
        table = pairwise_significance_table(values, lower_is_better=True)

        # 3 models → 3 unordered pairs.
        assert len(table) == 3
        # Sorted by raw pvalue ascending.
        pvalues = [r["pvalue"] for r in table]
        assert pvalues == sorted(pvalues)

        # Every pair should reject — gaps are huge.
        for r in table:
            assert r["reject"] is True
            assert r["mean_diff"] < 0.0  # a is the smaller of each (a, b)

    def test_reference_only(self):
        # With reference='mse', we get only (asym, mse) and (base, mse).
        values = {
            "asym": [10, 11, 9, 10, 10, 9],
            "mse":  [20, 21, 19, 20, 20, 19],
            "base": [30, 31, 29, 30, 30, 29],
        }
        table = pairwise_significance_table(
            values, reference="mse", lower_is_better=True
        )
        assert len(table) == 2
        for r in table:
            assert r["model_b"] == "mse"
            assert r["model_a"] in {"asym", "base"}

    def test_higher_is_better_flips_direction(self):
        # If higher is better, asym (means=9.83) is the WORST among these.
        values = {
            "asym": [10, 11, 9, 10, 10, 9],     # mean ~ 9.83
            "mse":  [20, 21, 19, 20, 20, 19],   # mean ~ 19.83
        }
        table = pairwise_significance_table(values, lower_is_better=False)
        # Only one pair — and asym should NOT beat mse under higher-is-better.
        assert len(table) == 1
        # Test 'asym > mse' alternative — large positive p-value expected.
        r = next(r for r in table if r["model_a"] == "asym" and r["model_b"] == "mse")
        assert r["pvalue"] > 0.5

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="same #seeds"):
            pairwise_significance_table({"a": [1, 2, 3], "b": [1, 2]})

    def test_too_few_models(self):
        with pytest.raises(ValueError, match=">=2"):
            pairwise_significance_table({"a": [1, 2, 3]})

    def test_unknown_reference(self):
        with pytest.raises(ValueError, match="reference"):
            pairwise_significance_table(
                {"a": [1, 2, 3], "b": [4, 5, 6]}, reference="c"
            )


# ---------------------------------------------------------------------------
# critical_difference_diagram + helper
# ---------------------------------------------------------------------------

class TestCriticalDifference:
    def test_cd_formula(self):
        # M=5, K=3, alpha=0.05: q=2.728, CD = 2.728 * sqrt(5*6/(6*3))
        #                              = 2.728 * sqrt(30/18)
        #                              = 2.728 * 1.2910 ≈ 3.5217.
        cd = _critical_difference(5, 3, alpha=0.05)
        assert cd == pytest.approx(2.728 * (30.0 / 18.0) ** 0.5, rel=1e-3)

    def test_cd_only_05(self):
        with pytest.raises(NotImplementedError):
            _critical_difference(5, 3, alpha=0.10)

    def test_cd_unsupported_M(self):
        with pytest.raises(ValueError, match="M="):
            _critical_difference(11, 3, alpha=0.05)

    def test_diagram_writes_png(self, tmp_path):
        # 4 models × 5 datasets, each cell = mean RMSE.
        rng = np.random.default_rng(0)
        values = {
            "patchtst":  rng.uniform(10, 20, size=5).tolist(),
            "lstm":      rng.uniform(20, 30, size=5).tolist(),
            "dlinear":   rng.uniform(15, 25, size=5).tolist(),
            "sarima":    rng.uniform(30, 40, size=5).tolist(),
        }
        out = tmp_path / "cd.png"
        info = critical_difference_diagram(
            values, str(out), lower_is_better=True, title="test CD"
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert info["M"] == 4
        assert info["K"] == 5
        # patchtst should have the lowest average rank (best).
        ranks = info["ranks"]
        assert ranks["patchtst"] == min(ranks.values())
        assert ranks["sarima"] == max(ranks.values())

    def test_diagram_too_few_datasets(self, tmp_path):
        # Need K >= 2 datasets to rank.
        values = {"a": [1.0], "b": [2.0]}
        with pytest.raises(ValueError, match="K="):
            critical_difference_diagram(values, str(tmp_path / "x.png"))
