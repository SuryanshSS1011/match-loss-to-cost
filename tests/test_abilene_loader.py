"""Regression guards on the cached Abilene tensor.

Reads `data/abilene_traffic.npz` (built once by `python -m src.data.abilene_loader`)
and asserts the invariants we relied on while building the loader. No
re-parsing of raw files happens here, so the test is fast (<1 s) and runs on
the laptop without touching the model.

Reference: STEPS.md 2026-04-28 entries on self-pair drop and the 142 Tbps
single-cell anomaly. If those invariants regress, the loader is broken.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.config import DATA_DIR


ABILENE_NPZ = os.path.join(DATA_DIR, "abilene_traffic.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ABILENE_NPZ),
    reason="abilene_traffic.npz not built — run `python -m src.data.abilene_loader`",
)


@pytest.fixture(scope="module")
def abilene():
    return np.load(ABILENE_NPZ, allow_pickle=False)


class TestSchema:
    def test_required_keys_present(self, abilene):
        for key in (
            "TM",
            "L",
            "R",
            "T_train",
            "T_val",
            "T_test",
            "train_end",
            "val_end",
            "num_links",
            "num_od",
        ):
            assert key in abilene.files, f"missing key: {key}"

    def test_shapes_consistent(self, abilene):
        T, num_od = abilene["TM"].shape
        assert abilene["L"].shape == (T, int(abilene["num_links"]))
        assert abilene["R"].shape == (int(abilene["num_links"]), num_od)
        assert int(abilene["num_od"]) == num_od

    def test_split_indices_partition_T(self, abilene):
        T = abilene["TM"].shape[0]
        T_train = int(abilene["T_train"])
        T_val = int(abilene["T_val"])
        T_test = int(abilene["T_test"])
        assert T_train + T_val + T_test == T
        assert int(abilene["train_end"]) == T_train
        assert int(abilene["val_end"]) == T_train + T_val


class TestSelfPairDrop:
    def test_132_od_pairs(self, abilene):
        # 144 - 12 self-pairs == 132. STEPS.md 2026-04-28.
        assert int(abilene["num_od"]) == 132

    def test_no_self_pairs_in_demands(self, abilene):
        demands = abilene["demands"]
        assert demands.shape == (132, 2)
        for s, d in demands:
            assert s != d, f"self-pair survived the drop: {s} → {d}"


class TestRoutingInvariants:
    def test_L_reconstructs_from_TM_and_R(self, abilene):
        # L = TM @ R.T should hold by construction. Check on the first
        # ~1000 timesteps to keep the assertion cheap.
        TM = abilene["TM"][:1000]
        R = abilene["R"]
        L_expected = TM @ R.T
        L_actual = abilene["L"][:1000]
        assert np.allclose(L_actual, L_expected, rtol=1e-4, atol=1e-3)

    def test_routing_matrix_in_unit_interval(self, abilene):
        # R entries are routing fractions, so they should live in [0, 1].
        R = abilene["R"]
        assert R.min() >= 0.0
        assert R.max() <= 1.0 + 1e-6


class TestLinkLoadMagnitudes:
    def test_no_link_exceeds_15_gbps(self, abilene):
        # Abilene nominal capacity was 9.92 Gbps per link. The 15 Gbps
        # threshold is a regression guard: anything above this almost
        # certainly means the 142 Tbps self-pair anomaly slipped back in.
        # STEPS.md 2026-04-28.
        L = abilene["L"]
        L_GBPS = L / 1000.0  # Mbps → Gbps
        max_load = float(L_GBPS.max())
        assert max_load <= 15.0, (
            f"some link load is {max_load:.1f} Gbps — self-pair anomaly?"
        )

    def test_no_negative_loads(self, abilene):
        # Link loads are nonnegative by physics; a negative value would mean
        # a sign error or a numerical underflow.
        assert float(abilene["L"].min()) >= 0.0
        assert float(abilene["TM"].min()) >= 0.0

    def test_traffic_is_nonconstant(self, abilene):
        # A zeroed-out file would still have shape and pass everything else.
        # Catch the "loader silently produced zeros" failure mode.
        assert float(abilene["L"].std()) > 0.0
