"""Unit tests for src/data/cesnet_loader.py.

Tests the loader on a fabricated parquet directory (no real CESNET data
download needed). Skips automatically if pyarrow is missing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.data.cesnet_loader import (  # noqa: E402
    BYTES_TO_MBPS,
    _list_institution_parquets,
    load_cesnet,
)


def _write_fake_parquet(path: str, n: int, byte_rate: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n_bytes = (rng.uniform(0.5, 1.5, size=n) * byte_rate).astype(np.int64)
    n_flows = rng.integers(low=1, high=100, size=n).astype(np.int64)
    table = pa.table({"n_bytes": n_bytes, "n_flows": n_flows})
    pq.write_table(table, path)


@pytest.fixture
def fake_cesnet(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # Three institutions, all length 200, distinct byte-rates so we can test
    # the top-N subsetting deterministically.
    _write_fake_parquet(str(raw / "inst_a.parquet"), n=200, byte_rate=1e9, seed=0)
    _write_fake_parquet(str(raw / "inst_b.parquet"), n=200, byte_rate=1e8, seed=1)
    _write_fake_parquet(str(raw / "inst_c.parquet"), n=200, byte_rate=1e10, seed=2)
    return raw


class TestListParquets:
    def test_finds_flat_layout(self, fake_cesnet):
        paths = _list_institution_parquets(str(fake_cesnet))
        assert len(paths) == 3
        assert paths == sorted(paths)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _list_institution_parquets(str(tmp_path / "nonexistent"))


class TestLoadCesnet:
    def test_basic_load(self, fake_cesnet):
        data = load_cesnet(str(fake_cesnet), metric="n_bytes")
        assert data["L"].shape == (200, 3)
        assert data["TM"].shape == (200, 3)
        assert data["R"].shape == (3, 3)
        # R should be the identity (CESNET has no real link structure).
        assert np.allclose(data["R"], np.eye(3))
        # Node/link IDs should match the parquet basenames.
        assert sorted(data["nodes"].tolist()) == ["inst_a", "inst_b", "inst_c"]

    def test_bytes_converted_to_mbps(self, fake_cesnet):
        data = load_cesnet(str(fake_cesnet), metric="n_bytes")
        # Roughly: 1e9 bytes/window → 1e9 * BYTES_TO_MBPS ≈ 13.3 Mbps.
        assert 1.0 < data["L"].mean() < 200.0  # institutions average in this range

    def test_top_n_keeps_highest_traffic(self, fake_cesnet):
        # inst_c has byte_rate 1e10, much larger than the others. top_n=1
        # should keep only inst_c.
        data = load_cesnet(str(fake_cesnet), metric="n_bytes",
                           top_n_institutions=1)
        assert data["num_links"] == 1
        assert data["nodes"].tolist() == ["inst_c"]

    def test_split_partitions_T(self, fake_cesnet):
        data = load_cesnet(str(fake_cesnet), metric="n_bytes")
        assert (data["T_train"] + data["T_val"] + data["T_test"]
                == data["T"])

    def test_alternative_metric(self, fake_cesnet):
        data = load_cesnet(str(fake_cesnet), metric="n_flows")
        # Flows are not byte-converted, so values are integer-ish flow counts.
        assert data["metric"] == "n_flows"
        assert data["L"].max() < 200  # we used flows in [1, 100)

    def test_handles_unequal_lengths(self, fake_cesnet, tmp_path):
        # Drop in a fourth parquet of length 150; loader should trim to 150.
        _write_fake_parquet(
            str(fake_cesnet / "inst_d.parquet"), n=150, byte_rate=1e9, seed=3
        )
        data = load_cesnet(str(fake_cesnet), metric="n_bytes")
        assert data["T"] == 150
        assert data["L"].shape[0] == 150
