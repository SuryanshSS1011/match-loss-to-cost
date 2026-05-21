"""Unit tests for src/data/cesnet_loader.py.

Tests the loader on a fabricated institution directory (no real CESNET data
download needed). Covers both the published CSV format and the parquet format
some mirrors repackage to. Parquet cases skip if pyarrow is missing.
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
    _list_institution_files,
    load_cesnet,
)


def _write_fake_csv(path: str, n: int, byte_rate: float, seed: int = 0) -> None:
    """Mimic the published institutions/agg_10_minutes/<id>.csv schema."""
    rng = np.random.default_rng(seed)
    n_bytes = (rng.uniform(0.5, 1.5, size=n) * byte_rate).astype(np.int64)
    n_flows = rng.integers(low=1, high=100, size=n).astype(np.int64)
    id_time = np.arange(n)
    with open(path, "w") as f:
        f.write("id_time,n_flows,n_bytes\n")
        for t, fl, b in zip(id_time, n_flows, n_bytes):
            f.write(f"{t},{fl},{b}\n")


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


class TestListFiles:
    def test_finds_flat_layout(self, fake_cesnet):
        paths = _list_institution_files(str(fake_cesnet))
        assert len(paths) == 3

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _list_institution_files(str(tmp_path / "nonexistent"))

    def test_finds_csv_in_agg_10_minutes_subdir(self, tmp_path):
        # Mirror the real institutions.tar.gz layout: agg_10_minutes/<id>.csv,
        # with numeric ids that must sort naturally (0,1,2,10 — not 0,1,10,2).
        sub = tmp_path / "agg_10_minutes"
        sub.mkdir()
        for i in (0, 1, 2, 10):
            _write_fake_csv(str(sub / f"{i}.csv"), n=50, byte_rate=1e9, seed=i)
        paths = _list_institution_files(str(tmp_path))
        assert [os.path.splitext(os.path.basename(p))[0] for p in paths] == \
            ["0", "1", "2", "10"]

    def test_loads_real_csv_layout(self, tmp_path):
        sub = tmp_path / "agg_10_minutes"
        sub.mkdir()
        _write_fake_csv(str(sub / "0.csv"), n=120, byte_rate=1e10, seed=0)
        _write_fake_csv(str(sub / "1.csv"), n=120, byte_rate=1e9, seed=1)
        data = load_cesnet(str(tmp_path), metric="n_bytes")
        assert data["L"].shape == (120, 2)
        assert np.allclose(data["R"], np.eye(2))


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

    def test_drops_far_too_short_institution(self, fake_cesnet, tmp_path):
        # A length-150 trace among length-200 traces is >5% short of the modal
        # length, so it should be DROPPED (not used to collapse everyone to 150).
        # This is the fix for the real CESNET release, where a few empty/stub
        # institutions otherwise dragged the global min to 0.
        _write_fake_parquet(
            str(fake_cesnet / "inst_d.parquet"), n=150, byte_rate=1e9, seed=3
        )
        data = load_cesnet(str(fake_cesnet), metric="n_bytes")
        assert data["T"] == 200
        assert data["num_links"] == 3  # inst_d dropped
        assert "inst_d" not in data["nodes"].tolist()

    def test_trims_within_tolerance_length(self, fake_cesnet, tmp_path):
        # A length-198 trace (1% short of modal 200) is within tolerance, so it
        # is KEPT and everyone is trimmed to the common min (198).
        _write_fake_parquet(
            str(fake_cesnet / "inst_d.parquet"), n=198, byte_rate=1e9, seed=3
        )
        data = load_cesnet(str(fake_cesnet), metric="n_bytes")
        assert data["T"] == 198
        assert data["num_links"] == 4

    def test_empty_institution_does_not_collapse_dataset(self, fake_cesnet):
        # An empty (length-0) institution must be dropped, not zero out T.
        _write_fake_parquet(
            str(fake_cesnet / "inst_empty.parquet"), n=0, byte_rate=1e9, seed=4
        )
        data = load_cesnet(str(fake_cesnet), metric="n_bytes")
        assert data["T"] == 200
        assert data["num_links"] == 3
