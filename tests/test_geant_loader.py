"""Unit tests for src/data/geant_loader.py.

Tests the topology + routing-matrix synthesis without touching any raw data
file. The actual CSV/tarball paths are exercised on the cloud box.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.geant_loader import (
    GEANT_EDGES,
    NUM_NODES,
    _build_directed_adjacency,
    _shortest_path_routing,
    load_geant,
)


class TestTopology:
    def test_directed_adjacency_doubles_undirected_edges(self):
        directed = _build_directed_adjacency(GEANT_EDGES)
        # Each undirected edge → 2 directed; loops would collapse but we
        # don't have any in GEANT_EDGES.
        assert len(directed) == 2 * len(GEANT_EDGES)

    def test_no_self_loops(self):
        directed = _build_directed_adjacency(GEANT_EDGES)
        assert all(u != v for u, v in directed)

    def test_sorted(self):
        directed = _build_directed_adjacency(GEANT_EDGES)
        assert directed == sorted(directed)


class TestShortestPathRouting:
    def test_R_shape(self):
        directed = _build_directed_adjacency(GEANT_EDGES)
        R, demands = _shortest_path_routing(NUM_NODES, directed)
        assert R.shape == (len(directed), NUM_NODES * NUM_NODES)
        assert len(demands) == NUM_NODES * NUM_NODES

    def test_self_pair_columns_zero(self):
        directed = _build_directed_adjacency(GEANT_EDGES)
        R, demands = _shortest_path_routing(NUM_NODES, directed)
        for k, (s, d) in enumerate(demands):
            if s == d:
                assert R[:, k].sum() == 0.0

    def test_routes_use_at_least_one_link(self):
        directed = _build_directed_adjacency(GEANT_EDGES)
        R, demands = _shortest_path_routing(NUM_NODES, directed)
        for k, (s, d) in enumerate(demands):
            if s != d:
                assert R[:, k].sum() >= 1.0, (
                    f"demand {s}→{d} routed on no link"
                )

    def test_routing_entries_binary(self):
        directed = _build_directed_adjacency(GEANT_EDGES)
        R, _ = _shortest_path_routing(NUM_NODES, directed)
        unique = np.unique(R)
        assert set(unique.tolist()).issubset({0.0, 1.0})


class TestLoadGeantValidation:
    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError, match="unknown source"):
            load_geant(source="parquet")

    def test_csv_requires_path(self):
        with pytest.raises(ValueError, match="csv_path"):
            load_geant(source="csv")

    def test_totem_requires_path(self):
        with pytest.raises(ValueError, match="tarball_path"):
            load_geant(source="totem")

    def test_csv_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_geant(source="csv", csv_path="/nonexistent/geant.csv")
