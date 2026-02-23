"""Unit tests for scripts/verify_geant_topology.py.

Builds fabricated TOTEM-shaped XML in tmp_path and exercises the verifier.
No real GÉANT XML download needed.
"""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)

import verify_geant_topology as vg  # noqa: E402
from src.data.geant_loader import GEANT_EDGES, _build_directed_adjacency


def _xml(nodes: list[str], directed_edges: list[tuple[str, str]]) -> str:
    node_block = "\n".join(f'    <node id="{n}"/>' for n in nodes)
    link_block = "\n".join(
        f'    <link id="L{i}"><from node="{u}"/><to node="{v}"/></link>'
        for i, (u, v) in enumerate(directed_edges)
    )
    return (
        "<topology>\n"
        f"  <nodes>\n{node_block}\n  </nodes>\n"
        f"  <links>\n{link_block}\n  </links>\n"
        "</topology>\n"
    )


def _xml_matching_constants() -> str:
    """Build an XML that should diff-clean against GEANT_EDGES."""
    # Constants are 0-indexed; XML labels are 1-indexed strings (TOTEM convention).
    nodes = [str(i + 1) for i in range(23)]
    directed = _build_directed_adjacency(GEANT_EDGES)
    edges_str = [(str(u + 1), str(v + 1)) for u, v in directed]
    return _xml(nodes, edges_str)


def test_diff_match(tmp_path):
    xml_path = tmp_path / "topo.xml"
    xml_path.write_text(_xml_matching_constants())
    d = vg.diff(str(xml_path))
    assert d["match"] is True
    assert d["node_count_xml"] == 23
    assert d["edge_count_xml"] == 2 * len(GEANT_EDGES)
    assert d["edge_count_const"] == 2 * len(GEANT_EDGES)


def test_diff_extra_edge_in_xml(tmp_path):
    nodes = [str(i + 1) for i in range(23)]
    directed = _build_directed_adjacency(GEANT_EDGES)
    edges_str = [(str(u + 1), str(v + 1)) for u, v in directed]
    # (1, 22) and (22, 1) are not in GEANT_EDGES — verified locally.
    edges_str.append(("2", "23"))   # 1-indexed → (1, 22) in 0-indexed.
    edges_str.append(("23", "2"))
    xml_path = tmp_path / "topo.xml"
    xml_path.write_text(_xml(nodes, edges_str))

    d = vg.diff(str(xml_path))
    assert d["match"] is False
    assert (1, 22) in d["in_xml_not_const"]
    assert (22, 1) in d["in_xml_not_const"]
    assert d["in_const_not_xml"] == []


def test_diff_missing_edge_in_xml(tmp_path):
    nodes = [str(i + 1) for i in range(23)]
    directed = _build_directed_adjacency(GEANT_EDGES)
    # Drop the first directed edge.
    dropped = directed[0]
    edges_str = [(str(u + 1), str(v + 1)) for u, v in directed[1:]]
    xml_path = tmp_path / "topo.xml"
    xml_path.write_text(_xml(nodes, edges_str))

    d = vg.diff(str(xml_path))
    assert d["match"] is False
    assert dropped in d["in_const_not_xml"]


def test_report_text(tmp_path):
    xml_path = tmp_path / "topo.xml"
    xml_path.write_text(_xml_matching_constants())
    text = vg.report(vg.diff(str(xml_path)))
    assert "MATCH" in text
    assert "nodes:" in text
    assert "directed edges:" in text


def test_diff_missing_xml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vg.diff(str(tmp_path / "no_such_file.xml"))


def test_parse_handles_duplicate_link_ids(tmp_path):
    nodes = ["1", "2"]
    edges = [("1", "2"), ("2", "1"), ("1", "2")]  # duplicate
    xml_path = tmp_path / "topo.xml"
    xml_path.write_text(_xml(nodes, edges))
    nodes_out, edges_out = vg.parse_totem_topology(str(xml_path))
    # Sets dedupe duplicates.
    assert nodes_out == {"1", "2"}
    assert edges_out == {("1", "2"), ("2", "1")}
