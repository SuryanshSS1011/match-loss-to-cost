#!/usr/bin/env python
"""Verify the hardcoded GÉANT topology against TOTEM's directed-geant-topology.xml.

Run on the cloud box (or anywhere with the TOTEM tarball unpacked):
    python scripts/verify_geant_topology.py \
        --xml-path data/raw/geant/directed-geant-topology.xml

Compares the directed adjacency built from `GEANT_EDGES` (in
`src/data/geant_loader.py`) against the directed link list in the XML.
Prints any mismatches; exits non-zero if found, zero otherwise.

This is the cheap insurance the loader docstring flags: if TOTEM ever
revises the topology (or our hardcoded constant is wrong), this script
flags it before we burn cloud-budget on a paper-grade sweep against a
silently-incorrect routing matrix.

XML schema (per the TOTEM published spec):
    <topology>
      <nodes><node id="..."/>...</nodes>
      <links>
        <link id="..."><from node="..."/><to node="..."/></link>
        ...
      </links>
    </topology>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable
from xml.etree import ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.geant_loader import (  # noqa: E402
    GEANT_EDGES,
    NUM_NODES,
    _build_directed_adjacency,
)


def parse_totem_topology(xml_path: str) -> tuple[set[str], set[tuple[str, str]]]:
    """Parse a TOTEM topology XML; return (node_ids, directed_edges).

    Both sets use the raw string node IDs from the XML — caller maps them
    to integers via a stable ordering to compare against `GEANT_EDGES`.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodes: set[str] = set()
    for node_el in root.iter("node"):
        nid = node_el.attrib.get("id")
        if nid is not None:
            nodes.add(nid)

    edges: set[tuple[str, str]] = set()
    for link_el in root.iter("link"):
        src_el = link_el.find("from")
        dst_el = link_el.find("to")
        if src_el is None or dst_el is None:
            continue
        src = src_el.attrib.get("node")
        dst = dst_el.attrib.get("node")
        if src is None or dst is None:
            continue
        edges.add((src, dst))
    return nodes, edges


def directed_edges_from_constants() -> set[tuple[int, int]]:
    """Directed adjacency from `GEANT_EDGES` (each undirected edge → 2)."""
    return set(_build_directed_adjacency(GEANT_EDGES))


def diff(xml_path: str) -> dict:
    """Return a diff between the XML topology and our hardcoded constant.

    Returns:
        {
          "match": bool,
          "node_count_xml": int,
          "node_count_const": int,
          "edge_count_xml": int,
          "edge_count_const": int,
          "in_xml_not_const": list of (u, v) tuples (ints if mappable),
          "in_const_not_xml": list of (u, v) tuples (ints),
          "node_label_map": dict[str, int]  (XML → constant index),
        }
    """
    xml_nodes, xml_edges = parse_totem_topology(xml_path)
    const_edges = directed_edges_from_constants()

    # Map XML node IDs to integer indices. The TOTEM release labels nodes
    # "1".."23"; we expect the integer parse to align with the 0..22
    # indices used in GEANT_EDGES (subtract 1).
    label_map: dict[str, int] = {}
    sorted_labels = sorted(xml_nodes, key=lambda s: (len(s), s))
    for i, label in enumerate(sorted_labels):
        # If the label is itself an integer, prefer that ordering.
        try:
            label_map[label] = int(label) - 1
        except ValueError:
            label_map[label] = i

    xml_edges_int: set[tuple[int, int]] = set()
    for u, v in xml_edges:
        if u in label_map and v in label_map:
            xml_edges_int.add((label_map[u], label_map[v]))

    in_xml_only = sorted(xml_edges_int - const_edges)
    in_const_only = sorted(const_edges - xml_edges_int)

    return {
        "match": len(in_xml_only) == 0 and len(in_const_only) == 0,
        "node_count_xml": len(xml_nodes),
        "node_count_const": NUM_NODES,
        "edge_count_xml": len(xml_edges_int),
        "edge_count_const": len(const_edges),
        "in_xml_not_const": in_xml_only,
        "in_const_not_xml": in_const_only,
        "node_label_map": label_map,
    }


def report(d: dict) -> str:
    lines = []
    lines.append(
        f"nodes: xml={d['node_count_xml']}  const={d['node_count_const']}"
    )
    lines.append(
        f"directed edges: xml={d['edge_count_xml']}  "
        f"const={d['edge_count_const']}"
    )
    if d["match"]:
        lines.append("MATCH — hardcoded GEANT_EDGES agrees with TOTEM XML.")
    else:
        lines.append("MISMATCH:")
        if d["in_xml_not_const"]:
            lines.append(f"  in XML but not in GEANT_EDGES "
                         f"({len(d['in_xml_not_const'])}):")
            for u, v in d["in_xml_not_const"]:
                lines.append(f"    ({u}, {v})")
        if d["in_const_not_xml"]:
            lines.append(f"  in GEANT_EDGES but not in XML "
                         f"({len(d['in_const_not_xml'])}):")
            for u, v in d["in_const_not_xml"]:
                lines.append(f"    ({u}, {v})")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="diff hardcoded GEANT_EDGES vs TOTEM topology XML"
    )
    parser.add_argument("--xml-path", required=True,
                        help="path to directed-geant-topology.xml")
    args = parser.parse_args()

    d = diff(args.xml_path)
    print(report(d))
    return 0 if d["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
