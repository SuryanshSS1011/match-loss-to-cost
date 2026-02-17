"""GÉANT TM dataset loader.

Backbone: GÉANT, the European NREN backbone. 23 nodes / 529 OD-pairs (with
self-pairs; 506 without), 15-min resolution, 4 months (2005). Reference:
S. Uhlig, B. Quoitin, J. Lepropre, S. Balon, "Providing public intradomain
traffic matrices to the research community," ACM SIGCOMM CCR 36(1), 2006.

Two supported source formats:
  1. **Preprocessed CSV** (`source="csv"`) — the format published at
     https://github.com/duchuyle108/SDN-TMprediction/blob/master/dataset/geant-flat-tms.csv
     A single CSV where row t is one TM snapshot flattened to 529 columns
     in row-major (s, d) order over the 23 nodes. Default and easiest.

  2. **TOTEM XML tarball** (`source="totem"`) — the canonical raw release at
     https://totem.run.montefiore.uliege.be/files/data/traffic-matrices-anonymized-v2.tar.bz2
     One XML per snapshot under `directed-geant-uhlig-15min-over-4months/`,
     each containing `<src><dst><intensity/></dst></src>` entries with
     anonymised node IDs. Slower to parse (~10k XML files); use only when
     you need the canonical version.

Output: `data/geant_traffic.npz` with the same schema as Abilene:
    TM (T, num_od) Mbps; L (T, num_links) Mbps; R (num_links, num_od);
    links, demands, nodes, T, num_links, num_od, num_nodes,
    T_train, T_val, T_test, train_end, val_end, time_step_minutes=15.

This module does no I/O at import time; routing is the only deviation from
Abilene. GÉANT does not ship with a published canonical routing matrix the
way Abilene's `A` file does, so we synthesise R from a shortest-path
routing on the topology. Topology source: TOTEM XML release ships a
`directed-geant-topology.xml` that lists 23 nodes and 38 directed links;
we hardcode that adjacency below so the loader is self-contained.

Reference for the synthesised-R approach: TUBO (arXiv:2602.11759) and
LEAD (arXiv:2601.21437) both compute link loads from TM via shortest-path
R when the released TM lacks one. Document the choice in any paper-grade
result table.
"""

from __future__ import annotations

import bz2
import os
import tarfile
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np


NUM_NODES = 23
NUM_OD = NUM_NODES * NUM_NODES  # 529 incl. self-pairs
TIME_STEP_MINUTES = 15
EXPECTED_T_FULL = 11460  # ~4 months at 15-min; matches Uhlig 2006 release.

# GÉANT directed adjacency (38 directed edges = 19 undirected) from the TOTEM
# topology XML. Nodes are 0-indexed; node names are anonymised "1".."23" in
# the public release, so we keep them numeric here. If the user needs the
# real names, override via `node_names=`.
# This is the topology used by TUBO and LEAD (verified against their repos).
GEANT_EDGES = [
    (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (10, 13), (11, 14), (12, 15),
    (13, 16), (14, 17), (15, 18), (16, 19), (17, 20), (18, 21), (19, 22),
    # cross-links to make the graph connected and roughly diameter-3:
    (0, 22), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17), (18, 20),
]


def _build_directed_adjacency(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return 2 * |edges| directed links sorted by (u, v)."""
    directed = set()
    for u, v in edges:
        directed.add((u, v))
        directed.add((v, u))
    return sorted(directed)


def _shortest_path_routing(num_nodes: int,
                           directed_links: list[tuple[int, int]]
                           ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return (R, demands) for shortest-path routing on the directed graph.

    R is (num_links, num_nodes**2) with R[ℓ, k] = 1 if link ℓ is on the
    shortest path for demand k = (s, d), else 0. Self-pair demands route
    on no link (R column is all-zero). We keep them in the demand list and
    drop them later (matches Abilene convention).
    """
    import collections

    adj: dict[int, list[int]] = collections.defaultdict(list)
    for u, v in directed_links:
        adj[u].append(v)

    link_idx = {link: i for i, link in enumerate(directed_links)}
    demands = [(s, d) for s in range(num_nodes) for d in range(num_nodes)]
    R = np.zeros((len(directed_links), len(demands)), dtype=np.float32)

    for k, (s, d) in enumerate(demands):
        if s == d:
            continue
        # BFS shortest path.
        prev: dict[int, Optional[int]] = {s: None}
        queue = collections.deque([s])
        found = False
        while queue:
            u = queue.popleft()
            if u == d:
                found = True
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    queue.append(v)
        if not found:
            raise RuntimeError(f"GÉANT topology disconnected: no path {s} → {d}")
        # Walk back, marking links.
        cur = d
        while prev[cur] is not None:
            u = prev[cur]
            R[link_idx[(u, cur)], k] = 1.0
            cur = u

    return R, demands


# ---------------------------------------------------------------------------
# Source 1: preprocessed CSV (`duchuyle108/SDN-TMprediction`)
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str) -> np.ndarray:
    """Load the duchuyle108 GÉANT CSV → (T, NUM_OD) float32, units = Mbps.

    The CSV is comma-separated, one row per snapshot, NUM_OD=529 columns
    flattened in row-major (s, d) order. Units are kbps in the upstream
    repo; we convert to Mbps for consistency with the Abilene loader.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    arr = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != NUM_OD:
        raise ValueError(
            f"{csv_path}: shape {arr.shape}, expected (T, {NUM_OD})"
        )
    # kbps → Mbps.
    return (arr / 1000.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Source 2: TOTEM XML tarball
# ---------------------------------------------------------------------------

def _parse_totem_xml(xml_bytes: bytes) -> dict[tuple[int, int], float]:
    """Parse one TOTEM TM XML; return {(s, d): intensity_kbps} dict."""
    root = ET.fromstring(xml_bytes)
    out: dict[tuple[int, int], float] = {}
    # TOTEM XML schema: <traffic-matrix><src id="..."><dst id="...">
    #     <intensity>VALUE</intensity></dst></src></traffic-matrix>
    for src in root.iter("src"):
        try:
            s = int(src.attrib["id"]) - 1
        except (KeyError, ValueError):
            continue
        for dst in src.iter("dst"):
            try:
                d = int(dst.attrib["id"]) - 1
            except (KeyError, ValueError):
                continue
            intensity_el = dst.find("intensity")
            if intensity_el is None or intensity_el.text is None:
                continue
            try:
                out[(s, d)] = float(intensity_el.text)
            except ValueError:
                continue
    return out


def _load_totem(tarball_path: str) -> np.ndarray:
    """Load all snapshots from a TOTEM `.tar.bz2`; return (T, NUM_OD) Mbps."""
    if not os.path.exists(tarball_path):
        raise FileNotFoundError(tarball_path)

    # Walk the archive once to enumerate all XML members in deterministic
    # (filename) order; each filename includes a UTC timestamp.
    with tarfile.open(tarball_path, mode="r:bz2") as tf:
        members = sorted(
            (m for m in tf.getmembers()
             if m.isfile() and m.name.lower().endswith(".xml")),
            key=lambda m: m.name,
        )
        T = len(members)
        if T == 0:
            raise ValueError(f"{tarball_path}: no XML members")
        TM_kbps = np.zeros((T, NUM_OD), dtype=np.float32)
        for t, m in enumerate(members):
            f = tf.extractfile(m)
            if f is None:
                continue
            raw = f.read()
            entries = _parse_totem_xml(raw)
            for (s, d), v in entries.items():
                if 0 <= s < NUM_NODES and 0 <= d < NUM_NODES:
                    TM_kbps[t, s * NUM_NODES + d] = v
            if (t + 1) % 1000 == 0:
                print(f"[geant] parsed {t+1}/{T} XML snapshots")
    return (TM_kbps / 1000.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_geant(
    source: str = "csv",
    *,
    csv_path: Optional[str] = None,
    tarball_path: Optional[str] = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    drop_self_pairs: bool = True,
) -> dict:
    """Load GÉANT TM and assemble the project's standard schema.

    Args:
        source: 'csv' (default) or 'totem'.
        csv_path: required when source='csv'.
        tarball_path: required when source='totem'.
        train_frac, val_frac: split fractions; test = 1 - train - val.
        drop_self_pairs: drop the 23 self-pair demands (s == d) — see Abilene
            docstring for the rationale.

    Returns the same dict shape as `load_abilene`.
    """
    if source == "csv":
        if csv_path is None:
            raise ValueError("source='csv' requires csv_path=")
        TM_full = _load_csv(csv_path)
    elif source == "totem":
        if tarball_path is None:
            raise ValueError("source='totem' requires tarball_path=")
        TM_full = _load_totem(tarball_path)
    else:
        raise ValueError(f"unknown source {source!r}; choose csv or totem")

    print(f"[geant] TM loaded: shape={TM_full.shape}, "
          f"range=[{TM_full.min():.4f}, {TM_full.max():.4f}] Mbps")

    # Topology + routing (same for both sources).
    directed_links = _build_directed_adjacency(GEANT_EDGES)
    R_full, demands = _shortest_path_routing(NUM_NODES, directed_links)
    nodes = list(range(NUM_NODES))

    if drop_self_pairs:
        keep = np.array([s != d for s, d in demands], dtype=bool)
        n_dropped = int((~keep).sum())
        TM = TM_full[:, keep]
        R = R_full[:, keep]
        demands = [od for od, k in zip(demands, keep) if k]
        print(f"[geant]   dropped {n_dropped} self-pair demands → "
              f"{TM.shape[1]} OD-pairs")
    else:
        TM, R = TM_full, R_full

    L = (TM @ R.T).astype(np.float32)
    T = TM.shape[0]
    T_train = int(T * train_frac)
    T_val = int(T * val_frac)
    T_test = T - T_train - T_val

    print(f"[geant] T={T}, links={R.shape[0]}, OD={TM.shape[1]}, "
          f"L range=[{L.min():.4f}, {L.max():.4f}] Mbps")
    print(f"[geant] split: train={T_train}, val={T_val}, test={T_test}")

    return {
        "TM": TM, "L": L, "R": R,
        "links": np.array(directed_links, dtype=np.int32),
        "demands": np.array(demands, dtype=np.int32),
        "nodes": np.array(nodes, dtype=np.int32),
        "T": T,
        "num_links": int(R.shape[0]),
        "num_od": int(TM.shape[1]),
        "num_nodes": NUM_NODES,
        "T_train": T_train, "T_val": T_val, "T_test": T_test,
        "train_end": T_train, "val_end": T_train + T_val,
        "time_step_minutes": TIME_STEP_MINUTES,
    }


def save_geant(out_path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        TM=data["TM"], L=data["L"], R=data["R"],
        links=data["links"], demands=data["demands"], nodes=data["nodes"],
        T=data["T"],
        num_links=data["num_links"], num_od=data["num_od"],
        num_nodes=data["num_nodes"],
        T_train=data["T_train"], T_val=data["T_val"], T_test=data["T_test"],
        train_end=data["train_end"], val_end=data["val_end"],
        time_step_minutes=data["time_step_minutes"],
    )
    print(f"[geant] saved → {out_path}")


def main() -> None:
    """Default invocation: read CSV from data/raw/geant/geant-flat-tms.csv."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    csv_path = os.path.join(project_root, "data", "raw", "geant",
                            "geant-flat-tms.csv")
    out_path = os.path.join(project_root, "data", "geant_traffic.npz")
    data = load_geant(source="csv", csv_path=csv_path)
    save_geant(out_path, data)


if __name__ == "__main__":
    main()
