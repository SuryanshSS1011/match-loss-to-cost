"""
Abilene TM dataset loader.

Source: Y. Zhang, "Abilene Traffic Matrices",
        https://www.cs.utexas.edu/~yzhang/research/AbileneTM/
Backbone: Internet2/Abilene, 12 routers, 54 directed links, 144 OD pairs,
          5-minute resolution, 24 weeks (2004-03-01 → 2004-09-04).

Raw data layout (in data/raw/abilene/):
  readme.txt, topo-2003-04-10.txt
  links     — 54 lines: "x,y idx type"  (type: 0 internal, 1 inbound, 2 outbound)
  demands   — 144 lines: "s,d idx"
  A         — routing matrix as triplets: "link_str dmd_str link_idx dmd_idx frac"
  X01.gz .. X24.gz — each: 2016 lines × 720 floats per line.
                   720 = 144 OD pairs × 5 estimates per pair.
                   Real OD is column 0 of every 5; gravity/tomogravity are 1..4.
                   Unit: "100 bytes / 5 min" (i.e., raw * 100 bytes per 5-min interval).

Output: data/abilene_traffic.npz with same schema as the synthetic pipeline:
  TM        — (T, num_od)   float32, units = Mbps
  L         — (T, num_links) float32, units = Mbps
  R         — (num_links, num_od) float32, routing matrix
  links     — (num_links, 2) str, directed link endpoints
  demands   — (num_od, 2)    str, OD endpoints
  nodes     — (num_nodes,)   str, router names
  train_end, val_end, T_train, T_val, T_test — split indices
"""

from __future__ import annotations

import gzip
import os
from typing import Optional

import numpy as np


RAW_BASE_URL = "https://www.cs.utexas.edu/~yzhang/research/AbileneTM"
NUM_OD = 144
NUM_LINKS = 54
NUM_WEEKS = 24
STEPS_PER_WEEK = 2016         # 12 * 24 * 7
TIME_STEP_MINUTES = 5
ESTIMATES_PER_OD = 5
# Raw unit: 100 bytes per 5-minute interval. Convert to Mbps:
#   bytes_per_5min = raw * 100
#   bits_per_5min  = raw * 800
#   bps            = raw * 800 / 300
#   Mbps           = raw * 800 / 300 / 1e6
RAW_TO_MBPS = 800.0 / 300.0 / 1e6


def _read_links(path: str) -> tuple[list[tuple[str, str]], list[int]]:
    """Read the `links` file. Returns (endpoints, types) ordered by 1-indexed idx."""
    rows: list[tuple[int, str, str, int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            edge_str, idx_str, type_str = line.split()
            x, y = edge_str.split(",")
            rows.append((int(idx_str), x, y, int(type_str)))
    rows.sort(key=lambda r: r[0])
    if [r[0] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"link indices not 1..{len(rows)} contiguous")
    endpoints = [(r[1], r[2]) for r in rows]
    types = [r[3] for r in rows]
    return endpoints, types


def _read_demands(path: str) -> list[tuple[str, str]]:
    """Read the `demands` file. Returns OD endpoints ordered by 1-indexed idx."""
    rows: list[tuple[int, str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            od_str, idx_str = line.split()
            s, d = od_str.split(",")
            rows.append((int(idx_str), s, d))
    rows.sort(key=lambda r: r[0])
    if [r[0] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"demand indices not 1..{len(rows)} contiguous")
    return [(r[1], r[2]) for r in rows]


def _read_routing_matrix(path: str, num_links: int, num_od: int) -> np.ndarray:
    """Read the `A` file; return dense (num_links, num_od) routing matrix."""
    R = np.zeros((num_links, num_od), dtype=np.float32)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # parts: link_str dmd_str link_idx dmd_idx frac
            link_idx = int(parts[2]) - 1
            dmd_idx = int(parts[3]) - 1
            frac = float(parts[4])
            R[link_idx, dmd_idx] = frac
    return R


def _read_week(path_gz: str) -> np.ndarray:
    """Read one X??.gz file → (2016, 144) float32 of real-OD values (raw units)."""
    arr = np.empty((STEPS_PER_WEEK, NUM_OD), dtype=np.float32)
    with gzip.open(path_gz, "rt") as f:
        for t, line in enumerate(f):
            tokens = line.split()
            if len(tokens) != NUM_OD * ESTIMATES_PER_OD:
                raise ValueError(
                    f"{path_gz}: line {t} has {len(tokens)} tokens, "
                    f"expected {NUM_OD * ESTIMATES_PER_OD}"
                )
            # Column 0 of every 5 is the real OD value.
            for k in range(NUM_OD):
                arr[t, k] = float(tokens[k * ESTIMATES_PER_OD])
    if t + 1 != STEPS_PER_WEEK:
        raise ValueError(f"{path_gz}: got {t+1} rows, expected {STEPS_PER_WEEK}")
    return arr


def load_abilene(
    raw_dir: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    num_weeks: Optional[int] = None,
    drop_self_pairs: bool = True,
) -> dict:
    """
    Load and assemble the Abilene TM dataset.

    Args:
        raw_dir: directory containing links, demands, A, X01.gz..X24.gz
        train_frac, val_frac: split fractions; test is 1 - train - val
        num_weeks: if given, only load the first N weeks (useful for smoke tests)
        drop_self_pairs: if True (default), drop the 12 demands where source == dest.
            These are intra-PoP/loopback artifacts in Yin Zhang's dataset, not real
            backbone OD-traffic. They carry ~56% of nominal "traffic" but include the
            single 142 Tbps anomaly at week 24 row 1349 (ATLAng→ATLAng). Dropping them
            yields 132 OD-pairs (matching TUBO/LEAD usage of this dataset).

    Returns:
        Dict with keys: TM, L, R, links, demands, nodes, T, num_links, num_od,
        num_nodes, train_end, val_end, T_train, T_val, T_test, time_step_minutes.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(raw_dir)

    links_path = os.path.join(raw_dir, "links")
    demands_path = os.path.join(raw_dir, "demands")
    A_path = os.path.join(raw_dir, "A")
    for p in (links_path, demands_path, A_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}")

    print(f"[abilene] reading topology from {raw_dir}")
    link_endpoints, link_types = _read_links(links_path)
    if len(link_endpoints) != NUM_LINKS:
        raise ValueError(f"expected {NUM_LINKS} links, got {len(link_endpoints)}")
    od_endpoints = _read_demands(demands_path)
    if len(od_endpoints) != NUM_OD:
        raise ValueError(f"expected {NUM_OD} demands, got {len(od_endpoints)}")

    R = _read_routing_matrix(A_path, NUM_LINKS, NUM_OD)
    print(f"[abilene]   R shape={R.shape}  density={R.mean():.3f}  "
          f"row-sums [{R.sum(axis=1).min():.1f}, {R.sum(axis=1).max():.1f}]")

    # Node list = unique routers across the demand list.
    nodes = sorted({s for s, _ in od_endpoints} | {d for _, d in od_endpoints})

    n_weeks = num_weeks if num_weeks is not None else NUM_WEEKS
    print(f"[abilene] reading {n_weeks} week files")
    TM_weeks = []
    for w in range(1, n_weeks + 1):
        path_gz = os.path.join(raw_dir, f"X{w:02d}.gz")
        if not os.path.exists(path_gz):
            raise FileNotFoundError(path_gz)
        TM_weeks.append(_read_week(path_gz))
        if w == 1 or w % 6 == 0 or w == n_weeks:
            print(f"[abilene]   X{w:02d}.gz done")
    TM_raw = np.concatenate(TM_weeks, axis=0)  # (T, 144), raw units

    # Convert to Mbps. Cast to float32 after the multiply.
    TM = (TM_raw * RAW_TO_MBPS).astype(np.float32)

    # Optionally drop self-pair demands (s == d). See docstring + STEPS.md (2026-04-28).
    if drop_self_pairs:
        keep = np.array([s != d for s, d in od_endpoints], dtype=bool)
        n_dropped = int((~keep).sum())
        TM = TM[:, keep]
        R = R[:, keep]
        od_endpoints = [od for od, k in zip(od_endpoints, keep) if k]
        print(f"[abilene]   dropped {n_dropped} self-pair demands → "
              f"{TM.shape[1]} OD-pairs, R shape={R.shape}")

    # Compute link loads using R.
    L = TM @ R.T  # (T, num_links)
    L = L.astype(np.float32)

    T = TM.shape[0]
    T_train = int(T * train_frac)
    T_val = int(T * val_frac)
    T_test = T - T_train - T_val

    print(f"[abilene] T={T} ({T // (24 * 60 // TIME_STEP_MINUTES)} days), "
          f"TM range [{TM.min():.4f}, {TM.max():.4f}] Mbps, "
          f"L range [{L.min():.4f}, {L.max():.4f}] Mbps")
    print(f"[abilene] split: train={T_train}  val={T_val}  test={T_test}")

    return {
        "TM": TM,
        "L": L,
        "R": R,
        "links": np.array(link_endpoints),
        "link_types": np.array(link_types, dtype=np.int32),
        "demands": np.array(od_endpoints),
        "nodes": np.array(nodes),
        "T": T,
        "num_links": NUM_LINKS,
        "num_od": TM.shape[1],
        "num_nodes": len(nodes),
        "T_train": T_train,
        "T_val": T_val,
        "T_test": T_test,
        "train_end": T_train,
        "val_end": T_train + T_val,
        "time_step_minutes": TIME_STEP_MINUTES,
    }


def save_abilene(out_path: str, data: dict) -> None:
    """Save loader output to a .npz file with the project's standard schema."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        TM=data["TM"],
        L=data["L"],
        R=data["R"],
        links=data["links"],
        link_types=data["link_types"],
        demands=data["demands"],
        nodes=data["nodes"],
        T=data["T"],
        num_links=data["num_links"],
        num_od=data["num_od"],
        num_nodes=data["num_nodes"],
        T_train=data["T_train"],
        T_val=data["T_val"],
        T_test=data["T_test"],
        train_end=data["train_end"],
        val_end=data["val_end"],
        time_step_minutes=data["time_step_minutes"],
    )
    print(f"[abilene] saved → {out_path}")


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(project_root, "data", "raw", "abilene")
    out_path = os.path.join(project_root, "data", "abilene_traffic.npz")
    data = load_abilene(raw_dir)
    save_abilene(out_path, data)


if __name__ == "__main__":
    main()
