"""CESNET-TimeSeries24 dataset loader.

Source: J. Koumar et al., "CESNET-TimeSeries24: A time-series dataset for
network traffic anomaly detection and forecasting," *Sci. Data* 12:338,
2025. CC-BY on Zenodo (DOI 10.5281/zenodo.13382427).
Repo: github.com/koumajos/CESNET-TimeSeries24-Example.

Dataset shape (per Koumar 2025):
  - 40 weeks, 10-min aggregates → ~40320 timesteps,
  - aggregations available at multiple resolutions (10-min, 1-hour, 1-day),
  - per-institution and per-IP granularity (we use per-institution),
  - metric columns include n_flows, n_packets, n_bytes, n_dest_ip, ...

This dataset does NOT have a TM or topology. For the Provision-Aware
pipeline (link-load forecasting + capacity planning), we treat each
institution-level trace as a "link," set R = I (identity), and TM = L.
This matches Lentini et al. (arXiv:2503.17410) usage: they treat per-
institution byte-rates as univariate time series.

Subsetting: the full release covers ~275 K unique IPs but only ~30
institutions at the institution aggregation level. We default to all
institutions; pass `top_n_institutions=K` to keep only the K highest-
total-traffic institutions (per the plan.md risk row: "Cap to a subset
of institutions; document the subset").

Output: `data/cesnet_traffic.npz` with the same schema as Abilene:
    TM (T, num_links) Mbps == L; R (num_links, num_links) == I;
    links (num_links, 2) — synthetic ("inst_k", "inst_k") since there are
        no real link endpoints, just institution IDs;
    demands == links;
    nodes (num_links,) — institution IDs;
    T_train, T_val, T_test, train_end, val_end, time_step_minutes.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np


TIME_STEP_MINUTES = 10
DEFAULT_METRIC = "n_bytes"
# n_bytes is the total bytes per 10-min window. To convert to Mbps:
#   bytes / (10 * 60 s) * 8 / 1e6 = bytes * 8 / (600 * 1e6)
#                                = bytes * 1.333e-8 Mbps.
BYTES_TO_MBPS = 8.0 / (600.0 * 1e6)


def _natural_key(path: str):
    """Sort key so institution files order 0,1,2,...,10 not 0,1,10,2."""
    base = os.path.splitext(os.path.basename(path))[0]
    return (int(base), base) if base.isdigit() else (1 << 62, base)


def _list_institution_files(raw_dir: str) -> list[str]:
    """Return sorted per-institution data files (CSV or parquet) under raw_dir.

    The published Zenodo release (`institutions.tar.gz`) ships **CSV** files at
    `institutions/agg_10_minutes/<institution_id>.csv` (one column per metric,
    `id_time` first). Some mirrors repackage these as parquet. We accept either
    extension and either a flat layout (files directly in raw_dir) or one level
    of nesting (e.g. an `agg_10_minutes/` subdir), preferring the 10-minute
    aggregation when multiple are present.
    """
    # Prefer the 10-minute aggregation subdir if the extracted tree is present.
    for sub in ("agg_10_minutes", os.path.join("institutions", "agg_10_minutes")):
        for ext in ("csv", "parquet"):
            hit = sorted(
                glob.glob(os.path.join(raw_dir, sub, f"*.{ext}")), key=_natural_key
            )
            if hit:
                return hit
    # Flat, then one-level-nested, for either extension.
    for pattern in ("*.{ext}", os.path.join("*", "*.{ext}")):
        for ext in ("csv", "parquet"):
            hit = sorted(
                glob.glob(os.path.join(raw_dir, pattern.format(ext=ext))),
                key=_natural_key,
            )
            if hit:
                return hit
    raise FileNotFoundError(
        f"no .csv or .parquet institution files found under {raw_dir} "
        f"(checked agg_10_minutes/, flat, and one level deeper)"
    )


def _read_one_institution(path: str, metric: str) -> np.ndarray:
    """Read one institution file (CSV or parquet); return metric col as float32.

    Rows are ordered by 10-minute timestamp (`id_time`) by construction, so we
    pull only the metric column. CSV is the published format; parquet support
    is kept for repackaged mirrors.
    """
    if path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "pyarrow is required to read CESNET parquets; install with "
                "`pip install pyarrow`"
            ) from e
        table = pq.read_table(path, columns=[metric])
        arr = table.column(metric).to_numpy(zero_copy_only=False)
        return np.asarray(arr, dtype=np.float32)

    # CSV path — read just the metric column to keep memory low.
    import pandas as pd
    col = pd.read_csv(path, usecols=[metric])[metric].to_numpy()
    return np.asarray(col, dtype=np.float32)


def load_cesnet(
    raw_dir: str,
    *,
    metric: str = DEFAULT_METRIC,
    top_n_institutions: Optional[int] = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    length_tolerance: float = 0.05,
) -> dict:
    """Load CESNET-TimeSeries24 institution-level traces.

    Args:
        raw_dir: directory holding `<institution>.parquet` files (or one
            level of nesting).
        metric: metric column to extract. Default 'n_bytes' converts to Mbps.
            Pass 'n_flows', 'n_packets', etc. to forecast a different signal.
        top_n_institutions: if given, keep only the K institutions with the
            highest total `metric` sum. Documented in plan.md as the
            mitigation for "CESNET ingestion heavier than expected."
        train_frac, val_frac: split fractions.

    Returns the same dict shape as `load_abilene`. R = identity since CESNET
    has no link-level structure; institution traces are the "links."
    """
    file_paths = _list_institution_files(raw_dir)
    print(f"[cesnet] found {len(file_paths)} institution files under {raw_dir}")

    institution_ids: list[str] = []
    traces: list[np.ndarray] = []
    for p in file_paths:
        inst = os.path.splitext(os.path.basename(p))[0]
        try:
            col = _read_one_institution(p, metric)
        except Exception as e:
            print(f"[cesnet] WARN: skipping {p}: {e}")
            continue
        institution_ids.append(inst)
        traces.append(col)

    if not traces:
        raise RuntimeError(f"no usable institution files under {raw_dir}")

    # Drop empty / truncated institutions BEFORE aligning. The real release has
    # a handful of institutions with zero or near-zero rows (joined monitoring
    # late or were trimmed); the canonical length is ~40k (40 weeks @ 10 min).
    # Taking a naive global min collapses everything to the shortest stub, so we
    # filter to traces within `length_tolerance` of the modal (most common)
    # length, then trim survivors to their common min.
    lengths = np.array([len(tr) for tr in traces])
    modal_len = int(np.bincount(lengths).argmax())  # most common length
    min_keep = int(modal_len * (1.0 - length_tolerance))
    keep = lengths >= max(min_keep, 1)
    n_dropped = int((~keep).sum())
    if n_dropped:
        dropped = [institution_ids[i] for i in np.where(~keep)[0]]
        print(f"[cesnet] dropped {n_dropped} institutions shorter than "
              f"{min_keep} steps (modal length {modal_len}): "
              f"{dropped[:10]}{'...' if n_dropped > 10 else ''}")
    institution_ids = [iid for iid, k in zip(institution_ids, keep) if k]
    traces = [tr for tr, k in zip(traces, keep) if k]
    if not traces:
        raise RuntimeError(
            f"all institutions under {raw_dir} were shorter than {min_keep} "
            f"steps; check the data or lower length_tolerance"
        )

    # Length-align survivors to their common (shortest) length.
    T = min(len(tr) for tr in traces)
    if any(len(tr) != T for tr in traces):
        n_trim = sum(1 for tr in traces if len(tr) != T)
        print(f"[cesnet] {n_trim} kept institutions had off-by-a-few lengths; "
              f"trimming all to T={T}")
    L_full = np.stack([tr[:T] for tr in traces], axis=1).astype(np.float32)

    # Subset to top-N by total metric.
    if top_n_institutions is not None and top_n_institutions < L_full.shape[1]:
        totals = L_full.sum(axis=0)
        keep_idx = np.argsort(totals)[::-1][:top_n_institutions]
        keep_idx = np.sort(keep_idx)  # preserve a stable order
        L_full = L_full[:, keep_idx]
        institution_ids = [institution_ids[i] for i in keep_idx]
        print(f"[cesnet] subset to top {top_n_institutions} institutions by total "
              f"{metric}")

    # Convert to Mbps if we're loading bytes.
    if metric == "n_bytes":
        L = L_full * BYTES_TO_MBPS
    else:
        L = L_full
    L = L.astype(np.float32)

    num_links = L.shape[1]
    R = np.eye(num_links, dtype=np.float32)
    TM = L.copy()

    T_train = int(T * train_frac)
    T_val = int(T * val_frac)
    T_test = T - T_train - T_val

    # Synthesise endpoints / nodes for schema parity.
    nodes = np.array(institution_ids, dtype=str)
    links = np.array([[inst, inst] for inst in institution_ids], dtype=str)
    demands = links.copy()

    print(f"[cesnet] T={T} ({T // (24 * 60 // TIME_STEP_MINUTES)} days), "
          f"num_links={num_links}, "
          f"L range=[{L.min():.4f}, {L.max():.4f}] "
          f"{'Mbps' if metric == 'n_bytes' else metric}")
    print(f"[cesnet] split: train={T_train}, val={T_val}, test={T_test}")

    return {
        "TM": TM, "L": L, "R": R,
        "links": links, "demands": demands, "nodes": nodes,
        "T": T,
        "num_links": num_links, "num_od": num_links, "num_nodes": num_links,
        "T_train": T_train, "T_val": T_val, "T_test": T_test,
        "train_end": T_train, "val_end": T_train + T_val,
        "time_step_minutes": TIME_STEP_MINUTES,
        "metric": metric,
    }


def save_cesnet(out_path: str, data: dict) -> None:
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
        metric=data["metric"],
    )
    print(f"[cesnet] saved → {out_path}")


def main() -> None:
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    raw_dir = os.path.join(project_root, "data", "raw", "cesnet")
    out_path = os.path.join(project_root, "data", "cesnet_traffic.npz")
    data = load_cesnet(raw_dir, top_n_institutions=20)
    save_cesnet(out_path, data)


if __name__ == "__main__":
    main()
