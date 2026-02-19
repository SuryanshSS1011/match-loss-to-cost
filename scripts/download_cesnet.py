#!/usr/bin/env python
"""Download the CESNET-TimeSeries24 dataset from Zenodo.

Resolves the record JSON via the Zenodo REST API, optionally filters file
names (default: parquet files under the 10-minute aggregation), downloads
each, and validates the MD5 checksum.

Run on the cloud box:
    python scripts/download_cesnet.py --out data/raw/cesnet \
        --include 'agg_10_minutes.*\\.parquet$'

Run dry on the laptop to inspect what would download without pulling bytes:
    python scripts/download_cesnet.py --dry-run --out /tmp/cesnet

The default record ID 13382427 corresponds to DOI 10.5281/zenodo.13382427
(Koumar et al., Sci. Data 12:338, 2025). Pass --record-id to point at a
different version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Iterable, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


ZENODO_API = "https://zenodo.org/api/records"
DEFAULT_RECORD_ID = "13382427"
DEFAULT_INCLUDE = r"agg_10_minutes.*\.parquet$"
CHUNK = 1 << 20  # 1 MiB streaming chunks


def fetch_record(record_id: str,
                 url_opener=urllib.request.urlopen) -> dict:
    """Return the parsed Zenodo record JSON for `record_id`.

    `url_opener` is dependency-injected so tests can stub it.
    """
    url = f"{ZENODO_API}/{record_id}"
    with url_opener(url) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def list_files(record: dict,
               include_pattern: Optional[str] = DEFAULT_INCLUDE
               ) -> list[dict]:
    """Return the subset of record['files'] matching `include_pattern`.

    Each returned dict has keys `key, link, size, md5` extracted from the
    Zenodo schema. `include_pattern` is a regex; pass None for all files.
    """
    files = record.get("files") or []
    if include_pattern is not None:
        regex = re.compile(include_pattern)
    else:
        regex = None

    out = []
    for f in files:
        key = f.get("key") or f.get("filename")
        if key is None:
            continue
        if regex is not None and not regex.search(key):
            continue
        # Zenodo exposes the download URL at links.self in v1 responses
        # and links.download in some legacy ones.
        link = (
            (f.get("links") or {}).get("self")
            or (f.get("links") or {}).get("download")
        )
        # Checksum format is "md5:<hex>"; older records may have just the hex.
        chksum = f.get("checksum") or ""
        if chksum.startswith("md5:"):
            md5 = chksum[len("md5:"):]
        else:
            md5 = chksum
        out.append({
            "key": key,
            "link": link,
            "size": int(f.get("size", 0)),
            "md5": md5,
        })
    return out


def md5_of_path(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(file_meta: dict, out_dir: str,
                 url_opener=urllib.request.urlopen,
                 verify_md5: bool = True) -> str:
    """Download one file, write to `out_dir/key`, optionally verify MD5.

    Returns the local path. Skips re-download if the file already exists
    AND its MD5 matches the manifest.
    """
    if not file_meta.get("link"):
        raise ValueError(f"file {file_meta.get('key')} has no download link")
    out_path = os.path.join(out_dir, file_meta["key"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if os.path.exists(out_path) and verify_md5 and file_meta.get("md5"):
        if md5_of_path(out_path) == file_meta["md5"]:
            print(f"[cesnet]   skip {file_meta['key']} (already present)")
            return out_path

    print(f"[cesnet] GET {file_meta['key']} "
          f"({file_meta.get('size', 0) / 1e6:.1f} MB)")
    with url_opener(file_meta["link"]) as resp, open(out_path, "wb") as f:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)

    if verify_md5 and file_meta.get("md5"):
        got = md5_of_path(out_path)
        if got != file_meta["md5"]:
            os.remove(out_path)
            raise RuntimeError(
                f"md5 mismatch for {file_meta['key']}: "
                f"expected {file_meta['md5']}, got {got}"
            )
        print(f"[cesnet]   md5 ok ({got[:8]}...)")
    return out_path


def download_record(
    record_id: str,
    out_dir: str,
    *,
    include_pattern: Optional[str] = DEFAULT_INCLUDE,
    dry_run: bool = False,
    verify_md5: bool = True,
    url_opener=urllib.request.urlopen,
) -> list[str]:
    """End-to-end: resolve record, list files, download each."""
    record = fetch_record(record_id, url_opener=url_opener)
    print(f"[cesnet] record {record_id}: "
          f"{record.get('metadata', {}).get('title', '<no title>')}")
    files = list_files(record, include_pattern=include_pattern)
    if not files:
        raise RuntimeError(
            f"no files in record {record_id} match pattern "
            f"{include_pattern!r}"
        )
    total_mb = sum(f["size"] for f in files) / 1e6
    print(f"[cesnet] {len(files)} files matching {include_pattern!r} "
          f"({total_mb:.1f} MB total)")

    if dry_run:
        for f in files:
            print(f"[dry-run]   would download {f['key']} "
                  f"({f['size'] / 1e6:.1f} MB) → {out_dir}/{f['key']}")
        return [os.path.join(out_dir, f["key"]) for f in files]

    paths = []
    for f in files:
        paths.append(download_one(f, out_dir, url_opener=url_opener,
                                  verify_md5=verify_md5))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="CESNET Zenodo downloader")
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID)
    parser.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "data", "raw", "cesnet"))
    parser.add_argument("--include", default=DEFAULT_INCLUDE,
                        help="regex pattern for file keys to download "
                             "(empty string = all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip MD5 verification")
    args = parser.parse_args()

    pattern = args.include or None
    download_record(
        args.record_id, args.out,
        include_pattern=pattern,
        dry_run=args.dry_run,
        verify_md5=not args.no_verify,
    )


if __name__ == "__main__":
    main()
