"""Unit tests for scripts/download_cesnet.py.

Stubs out `urllib.request.urlopen` so we exercise the real parsing and MD5
verification paths without hitting Zenodo. No network, no real download.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)

import download_cesnet as dl  # noqa: E402


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n == -1 or n is None:
            chunk = self._body[self._pos:]
            self._pos = len(self._body)
            return chunk
        chunk = self._body[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_opener(by_url: dict[str, bytes]):
    """Build a fake `urlopen(url)` that returns canned bytes per URL."""
    def _open(url):
        if url not in by_url:
            raise AssertionError(f"unexpected URL in test: {url}")
        return _FakeResponse(by_url[url])
    return _open


def _record(files: list[dict], title: str = "CESNET-TimeSeries24") -> bytes:
    return json.dumps({"metadata": {"title": title}, "files": files}).encode()


def _file_meta(key: str, body: bytes, *, link: str) -> dict:
    md5 = hashlib.md5(body).hexdigest()
    return {
        "key": key,
        "size": len(body),
        "checksum": f"md5:{md5}",
        "links": {"self": link},
    }


def test_fetch_record_parses_json():
    fake = _make_opener({
        "https://zenodo.org/api/records/13382427": _record(
            [_file_meta("a.parquet", b"hello", link="x")]
        ),
    })
    record = dl.fetch_record("13382427", url_opener=fake)
    assert record["metadata"]["title"] == "CESNET-TimeSeries24"
    assert len(record["files"]) == 1


def test_list_files_filters_by_pattern():
    record = json.loads(_record([
        _file_meta("agg_10_minutes/a.parquet", b"x", link="u1"),
        _file_meta("agg_1_hour/b.parquet", b"x", link="u2"),
        _file_meta("README.md", b"x", link="u3"),
    ]).decode())
    out = dl.list_files(record, include_pattern=r"agg_10_minutes.*\.parquet$")
    assert len(out) == 1
    assert out[0]["key"] == "agg_10_minutes/a.parquet"


def test_list_files_no_filter_returns_all():
    record = json.loads(_record([
        _file_meta("a.parquet", b"x", link="u1"),
        _file_meta("b.parquet", b"x", link="u2"),
    ]).decode())
    out = dl.list_files(record, include_pattern=None)
    assert len(out) == 2


def test_list_files_strips_md5_prefix():
    record = json.loads(_record([
        _file_meta("a.parquet", b"hello", link="u1"),
    ]).decode())
    out = dl.list_files(record, include_pattern=None)
    assert out[0]["md5"] == hashlib.md5(b"hello").hexdigest()
    assert ":" not in out[0]["md5"]


def test_download_one_writes_and_verifies(tmp_path):
    body = b"\x00" * 4096 + b"abc"
    md5 = hashlib.md5(body).hexdigest()
    fake = _make_opener({"http://example/a.parquet": body})
    meta = {
        "key": "a.parquet",
        "link": "http://example/a.parquet",
        "size": len(body),
        "md5": md5,
    }
    path = dl.download_one(meta, str(tmp_path), url_opener=fake)
    assert os.path.exists(path)
    assert dl.md5_of_path(path) == md5


def test_download_one_skips_existing_with_matching_md5(tmp_path, monkeypatch):
    body = b"already-here"
    md5 = hashlib.md5(body).hexdigest()
    out = tmp_path / "a.parquet"
    out.write_bytes(body)

    calls = []

    def _should_not_be_called(url):  # pragma: no cover - test guard
        calls.append(url)
        raise AssertionError("urlopen should not be called for cached file")

    meta = {
        "key": "a.parquet",
        "link": "http://example/a.parquet",
        "size": len(body),
        "md5": md5,
    }
    path = dl.download_one(meta, str(tmp_path), url_opener=_should_not_be_called)
    assert path == str(out)
    assert calls == []


def test_download_one_md5_mismatch_raises_and_removes(tmp_path):
    body = b"abc"
    fake = _make_opener({"http://example/a.parquet": body})
    meta = {
        "key": "a.parquet",
        "link": "http://example/a.parquet",
        "size": 3,
        "md5": "deadbeef" * 4,  # wrong
    }
    with pytest.raises(RuntimeError, match="md5 mismatch"):
        dl.download_one(meta, str(tmp_path), url_opener=fake)
    # Failed download should be cleaned up.
    assert not os.path.exists(os.path.join(tmp_path, "a.parquet"))


def test_download_record_dry_run_lists_without_writing(tmp_path):
    body = b"x" * 100
    record = _record([_file_meta("a.parquet", body, link="http://x/a")])
    fake = _make_opener({"https://zenodo.org/api/records/13382427": record})

    paths = dl.download_record(
        "13382427",
        str(tmp_path),
        include_pattern=None,
        dry_run=True,
        url_opener=fake,
    )
    assert paths == [str(tmp_path / "a.parquet")]
    # Nothing actually written.
    assert not (tmp_path / "a.parquet").exists()


def test_download_record_no_matches_raises(tmp_path):
    record = _record([_file_meta("README.md", b"x", link="http://x/r")])
    fake = _make_opener({"https://zenodo.org/api/records/13382427": record})
    with pytest.raises(RuntimeError, match="no files in record"):
        dl.download_record(
            "13382427",
            str(tmp_path),
            include_pattern=r"\.parquet$",
            dry_run=True,
            url_opener=fake,
        )
