"""Integrity and schema tests for the loader."""

from __future__ import annotations

import hashlib

import pytest

from bikedemand import data


def test_sha256_matches_hashlib(tmp_path):
    target = tmp_path / "x.bin"
    payload = b"bikes" * 500
    target.write_bytes(payload)
    assert data.sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_verify_accepts_and_rejects(tmp_path):
    target = tmp_path / "x.txt"
    target.write_text("ok", encoding="utf-8")
    digest = data.sha256_file(target)
    assert data.verify(target, digest) == target
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.verify(target, "f" * 64)


def test_describe_reports_period_and_totals(hourly_frame):
    summary = data.describe(hourly_frame)
    assert summary["rows"] == len(hourly_frame)
    assert summary["total_rentals"] == int(hourly_frame["cnt"].sum())
    assert summary["csv_sha256"] == data.CSV_SHA256


def test_load_raw_rejects_unexpected_schema(tmp_path, monkeypatch):
    path = tmp_path / "hour.csv"
    path.write_text("instant,dteday\n1,2011-01-01\n", encoding="utf-8")
    monkeypatch.setattr(data, "download", lambda force=False: path)
    with pytest.raises(ValueError, match="unexpected schema"):
        data.load_raw()
