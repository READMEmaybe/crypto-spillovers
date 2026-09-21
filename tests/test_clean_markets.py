"""Tests for src/clean/clean_markets.py.

Unit tests for volume parsing + small fixtures. Integration smoke tests
on the real bronze parquets (skipped if they aren't present).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.clean.clean_markets import clean_equity, clean_fx, parse_volume

BRONZE = Path("data/parquet")
HAS_BRONZE = (BRONZE / "equity.parquet").exists() and (BRONZE / "fx.parquet").exists()
needs_bronze = pytest.mark.skipif(not HAS_BRONZE, reason="bronze parquets not present")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.81M", 1_810_000.0),
        ("250K", 250_000.0),
        ("2.3B", 2_300_000_000.0),
        ("1,234", 1234.0),
        ("-", None),
        ("", None),
        (None, None),
        (float("nan"), None),
        ("garbage", None),
    ],
)
def test_parse_volume(raw, expected):
    assert parse_volume(raw) == expected


@needs_bronze
def test_clean_equity_country_count():
    df = clean_equity()
    # Expanded sample (2026-06): 92 unique country_ids with equity coverage in
    # silver (superset of the 89-country panel universe).
    assert df["country_id"].nunique() == 92


@needs_bronze
def test_clean_equity_no_duplicate_rows():
    df = clean_equity()
    assert not df.duplicated(subset=["country_id", "date"]).any()


@needs_bronze
def test_clean_equity_sources_valid():
    df = clean_equity()
    assert set(df["source"].unique()) <= {"yahoo", "investing"}


@needs_bronze
def test_clean_equity_investing_volume_is_numeric():
    df = clean_equity()
    inv = df[df["source"] == "investing"]
    # any non-null volume must be float-typed and non-negative
    vols = inv["volume"].dropna()
    assert (vols >= 0).all()
    assert pd.api.types.is_float_dtype(vols)


@needs_bronze
def test_clean_fx_country_count():
    df = clean_fx()
    # Expanded sample (2026-06): 86 unique country_ids with FX coverage in silver.
    assert df["country_id"].nunique() == 86


@needs_bronze
def test_clean_fx_no_duplicate_rows():
    df = clean_fx()
    assert not df.duplicated(subset=["country_id", "date"]).any()


@needs_bronze
def test_clean_fx_includes_investing_fx():
    # FX with no Yahoo/FRED coverage comes through the investing_csv path:
    # DZA/TUN (managed floats) + KGZ/MNG (not in Yahoo/FRED) + BIH (EUR-peg derived).
    df = clean_fx()
    inv = df[df["source"] == "investing"]
    assert set(inv["country_id"].unique()) == {"BIH", "DZA", "KGZ", "MNG", "TUN"}
