"""Tests for src/clean/clean_crypto.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.clean.clean_crypto import clean, load_monthly_parquets
from src.quality.schemas import CRYPTO_1MIN_CLEAN

BRONZE = Path("data/parquet/crypto")
HAS_BTC = (BRONZE / "btcusdt").exists() and any((BRONZE / "btcusdt").glob("*.parquet"))
HAS_ETH = (BRONZE / "ethusdt").exists() and any((BRONZE / "ethusdt").glob("*.parquet"))

needs_btc = pytest.mark.skipif(not HAS_BTC, reason="BTC bronze parquets not present")
needs_eth = pytest.mark.skipif(not HAS_ETH, reason="ETH bronze parquets not present")


@needs_btc
def test_btc_load_monthly_row_count_matches_sum():
    df = load_monthly_parquets("btc")
    # roughly 84 months × ~44k rows/month; just assert a non-trivial floor
    assert len(df) > 3_000_000
    assert set(df.columns) == {"timestamp_utc", "open", "high", "low", "close", "volume"}


@needs_btc
def test_btc_clean_is_monotonic_and_unique():
    df = clean("btc")
    assert df["timestamp_utc"].is_monotonic_increasing
    assert not df["timestamp_utc"].duplicated().any()


@needs_btc
def test_btc_clean_schema_validates():
    df = clean("btc")
    CRYPTO_1MIN_CLEAN.validate(df)


@needs_btc
def test_btc_clean_period_coverage():
    df = clean("btc")
    assert df["timestamp_utc"].min() <= pd.Timestamp("2019-01-01 00:05", tz="UTC")
    assert df["timestamp_utc"].max() >= pd.Timestamp("2025-12-30", tz="UTC")


@needs_eth
def test_eth_clean_smoke():
    df = clean("eth")
    assert df["timestamp_utc"].is_monotonic_increasing
    assert len(df) > 3_000_000
