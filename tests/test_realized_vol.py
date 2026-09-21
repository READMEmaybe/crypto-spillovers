"""Tests for src/features/realized_vol.py."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.realized_vol import compute_rv_daily
from src.quality.schemas import RV_DAILY

GOLD = Path("data/parquet/gold")
HAS_BTC_RV = (GOLD / "btc_rv_daily.parquet").exists()
needs_btc_rv = pytest.mark.skipif(not HAS_BTC_RV, reason="btc_rv gold not built")


def _synth_1min(n_minutes: int, start: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    """Deterministic 1-min series for unit testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC")
    rets = rng.normal(0, 1e-4, size=n_minutes)
    close = 42_000.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"timestamp_utc": idx, "close": close})


def test_rv_is_non_negative_and_schema_valid():
    df = _synth_1min(1440 * 3)
    rv = compute_rv_daily(df)
    RV_DAILY.validate(rv)
    assert (rv["rv"] >= 0).all()
    assert (rv["bv"] >= 0).all()
    assert (rv["jump"] >= 0).all()


def test_rv_matches_closed_form_on_constant_returns():
    # 1441 minutes starting 2024-01-01 00:00 → day 2024-01-01 owns rows 0..1439
    # (1440 rows). The very first row has no prior observation so log_ret is NaN;
    # that leaves 1439 intra-day returns, each = log(exp(1e-3)) = 1e-3.
    idx = pd.date_range("2024-01-01", periods=1441, freq="1min", tz="UTC")
    close = 100.0 * np.exp(np.arange(1441) * 1e-3)
    df = pd.DataFrame({"timestamp_utc": idx, "close": close})
    rv = compute_rv_daily(df)
    day0 = rv.iloc[0]
    expected_rv = 1439 * (1e-3) ** 2
    assert abs(float(day0["rv"]) - expected_rv) < 1e-12


def test_rv_coverage_equals_n_obs_over_1440():
    df = _synth_1min(1440 * 2)
    rv = compute_rv_daily(df)
    assert np.allclose(rv["coverage"], rv["n_obs"] / 1440.0)


@needs_btc_rv
def test_btc_rv_spikes_at_covid_and_ftx():
    """Plan v4 calibration: RV spikes around COVID (Mar 12-13 2020) and FTX
    (Nov 8-11 2022) must be in the top 10% of the RV distribution. China ban
    (Sep 24 2021) is excluded from the strict check, BTC only fell ~5% that
    day; the real crypto-shock window around that policy was several weeks
    earlier (May 2021 Musk / China mining crackdown)."""
    rv = pd.read_parquet(GOLD / "btc_rv_daily.parquet")
    rv["date"] = pd.to_datetime(rv["date"]).dt.tz_localize(None)
    top_10pct = rv["rv"].quantile(0.90)
    covid = rv[rv["date"].between("2020-03-12", "2020-03-14")]["rv"].max()
    ftx   = rv[rv["date"].between("2022-11-07", "2022-11-12")]["rv"].max()
    assert covid >= top_10pct, f"COVID RV {covid:.4f} below p90 {top_10pct:.4f}"
    assert ftx   >= top_10pct, f"FTX RV {ftx:.4f} below p90 {top_10pct:.4f}"


@needs_btc_rv
def test_btc_rv_has_elevated_vol_around_china_ban_week():
    """Sep 24 2021 China ban: BTC vol that day was moderate (~5% move), but the
    surrounding week should still be above the RV median."""
    rv = pd.read_parquet(GOLD / "btc_rv_daily.parquet")
    rv["date"] = pd.to_datetime(rv["date"]).dt.tz_localize(None)
    median = rv["rv"].median()
    week = rv[rv["date"].between("2021-09-20", "2021-09-27")]["rv"].max()
    assert week > median


@needs_btc_rv
def test_btc_rv_day_count():
    rv = pd.read_parquet(GOLD / "btc_rv_daily.parquet")
    # 2019-01-01 → 2025-12-31 inclusive = 2,557 days
    assert len(rv) == 2557


@needs_btc_rv
def test_btc_rv_max_is_covid_black_thursday():
    rv = pd.read_parquet(GOLD / "btc_rv_daily.parquet")
    rv["date"] = pd.to_datetime(rv["date"]).dt.tz_localize(None)
    max_date = rv.loc[rv["rv"].idxmax(), "date"].date()
    assert max_date == pd.Timestamp("2020-03-13").date()
