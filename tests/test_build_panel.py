"""Tests for src/features/build_panel.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features import crypto_ban
from src.quality.schemas import PANEL

GOLD = Path("data/parquet/gold")
HAS_PANEL = (GOLD / "panel.parquet").exists()
needs_panel = pytest.mark.skipif(not HAS_PANEL, reason="panel.parquet not built")


@needs_panel
def test_panel_countries_and_size():
    p = pd.read_parquet(GOLD / "panel.parquet")
    # Expanded sample (2026-06): 89 countries (~90-country universe minus VEN,
    # dropped for hyperinflation), ~166k rows. Tolerate ±5% on row count.
    assert p["country_id"].nunique() == 89
    assert 158_000 <= len(p) <= 175_000


@needs_panel
def test_panel_schema_validates():
    p = pd.read_parquet(GOLD / "panel.parquet")
    PANEL.validate(p)


@needs_panel
def test_panel_unique_country_date():
    p = pd.read_parquet(GOLD / "panel.parquet")
    assert not p.duplicated(subset=["country_id", "date"]).any()


@needs_panel
def test_panel_chinn_ito_in_unit_interval():
    p = pd.read_parquet(GOLD / "panel.parquet")
    ci = p["chinn_ito"].dropna()
    assert ((ci >= 0) & (ci <= 1)).all()


# The crypto_ban column is now *derived* (time-varying) from the episode table by
# build_panel via crypto_ban.derive(panel_index). These tests assert the derived
# binary over the real panel's (country_id, date) index, the exact integration
# path, independent of any (possibly stale) stored crypto_ban column.
def _derived_crypto_ban() -> pd.DataFrame:
    p = pd.read_parquet(GOLD / "panel.parquet", columns=["country_id", "date"])
    p["date"] = pd.to_datetime(p["date"])
    p["crypto_ban"] = crypto_ban.derive(p[["country_id", "date"]])
    return p


@needs_panel
def test_panel_crypto_ban_is_binary_no_nulls():
    d = _derived_crypto_ban()
    assert set(d["crypto_ban"].unique()) <= {0, 1}
    assert d["crypto_ban"].isna().sum() == 0
    assert d["crypto_ban"].dtype.kind == "i"


@needs_panel
def test_panel_crypto_ban_derived_transitions_and_stable():
    d = _derived_crypto_ban()

    # CHN: comprehensive ban transition on 2021-09-24 (0 the day before, 1 on/after).
    chn = d[d["country_id"] == "CHN"].set_index("date")["crypto_ban"]
    if pd.Timestamp("2021-09-23") in chn.index:
        assert chn.loc["2021-09-23"] == 0
    assert chn.loc[chn.index >= "2021-09-24"].eq(1).all()
    assert chn.loc[chn.index < "2021-09-24"].eq(0).all()

    # MAR: absolute ban predates 2019 and is open-ended -> 1 throughout.
    mar = d[d["country_id"] == "MAR"]["crypto_ban"]
    assert mar.eq(1).all()

    # A legal country (no episode) -> 0 throughout. BRA is in the panel and has
    # no crypto-ban episode.
    bra = d[d["country_id"] == "BRA"]["crypto_ban"]
    assert len(bra) > 0
    assert bra.eq(0).all()


@needs_panel
def test_panel_btc_rv_populated_every_date():
    p = pd.read_parquet(GOLD / "panel.parquet")
    # BTC trades 24/7 → btc_rv should be present for every date in the panel
    assert p["btc_rv"].isna().sum() == 0


@needs_panel
def test_panel_err_float_binary_and_complete():
    p = pd.read_parquet(GOLD / "panel.parquet")
    # Binary, no gaps, config/err_float.csv covers all 89 countries × 2019–2025.
    assert set(p["err_float"].unique()) <= {0, 1}
    assert p["err_float"].isna().sum() == 0


@needs_panel
def test_panel_err_float_modal_split():
    p = pd.read_parquet(GOLD / "panel.parquet")
    # Modal (majority-over-years) regime per country → 48 float / 41 non-float
    # for the expanded 89-country panel (see scripts/build_err_table.py).
    modal = p.groupby("country_id")["err_float"].mean()
    assert int((modal > 0.5).sum()) == 48
    assert int((modal <= 0.5).sum()) == 41


@needs_panel
def test_panel_err_float_switch_year_egypt():
    p = pd.read_parquet(GOLD / "panel.parquet")
    # EGY floated in Mar-2024: err_float 0 through 2023, 1 from 2024 on.
    egy = p[p["country_id"] == "EGY"].assign(y=lambda d: d["date"].dt.year)
    assert (egy.loc[egy["y"] <= 2023, "err_float"] == 0).all()
    assert (egy.loc[egy["y"] >= 2024, "err_float"] == 1).all()


@needs_panel
def test_panel_algeria_present_with_fx():
    p = pd.read_parquet(GOLD / "panel.parquet")
    dza = p[p["country_id"] == "DZA"]
    # Algeria is in the panel with FX returns present on most days. (DZA gained an
    # equity source in the 2026-06 expansion; verify the FX channel regardless.)
    assert len(dza) > 0
    assert dza["fx_ret"].notna().sum() > 1000
