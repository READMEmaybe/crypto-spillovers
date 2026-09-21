"""Assemble the country × date panel from silver + gold inputs.

Inputs (all local parquets):
  silver: equity.parquet, fx.parquet
  gold:   btc_rv_daily.parquet, eth_rv_daily.parquet
  bronze: controls.parquet, kaopen.parquet
  config: countries.yaml (country universe);
          crypto_ban_episodes.csv (time-varying crypto_ban, via src.features.crypto_ban)

Output:
  gold/panel.parquet with columns:
    date, country_id, equity_ret, fx_ret, btc_rv, eth_rv,
    chinn_ito, crypto_ban, err_float, vix, dxy, sp500, oil

Calendar convention:
  - Base grid = one row per (country_id, calendar date) spanning the intersection
    of silver coverage.
  - Equity and FX returns are NaN on non-trading days (no forward-fill).
  - BTC/ETH RV, controls, and KAOPEN are filled for every date (they're
    global / annual, not country-calendar).
  - KAOPEN is annual, forward-filled within country across days.
  - ERR (err_float) is annual per (country_id, year), exact merge, no ffill
    (config/err_float.csv covers all 31 countries × 2019–2025 with no gaps).
  - crypto_ban is derived daily from config/crypto_ban_episodes.csv via a
    date-range join (src.features.crypto_ban.derive); time-varying, int {0,1},
    no nulls. Countries with no episode are 0 throughout.

Note: `market_dev` (stock cap / GDP, World Bank) is listed in the plan-v4
schema but not yet fetched; we omit it here. Add it in a follow-up before
running the formal panel regression.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.features import crypto_ban
from src.features.returns import equity_log_returns, fx_log_returns
from src.quality.schemas import PANEL

BRONZE = Path("data/parquet")
SILVER = BRONZE / "clean"
GOLD = BRONZE / "gold"
CONFIG = Path("config/countries.yaml")
ERR_CSV = Path("config/err_float.csv")


def load_country_ids() -> list[str]:
    """Panel country universe (ISO3 ids) from countries.yaml."""
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    return [c["id"] for c in cfg["countries"]]


def load_kaopen() -> pd.DataFrame:
    """Chinn-Ito 2019–2023, one row per (iso3, year) → (country_id, year, chinn_ito)."""
    df = pd.read_parquet(BRONZE / "kaopen.parquet")
    df = df.rename(columns={"iso3": "country_id", "ka_open_normalized": "chinn_ito"})
    df = df[["country_id", "year", "chinn_ito"]].dropna()
    df["year"] = df["year"].astype(int)
    return df


def load_err() -> pd.DataFrame:
    """ERR float flag, one row per (country_id, year) → (country_id, year, err_float).

    Binary de-facto exchange-rate regime: 1 = floating/free-floating, 0 = managed/pegged.
    Source/curation: scripts/build_err_table.py (reads config/countries.yaml).
    """
    df = pd.read_csv(ERR_CSV)
    df = df[["country_id", "year", "err_float"]].copy()
    df["year"] = df["year"].astype(int)
    df["err_float"] = df["err_float"].astype(int)

    # Sentinel guard: -1 means "not yet determined" (placeholder TODO AREAER row).
    # It must NEVER silently become a real regressor, fail loudly until the user
    # fills the real 0/1 values from the IMF AREAER reports.
    unresolved = sorted(df.loc[df["err_float"] == -1, "country_id"].unique())
    if unresolved:
        raise ValueError(
            f"err_float has {len(unresolved)} unresolved -1 sentinels "
            f"(TODO AREAER): {unresolved}"
        )
    return df


def load_controls_wide() -> pd.DataFrame:
    """Long controls → wide: date, vix, dxy, sp500, oil."""
    df = pd.read_parquet(BRONZE / "controls.parquet")
    df["date"] = pd.to_datetime(df["date"])
    w = df.pivot(index="date", columns="series_id", values="value")
    w = w.rename(columns={"VIX": "vix", "DXY": "dxy", "SP500": "sp500", "BRENT": "oil"})
    return w.reset_index()[["date", "vix", "dxy", "sp500", "oil"]]


def load_rv(asset: str) -> pd.DataFrame:
    df = pd.read_parquet(GOLD / f"{asset}_rv_daily.parquet", columns=["date", "rv"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.rename(columns={"rv": f"{asset}_rv"})


def build_panel() -> pd.DataFrame:
    country_ids = load_country_ids()
    eq = pd.read_parquet(SILVER / "equity.parquet", columns=["date", "country_id", "close"])
    fx = pd.read_parquet(SILVER / "fx.parquet", columns=["date", "country_id", "fx_rate_vs_usd"])
    eq["date"] = pd.to_datetime(eq["date"])
    fx["date"] = pd.to_datetime(fx["date"])

    eq_ret = equity_log_returns(eq)
    fx_ret = fx_log_returns(fx)

    # Base grid: union of all (country, date) observed in either equity or fx,
    # filtered to panel countries. Holidays remain present on the grid of the
    # *other* series (equity holiday still gets FX row if FX observed, etc.).
    keys = pd.concat([
        eq_ret[["country_id", "date"]],
        fx_ret[["country_id", "date"]],
    ]).drop_duplicates().sort_values(["country_id", "date"]).reset_index(drop=True)
    keys = keys[keys["country_id"].isin(country_ids)]

    panel = (
        keys
        .merge(eq_ret, on=["country_id", "date"], how="left")
        .merge(fx_ret, on=["country_id", "date"], how="left")
    )

    # Crypto RVs (global) and controls, key on date only
    btc = load_rv("btc")
    eth = load_rv("eth")
    ctrl = load_controls_wide()
    panel = (
        panel
        .merge(btc, on="date", how="left")
        .merge(eth, on="date", how="left")
        .merge(ctrl, on="date", how="left")
    )

    # KAOPEN: annual → forward-fill to daily within country
    kao = load_kaopen()
    panel["year"] = panel["date"].dt.year
    panel = panel.merge(kao, on=["country_id", "year"], how="left")
    panel = panel.sort_values(["country_id", "date"])
    panel["chinn_ito"] = panel.groupby("country_id")["chinn_ito"].ffill()

    # ERR (err_float): annual binary float flag, exact (country_id, year) merge.
    # Coverage is complete (31 countries × 2019–2025), so no ffill and no NaN.
    err = load_err()
    panel = panel.merge(err, on=["country_id", "year"], how="left")
    assert panel["err_float"].notna().all(), (
        "err_float has gaps after merge, check config/err_float.csv coverage"
    )
    panel = panel.drop(columns=["year"])

    # Crypto ban flag: time-varying, derived daily from the episode table
    # (config/crypto_ban_episodes.csv) by date-range join under the default rule
    # {absolute, payment}. Countries with no episode -> 0; no nulls.
    panel["crypto_ban"] = crypto_ban.derive(panel[["country_id", "date"]])

    # Reorder to match schema
    panel = panel[[
        "date", "country_id",
        "equity_ret", "fx_ret",
        "btc_rv", "eth_rv",
        "chinn_ito", "crypto_ban", "err_float",
        "vix", "dxy", "sp500", "oil",
    ]].reset_index(drop=True)

    return PANEL.validate(panel)


def main() -> None:
    print("[build_panel]")
    panel = build_panel()
    GOLD.mkdir(parents=True, exist_ok=True)
    out = GOLD / "panel.parquet"
    panel.to_parquet(out, index=False)
    print(
        f"  wrote {out} ({len(panel):,} rows, "
        f"{panel.country_id.nunique()} countries, "
        f"{panel.date.min().date()} → {panel.date.max().date()})"
    )
    print("  non-null counts per column:")
    for col, n in panel.notna().sum().items():
        print(f"    {col:12s}  {n:>7,}")


if __name__ == "__main__":
    main()
