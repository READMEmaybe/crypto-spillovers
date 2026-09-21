"""Merge Yahoo + Investing bronze into clean/equity.parquet and clean/fx.parquet.

Silver contract (plan v4):
  equity: date, country_id, ticker, open, high, low, close, volume, currency, source
  fx:     date, country_id, ticker, fx_rate_vs_usd, source

Rules:
  - Long format, one row per (country_id, date).
  - Holidays / non-trading days → absent (NO forward-fill).
  - Investing `price` → `close`; `volume_raw` ('1.81M') → float.
  - Drop rows with close ≤ 0.
  - Currencies are backfilled from config/countries.yaml for Investing rows
    (the bronze investing loader left `currency` null).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.quality.schemas import EQUITY_CLEAN, FX_CLEAN

# A daily log-return of ±0.5 (~65% down / ~65% up) is the outer limit of any
# real FX/equity move in the 31-country panel. Observed exceedances are Yahoo
# decimal-shift glitches (e.g. GHA FX 2020-03-31) that revert next day.
_MAX_ABS_LOG_RET = 0.5

BRONZE_ROOT = Path("data/parquet")
SILVER_ROOT = Path("data/parquet/clean")
CONFIG = Path("config/countries.yaml")

_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def drop_reverting_glitches(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Drop rows where a single-day log return exceeds ±`_MAX_ABS_LOG_RET` AND
    reverses within the next two days (hallmark of a Yahoo decimal-shift glitch).
    Genuine crashes don't revert on the next trading day.
    """
    out = []
    for cid, grp in df.groupby("country_id", sort=False):
        g = grp.sort_values("date").reset_index(drop=True)
        log_ret = np.log(g[price_col] / g[price_col].shift(1))
        reverse_2d = np.log(g[price_col].shift(-2) / g[price_col])
        glitch = (log_ret.abs() > _MAX_ABS_LOG_RET) & \
                 (np.sign(log_ret) == -np.sign(reverse_2d)) & \
                 (reverse_2d.abs() > _MAX_ABS_LOG_RET)
        if glitch.any():
            dropped = g.loc[glitch, ["date", price_col]]
            for _, r in dropped.iterrows():
                print(f"  [glitch] dropped {cid} {r['date'].date()} "
                      f"{price_col}={r[price_col]:g}")
        out.append(g[~glitch])
    return pd.concat(out, ignore_index=True)


def parse_volume(raw: str | float | None) -> float | None:
    """Parse Investing-style volume string ('1.81M', '250K', '-', '') → float."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if s in {"", "-"}:
        return None
    m = re.fullmatch(r"([\d.,]+)\s*([KMBT]?)", s)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    mult = _SUFFIX.get(m.group(2), 1.0)
    return num * mult


def load_country_currencies() -> dict[str, str]:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    return {c["id"]: c.get("equity", {}).get("currency") for c in cfg["countries"]}


# ---------------------------------------------------------------------------
# Equity
# ---------------------------------------------------------------------------

def _yahoo_equity() -> pd.DataFrame:
    df = pd.read_parquet(BRONZE_ROOT / "equity.parquet")
    df["source"] = "yahoo"
    return df[
        ["date", "country_id", "ticker", "open", "high", "low",
         "close", "volume", "currency", "source"]
    ]


def _investing_equity(currencies: dict[str, str]) -> pd.DataFrame:
    df = pd.read_parquet(BRONZE_ROOT / "investing_manual.parquet")
    df = df[df["data_type"] == "equity"].copy()
    df["close"] = df["price"]
    df["volume"] = df["volume_raw"].map(parse_volume)
    df["ticker"] = df["filename"].str.replace("_equity.csv", "", regex=False)
    df["currency"] = df["country_id"].map(currencies)
    df["source"] = "investing"
    return df[
        ["date", "country_id", "ticker", "open", "high", "low",
         "close", "volume", "currency", "source"]
    ]


def clean_equity() -> pd.DataFrame:
    currencies = load_country_currencies()
    df = pd.concat([_yahoo_equity(), _investing_equity(currencies)], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df.drop_duplicates(subset=["country_id", "date"], keep="first")
    df = df.sort_values(["country_id", "date"]).reset_index(drop=True)
    df = drop_reverting_glitches(df, "close")
    return EQUITY_CLEAN.validate(df)


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------

def _yahoo_fx() -> pd.DataFrame:
    df = pd.read_parquet(BRONZE_ROOT / "fx.parquet")
    # Yahoo XXXUSD=X returns USD-per-local (EURUSD=X≈1.18, JPYUSD=X≈0.0064).
    # Invert to local-per-USD so this column means what its name says and
    # matches the investing_csv source (which is already local-per-USD).
    df["fx_rate_vs_usd"] = 1.0 / df["fx_rate_vs_usd"]
    df["source"] = "yahoo"
    return df[["date", "country_id", "ticker", "fx_rate_vs_usd", "source"]]


def _investing_fx() -> pd.DataFrame:
    df = pd.read_parquet(BRONZE_ROOT / "investing_manual.parquet")
    df = df[df["data_type"] == "fx"].copy()
    df["fx_rate_vs_usd"] = df["price"]
    df["ticker"] = df["filename"].str.replace("_fx.csv", "", regex=False)
    df["source"] = "investing"
    return df[["date", "country_id", "ticker", "fx_rate_vs_usd", "source"]]


def clean_fx() -> pd.DataFrame:
    df = pd.concat([_yahoo_fx(), _investing_fx()], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["fx_rate_vs_usd"].notna() & (df["fx_rate_vs_usd"] > 0)]
    df = df.drop_duplicates(subset=["country_id", "date"], keep="first")
    df = df.sort_values(["country_id", "date"]).reset_index(drop=True)
    df = drop_reverting_glitches(df, "fx_rate_vs_usd")
    return FX_CLEAN.validate(df)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    SILVER_ROOT.mkdir(parents=True, exist_ok=True)

    print("[clean_markets] equity")
    eq = clean_equity()
    out = SILVER_ROOT / "equity.parquet"
    eq.to_parquet(out, index=False)
    print(f"  wrote {out} ({len(eq):,} rows, {eq.country_id.nunique()} countries)")

    print("[clean_markets] fx")
    fx = clean_fx()
    out = SILVER_ROOT / "fx.parquet"
    fx.to_parquet(out, index=False)
    print(f"  wrote {out} ({len(fx):,} rows, {fx.country_id.nunique()} countries)")


if __name__ == "__main__":
    main()
