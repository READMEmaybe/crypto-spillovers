"""Build daily realized-volatility gold files from 1-min silver.

For each calendar day (UTC), from 1-min log returns r_i:

    RV   = Σ r_i²                                    (realized variance)
    BV   = (π/2) Σ |r_i| * |r_{i-1}|                 (bipower variation)
    Jump = max(RV - BV, 0)

Also emits `n_obs` (minutes observed) and `coverage = n_obs / 1440`. RV for
days with coverage < 0.8 should be treated cautiously; we do NOT drop at this
stage (leaves the decision to the modeling step). 21 days (0.8%) fall below
full coverage, all are Binance maintenance windows.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.quality.schemas import RV_DAILY

SILVER_ROOT = Path("data/parquet/clean")
GOLD_ROOT = Path("data/parquet/gold")


def compute_rv_daily(df_1min: pd.DataFrame) -> pd.DataFrame:
    """1-min OHLC + timestamp_utc → per-UTC-day RV, BV, Jump, coverage."""
    df = df_1min[["timestamp_utc", "close"]].copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["date"] = df["timestamp_utc"].dt.tz_convert("UTC").dt.normalize()

    # First observation of each day has no same-day return, drop from RV
    # computation but still count it in n_obs (number of price samples).
    df["log_ret_sq"] = df["log_ret"] ** 2
    df["abs_ret"] = df["log_ret"].abs()
    df["abs_ret_lag"] = df.groupby("date")["abs_ret"].shift(1)
    df["bv_term"] = df["abs_ret"] * df["abs_ret_lag"]

    grp = df.groupby("date", sort=True)
    out = pd.DataFrame({
        "rv": grp["log_ret_sq"].sum(min_count=1),
        "bv": grp["bv_term"].sum(min_count=1) * (math.pi / 2),
        "n_obs": grp.size(),
    }).reset_index()
    out["jump"] = (out["rv"] - out["bv"]).clip(lower=0.0)
    out["coverage"] = out["n_obs"] / 1440.0
    out = out[["date", "rv", "bv", "jump", "n_obs", "coverage"]]
    # Drop the day with only one obs (can happen on day boundaries)
    out = out[out["n_obs"] >= 2].reset_index(drop=True)
    return RV_DAILY.validate(out)


def main(asset: str) -> None:
    print(f"[realized_vol] asset={asset}")
    silver = SILVER_ROOT / f"{asset.lower()}_1min.parquet"
    df = pd.read_parquet(silver, columns=["timestamp_utc", "close"])
    rv = compute_rv_daily(df)

    GOLD_ROOT.mkdir(parents=True, exist_ok=True)
    out = GOLD_ROOT / f"{asset.lower()}_rv_daily.parquet"
    rv.to_parquet(out, index=False)

    low_cov = (rv["coverage"] < 0.9).sum()
    print(
        f"  wrote {out} ({len(rv):,} days, "
        f"{low_cov} with coverage < 0.9, "
        f"max RV = {rv['rv'].max():.4f} on {rv.loc[rv['rv'].idxmax(), 'date'].date()})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=["btc", "eth"], required=True)
    args = parser.parse_args()
    main(args.asset)
