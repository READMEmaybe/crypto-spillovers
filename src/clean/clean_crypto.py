"""Concatenate 84 monthly 1-min parquets into a single silver file per asset.

Keeps only the columns needed downstream for realized-volatility construction
(plan v4 schema contract: timestamp_utc, OHLC, volume). Dedupes on timestamp,
sorts, reports exchange-maintenance gaps, and enforces the Pandera schema on
write. Gaps are NOT forward-filled, silver preserves the true sampling grid;
the RV step decides how to treat them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.quality.schemas import CRYPTO_1MIN_CLEAN, check_monotonic

BRONZE_ROOT = Path("data/parquet/crypto")
SILVER_ROOT = Path("data/parquet/clean")
KEEP = ["timestamp_utc", "open", "high", "low", "close", "volume"]


def load_monthly_parquets(asset: str) -> pd.DataFrame:
    asset_dir = BRONZE_ROOT / f"{asset.lower()}usdt"
    files = sorted(asset_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No monthly parquets under {asset_dir}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"open_time_utc": "timestamp_utc"})
    return df[KEEP]


def clean(asset: str) -> pd.DataFrame:
    df = load_monthly_parquets(asset)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="first")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    check_monotonic(df, "timestamp_utc")
    report_gaps(df)
    return CRYPTO_1MIN_CLEAN.validate(df)


def report_gaps(df: pd.DataFrame) -> None:
    """Log contiguous gaps > 1 minute, Binance maintenance windows are real."""
    deltas = df["timestamp_utc"].diff().dropna()
    gaps = deltas[deltas > pd.Timedelta(minutes=1)]
    if len(gaps):
        total_missing = (gaps - pd.Timedelta(minutes=1)).sum()
        print(
            f"  gaps: {len(gaps)} windows, "
            f"{total_missing.total_seconds() / 60:.0f} missing minutes "
            f"(largest = {gaps.max()})"
        )
    else:
        print("  no gaps > 1 minute")


def main(asset: str) -> None:
    print(f"[clean_crypto] asset={asset}")
    df = clean(asset)
    SILVER_ROOT.mkdir(parents=True, exist_ok=True)
    out = SILVER_ROOT / f"{asset.lower()}_1min.parquet"
    df.to_parquet(out, index=False)
    span = f"{df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}"
    print(f"  wrote {out} ({len(df):,} rows, {span})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=["btc", "eth"], required=True)
    args = parser.parse_args()
    main(args.asset)
