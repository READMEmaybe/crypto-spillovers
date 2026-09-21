"""
Fetch daily equity, FX, and control variables from Yahoo Finance and FRED.

Writes:
  data/parquet/equity.parquet: daily OHLCV for all equity indices
  data/parquet/fx.parquet: daily FX rates (local per USD) for all countries
  data/parquet/controls.parquet: VIX, SP500, Brent (Yahoo) + DXY (FRED)

Each file is a flat Parquet (all countries × all years in one file).
Re-running overwrites with a fresh fetch.

Usage:
    python scripts/fetch_markets.py
    python scripts/fetch_markets.py --skip-fred   # if no FRED_API_KEY
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fetchers.yahoo_fetcher import YahooFetcher
from fetchers.fred_fetcher import FredFetcher

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "ingestion_settings.yaml"
COUNTRIES_PATH = Path(__file__).parent.parent / "config" / "countries.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch market data → Parquet")
    parser.add_argument("--skip-fred", action="store_true", help="Skip FRED fetches (no API key needed)")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    with open(COUNTRIES_PATH) as f:
        countries_cfg = yaml.safe_load(f)

    out_root = Path(cfg["paths"]["parquet_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    countries = countries_cfg["countries"]
    yahoo = YahooFetcher.from_config(str(CONFIG_PATH))

    # --- Equity ---
    equity_frames = []
    for country in countries:
        eq = country.get("equity", {})
        if eq.get("source") == "yahoo" and eq.get("ticker"):
            logger.info("Equity: %s (%s)", country["id"], eq["ticker"])
            df = yahoo.fetch_equity(
                country_id=country["id"],
                ticker=eq["ticker"],
                currency=eq.get("currency", ""),
            )
            if not df.empty:
                equity_frames.append(df)

    if equity_frames:
        equity_df = pd.concat(equity_frames, ignore_index=True)
        equity_df.to_parquet(out_root / "equity.parquet", index=False)
        logger.info("Wrote equity.parquet: %d rows, %d countries",
                    len(equity_df), equity_df["country_id"].nunique())
    else:
        logger.warning("No equity data fetched")

    # --- FX ---
    fx_frames = []
    for country in countries:
        fx = country.get("fx", {})
        if fx.get("source") == "yahoo" and fx.get("ticker"):
            logger.info("FX: %s (%s)", country["id"], fx["ticker"])
            df = yahoo.fetch_fx(country_id=country["id"], ticker=fx["ticker"])
            if not df.empty:
                fx_frames.append(df)

    # --- Controls (Yahoo: VIX, SP500, Brent) ---
    logger.info("Controls: VIX, SP500, Brent")
    controls_df = yahoo.fetch_controls()
    controls_frames = [controls_df] if not controls_df.empty else []

    # --- FRED (DXY + FX fallbacks) ---
    if not args.skip_fred:
        try:
            fred = FredFetcher.from_config(str(CONFIG_PATH))

            logger.info("Controls: DXY (FRED)")
            dxy_df = fred.fetch_dxy()
            if not dxy_df.empty:
                controls_frames.append(dxy_df)

            logger.info("FX fallbacks (FRED)")
            fred_fx_df = fred.fetch_fx_fallbacks(countries)
            if not fred_fx_df.empty:
                fx_frames.append(fred_fx_df)

        except ValueError as e:
            logger.warning("FRED skipped: %s", e)

    # Write FX
    if fx_frames:
        fx_df = pd.concat(fx_frames, ignore_index=True)
        fx_df.to_parquet(out_root / "fx.parquet", index=False)
        logger.info("Wrote fx.parquet: %d rows, %d countries",
                    len(fx_df), fx_df["country_id"].nunique())
    else:
        logger.warning("No FX data fetched")

    # Write controls
    if controls_frames:
        ctrl_df = pd.concat(controls_frames, ignore_index=True)
        ctrl_df.to_parquet(out_root / "controls.parquet", index=False)
        logger.info("Wrote controls.parquet: %d rows, series: %s",
                    len(ctrl_df), sorted(ctrl_df["series_id"].unique()))
    else:
        logger.warning("No controls data fetched")


if __name__ == "__main__":
    main()
