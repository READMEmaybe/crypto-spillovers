"""
Fetch 1-minute OHLCV klines for BTC and ETH from Binance.

Writes monthly Parquet files to data/parquet/crypto/{symbol}/{YYYY}-{MM}.parquet.
Idempotent: skips months already on disk.

Usage:
    python scripts/fetch_crypto.py
    python scripts/fetch_crypto.py --start 2024-01-01 --end 2024-03-31
    python scripts/fetch_crypto.py --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fetchers.binance_fetcher import BinanceFetcher

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "ingestion_settings.yaml"
COUNTRIES_PATH = Path(__file__).parent.parent / "config" / "countries.yaml"


def _crypto_symbols(countries_cfg: dict) -> list[str]:
    """Return configured crypto symbols (e.g. ['BTCUSDT', 'ETHUSDT'])."""
    return [c["symbol"].upper() for c in countries_cfg.get("crypto", [])]


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    with open(COUNTRIES_PATH) as f:
        countries_cfg = yaml.safe_load(f)

    output_root = Path(cfg["paths"]["parquet_root"]) / "crypto"

    parser = argparse.ArgumentParser(description="Fetch Binance crypto OHLCV → Parquet")
    parser.add_argument("--symbol", help="Single symbol to fetch (e.g. BTCUSDT). Fetches all by default.")
    parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--refetch", action="store_true", help="Ignore existing files and refetch everything")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else _crypto_symbols(countries_cfg)
    if not symbols:
        logger.error("No crypto symbols found in config/countries.yaml")
        sys.exit(1)

    for symbol in symbols:
        fetcher = BinanceFetcher.from_config(
            symbol=symbol,
            output_path=str(output_root),
            config_path=str(CONFIG_PATH),
        )
        if args.start:
            fetcher.start_ms = fetcher._date_to_ms(args.start)
        if args.end:
            fetcher.end_ms = fetcher._date_to_ms(args.end, end_of_day=True)

        logger.info("Fetching %s → %s", symbol, output_root / symbol.lower())
        fetcher.fetch_all(force=args.refetch)


if __name__ == "__main__":
    main()
