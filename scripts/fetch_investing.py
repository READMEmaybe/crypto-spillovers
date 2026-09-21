"""
Parse manually downloaded Investing.com CSV files → Parquet.

Scans data/raw/investing_csvs/ for files matching {COUNTRY_ID}_{type}.csv,
parses them all, and writes a single flat Parquet file.

Run fetch_nasi.py first to generate KEN_equity.csv before this script.

Output: data/parquet/investing_manual.parquet

Usage:
    python scripts/fetch_nasi.py   # generates KEN_equity.csv if needed
    python scripts/fetch_investing.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fetchers.investing_csv_parser import InvestingCsvParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "ingestion_settings.yaml"


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    csv_dir = Path(cfg["paths"]["investing_csvs"])
    out_root = Path(cfg["paths"]["parquet_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "investing_manual.parquet"

    if not csv_dir.exists():
        logger.error("CSV directory not found: %s", csv_dir)
        sys.exit(1)

    parser = InvestingCsvParser()
    files = parser.scan_directory(str(csv_dir))

    if not files:
        logger.warning("No CSV files found in %s", csv_dir)
        return

    frames = []
    for filepath in files:
        try:
            df = parser.parse_file(**filepath)
            frames.append(df)
            logger.info("Parsed %s: %d rows", Path(filepath["filepath"]).name, len(df))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", filepath, e)

    if not frames:
        logger.error("All CSV files failed to parse")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    logger.info(
        "Wrote %s: %d rows, countries: %s",
        out_path,
        len(combined),
        sorted(combined["country_id"].unique()),
    )


if __name__ == "__main__":
    main()
