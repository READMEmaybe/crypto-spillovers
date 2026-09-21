"""
Parser for manually downloaded Investing.com CSV files.

Investing.com historical data CSVs have a specific format:
    Date,Price,Open,High,Low,Vol.,Change %
    "Nov 01, 2023",12345.67,12300.00,12400.00,12200.00,1.5M,0.37%

Key quirks:
  - Date format: "Nov 01, 2023" (month name abbreviated)
  - Volume: "1.5M", "230.50K", "-" (string notation, not numeric)
  - Change %: "0.37%", "-1.23%" (string with % suffix)
  - Rows are in reverse chronological order (newest first)
  - FX CSVs use "Price" as the exchange rate (no OHLV typically)

Filename convention (enforced by this module):
    {COUNTRY_ID}_{data_type}.csv
    Examples: MAR_equity.csv, TUN_equity.csv, DZA_fx.csv, TUN_fx.csv

See data/raw/investing_csvs/README.md for full instructions.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Expected CSV columns from Investing.com download
INVESTING_CSV_COLUMNS = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]

# Output columns (before writing to Delta)
OUTPUT_COLUMNS = [
    "date", "country_id", "data_type", "filename",
    "price", "open", "high", "low", "volume_raw", "change_pct",
    "currency", "source",
]

# Regex: valid filename pattern
FILENAME_PATTERN = re.compile(r"^([A-Z]{2,3})_(equity|fx)\.csv$", re.IGNORECASE)


class InvestingCsvParser:
    """
    Parses Investing.com historical data CSV files into a standardized DataFrame.

    Usage:
        parser = InvestingCsvParser()

        # Parse a single file
        df = parser.parse_file(
            filepath="/opt/data/raw/investing_csvs/MAR_equity.csv",
            country_id="MAR",
            data_type="equity",
            currency="MAD",
        )

        # Auto-discover and parse all files in a directory
        files = InvestingCsvParser.scan_directory("/opt/data/raw/investing_csvs")
        for f in files:
            df = parser.parse_file(**f)
    """

    def parse_file(
        self,
        filepath: str,
        country_id: str,
        data_type: str,
        currency: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parse a single Investing.com CSV file.

        Args:
            filepath: Path to the CSV file
            country_id: ISO3 country code (e.g., "MAR")
            data_type: "equity" or "fx"
            currency: Local currency ISO code (optional, stored as metadata)

        Returns:
            DataFrame with OUTPUT_COLUMNS. Empty DataFrame on parse failure.
        """
        path = Path(filepath)
        if not path.exists():
            logger.error("File not found: %s", filepath)
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        try:
            raw = pd.read_csv(filepath, thousands=",")
        except Exception as e:
            logger.error("Failed to read CSV %s: %s", filepath, e)
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        # Normalize column names (strip whitespace)
        raw.columns = raw.columns.str.strip()

        # Validate expected columns are present
        required = {"Date", "Price"}
        missing = required - set(raw.columns)
        if missing:
            logger.error("Missing required columns %s in %s", missing, path.name)
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        df = pd.DataFrame()

        # Parse date
        df["date"] = raw["Date"].apply(self._parse_date)
        invalid_dates = df["date"].isna().sum()
        if invalid_dates > 0:
            logger.warning("%d invalid dates in %s", invalid_dates, path.name)

        # Price columns
        df["price"] = pd.to_numeric(raw["Price"], errors="coerce")
        df["open"] = pd.to_numeric(raw.get("Open"), errors="coerce") if "Open" in raw.columns else None
        df["high"] = pd.to_numeric(raw.get("High"), errors="coerce") if "High" in raw.columns else None
        df["low"] = pd.to_numeric(raw.get("Low"), errors="coerce") if "Low" in raw.columns else None

        # Volume: keep raw string, Silver job converts "1.5M" → float
        df["volume_raw"] = raw["Vol."].astype(str).str.strip() if "Vol." in raw.columns else None

        # Change %: strip % suffix and convert
        if "Change %" in raw.columns:
            df["change_pct"] = (
                raw["Change %"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )
        else:
            df["change_pct"] = None

        # Metadata
        df["country_id"] = country_id.upper()
        df["data_type"] = data_type.lower()
        df["filename"] = path.name
        df["currency"] = currency
        df["source"] = "investing_csv"

        # Sort chronologically (Investing.com CSV is newest-first)
        df = df.sort_values("date").reset_index(drop=True)

        # Drop rows with no date or price
        before = len(df)
        df = df.dropna(subset=["date", "price"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning("Dropped %d rows with null date/price in %s", dropped, path.name)

        logger.info(
            "Parsed %s: %d rows | %s → %s | country=%s type=%s",
            path.name,
            len(df),
            df["date"].min() if not df.empty else "N/A",
            df["date"].max() if not df.empty else "N/A",
            country_id,
            data_type,
        )

        return df[OUTPUT_COLUMNS]

    @staticmethod
    def scan_directory(csv_dir: str) -> list[dict]:
        """
        Auto-discover all Investing.com CSV files in a directory.
        Files must match the pattern: {COUNTRY_ID}_{data_type}.csv

        Returns:
            List of dicts with keys: filepath, country_id, data_type
            Files that don't match the naming convention are logged and skipped.
        """
        csv_path = Path(csv_dir)
        if not csv_path.exists():
            logger.warning("CSV directory not found: %s", csv_dir)
            return []

        found = []
        for f in sorted(csv_path.glob("*.csv")):
            match = FILENAME_PATTERN.match(f.name)
            if match:
                found.append({
                    "filepath": str(f),
                    "country_id": match.group(1).upper(),
                    "data_type": match.group(2).lower(),
                })
            else:
                logger.warning(
                    "Skipping file with unexpected name: %s "
                    "(expected format: COUNTRYID_equity.csv or COUNTRYID_fx.csv)",
                    f.name,
                )

        logger.info("Found %d CSV files in %s: %s",
                    len(found), csv_dir, [f["filepath"].split("/")[-1] for f in found])
        return found

    @staticmethod
    def _parse_date(raw_date: str) -> Optional[date]:
        """
        Parse Investing.com date format: "Nov 01, 2023".
        Falls back to standard ISO format if the primary format fails.
        """
        if pd.isna(raw_date):
            return None
        s = str(raw_date).strip().strip('"')
        for fmt in ("%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                return pd.to_datetime(s, format=fmt).date()
            except (ValueError, TypeError):
                continue
        logger.debug("Could not parse date: %r", raw_date)
        return None

    @staticmethod
    def parse_volume(volume_raw: str) -> Optional[float]:
        """
        Convert Investing.com volume strings to float.
        Called by the Silver cleaning job (not stored in Bronze).

        Examples:
            "1.5M"   → 1_500_000.0
            "230.50K" → 230_500.0
            "1.2B"   → 1_200_000_000.0
            "-"      → None
            ""       → None
        """
        if pd.isna(volume_raw) or str(volume_raw).strip() in ("-", "", "nan"):
            return None

        s = str(volume_raw).strip().upper()
        multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}

        for suffix, mult in multipliers.items():
            if s.endswith(suffix):
                try:
                    return float(s[:-1]) * mult
                except ValueError:
                    return None

        try:
            return float(s)
        except ValueError:
            return None
