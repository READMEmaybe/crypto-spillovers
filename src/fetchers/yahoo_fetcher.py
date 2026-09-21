"""
Yahoo Finance fetcher for daily equity indices, FX rates, and control variables.

Uses yfinance.download() with auto_adjust=True (adjusts for splits/dividends).
Returns standardized pandas DataFrames, no forward-filling of missing dates.
Non-trading days remain as gaps (NaN), consistent with the thesis methodology.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

# Standardized output columns for equity data
EQUITY_COLUMNS = ["date", "country_id", "ticker", "open", "high", "low", "close", "volume", "currency", "source"]

# Standardized output columns for FX data
FX_COLUMNS = ["date", "country_id", "ticker", "fx_rate_vs_usd", "source"]

# Standardized output columns for control variables
CONTROLS_COLUMNS = ["date", "series_id", "value", "source"]


class YahooFetcher:
    """
    Fetches daily OHLCV data from Yahoo Finance for equity indices, FX rates,
    and global control variables.

    Usage:
        fetcher = YahooFetcher.from_config(
            config_path="/opt/airflow/config/ingestion_settings.yaml"
        )
        countries = load_countries_config(...)

        df_equity = fetcher.fetch_equity("JPN", "^N225", "JPY", "2019-01-01", "2025-12-31")
        df_fx     = fetcher.fetch_fx("JPN", "JPYUSD=X", "2019-01-01", "2025-12-31")
        df_ctrl   = fetcher.fetch_controls("2019-01-01", "2025-12-31")
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    @classmethod
    def from_config(cls, config_path: str) -> "YahooFetcher":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        y = cfg.get("yahoo", {})
        return cls(
            start_date=cfg["dates"]["start"],
            end_date=cfg["dates"]["end"],
            max_retries=y.get("max_retries", 3),
            backoff_base=y.get("backoff_base_seconds", 1.0),
        )

    def fetch_equity(
        self,
        country_id: str,
        ticker: str,
        currency: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV for an equity index.

        Returns DataFrame with columns matching EQUITY_COLUMNS.
        Missing dates are NOT inserted, gaps remain as absent rows.
        """
        raw = self._download(ticker, start or self.start_date, end or self.end_date)
        if raw is None or raw.empty:
            logger.warning("No equity data for %s (%s)", country_id, ticker)
            return pd.DataFrame(columns=EQUITY_COLUMNS)

        df = self._flatten_columns(raw)
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })

        df["date"] = pd.to_datetime(df.index).date
        df = df.reset_index(drop=True)
        df["country_id"] = country_id
        df["ticker"] = ticker
        df["currency"] = currency
        df["source"] = "yahoo"

        # Keep only expected columns; volume may be absent for some indices
        cols_available = [c for c in EQUITY_COLUMNS if c in df.columns]
        missing = [c for c in EQUITY_COLUMNS if c not in df.columns]
        for c in missing:
            df[c] = None

        return df[EQUITY_COLUMNS].dropna(subset=["close"])

    def fetch_fx(
        self,
        country_id: str,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch the raw daily Yahoo close for an XXXUSD=X ticker.

        NOTE: Yahoo XXXUSD=X is quoted as USD per 1 unit of local currency
        (EURUSD=X ≈ 1.18, JPYUSD=X ≈ 0.0064), i.e. USD-per-local, NOT
        local-per-USD. Bronze stores this raw close as-is; the clean layer
        (_yahoo_fx in clean_markets.py) inverts it to local-per-USD so the
        column name is accurate and consistent with the investing_csv source.
        """
        raw = self._download(ticker, start or self.start_date, end or self.end_date)
        if raw is None or raw.empty:
            logger.warning("No FX data for %s (%s)", country_id, ticker)
            return pd.DataFrame(columns=FX_COLUMNS)

        df = self._flatten_columns(raw)
        df["date"] = pd.to_datetime(df.index).date
        df = df.reset_index(drop=True)

        # Use Close as the end-of-day rate
        close_col = next((c for c in df.columns if c.lower() == "close"), None)
        if close_col is None:
            logger.warning("No close column for FX ticker %s", ticker)
            return pd.DataFrame(columns=FX_COLUMNS)

        df["fx_rate_vs_usd"] = pd.to_numeric(df[close_col], errors="coerce")
        df["country_id"] = country_id
        df["ticker"] = ticker
        df["source"] = "yahoo"

        return df[FX_COLUMNS].dropna(subset=["fx_rate_vs_usd"])

    def fetch_controls(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch VIX, SP500, and Brent Oil from Yahoo Finance.
        Returns long-format DataFrame with series_id column.
        DXY is fetched separately from FRED (see fred_fetcher.py).
        """
        series = {
            "VIX": "^VIX",
            "SP500": "^GSPC",
            "BRENT": "BZ=F",
        }
        frames = []
        for series_id, ticker in series.items():
            raw = self._download(ticker, start or self.start_date, end or self.end_date)
            if raw is None or raw.empty:
                logger.warning("No data for control variable %s (%s)", series_id, ticker)
                continue

            df = self._flatten_columns(raw)
            close_col = next((c for c in df.columns if c.lower() == "close"), None)
            if close_col is None:
                continue

            out = pd.DataFrame()
            out["date"] = pd.to_datetime(df.index).date
            out["series_id"] = series_id
            # Use .values to avoid index-alignment NaN when df has DatetimeIndex
            out["value"] = pd.to_numeric(df[close_col].values, errors="coerce")
            out["source"] = "yahoo"
            frames.append(out.dropna(subset=["value"]))

        if not frames:
            return pd.DataFrame(columns=CONTROLS_COLUMNS)
        return pd.concat(frames, ignore_index=True)[CONTROLS_COLUMNS]

    def _download(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Retry-wrapped yfinance download."""
        for attempt in range(self.max_retries):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    timeout=30,
                )
                if df is not None and not df.empty:
                    return df
                logger.warning("Empty response for %s (attempt %d)", ticker, attempt + 1)
            except Exception as e:
                logger.warning("Download error for %s (attempt %d): %s", ticker, attempt + 1, e)

            if attempt < self.max_retries - 1:
                time.sleep(self.backoff_base * (2 ** attempt))

        return None

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten MultiIndex columns from yfinance (happens when downloading
        a single ticker but yfinance still returns multi-level columns).
        """
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]
        return df
