"""
FRED (Federal Reserve Economic Data) fetcher.

Fetches DXY (broad USD index) and FX rate fallbacks for countries
whose rates are not available or reliable on Yahoo Finance.

Requires FRED_API_KEY environment variable.
Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONTROLS_COLUMNS = ["date", "series_id", "value", "source"]
FX_COLUMNS = ["date", "country_id", "ticker", "fx_rate_vs_usd", "source"]


class FredFetcher:
    """
    Fetches time series from FRED API.

    Note on FRED FX series:
    - FRED uses "units of USD per local currency" for most EX* series
      (e.g., EXJPUS = USD per JPY ≈ 0.0067)
    - This is the INVERSE of what we want (local per USD)
    - fetch_fx_fallbacks() automatically inverts the rate so that
      fx_rate_vs_usd = local currency per 1 USD (consistent with Yahoo)

    Usage:
        fetcher = FredFetcher.from_config(config_path="...")
        df_dxy = fetcher.fetch_dxy()
        df_fx = fetcher.fetch_fx_fallbacks(countries_config)
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        # fredapi lazy import, only needed at runtime in containers
        self._api_key = api_key or os.getenv("FRED_API_KEY")
        if not self._api_key:
            raise ValueError(
                "FRED_API_KEY not set. Export it as an environment variable "
                "or pass api_key= explicitly."
            )

    @classmethod
    def from_config(cls, config_path: str) -> "FredFetcher":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        fred_cfg = cfg.get("fred", {})
        api_key_env = fred_cfg.get("api_key_env", "FRED_API_KEY")
        return cls(
            start_date=cfg["dates"]["start"],
            end_date=cfg["dates"]["end"],
            api_key=os.getenv(api_key_env),
            max_retries=fred_cfg.get("max_retries", 3),
            backoff_base=fred_cfg.get("backoff_base_seconds", 2.0),
        )

    def fetch_dxy(self) -> pd.DataFrame:
        """
        Fetch the Broad US Dollar Index (DTWEXBGS) from FRED.
        Higher = stronger USD. Used as a global control variable.

        Returns DataFrame with columns: [date, series_id, value, source]
        """
        df = self._fetch_series("DTWEXBGS")
        if df is None:
            return pd.DataFrame(columns=CONTROLS_COLUMNS)

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df.index).date
        out["series_id"] = "DXY"
        out["value"] = df.values.flatten()
        out["source"] = "fred"
        return out.dropna(subset=["value"])

    def fetch_fx_fallbacks(self, countries_config: list[dict]) -> pd.DataFrame:
        """
        Fetch FX rates from FRED for countries that have a fred_series configured.
        Only fetches when source is "fred" OR as a fallback check.

        IMPORTANT: FRED EX* series are in USD-per-local-currency (inverted).
        This function inverts them to local-per-USD for consistency.

        Returns DataFrame with columns matching FX_COLUMNS.
        """
        frames = []
        for country in countries_config:
            fx = country.get("fx", {})
            fred_series = fx.get("fred_series")
            if not fred_series:
                continue
            if fx.get("source") not in ("fred",):
                continue  # Only fetch FRED if source is explicitly "fred"

            country_id = country["id"]
            logger.info("Fetching FRED FX: %s → %s", country_id, fred_series)

            raw = self._fetch_series(fred_series)
            if raw is None or raw.empty:
                logger.warning("No FRED data for %s (%s)", country_id, fred_series)
                continue

            df = pd.DataFrame()
            df["date"] = pd.to_datetime(raw.index).date
            df["country_id"] = country_id
            df["ticker"] = fred_series

            # FRED reports USD per local → invert to local per USD
            values = pd.to_numeric(raw.values.flatten(), errors="coerce")
            df["fx_rate_vs_usd"] = 1.0 / values
            df["source"] = "fred"

            frames.append(df.dropna(subset=["fx_rate_vs_usd"]))

        if not frames:
            return pd.DataFrame(columns=FX_COLUMNS)
        return pd.concat(frames, ignore_index=True)[FX_COLUMNS]

    def _fetch_series(self, series_id: str) -> Optional[pd.Series]:
        """Retry-wrapped FRED series fetch."""
        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError("fredapi not installed. Run: uv add fredapi")

        fred = Fred(api_key=self._api_key)

        for attempt in range(self.max_retries):
            try:
                series = fred.get_series(
                    series_id,
                    observation_start=self.start_date,
                    observation_end=self.end_date,
                )
                return series
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error("FRED fetch failed for %s after %d retries: %s",
                                 series_id, self.max_retries, e)
                    return None
                sleep_secs = self.backoff_base ** (attempt + 1)
                logger.warning("FRED error (attempt %d): %s. Retrying in %.1fs",
                               attempt + 1, e, sleep_secs)
                time.sleep(sleep_secs)
        return None
