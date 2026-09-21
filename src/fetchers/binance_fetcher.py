"""
Binance REST API fetcher for 1-minute OHLCV klines.

Paginates from start_date to end_date in chunks of 1000 candles (1000 minutes),
writing monthly Parquet files to data/parquet/crypto/{symbol}/{YYYY}-{MM}.parquet.
Idempotent: skips months that already have a file on disk.

Rate limits (as of 2024):
  - Weight limit: 6000/min
  - Each klines request costs 2 weight units
  - Effective safe limit: ~1200 calls/min (using safety_factor=0.8)

For BTC 1-min Jan 2019–Dec 2025:
  ~3.65M minutes ÷ 1000 candles/request = ~3650 requests ≈ ~3 minutes to fetch
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

BINANCE_KLINES_COLUMNS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


class BinanceFetcher:
    """
    Fetches 1-minute OHLCV klines from Binance REST API and writes
    monthly Parquet files to {output_path}/{symbol}/{YYYY}-{MM}.parquet.

    Usage:
        fetcher = BinanceFetcher.from_config(
            symbol="BTCUSDT",
            output_path="data/parquet/crypto",
            config_path="config/ingestion_settings.yaml",
        )
        fetcher.fetch_all()
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        output_path: str,
        base_url: str = "https://api.binance.com/api/v3/klines",
        candles_per_request: int = 1000,
        rate_limit_per_minute: int = 1200,
        safety_factor: float = 0.8,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        progress_every: int = 50,
    ) -> None:
        self.symbol = symbol.upper()
        self.start_ms = self._date_to_ms(start_date)
        self.end_ms = self._date_to_ms(end_date, end_of_day=True)
        self.output_path = Path(output_path) / self.symbol.lower()
        self.base_url = base_url
        self.candles_per_request = candles_per_request
        self.effective_rps = (rate_limit_per_minute * safety_factor) / 60.0
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.progress_every = progress_every

        self._request_count = 0
        self._window_start = time.monotonic()

    @classmethod
    def from_config(cls, symbol: str, output_path: str, config_path: str) -> "BinanceFetcher":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        b = cfg["binance"]
        dates = cfg["dates"]
        return cls(
            symbol=symbol,
            start_date=dates["start"],
            end_date=dates["end"],
            output_path=output_path,
            base_url=b["base_url"],
            candles_per_request=b["candles_per_request"],
            rate_limit_per_minute=b["rate_limit_per_minute"],
            safety_factor=b["safety_factor"],
            max_retries=b["max_retries"],
            backoff_base=b["backoff_base_seconds"],
            progress_every=cfg["logging"]["progress_every_n_requests"],
        )

    def fetch_all(self, force: bool = False) -> None:
        """
        Main entrypoint. Paginates through date range and writes monthly Parquet files.

        Resumes automatically from the last written month unless force=True,
        in which case existing files are overwritten.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        resume_ms = self.start_ms if force else self._find_resume_ms()

        logger.info(
            "Fetching %s | %s → %s%s",
            self.symbol,
            self._ms_to_utc(resume_ms).date(),
            self._ms_to_utc(self.end_ms).date(),
            " (force refetch)" if force else "",
        )
        if resume_ms > self.start_ms and not force:
            logger.info(
                "Resuming: %d months already on disk, skipping to %s",
                len(list(self.output_path.glob("*.parquet"))),
                self._ms_to_utc(resume_ms).date(),
            )

        current_ms = resume_ms
        month_buf: dict[str, list[list]] = {}  # "YYYY-MM" → rows

        total_rows = 0
        request_num = 0

        while current_ms < self.end_ms:
            chunk_end_ms = min(
                current_ms + self.candles_per_request * 60_000,
                self.end_ms,
            )

            rows = self._fetch_chunk(current_ms, chunk_end_ms)
            if not rows:
                # Gap in Binance data (e.g., early 2019), advance and continue
                current_ms = chunk_end_ms
                continue

            for row in rows:
                ts = self._ms_to_utc(row[0])
                month_key = ts.strftime("%Y-%m")
                month_buf.setdefault(month_key, []).append(row)

            # Flush complete months (all months except the last in-progress one)
            all_months = sorted(month_buf.keys())
            for m in all_months[:-1]:
                self._flush_month(m, month_buf.pop(m), force=force)

            current_ms = rows[-1][0] + 60_000  # advance past last candle
            total_rows += len(rows)
            request_num += 1

            if request_num % self.progress_every == 0:
                logger.info(
                    "[%s] %d requests | %d rows | up to %s",
                    self.symbol,
                    request_num,
                    total_rows,
                    self._ms_to_utc(current_ms).date(),
                )

            self._throttle()

        # Flush the final partial month
        for month_key, rows in month_buf.items():
            self._flush_month(month_key, rows, force=force)

        logger.info(
            "Fetch complete: %s | %d total rows | %d requests",
            self.symbol, total_rows, request_num,
        )

    def _find_resume_ms(self) -> int:
        """
        Scan existing parquet files and return the start of the first missing month.
        Returns self.start_ms if no files exist yet.
        """
        existing = sorted(self.output_path.glob("*.parquet"))
        if not existing:
            return self.start_ms

        # File names are YYYY-MM.parquet, take the last one
        last_stem = existing[-1].stem  # e.g. "2024-03"
        try:
            year, month = map(int, last_stem.split("-"))
        except ValueError:
            return self.start_ms

        # Advance to the start of the next month
        if month == 12:
            next_year, next_month = year + 1, 1
        else:
            next_year, next_month = year, month + 1

        next_start = datetime(next_year, next_month, 1, tzinfo=timezone.utc)
        resume_ms = int(next_start.timestamp() * 1000)

        return resume_ms

    def _fetch_chunk(self, start_ms: int, end_ms: int) -> list[list]:
        """Fetch a single chunk from Binance API with exponential backoff retry."""
        params = {
            "symbol": self.symbol,
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms - 1,   # Binance end is inclusive, so subtract 1ms
            "limit": self.candles_per_request,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.get(self.base_url, params=params, timeout=30)

                if resp.status_code == 429 or resp.status_code == 418:
                    retry_after = int(resp.headers.get("Retry-After", self.backoff_base ** (attempt + 1)))
                    logger.warning("Rate limited. Sleeping %ds (attempt %d)", retry_after, attempt + 1)
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error("Failed after %d retries: %s", self.max_retries, e)
                    raise
                sleep_secs = self.backoff_base ** (attempt + 1)
                logger.warning("Request error (attempt %d/%d): %s. Retrying in %.1fs",
                               attempt + 1, self.max_retries, e, sleep_secs)
                time.sleep(sleep_secs)

        return []

    def _flush_month(self, month_key: str, rows: list[list], force: bool = False) -> None:
        """Write a month's rows to a Parquet file: {output_path}/{symbol}/{YYYY}-{MM}.parquet."""
        year, month = month_key.split("-")
        year_int, month_int = int(year), int(month)
        out_path = self.output_path / f"{month_key}.parquet"

        if out_path.exists() and not force:
            logger.debug("Skipping existing file: %s", out_path)
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows, columns=BINANCE_KLINES_COLUMNS)
        df = df.drop(columns=["ignore"])

        # Convert timestamps
        df["open_time_ms"] = df["open_time_ms"].astype("int64")
        df["close_time_ms"] = df["close_time_ms"].astype("int64")
        df["open_time_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)

        # Numeric casts
        for col in ["open", "high", "low", "close", "volume",
                    "quote_asset_volume", "taker_buy_base_volume",
                    "taker_buy_quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").astype("Int64")

        df["symbol"] = self.symbol
        df["source"] = "binance"
        df["ingest_timestamp"] = pd.Timestamp.now("UTC")
        df["year"] = year_int
        df["month"] = month_int

        df.to_parquet(out_path, index=False)
        logger.debug("Wrote %d rows → %s", len(df), out_path)

    def _throttle(self) -> None:
        """Simple sliding-window rate limiter: sleep if we're exceeding effective RPS."""
        self._request_count += 1
        elapsed = time.monotonic() - self._window_start
        expected_elapsed = self._request_count / self.effective_rps

        if expected_elapsed > elapsed:
            time.sleep(expected_elapsed - elapsed)

        # Reset window every 60 seconds
        if elapsed >= 60:
            self._request_count = 0
            self._window_start = time.monotonic()

    @staticmethod
    def _date_to_ms(date_str: str, end_of_day: bool = False) -> int:
        """Convert 'YYYY-MM-DD' to epoch milliseconds (UTC)."""
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _ms_to_utc(ms: int) -> datetime:
        """Convert epoch milliseconds to UTC datetime."""
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
