"""Tests for BinanceFetcher."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Adjust path for local imports when running tests from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fetchers.binance_fetcher import BinanceFetcher, BINANCE_KLINES_COLUMNS


def _make_candle(open_time_ms: int) -> list:
    """Return a fake Binance kline row."""
    return [
        open_time_ms,        # open_time_ms
        "50000.00",          # open
        "51000.00",          # high
        "49000.00",          # low
        "50500.00",          # close
        "123.456",           # volume
        open_time_ms + 59999,  # close_time_ms
        "6250000.00",        # quote_asset_volume
        1234,                # num_trades
        "60.00",             # taker_buy_base_volume
        "3025000.00",        # taker_buy_quote_volume
        "0",                 # ignore
    ]


def _make_fetcher(tmp_path: Path) -> BinanceFetcher:
    return BinanceFetcher(
        symbol="BTCUSDT",
        start_date="2023-01-01",
        end_date="2023-01-02",
        output_path=str(tmp_path),
        candles_per_request=1000,
        rate_limit_per_minute=1200,
        safety_factor=1.0,  # No throttling in tests
        max_retries=2,
        backoff_base=0.01,
        progress_every=9999,
    )


class TestBinanceFetcherInit:
    def test_symbol_uppercase(self, tmp_path):
        f = BinanceFetcher("btcusdt", "2023-01-01", "2023-01-02", str(tmp_path))
        assert f.symbol == "BTCUSDT"

    def test_start_ms_is_epoch(self, tmp_path):
        f = _make_fetcher(tmp_path)
        # 2023-01-01 00:00:00 UTC in ms
        assert f.start_ms == 1672531200000

    def test_output_path_includes_symbol(self, tmp_path):
        f = _make_fetcher(tmp_path)
        assert f.output_path == tmp_path / "btcusdt"


class TestFetchChunk:
    def test_successful_response(self, tmp_path):
        candles = [_make_candle(1672531200000 + i * 60_000) for i in range(100)]

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = candles
            mock_get.return_value = mock_resp

            fetcher = _make_fetcher(tmp_path)
            result = fetcher._fetch_chunk(1672531200000, 1672531200000 + 6_000_000)

        assert len(result) == 100
        assert result[0][0] == 1672531200000

    def test_retry_on_500(self, tmp_path):
        import requests as req
        candles = [_make_candle(1672531200000)]

        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.raise_for_status.side_effect = req.exceptions.HTTPError("500")

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = candles

        with patch("requests.get", side_effect=[error_resp, ok_resp]):
            fetcher = _make_fetcher(tmp_path)
            result = fetcher._fetch_chunk(1672531200000, 1672531260000)

        assert len(result) == 1

    def test_rate_limit_429_sleeps(self, tmp_path):
        candles = [_make_candle(1672531200000)]

        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "0"}  # 0 seconds for fast tests

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = candles

        with patch("requests.get", side_effect=[rate_limited, ok_resp]):
            with patch("time.sleep") as mock_sleep:
                fetcher = _make_fetcher(tmp_path)
                result = fetcher._fetch_chunk(1672531200000, 1672531260000)

        mock_sleep.assert_called()
        assert len(result) == 1


class TestFlushMonth:
    def test_creates_parquet_file(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        rows = [_make_candle(1672531200000 + i * 60_000) for i in range(10)]
        fetcher._flush_month("2023-01", rows)

        out_path = fetcher.output_path / "2023-01.parquet"
        assert out_path.exists()

    def test_parquet_has_correct_columns(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        rows = [_make_candle(1672531200000 + i * 60_000) for i in range(5)]
        fetcher._flush_month("2023-01", rows)

        out_path = fetcher.output_path / "2023-01.parquet"
        df = pd.read_parquet(out_path)

        assert "open_time_ms" in df.columns
        assert "open_time_utc" in df.columns
        assert "symbol" in df.columns
        assert "source" in df.columns
        assert df["symbol"].iloc[0] == "BTCUSDT"
        assert df["source"].iloc[0] == "binance"

    def test_skips_existing_partition(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        rows = [_make_candle(1672531200000)]
        fetcher._flush_month("2023-01", rows)

        out_path = fetcher.output_path / "2023-01.parquet"
        assert out_path.exists()

        # Second call should skip
        mtime_before = out_path.stat().st_mtime
        time.sleep(0.05)
        fetcher._flush_month("2023-01", rows)
        mtime_after = out_path.stat().st_mtime

        assert mtime_before == mtime_after  # File not modified

    def test_numeric_columns_are_float(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        rows = [_make_candle(1672531200000)]
        fetcher._flush_month("2023-01", rows)

        out_path = fetcher.output_path / "2023-01.parquet"
        df = pd.read_parquet(out_path)

        assert df["close"].dtype == float
        assert df["volume"].dtype == float


class TestResume:
    def test_no_files_returns_start_ms(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        assert fetcher._find_resume_ms() == fetcher.start_ms

    def test_resumes_after_last_written_month(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        rows = [_make_candle(1672531200000 + i * 60_000) for i in range(5)]
        fetcher._flush_month("2023-01", rows)

        resume_ms = fetcher._find_resume_ms()
        # Should resume at 2023-02-01 00:00:00 UTC
        from datetime import datetime, timezone
        expected = int(datetime(2023, 2, 1, tzinfo=timezone.utc).timestamp() * 1000)
        assert resume_ms == expected

    def test_force_overwrites_existing_file(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        rows = [_make_candle(1672531200000)]
        fetcher._flush_month("2023-01", rows)

        out_path = fetcher.output_path / "2023-01.parquet"
        mtime_before = out_path.stat().st_mtime
        time.sleep(0.05)
        fetcher._flush_month("2023-01", rows, force=True)
        mtime_after = out_path.stat().st_mtime

        assert mtime_after > mtime_before  # File was overwritten


class TestHelpers:
    def test_date_to_ms(self, tmp_path):
        f = _make_fetcher(tmp_path)
        assert f._date_to_ms("2019-01-01") == 1546300800000

    def test_ms_to_utc(self, tmp_path):
        f = _make_fetcher(tmp_path)
        dt = f._ms_to_utc(1546300800000)
        assert dt.year == 2019
        assert dt.month == 1
        assert dt.day == 1
        assert dt.tzinfo is not None
