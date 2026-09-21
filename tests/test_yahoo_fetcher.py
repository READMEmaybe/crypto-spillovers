"""Tests for YahooFetcher."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fetchers.yahoo_fetcher import YahooFetcher, EQUITY_COLUMNS, FX_COLUMNS, CONTROLS_COLUMNS

# Patch target: the yf module reference inside yahoo_fetcher
YF_PATCH = "fetchers.yahoo_fetcher.yf"


def _make_fetcher() -> YahooFetcher:
    return YahooFetcher(
        start_date="2023-01-01",
        end_date="2023-03-31",
        max_retries=1,
        backoff_base=0.0,
    )


def _make_yf_df(dates, close_values, include_volume=True) -> pd.DataFrame:
    """Build a minimal yfinance-style DataFrame with DatetimeIndex."""
    data = {
        "Open": close_values,
        "High": [v * 1.01 for v in close_values],
        "Low": [v * 0.99 for v in close_values],
        "Close": close_values,
    }
    if include_volume:
        data["Volume"] = [1000000] * len(dates)
    return pd.DataFrame(data, index=pd.to_datetime(dates))


class TestFetchEquity:
    def test_returns_expected_columns(self):
        dates = ["2023-01-03", "2023-01-04", "2023-01-05"]
        mock_df = _make_yf_df(dates, [100.0, 101.0, 99.0])

        with patch(YF_PATCH) as mock_yf:
            mock_yf.download.return_value = mock_df
            fetcher = _make_fetcher()
            result = fetcher.fetch_equity("JPN", "^N225", "JPY")

        assert list(result.columns) == EQUITY_COLUMNS
        assert len(result) == 3

    def test_empty_response_returns_empty_df(self):
        with patch(YF_PATCH) as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()
            fetcher = _make_fetcher()
            result = fetcher.fetch_equity("JPN", "^N225", "JPY")

        assert result.empty
        assert list(result.columns) == EQUITY_COLUMNS

    def test_country_id_populated(self):
        dates = ["2023-01-03"]
        mock_df = _make_yf_df(dates, [100.0])

        with patch(YF_PATCH) as mock_yf:
            mock_yf.download.return_value = mock_df
            fetcher = _make_fetcher()
            result = fetcher.fetch_equity("AUS", "^AXJO", "AUD")

        assert result["country_id"].iloc[0] == "AUS"
        assert result["ticker"].iloc[0] == "^AXJO"
        assert result["currency"].iloc[0] == "AUD"
        assert result["source"].iloc[0] == "yahoo"

    def test_no_forward_fill_on_missing_dates(self):
        """Missing trading days should be absent rows, not forward-filled."""
        dates = ["2023-01-03", "2023-01-05"]  # Jan 4 missing
        mock_df = _make_yf_df(dates, [100.0, 102.0])

        with patch(YF_PATCH) as mock_yf:
            mock_yf.download.return_value = mock_df
            fetcher = _make_fetcher()
            result = fetcher.fetch_equity("JPN", "^N225", "JPY")

        assert len(result) == 2  # No rows added for missing dates

    def test_drops_null_close(self):
        dates = ["2023-01-03", "2023-01-04"]
        mock_df = _make_yf_df(dates, [100.0, float("nan")])

        with patch(YF_PATCH) as mock_yf:
            mock_yf.download.return_value = mock_df
            fetcher = _make_fetcher()
            result = fetcher.fetch_equity("JPN", "^N225", "JPY")

        assert len(result) == 1
        assert result["close"].iloc[0] == 100.0


class TestFetchFX:
    def test_returns_fx_columns(self):
        dates = ["2023-01-03", "2023-01-04"]
        mock_df = _make_yf_df(dates, [130.0, 131.5], include_volume=False)

        with patch(YF_PATCH) as mock_yf:
            mock_yf.download.return_value = mock_df
            fetcher = _make_fetcher()
            result = fetcher.fetch_fx("JPN", "JPYUSD=X")

        assert list(result.columns) == FX_COLUMNS
        assert len(result) == 2
        assert result["fx_rate_vs_usd"].iloc[0] == pytest.approx(130.0)

    def test_source_is_yahoo(self):
        dates = ["2023-01-03"]
        mock_df = _make_yf_df(dates, [1.25], include_volume=False)

        with patch(YF_PATCH) as mock_yf:
            mock_yf.download.return_value = mock_df
            fetcher = _make_fetcher()
            result = fetcher.fetch_fx("AUS", "AUDUSD=X")

        assert result["source"].iloc[0] == "yahoo"

    def test_empty_response_returns_empty_df(self):
        with patch(YF_PATCH) as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()
            fetcher = _make_fetcher()
            result = fetcher.fetch_fx("JPN", "JPYUSD=X")

        assert result.empty


class TestFetchControls:
    def test_returns_three_series(self):
        dates = ["2023-01-03", "2023-01-04"]
        mock_df = _make_yf_df(dates, [20.0, 21.0])

        fetcher = _make_fetcher()
        with patch.object(fetcher, "_download", return_value=mock_df):
            result = fetcher.fetch_controls()

        assert set(result["series_id"].unique()) == {"VIX", "SP500", "BRENT"}
        assert list(result.columns) == CONTROLS_COLUMNS

    def test_series_id_column_present(self):
        dates = ["2023-01-03"]
        mock_df = _make_yf_df(dates, [4000.0])

        fetcher = _make_fetcher()
        with patch.object(fetcher, "_download", return_value=mock_df):
            result = fetcher.fetch_controls()

        assert "series_id" in result.columns
        assert "value" in result.columns


class TestRetry:
    def test_retries_on_exception(self):
        dates = ["2023-01-03"]
        mock_df = _make_yf_df(dates, [100.0])

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Timeout")
            return mock_df

        fetcher = YahooFetcher("2023-01-01", "2023-03-31", max_retries=2, backoff_base=0.0)
        with patch(YF_PATCH) as mock_yf:
            mock_yf.download.side_effect = side_effect
            result = fetcher.fetch_equity("JPN", "^N225", "JPY")

        assert call_count[0] == 2
        assert len(result) == 1
