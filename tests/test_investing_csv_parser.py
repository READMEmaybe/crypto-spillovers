"""Tests for InvestingCsvParser."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fetchers.investing_csv_parser import InvestingCsvParser, OUTPUT_COLUMNS


SAMPLE_CSV_CONTENT = '''"Date","Price","Open","High","Low","Vol.","Change %"
"Nov 03, 2023","12345.67","12300.00","12400.00","12200.00","1.5M","0.37%"
"Nov 02, 2023","12300.00","12250.00","12380.00","12100.00","950.00K","-0.12%"
"Nov 01, 2023","12314.70","12290.00","12350.00","12180.00","-","0.05%"
"Oct 31, 2023","12308.30","12280.00","12360.00","12150.00","2.3M","-0.89%"
"Oct 30, 2023","12418.50","12400.00","12500.00","12300.00","1.8M","0.91%"
'''

FX_CSV_CONTENT = '''"Date","Price","Open","High","Low","Vol.","Change %"
"Nov 03, 2023","10.2345","10.2100","10.2600","10.1900","-","0.24%"
"Nov 02, 2023","10.2100","10.1900","10.2300","10.1700","-","-0.10%"
'''


@pytest.fixture
def sample_csv(tmp_path) -> Path:
    p = tmp_path / "MAR_equity.csv"
    p.write_text(SAMPLE_CSV_CONTENT)
    return p


@pytest.fixture
def fx_csv(tmp_path) -> Path:
    p = tmp_path / "DZA_fx.csv"
    p.write_text(FX_CSV_CONTENT)
    return p


class TestParseFile:
    def test_output_columns(self, sample_csv):
        parser = InvestingCsvParser()
        df = parser.parse_file(str(sample_csv), "MAR", "equity", "MAD")
        assert list(df.columns) == OUTPUT_COLUMNS

    def test_row_count(self, sample_csv):
        parser = InvestingCsvParser()
        df = parser.parse_file(str(sample_csv), "MAR", "equity", "MAD")
        assert len(df) == 5

    def test_date_parsed_correctly(self, sample_csv):
        parser = InvestingCsvParser()
        df = parser.parse_file(str(sample_csv), "MAR", "equity", "MAD")
        # Should be sorted chronologically (Investing.com is newest-first)
        assert df["date"].iloc[0].strftime("%Y-%m-%d") == "2023-10-30"
        assert df["date"].iloc[-1].strftime("%Y-%m-%d") == "2023-11-03"

    def test_price_is_float(self, sample_csv):
        parser = InvestingCsvParser()
        df = parser.parse_file(str(sample_csv), "MAR", "equity", "MAD")
        assert df["price"].dtype == float
        assert df["price"].iloc[-1] == pytest.approx(12345.67)

    def test_change_pct_strips_percent_sign(self, sample_csv):
        parser = InvestingCsvParser()
        df = parser.parse_file(str(sample_csv), "MAR", "equity", "MAD")
        assert df["change_pct"].iloc[-1] == pytest.approx(0.37)
        assert df["change_pct"].iloc[0] == pytest.approx(0.91)

    def test_volume_raw_kept_as_string(self, sample_csv):
        parser = InvestingCsvParser()
        df = parser.parse_file(str(sample_csv), "MAR", "equity", "MAD")
        # pandas 2.x may use StringDtype instead of object for string columns
        assert pd.api.types.is_string_dtype(df["volume_raw"]) or df["volume_raw"].dtype == object
        assert "1.5M" in df["volume_raw"].values

    def test_metadata_populated(self, sample_csv):
        parser = InvestingCsvParser()
        df = parser.parse_file(str(sample_csv), "MAR", "equity", "MAD")
        assert (df["country_id"] == "MAR").all()
        assert (df["data_type"] == "equity").all()
        assert (df["source"] == "investing_csv").all()
        assert (df["currency"] == "MAD").all()
        assert (df["filename"] == "MAR_equity.csv").all()

    def test_fx_file_parses(self, fx_csv):
        parser = InvestingCsvParser()
        df = parser.parse_file(str(fx_csv), "DZA", "fx", "DZD")
        assert len(df) == 2
        assert df["data_type"].iloc[0] == "fx"

    def test_missing_file_returns_empty_df(self):
        parser = InvestingCsvParser()
        df = parser.parse_file("/nonexistent/path.csv", "MAR", "equity")
        assert df.empty
        assert list(df.columns) == OUTPUT_COLUMNS


class TestParseVolume:
    @pytest.mark.parametrize("raw,expected", [
        ("1.5M", 1_500_000.0),
        ("230.50K", 230_500.0),
        ("1.2B", 1_200_000_000.0),
        ("-", None),
        ("", None),
        ("nan", None),
        ("5000", 5000.0),
        ("0", 0.0),
    ])
    def test_volume_conversion(self, raw, expected):
        result = InvestingCsvParser.parse_volume(raw)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


class TestParseDate:
    @pytest.mark.parametrize("raw,expected", [
        ("Nov 01, 2023", "2023-11-01"),
        ("Jan 15, 2019", "2019-01-15"),
        ("Dec 31, 2025", "2025-12-31"),
        ("2023-11-01", "2023-11-01"),  # ISO fallback
    ])
    def test_date_formats(self, raw, expected):
        result = InvestingCsvParser._parse_date(raw)
        assert result is not None
        assert str(result) == expected

    def test_invalid_date_returns_none(self):
        result = InvestingCsvParser._parse_date("not a date")
        assert result is None


class TestScanDirectory:
    def test_discovers_matching_files(self, tmp_path):
        (tmp_path / "MAR_equity.csv").write_text(SAMPLE_CSV_CONTENT)
        (tmp_path / "TUN_equity.csv").write_text(SAMPLE_CSV_CONTENT)
        (tmp_path / "DZA_fx.csv").write_text(FX_CSV_CONTENT)
        (tmp_path / "README.md").write_text("not a csv")
        (tmp_path / "bad_format.csv").write_text("ignored")

        files = InvestingCsvParser.scan_directory(str(tmp_path))

        ids = {f["country_id"] for f in files}
        types = {f["data_type"] for f in files}

        assert "MAR" in ids
        assert "TUN" in ids
        assert "DZA" in ids
        assert len(files) == 3  # README and bad_format skipped
        assert "equity" in types
        assert "fx" in types

    def test_returns_empty_for_missing_dir(self):
        files = InvestingCsvParser.scan_directory("/nonexistent/dir")
        assert files == []

    def test_returns_empty_for_no_matching_files(self, tmp_path):
        (tmp_path / "README.md").write_text("nothing here")
        files = InvestingCsvParser.scan_directory(str(tmp_path))
        assert files == []
