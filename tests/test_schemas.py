"""Unit tests for Pandera schemas in src/quality/schemas.py."""
from __future__ import annotations

import pandas as pd
import pandera.errors as pae
import pytest

from src.quality.schemas import CRYPTO_1MIN_CLEAN, EQUITY_CLEAN, FX_CLEAN


def _equity_row(**over):
    base = dict(
        date=pd.Timestamp("2024-01-02"),
        country_id="AUS",
        ticker="^AXJO",
        open=7500.0,
        high=7550.0,
        low=7490.0,
        close=7520.0,
        volume=350_000.0,
        currency="AUD",
        source="yahoo",
    )
    base.update(over)
    return base


def test_equity_schema_accepts_valid_row():
    df = pd.DataFrame([_equity_row()])
    EQUITY_CLEAN.validate(df)


def test_equity_schema_rejects_non_positive_close():
    df = pd.DataFrame([_equity_row(close=0.0)])
    with pytest.raises(pae.SchemaError):
        EQUITY_CLEAN.validate(df)


def test_equity_schema_rejects_bad_country_id():
    df = pd.DataFrame([_equity_row(country_id="AU")])
    with pytest.raises(pae.SchemaError):
        EQUITY_CLEAN.validate(df)


def test_equity_schema_rejects_unknown_source():
    df = pd.DataFrame([_equity_row(source="bloomberg")])
    with pytest.raises(pae.SchemaError):
        EQUITY_CLEAN.validate(df)


def test_equity_schema_enforces_uniqueness():
    df = pd.DataFrame([_equity_row(), _equity_row()])
    with pytest.raises(pae.SchemaError):
        EQUITY_CLEAN.validate(df)


def test_equity_schema_allows_null_ohlv():
    df = pd.DataFrame([_equity_row(open=None, high=None, low=None, volume=None)])
    EQUITY_CLEAN.validate(df)


def test_fx_schema_accepts_valid_row():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2024-01-02"),
        "country_id": "JPN",
        "ticker": "JPYUSD=X",
        "fx_rate_vs_usd": 147.5,
        "source": "yahoo",
    }])
    FX_CLEAN.validate(df)


def test_fx_schema_rejects_negative_rate():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2024-01-02"),
        "country_id": "JPN",
        "ticker": "JPYUSD=X",
        "fx_rate_vs_usd": -0.5,
        "source": "yahoo",
    }])
    with pytest.raises(pae.SchemaError):
        FX_CLEAN.validate(df)


def test_crypto_schema_accepts_valid_row():
    df = pd.DataFrame([{
        "timestamp_utc": pd.Timestamp("2024-01-02 00:00:00", tz="UTC"),
        "open": 42000.0,
        "high": 42100.0,
        "low": 41950.0,
        "close": 42050.0,
        "volume": 12.3,
    }])
    CRYPTO_1MIN_CLEAN.validate(df)


def test_crypto_schema_coerces_to_utc():
    # coerce=True: pandera localizes naive timestamps to UTC.
    # Guard the output is tz-aware UTC regardless of input.
    df = pd.DataFrame([{
        "timestamp_utc": pd.Timestamp("2024-01-02 00:00:00"),
        "open": 42000.0,
        "high": 42100.0,
        "low": 41950.0,
        "close": 42050.0,
        "volume": 12.3,
    }])
    out = CRYPTO_1MIN_CLEAN.validate(df)
    assert str(out["timestamp_utc"].dt.tz) == "UTC"


def test_crypto_schema_rejects_non_positive_price():
    df = pd.DataFrame([{
        "timestamp_utc": pd.Timestamp("2024-01-02 00:00:00", tz="UTC"),
        "open": 0.0,
        "high": 42100.0,
        "low": 41950.0,
        "close": 42050.0,
        "volume": 12.3,
    }])
    with pytest.raises(pae.SchemaError):
        CRYPTO_1MIN_CLEAN.validate(df)


# append to tests/test_schemas.py
import pandas as pd
import pytest
import pandera.errors
from src.quality.schemas import COUNTRY_DEV, STEP2_ANALYSIS


def test_country_dev_accepts_valid():
    df = pd.DataFrame({"country_id": ["AUS"], "log_gdp_pc": [10.8], "priv_credit_gdp": [140.0]})
    COUNTRY_DEV.validate(df)


def test_country_dev_rejects_bad_country_id():
    df = pd.DataFrame({"country_id": ["AUSTRALIA"], "log_gdp_pc": [10.8], "priv_credit_gdp": [140.0]})
    with pytest.raises(pandera.errors.SchemaError):
        COUNTRY_DEV.validate(df)


def test_step2_analysis_accepts_valid():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-04"]),
        "country_id": ["AUS"],
        "spill_equity": [0.05], "spill_fx": [0.02],
        "chinn_ito": [1.0], "crypto_ban": [0], "err_float": [1],
        "crypto_vol": [-8.5], "crypto_stress": [0],
        "vix": [20.0], "dxy": [90.0], "sp500": [3700.0], "oil": [50.0],
        "log_gdp_pc": [10.8], "priv_credit_gdp": [140.0],
    })
    STEP2_ANALYSIS.validate(df)


from src.quality.schemas import SPILLOVER_COMPONENTS, SPILLOVER_3VAR


def test_spillover_components_accepts_valid():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-04"]), "country_id": ["AUS"],
        "A_equity": [0.5], "B_equity": [1.0], "localvar_equity": [2.0],
        "A_fx": [0.3], "B_fx": [0.9], "localvar_fx": [1.5],
    })
    SPILLOVER_COMPONENTS.validate(df)


def test_spillover_3var_accepts_valid_and_bounds_shares():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-04"]), "country_id": ["AUS"],
        "btc_share_vix_equity": [0.2], "global_share_vix_equity": [0.3],
        "btc_share_vix_fx": [0.1], "global_share_vix_fx": [0.4],
        "btc_share_dxy_equity": [0.2], "global_share_dxy_equity": [0.3],
        "btc_share_dxy_fx": [0.1], "global_share_dxy_fx": [0.4],
    })
    SPILLOVER_3VAR.validate(df)
    bad = df.copy(); bad.loc[0, "btc_share_vix_fx"] = 1.5
    with pytest.raises(pae.SchemaError):
        SPILLOVER_3VAR.validate(bad)
