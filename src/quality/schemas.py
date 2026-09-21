"""Pandera schemas for each parquet layer transition.

Call `SCHEMA.validate(df)` immediately before writing parquet; failures
surface as SchemaError at build time rather than corrupting gold outputs.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema, Index

# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------

EQUITY_CLEAN = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "country_id": Column(str, Check.str_length(3, 3), nullable=False),
        "ticker": Column(str, nullable=True),
        "open": Column(float, Check.gt(0), nullable=True),
        "high": Column(float, Check.gt(0), nullable=True),
        "low": Column(float, Check.gt(0), nullable=True),
        "close": Column(float, Check.gt(0), nullable=False),
        "volume": Column(float, Check.ge(0), nullable=True),
        "currency": Column(str, nullable=True),
        "source": Column(str, Check.isin(["yahoo", "investing"]), nullable=False),
    },
    unique=["country_id", "date"],
    strict=True,
    coerce=True,
)

FX_CLEAN = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "country_id": Column(str, Check.str_length(3, 3), nullable=False),
        "ticker": Column(str, nullable=True),
        "fx_rate_vs_usd": Column(float, Check.gt(0), nullable=False),
        "source": Column(str, Check.isin(["yahoo", "investing"]), nullable=False),
    },
    unique=["country_id", "date"],
    strict=True,
    coerce=True,
)

CRYPTO_1MIN_CLEAN = DataFrameSchema(
    {
        "timestamp_utc": Column(
            pd.DatetimeTZDtype(tz="UTC"), nullable=False
        ),
        "open": Column(float, Check.gt(0), nullable=False),
        "high": Column(float, Check.gt(0), nullable=False),
        "low": Column(float, Check.gt(0), nullable=False),
        "close": Column(float, Check.gt(0), nullable=False),
        "volume": Column(float, Check.ge(0), nullable=False),
    },
    unique=["timestamp_utc"],
    strict=True,
    coerce=True,
)


def check_monotonic(df: pd.DataFrame, col: str) -> None:
    """Guard that a timestamp column is strictly increasing after dedupe."""
    if not df[col].is_monotonic_increasing:
        raise ValueError(f"{col} is not monotonic increasing after cleaning")


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------

RV_DAILY = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "rv": Column(float, Check.ge(0), nullable=False),
        "bv": Column(float, Check.ge(0), nullable=False),
        "jump": Column(float, Check.ge(0), nullable=False),
        "n_obs": Column(int, Check.gt(0), nullable=False),
        "coverage": Column(float, Check.in_range(0.0, 1.0), nullable=False),
    },
    unique=["date"],
    strict=True,
    coerce=True,
)

SPILLOVERS = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "country_id": Column(str, Check.str_length(3, 3), nullable=False),
        # Directional Diebold-Yilmaz spillover BTC RV -> local vol proxy, 0-1.
        # Nullable: early (pre-first-window) dates and channels a country lacks.
        "spill_equity": Column(float, Check.in_range(0.0, 1.0), nullable=True),
        "spill_fx": Column(float, Check.in_range(0.0, 1.0), nullable=True),
        # Provenance of the rolling spec (constant across rows).
        "window": Column(int, Check.gt(0), nullable=False),
        "horizon": Column(int, Check.gt(0), nullable=False),
        "var_lag": Column(int, Check.gt(0), nullable=False),
    },
    unique=["country_id", "date"],
    strict=True,
    coerce=True,
)

# Companion-matrix spectral radius of each window's bivariate VAR (robustness:
# rho >= 1 flags a non-stationary window). Same window grid as SPILLOVERS, so the
# non-null mask matches it per channel; no horizon (rho is a coefficient property).
SPILLOVERS_STABILITY = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "country_id": Column(str, Check.str_length(3, 3), nullable=False),
        "sr_equity": Column(float, Check.ge(0.0), nullable=True),
        "sr_fx": Column(float, Check.ge(0.0), nullable=True),
        "window": Column(int, Check.gt(0), nullable=False),
        "var_lag": Column(int, Check.gt(0), nullable=False),
    },
    unique=["country_id", "date"],
    strict=True,
    coerce=True,
)

PANEL = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "country_id": Column(str, Check.str_length(3, 3), nullable=False),
        # Daily log-return sanity tripwire. The 90-country panel includes
        # currency-crisis economies whose genuine, documented daily moves exceed
        # ±0.5 (e.g. ARG Milei devaluation Dec-2023 ≈ +0.78; LBN official-rate
        # step devaluations 2023/24 up to ≈ +2.3 ×10). ±3.0 (≈ a 20× single-day
        # move) keeps those real shocks while still tripping on order-of-magnitude
        # data errors / redenominations (e.g. VEN ÷1000 ≈ −6.9). Glitch *reverts*
        # are caught separately in clean_markets.drop_reverting_glitches.
        "equity_ret": Column(float, Check.in_range(-3.0, 3.0), nullable=True),
        "fx_ret": Column(float, Check.in_range(-3.0, 3.0), nullable=True),
        "btc_rv": Column(float, Check.ge(0), nullable=True),
        "eth_rv": Column(float, Check.ge(0), nullable=True),
        "chinn_ito": Column(float, Check.in_range(0.0, 1.0), nullable=True),
        "crypto_ban": Column(int, Check.isin([0, 1]), nullable=False),
        "err_float": Column(int, Check.isin([0, 1]), nullable=False),
        "vix": Column(float, Check.gt(0), nullable=True),
        "dxy": Column(float, Check.gt(0), nullable=True),
        "sp500": Column(float, Check.gt(0), nullable=True),
        "oil": Column(float, Check.gt(0), nullable=True),
    },
    unique=["country_id", "date"],
    strict=True,
    coerce=True,
)

COUNTRY_DEV = DataFrameSchema(
    {
        "country_id": Column(str, Check.str_length(3, 3), nullable=False),
        "log_gdp_pc": Column(float, nullable=True),
        "priv_credit_gdp": Column(float, Check.ge(0), nullable=True),
    },
    unique=["country_id"],
    strict=True,
    coerce=True,
)

STEP2_ANALYSIS = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "country_id": Column(str, Check.str_length(3, 3), nullable=False),
        "spill_equity": Column(float, Check.in_range(0.0, 1.0), nullable=True),
        "spill_fx": Column(float, Check.in_range(0.0, 1.0), nullable=True),
        "chinn_ito": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "crypto_ban": Column(int, Check.isin([0, 1]), nullable=False),
        "err_float": Column(int, Check.isin([0, 1]), nullable=False),
        # log trailing-21d mean of btc_rv; a global (country-invariant) series.
        "crypto_vol": Column(float, nullable=False),
        "crypto_stress": Column(int, Check.isin([0, 1]), nullable=False),
        "vix": Column(float, Check.gt(0), nullable=True),
        "dxy": Column(float, Check.gt(0), nullable=True),
        "sp500": Column(float, Check.gt(0), nullable=True),
        "oil": Column(float, Check.gt(0), nullable=True),
        "log_gdp_pc": Column(float, nullable=True),
        "priv_credit_gdp": Column(float, Check.ge(0), nullable=True),
    },
    unique=["country_id", "date"],
    strict=True,
    coerce=True,
)

# A/B/local_var are non-negative variance-unit quantities; nullable for missing
# channels (e.g. DZA has no equity) and pre-window dates.
SPILLOVER_COMPONENTS = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "country_id": Column(str, Check.str_length(3, 3), nullable=False),
        "A_equity": Column(float, Check.ge(0), nullable=True),
        "B_equity": Column(float, Check.ge(0), nullable=True),
        "localvar_equity": Column(float, Check.ge(0), nullable=True),
        "A_fx": Column(float, Check.ge(0), nullable=True),
        "B_fx": Column(float, Check.ge(0), nullable=True),
        "localvar_fx": Column(float, Check.ge(0), nullable=True),
    },
    unique=["country_id", "date"],
    strict=True,
    coerce=True,
)

# Trivariate GFEVD shares are bounded in [0,1]; nullable for missing channels/windows.
_share = Column(float, Check.in_range(0.0, 1.0), nullable=True)
SPILLOVER_3VAR = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "country_id": Column(str, Check.str_length(3, 3), nullable=False),
        "btc_share_vix_equity": _share, "global_share_vix_equity": _share,
        "btc_share_vix_fx": _share, "global_share_vix_fx": _share,
        "btc_share_dxy_equity": _share, "global_share_dxy_equity": _share,
        "btc_share_dxy_fx": _share, "global_share_dxy_fx": _share,
    },
    unique=["country_id", "date"],
    strict=True,
    coerce=True,
)
