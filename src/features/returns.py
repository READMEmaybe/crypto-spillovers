"""Daily log returns for equity close and FX rate, per country.

Pure in-memory helpers used by build_panel. No I/O here; the panel builder
reads silver parquets, passes them in, and writes the merged gold panel.

NA policy (plan v4): non-trading days stay NA. Returns are log-differences
of consecutive observed closes, so a holiday gap produces a multi-day
return attributed to the first open day. Downstream models should treat
that as a feature of the local calendar, not impute it away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """date, country_id, close → date, country_id, equity_ret (log diff)."""
    df = df.sort_values(["country_id", "date"]).copy()
    df["equity_ret"] = (
        df.groupby("country_id")["close"]
          .transform(lambda s: np.log(s / s.shift(1)))
    )
    return df[["date", "country_id", "equity_ret"]]


def fx_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """date, country_id, fx_rate_vs_usd → date, country_id, fx_ret."""
    df = df.sort_values(["country_id", "date"]).copy()
    df["fx_ret"] = (
        df.groupby("country_id")["fx_rate_vs_usd"]
          .transform(lambda s: np.log(s / s.shift(1)))
    )
    return df[["date", "country_id", "fx_ret"]]
