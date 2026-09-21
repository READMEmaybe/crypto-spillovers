"""Audit engine: per-country / per-channel data-quality metrics.

For each (country, channel) the engine measures the three quantities the
inclusion rule keys on:

  coverage   = valid daily obs / business days in the full sample window
               (low -> late start, big gaps, or a short-lived series)
  density    = valid obs / business days WITHIN the country's own first..last span
               (low -> a real market calendar full of holes, not just a late start)
  stale_frac = share of trading days whose price equals the day before
               (high -> illiquid / synthetic / pegged: a series that "exists" but
                barely moves, which produces meaningless realized volatility)

`country_metrics` is the pure kernel; `compute_audit` maps it over both clean
market tables. Nothing here looks at a regression coefficient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COL = {"equity": "close", "fx": "fx_rate_vs_usd"}


def business_days(start, end) -> int:
    """Inclusive count of Mon-Fri days between two dates."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return int(np.busday_count(start.date(), end.date())) + 1


def country_metrics(dates: pd.Series, prices: pd.Series, bdays_full: int) -> dict:
    """Data-quality metrics for one country-channel price series.

    `dates` and `prices` are aligned (same length); NaN prices are treated as
    missing observations and excluded from every metric.
    """
    order = np.argsort(dates.to_numpy())
    d = pd.Series(dates.to_numpy()[order])
    p = pd.Series(prices.to_numpy(dtype="float64")[order])
    valid = p.notna().to_numpy()
    n = int(valid.sum())
    if n == 0:
        return {"n_obs": 0, "first": pd.NaT, "last": pd.NaT,
                "coverage": 0.0, "density": np.nan, "stale_frac": np.nan}
    dv = d[valid]
    first, last = dv.iloc[0], dv.iloc[-1]
    span = business_days(first, last)
    pv = p[valid].to_numpy()
    stale = float(np.mean(pv[1:] == pv[:-1])) if len(pv) > 1 else np.nan
    return {
        "n_obs": n,
        "first": first,
        "last": last,
        "coverage": round(n / bdays_full, 4),
        "density": round(n / span, 4) if span else np.nan,
        "stale_frac": round(stale, 4) if stale == stale else np.nan,
    }


def audit_channel(df: pd.DataFrame, channel: str, bdays_full: int) -> pd.DataFrame:
    price = PRICE_COL[channel]
    rows = [
        {"country_id": cid, "channel": channel,
         **country_metrics(g["date"], g[price], bdays_full)}
        for cid, g in df.groupby("country_id")
    ]
    return pd.DataFrame(rows)


def compute_audit(equity: pd.DataFrame, fx: pd.DataFrame,
                  window: tuple[str, str]) -> pd.DataFrame:
    """Audit both channels over the given (start, end) sample window."""
    bdays_full = business_days(*window)
    audit = pd.concat(
        [audit_channel(equity, "equity", bdays_full),
         audit_channel(fx, "fx", bdays_full)],
        ignore_index=True,
    )
    return audit.sort_values(["channel", "coverage"]).reset_index(drop=True)
