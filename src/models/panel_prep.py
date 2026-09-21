"""Build the canonical Step-2 analysis table.

Joins Step-1 spillovers (DV) with the RHS panel and the country development
controls, then derives the crypto-stress drivers:

  crypto_vol    = log of the trailing-21d mean of global btc_rv  (country-invariant)
  crypto_stress = 1 if crypto_vol is in the top decile (pooled over dates)

Output: data/parquet/gold/step2_analysis.parquet

Run:  uv run python -m src.models.panel_prep
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.quality.schemas import STEP2_ANALYSIS
from src.sample.rule import mask_nonmembers

GOLD = Path("data/parquet/gold")
SPILL = GOLD / "spillovers.parquet"
PANEL = GOLD / "panel.parquet"
DEV = Path("data/parquet/country_dev.parquet")
MEMBERSHIP = GOLD / "sample_membership.parquet"
OUT = GOLD / "step2_analysis.parquet"

CRYPTO_VOL_WINDOW = 21
RHS = ["chinn_ito", "crypto_ban", "err_float", "vix", "dxy", "sp500", "oil"]


def add_crypto_vol(panel: pd.DataFrame, window: int = CRYPTO_VOL_WINDOW,
                   min_periods: int = CRYPTO_VOL_WINDOW) -> pd.DataFrame:
    """Add `crypto_vol` = log trailing-`window` mean of the global btc_rv series.

    btc_rv is identical across countries, so the rolling mean is computed once on
    the unique (date, btc_rv) series and merged back.
    """
    glob = (panel[["date", "btc_rv"]].drop_duplicates("date").sort_values("date"))
    roll = glob["btc_rv"].rolling(window=window, min_periods=min_periods).mean()
    glob = glob.assign(crypto_vol=np.log(roll))
    return panel.merge(glob[["date", "crypto_vol"]], on="date", how="left")


def add_crypto_stress(df: pd.DataFrame, q: float = 0.9) -> pd.DataFrame:
    """Add `crypto_stress` = 1 where crypto_vol >= the q-quantile of the unique
    date-level crypto_vol series."""
    by_date = df[["date", "crypto_vol"]].drop_duplicates("date")
    thresh = by_date["crypto_vol"].quantile(q)
    return df.assign(crypto_stress=(df["crypto_vol"] >= thresh).astype(int))


def assemble(spillovers: pd.DataFrame, panel: pd.DataFrame, dev: pd.DataFrame,
             membership: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join DV + RHS + dev, derive drivers.

    crypto_vol is computed on the FULL panel btc_rv history (which predates the
    spillover sample by ~239 trading days) BEFORE the inner join, so every
    spillover-era row gets a complete trailing-21d window. min_periods=1 only
    affects the pre-spillover panel head, which the inner join discards anyway.

    If `membership` (the frozen-sample decision table) is given, each channel's
    spillover DV is NaN'd out for countries that did not qualify for that channel,
    so the frozen inclusion rule binds the regressions.
    """
    panel = add_crypto_vol(panel, min_periods=1)
    cols = ["date", "country_id", "crypto_vol"] + RHS
    df = spillovers[["date", "country_id", "spill_equity", "spill_fx"]].merge(
        panel[cols], on=["date", "country_id"], how="inner"
    )
    df = df.merge(dev, on="country_id", how="left")
    df = df.dropna(subset=["crypto_vol"])
    if membership is not None:
        df = mask_nonmembers(df, membership)
    df = add_crypto_stress(df)
    return df.sort_values(["country_id", "date"]).reset_index(drop=True)


def build() -> pd.DataFrame:
    spillovers = pd.read_parquet(SPILL)
    panel = pd.read_parquet(PANEL)
    dev = pd.read_parquet(DEV)
    membership = pd.read_parquet(MEMBERSHIP) if MEMBERSHIP.exists() else None
    df = assemble(spillovers, panel, dev, membership=membership)
    STEP2_ANALYSIS.validate(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  shape={df.shape}  countries={df.country_id.nunique()}  "
          f"dates {df.date.min().date()}..{df.date.max().date()}")
    return df


if __name__ == "__main__":
    build()
