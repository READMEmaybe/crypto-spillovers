"""Step 1 build: rolling Diebold-Yilmaz spillovers for all countries.

Loops every country in the panel, estimates the BTC RV -> equity and
BTC RV -> FX rolling directional spillovers, and writes the long
country-date artifact to ``data/parquet/gold/spillovers.parquet``.

Run:  uv run python -m src.spillovers.build_spillovers              # 200-day window
      uv run python -m src.spillovers.build_spillovers --window 60  # robustness
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.quality.schemas import SPILLOVERS
from src.spillovers.rolling_dy import compute_rolling_dy

PANEL = Path("data/parquet/gold/panel.parquet")
OUT = Path("data/parquet/gold/spillovers.parquet")

# Baseline spec: 200-day rolling bivariate VAR(1), generalized FEVD.
HORIZON, LAG = 10, 1


def _default_min_obs(window: int) -> int:
    """0.9 * window, matching the 200->180 baseline ratio."""
    return round(0.9 * window)


def _default_out(window: int, crypto_col: str = "btc_rv") -> Path:
    """Baseline file for Bitcoin + the 200-day window; a tagged file otherwise.

    Bitcoin keeps the canonical names (`spillovers.parquet`, `spillovers_w60.parquet`);
    a non-Bitcoin asset is tagged by its symbol, e.g. `spillovers_eth.parquet`.
    """
    asset = crypto_col.removesuffix("_rv")
    tag = "" if asset == "btc" else f"_{asset}"
    win = "" if window == 200 else f"_w{window}"
    if not tag and not win:
        return OUT
    return OUT.parent / f"spillovers{tag}{win}.parquet"


def _country_spillovers(
    d: pd.DataFrame, window: int, horizon: int, lag: int, min_obs: int,
    crypto_col: str = "btc_rv",
) -> pd.DataFrame | None:
    """Both channel spillovers for one country, indexed by date.

    Returns None if the country has neither equity nor FX coverage. Local
    volatility proxy is the squared daily return, per spec. `crypto_col` selects
    the transmitting asset's realized-variance series (default Bitcoin; pass
    ``eth_rv`` for the Ethereum robustness check).
    """
    d = d.sort_values("date").set_index("date")
    crypto = d[crypto_col]
    parts: dict[str, pd.Series] = {}
    if d["equity_ret"].notna().any():
        parts["spill_equity"] = compute_rolling_dy(
            crypto, d["equity_ret"] ** 2, window, horizon, lag, min_obs
        )
    if d["fx_ret"].notna().any():
        parts["spill_fx"] = compute_rolling_dy(
            crypto, d["fx_ret"] ** 2, window, horizon, lag, min_obs
        )
    if not parts:
        return None

    cdf = pd.DataFrame(parts).dropna(how="all")  # drop dates with no channel value
    if cdf.empty:
        return None
    for col in ("spill_equity", "spill_fx"):
        if col not in cdf:
            cdf[col] = np.nan
    return cdf


def build(
    window: int = 200,
    horizon: int = HORIZON,
    lag: int = LAG,
    min_obs: int | None = None,
    panel_path: Path = PANEL,
    out: Path | None = None,
    crypto_col: str = "btc_rv",
) -> pd.DataFrame:
    if min_obs is None:
        min_obs = _default_min_obs(window)
    if out is None:
        out = _default_out(window, crypto_col)

    panel = pd.read_parquet(panel_path)
    frames = []
    t0 = time.perf_counter()
    for cid, d in panel.groupby("country_id", sort=True):
        cdf = _country_spillovers(d, window, horizon, lag, min_obs, crypto_col)
        if cdf is None:
            print(f"  {cid}: no coverage, skipped")
            continue
        cdf = cdf.reset_index().rename(columns={"index": "date"})
        cdf["country_id"] = cid
        frames.append(cdf)
        ne = int(cdf["spill_equity"].notna().sum())
        nf = int(cdf["spill_fx"].notna().sum())
        print(f"  {cid}: equity={ne:4d}  fx={nf:4d}  rows={len(cdf):4d}")

    out_df = pd.concat(frames, ignore_index=True)
    out_df["window"], out_df["horizon"], out_df["var_lag"] = window, horizon, lag
    out_df = out_df[
        ["date", "country_id", "spill_equity", "spill_fx", "window", "horizon", "var_lag"]
    ].sort_values(["country_id", "date"]).reset_index(drop=True)

    SPILLOVERS.validate(out_df)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    eq_n = out_df["country_id"][out_df["spill_equity"].notna()].nunique()
    fx_n = out_df["country_id"][out_df["spill_fx"].notna()].nunique()
    print(
        f"\nwrote {out}  shape={out_df.shape}  countries={out_df.country_id.nunique()}\n"
        f"  equity channel: {eq_n} countries, {int(out_df.spill_equity.notna().sum())} obs\n"
        f"  fx channel:     {fx_n} countries, {int(out_df.spill_fx.notna().sum())} obs\n"
        f"  window={window} min_obs={min_obs}  elapsed: {time.perf_counter() - t0:.1f}s"
    )
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling Diebold-Yilmaz spillovers")
    ap.add_argument("--window", type=int, default=200,
                    help="rolling window in trading days (default 200)")
    ap.add_argument("--crypto", default="btc_rv",
                    help="transmitting-asset RV column (default btc_rv; eth_rv for "
                         "the Ethereum robustness check)")
    args = ap.parse_args()
    build(window=args.window, crypto_col=args.crypto)


if __name__ == "__main__":
    main()
