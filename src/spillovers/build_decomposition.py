"""Build the share-vs-level component tables (Step-2 decomposition).

Part 1 (components): bivariate BTC->local decomposition A/B/local_var per channel.
Part 2 (3var): trivariate [BTC, global, local] shares for global in {vix, dxy}.

200-day window (the headline window). Output:
  data/parquet/gold/spillover_components.parquet
  data/parquet/gold/spillover_3var.parquet

Run:  uv run python -m src.spillovers.build_decomposition --part all
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.quality.schemas import SPILLOVER_3VAR, SPILLOVER_COMPONENTS
from src.spillovers.rolling_dy import compute_rolling_3var, compute_rolling_components

GOLD = Path("data/parquet/gold")
PANEL = GOLD / "panel.parquet"
COMPONENTS_OUT = GOLD / "spillover_components.parquet"
TRIVAR_OUT = GOLD / "spillover_3var.parquet"
WINDOW, HORIZON, LAG, MIN_OBS = 200, 10, 1, 180


def dxy_ret2(panel: pd.DataFrame) -> pd.DataFrame:
    """Squared daily DXY return per observed DXY date (DXY is global, dedupe by date).

    NaNs are dropped BEFORE pct_change so the return is computed between consecutive
    *observed* DXY values; otherwise a return is NaN whenever the previous calendar
    date's DXY is missing (DXY is absent on ~790 of the panel's 2170 dates), which
    would collapse coverage and starve the trivariate windows below min_obs.
    """
    g = (panel[["date", "dxy"]].drop_duplicates("date")
         .dropna(subset=["dxy"]).sort_values("date"))
    g["dxy_ret2"] = g["dxy"].pct_change() ** 2
    return g[["date", "dxy_ret2"]]


def _wide_components(eq: pd.DataFrame | None, fx: pd.DataFrame | None) -> pd.DataFrame:
    """Combine per-channel component frames into one wide date-indexed frame."""
    parts = []
    for tag, comp in (("equity", eq), ("fx", fx)):
        if comp is None:
            continue
        parts.append(comp[["A", "B", "local_var"]].rename(
            columns={"A": f"A_{tag}", "B": f"B_{tag}", "local_var": f"localvar_{tag}"}))
    wide = pd.concat(parts, axis=1).dropna(how="all")
    return wide.reset_index().rename(columns={"index": "date"})


def build_components() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL)
    frames = []
    t0 = time.perf_counter()
    for cid, d in panel.groupby("country_id", sort=True):
        d = d.sort_values("date").set_index("date")
        btc = d["btc_rv"]
        eq = (compute_rolling_components(btc, d["equity_ret"] ** 2, WINDOW, HORIZON, LAG, MIN_OBS)
              if d["equity_ret"].notna().any() else None)
        fx = (compute_rolling_components(btc, d["fx_ret"] ** 2, WINDOW, HORIZON, LAG, MIN_OBS)
              if d["fx_ret"].notna().any() else None)
        if eq is None and fx is None:
            continue
        wide = _wide_components(eq, fx)
        wide["country_id"] = cid
        frames.append(wide)
        print(f"  {cid}: rows={len(wide)}")
    out = pd.concat(frames, ignore_index=True)
    for col in ["A_equity", "B_equity", "localvar_equity", "A_fx", "B_fx", "localvar_fx"]:
        if col not in out:
            out[col] = np.nan
    out = out[["date", "country_id", "A_equity", "B_equity", "localvar_equity",
               "A_fx", "B_fx", "localvar_fx"]].sort_values(
        ["country_id", "date"]).reset_index(drop=True)
    SPILLOVER_COMPONENTS.validate(out)
    COMPONENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(COMPONENTS_OUT, index=False)
    print(f"wrote {COMPONENTS_OUT}  shape={out.shape}  elapsed={time.perf_counter()-t0:.1f}s")
    return out


def build_3var() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL)
    panel = panel.merge(dxy_ret2(panel), on="date", how="left")
    globals_ = {"vix": "vix", "dxy": "dxy_ret2"}
    frames = []
    t0 = time.perf_counter()
    for cid, d in panel.groupby("country_id", sort=True):
        d = d.sort_values("date").set_index("date")
        btc = d["btc_rv"]
        cols = {}
        for gname, gcol in globals_.items():
            for chan, ret in (("equity", "equity_ret"), ("fx", "fx_ret")):
                if not d[ret].notna().any():
                    continue
                res = compute_rolling_3var(btc, d[gcol], d[ret] ** 2, WINDOW, HORIZON, LAG, MIN_OBS)
                cols[f"btc_share_{gname}_{chan}"] = res["btc_share"]
                cols[f"global_share_{gname}_{chan}"] = res["global_share"]
        if not cols:
            continue
        wide = pd.DataFrame(cols).dropna(how="all").reset_index().rename(columns={"index": "date"})
        wide["country_id"] = cid
        frames.append(wide)
        print(f"  {cid}: rows={len(wide)}")
    out = pd.concat(frames, ignore_index=True)
    share_cols = [f"{m}_{g}_{c}" for g in globals_ for c in ("equity", "fx")
                  for m in ("btc_share", "global_share")]
    for col in share_cols:
        if col not in out:
            out[col] = np.nan
    out = out[["date", "country_id"] + share_cols].sort_values(
        ["country_id", "date"]).reset_index(drop=True)
    SPILLOVER_3VAR.validate(out)
    TRIVAR_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TRIVAR_OUT, index=False)
    print(f"wrote {TRIVAR_OUT}  shape={out.shape}  elapsed={time.perf_counter()-t0:.1f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build share-vs-level component tables")
    ap.add_argument("--part", choices=["components", "3var", "all"], default="all")
    args = ap.parse_args()
    if args.part in ("components", "all"):
        build_components()
    if args.part in ("3var", "all"):
        build_3var()


if __name__ == "__main__":
    main()
