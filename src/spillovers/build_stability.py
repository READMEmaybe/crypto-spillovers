"""Step-1 robustness: rolling-VAR companion spectral radius per window.

For every country-day window underlying ``spillovers.parquet`` this re-fits the
bivariate ``[btc_rv, local]`` VAR (same windowing contract) and records the
companion-matrix spectral radius ``rho`` for each channel. ``rho >= 1`` marks a
non-stationary window whose generalized FEVD is unreliable; the spectral-radius
robustness check (``src/models/robustness_stability.py``, ``notebooks/09``) drops
those windows and re-runs H1/H2/H3.

The spec ``(window, var_lag)`` is read from ``spillovers.parquet`` provenance so it
cannot drift from the headline. Every spillover row is asserted to have a matching
spectral radius (the guarantee that makes the filter exact).

Run:  uv run python -m src.spillovers.build_stability
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.quality.schemas import SPILLOVERS_STABILITY
from src.spillovers.rolling_dy import compute_rolling_spectral_radius

GOLD = Path("data/parquet/gold")
PANEL = GOLD / "panel.parquet"
SPILL = GOLD / "spillovers.parquet"
OUT = GOLD / "spillovers_stability.parquet"


def _read_spec(spill: pd.DataFrame) -> tuple[int, int]:
    """(window, var_lag) from the frozen spillovers provenance (constant columns)."""
    return int(spill["window"].iloc[0]), int(spill["var_lag"].iloc[0])


def country_radius(
    d: pd.DataFrame, window: int, lag: int, min_obs: int
) -> pd.DataFrame | None:
    """``sr_equity`` / ``sr_fx`` for one country, indexed by date. None if no coverage.

    Mirrors ``build_spillovers._country_spillovers``: the local proxy is the squared
    daily return and a channel is computed only where it has observations.
    """
    d = d.sort_values("date").set_index("date")
    btc = d["btc_rv"]
    parts: dict[str, pd.Series] = {}
    if d["equity_ret"].notna().any():
        parts["sr_equity"] = compute_rolling_spectral_radius(
            btc, d["equity_ret"] ** 2, window, lag, min_obs)
    if d["fx_ret"].notna().any():
        parts["sr_fx"] = compute_rolling_spectral_radius(
            btc, d["fx_ret"] ** 2, window, lag, min_obs)
    if not parts:
        return None
    cdf = pd.DataFrame(parts).dropna(how="all")
    if cdf.empty:
        return None
    for col in ("sr_equity", "sr_fx"):
        if col not in cdf:
            cdf[col] = np.nan
    return cdf


def assert_covers_spillovers(out_df: pd.DataFrame, spill: pd.DataFrame) -> None:
    """Every spillover row/channel must have a spectral radius, so it can be filtered.

    (The reverse -- a radius on a window with no share -- is harmless and not
    required, so we check coverage of the share, not exact set equality.)
    """
    m = spill[["date", "country_id", "spill_equity", "spill_fx"]].merge(
        out_df[["date", "country_id", "sr_equity", "sr_fx"]],
        on=["date", "country_id"], how="left", indicator=True)
    missing_rows = int((m["_merge"] != "both").sum())
    assert missing_rows == 0, f"{missing_rows} spillover rows have no stability row"
    for chan in ("equity", "fx"):
        share_only = int((m[f"spill_{chan}"].notna() & m[f"sr_{chan}"].isna()).sum())
        assert share_only == 0, (
            f"{chan}: {share_only} windows have a spillover share but no spectral radius")


def build(panel_path: Path = PANEL, spill_path: Path = SPILL,
          out: Path = OUT) -> pd.DataFrame:
    spill = pd.read_parquet(spill_path)
    window, lag = _read_spec(spill)
    min_obs = round(0.9 * window)  # matches build_spillovers._default_min_obs
    panel = pd.read_parquet(panel_path)

    frames = []
    t0 = time.perf_counter()
    for cid, d in panel.groupby("country_id", sort=True):
        cdf = country_radius(d, window, lag, min_obs)
        if cdf is None:
            continue
        cdf = cdf.reset_index().rename(columns={"index": "date"})
        cdf["country_id"] = cid
        frames.append(cdf)
        ge = int((cdf["sr_equity"] >= 1).sum()); gf = int((cdf["sr_fx"] >= 1).sum())
        print(f"  {cid}: eq={int(cdf.sr_equity.notna().sum()):4d} fx={int(cdf.sr_fx.notna().sum()):4d}"
              f"  rho>=1: eq={ge} fx={gf}")

    out_df = pd.concat(frames, ignore_index=True)
    out_df["window"], out_df["var_lag"] = window, lag
    out_df = out_df[["date", "country_id", "sr_equity", "sr_fx", "window", "var_lag"]] \
        .sort_values(["country_id", "date"]).reset_index(drop=True)

    assert_covers_spillovers(out_df, spill)
    SPILLOVERS_STABILITY.validate(out_df)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    eq, fx = out_df["sr_equity"].dropna(), out_df["sr_fx"].dropna()
    print(f"\nwrote {out}  shape={out_df.shape}  window={window} lag={lag}")
    print(f"  equity: n={len(eq):5d}  rho>=1: {int((eq>=1).sum()):4d} "
          f"({(eq>=1).mean()*100:.2f}%)  max={eq.max():.2f}")
    print(f"  fx:     n={len(fx):5d}  rho>=1: {int((fx>=1).sum()):4d} "
          f"({(fx>=1).mean()*100:.2f}%)  max={fx.max():.2f}")
    print(f"  elapsed: {time.perf_counter()-t0:.1f}s")
    return out_df


def main() -> None:
    build()


if __name__ == "__main__":
    main()
