"""Share-vs-level moderation runner.

Re-runs the Step-2 ladder (run_all) on each derived measure by swapping it into
the DV slots of the existing step2_analysis frame, then tags the coefficients.

Part 1: log(A), log(B), log(local_var)      -> gold/decomp_coefs.parquet
Part 2: btc_share / global_share (VIX, DXY) -> gold/crowdout_coefs.parquet

Run:  uv run python -m src.models.decomposition
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.panel_regression import run_all
from src.models.robustness import logit_winsorize
from src.sample.rule import channel_members

GOLD = Path("data/parquet/gold")
ANALYSIS = GOLD / "step2_analysis.parquet"
COMPONENTS = GOLD / "spillover_components.parquet"
TRIVAR = GOLD / "spillover_3var.parquet"
PANEL = GOLD / "panel.parquet"
MEMBERSHIP = GOLD / "sample_membership.parquet"
DECOMP_OUT = GOLD / "decomp_coefs.parquet"
CROWDOUT_OUT = GOLD / "crowdout_coefs.parquet"
FOCUS = ["ci_x_cv", "chinn_ito"]


def mask_by_membership(df: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """NaN every per-channel measure column for countries outside that channel.

    Keeps the mechanism on the same frozen sample as the headline: any column
    named `*_equity` is NaN'd for non-equity-members, `*_fx` for non-fx-members.
    """
    out = df.copy()
    for channel in ("equity", "fx"):
        keep = channel_members(membership, channel)
        cols = [c for c in out.columns if c.endswith(f"_{channel}")]
        out.loc[~out["country_id"].isin(keep), cols] = np.nan
    return out


def log_winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Winsorize a positive series to its [lo, hi] quantiles, then log. NaNs kept."""
    p = s.clip(lower=s.quantile(lo), upper=s.quantile(hi))
    return np.log(p)


def run_measure(base: pd.DataFrame, eq_col: str, fx_col: str, transform, tags: dict) -> pd.DataFrame:
    """Swap a derived measure into the DV slots, run the ladder, tag the result."""
    df = base.copy()
    df["spill_equity"] = transform(df[eq_col])
    df["spill_fx"] = transform(df[fx_col])
    coefs = run_all(df)
    for k, v in tags.items():
        coefs[k] = v
    return coefs


def _load_base() -> pd.DataFrame:
    analysis = pd.read_parquet(ANALYSIS)
    comp = pd.read_parquet(COMPONENTS)
    tri = pd.read_parquet(TRIVAR)
    base = analysis.merge(comp, on=["country_id", "date"], how="inner").merge(
        tri, on=["country_id", "date"], how="inner")
    # Independent local realized variance (squared daily return), built straight from
    # returns: the VAR-free counterpart of `localvar` for the amplification check. If
    # open markets' OWN variance truly expands under crypto stress, openness x crypto
    # should be positive here too, not only inside the (numerically fragile) VAR.
    rets = pd.read_parquet(PANEL)[["country_id", "date", "equity_ret", "fx_ret"]]
    base = base.merge(rets, on=["country_id", "date"], how="left")
    # squared daily return = the one-observation realized variance; exact-zero
    # (unchanged-price) days carry no volatility signal and would make log undefined,
    # so null them (the VAR's localvar is likewise strictly positive).
    base["rv_equity"] = (base["equity_ret"] ** 2).replace(0.0, np.nan)
    base["rv_fx"] = (base["fx_ret"] ** 2).replace(0.0, np.nan)
    if MEMBERSHIP.exists():
        base = mask_by_membership(base, pd.read_parquet(MEMBERSHIP))
    return base


def _print_focus(coefs: pd.DataFrame, label: str, keys: list[str]) -> None:
    f = coefs[(coefs["term"].isin(FOCUS)) & (coefs["spec"].isin(["between", "pooled_full", "twfe"]))]
    pd.set_option("display.width", 200)
    print(f"\n=== {label}: interaction (ci_x_cv) + level (chinn_ito) ===")
    print(f[keys + ["channel", "spec", "term", "coef", "pval"]]
          .sort_values(keys + ["channel", "spec", "term"]).to_string(index=False))


def main() -> None:
    base = _load_base()

    # Part 1: bivariate decomposition (level measures -> log)
    part1 = []
    for measure, (eq, fx) in {
        "logA": ("A_equity", "A_fx"),
        "logB": ("B_equity", "B_fx"),
        "log_localvar": ("localvar_equity", "localvar_fx"),
        "local_rv_indep": ("rv_equity", "rv_fx"),
    }.items():
        part1.append(run_measure(base, eq, fx, log_winsorize, {"measure": measure}))
    decomp = pd.concat(part1, ignore_index=True)
    decomp.to_parquet(DECOMP_OUT, index=False)
    _print_focus(decomp, "PART 1 decomposition (logA/logB/log_localvar)", ["measure"])

    # Part 2: trivariate crowding-out (shares -> logit)
    part2 = []
    for glob in ("vix", "dxy"):
        for measure in ("btc_share", "global_share"):
            eq, fx = f"{measure}_{glob}_equity", f"{measure}_{glob}_fx"
            part2.append(run_measure(base, eq, fx, logit_winsorize,
                                     {"measure": measure, "global": glob}))
    crowd = pd.concat(part2, ignore_index=True)
    crowd.to_parquet(CROWDOUT_OUT, index=False)
    _print_focus(crowd, "PART 2 crowding-out (btc_share/global_share x VIX/DXY)", ["global", "measure"])

    print(f"\nwrote {DECOMP_OUT} ({len(decomp)} rows) and {CROWDOUT_OUT} ({len(crowd)} rows)")


if __name__ == "__main__":
    main()
