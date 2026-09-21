"""Step-2 robustness diagnostics: re-run the panel ladder across four cells.

Cells = DV transform {level, logit} x Step-1 window {200, 60}. The baseline core
panel was a (confounded) null for H2; these checks test whether that null is
robust or an artifact of the bounded/skewed DV and the 200-day smoothing.

Reuses the existing, tested `assemble` and `run_all` unchanged; logit is applied
in-memory to the DV before `run_all`.

Run:  uv run python -m src.models.robustness
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.panel_prep import assemble
from src.models.panel_regression import run_all

GOLD = Path("data/parquet/gold")
PANEL = GOLD / "panel.parquet"
DEV = Path("data/parquet/country_dev.parquet")
SPILL = GOLD / "spillovers.parquet"
SPILL60 = GOLD / "spillovers_w60.parquet"
MEMBERSHIP = GOLD / "sample_membership.parquet"
OUT = GOLD / "step2_robustness_coefs.parquet"

DV_COLS = ("spill_equity", "spill_fx")
FOCUS_TERMS = ["ci_x_cv", "chinn_ito", "errf_x_cv", "ci_x_errf_x_cv"]


def logit_winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Winsorize a (0,1) spillover series to its [lo, hi] quantiles, then logit.

    Clipping tames the long lower tail (near-zero spillovers -> ~ -16 logits) so a
    handful of values don't dominate the regression. NaNs are preserved.
    """
    p = s.clip(lower=s.quantile(lo), upper=s.quantile(hi))
    return np.log(p / (1.0 - p))


def transform_dv(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """Return a copy of the analysis frame with the DV columns transformed.

    `how="level"` is the identity; `how="logit"` applies `logit_winsorize` to each
    spillover channel (preserving NaNs, e.g. DZA has no equity).
    """
    if how == "level":
        return df.copy()
    if how == "logit":
        out = df.copy()
        for col in DV_COLS:
            out[col] = logit_winsorize(out[col])
        return out
    raise ValueError(f"unknown dv transform: {how!r}")


def run_cell(analysis: pd.DataFrame, dv: str, window_label: int) -> pd.DataFrame:
    """Run the full ladder on one (dv, window) cell and tag the coefficient table."""
    coefs = run_all(transform_dv(analysis, dv)).copy()
    coefs["dv"] = dv
    coefs["window"] = window_label
    return coefs


def _print_comparison(coefs: pd.DataFrame) -> None:
    focus = coefs[coefs["term"].isin(FOCUS_TERMS)]
    pd.set_option("display.width", 200)
    print("\n=== H2/H3 terms across cells (dv x window) ===")
    print(focus[["dv", "window", "channel", "spec", "term", "coef", "pval", "nobs"]]
          .sort_values(["term", "channel", "spec", "window", "dv"]).to_string(index=False))


def main() -> None:
    panel = pd.read_parquet(PANEL)
    dev = pd.read_parquet(DEV)
    # Bind the frozen sample, exactly as the headline panel does
    # (panel_prep). Without this the battery silently runs on the larger un-masked
    # sample and the pooled_full cell disagrees with the §4.3 headline; robustness
    # must vary window/transform/estimator on the SAME sample, not change the sample.
    membership = pd.read_parquet(MEMBERSHIP) if MEMBERSHIP.exists() else None
    cells = []
    for window_label, spill_path in ((200, SPILL), (60, SPILL60)):
        analysis = assemble(pd.read_parquet(spill_path), panel, dev, membership=membership)
        for dv in ("level", "logit"):
            cells.append(run_cell(analysis, dv, window_label))
    coefs = pd.concat(cells, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    coefs.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  rows={len(coefs)}  cells={coefs[['dv','window']].drop_duplicates().shape[0]}")
    _print_comparison(coefs)


if __name__ == "__main__":
    main()
