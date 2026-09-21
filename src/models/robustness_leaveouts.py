"""Step-2 leave-out / alternative-input robustness (§4.8, second axis).

Where `robustness.py` varies *estimator x window x transform* on a fixed DV, this
module varies the *sample and the inputs* at the PRIMARY spec (pooled OLS +
Driscoll-Kraay, full controls, 200-day window, frozen sample) and asks whether the
openness x crypto interaction (`ci_x_cv`) survives:

  primary      the headline frozen analysis (reference cell)
  crisis_drop  drop the 11 hyperinflation/devaluation-flagged countries
  eth          transmitting asset = Ethereum instead of Bitcoin (ETH->market
               rolling-DY share; requires gold/spillovers_eth.parquet)
  trivariate   DV = BTC's share from the BTC + DXY + market trivariate VAR, so the
               share is net of dollar-factor co-movement (gold/spillover_3var.parquet)

Both channels, both DV transforms (level, winsorized-logit). Output schema mirrors
`step2_robustness_coefs.parquet` plus a `leaveout` tag.

Run:  uv run python -m src.models.robustness_leaveouts
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.panel_prep import assemble
from src.models.panel_regression import fit_pooled_dk, tidy
from src.models.robustness import transform_dv
from src.sample.rule import channel_members

GOLD = Path("data/parquet/gold")
PANEL = GOLD / "panel.parquet"
DEV = Path("data/parquet/country_dev.parquet")
SPILL = GOLD / "spillovers.parquet"
SPILL_ETH = GOLD / "spillovers_eth.parquet"
TRIVAR = GOLD / "spillover_3var.parquet"
CRISIS = GOLD / "crisis_flags.parquet"
MEMBERSHIP = GOLD / "sample_membership.parquet"
OUT = GOLD / "step2_leaveout_coefs.parquet"

CHANNELS = {"equity": "spill_equity", "fx": "spill_fx"}


def _primary_cell(df: pd.DataFrame, channel: str, dv_col: str, dv: str,
                  leaveout: str) -> pd.DataFrame:
    """Fit the primary pooled+DK spec on one channel and return the tidy rows."""
    res = fit_pooled_dk(transform_dv(df, dv), dv_col, include_globals=True)
    out = tidy(res, channel, "pooled_full")
    out["dv"], out["window"], out["leaveout"] = dv, 200, leaveout
    return out


def _swap_dv(base: pd.DataFrame, eq_col: str, fx_col: str,
             membership: pd.DataFrame) -> pd.DataFrame:
    """Replace the spill_* DV slots with an alternative measure, frozen-masked.

    Any country outside a channel's frozen membership is NaN'd in that channel, so
    the alternative DV is evaluated on exactly the headline sample.
    """
    out = base.copy()
    out["spill_equity"], out["spill_fx"] = out[eq_col], out[fx_col]
    for ch, col in CHANNELS.items():
        keep = channel_members(membership, ch)
        out.loc[~out["country_id"].isin(keep), col] = float("nan")
    return out


def main() -> None:
    panel = pd.read_parquet(PANEL)
    dev = pd.read_parquet(DEV)
    membership = pd.read_parquet(MEMBERSHIP)
    base = assemble(pd.read_parquet(SPILL), panel, dev, membership=membership)

    # samples / DV swaps, each a frozen analysis frame ----------------------------
    crisis = pd.read_parquet(CRISIS)
    crisis_ids = set(crisis.loc[crisis["crisis"] == 1, "country_id"].unique())
    samples = {
        "primary": base,
        "crisis_drop": base[~base["country_id"].isin(crisis_ids)].copy(),
    }
    tri = pd.read_parquet(TRIVAR)
    samples["trivariate"] = _swap_dv(
        base.merge(tri, on=["country_id", "date"], how="left"),
        "btc_share_dxy_equity", "btc_share_dxy_fx", membership)
    if SPILL_ETH.exists():
        samples["eth"] = assemble(pd.read_parquet(SPILL_ETH), panel, dev,
                                  membership=membership)
    else:
        print(f"  NOTE: {SPILL_ETH} missing -> ETH cell skipped "
              f"(run: uv run python -m src.spillovers.build_spillovers --crypto eth_rv)")

    rows = []
    for leaveout, df in samples.items():
        for channel, dv_col in CHANNELS.items():
            for dv in ("level", "logit"):
                rows.append(_primary_cell(df, channel, dv_col, dv, leaveout))
    coefs = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    coefs.to_parquet(OUT, index=False)

    focus = coefs[(coefs["term"] == "ci_x_cv") & (coefs["dv"] == "level")]
    pd.set_option("display.width", 200)
    print(f"wrote {OUT}  rows={len(coefs)}  leave-outs={list(samples)}")
    print("\n=== ci_x_cv (level, pooled+DK, frozen) across leave-outs ===")
    print(focus[["leaveout", "channel", "coef", "se", "pval", "nobs"]]
          .sort_values(["channel", "leaveout"]).to_string(index=False))


if __name__ == "__main__":
    main()
