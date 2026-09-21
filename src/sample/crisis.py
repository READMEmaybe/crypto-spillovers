"""Crisis / hyperinflation flag for the 'keep + flag + robustness' design.

Idiosyncratic domestic macro crises (hyperinflation, currency collapse) are a
confound for the FX channel but are also the most thesis-relevant restricted
cases, so they are KEPT in-sample and FLAGGED. The flag is an objective
country-year criterion, never a hand-picked list:

  crisis = 1  if annual CPI inflation >= inflation_pct_min   (World Bank)
              OR annual local-per-USD depreciation >= fx_depreciation_min (our FX)

`annual_fx_depreciation`, `flag_crisis_years`, and `parse_inflation_panel` are
pure; `fetch_inflation_panel` is the World Bank network seam (reused from the
dev-controls fetcher).
"""
from __future__ import annotations

import pandas as pd

INFLATION_INDICATOR = "FP.CPI.TOTL.ZG"  # CPI inflation, annual %


def annual_fx_depreciation(fx: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year change of each country's year-end local-per-USD rate.

    Positive = the local currency depreciated (USD buys more local). The first
    observed year per country has no prior and is NaN.
    """
    df = fx.dropna(subset=["fx_rate_vs_usd"]).copy()
    df["year"] = df["date"].dt.year
    df = df.sort_values("date")
    year_end = (df.groupby(["country_id", "year"], as_index=False)["fx_rate_vs_usd"]
                  .last())
    year_end["fx_depreciation"] = (
        year_end.groupby("country_id")["fx_rate_vs_usd"].pct_change()
    )
    return year_end[["country_id", "year", "fx_depreciation"]]


def flag_crisis_years(inflation: pd.DataFrame, depreciation: pd.DataFrame,
                      params: dict) -> pd.DataFrame:
    """Per country-year crisis flag (1/0) with a reason string."""
    m = inflation.merge(depreciation, on=["country_id", "year"], how="outer")
    hi_infl = m["inflation_pct"] >= params["inflation_pct_min"]
    hi_depr = m["fx_depreciation"] >= params["fx_depreciation_min"]
    hi_infl, hi_depr = hi_infl.fillna(False), hi_depr.fillna(False)

    def _reason(infl, depr):
        return "+".join([t for t, f in (("inflation", infl), ("fx", depr)) if f])

    m["reason"] = [_reason(i, d) for i, d in zip(hi_infl, hi_depr)]
    m["crisis"] = (hi_infl | hi_depr).astype(int)
    return m[["country_id", "year", "inflation_pct", "fx_depreciation",
              "crisis", "reason"]]


def parse_inflation_panel(payload: list, keep: set[str]) -> pd.DataFrame:
    """Reduce a World Bank v2 JSON payload to per country-year inflation rows."""
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["country_id", "year", "inflation_pct"])
    rows = [
        {"country_id": rec.get("countryiso3code"), "year": int(rec["date"]),
         "inflation_pct": float(rec["value"])}
        for rec in payload[1]
        if rec.get("countryiso3code") in keep and rec.get("value") is not None
    ]
    return pd.DataFrame(rows, columns=["country_id", "year", "inflation_pct"])


def fetch_inflation_panel(country_ids: set[str]) -> pd.DataFrame:
    """World Bank CPI inflation panel for the given countries (network seam).

    The shared fetcher requests 2019-2023 (latest reliably published WB years);
    the FX-depreciation criterion covers the full 2019-2025 window from our own
    data, so the union still flags crises across the whole sample.
    """
    from scripts.fetch_worldbank_dev import fetch_indicator
    return parse_inflation_panel(fetch_indicator(INFLATION_INDICATOR), set(country_ids))


def main() -> None:
    from pathlib import Path

    from src.sample.rule import load_rule, study_membership

    gold = Path("data/parquet/gold")
    members = study_membership(pd.read_parquet(gold / "sample_membership.parquet"))
    fx = pd.read_parquet(Path("data/parquet/clean/fx.parquet"))
    params = load_rule()["crisis_flag"]

    depreciation = annual_fx_depreciation(fx[fx.country_id.isin(members)])
    inflation = fetch_inflation_panel(members)
    flags = flag_crisis_years(inflation, depreciation, params)

    out = gold / "crisis_flags.parquet"
    flags.to_parquet(out, index=False)
    flagged = flags[flags.crisis == 1].sort_values(["country_id", "year"])
    countries = sorted(flagged.country_id.unique())
    print(f"flagged {flags.crisis.sum()} country-years across {len(countries)} countries:")
    print(", ".join(countries))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
