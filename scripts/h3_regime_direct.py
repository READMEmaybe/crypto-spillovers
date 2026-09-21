"""Direct test of H3: does exchange-rate-regime flexibility moderate crypto->FX transmission?

NOTE (2026-06-20): the pooled pure/net FX test is now folded into the headline pipeline as
`fit_fx_h3_direct` in src/models/panel_regression.py (specs fx_h3_direct / fx_h3_direct_net;
level in step2_panel_coefs, logit in step2_robustness_coefs). This script is retained as the
EXTENDED diagnostic: it adds the two-way-FE rung and the equity placebo across level and
logit and prints the verdict table.

H3 (as stated) is a TWO-WAY moderation: cryptocurrency-volatility transmission to the
FX variance share is stronger where the currency floats. The published pipeline only
tested it as a *triple* interaction (openness x float x crypto, `fit_fx_h3`) plus a
regime subsample split; it never put err_float x crypto_vol in the driver's seat as a
focal two-way term. This script runs that direct interaction (`errf_x_cv`) the same way
the headline runs openness x crypto_vol, so H3 gets a direct, interpretable coefficient.

Sign reading. Two mechanisms compete for the sign of errf_x_cv:
  * Visibility (H3's own logic): under a float the rate is free to move, so crypto shocks
    surface in measured FX variance  ->  errf_x_cv > 0 (transmission stronger under float).
  * Composition (the thesis's dominant H2 finding): float regimes are also more open and
    developed, whose larger own-variance dilutes every external share  ->  errf_x_cv < 0,
    i.e. the openness/composition effect leaking through the regime dummy.
Because float <-> openness <-> development travel together, the PURE interaction cannot
separate these. We therefore run it both ways, per channel:
  (1) 1_pure            : err_float + crypto_vol + errf_x_cv          (what H3 literally asks)
  (2) 2_net_of_openness : add chinn_ito + ci_x_cv back               (regime net of openness)
If the pure errf_x_cv is significant but collapses once openness is restored, the regime
"effect" was openness in disguise -- the same verdict the development horse-race returned.

Channels: FX is the H3 channel; equity is a PLACEBO (regime is FX-specific, thesis §4.2.4).
Estimators: pooled OLS + Driscoll-Kraay (PRIMARY, mirrors the headline `pooled_full`) and
two-way FE (robustness). err_float is mostly a between-country trait, so the FE interaction
is identified only off the countries that switch regime within the sample; it is reported
with that caveat and on a standardized crypto_vol (z_cv) to avoid the sign/scale artifact
the global horse-race documented (raw crypto_vol is a negative log ~ -4).
DV: level and winsorized logit, reusing the pipeline's `transform_dv`.

Run:  uv run python scripts/h3_regime_direct.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS

from src.models.panel_regression import (
    DEV_CONTROLS,
    GLOBAL_CONTROLS,
    _design,
    _dk_kwargs,
    _panel_index,
    fit_meta,
)
from src.models.robustness import transform_dv

ANALYSIS = Path("data/parquet/gold/step2_analysis.parquet")
OUT = Path("data/parquet/gold/h3_regime_direct_coefs.parquet")

# Focal interaction term per estimator: pooled keeps raw crypto_vol so the coefficient is
# directly comparable to the headline ci_x_cv (-0.0065); FE uses the z-scored version.
FOCAL_TERMS = ["errf_x_cv", "errf_x_zcv"]


def add_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Build the regime and openness interactions, raw and z-scored on crypto_vol."""
    out = df.copy()
    out["errf_x_cv"] = out["err_float"] * out["crypto_vol"]
    out["ci_x_cv"] = out["chinn_ito"] * out["crypto_vol"]
    out["z_cv"] = (out["crypto_vol"] - out["crypto_vol"].mean()) / out["crypto_vol"].std()
    out["errf_x_zcv"] = out["err_float"] * out["z_cv"]
    out["ci_x_zcv"] = out["chinn_ito"] * out["z_cv"]
    return out


def _tidy(res, channel: str, dv: str, spec: str, model: str) -> pd.DataFrame:
    r2, ncty, ndate = fit_meta(res)
    return pd.DataFrame({
        "channel": channel, "dv": dv, "spec": spec, "model": model,
        "term": res.params.index, "coef": res.params.values,
        "se": res.std_errors.reindex(res.params.index).values,
        "pval": res.pvalues.reindex(res.params.index).values,
        "nobs": int(res.nobs),
        "r2": r2, "n_countries": ncty, "n_dates": ndate,
    })


def fit_pooled(df: pd.DataFrame, dv: str, regressors: list[str]):
    d = _panel_index(df.dropna(subset=[dv] + regressors))
    return PooledOLS(d[dv], _design(d, regressors)).fit(**_dk_kwargs(d))


def fit_twfe(df: pd.DataFrame, dv: str, regressors: list[str]):
    d = _panel_index(df.dropna(subset=[dv] + regressors))
    keep = [c for c in regressors if d[c].nunique(dropna=True) > 1]
    mod = PanelOLS(d[dv], d[keep], entity_effects=True, time_effects=True,
                   drop_absorbed=True)
    return mod.fit(**_dk_kwargs(d))


def run_channel(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Pure vs net-of-openness, pooled+DK and two-way FE, for one channel."""
    dv = f"spill_{channel}"
    # Pooled mirrors the headline `pooled_full` but swaps the moderator to err_float.
    pooled_pure = (["err_float", "crypto_vol", "errf_x_cv", "crypto_ban"]
                   + DEV_CONTROLS + GLOBAL_CONTROLS)
    pooled_net = pooled_pure + ["chinn_ito", "ci_x_cv"]
    # FE: main effects of err_float/crypto absorbed; keep err_float main (identified off
    # switchers) and use the z-scored interaction. drop_absorbed clears within-constant cols.
    twfe_pure = ["err_float", "errf_x_zcv", "crypto_ban"]
    twfe_net = ["err_float", "chinn_ito", "errf_x_zcv", "ci_x_zcv", "crypto_ban"]

    frames = []
    for dvt in ("level", "logit"):
        d = add_terms(transform_dv(df, dvt))
        frames += [
            _tidy(fit_pooled(d, dv, pooled_pure), channel, dvt, "1_pure", "pooled"),
            _tidy(fit_pooled(d, dv, pooled_net), channel, dvt, "2_net_of_openness", "pooled"),
            _tidy(fit_twfe(d, dv, twfe_pure), channel, dvt, "1_pure", "twfe"),
            _tidy(fit_twfe(d, dv, twfe_net), channel, dvt, "2_net_of_openness", "twfe"),
        ]
    return pd.concat(frames, ignore_index=True)


def regime_report(df: pd.DataFrame) -> None:
    """How is the regime moderator identified? Counts that bear on pooled vs FE reading."""
    fx = df.dropna(subset=["spill_fx"])
    per_country = fx.groupby("country_id")["err_float"].nunique()
    switchers = int((per_country > 1).sum())
    shares = fx.groupby("country_id")["err_float"].mean()
    always_float = int((shares == 1).sum())
    always_peg = int((shares == 0).sum())
    print("\n--- err_float identification (FX sample) ---")
    print(f"countries in FX channel : {fx.country_id.nunique()}")
    print(f"always floating         : {always_float}")
    print(f"always managed/pegged   : {always_peg}")
    print(f"switch regime in-sample : {switchers}  (the only FE-identifying variation)")
    print(f"country-days floating    : {fx['err_float'].mean():.3f} of FX rows")


def verdict(coefs: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    key = coefs[coefs["term"].isin(FOCAL_TERMS)].copy()
    for c in ("coef", "se", "pval"):
        key[c] = key[c].round(4)
    print("\n=== H3 direct test: err_float x crypto_vol interaction ===")
    print("    H3 predicts a POSITIVE coefficient (crypto transmits MORE to FX under a float).")
    print(key.sort_values(["channel", "model", "dv", "spec"])
             [["channel", "model", "dv", "spec", "term", "coef", "se", "pval", "nobs"]]
          .to_string(index=False))

    print("\n=== FX focus: does the regime effect survive controlling for openness? ===")
    fx = key[(key["channel"] == "fx") & (key["model"] == "pooled")]
    piv = fx.pivot_table(index=["dv"], columns="spec", values=["coef", "pval"])
    print(piv.round(4).to_string())

    print("\n=== Equity placebo (regime is FX-specific; expect nothing) ===")
    eq = key[(key["channel"] == "equity") & (key["model"] == "pooled")]
    print(eq[["dv", "spec", "coef", "pval", "nobs"]].to_string(index=False))


def main() -> None:
    df = pd.read_parquet(ANALYSIS)
    print(f"loaded {ANALYSIS}  shape={df.shape}  countries={df.country_id.nunique()}")
    regime_report(df)
    coefs = pd.concat([run_channel(df, "fx"), run_channel(df, "equity")],
                      ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    coefs.to_parquet(OUT, index=False)
    verdict(coefs)
    print(f"\nwrote {OUT}  rows={len(coefs)}")


if __name__ == "__main__":
    main()
