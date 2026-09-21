"""Decisive test: is openness x crypto-vol distinguishable from openness x global-risk?

`crypto_vol` is a SINGLE global time series and it spikes on the same calendar days
as VIX/DXY. The whole Step-2 story ("open markets dampen crypto's variance share
under crypto stress") could therefore be a generic "open markets swing harder during
global risk-off" effect, with crypto stress merely marking those episodes.

This horse-race adds openness x VIX and openness x DXY beside openness x crypto-vol
and asks: does the crypto interaction (ci_x_cv) SURVIVE? Run in the two specs that
carry the published result (pooled+Driscoll-Kraay and two-way FE) and both DV
transforms (level, logit), reusing the exact estimation machinery of the pipeline.

Verdict map:
  A  ci_x_cv stays significant with the right sign AFTER adding ci_x_vix/ci_x_dxy,
     incl. under two-way FE  -> a genuine crypto-specific moderation.
  B  ci_x_cv collapses (loses significance / shrinks) once the global interactions
     enter (esp. if ci_x_vix takes over) -> it's global-risk integration, not crypto.
  C  nothing is significant under FE regardless -> clean null.

SECOND horse-race (development): is the crypto MODERATION about *capital openness*,
or about *financial development*? Openness <-> GDP-per-capita correlate ~0.66 across
countries, and the headline only ever interacts crypto with openness. We standardize
the three candidate moderators (openness, GDP-pc, private-credit) and let them compete
to moderate crypto: openness alone, then with development added. If openness x crypto
collapses once development x crypto enters, the moderator is financial maturity, not
capital controls specifically. Output: `gold/dev_horse_race_coefs.parquet`.

Run:  uv run python scripts/horse_race_global_risk.py   (or `make horse-race`)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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
INTERACTIONS = ["ci_x_cv", "ci_x_vix", "ci_x_dxy"]
# Standardized global series -> comparable interaction scales and no artifactual
# collinearity (raw crypto_vol is a negative log, dxy a ~100 level; multiplied by the
# same non-negative chinn_ito they correlate ~ -0.97 purely from sign/scale mismatch).
Z_GLOBALS = {"crypto_vol": "z_cv", "vix": "z_vix", "dxy": "z_dxy"}


def add_hr_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for raw, z in Z_GLOBALS.items():
        out[z] = (out[raw] - out[raw].mean()) / out[raw].std()
    out["ci_x_cv"] = out["chinn_ito"] * out["z_cv"]
    out["ci_x_vix"] = out["chinn_ito"] * out["z_vix"]
    out["ci_x_dxy"] = out["chinn_ito"] * out["z_dxy"]
    return out


def _tidy(res, channel, dv, spec, model) -> pd.DataFrame:
    r2, ncty, ndate = fit_meta(res)
    return pd.DataFrame({
        "channel": channel, "dv": dv, "spec": spec, "model": model,
        "term": res.params.index, "coef": res.params.values,
        "se": res.std_errors.reindex(res.params.index).values,
        "pval": res.pvalues.reindex(res.params.index).values,
        "nobs": int(res.nobs),
        "r2": r2, "n_countries": ncty, "n_dates": ndate,
    })


def fit_pooled(df, dv, regressors):
    d = _panel_index(df.dropna(subset=[dv] + regressors))
    return PooledOLS(d[dv], _design(d, regressors)).fit(**_dk_kwargs(d))


def fit_twfe(df, dv, regressors):
    d = _panel_index(df.dropna(subset=[dv] + regressors))
    keep = [c for c in regressors if d[c].nunique(dropna=True) > 1]
    mod = PanelOLS(d[dv], d[keep], entity_effects=True, time_effects=True,
                   drop_absorbed=True)
    return mod.fit(**_dk_kwargs(d))


def run_channel(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    dv = f"spill_{channel}"
    # pooled main effects mirror the pipeline's `pooled_full`, but with the interacted
    # globals standardized (z_cv/z_vix/z_dxy) for comparability; sp500/oil stay as-is.
    pooled_base = (["chinn_ito", "z_cv", "ci_x_cv", "crypto_ban"] + DEV_CONTROLS
                   + ["z_vix", "z_dxy", "sp500", "oil"])
    pooled_hr = pooled_base + ["ci_x_vix", "ci_x_dxy"]
    # TWFE: global/openness main effects are absorbed by entity+time FE -> interactions only.
    twfe_base = ["ci_x_cv", "crypto_ban"]
    twfe_hr = ["ci_x_cv", "ci_x_vix", "ci_x_dxy", "crypto_ban"]

    frames = []
    for dvt in ("level", "logit"):
        d = add_hr_terms(transform_dv(df, dvt))
        frames += [
            _tidy(fit_pooled(d, dv, pooled_base), channel, dvt, "pooled", "1_baseline"),
            _tidy(fit_pooled(d, dv, pooled_hr), channel, dvt, "pooled", "2_horserace"),
            _tidy(fit_twfe(d, dv, twfe_base), channel, dvt, "twfe", "1_baseline"),
            _tidy(fit_twfe(d, dv, twfe_hr), channel, dvt, "twfe", "2_horserace"),
        ]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Development horse-race: is the crypto MODERATOR openness, or just development? #
# --------------------------------------------------------------------------- #
# Same driver (crypto stress), competing MODERATORS. Standardize all three so the
# interaction coefficients are comparable (effect of a +1 SD moderator on the
# crypto-stress slope of the spillover share).
Z_MODERATORS = {"chinn_ito": "z_open", "log_gdp_pc": "z_gdp", "priv_credit_gdp": "z_cred"}
DEV_INTERACTIONS = ["zopen_x_cv", "zgdp_x_cv", "zcred_x_cv"]


def add_dev_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["z_cv"] = (out["crypto_vol"] - out["crypto_vol"].mean()) / out["crypto_vol"].std()
    for raw, z in Z_MODERATORS.items():
        out[z] = (out[raw] - out[raw].mean()) / out[raw].std()
    out["zopen_x_cv"] = out["z_open"] * out["z_cv"]
    out["zgdp_x_cv"] = out["z_gdp"] * out["z_cv"]
    out["zcred_x_cv"] = out["z_cred"] * out["z_cv"]
    return out


def run_dev_channel(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    dv = f"spill_{channel}"
    # Pooled: keep the lower-order terms (standardized moderator mains + z_cv) so the
    # interaction is clean; carry the SAME global level controls as the headline
    # `pooled_full` (vix/dxy/sp500/oil) for parity, none is interacted here.
    pooled_base = (["z_open", "z_gdp", "z_cred", "z_cv", "zopen_x_cv", "crypto_ban"]
                   + GLOBAL_CONTROLS)
    pooled_dev = pooled_base + ["zgdp_x_cv", "zcred_x_cv"]
    # TWFE: moderator/crypto MAIN effects are absorbed by entity+time FE; keep the openness
    # main (time-varying for 10 countries) so the interaction is cleanly identified.
    twfe_base = ["z_open", "zopen_x_cv", "crypto_ban"]
    twfe_dev = ["z_open", "z_gdp", "z_cred", "zopen_x_cv", "zgdp_x_cv", "zcred_x_cv", "crypto_ban"]

    frames = []
    for dvt in ("level", "logit"):
        d = add_dev_terms(transform_dv(df, dvt))
        frames += [
            _tidy(fit_pooled(d, dv, pooled_base), channel, dvt, "pooled", "1_openness_only"),
            _tidy(fit_pooled(d, dv, pooled_dev), channel, dvt, "pooled", "2_dev_added"),
            _tidy(fit_twfe(d, dv, twfe_base), channel, dvt, "twfe", "1_openness_only"),
            _tidy(fit_twfe(d, dv, twfe_dev), channel, dvt, "twfe", "2_dev_added"),
        ]
    return pd.concat(frames, ignore_index=True)


def dev_collinearity_report(df: pd.DataFrame) -> None:
    cs = df.groupby("country_id")[["chinn_ito", "log_gdp_pc", "priv_credit_gdp"]].mean()
    print("\n--- Cross-country collinearity of the MODERATORS (country means) ---")
    print(cs.corr().round(3).to_string())


def dev_verdict_table(coefs: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    key = coefs[coefs["term"].isin(DEV_INTERACTIONS)].copy()
    for c in ("coef", "se", "pval"):
        key[c] = key[c].round(4)
    print("\n=== Development horse-race: openness vs development as the crypto moderator ===")
    print(key.sort_values(["channel", "dv", "spec", "model", "term"])
             [["channel", "dv", "spec", "model", "term", "coef", "se", "pval", "nobs"]]
          .to_string(index=False))
    print("\n=== Focus: openness x crypto (zopen_x_cv) before vs after development enters ===")
    cc = key[key["term"] == "zopen_x_cv"].pivot_table(
        index=["channel", "dv", "spec"], columns="model", values=["coef", "pval"])
    print(cc.round(4).to_string())


def collinearity_report(df: pd.DataFrame) -> None:
    g = df[["date", "crypto_vol", "vix", "dxy"]].drop_duplicates("date")
    print("\n--- Collinearity of the GLOBAL series (unique dates, n={}) ---".format(len(g)))
    print(g[["crypto_vol", "vix", "dxy"]].corr().round(3).to_string())
    d = add_hr_terms(df)
    print("\n--- Collinearity of the INTERACTIONS (full sample) ---")
    print(d[INTERACTIONS].corr().round(3).to_string())


def verdict_table(coefs: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    key = coefs[coefs["term"].isin(INTERACTIONS)].copy()
    key["coef"] = key["coef"].round(4)
    key["se"] = key["se"].round(4)
    key["pval"] = key["pval"].round(4)
    print("\n=== Interaction coefficients: baseline vs horse-race ===")
    print(key.sort_values(["channel", "dv", "spec", "model", "term"])
             [["channel", "dv", "spec", "model", "term", "coef", "se", "pval", "nobs"]]
          .to_string(index=False))

    print("\n=== Focus: what happens to ci_x_cv when the global interactions enter? ===")
    cc = key[key["term"] == "ci_x_cv"].pivot_table(
        index=["channel", "dv", "spec"], columns="model", values=["coef", "pval"])
    print(cc.round(4).to_string())


def main() -> None:
    df = pd.read_parquet(ANALYSIS)
    print(f"loaded {ANALYSIS}  shape={df.shape}  countries={df.country_id.nunique()}")
    collinearity_report(df)
    coefs = pd.concat([run_channel(df, "fx"), run_channel(df, "equity")], ignore_index=True)
    out = Path("data/parquet/gold/horse_race_coefs.parquet")
    coefs.to_parquet(out, index=False)
    verdict_table(coefs)
    print(f"\nwrote {out}")

    print("\n" + "=" * 78)
    print("DEVELOPMENT HORSE-RACE: is the crypto moderator openness, or development?")
    print("=" * 78)
    dev_collinearity_report(df)
    dev = pd.concat([run_dev_channel(df, "fx"), run_dev_channel(df, "equity")],
                    ignore_index=True)
    dev_out = Path("data/parquet/gold/dev_horse_race_coefs.parquet")
    dev.to_parquet(dev_out, index=False)
    dev_verdict_table(dev)
    print(f"\nwrote {dev_out}")


if __name__ == "__main__":
    main()
