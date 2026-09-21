"""Step-2 core panel regressions: the 3-rung ladder for H2 and H3.

Rung 0  Between estimator            cross-sectional H2 (mean spill ~ KAOPEN + dev)
Rung 1  Pooled OLS + Driscoll-Kraay  PRIMARY: KAOPEN level + interaction (core and full specs)
Rung 2  Two-way fixed effects + DK   robustness: interaction survives all FE

All inferential rungs use Driscoll-Kraay SEs (cov_type="kernel", Bartlett,
bandwidth ~ the 200-day Step-1 window). H3 is tested directly by `fit_fx_h3_direct`
(the focal err_float x crypto_vol two-way moderation, pure and net of openness); the
triple interaction `fit_fx_h3` and the regime subsamples `fit_regime_split` are secondary.

Run:  uv run python -m src.models.panel_regression
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import BetweenOLS, PanelOLS, PooledOLS

GOLD = Path("data/parquet/gold")
ANALYSIS = GOLD / "step2_analysis.parquet"
OUT_PARQUET = GOLD / "step2_panel_coefs.parquet"
OUT_CSV = GOLD / "step2_panel_coefs.csv"

DK_BANDWIDTH = 200  # matches the Step-1 rolling window
GLOBAL_CONTROLS = ["vix", "dxy", "sp500", "oil"]
DEV_CONTROLS = ["log_gdp_pc", "priv_credit_gdp"]


def add_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction columns used across specs."""
    out = df.copy()
    out["ci_x_cv"] = out["chinn_ito"] * out["crypto_vol"]
    if "err_float" in out.columns:
        out["errf_x_cv"] = out["err_float"] * out["crypto_vol"]
        out["ci_x_errf"] = out["chinn_ito"] * out["err_float"]
        out["ci_x_errf_x_cv"] = out["chinn_ito"] * out["err_float"] * out["crypto_vol"]
    return out


def _panel_index(df: pd.DataFrame) -> pd.DataFrame:
    """linearmodels wants a (entity, time) MultiIndex."""
    return df.set_index(["country_id", "date"])


def _design(d: pd.DataFrame, regressors: list[str]) -> pd.DataFrame:
    """add_constant over non-degenerate regressors.

    Drops any regressor that is constant within the estimation sample (e.g.
    crypto_ban inside a regime subsample with no banned country, or a global
    control with no variation). A constant column carries no information once an
    intercept is present and would make the design singular.
    """
    keep = [c for c in regressors if d[c].nunique(dropna=True) > 1]
    return sm.add_constant(d[keep])


def fit_between(df: pd.DataFrame, dv: str, regressors: list[str]):
    """Rung 0: between estimator on entity means."""
    d = _panel_index(df.dropna(subset=[dv] + regressors))
    return BetweenOLS(d[dv], _design(d, regressors)).fit()


def _dk_kwargs(df_index: pd.DataFrame) -> dict:
    """Driscoll-Kraay kwargs, bandwidth capped below the number of time periods."""
    n_periods = df_index.index.get_level_values(1).nunique()
    bw = min(DK_BANDWIDTH, max(1, n_periods - 1))
    return dict(cov_type="kernel", kernel="bartlett", bandwidth=bw)


def fit_pooled_dk(df: pd.DataFrame, dv: str, include_globals: bool = True):
    """Rung 1 PRIMARY: pooled OLS with Driscoll-Kraay SEs.

    spill ~ chinn_ito + crypto_vol + ci_x_cv + crypto_ban + dev (+ globals if full spec)
    `include_globals=False` is the core specification without the global controls.
    """
    regressors = (["chinn_ito", "crypto_vol", "ci_x_cv", "crypto_ban"]
                  + DEV_CONTROLS + (GLOBAL_CONTROLS if include_globals else []))
    d = _panel_index(add_terms(df).dropna(subset=[dv] + regressors))
    return PooledOLS(d[dv], _design(d, regressors)).fit(**_dk_kwargs(d))


def fit_twoway_fe(df: pd.DataFrame, dv: str):
    """Rung 2 robustness: entity + time fixed effects, interaction-only.

    Main effects (chinn_ito, crypto_vol, globals) are intentionally absent, they
    are absorbed by the FE. crypto_ban survives only via within-country time
    variation (e.g. China 2021); where it has none it is dropped (filter below
    removes fully-constant columns; drop_absorbed removes within-entity-constant
    ones such as an always-on ban).
    """
    d = _panel_index(add_terms(df).dropna(subset=[dv, "ci_x_cv", "crypto_ban"]))
    regressors = [c for c in ["ci_x_cv", "crypto_ban"] if d[c].nunique(dropna=True) > 1]
    mod = PanelOLS(d[dv], d[regressors], entity_effects=True, time_effects=True,
                   drop_absorbed=True)
    # Note: an always-on country ban (e.g. Morocco) is within-entity-constant and
    # is absorbed by entity FE -> linearmodels emits an expected AbsorbingEffectWarning.
    return mod.fit(**_dk_kwargs(d))


def fit_fx_h3(df: pd.DataFrame, include_globals: bool = True):
    """H3: FX spill with exchange-rate-regime terms (triple interaction + lower order)."""
    regressors = (["chinn_ito", "crypto_vol", "ci_x_cv", "err_float", "errf_x_cv",
                   "ci_x_errf", "ci_x_errf_x_cv", "crypto_ban"]
                  + DEV_CONTROLS + (GLOBAL_CONTROLS if include_globals else []))
    d = _panel_index(add_terms(df).dropna(subset=["spill_fx"] + regressors))
    return PooledOLS(d["spill_fx"], _design(d, regressors)).fit(**_dk_kwargs(d))


def fit_fx_h3_direct(df: pd.DataFrame, net: bool = False, include_globals: bool = True):
    """H3 PRIMARY: the direct two-way err_float x crypto_vol moderation of the FX share.

    This is the test that matches H3 as written ("crypto->FX is stronger where the currency
    floats"). The focal term `errf_x_cv` is the sample-average float-vs-peg difference in
    crypto's transmission. By contrast `fit_fx_h3` estimates the THREE-way openness x float
    x crypto interaction, whose errf_x_cv is the regime gap evaluated at openness = 0 (the
    closed end of the 0-1 scale, where floaters barely exist), not this average effect.

    net=False (PURE) drops the openness interaction, matching the literal H3 claim.
    net=True adds chinn_ito + ci_x_cv back, so errf_x_cv is read net of openness moderation;
    float and openness are collinear (r ~ 0.40) so the dummy can otherwise proxy openness.
    """
    openness = ["chinn_ito", "ci_x_cv"] if net else []
    regressors = (["err_float", "crypto_vol", "errf_x_cv", "crypto_ban"] + openness
                  + DEV_CONTROLS + (GLOBAL_CONTROLS if include_globals else []))
    d = _panel_index(add_terms(df).dropna(subset=["spill_fx"] + regressors))
    return PooledOLS(d["spill_fx"], _design(d, regressors)).fit(**_dk_kwargs(d))


def fit_regime_split(df: pd.DataFrame, include_globals: bool = True) -> dict:
    """Run the FX pooled+DK spec separately for floaters vs pegs (communicable H3)."""
    out = {}
    for label, val in (("float", 1), ("peg", 0)):
        sub = df[df["err_float"] == val]
        out[label] = fit_pooled_dk(sub, "spill_fx", include_globals=include_globals)
    return out


def fit_meta(res) -> tuple[float, int | None, int | None]:
    """(R^2, n_countries, n_dates) read off a fitted linearmodels result.

    `res.rsquared` is the spec-appropriate measure: overall for pooled/between,
    within for two-way fixed effects. Entity/time totals come from the fitted
    PanelData, so they reflect the ACTUAL estimation sample -- which is smaller
    than the estimable panel whenever a required regressor (the dev controls, the
    global controls on US holidays) is missing -- rather than an outside count.
    This is what guarantees the reported countries/dates match the regression.
    """
    r2 = float(getattr(res, "rsquared", float("nan")))
    try:
        ncty = int(res.entity_info["total"])
    except Exception:
        ncty = None
    try:
        ndate = int(res.time_info["total"])
    except Exception:
        ndate = None
    return r2, ncty, ndate


def tidy(res, channel: str, spec: str) -> pd.DataFrame:
    """Flatten a linearmodels result into tidy coefficient rows."""
    r2, ncty, ndate = fit_meta(res)
    return pd.DataFrame({
        "channel": channel,
        "spec": spec,
        "term": res.params.index,
        "coef": res.params.values,
        "se": res.std_errors.reindex(res.params.index).values,
        "tstat": res.tstats.reindex(res.params.index).values,
        "pval": res.pvalues.reindex(res.params.index).values,
        "nobs": int(res.nobs),
        "r2": r2,
        "n_countries": ncty,
        "n_dates": ndate,
    })


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full ladder for both channels and return one tidy coefficient table."""
    frames = []
    # Equity channel: plain ladder
    frames.append(tidy(fit_between(df, "spill_equity", ["chinn_ito"] + DEV_CONTROLS),
                       "equity", "between"))
    frames.append(tidy(fit_pooled_dk(df, "spill_equity", include_globals=True),
                       "equity", "pooled_full"))
    frames.append(tidy(fit_pooled_dk(df, "spill_equity", include_globals=False),
                       "equity", "pooled_nocontrols"))
    frames.append(tidy(fit_twoway_fe(df, "spill_equity"), "equity", "twfe"))
    # FX channel: plain ladder + H3
    frames.append(tidy(fit_between(df, "spill_fx", ["chinn_ito"] + DEV_CONTROLS),
                       "fx", "between"))
    frames.append(tidy(fit_pooled_dk(df, "spill_fx", include_globals=True),
                       "fx", "pooled_full"))
    frames.append(tidy(fit_pooled_dk(df, "spill_fx", include_globals=False),
                       "fx", "pooled_nocontrols"))
    frames.append(tidy(fit_twoway_fe(df, "spill_fx"), "fx", "twfe"))
    frames.append(tidy(fit_fx_h3(df, include_globals=True), "fx", "fx_h3"))
    # Direct H3: the focal err_float x crypto_vol two-way moderation (matches H3 as written),
    # pure and net of the openness interaction.
    frames.append(tidy(fit_fx_h3_direct(df, net=False, include_globals=True), "fx", "fx_h3_direct"))
    frames.append(tidy(fit_fx_h3_direct(df, net=True, include_globals=True), "fx", "fx_h3_direct_net"))
    split = fit_regime_split(df, include_globals=True)
    frames.append(tidy(split["float"], "fx", "fx_float"))
    frames.append(tidy(split["peg"], "fx", "fx_peg"))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = pd.read_parquet(ANALYSIS)
    coefs = run_all(df)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    coefs.to_parquet(OUT_PARQUET, index=False)
    coefs.to_csv(OUT_CSV, index=False)
    h2 = coefs[(coefs.channel == "equity") & (coefs.spec == "pooled_full") & (coefs.term == "ci_x_cv")]
    print(f"wrote {OUT_PARQUET} and {OUT_CSV}  rows={len(coefs)}")
    if not h2.empty:
        r = h2.iloc[0]
        print(f"H2 (equity, pooled_full) ci_x_cv: coef={r.coef:.4f} se={r.se:.4f} p={r.pval:.4f}")


if __name__ == "__main__":
    main()
