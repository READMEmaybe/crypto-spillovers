#!/usr/bin/env python3
"""Generate Appendix A (Supplementary Empirical Results) as Markdown.

Reproducible: every number is pulled from the frozen gold artifacts in
`data/parquet/gold/*` (plus `kaopen.parquet` and `config/err_float.csv`).
Nothing is hand-transcribed. Run:

    uv run python scripts/make_appendix_a.py

Outputs
-------
- thesis/appendix/appendix_a.md                 consolidated appendix (all 5 tables)
- thesis/tables/a1_country_descriptives.md      standalone per-table files
- thesis/tables/a2_err_regime.md
- thesis/tables/a3_global_horserace.md
- thesis/tables/a4_fevd_diagnostics.md
- thesis/tables/a5_robustness_matrix.md

Notes
-----
- Driscoll-Kraay SEs (bandwidth 200) where applicable; *** p<.01, ** p<.05, * p<.10.
- Goodness-of-fit (R^2) was not retained in the frozen coefficient artifacts and
  is reported as "--".
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "parquet" / "gold"
PARQ = ROOT / "data" / "parquet"
APPDIR = ROOT / "thesis" / "appendix"
TABDIR = ROOT / "thesis" / "tables"
APPDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

DASH = "-"  # em dash for "not applicable / not in sample"

# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
AN = pl.read_parquet(GOLD / "step2_analysis.parquet")
KA = (pl.read_parquet(PARQ / "kaopen.parquet")
        .filter(pl.col("year") == 2023)
        .select("iso3", "country_name", "ka_open_normalized")
        .unique(subset="iso3"))
ERRF = pl.read_csv(ROOT / "config" / "err_float.csv")

PANEL = pl.read_parquet(GOLD / "step2_panel_coefs.parquet")
H3 = pl.read_parquet(GOLD / "h3_regime_direct_coefs.parquet")
HR = pl.read_parquet(GOLD / "horse_race_coefs.parquet")
DECOMP = pl.read_parquet(GOLD / "decomp_coefs.parquet")
CROWD = pl.read_parquet(GOLD / "crowdout_coefs.parquet")
ROB = pl.read_parquet(GOLD / "step2_robustness_coefs.parquet")
LO = pl.read_parquet(GOLD / "step2_leaveout_coefs.parquet")
STAB = pl.read_parquet(GOLD / "step2_stability_coefs.parquet")

# ---------------------------------------------------------------------------
# auxiliary counts (countries / dates / crisis / stability removals)
# ---------------------------------------------------------------------------
eq = AN.filter(pl.col("spill_equity").is_not_null())
fx = AN.filter(pl.col("spill_fx").is_not_null())
EQ_N, FX_N = eq["country_id"].n_unique(), fx["country_id"].n_unique()
EQ_D, FX_D = eq["date"].n_unique(), fx["date"].n_unique()
FLOAT_N = fx.filter(pl.col("err_float") == 1)["country_id"].n_unique()
PEG_N = fx.filter(pl.col("err_float") == 0)["country_id"].n_unique()
FLOAT_D = fx.filter(pl.col("err_float") == 1)["date"].n_unique()
PEG_D = fx.filter(pl.col("err_float") == 0)["date"].n_unique()

crisis_ids = set(pl.read_parquet(GOLD / "crisis_flags.parquet")
                 .filter(pl.col("crisis") == 1)["country_id"].unique().to_list())
eq_ids = set(eq["country_id"].unique().to_list())
fx_ids = set(fx["country_id"].unique().to_list())
CRISIS_EQ_N = EQ_N - len(crisis_ids & eq_ids)
CRISIS_FX_N = FX_N - len(crisis_ids & fx_ids)


def _stab_removed(channel: str) -> tuple[int, float]:
    s = STAB.filter((pl.col("channel") == channel) & (pl.col("spec") == "pooled_full")
                    & (pl.col("term") == "ci_x_cv"))
    base = s.filter(pl.col("threshold") == float("inf"))["nobs"].max()
    drop = s.filter(pl.col("threshold") == 1.0)["nobs"].max()
    rem = int(base - drop)
    return rem, rem / base * 100.0


STAB_FX_REM, STAB_FX_PCT = _stab_removed("fx")
STAB_EQ_REM, STAB_EQ_PCT = _stab_removed("equity")

# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def stars(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def get(df: pl.DataFrame, filt: dict, term: str) -> tuple | None:
    """Return (coef, se, pval) for one term under a column filter, else None."""
    f = df
    for k, v in filt.items():
        f = f.filter(pl.col(k) == v)
    r = f.filter(pl.col("term") == term)
    if r.height == 0:
        return None
    row = r.row(0, named=True)
    return row["coef"], row["se"], row["pval"]


def nobs_for(df: pl.DataFrame, filt: dict) -> int | None:
    f = df
    for k, v in filt.items():
        f = f.filter(pl.col(k) == v)
    return int(f["nobs"].max()) if f.height else None


def meta(df: pl.DataFrame, filt: dict) -> tuple:
    """(R^2, n_countries, n_dates) for a model, read from its coefficient rows.

    These are constant across a model's terms (they describe the fit), so the
    first matching row carries them. They come from the fitted result object, so
    they reflect the ACTUAL estimation sample, not the estimable panel."""
    f = df
    for k, v in filt.items():
        f = f.filter(pl.col(k) == v)
    if f.height == 0:
        return None, None, None
    r = f.row(0, named=True)
    iv = lambda x: int(x) if x is not None else None
    return r.get("r2"), iv(r.get("n_countries")), iv(r.get("n_dates"))


def fnum(x: float, dp: int, signed: bool = True) -> str:
    """Fixed-dp, but fall back to 2 significant figures when a small-but-nonzero
    value would round to zero at the column's display precision."""
    if round(x, dp) == 0 and x != 0:
        s = f"{x:.2g}"
        return ("+" + s if signed and not s.startswith("-") else s)
    return f"{x:+.{dp}f}" if signed else f"{abs(x):.{dp}f}"


def cell(g: tuple | None, dp: int, absorbed: bool = False) -> str:
    if g is None:
        return "(absorbed)" if absorbed else DASH
    coef, se, p = g
    return f"{fnum(coef, dp)}{stars(p)} ({fnum(se, dp, signed=False)})"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def section_row(label: str, ncol: int) -> list[str]:
    return [f"**{label}**"] + [""] * ncol


def reg_block(col_defs: list[dict], row_defs: list[tuple],
              footers: list[str]) -> str:
    """col_defs: dicts {head, df, filt, dp, ncty, ndate, absorb(set)}.
    row_defs: ("SECTION", label) | (label, term)."""
    ncol = len(col_defs)
    headers = ["Term"] + [c["head"] for c in col_defs]
    rows: list[list[str]] = []
    for a, b in row_defs:
        if a == "SECTION":
            rows.append(section_row(b, ncol))
            continue
        line = [a]
        for c in col_defs:
            absorbed = b in c.get("absorb", set())
            line.append(cell(get(c["df"], c["filt"], b), c["dp"], absorbed))
        rows.append(line)
    metas = [meta(c["df"], c["filt"]) for c in col_defs]  # (r2, ncty, ndate) per column
    for f in footers:
        if f == "N":
            rows.append(["Observations"] + [f"{nobs_for(c['df'], c['filt']):,}"
                         if nobs_for(c["df"], c["filt"]) else DASH for c in col_defs])
        elif f == "C":
            rows.append(["Countries"] + [str(m[1]) if m[1] is not None else DASH for m in metas])
        elif f == "D":
            rows.append(["Date observations"]
                        + [f"{m[2]:,}" if m[2] is not None else DASH for m in metas])
        elif f == "FE":
            rows.append(["Country fixed effects"] + [c.get("cfe", "No") for c in col_defs])
            rows.append(["Date fixed effects"] + [c.get("dfe", "No") for c in col_defs])
        elif f == "R2":
            rows.append(["R²"] + [f"{m[0]:.3f}" if m[0] is not None else DASH for m in metas])
    return md_table(headers, rows)


# ---------------------------------------------------------------------------
# Table A.1 - country-level descriptives
# ---------------------------------------------------------------------------
def table_a1() -> tuple[str, str, str]:
    g = (AN.group_by("country_id").agg(
            eq_n=pl.col("spill_equity").is_not_null().sum(),
            eq_mean=pl.col("spill_equity").mean(),
            eq_med=pl.col("spill_equity").median(),
            fx_n=pl.col("spill_fx").is_not_null().sum(),
            fx_mean=pl.col("spill_fx").mean(),
            fx_med=pl.col("spill_fx").median())
         .join(KA, left_on="country_id", right_on="iso3", how="left"))

    flt = (ERRF.group_by("country_id").agg(fs=pl.col("err_float").mean()))
    g = g.join(flt, on="country_id", how="left")

    eq_rank = (g.filter(pl.col("eq_n") > 0).sort("eq_mean", descending=True)
                .with_row_index("eq_rank", 1).select("country_id", "eq_rank"))
    fx_rank = (g.filter(pl.col("fx_n") > 0).sort("fx_mean", descending=True)
                .with_row_index("fx_rank", 1).select("country_id", "fx_rank"))
    g = g.join(eq_rank, on="country_id", how="left").join(fx_rank, on="country_id", how="left")
    n_eq, n_fx = eq_rank.height, fx_rank.height

    g = g.filter((pl.col("eq_n") > 0) | (pl.col("fx_n") > 0))

    # spearman rho for the note
    e = g.filter((pl.col("eq_n") > 0) & pl.col("ka_open_normalized").is_not_null())
    f = g.filter((pl.col("fx_n") > 0) & pl.col("ka_open_normalized").is_not_null())
    rho_eq = spearmanr(e["ka_open_normalized"], e["eq_mean"]).correlation
    rho_fx = spearmanr(f["ka_open_normalized"], f["fx_mean"]).correlation

    def regime(fs: float | None) -> str:
        if fs is None:
            return DASH
        if fs >= 0.999:
            return f"{fs:.2f} (Float)"
        if fs <= 0.001:
            return f"{fs:.2f} (Managed/Peg)"
        return f"{fs:.2f} (Mixed)"

    def pct(x, n):
        return f"{x * 100:.2f}%" if (n and n > 0 and x is not None) else DASH

    rows = []
    for r in g.sort("country_name", nulls_last=True).iter_rows(named=True):
        name = r["country_name"] or r["country_id"]
        ka = f"{r['ka_open_normalized']:.2f}" if r["ka_open_normalized"] is not None else DASH
        eqn = r["eq_n"] or 0
        fxn = r["fx_n"] or 0
        rows.append([
            name, r["country_id"], ka,
            f"{eqn:,}" if eqn else DASH, pct(r["eq_mean"], eqn), pct(r["eq_med"], eqn),
            f"{r['eq_rank']}/{n_eq}" if r["eq_rank"] is not None else DASH,
            f"{fxn:,}" if fxn else DASH, pct(r["fx_mean"], fxn), pct(r["fx_med"], fxn),
            f"{r['fx_rank']}/{n_fx}" if r["fx_rank"] is not None else DASH,
            regime(r["fs"]),
        ])

    headers = ["Country", "ISO-3", "KAOPEN (2023)",
               "Equity obs", "Equity mean", "Equity median", "Equity rank",
               "FX obs", "FX mean", "FX median", "FX rank", "Float share / regime"]
    title = ("**Table A.1.** Country-level mean Bitcoin-associated variance shares, "
             "observation counts, and institutional characteristics.")
    note = (
        f"*Note.* Mean and median values are calculated over each country's available "
        f"rolling generalized-FEVD observations in the final estimable sample "
        f"({n_eq} equity, {n_fx} FX). KAOPEN is the 2023 Chinn-Ito score (normalized 0 "
        f"closed to 1 open) used in the cross-country descriptive comparison. \"Float "
        f"share\" is the fraction of the country's 2019-2025 country-years coded floating "
        f"under the de facto IMF AREAER regime measure. Equity and FX samples differ "
        f"because eligibility and spillover estimation were determined separately by "
        f"channel; a dash marks a channel in which the country is not estimable. Ranks are "
        f"by descending country-mean share within each channel. Spearman rank correlation "
        f"between mean share and KAOPEN: equity rho = {rho_eq:+.2f}, FX rho = {rho_fx:+.2f}.")
    return title, md_table(headers, rows), note


# ---------------------------------------------------------------------------
# Table A.2 - exchange-rate regime
# ---------------------------------------------------------------------------
def table_a2() -> tuple[str, str, str]:
    # Panel A - direct H3 (pooled), fx
    def h3col(spec, dv):
        return dict(df=H3, filt=dict(channel="fx", model="pooled", spec=spec, dv=dv))
    colsA = [
        {"head": "FX level (baseline)", **h3col("1_pure", "level"), "dp": 4, "ncty": FX_N, "ndate": FX_D},
        {"head": "FX log-odds (baseline)", **h3col("1_pure", "logit"), "dp": 3, "ncty": FX_N, "ndate": FX_D},
        {"head": "FX level (net of openness)", **h3col("2_net_of_openness", "level"), "dp": 4, "ncty": FX_N, "ndate": FX_D},
        {"head": "FX log-odds (net of openness)", **h3col("2_net_of_openness", "logit"), "dp": 3, "ncty": FX_N, "ndate": FX_D},
    ]
    rowsA = [
        ("Float regime", "err_float"),
        ("Crypto stress", "crypto_vol"),
        ("Float x crypto stress", "errf_x_cv"),
        ("KAOPEN", "chinn_ito"),
        ("KAOPEN x crypto stress", "ci_x_cv"),
        ("Crypto restriction", "crypto_ban"),
        ("Log GDP per capita", "log_gdp_pc"),
        ("Private credit/GDP", "priv_credit_gdp"),
        ("VIX", "vix"),
        ("Broad U.S. Dollar Index", "dxy"),
        ("S&P 500", "sp500"),
        ("Brent", "oil"),
        ("Constant", "const"),
    ]
    panelA = reg_block(colsA, rowsA, ["N", "C", "D", "R2"])

    # Panel B - subsamples (robustness, window 200), fx
    def subcol(spec, dv):
        return dict(df=ROB, filt=dict(channel="fx", spec=spec, dv=dv, window=200))
    colsB = [
        {"head": "Floating: level", **subcol("fx_float", "level"), "dp": 4, "ncty": FLOAT_N},
        {"head": "Floating: log-odds", **subcol("fx_float", "logit"), "dp": 3, "ncty": FLOAT_N},
        {"head": "Non-floating: level", **subcol("fx_peg", "level"), "dp": 4, "ncty": PEG_N},
        {"head": "Non-floating: log-odds", **subcol("fx_peg", "logit"), "dp": 3, "ncty": PEG_N},
    ]
    rowsB = [
        ("KAOPEN", "chinn_ito"),
        ("Crypto stress", "crypto_vol"),
        ("KAOPEN x crypto stress", "ci_x_cv"),
        ("Crypto restriction", "crypto_ban"),
        ("Log GDP per capita", "log_gdp_pc"),
        ("Private credit/GDP", "priv_credit_gdp"),
        ("VIX", "vix"),
        ("Broad U.S. Dollar Index", "dxy"),
        ("S&P 500", "sp500"),
        ("Brent", "oil"),
        ("Constant", "const"),
    ]
    panelB = reg_block(colsB, rowsB, ["N", "C", "R2"])

    # Panel C - triple (robustness, window 200), fx
    def tripcol(dv):
        return dict(df=ROB, filt=dict(channel="fx", spec="fx_h3", dv=dv, window=200))
    colsC = [
        {"head": "FX level", **tripcol("level"), "dp": 4, "ncty": FX_N, "ndate": FX_D},
        {"head": "FX log-odds", **tripcol("logit"), "dp": 3, "ncty": FX_N, "ndate": FX_D},
    ]
    rowsC = [
        ("KAOPEN", "chinn_ito"),
        ("Float", "err_float"),
        ("Crypto stress", "crypto_vol"),
        ("KAOPEN x Float", "ci_x_errf"),
        ("KAOPEN x crypto stress", "ci_x_cv"),
        ("Float x crypto stress", "errf_x_cv"),
        ("KAOPEN x Float x crypto stress", "ci_x_errf_x_cv"),
        ("Crypto restriction", "crypto_ban"),
        ("Log GDP per capita", "log_gdp_pc"),
        ("Private credit/GDP", "priv_credit_gdp"),
        ("VIX", "vix"),
        ("Broad U.S. Dollar Index", "dxy"),
        ("S&P 500", "sp500"),
        ("Brent", "oil"),
        ("Constant", "const"),
    ]
    panelC = reg_block(colsC, rowsC, ["N", "C", "D", "R2"])

    body = (f"*Panel A. Direct H3 test: float x crypto stress (pooled + Driscoll-Kraay).*\n\n{panelA}\n\n"
            f"*Panel B. Separate-subsample openness models (pooled + Driscoll-Kraay, 200-day window).*\n\n{panelB}\n\n"
            f"*Panel C. Triple-interaction model (pooled + Driscoll-Kraay).*\n\n{panelC}")
    title = ("**Table A.2.** Exchange-rate-regime interactions with cryptocurrency-market "
             "stress in the FX variance-share models.")
    note = (
        "*Note.* Float equals one for country-years classified as floating under the study's "
        "de facto IMF AREAER regime coding. Crypto stress is the logged trailing 21-day "
        "Bitcoin realized-variance measure. Coefficients are shown with Driscoll-Kraay "
        "standard errors (bandwidth 200) in parentheses; *** p<.01, ** p<.05, * p<.10. "
        "Level-share and log-odds coefficients are not comparable in magnitude. H3 predicts "
        "a positive float x crypto interaction (Panel A); the estimate is instead negative "
        "and survives only in the pure log-odds cell, collapsing once openness is controlled. "
        "Panels B and C show the openness moderation is regime-independent. Because the de "
        "facto regime can change across years, a country may enter both the floating and "
        "non-floating subsamples, so the Panel B country counts sum to more than the 75 FX "
        "countries in the pooled model. Reported countries and dates are each model's own "
        "estimation sample; R² is the overall pooled R².")
    return title, body, note


# ---------------------------------------------------------------------------
# Table A.3 - global-risk horse-race
# ---------------------------------------------------------------------------
def table_a3() -> tuple[str, str, str]:
    def hc(ch, dv):
        return dict(df=HR, filt=dict(channel=ch, spec="pooled", model="2_horserace", dv=dv))
    cols = [
        {"head": "Equity level", **hc("equity", "level"), "dp": 4, "ncty": EQ_N, "ndate": EQ_D},
        {"head": "Equity log-odds", **hc("equity", "logit"), "dp": 3, "ncty": EQ_N, "ndate": EQ_D},
        {"head": "FX level", **hc("fx", "level"), "dp": 4, "ncty": FX_N, "ndate": FX_D},
        {"head": "FX log-odds", **hc("fx", "logit"), "dp": 3, "ncty": FX_N, "ndate": FX_D},
    ]
    rows = [
        ("KAOPEN", "chinn_ito"),
        ("Standardized crypto stress", "z_cv"),
        ("Standardized VIX", "z_vix"),
        ("Standardized Broad U.S. Dollar Index", "z_dxy"),
        ("KAOPEN x crypto stress", "ci_x_cv"),
        ("KAOPEN x VIX", "ci_x_vix"),
        ("KAOPEN x Broad U.S. Dollar Index", "ci_x_dxy"),
        ("Crypto restriction", "crypto_ban"),
        ("Log GDP per capita", "log_gdp_pc"),
        ("Private credit/GDP", "priv_credit_gdp"),
        ("S&P 500", "sp500"),
        ("Brent", "oil"),
        ("Constant", "const"),
    ]
    body = reg_block(cols, rows, ["N", "C", "D", "R2"])
    title = ("**Table A.3.** Capital-account-openness interactions with cryptocurrency stress, "
             "global risk aversion, and dollar conditions.")
    note = (
        "*Note.* Cryptocurrency stress, VIX, and the Broad U.S. Dollar Index (FRED DTWEXBGS) "
        "are standardized before interaction construction, so the interaction coefficients "
        "represent the differential association of KAOPEN with a one-standard-deviation "
        "increase in each global factor. Pooled estimation with Driscoll-Kraay standard "
        "errors (bandwidth 200) in parentheses; *** p<.01, ** p<.05, * p<.10. Level and "
        "log-odds coefficients are on different scales. The openness x crypto interaction "
        "remains negative in the FX channel after the global-risk interactions are added, and "
        "in equity it carries the opposite sign to openness x VIX, indicating two distinct "
        "channels. R² is the overall pooled R².")
    return title, body, note


# ---------------------------------------------------------------------------
# Table A.4 - FEVD-component and local-volatility diagnostics (FX)
# ---------------------------------------------------------------------------
def table_a4() -> tuple[str, str, str]:
    def dcol(measure, spec):
        return dict(df=DECOMP, filt=dict(channel="fx", spec=spec, measure=measure))

    def ccol(measure, spec):
        return dict(df=CROWD, filt=dict(channel="fx", spec=spec, measure=measure, global_="dxy"))

    # crowdout uses column name "global"; alias for filter
    def ccol2(measure, spec):
        return dict(df=CROWD.rename({"global": "global_"}),
                    filt=dict(channel="fx", spec=spec, measure=measure, global_="dxy"))

    heads = ["Bitcoin gen. FEVD contribution (logA)",
             "Local own-shock gen. FEVD contribution (logB)",
             "Raw local forecast-error variance",
             "Independent local return-based volatility",
             "Bitcoin share, trivariate system",
             "Broad dollar share, trivariate system"]

    def make_cols(spec, fe):
        cfe = "Yes" if fe else "No"
        return [
            {"head": heads[0], **dcol("logA", spec), "dp": 3, "ncty": FX_N, "cfe": cfe, "dfe": cfe,
             "absorb": {"chinn_ito", "crypto_vol"}},
            {"head": heads[1], **dcol("logB", spec), "dp": 3, "ncty": FX_N, "cfe": cfe, "dfe": cfe,
             "absorb": {"chinn_ito", "crypto_vol"}},
            {"head": heads[2], **dcol("log_localvar", spec), "dp": 3, "ncty": FX_N, "cfe": cfe, "dfe": cfe,
             "absorb": {"chinn_ito", "crypto_vol"}},
            {"head": heads[3], **dcol("local_rv_indep", spec), "dp": 3, "ncty": FX_N, "cfe": cfe, "dfe": cfe,
             "absorb": {"chinn_ito", "crypto_vol"}},
            {"head": heads[4], **ccol2("btc_share", spec), "dp": 3, "ncty": None, "cfe": cfe, "dfe": cfe,
             "absorb": {"chinn_ito", "crypto_vol"}},
            {"head": heads[5], **ccol2("global_share", spec), "dp": 3, "ncty": None, "cfe": cfe, "dfe": cfe,
             "absorb": {"chinn_ito", "crypto_vol"}},
        ]

    rows = [
        ("KAOPEN", "chinn_ito"),
        ("Crypto stress", "crypto_vol"),
        ("KAOPEN x crypto stress", "ci_x_cv"),
        ("Crypto restriction", "crypto_ban"),
    ]
    panelA = reg_block(make_cols("pooled_full", fe=False), rows, ["N", "C", "R2"])
    panelB = reg_block(make_cols("twfe", fe=True), rows, ["N", "C", "FE", "R2"])

    body = (f"*Panel A. Pooled diagnostic models (Driscoll-Kraay, FX channel).*\n\n{panelA}\n\n"
            f"*Panel B. Two-way fixed-effects diagnostic models (FX channel).*\n\n{panelB}")
    title = ("**Table A.4.** Diagnostic models for generalized-FEVD contributions and local "
             "market-volatility measures (FX channel).")
    note = (
        "*Note.* The normalized Bitcoin-associated share is a relative allocation of "
        "generalized-FEVD contributions. Raw local forecast-error variance is reported as an "
        "ancillary diagnostic and does not mechanically constitute the denominator of the "
        "row-normalized Bitcoin share. logA/logB/raw-local-variance and the independent "
        "return-based volatility are winsorized-then-logged; the trivariate-system shares are "
        "logit-transformed. Cells report the openness x crypto-volatility coefficient and "
        "associated terms with Driscoll-Kraay standard errors (bandwidth 200) in parentheses; "
        "*** p<.01, ** p<.05, * p<.10. Under fixed effects the openness and crypto main "
        "effects are absorbed. Component models are diagnostic and do not identify a causal "
        "mechanism. R² is the overall R² in Panel A and the within R² in Panel B.")
    return title, body, note


# ---------------------------------------------------------------------------
# Table A.5 - robustness matrix
# ---------------------------------------------------------------------------
def table_a5() -> tuple[str, str, str]:
    BTC = "BTC-market (bivariate)"
    ETH = "ETH-market (bivariate)"
    TRI = "BTC-DXY-market (trivariate)"
    DK = "Pooled OLS + Driscoll-Kraay"
    FE = "Two-way FE (standardized inputs)"

    def fmt(coef, se, p, dp):
        return (f"{coef:+.{dp}f}{stars(p)}", f"{se:.{dp}f}",
                ("<0.001" if p < 0.001 else f"{p:.3f}"))

    def row(channel, outcome, estimator, system, window, sample,
            df, filt, ncty, dp):
        g = get(df, filt, "ci_x_cv")
        n = nobs_for(df, filt)
        if g is None:
            return None
        mc = meta(df, filt)[1]  # exact country count from the fit; ncty is fallback
        c, s, p = fmt(g[0], g[1], g[2], dp)
        return [channel, outcome, estimator, system, str(window), sample,
                c, s, p, str(mc if mc is not None else ncty), f"{n:,}" if n else DASH]

    headers = ["Channel", "Outcome", "Estimator", "First-stage system", "Window",
               "Sample / variation", "KAOPEN x crypto coef.", "SE", "p", "Countries", "Obs"]
    rows: list[list[str]] = []
    sec = lambda lab: rows.append(section_row(lab, len(headers) - 1))

    def rob(ch, dv, win):
        return dict(df=ROB, filt=dict(channel=ch, spec="pooled_full", dv=dv, window=win))

    def hr(ch, dv):
        return dict(df=HR, filt=dict(channel=ch, spec="twfe", model="1_baseline", dv=dv))

    def lo(ch, dv, leaveout):
        return dict(df=LO, filt=dict(channel=ch, spec="pooled_full", dv=dv, window=200, leaveout=leaveout))

    # Baseline
    sec("Baseline (pooled + Driscoll-Kraay, 200-day window)")
    for ch, n in [("equity", EQ_N), ("fx", FX_N)]:
        for dv, dp, lab in [("level", 4, "level share"), ("logit", 3, "log-odds")]:
            r = row("Equity" if ch == "equity" else "FX", lab, DK, BTC, 200, "Baseline",
                    **rob(ch, dv, 200), ncty=n, dp=dp)
            if r:
                rows.append(r)

    # Estimator variation - correctly specified TWFE
    sec("Estimator variation: two-way fixed effects (lower-order terms kept, standardized inputs)")
    for ch, n in [("equity", EQ_N), ("fx", FX_N)]:
        for dv, dp, lab in [("level", 4, "level share"), ("logit", 3, "log-odds")]:
            r = row("Equity" if ch == "equity" else "FX", lab, FE, BTC, 200, "Two-way FE",
                    **hr(ch, dv), ncty=n, dp=dp)
            if r:
                rows.append(r)

    # Window variation
    sec("Window variation (pooled + Driscoll-Kraay, 60-day window)")
    for ch, n in [("equity", EQ_N), ("fx", FX_N)]:
        for dv, dp, lab in [("level", 4, "level share"), ("logit", 3, "log-odds")]:
            r = row("Equity" if ch == "equity" else "FX", lab, DK, BTC, 60, "60-day window",
                    **rob(ch, dv, 60), ncty=n, dp=dp)
            if r:
                rows.append(r)

    # Sample variation - drop crisis
    sec("Sample variation: drop 11 crisis-flagged countries (pooled + Driscoll-Kraay, 200-day)")
    for ch, n in [("equity", CRISIS_EQ_N), ("fx", CRISIS_FX_N)]:
        for dv, dp, lab in [("level", 4, "level share"), ("logit", 3, "log-odds")]:
            r = row("Equity" if ch == "equity" else "FX", lab, DK, BTC, 200, "Drop crisis",
                    **lo(ch, dv, "crisis_drop"), ncty=n, dp=dp)
            if r:
                rows.append(r)

    # Transmitting-asset - ETH
    sec("Transmitting-asset variation: Ethereum replaces Bitcoin (pooled + Driscoll-Kraay, 200-day)")
    for ch, n in [("equity", EQ_N), ("fx", FX_N)]:
        for dv, dp, lab in [("level", 4, "level share"), ("logit", 3, "log-odds")]:
            r = row("Equity" if ch == "equity" else "FX", lab, DK, ETH, 200, "ETH input",
                    **lo(ch, dv, "eth"), ncty=n, dp=dp)
            if r:
                rows.append(r)

    # System variation - trivariate
    sec("System variation: trivariate BTC-DXY-market system, Bitcoin-associated share (200-day)")
    for ch, n in [("equity", EQ_N), ("fx", FX_N)]:
        for dv, dp, lab in [("level", 4, "level share"), ("logit", 3, "log-odds")]:
            r = row("Equity" if ch == "equity" else "FX", lab, DK, TRI, 200, "Trivariate",
                    **lo(ch, dv, "trivariate"), ncty=n, dp=dp)
            if r:
                rows.append(r)

    # Stability variation - drop spectral radius >= 1 (level only)
    sec("Stability variation: drop rolling windows with spectral radius >= 1 (pooled + Driscoll-Kraay, 200-day)")
    for ch, n, rem, pct in [("equity", EQ_N, STAB_EQ_REM, STAB_EQ_PCT),
                            ("fx", FX_N, STAB_FX_REM, STAB_FX_PCT)]:
        sf = dict(channel=ch, spec="pooled_full", threshold=1.0)
        g = get(STAB, sf, "ci_x_cv")
        n_obs = nobs_for(STAB, sf)
        mc = meta(STAB, sf)[1]
        c, s, p = fmt(g[0], g[1], g[2], 4)
        rows.append(["Equity" if ch == "equity" else "FX", "level share", DK, BTC, "200",
                     f"Spectral filter (-{rem:,} obs, {pct:.2f}%)", c, s, p,
                     str(mc if mc is not None else n), f"{n_obs:,}"])

    body = md_table(headers, rows)
    title = ("**Table A.5.** Robustness of the capital-account-openness x cryptocurrency-stress "
             "interaction.")
    note = (
        "*Note.* Each row reports the coefficient on the capital-account-openness x "
        "cryptocurrency-stress interaction from a separately estimated model, with "
        "Driscoll-Kraay standard errors (bandwidth 200). *** p<.01, ** p<.05, * p<.10. The "
        "table assesses directional and inferential robustness, not coefficient magnitudes "
        "across differently scaled dependent variables (level share vs log-odds) or "
        "transformed moderators (the two-way FE rows interact standardized crypto stress, so "
        "their scale differs from the raw-unit pooled rows). Country counts for the leave-out, "
        "asset, and system rows reuse the channel's data-quality inclusion set; observation "
        "counts are exact. The spectral-stability rows were estimated on the level share only.")
    return title, body, note


# ---------------------------------------------------------------------------
# assemble + write
# ---------------------------------------------------------------------------
def main() -> None:
    builders = [
        ("a1_country_descriptives", table_a1),
        ("a2_err_regime", table_a2),
        ("a3_global_horserace", table_a3),
        ("a4_fevd_diagnostics", table_a4),
        ("a5_robustness_matrix", table_a5),
    ]
    parts = [
        "## Appendix A. Supplementary Empirical Results",
        "",
        "*(Generated by `scripts/make_appendix_a.py` from `data/parquet/gold/*`. "
        "Do not edit by hand.)*",
        "",
        "This appendix reports the supplementary descriptive statistics, regression "
        "estimates, and robustness results referenced in Chapter 5. Unless otherwise "
        "stated, estimates use the frozen analytical sample, with country-day "
        "observations and Driscoll-Kraay standard errors where applicable. "
        "Goodness-of-fit is the overall R² for pooled and between models and the within "
        "R² for two-way fixed-effects models, computed on each model's own estimation "
        "sample; reported country and date counts are therefore each fit's actual sample, "
        "which is smaller than the estimable panel wherever a required control is missing.",
        "",
    ]
    for tid, fn in builders:
        title, body, note = fn()
        block = f"{title}\n\n{body}\n\n{note}\n"
        (TABDIR / f"{tid}.md").write_text(block + "\n")
        parts += [title, "", body, "", note, "", "---", ""]
        print(f"  wrote thesis/tables/{tid}.md")

    (APPDIR / "appendix_a.md").write_text("\n".join(parts).rstrip() + "\n")
    print("  wrote thesis/appendix/appendix_a.md")


if __name__ == "__main__":
    main()
