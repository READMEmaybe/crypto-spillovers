#!/usr/bin/env python3
"""Generate Results-chapter tables from the gold outputs (CSV + Markdown + LaTeX).

Reproducible: tables are built from `data/parquet/gold/*` and written to
`thesis/tables/<id>.{csv,md,tex}`. Run:

    uv run python scripts/make_tables.py          # all tables
    uv run python scripts/make_tables.py t1       # one table by id
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import skew, spearmanr

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "parquet" / "gold"
OUT = ROOT / "thesis" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def _write(tid: str, headers: list[str], rows: list[list[str]],
           caption: str, note: str) -> None:
    """Write a table as CSV, Markdown, and a booktabs LaTeX fragment."""
    # CSV
    import csv
    with (OUT / f"{tid}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    # Markdown
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", f"*{note}*"]
    (OUT / f"{tid}.md").write_text("\n".join(md) + "\n")
    # LaTeX (booktabs)
    esc = lambda s: (s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
                     .replace("<", r"$<$").replace(">", r"$>$"))
    tex = [r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{{esc(caption)}}}", rf"\label{{tab:{tid}}}",
           r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule",
           " & ".join(esc(h) for h in headers) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}",
            rf"\par\smallskip\footnotesize {esc(note)}", r"\end{table}"]
    (OUT / f"{tid}.tex").write_text("\n".join(tex) + "\n")
    print(f"  wrote {tid}.csv / .md / .tex")


def t1() -> None:
    """T1 - distribution of the Bitcoin variance share, by channel (frozen sample)."""
    df = pl.read_parquet(GOLD / "step2_analysis.parquet")

    def stats(col: str) -> dict[str, str]:
        s = df[col].drop_nulls().to_numpy()
        ncty = df.filter(pl.col(col).is_not_null())["country_id"].n_unique()
        p = lambda q: np.percentile(s, q)
        pct = lambda x, d=2: f"{x * 100:.{d}f}%"
        return {
            "N (country-days)": f"{len(s):,}",
            "Countries": f"{ncty}",
            "Mean": pct(s.mean()),
            "Median": pct(np.median(s)),
            "SD": pct(s.std()),
            "Min": "<0.01%",
            "P25": pct(p(25)),
            "P75": pct(p(75)),
            "P90": pct(p(90)),
            "P99": pct(p(99)),
            "Max": pct(s.max(), 1),
            "Skewness": f"{skew(s):.1f}",
            "Country-days > 10%": pct((s > 0.10).mean(), 1),
        }

    eq, fx = stats("spill_equity"), stats("spill_fx")
    rows = [[k, eq[k], fx[k]] for k in eq]
    _write(
        "t1_dv_distribution",
        headers=["Statistic", "Equity", "FX"],
        rows=rows,
        caption=("Distribution of the dependent variable: Bitcoin's share of "
                 "domestic forecast-error variance, by channel (frozen sample, "
                 "daily country-level observations)."),
        note=("All per-country means are positive in both channels. The share is a "
              "0-1 rolling generalized-FEVD quantity; see Methodology for construction."),
    )


def t_4_2() -> None:
    """4.2 - most- and least-exposed countries per channel, with openness.
    The five highest and five lowest country-mean Bitcoin variance shares in each
    channel, alongside each country's KAOPEN, showing that high and low exposure
    occur across the openness range (frozen sample)."""
    df = pl.read_parquet(GOLD / "step2_analysis.parquet")
    k = (pl.read_parquet(ROOT / "data" / "parquet" / "kaopen.parquet")
           .filter(pl.col("year") == 2023)
           .select(iso3="iso3", name="country_name", openness="ka_open_normalized")
           .unique(subset="iso3"))
    means = (df.group_by("country_id")
               .agg(eq=pl.col("spill_equity").mean(),
                    fx=pl.col("spill_fx").mean())
               .join(k, left_on="country_id", right_on="iso3", how="left"))

    K = 5
    rows: list[list[str]] = []
    rhos: dict[str, float] = {}
    for ch, clab in [("eq", "Equity"), ("fx", "FX")]:
        sub = (means.filter(pl.col(ch).is_not_null()
                            & pl.col("openness").is_not_null())
                    .sort(ch, descending=True)
                    .with_row_index("rank", 1))
        n = sub.height
        rhos[clab] = spearmanr(sub["openness"], sub[ch]).correlation
        fmt = lambda r: [f"{r['name']} ({r['country_id']})",
                         f"{r['openness']:.2f}", f"{r[ch] * 100:.1f}%"]
        for i, r in enumerate(sub.head(K).iter_rows(named=True)):
            grp = f"{clab} (top {K})" if i == 0 else ""
            rows.append([grp, str(r["rank"]), *fmt(r)])
        for i, r in enumerate(sub.tail(K).sort(ch).iter_rows(named=True)):
            grp = f"{clab} (bottom {K})" if i == 0 else ""
            rows.append([grp, f"{r['rank']}/{n}", *fmt(r)])

    _write(
        "t_4_2_exposure_extremes",
        headers=["Group", "Rank", "Country", "KAOPEN", "Mean BTC share"],
        rows=rows,
        caption=("Most- and least-exposed countries by channel: the five highest "
                 "and five lowest country-mean Bitcoin variance shares, with each "
                 "country's capital-account openness (frozen sample)."),
        note=(f"KAOPEN is the 2023 Chinn-Ito index, normalized to 0 (closed) and 1 "
              f"(open). Mean BTC share is the time-mean of the daily "
              f"generalized-FEVD variance share. Spearman rank correlation between "
              f"mean share and KAOPEN: equity {rhos['Equity']:+.2f}, "
              f"FX {rhos['FX']:+.2f}."),
    )


def t2() -> None:
    """T2 - the openness x crypto-volatility interaction across the estimator ladder.
    4 specs x {equity, FX} x {level, logit}. Level + raw specs from the panel/robustness
    coefs; the standardized two-way FE row from the horse-race baseline (crypto-vol
    z-scored, removing the negative-log sign/scale artifact)."""
    pc = pl.read_parquet(GOLD / "step2_panel_coefs.parquet")
    rb = pl.read_parquet(GOLD / "step2_robustness_coefs.parquet")
    hr = pl.read_parquet(GOLD / "horse_race_coefs.parquet")

    def grab(df, **f):
        out = df
        for k, v in f.items():
            out = out.filter(pl.col(k) == v)
        if out.height == 0:
            return None
        return float(out["coef"][0]), float(out["se"][0]), float(out["pval"][0])

    def cell(r, dec: int) -> str:
        if r is None:
            return "n/a"
        b, se, p = r
        st = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{b:+.{dec}f}{st} ({se:.{dec}f})"

    def level(spec, src):
        if src == "panel":
            return (lambda ch: grab(pc, channel=ch, spec=spec, term="ci_x_cv"))
        return (lambda ch: grab(hr, channel=ch, spec="twfe", model="1_baseline",
                                dv="level", term="ci_x_cv"))

    def logit(spec, src):
        if src == "panel":
            return (lambda ch: grab(rb, channel=ch, spec=spec, term="ci_x_cv",
                                    dv="logit", window=200))
        return (lambda ch: grab(hr, channel=ch, spec="twfe", model="1_baseline",
                                dv="logit", term="ci_x_cv"))

    specs = [("Pooled OLS (no controls)", "pooled_nocontrols", "panel"),
             ("Pooled OLS + Driscoll-Kraay", "pooled_full", "panel"),
             ("Two-way FE (raw crypto-vol)", "twfe", "panel"),
             ("Two-way FE (std. crypto-vol)", "twfe_std", "hr")]
    rows = []
    for lab, spec, src in specs:
        lv, lg = level(spec, src), logit(spec, src)
        rows.append([lab,
                     cell(lv("equity"), 4), cell(lg("equity"), 3),
                     cell(lv("fx"), 4), cell(lg("fx"), 3)])

    beq = grab(pc, channel="equity", spec="between", term="chinn_ito")
    bfx = grab(pc, channel="fx", spec="between", term="chinn_ito")
    _write(
        "t2_interaction_ladder",
        headers=["Specification", "Equity (level)", "Equity (logit)",
                 "FX (level)", "FX (logit)"],
        rows=rows,
        caption=("The openness x crypto-volatility interaction on Bitcoin's variance "
                 "share, across the estimator ladder and two DV transforms "
                 "(frozen sample)."),
        note=(f"Cells report the openness x crypto-volatility interaction coefficient; "
              f"Driscoll-Kraay SE (bandwidth 200) in parentheses; "
              f"*** p<0.01, ** p<0.05, * p<0.10. The hypothesized H2 sign is positive. "
              f"'Two-way FE (raw)' enters the interaction on raw crypto-volatility "
              f"(a negative log); 'std.' z-scores crypto-volatility first, removing a "
              f"sign/scale artifact. Logit DV = winsorized log-odds of the share; level "
              f"and logit coefficients are on different scales. For comparison, the "
              f"between-estimator openness level coefficient is {beq[0]:+.3f} "
              f"(equity, p={beq[2]:.2f}) and {bfx[0]:+.3f} (FX, p={bfx[2]:.2f}), and the "
              f"cross-country correlation of openness with the average share is +0.40 "
              f"(equity) / +0.16 (FX); the level relationship is positive, opposite in "
              f"sign to the interaction. n ranges from 102,000 to 131,000 "
              f"country-days per cell."),
    )


def t2b() -> None:
    """T2b (regime panel) - the DIRECT test of H3: the float x crypto interaction
    (errf_x_cv) on the FX variance share, pure and net of openness, in levels and
    log-odds, with the openness by-regime subsample split and the joint triple-interaction
    model as secondary checks. DK SEs. Level from the panel coefs, log-odds from the
    robustness coefs (200-day window)."""
    pc = pl.read_parquet(GOLD / "step2_panel_coefs.parquet")
    rb = pl.read_parquet(GOLD / "step2_robustness_coefs.parquet")

    def lvl(spec, term):
        r = pc.filter((pl.col("channel") == "fx") & (pl.col("spec") == spec)
                      & (pl.col("term") == term))
        if r.height == 0:
            return None
        return (float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0]),
                int(r["nobs"][0]))

    def lgt(spec, term):
        r = rb.filter((pl.col("channel") == "fx") & (pl.col("spec") == spec)
                      & (pl.col("term") == term) & (pl.col("dv") == "logit")
                      & (pl.col("window") == 200))
        if r.height == 0:
            return None
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    def cell(r, dec):
        if r is None:
            return "n/a"
        b, se, p = r[0], r[1], r[2]
        st = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{b:+.{dec}f}{st} ({se:.{dec}f})"

    def row(label, spec, term):
        return [label, cell(lvl(spec, term), 4), cell(lgt(spec, term), 3)]

    dir_pure = lvl("fx_h3_direct", "errf_x_cv")
    flt = lvl("fx_float", "ci_x_cv")
    peg = lvl("fx_peg", "ci_x_cv")
    joint = lvl("fx_h3", "ci_x_cv")

    rows = [
        ["Direct H3 test: float x crypto", "", ""],
        row("Pure (openness excluded)", "fx_h3_direct", "errf_x_cv"),
        row("Net of openness", "fx_h3_direct_net", "errf_x_cv"),
        ["Openness x crypto, by subsample", "", ""],
        row("All FX (pooled + DK)", "pooled_full", "ci_x_cv"),
        row("Floating regime", "fx_float", "ci_x_cv"),
        row("Managed / pegged regime", "fx_peg", "ci_x_cv"),
        ["Joint regime model (FX triple)", "", ""],
        row("openness x crypto", "fx_h3", "ci_x_cv"),
        row("float x crypto", "fx_h3", "errf_x_cv"),
        row("openness x float", "fx_h3", "ci_x_errf"),
        row("openness x float x crypto", "fx_h3", "ci_x_errf_x_cv"),
    ]
    _write(
        "t2b_regime",
        headers=["Sample / term", "Level coef. (SE)", "Log-odds coef. (SE)"],
        rows=rows,
        caption=("Panel B - the direct test of H3, the float x crypto-volatility "
                 "interaction on the FX variance share, pure and net of openness, with "
                 "the openness-by-regime subsample split and the joint triple-interaction "
                 "model as secondary checks (frozen sample)."),
        note=(f"err_float = 1 floating / 0 managed-or-pegged (IMF AREAER de-facto, per "
              f"country-year), a separate variable from capital openness. Driscoll-Kraay "
              f"SE (bandwidth 200) in parentheses. *** p<0.01, ** p<0.05, * p<0.10. Level "
              f"and log-odds coefficients are on different scales. H3 predicts the direct "
              f"float x crypto interaction is positive (transmission stronger under a "
              f"float); it is instead negative, significant only in the pure log-odds cell, "
              f"and that significance does not survive controlling for openness. The "
              f"subsample split and triple terms confirm the effect is regime-independent. "
              f"n = {dir_pure[3]:,} (direct) / {flt[3]:,} (floating) / {peg[3]:,} "
              f"(managed/peg) / {joint[3]:,} (joint)."),
    )


def t3() -> None:
    """T3 - global-risk horse-race: the openness x crypto interaction before/after
    adding openness x VIX and openness x DXY, plus those global interactions. Pooled,
    four cells (equity/FX x level/logit). Globals standardized (per-SD scale)."""
    hr = pl.read_parquet(GOLD / "horse_race_coefs.parquet")

    def grab(ch, dv, model, term):
        r = hr.filter((pl.col("channel") == ch) & (pl.col("spec") == "pooled")
                      & (pl.col("dv") == dv) & (pl.col("model") == model)
                      & (pl.col("term") == term))
        if r.height == 0:
            return None
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    def cell(t, dec):
        if t is None:
            return "n/a"
        b, se, p = t
        st = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{b:+.{dec}f}{st} ({se:.{dec}f})"

    def row4(label, model, term):
        return [label,
                cell(grab("equity", "level", model, term), 4),
                cell(grab("equity", "logit", model, term), 3),
                cell(grab("fx", "level", model, term), 4),
                cell(grab("fx", "logit", model, term), 3)]

    rows = [
        row4("openness x crypto (baseline)", "1_baseline", "ci_x_cv"),
        row4("openness x crypto (+ globals)", "2_horserace", "ci_x_cv"),
        row4("openness x VIX (+ globals)", "2_horserace", "ci_x_vix"),
        row4("openness x DXY (+ globals)", "2_horserace", "ci_x_dxy"),
    ]
    nb = lambda ch: int(hr.filter(
        (pl.col("channel") == ch) & (pl.col("spec") == "pooled")
        & (pl.col("model") == "2_horserace") & (pl.col("dv") == "level")
        & (pl.col("term") == "ci_x_cv"))["nobs"][0])
    _write(
        "t3_horserace",
        headers=["Term", "Equity (level)", "Equity (logit)", "FX (level)", "FX (logit)"],
        rows=rows,
        caption=("Global-risk horse-race: the openness x crypto-volatility interaction "
                 "before and after adding openness x VIX and openness x DXY, with those "
                 "global-risk interactions (pooled OLS, frozen sample)."),
        note=(f"Global series (crypto-volatility, VIX, DXY) are standardized before "
              f"interacting, so coefficients are on a comparable per-SD scale (the "
              f"baseline row therefore differs in scale from the raw-unit levels in "
              f"Table 4.2). Driscoll-Kraay SE (bandwidth 200) in parentheses; "
              f"*** p<0.01, ** p<0.05, * p<0.10. The first row is the openness x crypto "
              f"interaction with no global interactions; the lower three rows are the "
              f"joint model. Logit DV = winsorized log-odds; level and logit "
              f"coefficients are on different scales. n = {nb('equity'):,} (equity) / "
              f"{nb('fx'):,} (FX)."),
    )


def t3b() -> None:
    """T3b - development horse-race: openness x crypto alone vs after adding
    development x crypto, plus the development interactions. Pooled, four cells
    (equity/FX x level/logit). Standardized moderators (per-SD scale)."""
    dh = pl.read_parquet(GOLD / "dev_horse_race_coefs.parquet")

    def grab(ch, spec, dv, model, term):
        r = dh.filter((pl.col("channel") == ch) & (pl.col("spec") == spec)
                      & (pl.col("dv") == dv) & (pl.col("model") == model)
                      & (pl.col("term") == term))
        if r.height == 0:
            return None
        return (float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0]),
                int(r["nobs"][0]))

    def cell(t, dec):
        if t is None:
            return "n/a"
        b, se, p, _ = t
        st = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{b:+.{dec}f}{st} ({se:.{dec}f})"

    def row4(label, model, term):
        return [label,
                cell(grab("equity", "pooled", "level", model, term), 4),
                cell(grab("equity", "pooled", "logit", model, term), 3),
                cell(grab("fx", "pooled", "level", model, term), 4),
                cell(grab("fx", "pooled", "logit", model, term), 3)]

    rows = [
        row4("openness x crypto (alone)", "1_openness_only", "zopen_x_cv"),
        row4("openness x crypto (+ development)", "2_dev_added", "zopen_x_cv"),
        row4("GDP-pc x crypto (+ development)", "2_dev_added", "zgdp_x_cv"),
        row4("private credit x crypto (+ development)", "2_dev_added", "zcred_x_cv"),
    ]

    # cross-country collinearity + twfe headline numbers for the note
    an = pl.read_parquet(GOLD / "step2_analysis.parquet")
    cs = (an.group_by("country_id")
            .agg(o=pl.col("chinn_ito").mean(), g=pl.col("log_gdp_pc").mean(),
                 c=pl.col("priv_credit_gdp").mean())
            .drop_nulls())
    r_og = float(np.corrcoef(cs["o"].to_numpy(), cs["g"].to_numpy())[0, 1])
    fe_open = grab("fx", "twfe", "level", "2_dev_added", "zopen_x_cv")
    fe_geq = grab("equity", "twfe", "level", "2_dev_added", "zgdp_x_cv")
    fe_gfx = grab("fx", "twfe", "level", "2_dev_added", "zgdp_x_cv")
    nb = grab("equity", "pooled", "level", "2_dev_added", "zopen_x_cv")[3]
    nbfx = grab("fx", "pooled", "level", "2_dev_added", "zopen_x_cv")[3]

    _write(
        "t3b_dev_horserace",
        headers=["Term", "Equity (level)", "Equity (logit)", "FX (level)", "FX (logit)"],
        rows=rows,
        caption=("Development horse-race: the openness x crypto-volatility interaction "
                 "alone and after adding development x crypto-volatility, with the "
                 "development interactions (GDP-per-capita, private credit), pooled OLS "
                 "(frozen sample)."),
        note=(f"Moderators (openness, GDP-pc, private credit) are standardized, so the "
              f"interaction coefficients are per-SD and comparable; Driscoll-Kraay SE "
              f"(bandwidth 200) in parentheses; *** p<0.01, ** p<0.05, * p<0.10. "
              f"'alone' interacts only openness with crypto (development levels still "
              f"controlled); '+ development' adds the development interactions. "
              f"Cross-country r(openness, GDP-pc) = {r_og:.2f}. The pattern holds under "
              f"two-way FE: openness x crypto = {fe_open[0]:+.4f} (FX, p={fe_open[2]:.2f}), "
              f"GDP-pc x crypto = {fe_geq[0]:+.4f} (equity, p={fe_geq[2]:.3f}) / "
              f"{fe_gfx[0]:+.4f} (FX, p={fe_gfx[2]:.2f}). Logit DV = winsorized log-odds; "
              f"level and logit coefficients are on different scales. "
              f"n = {nb:,} (equity) / {nbfx:,} (FX)."),
    )


def t4() -> None:
    """T4 - decomposition of the variance share + an independent volatility check.
    The openness x crypto interaction on each component (log(A)=BTC absolute
    contribution, log(B)=local own-shock, log(localvar)=total local variance from the
    VAR, log(r^2)=independent realized variance from returns, DXY global share),
    pooled & two-way FE, both channels."""
    dc = pl.read_parquet(GOLD / "decomp_coefs.parquet")
    cw = pl.read_parquet(GOLD / "crowdout_coefs.parquet")

    def gd(df, ch, spec, **extra):
        f = ((pl.col("channel") == ch) & (pl.col("spec") == spec)
             & (pl.col("term") == "ci_x_cv"))
        for k, v in extra.items():
            f = f & (pl.col(k) == v)
        r = df.filter(f)
        if r.height == 0:
            return None
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    def cell(t):
        if t is None:
            return "n/a"
        b, se, p = t
        st = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{b:+.3f}{st} ({se:.3f})"

    measures = [
        ("BTC abs. contribution, log(A)", dc, dict(measure="logA")),
        ("local own-shock, log(B)", dc, dict(measure="logB")),
        ("local total variance, log (VAR)", dc, dict(measure="log_localvar")),
        ("independent local volatility, log(r2)", dc, dict(measure="local_rv_indep")),
        ("DXY global share (crowd-out)", cw, dict(measure="global_share",
                                                  **{"global": "dxy"})),
    ]
    rows = []
    for lab, df, extra in measures:
        rows.append([lab,
                     cell(gd(df, "fx", "pooled_full", **extra)),
                     cell(gd(df, "fx", "twfe", **extra)),
                     cell(gd(df, "equity", "pooled_full", **extra)),
                     cell(gd(df, "equity", "twfe", **extra))])

    sc = pl.read_parquet(GOLD / "spillover_components.parquet")
    amax = max(float(sc["A_fx"].max()), float(sc["B_fx"].max()),
               float(sc["localvar_fx"].max()))
    _write(
        "t4_decomposition",
        headers=["Measure (interaction on ...)", "FX (pooled)", "FX (FE)",
                 "Equity (pooled)", "Equity (FE)"],
        rows=rows,
        caption=("Decomposition of the variance share and an independent volatility "
                 "check: the openness x crypto-volatility interaction on each component "
                 "(frozen sample)."),
        note=(f"Cells report the openness x crypto-volatility coefficient (ci_x_cv) on "
              f"each measure; Driscoll-Kraay SE (bandwidth 200) in parentheses; "
              f"*** p<0.01, ** p<0.05, * p<0.10. The share is approximately "
              f"A / localvar, so a falling share with a flat log(A) and a rising "
              f"log(localvar) reflects a growing denominator (local variance), not a "
              f"shrinking BTC contribution. log(A), log(B), log(localvar) and the "
              f"independent log(r2) are winsorized-then-logged; the DXY share is logit. "
              f"'independent local volatility' is the squared daily return (realized "
              f"variance) built straight from returns, the VAR-free counterpart of "
              f"log(localvar). The raw FX VAR components are numerically unstable "
              f"(max magnitude of order 1e{int(np.log10(amax))}); equity components are "
              f"not. "
              f"Both channels on the frozen sample."),
    )


def t5() -> None:
    """T5 - robustness battery for the openness x crypto interaction (ci_x_cv).

    Two axes, both on the frozen sample. Block A varies estimator x window x DV
    transform at a fixed DV (the bivariate BTC share). Block B varies the sample and
    the transmitting input at the primary spec (pooled + Driscoll-Kraay, 200-day):
    drop the 11 crisis countries, swap Bitcoin for Ethereum, and recompute the share
    in a trivariate BTC + DXY + market VAR (net of the dollar factor)."""
    rb = pl.read_parquet(GOLD / "step2_robustness_coefs.parquet")
    hr = pl.read_parquet(GOLD / "horse_race_coefs.parquet")
    lo = (pl.read_parquet(GOLD / "step2_leaveout_coefs.parquet")
          if (GOLD / "step2_leaveout_coefs.parquet").exists() else None)

    # Correctly-specified two-way FE (standardized crypto-vol, lower-order term kept),
    # for the footnote: the raw twfe rows below carry a known negative-log sign artifact.
    _tw = hr.filter((pl.col("channel") == "fx") & (pl.col("spec") == "twfe")
                    & (pl.col("model") == "1_baseline") & (pl.col("dv") == "level")
                    & (pl.col("term") == "ci_x_cv"))
    tw_b, tw_p = float(_tw["coef"][0]), float(_tw["pval"][0])

    def fmt(r, dec: int) -> str:
        if r is None:
            return "n/a"
        b, se, p = r
        st = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return f"{b:+.{dec}f}{st} ({se:.{dec}f})"

    def grid(ch, spec, dv, win):
        r = rb.filter((pl.col("channel") == ch) & (pl.col("spec") == spec)
                      & (pl.col("term") == "ci_x_cv") & (pl.col("dv") == dv)
                      & (pl.col("window") == win))
        return None if r.height == 0 else (float(r["coef"][0]), float(r["se"][0]),
                                           float(r["pval"][0]))

    def lout(ch, tag, dv):
        if lo is None:
            return None
        r = lo.filter((pl.col("channel") == ch) & (pl.col("leaveout") == tag)
                      & (pl.col("term") == "ci_x_cv") & (pl.col("dv") == dv))
        return None if r.height == 0 else (float(r["coef"][0]), float(r["se"][0]),
                                           float(r["pval"][0]))

    def row_grid(label, spec, win):
        return [label,
                fmt(grid("equity", spec, "level", win), 4),
                fmt(grid("equity", spec, "logit", win), 3),
                fmt(grid("fx", spec, "level", win), 4),
                fmt(grid("fx", spec, "logit", win), 3)]

    def row_lout(label, tag):
        return [label,
                fmt(lout("equity", tag, "level"), 4), fmt(lout("equity", tag, "logit"), 3),
                fmt(lout("fx", tag, "level"), 4), fmt(lout("fx", tag, "logit"), 3)]

    div = lambda t: [t, "", "", "", ""]
    rows = [
        div("Estimator x window (bivariate BTC share)"),
        row_grid("Pooled OLS, no controls (200-day)", "pooled_nocontrols", 200),
        row_grid("Pooled OLS, no controls (60-day)", "pooled_nocontrols", 60),
        row_grid("Pooled + Driscoll-Kraay (200-day)", "pooled_full", 200),
        row_grid("Pooled + Driscoll-Kraay (60-day)", "pooled_full", 60),
        row_grid("Two-way fixed effects (200-day)", "twfe", 200),
        row_grid("Two-way fixed effects (60-day)", "twfe", 60),
        div("Leave-outs & alternative inputs (pooled + Driscoll-Kraay, 200-day)"),
        row_lout("Baseline (full frozen sample)", "primary"),
        row_lout("Drop 11 crisis countries", "crisis_drop"),
        row_lout("Ethereum replaces Bitcoin", "eth"),
        row_lout("Trivariate: BTC + DXY + market", "trivariate"),
    ]

    # n range across only the cells SHOWN in the table (3 displayed estimators + leave-outs)
    shown = ["pooled_nocontrols", "pooled_full", "twfe"]
    ns = rb.filter((pl.col("term") == "ci_x_cv")
                   & pl.col("spec").is_in(shown))["nobs"].to_list()
    if lo is not None:
        ns += lo.filter(pl.col("term") == "ci_x_cv")["nobs"].to_list()
    nlo, nhi = min(ns), max(ns)
    eth_note = ("" if (lo is not None and lo.filter(pl.col("leaveout") == "eth").height)
                else " The Ethereum row is pending its Step-1 rebuild and shows n/a.")
    _write(
        "t5_robustness",
        headers=["Specification", "Equity (level)", "Equity (logit)",
                 "FX (level)", "FX (logit)"],
        rows=rows,
        caption=("Robustness of the openness x crypto-volatility interaction across "
                 "window length, dependent-variable transform, estimator, and sample / "
                 "input leave-outs (frozen sample)."),
        note=(f"Each cell is the openness x crypto-volatility coefficient (ci_x_cv); "
              f"Driscoll-Kraay SE (bandwidth 200) in parentheses; *** p<0.01, ** p<0.05, "
              f"* p<0.10. The hypothesized H2 sign is positive. The upper block holds the "
              f"DV fixed (the bivariate Bitcoin variance share) and varies the Step-1 "
              f"window (200/60 days), the DV transform (level / winsorized log-odds), and "
              f"the estimator. The lower block fixes the estimator at pooled + "
              f"Driscoll-Kraay (200-day) and varies the sample or transmitting input: "
              f"dropping the 11 hyperinflation/devaluation-flagged countries, substituting "
              f"Ethereum for Bitcoin, and recomputing the share inside a trivariate "
              f"BTC + DXY + market VAR (the Bitcoin share net of the dollar factor). The "
              f"two-way fixed-effects rows enter the interaction on raw crypto-volatility "
              f"(a negative log) and carry a sign/scale artifact; the correctly specified "
              f"two-way FE (interaction kept with its lower-order term and standardized "
              f"inputs, Table 4.2) is {tw_b:+.4f} (FX, p={tw_p:.3f}). Level and logit "
              f"coefficients are on different scales and are not directly "
              f"comparable.{eth_note} n ranges from {nlo:,} to {nhi:,} country-days."),
    )


TABLES = {"t1": t1, "t_4_2": t_4_2, "t2": t2, "t2b": t2b, "t3": t3, "t3b": t3b,
          "t4": t4, "t5": t5}


def main(argv: list[str]) -> None:
    todo = argv[1:] if len(argv) > 1 else list(TABLES)
    for tid in todo:
        fn = TABLES.get(tid.lower())
        if fn is None:
            print(f"  unknown table '{tid}' (have {list(TABLES)})")
            continue
        print(f"building {tid} ...")
        fn()


if __name__ == "__main__":
    main(sys.argv)
