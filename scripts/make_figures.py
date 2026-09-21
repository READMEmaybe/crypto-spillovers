#!/usr/bin/env python3
"""Generate the thesis Results-chapter figures from the gold outputs.

Reproducible: every figure is built from `data/parquet/gold/*` (and `kaopen.parquet`),
saved as vector PDF + PNG into `thesis/figures/`. Run:

    uv run python scripts/make_figures.py            # all figures
    uv run python scripts/make_figures.py 4_3        # one figure by id

Design notes:
- Consistent system: EQUITY vs FX colors; neutral gray = non-significant.
- Coefficient plots show 95% CIs and a marked zero line.
- Honesty: y-axes are not truncated to inflate effects; CIs always shown where
  a coefficient SE is available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

# ---------------------------------------------------------------------------
# Paths & shared style
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "parquet" / "gold"
PARQ = ROOT / "data" / "parquet"
OUT = ROOT / "thesis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette (color-blind safe Okabe-Ito subset)
C_EQUITY = "#0072B2"   # blue
C_FX = "#D55E00"       # vermillion
C_NEUTRAL = "#999999"  # gray (non-significant)
C_POS = "#0072B2"      # positive coefficient
C_NEG = "#D55E00"      # negative coefficient
C_ACCENT = "#009E73"   # green accent

# Crisis episodes shaded on every time series (start, end, label)
CRISES = [
    ("2020-02-20", "2020-04-15", "COVID-19"),
    ("2022-05-07", "2022-05-18", "Terra/LUNA"),
    ("2022-11-06", "2022-11-20", "FTX"),
]


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 120,
    })


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def add_crisis_shading(ax, label=True) -> None:
    for start, end, name in CRISES:
        ax.axvspan(np.datetime64(start), np.datetime64(end),
                   color="#000000", alpha=0.06, lw=0, zorder=0)
        if label:
            mid = np.datetime64(start) + (np.datetime64(end) - np.datetime64(start)) / 2
            ax.annotate(name, xy=(mid, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7.5, color="#555555")


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def openness_map() -> pl.DataFrame:
    """iso3 -> latest (2023) normalized openness + country name."""
    k = pl.read_parquet(PARQ / "kaopen.parquet")
    return (k.filter(pl.col("year") == 2023)
             .select(iso3="iso3", name="country_name",
                     openness="ka_open_normalized")
             .unique(subset="iso3"))


def coef(df: pl.DataFrame, **filt) -> pl.DataFrame:
    out = df
    for col, val in filt.items():
        out = out.filter(pl.col(col) == val)
    return out


# ---------------------------------------------------------------------------
# 4.1 - The DV over time, by channel (time profile; distribution -> Table T1)
# ---------------------------------------------------------------------------
def fig_4_1():
    """Cross-country monthly mean of the Bitcoin variance share, equity vs FX,
    on the frozen analysis sample. Two stacked panels with independent y-scales
    (the channels differ ~3-4x in level); greyscale, one line per panel.
    COVID-19 shaded; Terra/LUNA and FTX marked to show they carry no
    cross-country spike."""
    df = pl.read_parquet(GOLD / "step2_analysis.parquet")
    m = (df.with_columns(pl.col("date").dt.truncate("1mo").alias("mo"))
           .group_by("mo")
           .agg(eq=pl.col("spill_equity").mean(),
                fx=pl.col("spill_fx").mean(),
                n_eq=pl.col("spill_equity").drop_nulls().len(),
                n_fx=pl.col("spill_fx").drop_nulls().len())
           .sort("mo")
           # require a stable cross-section so the early ramp-in months do not
           # produce a spurious cross-country-mean spike
           .filter((pl.col("n_eq") >= 100) & (pl.col("n_fx") >= 100)))
    d = m["mo"].to_numpy()

    covid = (np.datetime64("2020-02-20"), np.datetime64("2020-04-30"))
    events = [(np.datetime64("2022-05-11"), "Terra/LUNA"),
              (np.datetime64("2022-11-11"), "FTX")]

    fig, (axE, axF) = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True,
                                   gridspec_kw={"hspace": 0.16})
    for ax in (axE, axF):
        ax.axvspan(*covid, color="#000000", alpha=0.07, lw=0, zorder=0)
        for dt, _ in events:
            ax.axvline(dt, color="#888888", lw=0.8, ls=":", zorder=1)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    axE.plot(d, m["eq"].to_numpy(), color="black", lw=1.6)
    axF.plot(d, m["fx"].to_numpy(), color="black", lw=1.6)
    # independent, tight y-scales (the channels differ ~3-4x in level)
    axE.set_ylim(0, float(m["eq"].max()) * 1.18)
    axF.set_ylim(0, float(m["fx"].max()) * 1.18)

    axE.text(0.012, 0.86, "Equity", transform=axE.transAxes,
             fontsize=10, fontweight="bold")
    axF.text(0.012, 0.86, "FX", transform=axF.transAxes,
             fontsize=10, fontweight="bold")

    # event labels above the top panel
    mid_covid = covid[0] + (covid[1] - covid[0]) // 2
    axE.annotate("COVID-19", xy=(mid_covid, 1.03),
                 xycoords=("data", "axes fraction"), ha="center", va="bottom",
                 fontsize=7.5, color="#555555")
    for dt, lab in events:
        axE.annotate(lab, xy=(dt, 1.03), xycoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=7.5, color="#888888")

    axF.xaxis.set_major_locator(mdates.YearLocator())
    axF.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.supylabel("Bitcoin share of domestic forecast-error variance", fontsize=10)
    #fig.suptitle("Bitcoin's variance share over time, by channel "
    #             "(cross-country monthly mean)", x=0.02, ha="left",
    #             fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.90, left=0.10)
    save(fig, "fig_4_1_share_over_time")


def fig_4_1_dist():
    """4.1 distribution - empirical CDF of the Bitcoin variance share, equity vs FX.
    Greyscale (equity solid, FX dashed); x truncated at 50% for legibility (the
    maxima reach 93%/99%, in <0.5% of the right tail)."""
    df = pl.read_parquet(GOLD / "step2_analysis.parquet")
    eq = np.sort(df["spill_equity"].drop_nulls().to_numpy())
    fx = np.sort(df["spill_fx"].drop_nulls().to_numpy())
    ce = np.arange(1, len(eq) + 1) / len(eq)
    cf = np.arange(1, len(fx) + 1) / len(fx)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.5, color="#BBBBBB", lw=0.8, ls=":", zorder=0)
    ax.plot(fx, cf, color="#777777", lw=1.7, ls="--", label="FX")
    ax.plot(eq, ce, color="black", lw=1.7, label="Equity")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.001)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Bitcoin share of domestic forecast-error variance")
    ax.set_ylabel("Cumulative proportion of country-days")
    ax.legend(loc="center right")
    ax.annotate(f"medians: equity {np.median(eq):.1%},  FX {np.median(fx):.1%}",
                xy=(0.5, 0.52), xytext=(0.22, 0.40), fontsize=8.5, color="#444444")
    #ax.set_title("Distribution of the Bitcoin variance share (empirical CDF; "
    #             "x truncated at 50%, maxima 93%/99%)", loc="left", fontsize=10)
    save(fig, "fig_4_1_distribution_ecdf")


# ---------------------------------------------------------------------------
# 4.2 - Cross-country exposure vs openness (does mean exposure rank with openness?)
# ---------------------------------------------------------------------------
def fig_4_2():
    """4.2 - Country-mean Bitcoin variance share against capital-account openness,
    equity and FX. Greyscale two-panel scatter, one point per country. KAOPEN
    takes ~7 discrete values, so x is lightly jittered to de-overlap; a faint OLS
    guide line and the Spearman rank correlation are shown per panel, with a few
    ISO3 anchors labelled to convey the exposure spread at each openness level.
    Frozen analysis sample."""
    df = pl.read_parquet(GOLD / "step2_analysis.parquet")
    om = openness_map()
    means = (df.group_by("country_id")
               .agg(eq=pl.col("spill_equity").mean(),
                    fx=pl.col("spill_fx").mean())
               .join(om, left_on="country_id", right_on="iso3", how="left"))

    panels = [("eq", "Equity", {"AUS", "LTU", "TUN", "MEX", "LAO"}),
              ("fx", "FX", {"AUS", "HKG", "COL", "ZMB", "RUS"})]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3))
    for ax, (ch, title, anchors) in zip(axes, panels):
        sub = means.filter(pl.col(ch).is_not_null()
                           & pl.col("openness").is_not_null())
        op = sub["openness"].to_numpy()
        yv = sub[ch].to_numpy()
        iso = sub["country_id"].to_list()
        rho, p = spearmanr(op, yv)
        xj = op + rng.uniform(-0.018, 0.018, size=len(op))

        slope, intercept = np.polyfit(op, yv, 1)
        gx = np.array([0.0, 1.0])
        ax.plot(gx, intercept + slope * gx, color="#999999", lw=1.0, ls="--",
                zorder=1)
        ax.scatter(xj, yv, s=28, c="black", alpha=0.55, edgecolor="white",
                   linewidth=0.3, zorder=3)
        for x, y, code in zip(xj, yv, iso):
            if code in anchors:
                ax.annotate(code, (x, y), fontsize=7, color="#333333",
                            xytext=(4, 3), textcoords="offset points", zorder=4)

        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, float(yv.max()) * 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x", visible=False)
        ax.annotate(rf"Spearman $\rho$ = {rho:+.2f}{stars(p)}  (n = {len(op)})",
                    xy=(0.04, 0.95), xycoords="axes fraction",
                    ha="left", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#DDDDDD", lw=0.5, alpha=0.85))

    fig.supxlabel("Capital-account openness  (Chinn-Ito KAOPEN, 0 = closed … 1 = open)",
                  fontsize=10)
    fig.supylabel("Country-mean Bitcoin variance share", fontsize=10)
    #fig.suptitle("Country-mean Bitcoin variance share against capital-account "
    #             "openness, by channel", x=0.02, ha="left", fontsize=11,
    #             fontweight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.09, right=0.98, wspace=0.22)
    save(fig, "fig_4_2_exposure_vs_openness")


# ---------------------------------------------------------------------------
# 4.3 - Openness x crypto-volatility interaction (sign/size/significance + magnitude)
# ---------------------------------------------------------------------------
def fig_4_3():
    """4.3 - the openness x crypto-volatility interaction on the variance share.
    LEFT: coefficient forest of ci_x_cv (level DV) across the estimator ladder,
    equity (grey) vs FX (black); filled = p<0.05, hollow = n.s.; 95% CI whiskers.
    The 'std.' two-way FE row standardizes crypto-volatility first (the raw series
    is a negative log, which puts a sign/scale artifact into the interaction).
    RIGHT: implied open-vs-closed gap in crypto's FX share, relative to calm, as
    crypto volatility rises from its 10th to 90th percentile, with a 95% band
    (pooled+DK coefficients). Greyscale."""
    pc = pl.read_parquet(GOLD / "step2_panel_coefs.parquet")
    hr = pl.read_parquet(GOLD / "horse_race_coefs.parquet")
    an = pl.read_parquet(GOLD / "step2_analysis.parquet")

    def grab(df, **f):
        out = df
        for k, v in f.items():
            out = out.filter(pl.col(k) == v)
        if out.height == 0:
            return None
        return float(out["coef"][0]), float(out["se"][0]), float(out["pval"][0])

    C_FX, C_EQ = "black", "#888888"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                   gridspec_kw={"width_ratios": [1.2, 1.0]})

    # ---- LEFT: forest of the interaction across specifications ----
    specs = [("Pooled (no controls)", "pooled_nocontrols", "panel"),
             ("Pooled + Driscoll-Kraay", "pooled_full", "panel"),
             ("Two-way FE (raw)", "twfe", "panel"),
             ("Two-way FE (std.)", "twfe_std", "hr")]
    ypos, ylab = [], []
    for i, (lab, spec, src) in enumerate(specs):
        y = -i
        for ch, col, off in [("equity", C_EQ, 0.16), ("fx", C_FX, -0.16)]:
            r = (grab(pc, channel=ch, spec=spec, term="ci_x_cv") if src == "panel"
                 else grab(hr, channel=ch, spec="twfe", model="1_baseline",
                           dv="level", term="ci_x_cv"))
            if r is None:
                continue
            b, se, p = r
            sig = p < 0.05
            axL.errorbar(b, y + off, xerr=1.96 * se, fmt="o", color=col, ms=6.5,
                         mfc=col if sig else "white", mec=col, mew=1.2,
                         ecolor=col, elinewidth=1.0, capsize=2.5, zorder=3)
        ypos.append(y); ylab.append(lab)
    axL.axvline(0, color="#444444", lw=1.0, ls="--", zorder=1)
    axL.set_yticks(ypos); axL.set_yticklabels(ylab, fontsize=8.5)
    axL.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
    axL.set_xlabel("openness × crypto-volatility coefficient  (level, 95% CI)",
                   fontsize=8.5)
    axL.set_title("Interaction across specifications", loc="left", fontsize=10)
    axL.grid(axis="y", visible=False)
    leg = [Line2D([], [], marker="o", color=C_FX, ls="", mfc=C_FX, label="FX"),
           Line2D([], [], marker="o", color=C_EQ, ls="", mfc=C_EQ, label="Equity"),
           Line2D([], [], marker="o", color="#444444", ls="", mfc="white",
                  label="hollow = n.s.")]
    axL.legend(handles=leg, loc="upper right", fontsize=7.5, handletextpad=0.3)

    # ---- RIGHT: open-vs-closed FX gap vs crypto stress (relative to calm) ----
    cv = an["crypto_vol"].drop_nulls().to_numpy()
    p10, p50, p90 = np.percentile(cv, [10, 50, 90])
    zs = np.linspace(p10, p90, 60)
    b, se, _ = grab(pc, channel="fx", spec="pooled_full", term="ci_x_cv")
    gap = b * (zs - p10) * 100
    band = 1.96 * se * np.abs(zs - p10) * 100
    axR.axhline(0, color="#444444", lw=0.8, ls="--", zorder=1)
    axR.fill_between(zs, gap - band, gap + band, color="#000000", alpha=0.10,
                     lw=0, zorder=2)
    axR.plot(zs, gap, color="black", lw=1.9, zorder=3)
    for z in (p10, p50, p90):
        val = b * (z - p10) * 100
        val = 0.0 if abs(val) < 1e-9 else val  # avoid IEEE "-0.00" at calm
        axR.plot(z, val, "o", color="black", ms=4, zorder=4)
        axR.annotate(f"{val:+.2f}", (z, val), fontsize=7.5, ha="center",
                     va="bottom", xytext=(0, 6), textcoords="offset points")
    axR.set_xticks([p10, p50, p90])
    axR.set_xticklabels(["calm\n(p10)", "median", "stress\n(p90)"], fontsize=8)
    axR.set_xlim(p10, p90)
    axR.set_ylabel("open-vs-closed gap in crypto's FX share\n(relative to calm, pp)",
                   fontsize=8.5)
    axR.set_xlabel("crypto volatility", fontsize=9)
    axR.set_title("Implied magnitude (FX, pooled+DK)", loc="left", fontsize=10)

    #fig.suptitle("The openness × crypto-volatility interaction on Bitcoin's "
    #             "variance share", x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.22, right=0.98, wspace=0.50)
    save(fig, "fig_4_3_interaction")


# ---------------------------------------------------------------------------
# 4.4 - The FX interaction across exchange-rate regimes (H3)
# ---------------------------------------------------------------------------
def _h3_forest(rows, title, fname, *, sep_ys=(), figsize=(9.4, 3.8), ylim=None):
    """Greyscale two-panel (level | log-odds) forest shared by the H3 figures.

    rows: list of (label, y, lvl, logit) where lvl/logit are (coef, se, p) or None.
    Filled marker = p < 0.05, hollow = n.s.; 95% CIs; dashed zero line. sep_ys draws
    faint horizontal separators between row groups; the two panels share the y axis."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    ypos = [r[1] for r in rows]
    ylab = [r[0] for r in rows]
    for ax, idx, xl, dec in ((axes[0], 2, "level coefficient  (95% CI)", 4),
                             (axes[1], 3, "log-odds coefficient  (95% CI)", 3)):
        ax.axvline(0, color="#444444", lw=1.0, ls="--", zorder=1)
        for y_sep in sep_ys:
            ax.axhline(y_sep, color="#DDDDDD", lw=0.8, zorder=0)
        for lab, y, lvl_t, lgt_t in rows:
            cell = lvl_t if idx == 2 else lgt_t
            if cell is None:
                continue
            b, se, p = cell
            sig = p < 0.05
            ax.errorbar(b, y, xerr=1.96 * se, fmt="o", color="black", ms=6.5,
                        mfc="black" if sig else "white", mec="black", mew=1.3,
                        ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
            ax.annotate(f"{b:+.{dec}f}{stars(p)}", (b, y), fontsize=7,
                        ha="center", va="bottom", xytext=(0, 7),
                        textcoords="offset points")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xl, fontsize=8.5)
        ax.margins(x=0.20)
        ax.set_title("Level" if idx == 2 else "Log-odds", fontsize=9.5,
                     fontweight="normal")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(ylab, fontsize=8.5)
    if ylim:
        axes[0].set_ylim(*ylim)
    leg = [Line2D([], [], marker="o", color="black", ls="", mfc="black",
                  label="p < 0.05"),
           Line2D([], [], marker="o", color="black", ls="", mfc="white",
                  label="n.s.")]
    axes[1].legend(handles=leg, loc="upper right", fontsize=8, handletextpad=0.3)
    #fig.suptitle(title, x=0.012, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, fname)


def fig_4_4a():
    """4.4a - the DIRECT test of H3 on the FX channel: the float x crypto interaction
    (errf_x_cv), pure and net of openness, in levels and log-odds, with the equity
    channel as a placebo (the regime should not operate there). H3 predicts a positive
    coefficient; it is negative throughout, significant only in the FX pure log-odds
    cell, and that significance does not survive controlling for openness. Source:
    h3_regime_direct_coefs (pooled + Driscoll-Kraay)."""
    hr = pl.read_parquet(GOLD / "h3_regime_direct_coefs.parquet")

    def g(ch, spec, dv):
        r = hr.filter((pl.col("channel") == ch) & (pl.col("spec") == spec)
                      & (pl.col("model") == "pooled") & (pl.col("dv") == dv)
                      & (pl.col("term") == "errf_x_cv"))
        if r.height == 0:
            return None
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    rows = [
        ("FX, pure", 0.0, g("fx", "1_pure", "level"), g("fx", "1_pure", "logit")),
        ("FX, net of openness", -1.0, g("fx", "2_net_of_openness", "level"),
         g("fx", "2_net_of_openness", "logit")),
        ("Equity placebo, pure", -2.3, g("equity", "1_pure", "level"),
         g("equity", "1_pure", "logit")),
        ("Equity placebo, net", -3.3, g("equity", "2_net_of_openness", "level"),
         g("equity", "2_net_of_openness", "logit")),
    ]
    _h3_forest(rows,
               "The direct test of H3: float × crypto on the FX variance share",
               "fig_4_4a_direct", sep_ys=(-1.65,), ylim=(-4.05, 0.85))


def fig_4_4b():
    """4.4b - heterogeneity: does the openness moderation of crypto -> FX transmission
    differ by exchange-rate regime? Shows the openness x crypto interaction on the full
    FX sample and split into floating vs managed/pegged subsamples, plus the joint
    openness x float x crypto triple term (the regime-difference statistic). Near-identical
    subsample slopes and a null triple imply the moderation is regime-independent. Level
    from step2_panel_coefs, log-odds from step2_robustness_coefs (200-day window)."""
    pc = pl.read_parquet(GOLD / "step2_panel_coefs.parquet")
    rb = pl.read_parquet(GOLD / "step2_robustness_coefs.parquet")

    def lvl(spec, term):
        r = pc.filter((pl.col("channel") == "fx") & (pl.col("spec") == spec)
                      & (pl.col("term") == term))
        if r.height == 0:
            return None
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    def lgt(spec, term):
        r = rb.filter((pl.col("channel") == "fx") & (pl.col("spec") == spec)
                      & (pl.col("term") == term) & (pl.col("dv") == "logit")
                      & (pl.col("window") == 200))
        if r.height == 0:
            return None
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    rows = [
        ("All FX", 0.0, lvl("pooled_full", "ci_x_cv"), lgt("pooled_full", "ci_x_cv")),
        ("Floating regime", -1.0, lvl("fx_float", "ci_x_cv"),
         lgt("fx_float", "ci_x_cv")),
        ("Managed / pegged regime", -2.0, lvl("fx_peg", "ci_x_cv"),
         lgt("fx_peg", "ci_x_cv")),
        ("openness × float × crypto", -3.3, lvl("fx_h3", "ci_x_errf_x_cv"),
         lgt("fx_h3", "ci_x_errf_x_cv")),
    ]
    _h3_forest(rows,
               "Heterogeneity: openness × crypto moderation by exchange-rate regime",
               "fig_4_4b_hetero", sep_ys=(-2.65,), ylim=(-4.05, 0.85))


# ---------------------------------------------------------------------------
# 4.5 - Global-risk horse-race: openness x crypto beside openness x VIX / x DXY
# ---------------------------------------------------------------------------
def fig_4_5():
    """4.5 - global-risk horse-race (logit DV, pooled): the openness x crypto
    interaction beside openness x VIX and openness x DXY. Two panels (equity, FX);
    each a forest of the three interaction terms with 95% CIs and a zero line;
    filled = p<0.05, hollow = n.s. In equity the crypto and VIX interactions fall on
    opposite sides of zero. Greyscale."""
    hr = pl.read_parquet(GOLD / "horse_race_coefs.parquet")

    def grab(ch, term):
        r = hr.filter((pl.col("channel") == ch) & (pl.col("spec") == "pooled")
                      & (pl.col("model") == "2_horserace") & (pl.col("dv") == "logit")
                      & (pl.col("term") == term))
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    terms = [("ci_x_cv", "openness × crypto"),
             ("ci_x_vix", "openness × VIX"),
             ("ci_x_dxy", "openness × DXY")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharex=True)
    for ax, ch, title in zip(axes, ["equity", "fx"], ["Equity", "FX"]):
        for i, (term, _) in enumerate(terms):
            b, se, p = grab(ch, term)
            sig = p < 0.05
            ax.errorbar(b, -i, xerr=1.96 * se, fmt="o", color="black", ms=7,
                        mfc="black" if sig else "white", mec="black", mew=1.3,
                        ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
            ax.annotate(f"{b:+.2f}{stars(p)}", (b, -i), fontsize=8, ha="center",
                        va="bottom", xytext=(0, 7), textcoords="offset points")
        ax.axvline(0, color="#444444", lw=1.0, ls="--", zorder=1)
        ax.set_yticks([0, -1, -2])
        ax.set_yticklabels([t[1] for t in terms], fontsize=8.5)
        ax.set_ylim(-2.6, 0.7)
        ax.set_xlim(-0.72, 0.68)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])  # shared rows; label once on the left
    leg = [Line2D([], [], marker="o", color="black", ls="", mfc="black",
                  label="p < 0.05"),
           Line2D([], [], marker="o", color="black", ls="", mfc="white", label="n.s.")]
    axes[0].legend(handles=leg, loc="lower left", fontsize=8, handletextpad=0.3)
    fig.supxlabel("openness × global-factor interaction coefficient  (logit DV, 95% CI)",
                  fontsize=9)
    #fig.suptitle("Openness × crypto-volatility beside openness × VIX and openness × DXY "
    #             "(horse-race, logit DV)", x=0.02, ha="left", fontsize=11,
    #             fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.20, left=0.16, right=0.98, wspace=0.08)
    save(fig, "fig_4_5_horserace")


# ---------------------------------------------------------------------------
# 4.6 - Development horse-race: openness x crypto vs development x crypto
# ---------------------------------------------------------------------------
def fig_4_6():
    """4.6 - competing moderators (development horse-race, pooled, level DV): the
    openness x crypto interaction alone vs after development x crypto is added, beside
    the development interactions (GDP-pc, private credit). Standardized moderators, so
    coefficients are per-SD and comparable. Two panels (equity, FX); filled = p<0.05,
    hollow = n.s.; 95% CIs; zero line. Greyscale."""
    dh = pl.read_parquet(GOLD / "dev_horse_race_coefs.parquet")

    def grab(ch, model, term):
        r = dh.filter((pl.col("channel") == ch) & (pl.col("spec") == "pooled")
                      & (pl.col("dv") == "level") & (pl.col("model") == model)
                      & (pl.col("term") == term))
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    rows = [("openness × crypto (alone)", "1_openness_only", "zopen_x_cv"),
            ("openness × crypto (+dev)", "2_dev_added", "zopen_x_cv"),
            ("GDP × crypto (+dev)", "2_dev_added", "zgdp_x_cv"),
            ("credit × crypto (+dev)", "2_dev_added", "zcred_x_cv")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
    for ax, ch, title in zip(axes, ["equity", "fx"], ["Equity", "FX"]):
        for i, (lab, model, term) in enumerate(rows):
            b, se, p = grab(ch, model, term)
            sig = p < 0.05
            ax.errorbar(b, -i, xerr=1.96 * se, fmt="o", color="black", ms=7,
                        mfc="black" if sig else "white", mec="black", mew=1.3,
                        ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
            ax.annotate(f"{b:+.4f}{stars(p)}", (b, -i), fontsize=7.5, ha="center",
                        va="bottom", xytext=(0, 7), textcoords="offset points")
        ax.axvline(0, color="#444444", lw=1.0, ls="--", zorder=1)
        ax.set_yticks([0, -1, -2, -3])
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.set_ylim(-3.6, 0.7)
        ax.set_xlim(-0.0085, 0.0055)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])  # shared rows; label once on the left
    leg = [Line2D([], [], marker="o", color="black", ls="", mfc="black",
                  label="p < 0.05"),
           Line2D([], [], marker="o", color="black", ls="", mfc="white", label="n.s.")]
    axes[1].legend(handles=leg, loc="upper right", fontsize=8, handletextpad=0.3)
    # collinearity on the dev horse-race sample (all three moderators present)
    cs = (pl.read_parquet(GOLD / "step2_analysis.parquet").group_by("country_id")
          .agg(o=pl.col("chinn_ito").mean(), g=pl.col("log_gdp_pc").mean(),
               c=pl.col("priv_credit_gdp").mean()).drop_nulls())
    r_og = float(np.corrcoef(cs["o"].to_numpy(), cs["g"].to_numpy())[0, 1])
    fig.supxlabel("standardized moderator × crypto-volatility coefficient  "
                  "(level DV, 95% CI)", fontsize=8.5)
    #fig.suptitle(f"Competing moderators: openness × crypto vs development × crypto  "
    #             f"(openness-GDP r = {r_og:.2f})", x=0.02, ha="left", fontsize=10.5,
    #             fontweight="bold")
    fig.subplots_adjust(top=0.87, bottom=0.20, left=0.23, right=0.98, wspace=0.08)
    save(fig, "fig_4_6_dev_horserace")


# ---------------------------------------------------------------------------
# 4.7 - Decomposition of the FX share + an independent volatility check
# ---------------------------------------------------------------------------
def fig_4_7():
    """4.7 - decomposition of the FX variance share. Forest of the openness x crypto
    interaction on each component: BTC's absolute contribution log(A), local own-shock
    log(B), local total variance log(localvar) from the VAR, an independent local
    volatility log(r^2) built straight from returns, and the trivariate DXY global
    share. Pooled (grey) vs two-way FE (black); filled = p<0.05, hollow = n.s.; 95% CI.
    Greyscale. The independent log(r^2) is the VAR-free counterpart of log(localvar)."""
    dc = pl.read_parquet(GOLD / "decomp_coefs.parquet")
    cw = pl.read_parquet(GOLD / "crowdout_coefs.parquet")

    def gd(df, spec, **extra):
        f = ((pl.col("channel") == "fx") & (pl.col("spec") == spec)
             & (pl.col("term") == "ci_x_cv"))
        for k, v in extra.items():
            f = f & (pl.col(k) == v)
        r = df.filter(f)
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    rows = [("BTC abs. contribution  log(A)", dc, dict(measure="logA")),
            ("local own-shock  log(B)", dc, dict(measure="logB")),
            ("local total variance  log(VAR)", dc, dict(measure="log_localvar")),
            ("independent local vol  log(r²)", dc, dict(measure="local_rv_indep")),
            ("DXY global share", cw, dict(measure="global_share", **{"global": "dxy"}))]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.axvline(0, color="#444444", lw=1.0, ls="--", zorder=1)
    ypos, ylab = [], []
    for i, (lab, df, extra) in enumerate(rows):
        for spec, col, off in [("pooled_full", "#888888", 0.16), ("twfe", "black", -0.16)]:
            b, se, p = gd(df, spec, **extra)
            sig = p < 0.05
            ax.errorbar(b, -i + off, xerr=1.96 * se, fmt="o", color=col, ms=6.5,
                        mfc=col if sig else "white", mec=col, mew=1.2, ecolor=col,
                        elinewidth=1.0, capsize=2.5, zorder=3)
        ypos.append(-i); ylab.append(lab)
    ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=8.5)
    ax.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("openness × crypto-volatility coefficient  (FX; log/logit DV, 95% CI)",
                  fontsize=8.5)
    leg = [Line2D([], [], marker="o", color="black", ls="", mfc="black",
                  label="two-way FE"),
           Line2D([], [], marker="o", color="#888888", ls="", mfc="#888888",
                  label="pooled"),
           Line2D([], [], marker="o", color="#444444", ls="", mfc="white",
                  label="hollow = n.s.")]
    ax.legend(handles=leg, loc="lower right", fontsize=7.5, handletextpad=0.3)
    #ax.set_title("Decomposition of the FX variance share, with an independent "
    #             "volatility check", loc="left", x=0.0, fontsize=10, fontweight="bold")
    fig.subplots_adjust(top=0.91, bottom=0.13, left=0.31, right=0.97)
    save(fig, "fig_4_7_decomposition")


# ---------------------------------------------------------------------------
# 4.8 - Robustness across window, transform, estimator, and leave-outs
# ---------------------------------------------------------------------------
def fig_4_8():
    """4.8 - robustness of the openness x crypto interaction across window x transform
    x estimator. 2x2 grid: rows = channel (FX top, Equity bottom), columns = DV
    transform (Level | Logit; different x-scales, shared within a column). Each panel
    plots ci_x_cv for the three estimators (pooled no-controls / pooled+DK / two-way FE)
    at two Step-1 windows -- 200-day (black) and 60-day (grey); filled = p<0.05,
    hollow = n.s.; 95% CI whiskers; dashed zero line. Frozen sample. Greyscale."""
    rb = pl.read_parquet(GOLD / "step2_robustness_coefs.parquet")

    def grab(ch, spec, dv, win):
        r = rb.filter((pl.col("channel") == ch) & (pl.col("spec") == spec)
                      & (pl.col("term") == "ci_x_cv") & (pl.col("dv") == dv)
                      & (pl.col("window") == win))
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    estimators = [("Pooled\n(no controls)", "pooled_nocontrols"),
                  ("Pooled + DK", "pooled_full"),
                  ("Two-way FE", "twfe")]
    windows = [(200, "black", 0.17), (60, "#8A8A8A", -0.17)]
    channels = [("fx", "FX"), ("equity", "Equity")]
    transforms = [("level", "Level share"), ("logit", "Log-odds-transformed share")]

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), sharex="col", sharey=True)
    for ri, (ch, chlab) in enumerate(channels):
        for ci, (dv, dvlab) in enumerate(transforms):
            ax = axes[ri][ci]
            ax.axvline(0, color="#444444", lw=1.0, ls="--", zorder=1)
            for ei, (elab, spec) in enumerate(estimators):
                for win, col, off in windows:
                    b, se, p = grab(ch, spec, dv, win)
                    sig = p < 0.05
                    ax.errorbar(b, -ei + off, xerr=1.96 * se, fmt="o", color=col, ms=6,
                                mfc=col if sig else "white", mec=col, mew=1.2,
                                ecolor=col, elinewidth=1.0, capsize=2.5, zorder=3)
            ax.set_yticks([0, -1, -2])
            ax.set_ylim(-2.6, 0.6)
            ax.grid(axis="y", visible=False)
            if ci == 0:  # label estimators on the left column only
                ax.set_yticklabels([e[0] for e in estimators], fontsize=8.5)
                ax.tick_params(axis="y", labelleft=True)
                ax.set_ylabel(chlab, fontsize=11.5, fontweight="bold", labelpad=12)
            else:        # hide (do NOT clear) labels on the shared right column
                ax.tick_params(axis="y", labelleft=False)
            if ri == 0:
                ax.set_title(dvlab, fontsize=10, loc="center")
    axes[1][0].set_xlabel("openness × crypto-volatility coefficient  (level, 95% CI)",
                          fontsize=8.5)
    axes[1][1].set_xlabel("openness × crypto-volatility coefficient  (logit, 95% CI)",
                          fontsize=8.5)
    leg = [Line2D([], [], marker="o", color="black", ls="", mfc="black",
                  label="200-day window"),
           Line2D([], [], marker="o", color="#8A8A8A", ls="", mfc="#8A8A8A",
                  label="60-day window"),
           Line2D([], [], marker="o", color="#444444", ls="", mfc="white",
                  label="hollow = n.s.")]
    # FX coefficients are all negative, so the right side of the FX-Level panel is
    # empty -- the least-cluttered home for the legend.
    axes[0][0].legend(handles=leg, loc="upper right", fontsize=7.5, handletextpad=0.3,
                      labelspacing=0.3)
    # fig.suptitle("Robustness of the openness × crypto-volatility interaction across "
    #             "window, transform, and estimator", x=0.02, ha="left", fontsize=11,
    #             fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.205, right=0.98,
                        hspace=0.20, wspace=0.07)
    save(fig, "fig_4_8_robustness")


def fig_4_8b():
    """4.8 (leave-outs) - the FX openness x crypto interaction at the primary spec
    (pooled + DK, 200-day, frozen) across sample / input leave-outs: baseline, drop the
    11 crisis countries, Ethereum for Bitcoin, and the trivariate BTC + DXY + market
    share (net of the dollar factor). Two panels (Level | Logit, different x-scales);
    filled = p<0.05, hollow = n.s.; 95% CI; dashed zero line. Greyscale."""
    lo = pl.read_parquet(GOLD / "step2_leaveout_coefs.parquet")

    def grab(tag, dv):
        r = lo.filter((pl.col("channel") == "fx") & (pl.col("leaveout") == tag)
                      & (pl.col("term") == "ci_x_cv") & (pl.col("dv") == dv))
        if r.height == 0:
            return None
        return float(r["coef"][0]), float(r["se"][0]), float(r["pval"][0])

    rows = [("Baseline (frozen sample)", "primary"),
            ("Drop 11 crisis countries", "crisis_drop"),
            ("Ethereum for Bitcoin", "eth"),
            ("Trivariate (BTC + DXY)", "trivariate")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for ax, (dv, dvlab, dec) in zip(axes, [("level", "Level share", 4),
                                           ("logit", "Log-odds-transformed share", 2)]):
        ax.axvline(0, color="#444444", lw=1.0, ls="--", zorder=1)
        for i, (lab, tag) in enumerate(rows):
            g = grab(tag, dv)
            if g is None:
                ax.annotate("pending rebuild", (0, -i), fontsize=7, style="italic",
                            color="#999999", ha="center", va="center")
                continue
            b, se, p = g
            sig = p < 0.05
            ax.errorbar(b, -i, xerr=1.96 * se, fmt="o", color="black", ms=7,
                        mfc="black" if sig else "white", mec="black", mew=1.3,
                        ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
            ax.annotate(f"{b:+.{dec}f}{stars(p)}", (b, -i), fontsize=7.5, ha="center",
                        va="bottom", xytext=(0, 7), textcoords="offset points")
        ax.set_yticks([0, -1, -2, -3])
        ax.set_ylim(-3.6, 0.7)
        ax.set_yticklabels([r[0] for r in rows] if dv == "level" else [], fontsize=8.5)
        ax.set_title(dvlab, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"openness × crypto-volatility coefficient  ({dv}, 95% CI)",
                      fontsize=8.5)
    leg = [Line2D([], [], marker="o", color="black", ls="", mfc="black", label="p < 0.05"),
           Line2D([], [], marker="o", color="black", ls="", mfc="white", label="n.s.")]
    axes[1].legend(handles=leg, loc="lower right", fontsize=8, handletextpad=0.3)
    #fig.suptitle("FX openness × crypto-volatility interaction across sample and input "
    #             "leave-outs", x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.21, right=0.98, wspace=0.08)
    save(fig, "fig_4_8b_leaveouts")


def fig_meth_audit(path: str | Path | None = None):
    """Plot the country-channel sample audit in an APA-style monochrome design."""
    path = Path(path) if path is not None else GOLD / "sample_membership.parquet"
    audit = pl.read_parquet(path)
    stale_caps = {"equity": 0.20, "fx": 0.40}
    panels = [("equity", "A  Equity"), ("fx", "B  Foreign exchange")]
    label_offsets = {
        ("equity", "ARE"): (-2, 6), ("equity", "KHM"): (-4, -7),
        ("equity", "IRQ"): (5, 6), ("equity", "IRN"): (0, -7),
        ("equity", "GHA"): (0, -7), ("equity", "ECU"): (0, 7),
        ("equity", "ZMB"): (-10, -7), ("equity", "PAN"): (10, -7),
        ("equity", "BWA"): (0, 7), ("equity", "SVK"): (0, -7),
        ("equity", "DZA"): (0, 7), ("equity", "MDV"): (0, -7),
        ("equity", "KGZ"): (0, 7), ("equity", "UKR"): (0, -7),
        ("fx", "HRV"): (0, 7), ("fx", "QAT"): (0, 7),
        ("fx", "MDV"): (0, -7), ("fx", "ARE"): (0, 7),
        ("fx", "JOR"): (0, -7),
    }

    # A shared scale preserves comparability and, importantly, does not hide the
    # few observed coverage values above one.
    coverage_max = max(1.05, float(audit["coverage"].max()) + 0.04)
    coverage_min = max(0.0, float(audit["coverage"].min()) - 0.08)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (channel, panel_label) in zip(axes, panels):
        d = audit.filter(pl.col("channel") == channel)
        kept = d.filter(pl.col("passes"))
        dropped = d.filter(~pl.col("passes"))
        cap = stale_caps[channel]

        # Very light shading identifies the region satisfying both rules while
        # keeping the figure fully legible in grayscale or black-and-white print.
        ax.fill_between([0, cap], 0.80, coverage_max,
                        color="#F0F0F0", lw=0, zorder=0)
        ax.scatter(kept["stale_frac"], kept["coverage"], s=27,
                   marker="o", facecolors="white", edgecolors="#4D4D4D",
                   linewidths=0.8, label="Included", zorder=2)
        ax.scatter(dropped["stale_frac"], dropped["coverage"], s=32,
                   marker="^", facecolors="#1A1A1A", edgecolors="#1A1A1A",
                   linewidths=0.6, label="Excluded", zorder=3)

        ax.axhline(0.80, ls=(0, (4, 3)), color="#666666", lw=0.9, zorder=1)
        ax.axvline(cap, ls=(0, (4, 3)), color="#666666", lw=0.9, zorder=1)
        ax.text(0.99, 0.80, "80% minimum coverage", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=7, color="#4D4D4D")
        ax.text(cap, coverage_min + 0.01 * (coverage_max - coverage_min),
                f"{cap:.0%} maximum staleness", rotation=90,
                ha="right", va="bottom", fontsize=7, color="#4D4D4D")

        # Small, fixed offsets keep the dense labels deterministic and separable.
        for row in dropped.iter_rows(named=True):
            dx, dy = label_offsets.get((channel, row["country_id"]), (0, 6))
            ax.annotate(row["country_id"],
                        (row["stale_frac"], row["coverage"]),
                        xytext=(dx, dy), textcoords="offset points",
                        ha="center", va="bottom" if dy >= 0 else "top",
                        fontsize=6.5, color="#111111", zorder=4,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        ax.set_title(panel_label, loc="left", fontsize=10, fontweight="normal", pad=8)
        ax.set_xlabel("Stale observations")
        ax.set_xlim(-0.02, 1.00)
        ax.set_ylim(coverage_min, coverage_max)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.20))
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#E3E3E3", linewidth=0.6)
        ax.tick_params(labelsize=8.5)

    axes[0].set_ylabel("Available observations (coverage)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, handletextpad=0.5, columnspacing=1.8,
               bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.19, wspace=0.08)
    save(fig, "fig_meth_audit")


def fig_meth_pipeline():
    """Draw the research workflow from source inputs to analytical datasets."""
    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    C_INPUT = "#F2F0ED"
    C_CONTROL = "#E3EBEF"
    C_OUTPUT = "#E5ECE8"
    C_LINE = "#3D4548"

    def box(x, y, w, h, title, body, fill, *, body_size=7.4,
            title_size=8.0, dashed=False, linewidth=0.9):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor=fill, edgecolor=C_LINE, linewidth=linewidth,
            linestyle=(0, (4, 3)) if dashed else "solid",
            transform=ax.transAxes, zorder=3)
        ax.add_patch(patch)
        title_y = y + h - 0.031
        ax.text(x + 0.012, title_y, title, transform=ax.transAxes,
                ha="left", va="center", fontsize=title_size,
                fontweight="bold", color="#202426", zorder=4)
        rule_y = y + h - 0.054
        ax.plot([x + 0.012, x + w - 0.012], [rule_y, rule_y],
                transform=ax.transAxes, color="#A5ACAE", lw=0.55, zorder=4)
        ax.text(x + w / 2, y + (h - 0.054) / 2, body,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=body_size, linespacing=1.25, color="#202426", zorder=4)
        return (x, y, w, h)

    def arrow(start, end, *, dashed=False, connection="arc3", zorder=2):
        ax.add_patch(FancyArrowPatch(
            start, end, transform=ax.transAxes, arrowstyle="-|>",
            mutation_scale=9, linewidth=0.9, color=C_LINE,
            linestyle=(0, (4, 3)) if dashed else "solid",
            connectionstyle=connection, shrinkA=0, shrinkB=0, zorder=zorder))

    # Source inputs
    input_boxes = [
        box(0.018, 0.800, 0.205, 0.145, "Cryptocurrency market data",
            "Binance BTCUSDT and ETHUSDT\nOne-minute OHLCV", C_INPUT),
        box(0.018, 0.625, 0.205, 0.145, "Domestic market data",
            "Equity indices · FX rates\nDaily close prices", C_INPUT),
        box(0.018, 0.430, 0.205, 0.165, "Institutional and development data",
            "KAOPEN · de facto regime · crypto restrictions\nGDP per capita · private credit",
            C_INPUT, body_size=7.0, title_size=7.5),
        box(0.018, 0.265, 0.205, 0.135, "Global conditions",
            "VIX · DXY · S&P 500 · Brent\nDaily observations", C_INPUT),
    ]

    # All source groups enter the same cleaning and harmonisation stage.
    collector_x = 0.257
    centers = [y + h / 2 for _, y, _, h in input_boxes]
    for x, y, w, h in input_boxes:
        ax.plot([x + w, collector_x], [y + h / 2, y + h / 2],
                transform=ax.transAxes, color=C_LINE, lw=0.75, zorder=1)
    ax.plot([collector_x, collector_x], [min(centers), max(centers)],
            transform=ax.transAxes, color=C_LINE, lw=0.75, zorder=1)

    cleaning = box(
        0.292, 0.620, 0.225, 0.245, "Cleaning and harmonisation",
        "Common ISO3 country identifiers and dates\n"
        "FX standardised to local currency per USD\n"
        "Duplicate and invalid-price checks\n"
        "Missing observations retained as missing",
        C_CONTROL, body_size=7.1)
    arrow((collector_x, 0.742), (cleaning[0], 0.742))
    schema = box(
        0.292, 0.535, 0.225, 0.065, "Schema validation",
        "Valid identifiers · positive prices · unique country-date keys",
        C_CONTROL, body_size=6.5, title_size=7.2, linewidth=0.7)
    arrow((cleaning[0] + cleaning[2] / 2, cleaning[1]),
          (schema[0] + schema[2] / 2, schema[1] + schema[3]))

    # Three parallel analytical-variable lines.
    crypto_line = box(
        0.570, 0.745, 0.215, 0.205, "Crypto line",
        "BTC/ETH one-minute prices\n↓\nDaily realised variance\n↓\n"
        "Bitcoin crypto-stress measure\n21-day trailing log mean",
        C_OUTPUT, body_size=7.0)
    domestic_line = box(
        0.570, 0.505, 0.215, 0.205, "Domestic-market line",
        "Daily equity and FX prices\n↓\nDaily log returns\n↓\n"
        "Squared-return volatility proxies",
        C_OUTPUT, body_size=7.0)
    institutional_line = box(
        0.570, 0.335, 0.215, 0.135, "Institutional and global line",
        "Annual country-year variables\nmerged with daily controls",
        C_OUTPUT, body_size=7.0, title_size=7.6)
    for target in (crypto_line, domestic_line, institutional_line):
        arrow((cleaning[0] + cleaning[2], cleaning[1] + cleaning[3] / 2),
              (target[0], target[1] + target[3] / 2),
              connection="arc3,rad=0.05")

    # Sample audit branch, derived specifically from cleaned domestic series.
    audit = box(0.292, 0.355, 0.225, 0.145, "Data-quality sample audit",
                "Coverage · staleness · density", C_CONTROL)
    rule = box(0.292, 0.155, 0.225, 0.165, "Frozen inclusion rule",
               "Equity: coverage ≥ .80; staleness ≤ .20\n"
               "FX: coverage ≥ .80; staleness ≤ .40",
               C_CONTROL, body_size=7.0)
    eligible = box(0.570, 0.125, 0.215, 0.165, "Eligible analytical samples",
                   "77 equity markets · 81 FX markets\n"
                   "86 countries in at least one channel",
                   C_CONTROL, body_size=7.0, title_size=7.6)
    arrow((schema[0] + schema[2] / 2, schema[1]),
          (audit[0] + audit[2] / 2, audit[1] + audit[3]))
    arrow((audit[0] + audit[2] / 2, audit[1]),
          (rule[0] + rule[2] / 2, rule[1] + rule[3]))
    arrow((rule[0] + rule[2], rule[1] + rule[3] / 2),
          (eligible[0], eligible[1] + eligible[3] / 2))

    config = box(
        0.018, 0.035, 0.235, 0.185, "Explicit configuration inputs",
        "Country universe · sample thresholds\n"
        "Regime coding · crypto-restriction episodes",
        "#F7F7F5", body_size=7.0, title_size=7.6, dashed=True)
    arrow((config[0] + config[2], config[1] + config[3] * 0.72),
          (audit[0], audit[1] + audit[3] * 0.35), dashed=True,
          connection="arc3,rad=-0.08")

    # The three variable lines and eligible sample converge at the validated panel.
    panel = box(
        0.820, 0.655, 0.162, 0.180, "Validated country-day\nanalytical panel",
        "Schema-checked merge of\nconstructed variables and\neligible channels",
        C_OUTPUT, body_size=6.9, title_size=7.5)
    panel_collector_x = 0.802
    variable_centers = [b[1] + b[3] / 2
                        for b in (crypto_line, domestic_line, institutional_line)]
    eligible_center = eligible[1] + eligible[3] / 2
    for b, yc in zip((crypto_line, domestic_line, institutional_line), variable_centers):
        ax.plot([b[0] + b[2], panel_collector_x], [yc, yc],
                transform=ax.transAxes, color=C_LINE, lw=0.75, zorder=1)
    ax.plot([eligible[0] + eligible[2], panel_collector_x],
            [eligible_center, eligible_center], transform=ax.transAxes,
            color=C_LINE, lw=0.75, zorder=1)
    ax.plot([panel_collector_x, panel_collector_x],
            [eligible_center, max(variable_centers)],
            transform=ax.transAxes, color=C_LINE, lw=0.75, zorder=1)
    arrow((panel_collector_x, panel[1] + panel[3] / 2),
          (panel[0], panel[1] + panel[3] / 2))

    # Configuration also governs the final panel merge. Route the dashed guide
    # through the open lower margin so it does not obscure substantive stages.
    ax.plot([config[0] + config[2], panel_collector_x], [0.105, 0.105],
            transform=ax.transAxes, color=C_LINE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    arrow((panel_collector_x, 0.105), (panel_collector_x, eligible_center),
          dashed=True, zorder=1)

    step1 = box(
        0.820, 0.405, 0.162, 0.165, "Step 1",
        "Country-channel\nBitcoin-associated\nvariance shares",
        C_OUTPUT, body_size=7.1)
    step2 = box(
        0.820, 0.145, 0.162, 0.195, "Step 2",
        "Regression-ready panel:\nvariance shares, moderators,\n"
        "development indicators,\nand global controls",
        C_OUTPUT, body_size=6.8)
    arrow((panel[0] + panel[2] / 2, panel[1]),
          (step1[0] + step1[2] / 2, step1[1] + step1[3]))
    arrow((step1[0] + step1[2] / 2, step1[1]),
          (step2[0] + step2[2] / 2, step2[1] + step2[3]))

    # A restrained footer identifies the build-level provenance control.
    ax.add_patch(FancyBboxPatch(
        (0.292, 0.035), 0.690, 0.055,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        facecolor="#F7F7F5", edgecolor="#798084", linewidth=0.7,
        transform=ax.transAxes, zorder=3))
    ax.text(0.637, 0.062,
            "Dependency-tracked rebuilds preserve source-to-output lineage",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.2, color="#303638", zorder=4)

    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.02)
    save(fig, "fig_meth_pipeline")


def fig_d_coverage_matrix(path: str | Path | None = None):
    """Appendix D: country-by-channel inclusion decision matrix.

    Every audited country-channel as a pass/fail cell, so the data-quality screen
    is read as the inclusion decision itself. Cells: green = passes; vermillion =
    excluded (with the binding reason cov / stl / c+s); amber = passes the screen
    but is not estimable (Venezuela: hyperinflation/redenomination artifacts)."""
    audit = pl.read_parquet(GOLD / "sample_membership.parquet")
    an = pl.read_parquet(GOLD / "step2_analysis.parquet")
    est = {"equity": set(an.filter(pl.col("spill_equity").is_not_null())["country_id"].to_list()),
           "fx": set(an.filter(pl.col("spill_fx").is_not_null())["country_id"].to_list())}
    abbr = {"coverage": "cov", "stale": "stl", "coverage+stale": "c+s"}
    rec: dict[str, dict[str, tuple[str, str]]] = {}
    for r in audit.iter_rows(named=True):
        iso, ch = r["country_id"], r["channel"]
        if not r["passes"]:
            cell = ("fail", abbr.get(r["reason"], (r["reason"] or "?")[:3]))
        elif iso in est[ch]:
            cell = ("pass", "")
        else:
            cell = ("nest", "est")
        rec.setdefault(iso, {})[ch] = cell

    isos = sorted(rec)
    ncols, n = 3, len(rec)
    per = -(-n // ncols)
    FC = {"pass": "#CDE9DC", "fail": "#F6D6C6", "nest": "#FBE7C4", "na": "#ECECEC"}
    EC = {"pass": "#2E8B57", "fail": "#D55E00", "nest": "#E69F00", "na": "#BDBDBD"}
    TC = {"pass": "#1B6B3A", "fail": "#9C3D08", "nest": "#8A6100", "na": "#888888"}

    fig, axes = plt.subplots(1, ncols, figsize=(11.5, 9))
    for p, ax in enumerate(axes):
        block = isos[p * per:(p + 1) * per]
        for ri, iso in enumerate(block):
            y = per - 1 - ri
            for cj, ch in enumerate(("equity", "fx")):
                st, lab = rec[iso].get(ch, ("na", "n/a"))
                ax.add_patch(plt.Rectangle((cj, y), 1, 1, facecolor=FC[st],
                                           edgecolor="white", lw=1.4, zorder=1))
                if lab:
                    ax.text(cj + 0.5, y + 0.5, lab, ha="center", va="center",
                            fontsize=6.3, color=TC[st], zorder=2)
            ax.text(-0.12, y + 0.5, iso, ha="right", va="center", fontsize=6.8,
                    color="#222222", family="monospace")
        for cj, lab in enumerate(("Equity", "FX")):
            ax.text(cj + 0.5, 1.005, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_xlim(-0.95, 2.02)
        ax.set_ylim(0, per)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    handles = [Patch(facecolor=FC["pass"], edgecolor=EC["pass"], label="Included (passes screen)"),
               Patch(facecolor=FC["fail"], edgecolor=EC["fail"],
                     label="Excluded: cov(erage) / st(a)l(eness) / c+s (both)"),
               Patch(facecolor=FC["nest"], edgecolor=EC["nest"],
                     label="Passes screen but not estimable (est)"),
               Patch(facecolor=FC["na"], edgecolor=EC["na"],
                     label="No independent series (n/a)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.005), handletextpad=0.5, columnspacing=1.4)
    fig.suptitle("Country-channel inclusion decision under the data-quality screen",
                 y=0.985, fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.05, wspace=0.30)
    save(fig, "fig_d_coverage_matrix")


def fig_d_coverage_heatmap(path: str | Path | None = None):
    """Appendix D: monthly data coverage by country and channel, 2019-2025.

    Rows are the 89 configured countries, columns the 84 months; colour is each
    month's coverage (observed business days / calendar business days), clipped to
    [0,1]. Dark purple = a month with little or no data (e.g. Bangladesh's Apr-May
    2020 market closure); grey = no independent series in that channel (dollarized
    or numéraire currencies). Read alongside the inclusion decision in
    `fig_d_coverage_matrix`: this figure shows WHEN coverage thins, not the verdict."""
    p = pl.read_parquet(GOLD / "panel.parquet")
    years = list(range(2019, 2026))
    months = [(y, m) for y in years for m in range(1, 13)]
    midx = {f"{y}-{m:02d}": i for i, (y, m) in enumerate(months)}

    def _busdays(y, m):
        start = np.datetime64(f"{y}-{m:02d}-01")
        end = np.datetime64(f"{y + 1}-01-01") if m == 12 else np.datetime64(f"{y}-{m + 1:02d}-01")
        return int(np.busday_count(start, end))

    busd = np.array([_busdays(y, m) for (y, m) in months], dtype=float)
    countries = sorted(p["country_id"].unique().to_list())
    cidx = {c: i for i, c in enumerate(countries)}

    def matrix(col):
        sub = (p.filter(pl.col(col).is_not_null())
                 .with_columns(ym=pl.col("date").dt.strftime("%Y-%m"))
                 .group_by("country_id", "ym").len())
        M = np.full((len(countries), len(months)), np.nan)
        for c in sub["country_id"].unique().to_list():
            M[cidx[c], :] = 0.0                        # in channel: missing month = 0
        for r in sub.iter_rows(named=True):
            j = midx.get(r["ym"])
            if j is not None:
                M[cidx[r["country_id"]], j] = min(1.0, r["len"] / busd[j])
        return M

    Meq, Mfx = matrix("equity_ret"), matrix("fx_ret")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E6E6E6")

    fig, axes = plt.subplots(1, 2, figsize=(13, 11.2), sharey=True)
    im = None
    for ax, (M, lab) in zip(axes, [(Meq, "A  Equity"), (Mfx, "B  Foreign exchange")]):
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap,
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(lab, loc="left", fontsize=10, fontweight="bold", pad=6)
        jan = [midx[f"{y}-01"] for y in years]
        ax.set_xticks(jan)
        ax.set_xticklabels([str(y) for y in years], fontsize=7)
        for j in jan[1:]:
            ax.axvline(j - 0.5, color="white", lw=0.4, alpha=0.6)
        ax.tick_params(length=0)
        ax.grid(False)
    axes[0].set_yticks(range(len(countries)))
    axes[0].set_yticklabels(countries, fontsize=5.2, family="monospace")
    axes[1].tick_params(axis="y", length=0)
    axes[0].set_ylabel("Country (ISO-3)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cbar.set_label("Monthly coverage (observed / business days)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Monthly data coverage by country and channel, 2019-2025",
                 y=0.985, fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.92, top=0.95, bottom=0.04, wspace=0.04)
    save(fig, "fig_d_coverage_heatmap")


# ---------------------------------------------------------------------------
FIGURES = {"4_1": fig_4_1, "4_1d": fig_4_1_dist, "4_2": fig_4_2,
           "4_3": fig_4_3, "4_4a": fig_4_4a, "4_4b": fig_4_4b,
           "4_5": fig_4_5, "4_6": fig_4_6,
           "4_7": fig_4_7, "4_8": fig_4_8, "4_8b": fig_4_8b,
           "meth_audit": fig_meth_audit, "meth_pipeline": fig_meth_pipeline,
           "d_audit": fig_d_coverage_matrix, "d_cov": fig_d_coverage_heatmap}


def main(argv):
    set_style()
    todo = argv[1:] if len(argv) > 1 else list(FIGURES)
    for fid in todo:
        fn = FIGURES.get(fid.lower())
        if fn is None:
            print(f"  unknown figure '{fid}' (have {list(FIGURES)})")
            continue
        print(f"building {fid} ...")
        fn()


if __name__ == "__main__":
    main(sys.argv)
