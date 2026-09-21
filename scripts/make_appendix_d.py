#!/usr/bin/env python3
"""Generate Appendix D (Data-quality audit and sample construction) as Markdown.

Reproducible: built from `data/parquet/gold/sample_membership.parquet` (the frozen
data-quality screen) and `step2_analysis.parquet` (which countries are estimable).
The companion figure (the pass/fail decision matrix) is `fig_d_coverage_matrix`,
produced by `scripts/make_figures.py d_audit`.

Run:  uv run python scripts/make_appendix_d.py

Outputs
-------
- thesis/appendix/appendix_d.md            consolidated appendix
- thesis/tables/d1_quality_audit.md        full 92-row country-channel audit
- thesis/tables/d2_exclusions.md           exclusions and special cases
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "parquet" / "gold"
PARQ = ROOT / "data" / "parquet"
APPDIR = ROOT / "thesis" / "appendix"
TABDIR = ROOT / "thesis" / "tables"
APPDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)
DASH = "-"

CAPS = {"equity": (0.80, 0.20), "fx": (0.80, 0.40)}  # (min coverage, max staleness)

# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
M = pl.read_parquet(GOLD / "sample_membership.parquet")
AN = pl.read_parquet(GOLD / "step2_analysis.parquet")
EST = {"equity": set(AN.filter(pl.col("spill_equity").is_not_null())["country_id"].to_list()),
       "fx": set(AN.filter(pl.col("spill_fx").is_not_null())["country_id"].to_list())}

_ka = (pl.read_parquet(PARQ / "kaopen.parquet")
         .select("iso3", "country_name").unique(subset="iso3"))
NAME = {r["iso3"]: r["country_name"] for r in _ka.iter_rows(named=True)}
NAME.update({"IRN": "Iran", "IRQ": "Iraq", "VEN": "Venezuela"})  # screened out of config

# per-country record: {channel: row-dict}
REC: dict[str, dict[str, dict]] = {}
for r in M.iter_rows(named=True):
    REC.setdefault(r["country_id"], {})[r["channel"]] = r
ISOS = sorted(REC)

REASON = {"coverage": "coverage", "stale": "staleness", "coverage+stale": "coverage+staleness"}


def name(iso: str) -> str:
    return NAME.get(iso, iso)


def cell(row: dict | None, key: str) -> str:
    return f"{row[key]:.2f}" if row is not None else DASH


def ch_status(iso: str, channel: str) -> str:
    row = REC[iso].get(channel)
    if row is None:
        return "No series"
    if not row["passes"]:
        return f"Excluded ({REASON.get(row['reason'], row['reason'])})"
    return "Pass" if iso in EST[channel] else "Pass (not estimable)"


def estimable(iso: str, channel: str) -> bool:
    row = REC[iso].get(channel)
    return bool(row and row["passes"] and iso in EST[channel])


def final_outcome(iso: str) -> str:
    eq_e, fx_e = estimable(iso, "equity"), estimable(iso, "fx")
    eq = REC[iso].get("equity")
    if eq_e and fx_e:
        return "Included: both channels"
    if eq_e:
        return "Included: equity only"
    if fx_e:
        return "Included: FX only"
    if eq is not None and eq["passes"]:           # passes screen but not estimable
        return "Passes equity screen but not estimable; excluded from estimation"
    return "Excluded: fails the data-quality screen in both channels"


# ---------------------------------------------------------------------------
# Table D.1 - full audit
# ---------------------------------------------------------------------------
def table_d1() -> tuple[str, str, str]:
    H = ["Country", "ISO-3", "Eq coverage", "Eq staleness", "Eq density", "Eq status",
         "FX coverage", "FX staleness", "FX density", "FX status",
         "Final inclusion / exclusion"]
    rows = []
    for iso in ISOS:
        eq, fx = REC[iso].get("equity"), REC[iso].get("fx")
        rows.append([
            name(iso), iso,
            cell(eq, "coverage"), cell(eq, "stale_frac"), cell(eq, "density"), ch_status(iso, "equity"),
            cell(fx, "coverage"), cell(fx, "stale_frac"), cell(fx, "density"), ch_status(iso, "fx"),
            final_outcome(iso),
        ])
    title = "**Table D.1.** Full country-channel data-quality audit (all audited countries)."
    note = ("*Note.* Coverage is the share of in-window business days with a usable price; "
            "staleness is the share of consecutive identical prices; density is descriptive "
            "and does not bind. Frozen rules: equity requires coverage ≥ 0.80 and staleness ≤ "
            "0.20; FX requires coverage ≥ 0.80 and staleness ≤ 0.40 (the looser FX cap retains "
            "managed floats while dropping hard pegs). A dash marks a channel with no "
            "independent series (dollarized or numéraire currencies, and countries screened out "
            "before collection). Coverage can exceed 1.00 where stale or duplicated quotes "
            "inflate the observation count. Of 92 audited countries, 78 pass the equity screen "
            "(77 estimable; Venezuela passes but is not estimable) and 81 pass the FX screen; 86 "
            "qualify on at least one channel and form the study sample.")
    return title, md_table(H, rows), note


# ---------------------------------------------------------------------------
# Table D.2 - exclusions and special cases
# ---------------------------------------------------------------------------
def table_d2() -> tuple[str, str, str]:
    H = ["Group", "Country", "ISO-3", "Equity (cov / stale → result)",
         "FX (cov / stale → result)", "Net outcome"]

    def chsum(iso, ch):
        row = REC[iso].get(ch)
        if row is None:
            return "no series"
        return f"{row['coverage']:.2f} / {row['stale_frac']:.2f} → {ch_status(iso, ch).lower()}"

    groups = [
        ("Excluded: fails both screens",
         ["ARE", "ECU", "IRN", "IRQ", "MDV", "PAN"]),
        ("Passes screen but not estimable",
         ["VEN"]),
        ("Included on equity only (FX excluded)",
         ["HRV", "JOR", "QAT", "USA"]),
        ("Included on FX only (equity excluded)",
         ["BWA", "DZA", "GHA", "KGZ", "KHM", "SVK", "UKR", "ZMB"]),
    ]
    rows = []
    for label, isos in groups:
        for i, iso in enumerate(isos):
            rows.append([label if i == 0 else "", name(iso), iso,
                         chsum(iso, "equity"), chsum(iso, "fx"),
                         final_outcome(iso)])
    title = "**Table D.2.** Excluded countries and special cases."
    note = ("*Note.* The 73 countries qualifying and estimable on both channels are omitted "
            "(see Table D.1). Hard or near-fixed pegs (JOR, ARE, MDV, QAT) fail the FX screen "
            "on staleness despite near-complete coverage; Croatia fails FX on coverage after "
            "adopting the euro in 2023. Venezuela passes the equity quality screen but its "
            "bolívar-denominated index is inflation-dominated and carries redenomination "
            "artifacts, so it is not estimable and is excluded from estimation. Iran and Iraq "
            "were screened out before entering the analytical configuration; the study "
            "configuration therefore lists 89 countries, of which 86 enter the study sample.")
    return title, md_table(H, rows), note


# ---------------------------------------------------------------------------
def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    parts = [
        "## Appendix D. Data-Quality Audit and Sample Construction",
        "",
        "*(Generated by `scripts/make_appendix_d.py` from "
        "`data/parquet/gold/sample_membership.parquet`. Do not edit by hand.)*",
        "",
        "Countries are selected on data quality alone, blind to results. A country enters the "
        "study if it qualifies on at least one channel, and each channel's regression uses only "
        "its qualifying countries. The frozen rules are: **equity** requires coverage ≥ 0.80 "
        "and staleness ≤ 0.20; **FX** requires coverage ≥ 0.80 and staleness ≤ 0.40. Density is reported "
        "as a descriptive diagnostic and does not bind. Figure D.1 shows monthly coverage over "
        "time; Figure D.2 shows the pass/fail verdict per channel; Tables D.1 and D.2 give the "
        "underlying values and the exclusions.",
        "",
        "**Figure D.1.** Monthly data coverage by country and channel, 2019-2025 "
        "(`thesis/figures/fig_d_coverage_heatmap.png`). Each cell is a country-month's coverage "
        "(observed business days divided by calendar business days); darker cells mark thin or "
        "missing months (for example Bangladesh's April-May 2020 market closure) and grey marks "
        "a channel with no independent series (dollarized or numéraire currencies). Because "
        "coverage and staleness are separate rules, this figure shows when coverage thins but "
        "not the staleness-based exclusions: hard pegs appear fully covered yet fail on "
        "staleness (see Figure D.2 and Table D.2).",
        "",
        "**Figure D.2.** Country-channel inclusion decision under the data-quality screen "
        "(`thesis/figures/fig_d_coverage_matrix.png`): the pass/fail verdict per channel, "
        "including the staleness exclusions and the not-estimable case (Venezuela).",
        "",
    ]
    for tid, fn in [("d1_quality_audit", table_d1), ("d2_exclusions", table_d2)]:
        title, body, note = fn()
        (TABDIR / f"{tid}.md").write_text(f"{title}\n\n{body}\n\n{note}\n\n")
        parts += [title, "", body, "", note, "", "---", ""]
        print(f"  wrote thesis/tables/{tid}.md")
    (APPDIR / "appendix_d.md").write_text("\n".join(parts).rstrip() + "\n")
    print("  wrote thesis/appendix/appendix_d.md")


if __name__ == "__main__":
    main()
