"""Build the per-country-year exchange-rate-regime (ERR) reference table.

`err_float[country, year]` for 2019-2025, binary:
    1 = de-facto floating / free-floating
    0 = managed or pegged

Source: IMF AREAER de-facto classification. This is HAND-CURATED reference data
(not fetched at runtime): the logic is `stable groups + explicit switch overrides`.
The stable/switch assignments were verified against the AREAER de-facto tables for
the 2019-2023 report cycle, stored (human-readable, source evidence) at
    data/raw/areaer/areaer_{2019..2023}.csv   (data/raw is gitignored)
Two override flavours sit in SWITCHERS: (a) lag-corrected COVID/crisis episodes
where the AREAER annual snapshot misses the within-year transition month (PER, PHL,
GHA, PAK, TUR, NGA, EGY, MUS, JAM, KEN), and (b) plain AREAER moves where the binary
flag changes between vintages (ARG, CHE, HRV, UKR, ZMB). Full justification, the
crosswalk, the switch table and the annual-vs-monthly decision:
    config/err_transitions.csv  (curated switch log with sources)

Output (long format, one row per country-year):
    config/err_float.csv  ->  country_id, year, err_float, is_switch_year, provisional, note

Run:  python3 scripts/build_err_table.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

YEARS = list(range(2019, 2026))  # 2019..2025 inclusive

# ── Stable groups: same regime every year ───────────────────────────────────
# Verified against the AREAER de-facto classification tables for the 2019-2023
# report cycle (data/raw/areaer/areaer_{2019..2023}.csv). "Stable" = the binary
# float flag is constant across every vintage in that window; 2024-2025 are
# carry-forward (AREAER lag). Countries whose binary flag *moves* live in
# SWITCHERS, not here.
STABLE_FLOAT = [  # err_float = 1 for all years
    "AUS", "CAN", "JPN", "NOR", "SWE", "MEX", "POL",        # free floating
    "BRA", "COL", "HUN", "KOR", "ZAF", "THA", "IND", "IDN", # floating
    "CZE", "CHL",                                            # within-float crossers (Free<->Floating)
    "KAZ",                                                   # KZT free-float since 2015; Floating every AREAER vintage
    "UGA",                                                   # UGX de-facto Floating every AREAER vintage (BoU, market-determined). PROVISIONAL, verify vs AREAER table.
    # AREAER 2019-2023 cycle fill (all-float every vintage):
    "USA", "GBR", "NZL", "ISL", "RUS",                      # non-euro free/floating
    "MYS",                                                  # ringgit floating
    "AUT", "BEL", "CYP", "DEU", "ESP", "EST", "FIN", "FRA", # euro-area members: the
    "GRC", "IRL", "ITA", "LTU", "LVA", "MLT", "NLD", "PRT", # euro floats freely ->
    "SVK", "SVN",                                            # AREAER codes each as Free floating
]
STABLE_NONFLOAT = ["MAR", "CHN", "DZA", "TUN", "VNM", "BGD",  # err_float = 0 for all years
                   "JOR",                                     # JOD conventional peg (USD/SDR) since 1995; every vintage
    # AREAER 2019-2023 cycle fill (managed/pegged every vintage):
    "DNK",                                                  # DKK conventional peg to EUR (ERM II)
    "HKG",                                                  # currency board (USD)
    "BHR", "KWT", "OMN", "QAT", "SAU", "ARE",               # Gulf USD pegs
    "BGR",                                                  # currency board (EUR)
    "BIH",                                                  # currency board (EUR)
    "SGP",                                                  # NEER band (stabilized/other-managed)
    "ROU",                                                  # leu managed (stabilized/crawl-like)
    "ECU", "PAN",                                           # fully dollarized (no separate legal tender)
    "LBN", "KGZ", "KHM", "LAO", "MDV",                      # pegged / stabilized / other-managed
    "MNG", "MWI", "NAM", "TZA", "BWA",                      # managed / crawl-like / other-managed
]

# ── Switchers: explicit per-year err_float (the boundary-crossers) ───────────
# Effective-year coding (AREAER report-lag corrected): the AREAER vintage Y
# reports the regime in force ~Y-1, so e.g. JAM "2023 Regime: Crawl-like"
# => effective 2022. New (2026-06-08) additions MUS/JAM derived from
# data/raw/err_from_IMF_AREAER sheet; 2023-2025 tail is carry-forward (sheet
# has no 2024 vintage), provisional, verify vs Article IV. See [[project-openness-sample-additions]].
SWITCHERS = {
    "PER": {2019: 1, 2020: 0, 2021: 1, 2022: 1, 2023: 1, 2024: 1, 2025: 1},
    "PHL": {2019: 1, 2020: 0, 2021: 1, 2022: 1, 2023: 1, 2024: 1, 2025: 1},
    "GHA": {2019: 1, 2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0},
    "PAK": {2019: 0, 2020: 1, 2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0},
    "TUR": {2019: 1, 2020: 1, 2021: 1, 2022: 0, 2023: 0, 2024: 0, 2025: 0},
    "KEN": {2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0},
    "NGA": {2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 1, 2024: 1, 2025: 1},
    "EGY": {2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 1, 2025: 1},
    "MUS": {2019: 1, 2020: 0, 2021: 1, 2022: 1, 2023: 1, 2024: 1, 2025: 1},
    "JAM": {2019: 1, 2020: 1, 2021: 1, 2022: 0, 2023: 0, 2024: 0, 2025: 0},
    # ── AREAER 2019-2023 cycle: binary flag moves within the window ──────────
    # These come straight from the AREAER de-facto tables (not lag-corrected
    # overrides like the COVID cluster above); 2024-2025 carry forward 2023.
    "ARG": {2019: 1, 2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0},
    "CHE": {2019: 1, 2020: 1, 2021: 0, 2022: 0, 2023: 1, 2024: 1, 2025: 1},
    "HRV": {2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 1, 2024: 1, 2025: 1},
    "UKR": {2019: 1, 2020: 1, 2021: 1, 2022: 1, 2023: 0, 2024: 0, 2025: 0},
    "ZMB": {2019: 1, 2020: 1, 2021: 0, 2022: 1, 2023: 0, 2024: 0, 2025: 0},
}

# Transition provenance, surfaced on the switch-year row.
SWITCH_NOTE = {
    "PER": "managed Apr-2020 -> float Mar-2021 (COVID)",
    "PHL": "managed Mar-2020 -> float Jun-2021 (COVID)",
    "GHA": "-> managed Apr-2020 (COVID/BoP)",
    "PAK": "float Mar-2020 -> Other Managed 6 May 2021",
    "TUR": "-> Crawl-like 28 Jul 2022 (lira crisis)",
    "NGA": "-> float Jun-2023 (reform)",
    "EGY": "-> float Mar-2024 (reform; H6)",
    "MUS": "Crawl-like 2020 (COVID); Floating before/after (AREAER lag-corrected)",
    "JAM": "Floating -> Crawl-like 2022 (AREAER 2023 vintage; tail carry-forward, provisional)",
    "ARG": "Floating -> Stabilized 2020 (cepo capital controls; AREAER)",
    "CHE": "Floating -> Other Managed 2021-22 (SNB FX intervention) -> Floating 2023 (AREAER)",
    "HRV": "-> Free Floating 2023 (euro adoption 1 Jan 2023; AREAER)",
    "UKR": "Floating -> fixed/managed 2023 (wartime measures Feb-2022; AREAER 2023 vintage)",
    "ZMB": "kwacha volatility: Floating<->Other Managed flips 2021-23 (AREAER)",
}

REPO = Path(__file__).resolve().parents[1]


def panel_ids() -> set[str]:
    """Panel countries = config entries with an equity or fx source (drops source-less)."""
    cfg = yaml.safe_load((REPO / "config" / "countries.yaml").read_text())
    return {
        c["id"]
        for c in cfg["countries"]
        if (c.get("equity", {}) or {}).get("source") or (c.get("fx", {}) or {}).get("source")
    }


def build_series(panel: set[str]) -> dict[str, dict[int, int]]:
    """Hand-curated regimes for the countries read from AREAER; every other panel
    country gets the -1 sentinel ("TODO AREAER") until its de-facto regime is read
    and moved into one of the curated structures above. The -1 is allowed to flow
    into config/err_float.csv but the panel build (build_panel.load_err) refuses to
    consume it, so a sentinel can never silently become a regressor."""
    series: dict[str, dict[int, int]] = {}
    for cid in STABLE_FLOAT:
        series[cid] = {y: 1 for y in YEARS}
    for cid in STABLE_NONFLOAT:
        series[cid] = {y: 0 for y in YEARS}
    for cid, vals in SWITCHERS.items():
        series[cid] = {y: vals[y] for y in YEARS}
    # Auto-fill: any panel country not yet hand-curated -> sentinel.
    for cid in sorted(panel):
        if cid not in series:
            series[cid] = {y: -1 for y in YEARS}
    return series


def validate_against_config(series: dict[str, dict[int, int]], panel: set[str]) -> None:
    """Coverage is full by construction (build_series auto-fills the panel); this
    only catches a curated country that has fallen out of the panel (`extra`)."""
    extra = set(series) - panel
    if extra:
        print(f"[FAIL] coded countries not in config panel: extra={sorted(extra)}")
        sys.exit(1)
    n_todo = sum(1 for cid in series if all(v == -1 for v in series[cid].values()))
    print(f"[ok] coverage matches config panel: {len(panel)} countries "
          f"({len(panel) - n_todo} curated, {n_todo} TODO-AREAER sentinels)")


def to_rows(series: dict[str, dict[int, int]]) -> list[tuple]:
    rows = []
    for cid in sorted(series):
        prev = None
        for y in YEARS:
            v = series[cid][y]
            is_switch = int(prev is not None and v != prev)
            note = SWITCH_NOTE.get(cid, "") if is_switch else ""
            if v == -1:
                note = "TODO AREAER"  # de-facto regime not yet read; sentinel
            if cid == "KEN" and y == 2020:
                note = "brief float late-2020 not coded (kept 0)"
            provisional = int(v == -1 or y >= 2024)  # unread or vintage-not-published
            rows.append((cid, y, v, is_switch, provisional, note))
            prev = v
    return rows


def main() -> None:
    # The generator intentionally EMITS -1 sentinels for not-yet-read countries.
    # The guard against a sentinel reaching the regression lives in the consumer
    # (src/features/build_panel.load_err), not here, otherwise the generator could
    # never produce the placeholder table in the first place.
    panel = panel_ids()
    series = build_series(panel)
    validate_against_config(series, panel)
    rows = to_rows(series)

    out = REPO / "config" / "err_float.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country_id", "year", "err_float", "is_switch_year", "provisional", "note"])
        w.writerows(rows)

    # ── sanity report ──
    by_c: dict[str, list[int]] = defaultdict(list)
    for cid, _y, v, *_ in rows:
        by_c[cid].append(v)
    n_float = sum(1 for vs in by_c.values() if sum(vs) > len(vs) / 2)  # majority-float = modal float
    n_switch_rows = sum(r[3] for r in rows)
    print(f"[ok] wrote {len(rows)} rows, {len(by_c)} countries, {len(YEARS)} years -> {out}")
    print(f"[ok] modal split: {n_float} float / {len(by_c) - n_float} non-float")
    print(f"[ok] switch-year rows flagged: {n_switch_rows}")


if __name__ == "__main__":
    main()
