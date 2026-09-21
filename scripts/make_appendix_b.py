#!/usr/bin/env python3
"""Generate Appendix B (Variable Dictionary, Market-Series Mapping, Data Sources).

B.1 (variable dictionary) and B.4 (source details) are curated documentation,
verified against the repo (config, fetchers, methodology docs). B.2 (equity map)
and B.3 (FX map) are generated from `config/countries.yaml` (declared series) joined
to `data/parquet/gold/panel.parquet` (the actual analyzed coverage dates).

Run:  uv run python scripts/make_appendix_b.py

Outputs
-------
- thesis/appendix/appendix_b.md                 consolidated appendix
- thesis/tables/b1_variable_dictionary.md       standalone per-table files
- thesis/tables/b2_equity_series_map.md
- thesis/tables/b3_fx_series_map.md
- thesis/tables/b4_source_details.md

Conventions: DTWEXBGS is the *Broad U.S. Dollar Index* (FRED), not the ICE "DXY".
FX is uniformly local-currency-per-USD; Yahoo XXXUSD=X quotes are USD-per-local and
inverted in src/clean/clean_markets.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "countries.yaml"
PANEL = ROOT / "data" / "parquet" / "gold" / "panel.parquet"
APPDIR = ROOT / "thesis" / "appendix"
TABDIR = ROOT / "thesis" / "tables"
APPDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)
DASH = "-"

# ---------------------------------------------------------------------------
# curated equity index names (compiled from config annotations + exchange refs)
# ---------------------------------------------------------------------------
EQUITY_INDEX = {
    "AUS": "S&P/ASX 200", "CAN": "S&P/TSX Composite", "CZE": "PX Index (Prague SE)",
    "HUN": "BUX (Budapest SE)", "JPN": "Nikkei 225", "NOR": "OSE All-Share (OSEAX)",
    "PER": "S&P/BVL Peru General", "KOR": "KOSPI", "SWE": "OMX Stockholm 30",
    "JOR": "ASE General Index (Amman SE)", "CHL": "S&P IPSA (Santiago SE)",
    "KEN": "NSE All-Share (NASI)", "MEX": "S&P/BMV IPC", "POL": "WIG20",
    "MUS": "SEMDEX (Mauritius SE)", "COL": "COLCAP", "EGY": "EGX 30",
    "IDN": "IDX Composite", "NGA": "NGX All-Share", "PHL": "PSEi", "THA": "SET Index",
    "VNM": "VN-Index", "JAM": "JSE Main Market Index", "BGD": "DSEX (Dhaka SE)",
    "BRA": "Ibovespa", "CHN": "SSE Composite", "IND": "BSE Sensex",
    "MAR": "MASI (Casablanca SE)", "PAK": "KSE 100", "ZAF": "FTSE/JSE Top 40",
    "TUN": "Tunindex", "TUR": "BIST 100", "KAZ": "KASE Index", "ARG": "S&P MERVAL",
    "AUT": "ATX (Vienna SE)", "DNK": "OMX Copenhagen 20", "ECU": "Quito SE (BVQ)",
    "FRA": "CAC 40", "DEU": "DAX", "GRC": "Athex Composite", "HKG": "Hang Seng Index",
    "ISL": "OMX Iceland Main", "ITA": "FTSE MIB", "KWT": "Kuwait All Share",
    "KGZ": "KSE Index (Kyrgyz SE)", "LAO": "LSX Composite", "LBN": "BLOM Stock Index",
    "MWI": "MSE All Share", "MYS": "FTSE Bursa Malaysia KLCI", "MDV": "MSE Equity Index",
    "MNG": "MSE Top 20", "NAM": "NSX Overall Index", "NZL": "S&P/NZX 50", "OMN": "MSM 30",
    "QAT": "QE General Index", "RUS": "MOEX Russia Index", "SAU": "Tadawul All Share (TASI)",
    "SGP": "Straits Times Index", "ESP": "IBEX 35", "CHE": "SMI", "TZA": "DSE All Share",
    "UKR": "PFTS Index", "ARE": "DFM General Index", "GBR": "FTSE 100", "USA": "S&P 500",
    "ZMB": "LuSE All Share", "BHR": "Bahrain All Share", "BEL": "BEL 20",
    "BIH": "SASE (Sarajevo SE)", "BWA": "BSE Domestic Company", "BGR": "SOFIX",
    "KHM": "CSX Index", "HRV": "CROBEX", "CYP": "CSE General Index", "EST": "OMX Tallinn",
    "FIN": "OMX Helsinki 25", "IRL": "ISEQ Overall", "LVA": "OMX Riga", "LTU": "OMX Vilnius",
    "MLT": "MSE Equity Index", "NLD": "AEX", "PAN": "BVPSE General Index", "PRT": "PSI 20",
    "ROU": "BET Index (Bucharest SE)", "SVK": "SAX Index", "SVN": "SBI TOP",
    "UGA": "USE All Share",
    # DZA: config annotation is a copy-paste error (it reads "KASE Index"); the
    # Algeria CSV index label is unverified, so it is left blank and flagged.
    "DZA": "",
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def src_label(s: str | None) -> str:
    if s is None:
        return "Not obtained"
    return {"yahoo": "Yahoo Finance", "investing_csv": "Investing.com",
            "fred": "FRED"}.get(s) or f"Exchange/curated CSV ({s})"


def parse_comments() -> tuple[dict, dict]:
    """First inline comment in each country's equity / fx sub-block."""
    eq, fx, cur, sub = {}, {}, None, None
    for line in CFG.read_text().splitlines():
        m = re.match(r"\s*-\s*id:\s*(\w+)", line)
        if m:
            cur, sub = m.group(1), None
            continue
        if re.match(r"\s*equity:", line):
            sub = "eq"; continue
        if re.match(r"\s*fx:", line):
            sub = "fx"; continue
        if re.match(r"\s*(controls|crypto):", line):
            cur = None; continue
        if cur and sub and "#" in line:
            if "fred_series" in line:   # the FRED fallback note is not an FX data source
                continue
            com = line.split("#", 1)[1].strip()
            tgt = eq if sub == "eq" else fx
            tgt.setdefault(cur, com)
    return eq, fx


_BOILER = ("yfinance", "not on yahoo", "not found", "poor coverage", "unreliable",
           "download csv", "returns no data", "if unavailable", "can't resolve",
           "dropped", "alternative")


def clean_com(c: str | None) -> str:
    """Reduce a config comment to its informative head (peg note / index)."""
    if not c:
        return ""
    c = c.split(";")[0].split("→")[0].strip()
    return "" if any(b in c.lower() for b in _BOILER) else c


def fmt_date(d) -> str:
    return d.strftime("%Y-%m-%d") if d is not None else DASH


# ---------------------------------------------------------------------------
# load config + coverage
# ---------------------------------------------------------------------------
CFG_DATA = yaml.safe_load(CFG.read_text())
COUNTRIES = CFG_DATA["countries"]
EQ_COM, FX_COM = parse_comments()

_p = pl.read_parquet(PANEL)


def _cov(col: str) -> dict:
    g = (_p.filter(pl.col(col).is_not_null())
           .group_by("country_id")
           .agg(start=pl.col("date").min(), end=pl.col("date").max(), n=pl.len()))
    return {r["country_id"]: (r["start"], r["end"], r["n"]) for r in g.iter_rows(named=True)}


EQ_COV, FX_COV = _cov("equity_ret"), _cov("fx_ret")


# ---------------------------------------------------------------------------
# Table B.1 - analytical variable dictionary (curated, verified)
# ---------------------------------------------------------------------------
def table_b1() -> tuple[str, str, str]:
    H = ["Variable group", "Variable", "Symbol", "Definition", "Source",
         "Native frequency", "Transformation / final use", "Coverage"]
    R = [
        ["Crypto market", "Bitcoin realized variance", "RV_t^BTC",
         "Sum of squared one-minute BTCUSDT log returns within a UTC day", "Binance",
         "1 minute", "Step 1 transmitting-volatility input", "2019–2025"],
        ["Crypto market", "Ethereum realized variance", "RV_t^ETH",
         "Same RV pipeline applied to ETHUSDT one-minute returns", "Binance",
         "1 minute", "Robustness transmitter (replaces BTC in Step 1)", "2019–2025"],
        ["Crypto stress", "Bitcoin stress", "C_t",
         "Log of the trailing 21-day mean Bitcoin realized variance", "Constructed (BTC RV)",
         "Daily", "Step 2 explanatory variable (country-invariant)", "2019–2025"],
        ["Equity market", "Equity volatility", "v_{i,t}^EQ",
         "Squared daily index log return", "Yahoo / Investing.com / exchange CSV",
         "Daily", "Step 1 receiving-market input (→ spill_equity)", "Country-specific"],
        ["FX market", "FX volatility", "v_{i,t}^FX",
         "Squared daily log return of local currency per USD",
         "Yahoo / Investing.com / peg-derived", "Daily",
         "Step 1 receiving-market input (→ spill_fx)", "Country-specific"],
        ["Outcome", "Bitcoin variance share", "s_{i,t}",
         "Rolling generalized-FEVD share of domestic forecast-error variance "
         "attributable to Bitcoin RV (200-day window)", "Constructed (Step 1)",
         "Daily", "Step 2 dependent variable", "Per channel"],
        ["Institutional", "Capital-account openness", "KAOPEN_{i,t}",
         "Chinn-Ito annual openness index, normalized to [0,1]", "Chinn-Ito",
         "Annual", "Main moderator; 2023 cross-section carried forward", "2019–2025"],
        ["Institutional", "Float regime", "Float_{i,t}",
         "Binary de facto exchange-rate-regime classification (1=float, 0=managed/peg)",
         "IMF AREAER", "Country-year", "H3 moderator", "2019–2025"],
        ["Regulation", "Crypto restriction", "Ban_{i,t}",
         "Absolute or payment-system crypto prohibition in force", "Curated episode file",
         "Country-day", "Control", "2019–2025"],
        ["Development", "GDP per capita", "GDPpc_i",
         "Log GDP per capita (current USD), 2019–2023 average", "World Bank",
         "Annual", "Structural control (time-invariant)", "Country-specific"],
        ["Development", "Private credit", "Credit_i",
         "Domestic credit to private sector (% GDP), 2019–2023 average", "World Bank",
         "Annual", "Structural control (time-invariant)", "Country-specific"],
        ["Global conditions", "VIX", "VIX_t", "CBOE Volatility Index (global risk-aversion proxy)",
         "Yahoo Finance", "Daily", "Control / interaction robustness", "2019–2025"],
        ["Global conditions", "Broad U.S. Dollar Index", "USD_t",
         "FRED DTWEXBGS broad trade-weighted dollar index", "FRED", "Daily",
         "Control / interaction robustness", "2019–2025"],
        ["Global conditions", "S&P 500", "SP500_t", "Broad U.S. equity-market condition",
         "Yahoo Finance", "Daily", "Control", "2019–2025"],
        ["Global conditions", "Brent crude", "Brent_t", "Global oil-price proxy",
         "Yahoo Finance", "Daily", "Control", "2019–2025"],
    ]
    title = "**Table B.1.** Analytical variable dictionary."
    note = ("*Note.* Symbols match the notation used in Chapters 4 and 5. Realized "
            "variance, the crypto-stress series, and the Bitcoin variance share are "
            "constructed; all other series are sourced as listed. The Broad U.S. Dollar "
            "Index is FRED's DTWEXBGS, not the ICE \"DXY\" series. Per-country equity and "
            "FX sources are detailed in Tables B.2 and B.3; exact dataset codes and "
            "retrieval methods in Table B.4.")
    return title, md_table(H, R), note


# ---------------------------------------------------------------------------
# Table B.2 - equity-series map (generated)
# ---------------------------------------------------------------------------
def table_b2() -> tuple[str, str, str]:
    H = ["Country", "ISO-3", "Equity index", "Ticker / identifier", "Source",
         "Currency", "Start date", "End date", "Notes"]
    rows = []
    for c in sorted(COUNTRIES, key=lambda x: x["name"]):
        eq = c.get("equity") or {}
        if eq.get("source") is None:
            continue  # no equity series declared (e.g. GHA: GSE unavailable)
        iso = c["id"]
        # explicit dict entries win (even when deliberately blank, e.g. DZA); only
        # countries absent from the dict fall back to the config comment.
        idx = EQUITY_INDEX[iso] if iso in EQUITY_INDEX else clean_com(EQ_COM.get(iso))
        idx = idx or DASH
        ident = eq.get("ticker") or eq.get("filename") or DASH
        cov = EQ_COV.get(iso)
        start, end = (fmt_date(cov[0]), fmt_date(cov[1])) if cov else (DASH, DASH)
        note = "Index label unverified (source annotation error)" if iso == "DZA" else ""
        rows.append([c["name"], iso, idx, f"`{ident}`" if ident != DASH else DASH,
                     src_label(eq.get("source")), eq.get("currency") or DASH,
                     start, end, note])
    title = "**Table B.2.** Country-level domestic equity-series map (candidate markets)."
    note = ("*Note.* One row per candidate equity market with a declared series. \"Source\" "
            "distinguishes Yahoo Finance, Investing.com CSV downloads, and official-exchange "
            "or curated CSVs. Tickers are Yahoo symbols; identifiers ending in `.csv` are "
            "manual files in `data/parquet/investing_manual.parquet`. Start and end dates are "
            "the first and last dates with a usable return in the analyzed panel "
            "(`gold/panel.parquet`); they describe coverage, not sample eligibility, which is "
            "determined in Appendix D. Ghana's equity series was sought but not obtained (GSE "
            "Composite unavailable) and is omitted. The Algeria index label is unverified "
            "(source annotation error) and left blank.")
    return title, md_table(H, rows), note


# ---------------------------------------------------------------------------
# Table B.3 - FX-series map (generated)
# ---------------------------------------------------------------------------
def table_b3() -> tuple[str, str, str]:
    H = ["Country", "ISO-3", "Currency", "Series / ticker", "Source", "Native quote",
         "Final quote convention", "Start date", "End date", "Notes"]
    rows = []
    for c in sorted(COUNTRIES, key=lambda x: x["name"]):
        fx = c.get("fx") or {}
        iso, ccy = c["id"], (c.get("equity") or {}).get("currency") or DASH
        source = fx.get("source")
        ident = fx.get("ticker") or fx.get("filename") or DASH
        cov = FX_COV.get(iso)
        start, end = (fmt_date(cov[0]), fmt_date(cov[1])) if cov else (DASH, DASH)
        if iso == "BIH":
            native, conv = "Derived (BAM/EUR peg)", "Local per USD"
        elif source == "yahoo":
            native, conv = "USD per local (XXXUSD=X)", "Local per USD (inverted)"
        elif source == "investing_csv":
            native, conv = "Local per USD", "Local per USD (as-is)"
        else:
            native, conv = DASH, "Local per USD"
        peg = clean_com(FX_COM.get(iso))
        extra = []
        if iso == "BIH":
            extra.append("BAM fixed to EUR; peg-derived via scripts/generate_bih_fx.py")
        elif ccy == "USD":
            extra.append("Dollarized / USD numéraire")
        if peg:
            extra.append(peg)
        rows.append([c["name"], iso, ccy, f"`{ident}`" if ident != DASH else DASH,
                     src_label(source), native, conv, start, end, "; ".join(extra)])
    title = "**Table B.3.** Country-level FX-series map (candidate markets)."
    note = ("*Note.* All FX series are expressed as local currency per U.S. dollar. Yahoo "
            "`XXXUSD=X` quotes are natively USD-per-local and are inverted in "
            "`src/clean/clean_markets.py`; Investing.com CSVs are already local-per-USD and "
            "kept as-is. Bosnia and Herzegovina's series is peg-derived (BAM is fixed to the "
            "euro) rather than quoted. Euro-area and euro-pegged countries share the EUR/USD "
            "series, so their FX volatility is identical across countries (relevant to the FX "
            "results; see Appendix A). Dollarized economies (Ecuador, Panama, United States) "
            "have a near-flat or numéraire FX series. Start and end dates are coverage in "
            "`gold/panel.parquet`, not sample eligibility (Appendix D). FRED `fred_series` "
            "entries in the configuration are an unused fallback and are not a data source here.")
    return title, md_table(H, rows), note


# ---------------------------------------------------------------------------
# Table B.4 - source details (curated, verified)
# ---------------------------------------------------------------------------
def table_b4() -> tuple[str, str, str]:
    H = ["Variable", "Exact source dataset / code", "Retrieval method",
         "Original frequency", "Treatment of 2024–2025", "Notes"]
    R = [
        ["Bitcoin", "Binance `GET /api/v3/klines`, symbol=BTCUSDT, interval=1m",
         "REST API (`src/fetchers/binance_fetcher.py`)", "1-minute",
         "Live API; full window fetched", "Primary transmitter"],
        ["Ethereum", "Binance `GET /api/v3/klines`, symbol=ETHUSDT, interval=1m",
         "REST API", "1-minute", "Live API; full window fetched", "Robustness transmitter"],
        ["Equity indices", "Per-country Yahoo tickers; Investing.com & exchange CSVs (Table B.2)",
         "yfinance + manual CSV (`investing_manual.parquet`)", "Daily",
         "Live / CSV; full window", "See Table B.2"],
        ["FX rates", "Yahoo `XXXUSD=X`; Investing.com CSVs; BIH peg-derived (Table B.3)",
         "yfinance + CSV + `scripts/generate_bih_fx.py`", "Daily", "Live / CSV; full window",
         "Uniform local-per-USD; Yahoo inverted"],
        ["Capital-account openness (KAOPEN)", "Chinn-Ito `ka_open` (vintage through 2023)",
         "Official `.xls` → `kaopen.parquet` (`scripts/fetch_kaopen.py`)", "Annual",
         "2023 cross-section carried forward (moderator is the 2023 value)",
         "Normalized to [0,1]; time-varying for 10/89 countries"],
        ["Exchange-rate regime (Float)", "IMF AREAER, de-facto classification, latest vintage",
         "Curated → `config/err_float.csv` (`scripts/build_err_table.py`)", "Country-year",
         "Carry-forward of last vintage + documented overrides (14 switchers), sourced "
         "from central-bank announcements / IMF Article IV", "Binary float vs non-float; "
         "~10 AREAER categories crosswalked to binary (Appendix C)"],
        ["GDP per capita", "World Bank `NY.GDP.PCAP.CD`",
         "WB REST `api.worldbank.org/v2` (`scripts/fetch_worldbank_dev.py`)", "Annual",
         "2019–2023 average, time-invariant across years", "Log-transformed (`log_gdp_pc`)"],
        ["Private credit / GDP", "World Bank `FS.AST.PRVT.GD.ZS`", "WB REST", "Annual",
         "2019–2023 average, time-invariant", "`priv_credit_gdp`"],
        ["VIX", "Yahoo `^VIX`", "yfinance", "Daily", "Live; full window", "Index level"],
        ["Broad U.S. Dollar Index", "FRED `DTWEXBGS`",
         "FRED API (`src/fetchers/fred_fetcher.py`)", "Daily", "Live; full window",
         "Broad trade-weighted index; NOT the ICE \"DXY\""],
        ["S&P 500", "Yahoo `^GSPC`", "yfinance", "Daily", "Live; full window", "Index level"],
        ["Brent crude", "Yahoo `BZ=F`", "yfinance", "Daily", "Live; full window",
         "USD per barrel"],
        ["Crypto-restriction episodes", "`config/crypto_ban_episodes.csv`",
         "Curated from primary regulator documents (central-bank notices, circulars, "
         "official gazettes)", "Event-dated", "Open end-dates where restrictions remain in "
         "force through 2025", "Typed absolute / payment / banking; date-range join in "
         "`src/features/crypto_ban.py`"],
        ["Crisis-flag inputs", "World Bank `FP.CPI.TOTL.ZG` (CPI) + computed FX depreciation",
         "WB REST + own computation (`src/sample/`)", "Annual", "Through 2025",
         "Hyperinflation (CPI ≥ 25%) / devaluation (≥ 30%) flag; Appendix D"],
    ]
    title = "**Table B.4.** Global, institutional, and macroeconomic source details."
    note = ("*Note.* Dataset codes and tickers are reproduced exactly as queried. The dollar "
            "control is FRED's Broad U.S. Dollar Index (DTWEXBGS), not the conventional ICE "
            "U.S. Dollar Index. World Bank development controls are time-invariant 2019–2023 "
            "averages, so they raise no 2024–2025 vintage question; KAOPEN and the AREAER "
            "regime are sticky and carried forward past their last published vintage, with "
            "regime overrides documented per country-year. The institutional coding protocol "
            "is detailed in Appendix C and the data-quality screen in Appendix D.")
    return title, md_table(H, R), note


# ---------------------------------------------------------------------------
# assemble + write
# ---------------------------------------------------------------------------
def main() -> None:
    builders = [
        ("b1_variable_dictionary", table_b1),
        ("b2_equity_series_map", table_b2),
        ("b3_fx_series_map", table_b3),
        ("b4_source_details", table_b4),
    ]
    parts = [
        "## Appendix B. Variable Dictionary, Market-Series Mapping, and Data Sources",
        "",
        "*(Generated by `scripts/make_appendix_b.py` from `config/countries.yaml` and "
        "`data/parquet/gold/panel.parquet`. Do not edit by hand.)*",
        "",
        "This appendix documents the construction, source, frequency, transformation, and "
        "coverage of all analytical variables used in the empirical analysis. It also "
        "identifies the country-specific equity-index and foreign-exchange series used to "
        "construct the domestic-market volatility measures.",
        "",
    ]
    for tid, fn in builders:
        title, body, note = fn()
        (TABDIR / f"{tid}.md").write_text(f"{title}\n\n{body}\n\n{note}\n\n")
        parts += [title, "", body, "", note, "", "---", ""]
        print(f"  wrote thesis/tables/{tid}.md")
    (APPDIR / "appendix_b.md").write_text("\n".join(parts).rstrip() + "\n")
    print("  wrote thesis/appendix/appendix_b.md")


if __name__ == "__main__":
    main()
