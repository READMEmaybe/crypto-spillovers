"""Fetch country-level development controls from the World Bank REST API.

Indicators:
  NY.GDP.PCAP.CD     GDP per capita (current US$)        -> log_gdp_pc
  FS.AST.PRVT.GD.ZS  Domestic credit to private sector (% GDP) -> priv_credit_gdp

Each indicator is averaged over 2019-2023 per country (ignoring missing years).
Country universe is read from gold/panel.parquet (ISO3 ids). Output is one row
per country: data/parquet/country_dev.parquet.

Run:  uv run python scripts/fetch_worldbank_dev.py
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import pandas as pd
import requests

PANEL = Path("data/parquet/gold/panel.parquet")
OUT = Path("data/parquet/country_dev.parquet")
BASE = "https://api.worldbank.org/v2/country/all/indicator/{ind}"
YEARS = "2019:2023"

# The World Bank country/all endpoint sporadically throttles otherwise-valid
# requests with a 400/429/5xx; treat those (and connection errors) as transient
# and retry with exponential backoff rather than aborting the whole pipeline.
RETRY_STATUS = {400, 408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
BACKOFF_BASE = 1.5  # seconds; doubles each attempt

logger = logging.getLogger(__name__)


def parse_indicator(payload: list, keep: set[str]) -> dict[str, float]:
    """Mean of non-null values per country, restricted to `keep` ISO3 codes."""
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for rec in payload[1]:
        iso3 = rec.get("countryiso3code")
        val = rec.get("value")
        if iso3 not in keep or val is None:
            continue
        sums[iso3] = sums.get(iso3, 0.0) + float(val)
        counts[iso3] = counts.get(iso3, 0) + 1
    return {c: sums[c] / counts[c] for c in sums}


def fetch_indicator(indicator: str, session: requests.Session | None = None,
                    max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Hit the World Bank REST endpoint for one indicator over YEARS.

    Retries transient failures (sporadic 400/429/5xx throttling and connection
    errors) with exponential backoff; re-raises genuine client errors such as a
    404 from a bad indicator code immediately.
    """
    sess = session or requests.Session()
    url = BASE.format(ind=indicator)
    for attempt in range(1, max_attempts + 1):
        try:
            resp = sess.get(
                url, params={"format": "json", "per_page": 20000, "date": YEARS}, timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = status is None or status in RETRY_STATUS
            if not transient or attempt == max_attempts:
                raise
            wait = BACKOFF_BASE * 2 ** (attempt - 1)
            logger.warning(
                "World Bank %s attempt %d/%d failed (%s); retrying in %.1fs",
                indicator, attempt, max_attempts, status or type(exc).__name__, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"World Bank fetch for {indicator} exhausted retries")


def parse_indicator_or_fetch(indicator: str, keep: set[str]) -> dict[str, float]:
    """Fetch one World Bank indicator and reduce it to a per-country mean."""
    return parse_indicator(fetch_indicator(indicator), keep)


def build_country_dev(country_ids: list[str]) -> pd.DataFrame:
    keep = set(country_ids)
    # parse_indicator_or_fetch is the seam tests monkeypatch
    gdp = parse_indicator_or_fetch("NY.GDP.PCAP.CD", keep)
    cred = parse_indicator_or_fetch("FS.AST.PRVT.GD.ZS", keep)
    rows = []
    for cid in sorted(keep):
        g = gdp.get(cid)
        rows.append(
            {
                "country_id": cid,
                "log_gdp_pc": math.log(g) if g is not None and g > 0 else float("nan"),
                "priv_credit_gdp": cred.get(cid, float("nan")),
            }
        )
    return pd.DataFrame(rows, columns=["country_id", "log_gdp_pc", "priv_credit_gdp"])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Imported lazily so this module's tests don't depend on the schema (added in Task 2).
    from src.quality.schemas import COUNTRY_DEV

    countries = pd.read_parquet(PANEL, columns=["country_id"])["country_id"].unique().tolist()
    logger.info("Fetching NY.GDP.PCAP.CD from World Bank API")
    logger.info("Fetching FS.AST.PRVT.GD.ZS from World Bank API")
    df = build_country_dev(countries)
    COUNTRY_DEV.validate(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    n_missing = int(df["priv_credit_gdp"].isna().sum())
    logger.info("Wrote %s  shape=%s  priv_credit_gdp missing: %d", OUT, df.shape, n_missing)


if __name__ == "__main__":
    main()
