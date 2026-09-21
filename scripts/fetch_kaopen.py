"""
Parse kaopen_2023.xls (Chinn-Ito KAOPEN dataset) → Parquet.

Wide format (countries × years) → long format with min-max normalized index.

Output: data/parquet/kaopen.parquet
Columns: country_name, iso2, year, ka_open, ka_open_normalized

Usage:
    python scripts/fetch_kaopen.py
    python scripts/fetch_kaopen.py --xls path/to/kaopen.xls
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "ingestion_settings.yaml"
DEFAULT_XLS = Path(__file__).parent.parent / "data" / "raw" / "kaopen_2023.xls"

# Countries to log for verification
VERIFY_COUNTRIES = [
    "Morocco", "Tunisia", "Egypt", "Japan", "Australia",
    "China", "Brazil", "India", "South Africa",
]


def read_kaopen_xls(xls_path: Path) -> pd.DataFrame:
    """Read kaopen_2023.xls. The file is already in long format.

    Columns: cn (IMF code), ccode (ISO3), country_name, year, kaopen (raw),
    ka_open (0–1 normalized by the dataset author).

    Returns columns: country_name, iso3, year, kaopen_raw, ka_open (normalized).
    """
    logger.info("Reading: %s", xls_path)
    df = pd.read_excel(xls_path, engine="xlrd")
    required = {"ccode", "country_name", "year", "kaopen", "ka_open"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected KAOPEN columns: {missing}")

    out = pd.DataFrame({
        "country_name": df["country_name"].astype(str).str.strip(),
        "iso3": df["ccode"].astype(str).str.strip(),
        "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
        "kaopen_raw": pd.to_numeric(df["kaopen"], errors="coerce"),
        "ka_open": pd.to_numeric(df["ka_open"], errors="coerce"),
    })
    out = out.dropna(subset=["country_name", "iso3", "year"])
    out = out[out["country_name"] != ""]
    return out.reset_index(drop=True)


def normalize_kaopen(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset already ships a 0–1 normalized `ka_open`. Keep it as-is under
    `ka_open_normalized` for downstream use; retain the raw Chinn-Ito index
    under `ka_open` for audit."""
    out = df.drop(columns=["ka_open"]).rename(columns={"kaopen_raw": "ka_open"})
    out["ka_open_normalized"] = df["ka_open"].clip(0.0, 1.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse KAOPEN XLS → Parquet")
    parser.add_argument("--xls", default=str(DEFAULT_XLS), help="Path to kaopen_2023.xls")
    args = parser.parse_args()

    xls_path = Path(args.xls)
    if not xls_path.exists():
        logger.error("XLS not found: %s", xls_path)
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    out_root = Path(cfg["paths"]["parquet_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "kaopen.parquet"

    df = read_kaopen_xls(xls_path)
    df = normalize_kaopen(df)
    df["ingest_timestamp"] = datetime.now(tz=timezone.utc)

    logger.info(
        "KAOPEN: %d rows | %d countries | years %d–%d",
        len(df),
        df["country_name"].nunique(),
        int(df["year"].min()),
        int(df["year"].max()),
    )

    # Spot-check known countries for year 2023
    check = df[df["country_name"].isin(VERIFY_COUNTRIES) & (df["year"] == 2023)][
        ["country_name", "iso3", "year", "ka_open", "ka_open_normalized"]
    ]
    if not check.empty:
        logger.info("KAOPEN 2023 spot-check:\n%s", check.to_string(index=False))

    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
