"""
Generate BIH_fx.csv (BAM per USD) from the EUR peg.

Bosnia's convertible mark (BAM) is hard-pegged to the euro at
1 EUR = 1.95583 BAM (currency board, unchanged since 1997). There is no
BAM/USD quote on Yahoo Finance or FRED, so we derive it exactly from EURUSD:

    BAM per USD = 1.95583 / (USD per EUR)

EURUSD=X is pulled through the project's own YahooFetcher so BIH tracks the
same euro series the euro-area countries use. OHLC is inverted correctly
(a EUR high is a BAM/USD low). Output matches the Investing.com CSV layout
that InvestingCsvParser expects (Date, Price, Open, High, Low, Vol., Change %;
MM/DD/YYYY; newest-first), so the file drops straight into the existing
investing_csv ingestion path.

Usage:
    python scripts/generate_bih_fx.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fetchers.yahoo_fetcher import YahooFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PEG = 1.95583  # BAM per 1 EUR (KM currency board, fixed since 1997)
EUR_TICKER = "EURUSD=X"  # Yahoo: USD per 1 EUR

CONFIG_PATH = Path(__file__).parent.parent / "config" / "ingestion_settings.yaml"


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["paths"]["investing_csvs"])
    out_path = out_dir / "BIH_fx.csv"

    yahoo = YahooFetcher.from_config(str(CONFIG_PATH))
    raw = yahoo._download(EUR_TICKER, yahoo.start_date, yahoo.end_date)
    if raw is None or raw.empty:
        logger.error("No EURUSD data returned, aborting, BIH_fx.csv not written")
        sys.exit(1)

    df = yahoo._flatten_columns(raw).copy()
    df.index = pd.to_datetime(df.index)

    def col(name: str) -> pd.Series:
        match = next((c for c in df.columns if c.lower() == name), None)
        if match is None:
            raise KeyError(f"EURUSD download missing '{name}' column: {list(df.columns)}")
        return pd.to_numeric(df[match], errors="coerce")

    eur_open, eur_high, eur_low, eur_close = col("open"), col("high"), col("low"), col("close")

    out = pd.DataFrame(index=df.index)
    out["Price"] = PEG / eur_close            # BAM per USD, end-of-day
    out["Open"] = PEG / eur_open
    out["High"] = PEG / eur_low               # EUR low  -> BAM/USD high
    out["Low"] = PEG / eur_high               # EUR high -> BAM/USD low
    out = out.dropna(subset=["Price"]).round(4)

    out["Vol."] = ""
    out["Change %"] = (out["Price"].pct_change() * 100).round(2)

    out = out.sort_index(ascending=False)                        # newest-first
    out.insert(0, "Date", out.index.strftime("%m/%d/%Y"))
    out = out[["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]].reset_index(drop=True)
    out["Change %"] = out["Change %"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}%")

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, quoting=1)  # quote all, like Investing.com exports

    logger.info(
        "Wrote %s: %d rows | %s → %s | BAM/USD range %.4f–%.4f",
        out_path, len(out), out["Date"].iloc[-1], out["Date"].iloc[0],
        out["Price"].min(), out["Price"].max(),
    )


if __name__ == "__main__":
    main()
