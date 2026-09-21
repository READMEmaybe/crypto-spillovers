"""
Interactive DuckDB shell with all parquet files registered as views.

Usage:
    uv run python scripts/shell.py

Views available after startup:
  equity, fx, controls, kaopen, investing_manual
  crypto_btc, crypto_eth   (union of all monthly files)
"""
import sys
from pathlib import Path

import duckdb

PARQUET_ROOT = Path(__file__).parent.parent / "data" / "parquet"


def register_views(con: duckdb.DuckDBPyConnection) -> list[str]:
    views = []

    # Flat parquet files → view by stem name
    for f in sorted(PARQUET_ROOT.glob("*.parquet")):
        con.execute(f"CREATE VIEW {f.stem} AS SELECT * FROM '{f}'")
        views.append(f.stem)

    # Crypto: one view per symbol (glob across all months)
    for symbol_dir in sorted(PARQUET_ROOT.glob("crypto/*")):
        if symbol_dir.is_dir() and list(symbol_dir.glob("*.parquet")):
            view_name = f"crypto_{symbol_dir.name.replace('usdt', '')}"
            con.execute(
                f"CREATE VIEW {view_name} AS SELECT * FROM '{symbol_dir}/*.parquet'"
            )
            views.append(view_name)

    return views


def main() -> None:
    con = duckdb.connect()
    views = register_views(con)

    if views:
        print(f"Views: {', '.join(views)}")
    else:
        print(f"No parquet files found in {PARQUET_ROOT}")

    print("Type SQL queries, '.tables' to list views, or Ctrl-D to exit.\n")

    while True:
        try:
            query = input("sql> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query == ".tables":
            con.sql("SHOW TABLES").show()
            continue

        try:
            result = con.sql(query)
            if result is not None:
                result.show()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
