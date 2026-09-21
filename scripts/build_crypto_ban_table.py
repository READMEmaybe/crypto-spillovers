"""Validate the curated crypto-ban episode table and print its summary.

The panel derives `crypto_ban` on the fly (no materialized file): this CLI just
runs schema validation via `crypto_ban.load_episodes` and prints
`crypto_ban.annual_summary` under the default rule {absolute, payment} -
episode count, per-country ban-days in [2019-01-01, 2025-12-31], and the list
of within-sample transitions (the rows carrying the most analytical weight).

Source of truth: config/crypto_ban_episodes.csv
Module:          src/features/crypto_ban.py

Run:  python scripts/build_crypto_ban_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.features import crypto_ban  # noqa: E402


def main() -> None:
    eps = crypto_ban.load_episodes()
    n_countries = eps["country_id"].nunique()
    print(f"[ok] validated {len(eps)} episodes across {n_countries} countries "
          f"-> {crypto_ban.EPISODES_CSV}")
    print(f"[ok] default rule ACTIVE_TYPES = {sorted(crypto_ban.DEFAULT_ACTIVE_TYPES)}")
    print(f"     ban_type counts: {eps['ban_type'].value_counts().to_dict()}")

    summary = crypto_ban.annual_summary()
    active = summary[summary["ban_days"] > 0].sort_values("ban_days", ascending=False)
    print(f"\n[ok] per-country ban-days in window (default rule), "
          f"{len(active)} countries with any ban-day:")
    for cid, row in active.iterrows():
        flag = "  <-- in-window transition" if row["in_window_transition"] else ""
        print(f"     {cid}  ban_days={int(row['ban_days']):>5}  "
              f"episodes={int(row['n_episodes'])}{flag}")

    transitions = summary.attrs["transitions"]
    trans_countries = sorted({t.split(":")[0] for t in transitions})
    print(f"\n[ok] within-sample transitions ({len(transitions)}), "
          f"countries: {trans_countries}")
    for t in transitions:
        print(f"     {t}")


if __name__ == "__main__":
    main()
