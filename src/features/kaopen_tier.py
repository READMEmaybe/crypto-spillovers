"""Derive the KAOPEN openness tier from the Chinn-Ito normalized index.

The canonical KAOPEN values live in `data/parquet/kaopen.parquet`
(`ka_open_normalized`, 0-1). This module is the single place that bins those
values into the four openness tiers used throughout the analysis, replacing the
previously hard-coded `kaopen_tier` field in `config/countries.yaml`.

Thresholds (on `ka_open_normalized`):

    >= 0.9 -> "open"
    >= 0.6 -> "partial"
    >= 0.3 -> "intermediate"
    else   -> "restricted"
"""
from __future__ import annotations


def tier(ka_open_normalized: float) -> str:
    """Bin a normalized KAOPEN value (0-1) into an openness tier."""
    if ka_open_normalized >= 0.9:
        return "open"
    if ka_open_normalized >= 0.6:
        return "partial"
    if ka_open_normalized >= 0.3:
        return "intermediate"
    return "restricted"
