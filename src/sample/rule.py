"""Apply the frozen, configuration-based, data-quality-only inclusion rule to the audit table.

Pure functions only: given the audit metrics and the rule dict (loaded from
`config/sample_rule.yaml`), decide per-channel membership and a human-readable
reason for every drop. The rule is data-quality-only by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# DV column each channel governs in the Step-2 analysis table.
CHANNEL_DV = {"equity": "spill_equity", "fx": "spill_fx"}


def load_rule(path: str | Path = "config/sample_rule.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _evaluate(row: pd.Series, thresholds: dict) -> tuple[bool, str]:
    fails = []
    if row["coverage"] < thresholds["min_coverage"]:
        fails.append("coverage")
    # a NaN stale_frac (no data) cannot satisfy the cap -> treat as a failure
    stale = row["stale_frac"]
    if pd.isna(stale) or stale > thresholds["max_stale_frac"]:
        fails.append("stale")
    return (len(fails) == 0), "+".join(fails)


def apply_rule(audit: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Return the audit table with `passes` (bool) and `reason` (str) per row."""
    channels = rule["channels"]
    out = audit.copy()
    evald = out.apply(lambda r: _evaluate(r, channels[r["channel"]]), axis=1)
    out["passes"] = [p for p, _ in evald]
    out["reason"] = [r for _, r in evald]
    return out


def study_membership(decided: pd.DataFrame) -> set[str]:
    """Countries qualifying on >= 1 channel (the `any_channel` rule)."""
    return set(decided.loc[decided["passes"], "country_id"])


def channel_members(decided: pd.DataFrame, channel: str) -> set[str]:
    """Countries qualifying for a specific channel."""
    return set(decided.loc[decided["passes"] & (decided["channel"] == channel),
                           "country_id"])


def mask_nonmembers(df: pd.DataFrame, decided: pd.DataFrame) -> pd.DataFrame:
    """NaN out each channel's spillover DV for countries that did not qualify.

    The frozen sample binds without touching the regression code: the per-channel
    `dropna(subset=[dv])` already in the estimators then excludes the masked rows.
    """
    out = df.copy()
    for channel, dv in CHANNEL_DV.items():
        if dv not in out.columns:
            continue
        keep = channel_members(decided, channel)
        out.loc[~out["country_id"].isin(keep), dv] = np.nan
    return out
