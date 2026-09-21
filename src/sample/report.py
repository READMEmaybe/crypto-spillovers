"""Render the audit + rule outcome as a Markdown table for the thesis appendix.

This is the human-facing "show your work" artifact: it states the frozen rule
thresholds, the pass/drop counts per channel, and every excluded country with the
metric that excluded it. Pure string-building, no I/O.
"""
from __future__ import annotations

import pandas as pd


def _channel_block(decided: pd.DataFrame, channel: str, th: dict) -> str:
    sub = decided[decided.channel == channel].sort_values(["passes", "coverage"])
    n_pass = int(sub["passes"].sum())
    lines = [
        f"### {channel.capitalize()} channel",
        "",
        f"Rule: `coverage >= {th['min_coverage']}` and "
        f"`stale_frac <= {th['max_stale_frac']}`.  "
        f"**Qualify: {n_pass} / {len(sub)}.**",
        "",
    ]
    dropped = sub[~sub["passes"]]
    if dropped.empty:
        lines.append("_No countries dropped on this channel._")
    else:
        lines += [
            "| Country | n_obs | coverage | stale_frac | dropped on |",
            "|---|---:|---:|---:|---|",
        ]
        for _, r in dropped.iterrows():
            lines.append(
                f"| {r.country_id} | {int(r.n_obs)} | {r.coverage:.3f} | "
                f"{r.stale_frac:.3f} | {r.reason} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_markdown(decided: pd.DataFrame, rule: dict) -> str:
    win = rule.get("window", {})
    members = sorted(set(decided.loc[decided["passes"], "country_id"]))
    head = [
        "# Appendix: Country-Inclusion Audit",
        "",
        f"Sample window: **{win.get('start', '?')} .. {win.get('end', '?')}**.  "
        "Countries are selected on data quality alone, blind to results. A country "
        "enters the study if it qualifies on at least one channel; each channel's "
        "regression uses only its qualifying countries.",
        "",
        f"**Study sample: {len(members)} countries.**",
        "",
    ]
    blocks = [_channel_block(decided, ch, rule["channels"][ch])
              for ch in ("equity", "fx") if (decided.channel == ch).any()]
    return "\n".join(head + blocks)
