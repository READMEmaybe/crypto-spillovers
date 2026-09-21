"""Build the frozen Step-2 sample: audit -> rule -> membership + appendix.

Reads the clean market tables and `config/sample_rule.yaml`, then writes:
  data/parquet/gold/coverage_audit.parquet      raw per-country/channel metrics
  data/parquet/gold/sample_membership.parquet   per-channel pass/fail + reason
  thesis/appendix/sample_audit.md               human-facing showcase table

Run:  uv run python -m src.sample.build
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sample.audit import compute_audit
from src.sample.report import render_markdown
from src.sample.rule import apply_rule, load_rule, study_membership

CLEAN = Path("data/parquet/clean")
GOLD = Path("data/parquet/gold")
AUDIT_OUT = GOLD / "coverage_audit.parquet"
MEMBER_OUT = GOLD / "sample_membership.parquet"
APPENDIX_OUT = Path("thesis/appendix/sample_audit.md")


def build() -> pd.DataFrame:
    rule = load_rule()
    equity = pd.read_parquet(CLEAN / "equity.parquet")
    fx = pd.read_parquet(CLEAN / "fx.parquet")
    window = (rule["window"]["start"], rule["window"]["end"])

    audit = compute_audit(equity, fx, window)
    decided = apply_rule(audit, rule)
    members = study_membership(decided)

    GOLD.mkdir(parents=True, exist_ok=True)
    APPENDIX_OUT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(AUDIT_OUT, index=False)
    decided.to_parquet(MEMBER_OUT, index=False)
    APPENDIX_OUT.write_text(render_markdown(decided, rule))

    for ch in ("equity", "fx"):
        sub = decided[decided.channel == ch]
        n = int(sub["passes"].sum())
        dropped = sub.loc[~sub["passes"], ["country_id", "reason"]]
        print(f"{ch:6s}: {n}/{len(sub)} qualify | dropped: "
              + ", ".join(f"{c}({r})" for c, r in dropped.itertuples(index=False)))
    print(f"study sample (any channel): {len(members)} countries")
    print(f"wrote {AUDIT_OUT}, {MEMBER_OUT}, {APPENDIX_OUT}")
    return decided


if __name__ == "__main__":
    build()
