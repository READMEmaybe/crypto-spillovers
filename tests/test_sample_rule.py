import numpy as np
import pandas as pd

from src.sample.rule import apply_rule, load_rule, mask_nonmembers, study_membership

RULE = {
    "channels": {
        "equity": {"min_coverage": 0.80, "max_stale_frac": 0.20},
        "fx": {"min_coverage": 0.80, "max_stale_frac": 0.40},
    },
    "membership": "any_channel",
}


def _audit(rows):
    return pd.DataFrame(rows, columns=["country_id", "channel", "coverage", "stale_frac"])


def test_passes_when_both_thresholds_met():
    a = _audit([("AAA", "equity", 0.95, 0.01)])
    out = apply_rule(a, RULE)
    row = out.iloc[0]
    assert bool(row["passes"]) is True
    assert row["reason"] == ""


def test_fails_and_labels_low_coverage():
    a = _audit([("AAA", "equity", 0.70, 0.01)])
    row = apply_rule(a, RULE).iloc[0]
    assert bool(row["passes"]) is False
    assert "coverage" in row["reason"]


def test_fails_and_labels_staleness():
    a = _audit([("AAA", "equity", 0.95, 0.50)])
    row = apply_rule(a, RULE).iloc[0]
    assert bool(row["passes"]) is False
    assert "stale" in row["reason"]


def test_per_channel_thresholds_differ():
    # stale_frac 0.30 fails equity (cap 0.20) but passes fx (cap 0.40)
    a = _audit([("AAA", "equity", 0.95, 0.30), ("AAA", "fx", 0.95, 0.30)])
    out = apply_rule(a, RULE).set_index("channel")
    assert bool(out.loc["equity", "passes"]) is False
    assert bool(out.loc["fx", "passes"]) is True


def test_study_membership_is_any_channel():
    a = _audit([
        ("AAA", "equity", 0.95, 0.01),  # passes
        ("AAA", "fx", 0.50, 0.01),       # fails
        ("BBB", "equity", 0.50, 0.01),  # fails
        ("BBB", "fx", 0.50, 0.01),       # fails
    ])
    members = study_membership(apply_rule(a, RULE))
    assert members == {"AAA"}


def test_mask_nonmembers_nulls_only_disqualified_channel_dv():
    decided = _audit([
        ("AAA", "equity", 0.95, 0.01), ("AAA", "fx", 0.95, 0.01),  # both pass
        ("BBB", "equity", 0.50, 0.01), ("BBB", "fx", 0.95, 0.01),  # equity fails
    ])
    decided = apply_rule(decided, RULE)
    df = pd.DataFrame({
        "country_id": ["AAA", "BBB"],
        "spill_equity": [0.3, 0.4],
        "spill_fx": [0.5, 0.6],
    })
    out = mask_nonmembers(df, decided).set_index("country_id")
    assert out.loc["AAA", "spill_equity"] == 0.3   # qualified -> kept
    assert np.isnan(out.loc["BBB", "spill_equity"])  # disqualified -> NaN
    assert out.loc["BBB", "spill_fx"] == 0.6        # fx still qualified -> kept


def test_load_rule_reads_config_thresholds():
    rule = load_rule("config/sample_rule.yaml")
    assert rule["channels"]["equity"]["max_stale_frac"] == 0.20
    assert rule["channels"]["fx"]["max_stale_frac"] == 0.40
    assert rule["window"]["start"] == "2019-01-01"
