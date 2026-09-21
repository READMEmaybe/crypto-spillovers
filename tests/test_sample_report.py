import pandas as pd

from src.sample.report import render_markdown

RULE = {
    "window": {"start": "2019-01-01", "end": "2025-12-31"},
    "channels": {
        "equity": {"min_coverage": 0.80, "max_stale_frac": 0.20},
        "fx": {"min_coverage": 0.80, "max_stale_frac": 0.40},
    },
}


def _decided(rows):
    cols = ["country_id", "channel", "n_obs", "coverage", "density",
            "stale_frac", "passes", "reason"]
    return pd.DataFrame(rows, columns=cols)


def test_report_states_thresholds_and_counts_and_drops():
    decided = _decided([
        ("AAA", "equity", 1700, 0.95, 0.96, 0.01, True, ""),
        ("UKR", "equity", 1781, 0.97, 0.98, 0.82, False, "stale"),
        ("IRN", "equity", 1232, 0.67, 0.67, 0.03, False, "coverage"),
    ])
    md = render_markdown(decided, RULE)
    # thresholds disclosed
    assert "0.8" in md and "0.2" in md
    # pass count for equity: 1 of 3
    assert "1" in md and "3" in md
    # dropped countries appear with their reason
    assert "UKR" in md and "stale" in md
    assert "IRN" in md and "coverage" in md
    # a passing country need not be listed in the drop section, but the doc renders
    assert md.strip().startswith("#")
