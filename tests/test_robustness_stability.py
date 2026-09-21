"""Tests for the spectral-radius robustness re-run (src/models/robustness_stability.py).

The filter NaNs VAR-derived columns on windows with rho >= threshold; because every
regression dropna's its DV, NaN == dropped. The key correctness guarantee is the
equivalence guard: a no-op filter must reproduce the frozen headline coefficients.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.panel_regression import run_all
from src.models.robustness_stability import (
    HEADLINE_COLS, _compare_block, apply_filter, run_headline,
)

ANALYSIS = Path("data/parquet/gold/step2_analysis.parquet")


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    df = pd.DataFrame({
        "date": dates, "country_id": ["AAA"] * 3,
        "spill_equity": [0.1, 0.2, 0.3], "spill_fx": [0.4, 0.5, 0.6],
    })
    stab = pd.DataFrame({
        "date": dates, "country_id": ["AAA"] * 3,
        "sr_equity": [0.5, 1.2, 0.9], "sr_fx": [0.8, 0.7, 3.0],
    })
    return df, stab


def test_apply_filter_nans_only_unstable_rows():
    df, stab = _toy()
    out = apply_filter(df, stab, 1.0, HEADLINE_COLS)
    # equity unstable at row 1 (rho 1.2); fx unstable at row 2 (rho 3.0)
    assert out["spill_equity"].iloc[0] == 0.1
    assert np.isnan(out["spill_equity"].iloc[1])
    assert out["spill_equity"].iloc[2] == 0.3
    assert out["spill_fx"].iloc[0] == 0.4
    assert out["spill_fx"].iloc[1] == 0.5
    assert np.isnan(out["spill_fx"].iloc[2])


def test_apply_filter_preserves_columns_and_order_and_drops_sr():
    df, stab = _toy()
    out = apply_filter(df, stab, 1.0, HEADLINE_COLS)
    assert list(out.columns) == list(df.columns)        # sr helper cols dropped
    pd.testing.assert_index_equal(out.index, df.index)   # left-merge keeps order


def test_apply_filter_threshold_inf_is_noop():
    df, stab = _toy()
    out = apply_filter(df, stab, np.inf, HEADLINE_COLS)
    pd.testing.assert_frame_equal(out, df)


def test_unmatched_rows_are_not_filtered():
    df, stab = _toy()
    stab = stab.iloc[:1]  # only the first date has a spectral radius
    out = apply_filter(df, stab, 1.0, HEADLINE_COLS)
    # rows with no radius (NaN >= thr is False) are left untouched
    assert out["spill_equity"].iloc[1] == 0.2
    assert out["spill_fx"].iloc[2] == 0.6


def test_compare_block_flags_flip_and_significance_crossing():
    cur = pd.DataFrame({"channel": ["fx", "fx"], "spec": ["s", "s"], "term": ["a", "b"],
                        "coef": [0.5, -0.20], "pval": [0.01, 0.20], "nobs": [100, 100]})
    flt = pd.DataFrame({"channel": ["fx", "fx"], "spec": ["s", "s"], "term": ["a", "b"],
                        "coef": [-0.4, -0.25], "pval": [0.30, 0.04], "nobs": [90, 90]})
    out = _compare_block(cur, flt, ["channel", "spec", "term"], "headline")
    a = out[out.term == "a"].iloc[0]
    b = out[out.term == "b"].iloc[0]
    assert a.sign_flip and a.sig_change       # 0.5 -> -0.4 flips; p 0.01 -> 0.30 crosses up
    assert (not b.sign_flip) and b.sig_change  # stays negative; p 0.20 -> 0.04 crosses down
    assert abs(a.d_coef - (-0.9)) < 1e-12
    assert a.d_nobs == -10
    assert (out["family"] == "headline").all()


def test_compare_block_identical_frames_no_flags():
    cur = pd.DataFrame({"channel": ["fx"], "spec": ["s"], "term": ["a"],
                        "coef": [0.3], "pval": [0.02], "nobs": [100]})
    out = _compare_block(cur, cur.copy(), ["channel", "spec", "term"], "headline")
    assert out["d_coef"].abs().max() == 0
    assert not out["sign_flip"].any() and not out["sig_change"].any()


@pytest.mark.skipif(not ANALYSIS.exists(), reason="needs gold/step2_analysis.parquet")
def test_noop_filter_reproduces_headline_coefs():
    """Equivalence guard: filtering nothing must reproduce run_all exactly."""
    analysis = pd.read_parquet(ANALYSIS)
    keys = analysis[["date", "country_id"]].drop_duplicates()
    stab = keys.assign(sr_equity=0.0, sr_fx=0.0)  # all stable -> nothing dropped
    base = run_all(analysis).sort_values(["channel", "spec", "term"]).reset_index(drop=True)
    filt = (run_headline(analysis, stab, 1.0)
            .drop(columns=["threshold"])
            .sort_values(["channel", "spec", "term"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(filt, base, rtol=1e-9, atol=1e-12)
