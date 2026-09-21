import numpy as np
import pandas as pd

from src.sample.audit import country_metrics, compute_audit


def _dates(n, start="2019-01-01"):
    # n consecutive business days
    return pd.bdate_range(start=start, periods=n)


def test_country_metrics_counts_valid_obs_and_coverage():
    dates = _dates(5)
    prices = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0])
    m = country_metrics(dates, prices, bdays_full=10)
    assert m["n_obs"] == 5
    assert m["coverage"] == 0.5  # 5 / 10
    assert m["first"] == dates[0]
    assert m["last"] == dates[-1]


def test_country_metrics_staleness_is_share_of_frozen_prices():
    dates = _dates(5)
    # consecutive equal pairs: (1,1)=frozen, (1,2),(2,2)=frozen,(2,3) -> 2/4
    prices = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0])
    m = country_metrics(dates, prices, bdays_full=5)
    assert m["stale_frac"] == 0.5


def test_country_metrics_ignores_nan_prices_for_count_and_staleness():
    dates = _dates(5)
    prices = pd.Series([1.0, 1.0, np.nan, 2.0, 3.0])
    m = country_metrics(dates, prices, bdays_full=5)
    assert m["n_obs"] == 4  # the NaN day is not an observation
    # valid sequence is [1,1,2,3] -> frozen pairs: (1,1) only -> 1/3 (rounded 4dp)
    assert m["stale_frac"] == 0.3333
    assert m["last"] == dates[4]  # last *valid* date, not the NaN


def test_compute_audit_emits_one_row_per_country_channel():
    eq = pd.DataFrame({
        "date": list(_dates(3)) * 2,
        "country_id": ["AAA"] * 3 + ["BBB"] * 3,
        "close": [1.0, 2.0, 3.0, 5.0, 5.0, 5.0],
    })
    fx = pd.DataFrame({
        "date": list(_dates(3)),
        "country_id": ["AAA"] * 3,
        "fx_rate_vs_usd": [1.0, 1.1, 1.2],
    })
    audit = compute_audit(eq, fx, window=("2019-01-01", "2019-01-31"))
    assert set(audit["channel"]) == {"equity", "fx"}
    assert len(audit) == 3  # AAA+BBB equity, AAA fx
    bbb = audit[(audit.country_id == "BBB") & (audit.channel == "equity")].iloc[0]
    assert bbb["stale_frac"] == 1.0  # 5,5,5 -> fully frozen
