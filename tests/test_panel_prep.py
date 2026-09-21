import math
import numpy as np
import pandas as pd
from src.models.panel_prep import add_crypto_vol, add_crypto_stress, assemble


def _spillovers():
    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    rows = []
    for cid in ["AUS", "BGD"]:
        for d in dates:
            rows.append({"date": d, "country_id": cid, "spill_equity": 0.05,
                         "spill_fx": 0.02, "window": 200, "horizon": 10, "var_lag": 1})
    return pd.DataFrame(rows)


def _panel():
    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    rows = []
    for cid, ci, errf in [("AUS", 1.0, 1), ("BGD", 0.16, 0)]:
        for i, d in enumerate(dates):
            rows.append({"date": d, "country_id": cid, "equity_ret": 0.0, "fx_ret": 0.0,
                         "btc_rv": float(i + 1), "eth_rv": 0.0, "chinn_ito": ci,
                         "crypto_ban": 0, "err_float": errf, "vix": 20.0, "dxy": 90.0,
                         "sp500": 3700.0, "oil": 50.0})
    return pd.DataFrame(rows)


def _dev():
    return pd.DataFrame({"country_id": ["AUS", "BGD"], "log_gdp_pc": [10.8, 7.6],
                         "priv_credit_gdp": [140.0, 40.0]})


def test_add_crypto_vol_is_trailing_log_mean():
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=4, freq="D"),
                       "btc_rv": [1.0, 3.0, 5.0, 7.0]})
    out = add_crypto_vol(df, window=2, min_periods=1)
    # day2 trailing-2 mean = (1+3)/2 = 2 -> log(2)
    assert abs(out["crypto_vol"].iloc[1] - math.log(2.0)) < 1e-9


def test_add_crypto_stress_top_decile():
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=10, freq="D"),
                       "crypto_vol": list(range(10))})
    out = add_crypto_stress(df, q=0.9)
    assert out["crypto_stress"].sum() == 1  # only the max is >= 90th pct
    assert out["crypto_stress"].iloc[-1] == 1


def test_assemble_no_row_explosion_and_columns():
    out = assemble(_spillovers(), _panel(), _dev())
    # 2 countries x 5 dates = 10 rows, no fan-out
    assert len(out) == 10
    for col in ["crypto_vol", "crypto_stress", "log_gdp_pc", "priv_credit_gdp",
                "chinn_ito", "err_float", "spill_equity", "spill_fx"]:
        assert col in out.columns
    # crypto_vol identical across countries on a given date (global series)
    piv = out.pivot_table(index="date", columns="country_id", values="crypto_vol")
    assert np.allclose(piv["AUS"].values, piv["BGD"].values)


def test_assemble_masks_disqualified_channel_via_membership():
    # BGD fails the equity channel -> its spill_equity must be NaN, fx kept.
    membership = pd.DataFrame([
        {"country_id": "AUS", "channel": "equity", "passes": True},
        {"country_id": "AUS", "channel": "fx", "passes": True},
        {"country_id": "BGD", "channel": "equity", "passes": False},
        {"country_id": "BGD", "channel": "fx", "passes": True},
    ])
    out = assemble(_spillovers(), _panel(), _dev(), membership=membership)
    bgd = out[out.country_id == "BGD"]
    assert bgd["spill_equity"].isna().all()
    assert bgd["spill_fx"].notna().all()
    assert out[out.country_id == "AUS"]["spill_equity"].notna().all()
