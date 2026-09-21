import numpy as np
import pandas as pd
from src.models.decomposition import log_winsorize, mask_by_membership, run_measure


def test_log_winsorize_finite_monotone_keeps_nan():
    s = pd.Series([1e-12, 0.5, 1.0, 5.0, 20.0, np.nan])
    out = log_winsorize(s, lo=0.1, hi=0.9)
    assert np.isfinite(out.dropna()).all()           # extreme-low outlier tamed, no -inf
    assert out.isna().tolist() == [False, False, False, False, False, True]  # NaN preserved
    v = out.dropna().tolist()
    assert v == sorted(v)                             # log is order-preserving


def _base_frame(seed=21):
    rng = np.random.default_rng(seed)
    cids = [f"C{i:02d}" for i in range(16)]
    chinn = {c: rng.uniform(0, 1) for c in cids}
    errf = {c: int(i % 2 == 0) for i, c in enumerate(cids)}
    gdp = {c: rng.uniform(8, 12) for c in cids}
    cred = {c: rng.uniform(40, 160) for c in cids}
    dates = pd.date_range("2021-01-01", periods=70, freq="D")
    cv = rng.normal(0, 1, size=len(dates))
    rows = []
    for c in cids:
        for d, v in zip(dates, cv):
            rows.append({"country_id": c, "date": d, "chinn_ito": chinn[c], "crypto_vol": v,
                         "crypto_stress": 0, "err_float": errf[c], "crypto_ban": 0,
                         "log_gdp_pc": gdp[c], "priv_credit_gdp": cred[c],
                         "vix": 20.0, "dxy": 90.0, "sp500": 3700.0, "oil": 50.0,
                         "spill_equity": 0.1, "spill_fx": 0.1,
                         "A_equity": rng.uniform(0.1, 2.0), "A_fx": rng.uniform(0.1, 2.0)})
    return pd.DataFrame(rows)


def test_mask_by_membership_nulls_channel_measure_columns():
    # BBB fails the equity channel -> every *_equity measure NaN, *_fx kept.
    df = pd.DataFrame({
        "country_id": ["AAA", "BBB"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
        "A_equity": [1.0, 2.0], "B_equity": [0.5, 0.6],
        "A_fx": [0.3, 0.4], "chinn_ito": [0.9, 0.2],
    })
    membership = pd.DataFrame([
        {"country_id": "AAA", "channel": "equity", "passes": True},
        {"country_id": "AAA", "channel": "fx", "passes": True},
        {"country_id": "BBB", "channel": "equity", "passes": False},
        {"country_id": "BBB", "channel": "fx", "passes": True},
    ])
    out = mask_by_membership(df, membership).set_index("country_id")
    assert np.isnan(out.loc["BBB", "A_equity"]) and np.isnan(out.loc["BBB", "B_equity"])
    assert out.loc["BBB", "A_fx"] == 0.4         # fx still qualified
    assert out.loc["AAA", "A_equity"] == 1.0     # equity qualified
    assert out.loc["BBB", "chinn_ito"] == 0.2    # non-channel column untouched


def test_run_measure_tags_and_covers_specs():
    base = _base_frame()
    coefs = run_measure(base, "A_equity", "A_fx", log_winsorize, {"measure": "logA"})
    assert (coefs["measure"] == "logA").all()
    specs = set(coefs["spec"].unique())
    assert {"between", "pooled_full", "twfe", "fx_h3"}.issubset(specs)
