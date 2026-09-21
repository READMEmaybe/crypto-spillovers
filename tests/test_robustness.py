import numpy as np
import pandas as pd
from src.models.robustness import logit_winsorize, transform_dv


def test_logit_winsorize_clips_outliers_and_is_finite():
    # one extreme low outlier (1e-9) plus a normal spread in (0,1)
    s = pd.Series([1e-9, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    out = logit_winsorize(s, lo=0.05, hi=0.95)
    assert np.isfinite(out).all()
    # the 1e-9 outlier is winsorized up to the 5th-pct floor, so its logit is
    # bounded well above logit(1e-9) (= -20.7)
    assert out.iloc[0] > -10
    # order preserved for in-range values
    assert out.iloc[1] < out.iloc[5] < out.iloc[-1]


def test_logit_winsorize_preserves_nan():
    s = pd.Series([0.1, np.nan, 0.3])
    out = logit_winsorize(s)
    assert out.isna().tolist() == [False, True, False]


def test_transform_dv_level_is_identity():
    df = pd.DataFrame({"spill_equity": [0.1, np.nan, 0.3], "spill_fx": [0.2, 0.4, 0.5],
                       "other": [1, 2, 3]})
    out = transform_dv(df, "level")
    pd.testing.assert_frame_equal(out, df)


def test_transform_dv_logit_is_monotone_and_keeps_nan():
    df = pd.DataFrame({"spill_equity": [0.1, 0.2, 0.3, np.nan],
                       "spill_fx": [0.5, 0.4, 0.3, 0.2], "other": [1, 2, 3, 4]})
    out = transform_dv(df, "logit")
    # spill_equity ascending -> logit ascending (ignoring NaN)
    eq = out["spill_equity"].dropna().tolist()
    assert eq == sorted(eq)
    assert out["spill_equity"].isna().tolist() == [False, False, False, True]
    # non-DV column untouched
    assert out["other"].tolist() == [1, 2, 3, 4]


from src.models.robustness import run_cell


def _analysis_frame(seed=11):
    """Synthetic analysis-shaped frame with spill in (0,1) and varying dev/regime."""
    rng = np.random.default_rng(seed)
    cids = [f"C{i:02d}" for i in range(18)]
    chinn = {c: rng.uniform(0, 1) for c in cids}
    errf = {c: int(i % 2 == 0) for i, c in enumerate(cids)}
    gdp = {c: rng.uniform(8, 12) for c in cids}
    cred = {c: rng.uniform(40, 160) for c in cids}
    dates = pd.date_range("2021-01-01", periods=80, freq="D")
    cv = rng.normal(0.0, 1.0, size=len(dates))
    rows = []
    for c in cids:
        for d, v in zip(dates, cv):
            rows.append({"country_id": c, "date": d, "chinn_ito": chinn[c],
                         "crypto_vol": v, "crypto_stress": 0, "err_float": errf[c],
                         "crypto_ban": 0, "log_gdp_pc": gdp[c], "priv_credit_gdp": cred[c],
                         "vix": 20.0, "dxy": 90.0, "sp500": 3700.0, "oil": 50.0,
                         "spill_equity": rng.uniform(0.02, 0.6),
                         "spill_fx": rng.uniform(0.02, 0.6)})
    return pd.DataFrame(rows)


def test_run_cell_tags_dv_window_and_covers_specs():
    coefs = run_cell(_analysis_frame(), dv="logit", window_label=60)
    assert (coefs["dv"] == "logit").all()
    assert (coefs["window"] == 60).all()
    specs = set(coefs["spec"].unique())
    assert {"between", "pooled_full", "pooled_nocontrols", "twfe",
            "fx_h3", "fx_float", "fx_peg"}.issubset(specs)
    assert {"equity", "fx"}.issubset(set(coefs["channel"].unique()))
