import numpy as np
import pandas as pd
import pytest
from src.models.panel_regression import add_terms, fit_between


@pytest.fixture
def synth():
    """Panel with a KNOWN cross-sectional slope: mean spill = 0.1 + 0.3*chinn_ito."""
    rng = np.random.default_rng(0)
    cids = [f"C{i:02d}" for i in range(20)]
    chinn = {c: rng.uniform(0, 1) for c in cids}
    # dev controls vary ACROSS countries (independent of chinn), as in real data;
    # a single constant would be collinear with the intercept and make exog singular.
    gdp = {c: rng.uniform(8, 12) for c in cids}
    cred = {c: rng.uniform(40, 160) for c in cids}
    dates = pd.date_range("2021-01-01", periods=60, freq="D")
    cv = rng.normal(-8, 0.5, size=len(dates))  # global crypto_vol
    rows = []
    for c in cids:
        base = 0.1 + 0.3 * chinn[c]
        for d, v in zip(dates, cv):
            rows.append({"country_id": c, "date": d, "chinn_ito": chinn[c],
                         "crypto_vol": v, "err_float": 0, "crypto_ban": 0,
                         "log_gdp_pc": gdp[c], "priv_credit_gdp": cred[c],
                         "vix": 20.0, "dxy": 90.0, "sp500": 3700.0, "oil": 50.0,
                         "spill_equity": base + rng.normal(0, 0.01),
                         "spill_fx": base + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_add_terms_builds_interactions():
    df = pd.DataFrame({"chinn_ito": [0.5], "crypto_vol": [-8.0], "err_float": [1]})
    out = add_terms(df)
    assert abs(out["ci_x_cv"].iloc[0] - (0.5 * -8.0)) < 1e-12
    assert abs(out["errf_x_cv"].iloc[0] - (1 * -8.0)) < 1e-12
    assert abs(out["ci_x_errf"].iloc[0] - (0.5 * 1)) < 1e-12
    assert abs(out["ci_x_errf_x_cv"].iloc[0] - (0.5 * 1 * -8.0)) < 1e-12


def test_between_recovers_cross_sectional_slope(synth):
    res = fit_between(synth, "spill_equity", ["chinn_ito", "log_gdp_pc", "priv_credit_gdp"])
    assert abs(res.params["chinn_ito"] - 0.3) < 0.05


from src.models.panel_regression import fit_pooled_dk


def test_pooled_recovers_interaction_beta():
    """spill = 0.05 + 0.2*ci + 0.01*cv + 0.5*(ci*cv) + noise; recover 0.5."""
    rng = np.random.default_rng(1)
    cids = [f"C{i:02d}" for i in range(15)]
    chinn = {c: rng.uniform(0, 1) for c in cids}
    gdp = {c: rng.uniform(8, 12) for c in cids}
    cred = {c: rng.uniform(40, 160) for c in cids}
    dates = pd.date_range("2021-01-01", periods=80, freq="D")
    cv = rng.normal(0.0, 1.0, size=len(dates))
    rows = []
    for c in cids:
        for d, v in zip(dates, cv):
            y = 0.05 + 0.2 * chinn[c] + 0.01 * v + 0.5 * (chinn[c] * v) + rng.normal(0, 0.02)
            rows.append({"country_id": c, "date": d, "chinn_ito": chinn[c],
                         "crypto_vol": v, "crypto_ban": 0, "err_float": 0,
                         "log_gdp_pc": gdp[c], "priv_credit_gdp": cred[c],
                         "vix": 20.0, "dxy": 90.0, "sp500": 3700.0, "oil": 50.0,
                         "spill_equity": y})
    df = pd.DataFrame(rows)
    res = fit_pooled_dk(df, "spill_equity", include_globals=False)
    assert abs(res.params["ci_x_cv"] - 0.5) < 0.05
    assert abs(res.params["chinn_ito"] - 0.2) < 0.05


def test_h5_toggle_changes_regressor_set():
    rng = np.random.default_rng(2)
    cids = [f"C{i:02d}" for i in range(12)]
    chinn = {c: rng.uniform(0, 1) for c in cids}
    gdp = {c: rng.uniform(8, 12) for c in cids}
    cred = {c: rng.uniform(40, 160) for c in cids}
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    rows = []
    for c in cids:
        for d in dates:
            rows.append({"country_id": c, "date": d, "chinn_ito": chinn[c],
                         "crypto_vol": rng.normal(), "crypto_ban": 0, "err_float": 0,
                         "log_gdp_pc": gdp[c], "priv_credit_gdp": cred[c],
                         "vix": rng.uniform(15, 30), "dxy": 90.0, "sp500": 3700.0, "oil": 50.0,
                         "spill_equity": rng.normal(0.05, 0.01)})
    df = pd.DataFrame(rows)
    with_g = fit_pooled_dk(df, "spill_equity", include_globals=True)
    without_g = fit_pooled_dk(df, "spill_equity", include_globals=False)
    assert "vix" in with_g.params.index
    assert "vix" not in without_g.params.index


from src.models.panel_regression import fit_twoway_fe


def test_twoway_fe_recovers_interaction_and_drops_mains():
    """With entity+time FE, only ci_x_cv (+ crypto_ban) are estimable."""
    rng = np.random.default_rng(3)
    cids = [f"C{i:02d}" for i in range(15)]
    chinn = {c: rng.uniform(0, 1) for c in cids}
    dates = pd.date_range("2021-01-01", periods=80, freq="D")
    cv = rng.normal(0.0, 1.0, size=len(dates))
    rows = []
    for c in cids:
        for d, v in zip(dates, cv):
            y = 2.0 * chinn[c] + 0.3 * v + 0.5 * (chinn[c] * v) + rng.normal(0, 0.02)
            # crypto_ban needs within-entity variation to be estimable under entity FE
            ban = int(c == "C00" and d >= dates[40])
            rows.append({"country_id": c, "date": d, "chinn_ito": chinn[c],
                         "crypto_vol": v, "crypto_ban": ban, "spill_equity": y})
    df = pd.DataFrame(rows)
    res = fit_twoway_fe(df, "spill_equity")
    assert abs(res.params["ci_x_cv"] - 0.5) < 0.05
    assert "chinn_ito" not in res.params.index   # absorbed by entity FE
    assert "crypto_vol" not in res.params.index   # absorbed by time FE


from src.models.panel_regression import fit_fx_h3, fit_regime_split


def _fx_synth(seed=4):
    """FX spill responds to crypto_vol ONLY for floaters (err_float=1)."""
    rng = np.random.default_rng(seed)
    cids = [f"C{i:02d}" for i in range(16)]
    chinn = {c: rng.uniform(0, 1) for c in cids}
    errf = {c: int(i % 2 == 0) for i, c in enumerate(cids)}
    dates = pd.date_range("2021-01-01", periods=70, freq="D")
    cv = rng.normal(0.0, 1.0, size=len(dates))
    rows = []
    for c in cids:
        for d, v in zip(dates, cv):
            y = 0.05 + 0.4 * errf[c] * v + rng.normal(0, 0.02)  # slope only if float
            rows.append({"country_id": c, "date": d, "chinn_ito": chinn[c],
                         "crypto_vol": v, "err_float": errf[c], "crypto_ban": 0,
                         "log_gdp_pc": 10.0, "priv_credit_gdp": 100.0,
                         "vix": 20.0, "dxy": 90.0, "sp500": 3700.0, "oil": 50.0,
                         "spill_fx": y})
    return pd.DataFrame(rows)


def test_fx_h3_detects_regime_interaction():
    res = fit_fx_h3(_fx_synth())
    # errf_x_cv should be clearly positive (~0.4); regime makes FX visible
    assert res.params["errf_x_cv"] > 0.2


def test_regime_split_returns_both_subsamples():
    out = fit_regime_split(_fx_synth())
    assert set(out) == {"float", "peg"}
    # crypto_vol slope present for floaters, ~0 for pegs
    assert out["float"].params["crypto_vol"] > 0.2
    assert abs(out["peg"].params["crypto_vol"]) < 0.1


# append to tests/test_panel_regression.py
from src.models.panel_regression import tidy, run_all


def test_tidy_extracts_coef_rows():
    df = _fx_synth(seed=9)
    res = fit_pooled_dk(df, "spill_fx", include_globals=False)
    out = tidy(res, channel="fx", spec="pooled_nocontrols")
    assert set(["channel", "spec", "term", "coef", "se", "tstat", "pval", "nobs"]).issubset(out.columns)
    assert (out["spec"] == "pooled_nocontrols").all()
    assert "ci_x_cv" in out["term"].values


def test_run_all_covers_every_spec():
    # build a combined equity+fx synthetic analysis frame
    rng = np.random.default_rng(7)
    cids = [f"C{i:02d}" for i in range(18)]
    chinn = {c: rng.uniform(0, 1) for c in cids}
    errf = {c: int(i % 2 == 0) for i, c in enumerate(cids)}
    dates = pd.date_range("2021-01-01", periods=90, freq="D")
    cv = rng.normal(0.0, 1.0, size=len(dates))
    rows = []
    for c in cids:
        for d, v in zip(dates, cv):
            rows.append({"country_id": c, "date": d, "chinn_ito": chinn[c],
                         "crypto_vol": v, "crypto_stress": 0, "err_float": errf[c],
                         "crypto_ban": 0, "log_gdp_pc": 10.0, "priv_credit_gdp": 100.0,
                         "vix": 20.0, "dxy": 90.0, "sp500": 3700.0, "oil": 50.0,
                         "spill_equity": 0.05 + 0.3 * chinn[c] * v + rng.normal(0, 0.02),
                         "spill_fx": 0.02 + 0.2 * errf[c] * v + rng.normal(0, 0.02)})
    df = pd.DataFrame(rows)
    coefs = run_all(df)
    specs = set(coefs["spec"].unique())
    assert {"between", "pooled_full", "pooled_nocontrols", "twfe"}.issubset(specs)
    assert {"fx_h3", "fx_float", "fx_peg"}.issubset(specs)
    assert {"fx_h3_direct", "fx_h3_direct_net"}.issubset(specs)
    assert {"equity", "fx"}.issubset(set(coefs["channel"].unique()))


from src.models.panel_regression import fit_fx_h3_direct


def test_fx_h3_direct_pure_is_errf_interaction_without_openness():
    """Direct H3: err_float x crypto_vol as the focal term, openness interaction OUT.

    _fx_synth has FX responding to crypto ONLY for floaters (true errf_x_cv ~ 0.4 and no
    openness effect), so the pure spec must recover a clearly positive errf_x_cv and must
    NOT contain the openness interaction ci_x_cv.
    """
    res = fit_fx_h3_direct(_fx_synth(), net=False)
    assert "errf_x_cv" in res.params.index
    assert res.params["errf_x_cv"] > 0.2
    assert "ci_x_cv" not in res.params.index


def test_fx_h3_direct_net_restores_openness_interaction():
    """The net-of-openness variant adds chinn_ito and ci_x_cv back as controls."""
    res = fit_fx_h3_direct(_fx_synth(), net=True)
    assert "errf_x_cv" in res.params.index
    assert "ci_x_cv" in res.params.index
    assert "chinn_ito" in res.params.index
