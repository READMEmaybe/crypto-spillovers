"""Tests for src/spillovers/rolling_dy.py.

Contract: shape, bounds, sparsity, expected date coverage. Plus one GFEVD
signal-recovery sanity test so a degenerate decomposition can't pass on
shape/bounds alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.spillovers.rolling_dy import compute_rolling_dy

# VAR(1) coefficient matrices. Column/row 0 = btc, 1 = local.
A_DRIVEN = np.array([[0.2, 0.0], [0.7, 0.2]])  # local_t depends on btc_{t-1}
A_INDEP = np.array([[0.2, 0.0], [0.0, 0.2]])   # the two series are unrelated


def _var1(n: int, A: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Simulate a VAR(1): y_t = A y_{t-1} + e_t, e ~ N(0, I). Cols: btc, local."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, 2))
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = A @ y[t - 1] + e[t]
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.DataFrame({"btc": y[:, 0], "local": y[:, 1]}, index=idx)


def test_output_shape_matches_local_trading_days():
    df = _var1(400, A_DRIVEN)
    out = compute_rolling_dy(df["btc"], df["local"], window=200, min_obs=180)
    assert isinstance(out, pd.Series)
    assert len(out) == 400
    assert out.index.equals(df.index)


def test_values_are_bounded_in_unit_interval():
    df = _var1(400, A_DRIVEN)
    out = compute_rolling_dy(df["btc"], df["local"], window=200, min_obs=180)
    valid = out.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()


def test_early_dates_before_first_full_window_are_nan():
    df = _var1(400, A_DRIVEN)
    out = compute_rolling_dy(df["btc"], df["local"], window=200, min_obs=180)
    # Window ending at positional index i covers i-199..i; first full window at i=199.
    assert out.iloc[:199].isna().all()
    assert out.iloc[199:].notna().all()


def test_sparse_windows_below_min_obs_are_nan():
    df = _var1(400, A_DRIVEN)
    # Blank btc on 30 consecutive trading days; any 200-window fully containing
    # them has only 170 synchronized obs (< min_obs=180) and must be NaN.
    df.iloc[100:130, df.columns.get_loc("btc")] = np.nan
    out = compute_rolling_dy(df["btc"], df["local"], window=200, min_obs=180)
    assert np.isnan(out.iloc[200])  # window 1..200 contains all 30 gaps -> 170 obs
    assert out.notna().iloc[399]    # window 200..399 has no gaps -> 200 obs


def test_gfevd_recovers_transmission_direction():
    driven = _var1(500, A_DRIVEN, seed=1)
    indep = _var1(500, A_INDEP, seed=1)
    s_driven = compute_rolling_dy(driven["btc"], driven["local"], window=200).dropna()
    s_indep = compute_rolling_dy(indep["btc"], indep["local"], window=200).dropna()
    # Analytic population share for A_DRIVEN with Sigma=I is ~0.356
    # (num10/(num10+num11) = 0.576/1.618). When the series are unrelated the
    # share collapses toward zero. Bracket the analytic value, don't pin it.
    assert 0.30 < s_driven.median() < 0.45
    assert s_indep.median() < 0.1
    assert s_driven.median() > s_indep.median()


from src.spillovers.rolling_dy import compute_rolling_components


def test_components_share_equals_A_over_A_plus_B():
    df = _var1(400, A_DRIVEN)
    comp = compute_rolling_components(df["btc"], df["local"], window=200, min_obs=180)
    assert list(comp.columns) == ["spillover", "A", "B", "local_var"]
    row = comp.dropna().iloc[0]
    assert abs(row["spillover"] - row["A"] / (row["A"] + row["B"])) < 1e-9
    assert row["A"] > 0 and row["B"] > 0 and row["local_var"] > 0


def test_components_spillover_matches_compute_rolling_dy():
    df = _var1(400, A_DRIVEN)
    comp = compute_rolling_components(df["btc"], df["local"], window=200, min_obs=180)
    share = compute_rolling_dy(df["btc"], df["local"], window=200, min_obs=180)
    pd.testing.assert_series_equal(comp["spillover"], share, check_names=False)


from src.spillovers.rolling_dy import compute_rolling_3var


def _var1_3(n, A, seed=0):
    """Simulate a 3-var VAR(1). Cols: btc, glob, local."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, 3))
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t] = A @ y[t - 1] + e[t]
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.DataFrame({"btc": y[:, 0], "glob": y[:, 1], "local": y[:, 2]}, index=idx)


def test_3var_shares_btc_drives_local():
    # local (idx 2) depends on btc (idx 0); glob (idx 1) is unrelated to local
    A = np.array([[0.2, 0.0, 0.0],
                  [0.0, 0.2, 0.0],
                  [0.7, 0.0, 0.2]])
    df = _var1_3(500, A, seed=1)
    out = compute_rolling_3var(df["btc"], df["glob"], df["local"], window=200).dropna()
    assert list(out.columns) == ["btc_share", "global_share"]
    assert (out >= 0).all().all() and (out <= 1).all().all()
    assert out["btc_share"].median() > out["global_share"].median()


def test_3var_shares_global_drives_local():
    # local depends on glob (idx 1), not btc
    A = np.array([[0.2, 0.0, 0.0],
                  [0.0, 0.2, 0.0],
                  [0.0, 0.7, 0.2]])
    df = _var1_3(500, A, seed=1)
    out = compute_rolling_3var(df["btc"], df["glob"], df["local"], window=200).dropna()
    assert out["global_share"].median() > out["btc_share"].median()


# --- rolling spectral radius (VAR-stability robustness) -----------------------
from statsmodels.tsa.api import VAR

from src.spillovers.rolling_dy import compute_rolling_spectral_radius

# Lower-triangular VAR(1) with an explosive own-root for the local variable
# (eigenvalues = diagonal = 0.2, 1.1 -> companion spectral radius 1.1 >= 1).
A_EXPLOSIVE = np.array([[0.2, 0.0], [0.6, 1.1]])


def test_spectral_radius_matches_companion_eigenvalue_and_stability():
    df = _var1(400, A_DRIVEN, seed=2)
    out = compute_rolling_spectral_radius(df["btc"], df["local"], window=200, min_obs=180)
    # The value dated at the last row is the VAR fit on the trailing 200 obs.
    res = VAR(df.iloc[-200:][["btc", "local"]].to_numpy()).fit(1)
    expected_root = 1.0 / np.min(np.abs(res.roots))
    expected_eig = np.max(np.abs(np.linalg.eigvals(res.coefs[0])))
    got = out.iloc[-1]
    assert abs(got - expected_root) < 1e-9
    assert abs(got - expected_eig) < 1e-9          # = max|eig(coefs[0])| for VAR(1)
    assert (got < 1.0) == res.is_stable()           # rho < 1 <=> stable


def test_spectral_radius_below_one_on_stable_data():
    df = _var1(400, A_DRIVEN)
    out = compute_rolling_spectral_radius(df["btc"], df["local"], window=200, min_obs=180)
    valid = out.dropna()
    assert len(valid) > 0
    assert (valid < 1.0).all()


def test_spectral_radius_flags_explosive_window():
    df = _var1(200, A_EXPLOSIVE, seed=3)
    out = compute_rolling_spectral_radius(df["btc"], df["local"], window=200, min_obs=180)
    valid = out.dropna()
    assert len(valid) >= 1
    assert valid.iloc[-1] >= 1.0
    res = VAR(df[["btc", "local"]].to_numpy()).fit(1)
    assert not res.is_stable()


def test_spectral_radius_index_and_mask_align_with_rolling_dy():
    df = _var1(400, A_DRIVEN)
    # Blank btc on 30 consecutive days so some windows fall below min_obs -> NaN,
    # exercising that sr and the share share the exact same window coverage.
    df.iloc[100:130, df.columns.get_loc("btc")] = np.nan
    sr = compute_rolling_spectral_radius(df["btc"], df["local"], window=200, min_obs=180)
    share = compute_rolling_dy(df["btc"], df["local"], window=200, min_obs=180)
    assert isinstance(sr, pd.Series)
    assert len(sr) == 400
    assert sr.index.equals(share.index)
    pd.testing.assert_series_equal(sr.notna(), share.notna(), check_names=False)
