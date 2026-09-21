"""Rolling-window Diebold-Yilmaz directional spillover (BTC -> local market).

Implements the Step 1 contract:

  * bivariate VAR(p) on [btc_rv, local_vol_proxy], re-estimated on each window
  * generalized FEVD (Pesaran & Shin 1998 / KPPS), as used in Diebold-Yilmaz
    (2012) -- order-invariant, so no Cholesky-ordering assumption to defend
  * the reported spillover is the row-normalized share of the *local* variable's
    H-step forecast-error variance attributable to shocks in btc_rv

statsmodels' VAR only ships the orthogonalized (Cholesky) decomposition, so the
generalized variant is computed by hand from the MA coefficients (``ma_rep``)
and the residual covariance (``sigma_u``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def _gfevd_matrices(data: np.ndarray, horizon: int, lag: int):
    """Row-normalized generalized FEVD plus its raw pieces, or None if the VAR fails.

    Returns ``(gfevd, num, den, sigma_diag)`` where ``gfevd`` is the (k, k)
    Diebold-Yilmaz table (rows = targets, cols = sources, each row sums to 1),
    ``num[i, j] = sum_h (Theta_h Sigma)_ij^2``, ``den[i] = sum_h
    (Theta_h Sigma Theta_h')_ii`` (variable i's H-step forecast-error variance),
    and ``sigma_diag = diag(Sigma)``.
    """
    try:
        res = VAR(data).fit(lag)
        sigma = np.asarray(res.sigma_u)            # residual covariance (k, k)
        theta = res.ma_rep(maxn=horizon - 1)       # MA coeffs Theta_0..Theta_{H-1}
    except Exception:
        return None

    sigma_diag = np.diag(sigma)
    k = sigma.shape[0]
    num = np.zeros((k, k))   # sum_h (Theta_h Sigma)_ij^2
    den = np.zeros(k)        # sum_h (Theta_h Sigma Theta_h')_ii
    for h in range(horizon):
        ts = theta[h] @ sigma
        num += ts**2
        den += np.einsum("ij,ij->i", ts, theta[h])  # diag(Theta_h Sigma Theta_h')

    # Unnormalized generalized variance shares, then row-normalize so each
    # target variable's contributions sum to 1 (the Diebold-Yilmaz table).
    gfevd = (num / sigma_diag[np.newaxis, :]) / den[:, np.newaxis]
    gfevd /= gfevd.sum(axis=1, keepdims=True)
    return gfevd, num, den, sigma_diag


def _gfevd_share(data: np.ndarray, horizon: int, lag: int) -> float:
    """Generalized-FEVD share of variable 1's variance explained by variable 0.

    ``data`` is an (n, 2) array with column 0 = btc, column 1 = local. Returns
    the H-step directional spillover btc -> local on a 0-1 scale, or ``nan`` if
    the VAR cannot be estimated on this window.
    """
    m = _gfevd_matrices(data, horizon, lag)
    if m is None:
        return float("nan")
    return float(m[0][1, 0])  # target = local (row 1), source = btc (col 0)


def compute_rolling_dy(
    btc_rv: pd.Series,
    local: pd.Series,
    window: int = 200,
    horizon: int = 10,
    lag: int = 1,
    min_obs: int = 180,
) -> pd.Series:
    """Rolling directional spillover btc_rv -> local volatility proxy.

    The window slides over the local market's *trading days* (dates where
    ``local`` is observed); btc trades every day, so this makes ``window`` a
    count of trading days. Within each trailing ``window``-day span only
    synchronized rows (both series observed) are used; if fewer than ``min_obs``
    remain the window is skipped and ``nan`` recorded. The value dated ``t``
    corresponds to the window *ending* on ``t``.

    Returns a Series indexed by the local trading days, named ``spillover``.
    """
    df = pd.concat([btc_rv.rename("btc"), local.rename("local")], axis=1).sort_index()
    trading = df.loc[df["local"].notna()]
    out = pd.Series(np.nan, index=trading.index, name="spillover", dtype=float)

    values = trading[["btc", "local"]].to_numpy()
    for i in range(window - 1, len(trading)):
        win = values[i - window + 1 : i + 1]
        synced = win[~np.isnan(win).any(axis=1)]
        if len(synced) < min_obs:
            continue
        out.iloc[i] = _gfevd_share(synced, horizon=horizon, lag=lag)
    return out


def _bivariate_components(data: np.ndarray, horizon: int, lag: int):
    """``(share, A, B, local_var)`` for the bivariate BTC->local decomposition.

    ``share = A/(A+B)``; ``A`` = BTC's standardized contribution
    ``num[1,0]/sigma_btc``; ``B`` = local own-shock contribution
    ``num[1,1]/sigma_local``; ``local_var`` = ``den[1]``. NaNs if the VAR fails.
    """
    m = _gfevd_matrices(data, horizon, lag)
    if m is None:
        return (float("nan"),) * 4
    gfevd, num, den, sigma_diag = m
    A = num[1, 0] / sigma_diag[0]
    B = num[1, 1] / sigma_diag[1]
    return float(gfevd[1, 0]), float(A), float(B), float(den[1])


def compute_rolling_components(
    btc_rv: pd.Series,
    local: pd.Series,
    window: int = 200,
    horizon: int = 10,
    lag: int = 1,
    min_obs: int = 180,
) -> pd.DataFrame:
    """Rolling bivariate decomposition. Columns: ``spillover, A, B, local_var``.

    Same windowing/synchronization contract as :func:`compute_rolling_dy`.
    """
    df = pd.concat([btc_rv.rename("btc"), local.rename("local")], axis=1).sort_index()
    trading = df.loc[df["local"].notna()]
    cols = ["spillover", "A", "B", "local_var"]
    out = pd.DataFrame(np.nan, index=trading.index, columns=cols, dtype=float)
    values = trading[["btc", "local"]].to_numpy()
    for i in range(window - 1, len(trading)):
        win = values[i - window + 1 : i + 1]
        synced = win[~np.isnan(win).any(axis=1)]
        if len(synced) < min_obs:
            continue
        out.iloc[i] = _bivariate_components(synced, horizon=horizon, lag=lag)
    return out


def _gfevd_3var_shares(data: np.ndarray, horizon: int, lag: int):
    """``(btc_share, global_share)`` of the local variable's variance.

    ``data`` columns are ordered ``[btc, global, local]``; the local target is
    row 2. NaNs if the VAR cannot be estimated.
    """
    m = _gfevd_matrices(data, horizon, lag)
    if m is None:
        return float("nan"), float("nan")
    gfevd = m[0]
    return float(gfevd[2, 0]), float(gfevd[2, 1])


def compute_rolling_3var(
    btc_rv: pd.Series,
    global_vol: pd.Series,
    local: pd.Series,
    window: int = 200,
    horizon: int = 10,
    lag: int = 1,
    min_obs: int = 180,
) -> pd.DataFrame:
    """Rolling trivariate ``[btc, global, local]`` GFEVD shares of local variance.

    Columns: ``btc_share, global_share``. Same windowing contract as
    :func:`compute_rolling_dy`; a window is used only where all three series are
    observed.
    """
    df = pd.concat(
        [btc_rv.rename("btc"), global_vol.rename("glob"), local.rename("local")], axis=1
    ).sort_index()
    trading = df.loc[df["local"].notna()]
    out = pd.DataFrame(
        np.nan, index=trading.index, columns=["btc_share", "global_share"], dtype=float
    )
    values = trading[["btc", "glob", "local"]].to_numpy()
    for i in range(window - 1, len(trading)):
        win = values[i - window + 1 : i + 1]
        synced = win[~np.isnan(win).any(axis=1)]
        if len(synced) < min_obs:
            continue
        out.iloc[i] = _gfevd_3var_shares(synced, horizon=horizon, lag=lag)
    return out


def _window_spectral_radius(data: np.ndarray, lag: int) -> float:
    """Companion-matrix spectral radius of a VAR(lag) on ``data``, or nan if it fails.

    ``rho = 1 / min(|characteristic roots|) = max |eigenvalue of the companion
    matrix|``. ``rho >= 1`` iff the fitted VAR is non-stationary (statsmodels'
    ``is_stable()`` is False); for a VAR(1) this is just ``max|eig(coefs[0])|``.
    """
    try:
        res = VAR(data).fit(lag)
        return float(1.0 / np.min(np.abs(res.roots)))
    except Exception:
        return float("nan")


def compute_rolling_spectral_radius(
    btc_rv: pd.Series,
    local: pd.Series,
    window: int = 200,
    lag: int = 1,
    min_obs: int = 180,
) -> pd.Series:
    """Rolling companion spectral radius of the bivariate ``[btc_rv, local]`` VAR(lag).

    Mirrors :func:`compute_rolling_dy`'s windowing/synchronization contract exactly
    (same trailing ``window`` of the local market's trading days, same ``min_obs``
    synchronized-row floor, value dated at the window's end), so each radius aligns
    one-to-one with that window's spillover share. The value is the window's
    non-stationarity diagnostic: ``>= 1`` flags an explosive VAR whose generalized
    FEVD is unreliable. ``nan`` where the window is too sparse or the VAR fails --
    the same windows the share is ``nan`` on. ``horizon`` is irrelevant here (the
    radius is a property of the VAR coefficients, not the FEVD horizon).
    """
    df = pd.concat([btc_rv.rename("btc"), local.rename("local")], axis=1).sort_index()
    trading = df.loc[df["local"].notna()]
    out = pd.Series(np.nan, index=trading.index, name="spectral_radius", dtype=float)
    values = trading[["btc", "local"]].to_numpy()
    for i in range(window - 1, len(trading)):
        win = values[i - window + 1 : i + 1]
        synced = win[~np.isnan(win).any(axis=1)]
        if len(synced) < min_obs:
            continue
        out.iloc[i] = _window_spectral_radius(synced, lag)
    return out
