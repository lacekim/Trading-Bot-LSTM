"""Chapter 5: fractional differentiation with a fixed-width window (Snippets 5.1, 5.3) and the search for the smallest d whose
differenced log-price passes the augmented Dickey-Fuller test (Snippet 5.4).  numpy only; no project imports.

ADF here is the book's setting: regression with a constant, lagged level and `lags` lagged differences (the book uses maxlag=1, no automatic
lag selection), 5% critical value from the MacKinnon response surface for the constant-only case.  statsmodels is not required.
"""
from __future__ import annotations

import numpy as np

# MacKinnon (2010) response-surface coefficients, regression 'c', one series, 5% level: crit = b0 + b1/T + b2/T^2 + b3/T^3
_CRIT5_C = (-2.86154, -2.8903, -4.234, -40.040)


def weights_ffd(d: float, thres: float = 1e-5, max_len: int = 100000) -> np.ndarray:
    """Weights of the fixed-width window, oldest first, ending with w0 = 1 (Snippet 5.3's getWeights_FFD): w_k = -w_{k-1}/k * (d-k+1),
    stopped when |w_k| < thres."""
    w = [1.0]
    k = 1
    while k < max_len:
        wk = -w[-1] / k * (d - k + 1)
        if abs(wk) < thres:
            break
        w.append(wk)
        k += 1
    return np.array(w[::-1], dtype=np.float64)


def frac_diff_ffd(x, d: float, thres: float = 1e-5) -> np.ndarray:
    """Fixed-width-window fractional difference of a 1-D array. Element t uses x[t-width .. t]; the first `width` elements are NaN.
    A NaN anywhere in a window gives NaN (the book forward-fills first; callers here pass gap-free daily data)."""
    x = np.asarray(x, dtype=np.float64)
    w = weights_ffd(d, thres)
    width = len(w) - 1
    out = np.full(len(x), np.nan)
    if len(x) <= width:
        return out
    # sliding dot product via a strided view
    win = np.lib.stride_tricks.sliding_window_view(x, width + 1)
    out[width:] = win @ w
    return out


def adf_crit5(nobs: int) -> float:
    t = float(nobs)
    b0, b1, b2, b3 = _CRIT5_C
    return b0 + b1 / t + b2 / t ** 2 + b3 / t ** 3


def adf_stat(y, lags: int = 1):
    """ADF t-statistic of the lagged level in:  dy_t = a + g*y_{t-1} + sum_{i=1..lags} c_i*dy_{t-i} + e_t.  Returns (stat, nobs)."""
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    dy = np.diff(y)
    n = len(dy) - lags
    if n <= lags + 3:
        return float("nan"), 0
    target = dy[lags:]
    cols = [np.ones(n), y[lags:-1] if lags >= 0 else y[:-1]]
    # y_{t-1} aligned with target dy_t = y_t - y_{t-1}: y index (lags .. len(y)-2)
    cols[1] = y[lags:len(y) - 1]
    for i in range(1, lags + 1):
        cols.append(dy[lags - i:len(dy) - i])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    dof = n - X.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(beta[1] / np.sqrt(cov[1, 1])), n


def min_d(log_price_daily, grid=None, thres: float = 0.01, lags: int = 1):
    """Smallest d on `grid` whose FFD(d) of the (daily log-price) series has ADF stat below the 5% critical value (Snippet 5.4).
    Returns dict(d, adf, crit, corr, table) where table lists (d, adf, crit, corr, nobs) for every d tried; d = 1.0 if none passes."""
    x = np.asarray(log_price_daily, dtype=np.float64)
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2) if grid is None else grid
    table = []
    chosen = None
    for d in grid:
        f = frac_diff_ffd(x, float(d), thres)
        ok = np.isfinite(f)
        if ok.sum() < 50:
            table.append((float(d), float("nan"), float("nan"), float("nan"), int(ok.sum())))
            continue
        stat, nobs = adf_stat(f[ok], lags)
        crit = adf_crit5(nobs)
        corr = float(np.corrcoef(x[ok], f[ok])[0, 1])
        table.append((float(d), stat, crit, corr, nobs))
        if chosen is None and np.isfinite(stat) and stat < crit:
            chosen = (float(d), stat, crit, corr)
    if chosen is None:
        chosen = (1.0, float("nan"), float("nan"), float("nan"))
    return {"d": chosen[0], "adf": chosen[1], "crit": chosen[2], "corr": chosen[3], "table": table}
