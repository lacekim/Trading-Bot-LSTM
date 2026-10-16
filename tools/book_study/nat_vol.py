"""Natenberg's volatility (Appendix B, 'Volatility') as the replacement for the bot's volatility estimates.  New file.

The book: the annualised volatility is the sample standard deviation of x_i = ln(p_i / p_{i-1}) with mean 0, divided by sqrt(t) (t = years between prices):
    sigma = sqrt( sum x_i^2 / (n - 1) ) / sqrt(t)          (the 'estimated standard deviation, 0 mean' line of Figure B-2: 39.93%)
The bot works in per-bar and per-day units, so the factor 1/sqrt(t) is left out (sigma x sqrt(t) = sqrt( sum x^2 / (n - 1) )).
The book does not say how many observations n to use.  Assumption: the same count as the estimator replaced: n = 336 bars (bar_events.SPAN) for the per-bar sigma, n = 100 daily returns (Snippet 3.1's span) for the daily target.
Missing returns (a data hole) are skipped, as the replaced exponentially weighted estimator skips them: the window is the last n valid returns, and at a missing return the last value is carried.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY_S = 86400


def appendix_b_vol(r, n: int) -> np.ndarray:
    """Rolling sqrt( sum x^2 / (n-1) ) over the last n valid log returns r (NaN = missing).  NaN until n valid returns exist."""
    s = pd.Series(np.asarray(r, float))
    v = s.dropna()
    out = np.sqrt((v ** 2).rolling(n, min_periods=n).sum() / (n - 1))
    return out.reindex(s.index).ffill().to_numpy()


def daily_vol(t: np.ndarray, close: np.ndarray, n: int = 100) -> np.ndarray:
    """Snippet 3.1's bar selection (the last bar strictly before t - 1 day, rows kept as printed) with Natenberg's volatility of the one-day LOG return in place of the EWM std of the simple return."""
    m = len(t)
    ss = np.searchsorted(t, t - DAY_S, side="left")
    keep = ss > 0
    r = np.full(m, np.nan)
    r[keep] = np.log(close[keep] / close[ss[keep] - 1])
    first = int(np.argmax(keep)) if keep.any() else m
    out = np.full(m, np.nan)
    if first < m:
        s = pd.Series(r[first:])
        out[first:] = np.sqrt((s ** 2).rolling(n, min_periods=n).sum() / (n - 1)).to_numpy()
    return out
