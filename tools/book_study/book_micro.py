"""Chapter 17 and 19 features that the book defines by formula and that can be built from the hourly reference series (high, low, close, volume) of the dollar-bar folders.  New file.
Every one is computed from rows up to and including the row's own hour (the row is stamped with the END of its hour), then read at the last completed hour at or before an event (bar_events.asof).
Window length: 100 hourly rows for the rolling estimates, the same 100 as the project's SADF and entropy windows; 24 rows for the Becker-Parkinson mean, the same 24 as the Corwin-Schultz mean.  Both are the project's choices (the book leaves them
to the user); they are fixed here before any result and are not searched.

  bp_sigma     Snippet 19.2 (Becker-Parkinson high-low volatility) from the Corwin-Schultz beta and gamma of Snippet 19.1 (same 24-row mean of the two-row sum of log(H/L)^2)
  roll         Equations 82-83: c = sqrt(max(0, -cov(dp_t, dp_{t-1}))) over the window (the bid-ask-bounce estimate of half the spread, in price units)
  kyle         Equation 90: dp_t = lambda * (b_t V_t) + e_t, no intercept, b_t = tick rule (Equation 81) on the row's close, V_t = the row's volume
  amihud       Equation 91: |dlog p_t| = lambda * (p_t V_t) + e_t (dollar volume of the row)
  hasbrouck    Equation 92: dlog p_t = lambda * b_t sqrt(p_t V_t) + e_t
  csw          Section 17.3.2, Chu-Stinchcombe-White CUSUM on levels: S_{n,t} = (y_t - y_n) / (sigma_t sqrt(t - n)), y = log close, sigma_t^2 = (t-1)^-1 sum (dy_i)^2; the feature is sup over n in the window
The book computes the three lambdas from trades inside each bar; we hold no trades before 2026 (Kraken) or at all (Alpaca), so the 'bar' is the hourly row, b_t is the tick rule on its close, and the row's volume is the whole V_t.  This is a declared deviation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import micro_features as MF

W_ROLL = 100
SL_BP = 24
NAMES = ("bp_sigma", "roll", "kyle", "amihud", "hasbrouck", "csw")


def tick_rule(close: np.ndarray) -> np.ndarray:
    """Equation 81: b_t = sign(dp_t), carried forward when dp_t = 0; the first row has no sign (0)."""
    d = np.diff(close, prepend=np.nan)
    b = np.sign(np.nan_to_num(d, nan=0.0))
    out = np.zeros(len(b))
    last = 0.0
    for i in range(len(b)):                       # carry-forward is path dependent; hourly series are short enough
        if b[i] != 0.0:
            last = b[i]
        out[i] = last
    return out


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sum of x[t-w+1..t]; NaN where the window is incomplete or holds a NaN."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    bad = np.isnan(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(bad, 0.0, x))])
    cb = np.concatenate([[0], np.cumsum(bad)])
    t = np.arange(w - 1, n)
    ok = (cb[t + 1] - cb[t + 1 - w]) == 0
    out[t] = np.where(ok, cs[t + 1] - cs[t + 1 - w], np.nan)
    return out


def bp_sigma(high: np.ndarray, low: np.ndarray, sl: int = SL_BP) -> np.ndarray:
    """Snippet 19.2 on top of Snippet 19.1's beta (rolling mean sl of the 2-row sum of log(H/L)^2) and gamma (log of the 2-row high over the 2-row low, squared)."""
    h, l = np.asarray(high, float), np.asarray(low, float)
    n = len(h)
    out = np.full(n, np.nan)
    if n <= sl:
        return out
    invalid = ~(h > 0) | ~(l > 0) | (h < l)
    h, l = np.where(invalid, np.nan, h), np.where(invalid, np.nan, l)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(h / l) ** 2
        b2 = np.full(n, np.nan)
        b2[1:] = hl[1:] + hl[:-1]
        beta = _rolling_sum(b2, sl) / sl
        h2, l2 = np.full(n, np.nan), np.full(n, np.nan)
        h2[1:], l2[1:] = np.maximum(h[1:], h[:-1]), np.minimum(l[1:], l[:-1])
        gamma = np.log(h2 / l2) ** 2
        k2 = (8 / np.pi) ** .5
        den = 3 - 2 * 2 ** .5
        sig = (2 ** -.5 - 1) * np.sqrt(beta) / (k2 * den) + np.sqrt(gamma / (k2 ** 2 * den))
        sig = np.where(sig < 0, 0.0, sig)
    sig[:sl] = np.nan
    return sig


def _slope_through_origin(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    sxy, sxx = _rolling_sum(x * y, w), _rolling_sum(x * x, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sxx > 0, sxy / sxx, np.nan)


def roll_measure(close: np.ndarray, w: int = W_ROLL) -> np.ndarray:
    dp = np.diff(close, prepend=np.nan)
    d0, d1 = dp[1:], dp[:-1]                                       # pairs (dp_t, dp_{t-1}) at t = 1..n-1
    n = len(close)
    out = np.full(n, np.nan)
    pr = np.full(n, np.nan)
    pr[1:] = d0 * d1
    s_pr = _rolling_sum(pr, w)
    s0, s1 = _rolling_sum(np.concatenate([[np.nan], d0]), w), _rolling_sum(np.concatenate([[np.nan], d1]), w)
    with np.errstate(invalid="ignore"):
        cov = s_pr / w - (s0 / w) * (s1 / w)
        out = np.sqrt(np.maximum(0.0, -cov))
    out[~np.isfinite(cov)] = np.nan
    return out


def lambdas(close: np.ndarray, vol: np.ndarray, w: int = W_ROLL):
    c, v = np.asarray(close, float), np.asarray(vol, float)
    b = tick_rule(c)
    dp = np.diff(c, prepend=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.diff(np.log(c), prepend=np.nan)
        dollar = c * v
        kyle = _slope_through_origin(b * v, dp, w)
        amihud = _slope_through_origin(dollar, np.abs(dlog), w)
        hasb = _slope_through_origin(b * np.sqrt(dollar), dlog, w)
    return kyle, amihud, hasb


def csw_sup(close: np.ndarray, w: int = W_ROLL) -> np.ndarray:
    """Section 17.3.2: sup over n in [t-w, t-1] of (y_t - y_n) / (sigma_t sqrt(t - n)); sigma_t^2 = (t-1)^-1 sum_{i<=t} (dy_i)^2 over the whole series up to t."""
    y = np.log(np.asarray(close, float))
    n = len(y)
    out = np.full(n, np.nan)
    if n <= w + 2:
        return out
    dy = np.diff(y, prepend=np.nan)
    cs = np.cumsum(np.nan_to_num(dy[1:] ** 2, nan=0.0))            # dy_2..dy_t  (index i -> t = i + 2 in 1-based)
    t_idx = np.arange(n)
    sig2 = np.full(n, np.nan)
    sig2[2:] = cs[1:] / t_idx[2:]                                  # (t-1)^-1 with t 1-based = idx+1 -> idx
    sig = np.sqrt(sig2)
    best = np.full(n, -np.inf)
    for k in range(1, w + 1):
        s = np.full(n, np.nan)
        s[k:] = (y[k:] - y[:-k]) / (sig[k:] * np.sqrt(k))
        best = np.fmax(best, s)                                    # fmax ignores NaN
    best[~np.isfinite(best)] = np.nan
    best[:w] = np.nan
    return best


def ref_features(ref_high, ref_low, ref_close, ref_vol) -> dict:
    c = np.asarray(ref_close, float)
    k, a, h = lambdas(c, ref_vol)
    return {"bp_sigma": bp_sigma(ref_high, ref_low), "roll": roll_measure(c), "kyle": k, "amihud": a, "hasbrouck": h, "csw": csw_sup(c)}
