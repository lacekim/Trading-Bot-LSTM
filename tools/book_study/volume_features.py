"""Volume-based microstructure features from Chapter 19 of the book, computed on HOURLY BARS (numpy only, no project imports).

The book builds these from tick / trade data.  Only bars are available here, so each feature is a bar-level proxy (declared in
meta_eval.DEVIATIONS through run_meta_volume.py).  All four were fixed before any result was seen.  Everything is causal: the value at bar t
uses bars up to and including t.  Windows are counted in bars (positions), like the other hourly features.

  dv_t   = close_t * volume_t                      dollar volume of the bar
  r_t    = log(close_t / close_{t-1})              bar return
  b_t    = sign(close_t - close_{t-1}), tick rule: a zero change repeats the previous sign; 0 until the first non-zero change

  kyle_lam   (Ch.19 Kyle's lambda, bar version)    OLS slope (with intercept) of r_i on b_i * dv_i over the trailing W_REG = 168 bars,
                                                   times the mean dv of the same bars (units: log return per average hour of flow)
  amihud_lam (Ch.19 Amihud's lambda, bar version)  OLS slope (with intercept) of |r_i| on dv_i over the same 168 bars, times the same mean dv
  vpin       (Ch.19 VPIN, bar version)             sum over the trailing W_FLOW = 24 bars of |V_buy - V_sell| / sum of volume, with the buy share
                                                   of each bar from bulk volume classification: V_buy = V * Phi(r_i / sigma_i),
                                                   sigma_i = population std of r over the 168 bars ending at i (time bars instead of volume buckets)
  dvol_z     (volume level, not a book snippet)    z-score of m_t = log(1 + mean dv over 24 bars) against its own trailing W_Z = 720 values
                                                   (population std)

A value is NaN when its window is incomplete, contains a missing volume, or its denominator is zero (zero variance of the regressor,
zero summed volume, zero sigma).  Hasbrouck's lambda needs trade-level signed flow and is not built.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

VOL_FEATURES = ["kyle_lam", "amihud_lam", "vpin", "dvol_z"]
W_REG, W_FLOW, W_Z = 168, 24, 720
_CHUNK = 4000
_erf = np.frompyfunc(math.erf, 1, 1)


def load_volumes(candle_dir, symbol: str, timeframe: str = "1h") -> pd.Series:
    """Volume column on exactly the bars venue_loader keeps: close present and > 0, duplicate timestamps keep the last row, sorted by date."""
    from pathlib import Path

    d = pd.read_csv(Path(candle_dir) / f"{symbol}_{timeframe}.csv", usecols=["date", "close", "volume"], keep_default_na=False, na_values=[""])
    d["date"] = pd.to_datetime(d["date"])
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["close"])
    d = d[d["close"] > 0]
    d = d.drop_duplicates("date", keep="last").sort_values("date")
    return pd.Series(d["volume"].astype(float).to_numpy(), index=pd.DatetimeIndex(d["date"]))


def tick_sign(close: np.ndarray) -> np.ndarray:
    """Tick rule on bar closes: sign of the change; a zero change repeats the previous sign; 0 before the first non-zero change."""
    c = np.asarray(close, dtype=np.float64)
    b = np.zeros(len(c))
    cur = 0.0
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        if d > 0:
            cur = 1.0
        elif d < 0:
            cur = -1.0
        b[i] = cur
    return b


def _windows(a: np.ndarray, w: int):
    return np.lib.stride_tricks.sliding_window_view(a, w)


def _ols_slope(x: np.ndarray, y: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Slope of y on x (with intercept) and mean of x over every trailing window of w values; out[t] uses x[t-w+1..t]; NaN where undefined."""
    n = len(x)
    slope = np.full(n, np.nan)
    xmean = np.full(n, np.nan)
    if n < w:
        return slope, xmean
    X, Y = _windows(x, w), _windows(y, w)
    for a in range(0, X.shape[0], _CHUNK):
        xc, yc = X[a:a + _CHUNK], Y[a:a + _CHUNK]
        xm = xc.mean(axis=1)
        ym = yc.mean(axis=1)
        dx = xc - xm[:, None]
        cov = (dx * (yc - ym[:, None])).sum(axis=1)
        var = (dx * dx).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(var > 0, cov / var, np.nan)
        slope[w - 1 + a: w - 1 + a + len(s)] = s
        xmean[w - 1 + a: w - 1 + a + len(s)] = xm
    return slope, xmean


def _roll_sum(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= w:
        W = _windows(a, w)
        for k in range(0, W.shape[0], _CHUNK):
            out[w - 1 + k: w - 1 + k + len(W[k:k + _CHUNK])] = W[k:k + _CHUNK].sum(axis=1)
    return out


def _roll_mean_std(a: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std of every trailing window of w values (two-pass, so no cancellation)."""
    n = len(a)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    if n < w:
        return mean, std
    W = _windows(a, w)
    for k in range(0, W.shape[0], _CHUNK):
        c = W[k:k + _CHUNK]
        m = c.mean(axis=1)
        s = np.sqrt(((c - m[:, None]) ** 2).mean(axis=1))
        mean[w - 1 + k: w - 1 + k + len(m)] = m
        std[w - 1 + k: w - 1 + k + len(m)] = s
    return mean, std


def volume_features(close, volume) -> dict:
    """The four features for every bar of one symbol.  close > 0; volume >= 0 or NaN (same length)."""
    c = np.asarray(close, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    n = len(c)
    out = {k: np.full(n, np.nan) for k in VOL_FEATURES}
    if n < 3:
        return out
    r = np.full(n, np.nan)
    r[1:] = np.diff(np.log(c))
    dv = c * v
    b = tick_sign(c)

    # Kyle and Amihud: windows of the last W_REG bars with a defined return (bars 1 .. t)
    if n > W_REG:
        k_slope, _ = _ols_slope(b[1:] * dv[1:], r[1:], W_REG)
        _, dv_mean = _ols_slope(dv[1:], r[1:], W_REG)
        a_slope, _ = _ols_slope(dv[1:], np.abs(r[1:]), W_REG)
        out["kyle_lam"][1:] = k_slope * dv_mean
        out["amihud_lam"][1:] = a_slope * dv_mean

    # VPIN: sigma_i over the W_REG returns ending at bar i, bulk volume classification, then W_FLOW bars
    if n > W_REG:
        _, sig_r = _roll_mean_std(r[1:], W_REG)
        sig = np.full(n, np.nan)
        sig[1:] = sig_r
        with np.errstate(divide="ignore", invalid="ignore"):
            z = r / (sig * math.sqrt(2.0))
        ok = np.isfinite(z) & (sig > 0)
        imb = np.full(n, np.nan)
        imb[ok] = np.abs(_erf(z[ok]).astype(np.float64))          # |2 Phi(r / sigma) - 1|
        num = _roll_sum(v * imb, W_FLOW)
        den = _roll_sum(v, W_FLOW)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["vpin"] = np.where(den > 0, num / den, np.nan)

    # dollar-volume level
    mdv = _roll_sum(dv, W_FLOW) / W_FLOW
    m = np.log1p(mdv)
    mean_m, std_m = _roll_mean_std(m, W_Z)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dvol_z"] = np.where(std_m > 0, (m - mean_m) / std_m, np.nan)
    return out
