"""Features from Chapters 17-19 of the book, numpy only (no project imports).

SADF (Ch.17, Snippets 17.1-17.4): supremum over start points of the augmented Dickey-Fuller t-statistic of the lagged level, on a trailing
window of log prices.  Entropy (Ch.18, Snippets 18.1, 18.3, 18.4): plug-in entropy rate and the Kontoyiannis / Lempel-Ziv estimate of a
message of signs.  Corwin-Schultz bid-ask spread (Ch.19, Snippet 19.1) from High / Low.

Everything here is causal: the value at index t uses data up to and including index t.
Declared choices (not in the condensed edition, or changed to make hourly data feasible) are listed in meta_eval.DEVIATIONS.
"""
from __future__ import annotations

import math

import numpy as np

SADF_MIN_DET = 1e-14          # a start whose 3x3 normal-equation matrix is this close to singular is skipped (flat prices)


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 17: SADF
# ---------------------------------------------------------------------------------------------------------------------
def sadf_literal(logp, minsl: int, lags: int = 1):
    """Snippets 17.1-17.4 for one window of log prices, constant 'c', written the way the book writes it (one lstsq-style fit per start).
    Regressors per row t: lagged level, `lags` lagged differences, constant; target: the difference.  Returns the supremum over start points
    0 .. rows+lags-minsl of the t-statistic of the lagged level (starts with a singular design are skipped)."""
    p = np.asarray(logp, dtype=np.float64)
    dy = np.diff(p)                                         # series_ = series.diff().dropna()
    n_rows = len(dy) - lags
    # lagDF(series_, lags): columns shift 0..lags of the differences; column 0 is replaced by the lagged level
    x = np.empty((n_rows, lags + 1))
    x[:, 0] = p[-n_rows - 1:-1]
    for k in range(1, lags + 1):
        x[:, k] = dy[lags - k: len(dy) - k]
    x = np.append(x, np.ones((n_rows, 1)), axis=1)
    y = dy[-n_rows:]
    best = None
    for start in range(0, n_rows + lags - minsl + 1):
        y_, x_ = y[start:], x[start:]
        xx = x_.T @ x_
        if abs(np.linalg.det(xx)) < SADF_MIN_DET:
            continue
        xxinv = np.linalg.inv(xx)
        b = xxinv @ (x_.T @ y_)
        err = y_ - x_ @ b
        var = (err @ err) / (x_.shape[0] - x_.shape[1]) * xxinv
        stat = b[0] / math.sqrt(var[0, 0])
        if best is None or stat > best:
            best = stat
    return float("nan") if best is None else float(best)


def sadf_windows(x, window: int, minsl: int, lags: int = 1, chunk: int = 1500):
    """SADF of every trailing window of `window` observations of x (log prices): out[t] uses x[t-window+1 .. t]; NaN before that.
    Same quantity as sadf_literal, computed for many windows at once with suffix sums of the normal equations.  The window is shifted by its
    last value (the constant absorbs the shift; the statistic is unchanged) to keep the 3x3 systems well conditioned.  lags must be 1."""
    assert lags == 1
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out
    wins = np.lib.stride_tricks.sliding_window_view(x, window)          # (n-window+1, window)
    n_rows = window - 1 - lags
    n_starts = n_rows + lags - minsl + 1
    rows_in = n_rows - np.arange(n_starts)                               # observations per start
    for a in range(0, len(wins), chunk):
        w = wins[a:a + chunk]
        w = w - w[:, -1:]
        lvl = w[:, 1:window - 1]                                         # y[t-1] for t = 2 .. window-1
        dl = w[:, 1:window - 1] - w[:, 0:window - 2]                     # y[t-1] - y[t-2]
        tgt = w[:, 2:window] - w[:, 1:window - 1]                        # y[t] - y[t-1]
        X = np.stack([lvl, dl, np.ones_like(lvl)], axis=2)               # (nw, rows, 3)
        O = X[:, :, :, None] * X[:, :, None, :]
        Xy = X * tgt[:, :, None]
        yy = tgt * tgt
        S_O = np.cumsum(O[:, ::-1], axis=1)[:, ::-1][:, :n_starts]       # suffix sums: rows s .. end
        S_Xy = np.cumsum(Xy[:, ::-1], axis=1)[:, ::-1][:, :n_starts]
        S_yy = np.cumsum(yy[:, ::-1], axis=1)[:, ::-1][:, :n_starts]
        det = np.linalg.det(S_O)
        bad = np.abs(det) < SADF_MIN_DET
        S_O = np.where(bad[..., None, None], np.eye(3), S_O)
        inv = np.linalg.inv(S_O)
        beta = np.einsum("wsij,wsj->wsi", inv, S_Xy)
        ssr = S_yy - np.einsum("wsi,wsi->ws", beta, S_Xy)
        var0 = np.maximum(ssr, 0.0) / (rows_in[None, :] - 3) * inv[:, :, 0, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = beta[:, :, 0] / np.sqrt(var0)
        stat = np.where(bad | ~np.isfinite(stat), np.nan, stat)
        all_nan = np.isnan(stat).all(axis=1)
        best = np.where(all_nan, np.nan, np.nanmax(np.where(np.isnan(stat), -np.inf, stat), axis=1))
        out[window - 1 + a: window - 1 + a + len(w)] = best
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 18: entropy
# ---------------------------------------------------------------------------------------------------------------------
def plugin_entropy_literal(msg: str, w: int):
    """Snippet 18.1 (Python 3): plug-in entropy rate over all words of length w that END before the last character
    (the snippet's loop runs i = w .. len-1 over msg[i-w:i], so the final character is never part of a word)."""
    lib = {}
    for i in range(w, len(msg)):
        lib.setdefault(msg[i - w:i], []).append(i - w)
    tot = float(len(msg) - w)
    pmf = {k: len(v) / tot for k, v in lib.items()}
    return -sum(p * math.log2(p) for p in pmf.values()) / w, pmf


def match_length_literal(msg: str, i: int, n: int):
    """Snippet 18.3, as written: longest prefix of msg[i:] that also starts somewhere in msg[i-n:i] (overlap allowed); returns (length + 1, substring)."""
    sub = ""
    for l in range(n):
        m1 = msg[i:i + l + 1]
        for j in range(i - n, i):
            if m1 == msg[j:j + l + 1]:
                sub = m1
                break
    return len(sub) + 1, sub


def konto_literal(msg: str):
    """Snippet 18.4 with window=None (expanding), as written: returns dict(h, r, num).  len(msg) must be even."""
    s, num = 0.0, 0
    for i in range(1, len(msg) // 2 + 1):
        l, _ = match_length_literal(msg, i, i)
        s += math.log2(i + 1) / l
        num += 1
    h = s / num
    return {"h": h, "r": 1 - h / math.log2(len(msg)), "num": num}


def konto_fast(msg: str) -> float:
    """Same value as konto_literal(msg)['h'], using str.find for the match search (a longer match implies every shorter one, so the search
    stops at the first failure)."""
    n2 = len(msg) // 2
    s = 0.0
    for i in range(1, n2 + 1):
        m = 0
        while m < i and msg.find(msg[i:i + m + 1], 0, i + m) != -1:
            m += 1
        s += math.log2(i + 1) / (m + 1)
    return s / n2


def sign_bits(x):
    """Binary encoding of the daily log-return series: 1 if the return is positive, else 0 (the condensed edition has no encoding section; this
    plain sign encoding is a declared choice).  out[k] is the bit of x[k]-x[k-1]; index 0 has no return (-1)."""
    x = np.asarray(x, dtype=np.float64)
    b = np.full(len(x), -1, dtype=np.int8)
    b[1:] = (np.diff(x) > 0).astype(np.int8)
    return b


def entropy_windows(x, window: int = 100, w: int = 2):
    """For every index k with a full message (the last `window` returns, k >= window): the plug-in entropy rate (word length w) of the message
    in time order, and the Kontoyiannis estimate of the REVERSED message (most recent character first; the snippet's docstring suggests this
    when the end of the message is the more relevant part).  Returns (plugin, konto) arrays, NaN where unavailable."""
    assert window % 2 == 0 and w == 2
    b = sign_bits(x)
    n = len(x)
    plug = np.full(n, np.nan)
    kon = np.full(n, np.nan)
    if n <= window:
        return plug, kon
    # plug-in, vectorised: words are the adjacent pairs of msg[0 .. window-2]; msg covers bits k-window+1 .. k
    code = np.zeros(n, dtype=np.int64)
    code[:-1] = 2 * b[:-1].astype(np.int64) + b[1:].astype(np.int64)      # pair starting at g
    cums = [np.concatenate([[0], np.cumsum((code == c) & (np.arange(n) >= 1))]) for c in range(4)]
    n_words = window - w
    for k in range(window, n):
        g0, g1 = k - window + 1, k - w                                    # pairs starting at g0 .. g1 inclusive (n_words of them)
        h = 0.0
        for c in range(4):
            cnt = cums[c][g1 + 1] - cums[c][g0]
            if cnt:
                p = cnt / n_words
                h -= p * math.log2(p)
        plug[k] = h / w
    s_all = "".join("1" if v > 0 else "0" for v in b)                      # b[0] = -1 is encoded as "0" but never lies in a window (k >= window)
    rev = s_all[::-1]
    for k in range(window, n):
        r0 = n - 1 - k
        kon[k] = konto_fast(rev[r0:r0 + window])
    return plug, kon


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 19: Corwin-Schultz
# ---------------------------------------------------------------------------------------------------------------------
def corwin_schultz(high, low, sl: int = 1):
    """Snippet 19.1 with modern rolling operations.  beta = rolling mean (window sl) of the 2-bar rolling sum of log(H/L)^2; gamma = log of
    the 2-bar high over the 2-bar low, squared; alpha as in the snippet (negative alphas set to 0); spread = 2(e^alpha - 1)/(1 + e^alpha).
    out[t] uses bars up to and including t; NaN for t < sl.  Bars with high < low or non-positive prices give NaN."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    n = len(h)
    out = np.full(n, np.nan)
    if n <= sl:
        return out
    invalid = ~(h > 0) | ~(l > 0) | (h < l)
    h = np.where(invalid, np.nan, h)
    l = np.where(invalid, np.nan, l)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(h / l) ** 2
        b2 = np.full(n, np.nan)
        b2[1:] = hl[1:] + hl[:-1]
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(b2, nan=0.0))])
        bad = np.concatenate([[0], np.cumsum(np.isnan(b2) | ~np.isfinite(b2))])
        beta = np.full(n, np.nan)
        t = np.arange(sl, n)
        ok = (bad[t + 1] - bad[t - sl + 1]) == 0                          # window b2[t-sl+1 .. t], all valid
        beta[t] = np.where(ok, (cs[t + 1] - cs[t - sl + 1]) / sl, np.nan)
        h2 = np.full(n, np.nan)
        l2 = np.full(n, np.nan)
        h2[1:] = np.maximum(h[1:], h[:-1])
        l2[1:] = np.minimum(l[1:], l[:-1])
        gamma = np.log(h2 / l2) ** 2
        den = 3 - 2 * 2 ** 0.5
        alpha = (2 ** 0.5 - 1) * np.sqrt(beta) / den - np.sqrt(gamma / den)
        alpha = np.where(alpha < 0, 0.0, alpha)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    out[:] = spread
    out[: sl] = np.nan
    return out
