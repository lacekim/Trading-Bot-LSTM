"""Natenberg, Option Volatility and Pricing: engines copied from the book's own formulas (Figure 18-6 Black-Scholes and sensitivities, Figure 19-13 binomial model, Appendix B statistics).
New file; reads saved bars, changes nothing audited.

Everything here is the book's formula as printed.  Where the book's printed formula fails its own finite-difference check, nat_checks reports that; the formula here is NOT altered.

Conventions of Figure 18-6 (kept): theta per year (divide by 365 for a day), vega and rho per 1.00 of volatility / rate (divide by 100 for one point).
b = r stock, b = 0 Black (futures, stock-type settlement), b = r = 0 futures-type settlement, b = r - rf Garman-Kohlhagen, b = r - q dividend yield.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

SQ2PI = np.sqrt(2 * np.pi)


def N(x):
    return ndtr(x)


def n(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def d1d2(S, X, t, r, sig, b=None):
    b = r if b is None else b
    S, X, t, sig = (np.asarray(v, float) for v in (S, X, t, sig))
    d1 = (np.log(S / X) + (b + sig ** 2 / 2) * t) / (sig * np.sqrt(t))
    d2 = (np.log(S / X) + (b - sig ** 2 / 2) * t) / (sig * np.sqrt(t))
    return d1, d2


def bs(S, X, t, r, sig, b=None):
    """Figure 18-6: C = S e^{(b-r)t} N(d1) - X e^{-rt} N(d2);  P = X e^{-rt} N(-d2) - S e^{(b-r)t} N(-d1)."""
    b = r if b is None else b
    d1, d2 = d1d2(S, X, t, r, sig, b)
    C = S * np.exp((b - r) * t) * N(d1) - X * np.exp(-r * t) * N(d2)
    P = X * np.exp(-r * t) * N(-d2) - S * np.exp((b - r) * t) * N(-d1)
    return C, P


def greeks(S, X, t, r, sig, b=None):
    """Every row of Figure 18-6 (and its continuation), call and put, exactly as printed.  Returns dict name -> (call, put); where the book gives one formula for both the two are equal."""
    b = r if b is None else b
    S, X, t, sig = (np.asarray(v, float) for v in (S, X, t, sig))
    d1, d2 = d1d2(S, X, t, r, sig, b)
    E = np.exp((b - r) * t)
    C, P = bs(S, X, t, r, sig, b)
    sq = np.sqrt(t)
    g = {}
    g["delta"] = (E * N(d1), E * (N(d1) - 1))
    gam = E * n(d1) / (S * sig * sq)
    g["gamma"] = (gam, gam)
    th_c = -S * E * n(d1) * sig / (2 * sq) - (b - r) * S * E * N(d1) - r * X * np.exp(-r * t) * N(d2)
    th_p = -S * E * n(d1) * sig / (2 * sq) + (b - r) * S * E * N(d1) + r * X * np.exp(-r * t) * N(-d2)   # as printed: N(d1) in the second term
    g["theta"] = (th_c, th_p)
    vega = S * E * n(d1) * sq
    g["vega"] = (vega, vega)
    if np.all(np.asarray(b) != 0):
        g["rho"] = (t * X * np.exp(-r * t) * N(d2), -t * X * np.exp(-r * t) * N(-d2))
    else:
        g["rho"] = (-t * C, -t * P)
    g["phi"] = (-t * S * E * N(d1), t * S * E * N(-d1))
    g["lambda"] = (g["delta"][0] * S / C, g["delta"][1] * S / P)
    van = -E * n(d1) * d2 / sig
    g["vanna"] = (van, van)
    g["charm"] = (-E * (n(d1) * (b / (sig * sq) - d2 / (2 * t)) + (b - r) * N(d1)), -E * (n(d1) * (b / (sig * sq) - d2 / (2 * t)) - (b - r) * N(d1)))
    sp = -gam / S * (1 + d1 / (sig * sq))
    g["speed"] = (sp, sp)
    col = gam * (r - b + b * d1 / (sig * sq) + (1 - d1 * d2) / (2 * t))
    g["color"] = (col, col)
    vol = vega * (d1 * d2 / sig)
    g["volga"] = (vol, vol)
    vd = vega * (r - b + b * d1 / (sig * sq) - (1 - d1 * d2) / (2 * t))
    g["vegadecay"] = (vd, vd)
    zom = gam * ((d1 * d2 - 1) / sig)
    g["zomma"] = (zom, zom)
    return g


def implied_vol(price, S, X, t, r, kind="call", b=None, lo=1e-4, hi=5.0):
    """Bisection on the book's own value formula (the book prints no solver; the book defines implied volatility as the volatility that returns the price)."""
    f = (lambda s: bs(S, X, t, r, s, b)[0]) if kind == "call" else (lambda s: bs(S, X, t, r, s, b)[1])
    a, c = lo, hi
    for _ in range(200):
        m = 0.5 * (a + c)
        if f(m) > price:
            c = m
        else:
            a = m
    return 0.5 * (a + c)


# ------------------------------------------------------------------ Chapter 19: binomial (Figure 19-13)
def tree(S, X, t, r, sig, n_, kind="call", b=None, american=False, u=None, d=None, div=None):
    """Binomial model of Figure 19-13.  u = e^{sig sqrt(t/n)}, d = 1/u, p = [(1 + b t/n) - d] / (u - d); node values C_ij = [p C_{i+1,j+1} + (1-p) C_{i+1,j}] / (1 + r t/n).
    i = period (0..n), j = number of up moves (0..i).  u (and d = 1/u) may be given instead of sig (the book's three-period example sets u = 1.05).
    div = (period i_div, amount): the stock price of every node of period >= i_div is reduced by the amount (Figure 19-7 style, 'reduce all subsequent values')."""
    b = r if b is None else b
    h = t / n_
    if u is None:
        u = np.exp(sig * np.sqrt(h))
    d = 1.0 / u if d is None else d
    p = ((1 + b * h) - d) / (u - d)
    Sx = [np.array([S * u ** j * d ** (i - j) for j in range(i + 1)]) for i in range(n_ + 1)]
    if div is not None:
        for i in range(div[0], n_ + 1):
            Sx[i] = Sx[i] - div[1]
    pay = (lambda s: np.maximum(s - X, 0.0)) if kind == "call" else (lambda s: np.maximum(X - s, 0.0))
    V = [None] * (n_ + 1)
    V[n_] = pay(Sx[n_])
    for i in range(n_ - 1, -1, -1):
        v = (p * V[i + 1][1:] + (1 - p) * V[i + 1][:-1]) / (1 + r * h)
        if american:
            v = np.maximum(v, pay(Sx[i]))
        V[i] = v
    # delta and gamma and theta exactly as printed in Figure 19-13
    dl = [None] * (n_ + 1)
    for i in range(n_):
        dl[i] = (V[i + 1][1:] - V[i + 1][:-1]) / (Sx[i + 1][1:] - Sx[i + 1][:-1])
    gm = [None] * (n_ + 1)
    for i in range(n_ - 1):
        gm[i] = (dl[i + 1][1:] - dl[i + 1][:-1]) / (Sx[i + 1][1:] - Sx[i + 1][:-1])
    th = [None] * (n_ + 1)
    for i in range(n_ - 1):
        th[i] = (V[i + 2][1:-1] - V[i]) / t
    return {"S": Sx, "V": V, "delta": dl, "gamma": gm, "theta": th, "u": u, "d": d, "p": p, "h": h}


def tree_value(*a, **k):
    return float(tree(*a, **k)["V"][0][0])


def european_sum(S, X, t, r, sig, n_, kind="call", b=None, u=None):
    """The closed sum of Figure 19-13: value = 1/(1 + r t/n)^n * sum_j n!/(j!(n-j)!) p^j (1-p)^(n-j) max[S u^j d^(n-j) - X, 0]."""
    from math import comb
    b = r if b is None else b
    h = t / n_
    uu = np.exp(sig * np.sqrt(h)) if u is None else u
    d = 1.0 / uu
    p = ((1 + b * h) - d) / (uu - d)
    tot = 0.0
    for j in range(n_ + 1):
        s = S * uu ** j * d ** (n_ - j)
        pay = max(s - X, 0.0) if kind == "call" else max(X - s, 0.0)
        tot += comb(n_, j) * p ** j * (1 - p) ** (n_ - j) * pay
    return tot / (1 + r * h) ** n_


# ------------------------------------------------------------------ Appendix B
def rate_of_return(PV, r, t, n_=1):
    """Appendix B: simple, compound (n per year) and continuous future values."""
    return {"simple": PV * (1 + r * t), "compound": PV * (1 + r / n_) ** (n_ * t), "continuous": PV * np.exp(r * t)}


def moments(x):
    """Appendix B: population mean, st. dev., skewness m3/m2^1.5 and kurtosis m4/m2^2 (normal = 3)."""
    x = np.asarray(x, float)
    mu = x.mean()
    m2, m3, m4 = (np.mean((x - mu) ** k) for k in (2, 3, 4))
    return {"mean": mu, "std_pop": np.sqrt(m2), "std_sample": np.sqrt(m2 * len(x) / (len(x) - 1)), "skew": m3 / m2 ** 1.5, "kurt": m4 / m2 ** 2, "m2": m2, "m3": m3, "m4": m4}


def vol_appendix_b(prices, dt_years, mean="actual", sample=False):
    """Appendix B volatility: sigma = sqrt( sum (x_i - mu)^2 / n ) / sqrt(t)  with x_i = ln(p_i / p_{i-1}); mu = 0 for the zero-mean variants; n-1 for the estimated st. dev."""
    p = np.asarray(prices, float)
    x = np.diff(np.log(p))
    mu = x.mean() if mean == "actual" else 0.0
    k = len(x) - 1 if sample else len(x)
    return float(np.sqrt(np.sum((x - mu) ** 2) / k) / np.sqrt(dt_years))


def forward_vol(s1, t1, s2, t2):
    """Chapter 20: sigma_f = sqrt( (sigma2^2 t2 - sigma1^2 t1) / (t2 - t1) )."""
    return float(np.sqrt((s2 ** 2 * t2 - s1 ** 2 * t1) / (t2 - t1)))
