"""Natenberg study: shared pieces (venue inputs from our data, position evaluator, item/check helpers).  New file.

Data sources are the ONLY thing that changes from the book: S is our last daily close, sigma our one-year annualised daily-log-return st. dev.  Everything else in an 'ours' run is the book's formula with the book's inputs
where the book gives them; where the book gives no number for a picture the input is stated in the entry (X = S, 3 months, r = 0 unless the entry says otherwise).
"""
from __future__ import annotations

import numpy as np

import nat_core as C
import writeup_core as W
from writeup_core import VENUES, SYM, tbl, fmt, save

ANN = {"Kraken": 365, "Alpaca": 252}     # assumption: trading days per year used to annualise daily returns (crypto trades every day)
T3M = 0.25                                # assumption for pictures that give no time to expiration
_ctx = {}


def ctx(v):
    if v not in _ctx:
        s = W.daily_close(v, SYM[v])
        lr = np.log(s).diff().dropna()
        yr = lr.iloc[-ANN[v]:]
        _ctx[v] = dict(venue=v, sym=SYM[v], close=s, ret=lr, S=float(s.iloc[-1]), sig=float(yr.std(ddof=1) * np.sqrt(ANN[v])), last=str(s.index[-1].date()), first=str(s.index[0].date()), n=len(s), ann=ANN[v])
    return _ctx[v]


def ctx_line(v):
    c = ctx(v)
    return f"{v} {c['sym']}: S = {c['S']:,.2f} (close of {c['last']}), sigma = {c['sig']*100:.2f}% (std of the last {c['ann']} daily log returns x sqrt({c['ann']}))"


def item(id_, page, title, basis, fig=None, table=None, text=None, note=None, checks=None, table2=None, table2_title=None, table3=None, table3_title=None):
    d = W.item(id_, title, basis, fig=fig, table=table, text=text, note=note)
    d["page"] = page
    d["checks"] = checks or []
    if table2:
        d["table2"] = table2
        d["table2_title"] = table2_title
    if table3:
        d["table3"] = table3
        d["table3_title"] = table3_title
    return d


def chk(what, book, ours, tol=None, kind="num", ok=None):
    """One recorded check: book = what the book prints; ours = what the saved run computes.  kind num (|ours-book| <= tol), bool (equal), text."""
    if ok is None:
        if kind == "num":
            ok = bool(abs(float(ours) - float(book)) <= tol)
        else:
            ok = bool(ours == book)
    return {"what": what, "book": book if kind != "num" else float(book), "ours": ours if kind != "num" else float(ours), "tol": tol, "ok": bool(ok), "kind": kind}


def chk_table(checks, nd=4):
    rows = []
    for c in checks:
        if c["kind"] == "num":
            rows.append([c["what"], fmt(c["book"], nd), fmt(c["ours"], nd), "pass" if c["ok"] else "DIFFERS"])
        else:
            rows.append([c["what"], str(c["book"]), str(c["ours"]), "pass" if c["ok"] else "DIFFERS"])
    return tbl(["Check", "Book prints", "Ours", "Result"], rows)


# ---------------------------------------------------------------------------------------------- positions
def leg_value(kind, S, X, t, r, sig, b=None):
    if kind == "U":
        return np.asarray(S, float)
    if t <= 1e-12:
        return np.maximum(S - X, 0.0) if kind == "C" else np.maximum(X - S, 0.0)
    c, p = C.bs(S, X, t, r, sig, b)
    return c if kind == "C" else p


def pos_value(legs, S, X0, r, sig, dt=0.0, dsig=0.0, b=None):
    """legs = [(qty, 'C'|'P'|'U', strike / X0, years to expiration)] ; every leg is valued dt years later and at sig+dsig."""
    S = np.asarray(S, float)
    tot = np.zeros_like(S)
    for q, k, xr, t in legs:
        tot = tot + q * leg_value(k, S, xr * X0, max(t - dt, 0.0), r, sig + dsig, b)
    return tot


def pos_greeks(legs, S, X0, r, sig, b=None):
    S = np.asarray(S, float)
    out = {g: np.zeros_like(S) for g in ("delta", "gamma", "theta", "vega")}
    for q, k, xr, t in legs:
        if k == "U":
            out["delta"] = out["delta"] + q
            continue
        g = C.greeks(S, xr * X0, t, r, sig, b)
        i = 0 if k == "C" else 1
        for name in out:
            out[name] = out[name] + q * g[name][i]
    return out


def sgn(x, eps=1e-9):
    return "+" if x > eps else ("-" if x < -eps else "0")


def panel(draw, name, w=7.2, h=2.6):
    return W.two_panel(draw, name, w=w, h=h)
