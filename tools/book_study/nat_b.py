"""Natenberg write-up items: Chapter 7 (risk measures) and Chapter 9 (the sensitivities as the inputs change).  New file.

No figure in these chapters prints a number, so each picture is drawn from the book's own formulas (Figure 18-6) on our two prices and the claims the book states about the picture (in the figure, its legend, Figures 7-1, 7-10, 7-11, 9-25 and 9-26) are tested numerically.
Inputs the book does not print are fixed here and stated: X = last close unless the legend says ITM / OTM (0.9 and 1.1 of it), 3 months, r = b = 0, sigma = our one-year volatility.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import nat_core as C
import nat_ctx as X
from nat_ctx import item, chk, chk_table, ctx, tbl, fmt, VENUES, panel, sgn


def fid(s):
    return s.replace(" ", "_").replace(".", "_").replace("-", "_")


def G(S, Xs, t, sig, name, kind="call", r=0.0, b=None):
    g = C.greeks(S, Xs, t, r, sig, b)
    return g[name][0 if kind == "call" else 1]


def V(S, Xs, t, sig, kind="call", r=0.0, b=None):
    c, p = C.bs(S, Xs, t, r, sig, b)
    return c if kind == "call" else p


# ----------------------------------------------------------------------------------------------------------------- Chapter 7
def ch7():
    out = []
    # 7-1
    rows, ck = [], []
    book = {"S up": ("rise", "fall"), "S down": ("fall", "rise"), "vol up": ("rise", "rise"), "vol down": ("fall", "fall"), "time passes": ("fall", "fall")}
    for v in VENUES:
        c = ctx(v)
        S, sig, t = c["S"], c["sig"], X.T3M
        base = (V(S, S, t, sig, "call"), V(S, S, t, sig, "put"))
        moves = {"S up": (S * 1.01, sig, t), "S down": (S * 0.99, sig, t), "vol up": (S, sig + 0.01, t), "vol down": (S, sig - 0.01, t), "time passes": (S, sig, t - 1 / 365)}
        for k, (s2, sg2, t2) in moves.items():
            cc, pp = V(s2, S, t2, sg2, "call"), V(s2, S, t2, sg2, "put")
            dc, dp = ("rise" if cc > base[0] else "fall"), ("rise" if pp > base[1] else "fall")
            rows.append([v, k, book[k][0], dc, book[k][1], dp])
            ck.append(chk(f"{v}: {k}: call", book[k][0], dc, kind="text")); ck.append(chk(f"{v}: {k}: put", book[k][1], dp, kind="text"))
    out.append(item("Figure 7-1", 22, "Effect of changing market conditions on option values (call and put, five changes)", "data", table=chk_table(ck),
                    table2=tbl(["Venue", "Change", "Book: call", "Ours: call", "Book: put", "Ours: put"], rows), table2_title="At-the-money 3-month option, r = 0; S +/-1%, vol +/-1 point, one day passes",
                    text="Each change is applied to the book's value formula on our two prices and the direction of the value change is compared with the book's table.", checks=ck))
    # 7-2 / 7-3
    for id_, pg, kind in (("Figure 7-2", 22, "call"), ("Figure 7-3", 23, "put")):
        ck = []
        rows = []
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.5 * S0, 1.5 * S0, 1001)
            val = V(g, S0, X.T3M, sig, kind)
            intr = np.maximum(g - S0, 0) if kind == "call" else np.maximum(S0 - g, 0)
            ck.append(chk(f"{v}: theoretical value is never below intrinsic value (min of value - intrinsic)", True, bool(np.all(val - intr >= -1e-12)), kind="bool"))
            d = G(np.array([0.05 * S0, 0.5 * S0, 1.5 * S0, 20 * S0]), S0, X.T3M, sig, "delta", kind)
            lo, hi = (0.0, 1.0) if kind == "call" else (-1.0, 0.0)
            ck.append(chk(f"{v}: slope at 5% of the exercise price", lo, float(d[0]), 1e-3)); ck.append(chk(f"{v}: slope at 2,000% of the exercise price", hi, float(d[3]), 1e-3))
            ck.append(chk(f"{v}: the slope rises steadily from its low-price limit to its high-price limit (5%, 50%, 150%, 2,000% of X)", True, bool(d[0] <= d[1] <= d[2] <= d[3]), kind="bool"))
            rows.append([v, fmt(float(V(S0, S0, X.T3M, sig, kind)), 2), fmt(float(np.max(val - intr)), 2), fmt(float(g[np.argmax(val - intr)] / S0 * 100), 1)])

        def draw(ax, v, kind=kind):
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.6 * S0, 1.4 * S0, 400)
            ax.plot(g / S0 * 100, V(g, S0, X.T3M, sig, kind) / S0 * 100, label="theoretical value")
            ax.plot(g / S0 * 100, (np.maximum(g - S0, 0) if kind == "call" else np.maximum(S0 - g, 0)) / S0 * 100, ls="--", label="intrinsic value")
            ax.set_xlabel("underlying price (% of exercise price)"); ax.set_ylabel("% of exercise price"); ax.legend()
        p = panel(draw, f"nat_{fid(id_)}")
        out.append(item(id_, pg, f"Theoretical value of a {kind} against the underlying price, with its intrinsic value; slope tends to {'0 and 1' if kind == 'call' else '-1 and 0'}", "formula", fig=p, table=chk_table(ck, 3),
                        table2=tbl(["Venue", "At-the-money value", "Largest time value", "...at price (% of X)"], rows), table2_title="3-month, r = 0", checks=ck, text="Drawn from the book's formula (Figure 18-6) with X = our last close and 3 months."))
    # 7-4 delta and gamma
    ck, rows = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        g = np.linspace(0.7 * S0, 1.3 * S0, 2001)
        dl = G(g, S0, X.T3M, sig, "delta")
        gm = G(g, S0, X.T3M, sig, "gamma")
        fd = np.gradient(dl, g)
        ck.append(chk(f"{v}: gamma = slope of the delta curve (largest gap, per unit)", 0.0, float(np.max(np.abs(fd[2:-2] - gm[2:-2])) / float(np.max(gm))), 1e-4))
        dp = G(g, S0, X.T3M, sig, "delta", "put")
        ck.append(chk(f"{v}: call delta - put delta = 1 everywhere", 1.0, float(np.max(np.abs(dl - dp - 1)) + 1.0), 1e-12))
        rows.append([v, fmt(float(G(S0, S0, X.T3M, sig, 'delta')) * 100, 2), fmt(float(G(S0, S0, X.T3M, sig, 'delta', 'put')) * 100, 2), fmt(float(G(S0, S0, X.T3M, sig, 'gamma')) * S0, 4)])

    def draw(ax, v):
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        g = np.linspace(0.7 * S0, 1.3 * S0, 400)
        ax.plot(g / S0 * 100, G(g, S0, X.T3M, sig, "delta") * 100, label="call delta"); ax.plot(g / S0 * 100, G(g, S0, X.T3M, sig, "delta", "put") * 100, label="put delta")
        ax.set_xlabel("underlying price (% of exercise price)"); ax.set_ylabel("delta"); ax.legend()
    p = panel(draw, "nat_Figure_7_4")
    out.append(item("Figure 7-4", 24, "Delta values of a call and a put; gamma is the slope of the delta graph", "formula", fig=p, table=chk_table(ck, 4), table2=tbl(["Venue", "ATM call delta", "ATM put delta", "ATM gamma x S"], rows), table2_title="3-month, X = S, r = 0", checks=ck))
    # 7-5, 7-6
    for id_, pg, kind in (("Figure 7-5", 25, "call"), ("Figure 7-6", 25, "put")):
        ck = []
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.7 * S0, 1.3 * S0, 801)
            a, b_, c_ = (V(g, S0, t, sig, kind) for t in (0.25, 0.125, 1e-9))
            ck.append(chk(f"{v}: value falls at every price as time passes (3 months > 1.5 months > expiry)", True, bool(np.all(a >= b_ - 1e-12) and np.all(b_ >= c_ - 1e-12)), kind="bool"))

        def draw(ax, v, kind=kind):
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.7 * S0, 1.3 * S0, 400)
            for t, ls in ((0.25, "-"), (0.125, "--"), (1e-9, ":")):
                ax.plot(g / S0 * 100, V(g, S0, t, sig, kind) / S0 * 100, ls=ls, label=f"{t * 12:.1f} months" if t > 1e-3 else "expiration")
            ax.set_xlabel("underlying price (% of exercise price)"); ax.set_ylabel("% of exercise price"); ax.legend()
        p = panel(draw, f"nat_{fid(id_)}")
        out.append(item(id_, pg, f"Theoretical value of a {kind} as time passes (3 months, 1.5 months, expiration)", "formula", fig=p, table=chk_table(ck), checks=ck, text="The curve shifts down toward intrinsic value as time passes."))
    # 7-7 negative time value
    ck, rows = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        r = 0.06
        t = 1.0
        Sx = 0.5 * S0
        vp = float(V(Sx, S0, t, sig, "put", r))
        th = float(G(Sx, S0, t, sig, "theta", "put", r))
        rows.append([v, fmt(Sx, 2), fmt(vp, 2), fmt(S0 - Sx, 2), fmt(vp - (S0 - Sx), 2), fmt(th / 365, 4)])
        ck.append(chk(f"{v}: deep in-the-money put (S = 50% of X, r = 6%, 1 year) has value below intrinsic", True, bool(vp < S0 - Sx), kind="bool"))
        ck.append(chk(f"{v}: ...and its theta is positive", True, bool(th > 0), kind="bool"))
    out.append(item("Figure 7-7", 26, "An option with negative time value has a positive theta: its value rises toward intrinsic value as time passes", "formula", table=chk_table(ck),
                    table2=tbl(["Venue", "S", "Put value", "Intrinsic", "Time value", "Theta per day"], rows), table2_title="Deep in-the-money European put, r = 6% (the book's rate in Figure 6-5; the figure gives none), 1 year, X = last close, S = 50% of X", checks=ck))
    # 7-8, 7-9
    for id_, pg, kind in (("Figure 7-8", 27, "call"), ("Figure 7-9", 27, "put")):
        ck = []
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.7 * S0, 1.3 * S0, 801)
            lo, hi = V(g, S0, X.T3M, sig * 0.7, kind), V(g, S0, X.T3M, sig * 1.3, kind)
            ck.append(chk(f"{v}: higher volatility gives a higher value at every price", True, bool(np.all(hi >= lo - 1e-12)), kind="bool"))

        def draw(ax, v, kind=kind):
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.7 * S0, 1.3 * S0, 400)
            for m, ls in ((0.7, "-"), (1.3, "--")):
                ax.plot(g / S0 * 100, V(g, S0, X.T3M, sig * m, kind) / S0 * 100, ls=ls, label=f"{m} x sigma")
            ax.set_xlabel("underlying price (% of exercise price)"); ax.set_ylabel("% of exercise price"); ax.legend()
        p = panel(draw, f"nat_{fid(id_)}")
        out.append(item(id_, pg, f"Theoretical value of a {kind} at lower and higher volatility", "formula", fig=p, table=chk_table(ck), checks=ck))
    # 7-10 sign table
    rows, ck = [], []
    book = {"Long underlying": ["+", "0", "0", "0", "0"], "Short underlying": ["+", "0", "0", "0", "0"], "Long calls": ["+", "+", "-", "+", "+ stock / - futures"], "Short calls": ["-", "-", "+", "-", "- stock / + futures"],
            "Long puts": ["-", "+", "-", "+", "- stock / - futures"], "Short puts": ["+", "-", "+", "-", "- stock / + futures"]}
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        t = X.T3M
        rate = 0.05
        gs = C.greeks(S0, S0, t, rate, sig)            # stock: b = r
        gf = C.greeks(S0, S0, t, rate, sig, b=0.0)     # futures with stock-type settlement: b = 0
        calc = {"Long underlying": ["+", "0", "0", "0", "0"], "Short underlying": ["-", "0", "0", "0", "0"]}
        for nm, kind, q in (("Long calls", 0, 1), ("Short calls", 0, -1), ("Long puts", 1, 1), ("Short puts", 1, -1)):
            calc[nm] = [sgn(q * gs["delta"][kind]), sgn(q * gs["gamma"][kind]), sgn(q * gs["theta"][kind]), sgn(q * gs["vega"][kind]), f"{sgn(q * gs['rho'][kind])} stock / {sgn(q * gf['rho'][kind])} futures"]
        for nm in book:
            for j, gn in enumerate(["delta", "gamma", "theta", "vega", "rho"]):
                rows.append([v, nm, gn, book[nm][j], calc[nm][j], "pass" if book[nm][j] == calc[nm][j] else "DIFFERS"])
                ck.append(chk(f"{v}: {nm}: {gn}", book[nm][j], calc[nm][j], kind="text"))
    out.append(item("Figure 7-10", 28, "Summary of the signs of the risk measures for an underlying contract, calls and puts (long and short)", "data", table=tbl(["Venue", "Position", "Risk measure", "Book prints", "Ours", "Result"], rows), checks=ck,
                    text="The sign of every entry recomputed from the book's formulas (Figure 18-6) for a 3-month at-the-money option, r = 5% (rho needs a rate; the book gives none here), rho for stock (b = r) and for futures with stock-type settlement (b = 0).",
                    note="The book prints '+' for the delta of being short the underlying contract; a short position has a negative delta, so that row DIFFERS. The book's rho for short puts on stock is printed '-'; the computed sign is the opposite of long puts."))
    # 7-11
    rows, ck = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        t = X.T3M
        r = 0.05
        v0 = float(V(S0, S0, t, sig, "call", r))
        gtab = C.greeks(S0, S0, t, r, sig)
        d, gm, th, vg, rh = (float(gtab[k][0]) for k in ("delta", "gamma", "theta", "vega", "rho"))
        up = float(V(S0 * 1.01, S0, t, sig, "call", r)) - v0
        small = float(V(S0 * 1.005, S0, t, sig, "call", r)) + float(V(S0 * 0.995, S0, t, sig, "call", r)) - 2 * v0 - 0                   # long call, no hedge: curvature of the value (positive gamma)
        big = float(V(S0 * 1.05, S0, t, sig, "call", r)) + float(V(S0 * 0.95, S0, t, sig, "call", r)) - 2 * v0
        hedge_small = (float(V(S0 * 1.005, S0, t, sig, "call", r)) - v0 - d * S0 * 0.005) + (float(V(S0 * 0.995, S0, t, sig, "call", r)) - v0 + d * S0 * 0.005)
        hedge_big = (float(V(S0 * 1.05, S0, t, sig, "call", r)) - v0 - d * S0 * 0.05) + (float(V(S0 * 0.95, S0, t, sig, "call", r)) - v0 + d * S0 * 0.05)
        rows += [[v, "delta position positive", "wants the underlying to rise", "rise" if up > 0 else "fall"],
                 [v, "gamma position positive (delta-hedged long call)", "wants big or quick moves", f"hedged P&L after +/-5% = {hedge_big:+.4f}, after +/-0.5% = {hedge_small:+.4f}"],
                 [v, "theta position negative (long call)", "time reduces the value", "reduces" if float(V(S0, S0, t - 1 / 365, sig, 'call', r)) < v0 else "increases"],
                 [v, "vega position positive (long call)", "wants volatility to rise", "rise" if float(V(S0, S0, t, sig + .01, 'call', r)) > v0 else "fall"],
                 [v, "rho position positive (long call on stock)", "wants rates to rise", "rise" if float(V(S0, S0, t, sig, 'call', r + .01)) > v0 else "fall"]]
        ck += [chk(f"{v}: positive delta wants the underlying to rise", "rise", "rise" if up > 0 else "fall", kind="text"), chk(f"{v}: positive gamma (delta-hedged) profits from a big move, more than from a small one", True, bool(hedge_big > hedge_small > 0), kind="bool"),
               chk(f"{v}: negative theta: time reduces the value", "reduces", "reduces" if float(V(S0, S0, t - 1 / 365, sig, 'call', r)) < v0 else "increases", kind="text"),
               chk(f"{v}: positive vega wants volatility to rise", "rise", "rise" if float(V(S0, S0, t, sig + .01, 'call', r)) > v0 else "fall", kind="text"), chk(f"{v}: positive rho (stock call) wants rates to rise", "rise", "rise" if float(V(S0, S0, t, sig, 'call', r + .01)) > v0 else "fall", kind="text")]
    out.append(item("Figure 7-11", 29, "Summary of the risks and rewards of each risk measure: what you want the underlying, time, volatility and rates to do", "data", table=chk_table(ck), table2=tbl(["Venue", "Position", "Book: you want", "Ours"], rows), table2_title="Long at-the-money 3-month call, r = 5%", checks=ck))
    return out


# ----------------------------------------------------------------------------------------------------------------- Chapter 9
def sweep(spec, c):
    """spec: dict(x='S'|'vol'|'time'|'delta'|'pdelta', y=greek, series=[(label, dict(xr=, t=, sm=, kind=))])  -> list of (label, x array, y array) for venue context c."""
    S0, sig = c["S"], c["sig"]
    out = []
    for label, p in spec["series"]:
        Xs = p.get("xr", 1.0) * S0
        t = p.get("t", X.T3M)
        sm = p.get("sm", 1.0)
        kind = p.get("kind", "call")
        xv = spec["x"]
        if xv == "S":
            S = np.linspace(0.6 * S0, 1.4 * S0, 801); x = S / S0 * 100; y = G(S, Xs, t, sig * sm, spec["y"], kind) if spec["y"] != "value" else V(S, Xs, t, sig * sm, kind)
        elif xv == "vol":
            sg = np.linspace(0.05, 1.0, 400); x = sg * 100; y = G(S0, Xs, t, sg, spec["y"], kind) if spec["y"] != "value" else V(S0, Xs, t, sg, kind)
        elif xv == "time":
            tt = np.linspace(0.005, 1.0, 400); x = tt * 12; y = G(S0, Xs, tt, sig * sm, spec["y"], kind) if spec["y"] != "value" else V(S0, Xs, tt, sig * sm, kind)
        else:
            S = np.linspace(0.45 * S0, 1.8 * S0, 1601)
            cd = G(S, Xs, t, sig * sm, "delta", "call") * 100
            yy = G(S, Xs, t, sig * sm, spec["y"], "call")
            if xv == "delta":
                x, y = cd, yy
            else:
                x, y = cd - 100, yy
        out.append((label, np.asarray(x), np.asarray(y)))
    return out


def fig_item(id_, pg, title, spec, claims=None, ylab=None, xlab=None, scale=1.0, text=None):
    def draw(ax, v):
        c = ctx(v)
        for (lab, x, y), ls in zip(sweep(spec, c), ("-", "--", ":", "-.")):
            ax.plot(x, y * spec.get("scale", 1.0), ls=ls, label=lab)
        ax.set_xlabel(xlab or spec["x"]); ax.set_ylabel(ylab or spec["y"]); ax.legend(fontsize=6)
    p = panel(draw, f"nat_{fid(id_)}")
    rows, ck = [], (claims(spec) if claims else [])
    for v in VENUES:
        c = ctx(v)
        for lab, x, y in sweep(spec, c):
            y = y * spec.get("scale", 1.0)
            rows.append([v, lab, fmt(float(x[int(np.argmax(y))]), 2), fmt(float(np.max(y)), 5), fmt(float(x[int(np.argmin(y))]), 2), fmt(float(np.min(y)), 5)])
    return item(id_, pg, title, "formula", fig=p, table=tbl(["Venue", "Series", "x at maximum", "maximum", "x at minimum", "minimum"], rows), table2=chk_table(ck, 4) if ck else None, table2_title="Statements the book makes about this picture, tested" if ck else None,
                text=text or f"Drawn from the book's formulas (Figure 18-6) on our two prices: X = last close (ITM / OTM series at 0.9 / 1.1 of it), 3 months unless the legend varies time, r = b = 0, sigma = our one-year volatility. y = {spec['y']}.", checks=ck)


def _delta_at(c, xr, t, sm, kind="call", S=None):
    S0, sig = c["S"], c["sig"]
    return float(G(S0 if S is None else S, xr * S0, t, sig * sm, "delta", kind))


def ch9():
    out = []
    ser3 = lambda: [("ITM (X = 0.9 S)", dict(xr=0.9)), ("ATM (X = S)", dict(xr=1.0)), ("OTM (X = 1.1 S)", dict(xr=1.1))]
    ser3x = lambda: [("lower exercise (0.9 S)", dict(xr=0.9)), ("intermediate (S)", dict(xr=1.0)), ("higher exercise (1.1 S)", dict(xr=1.1))]

    def claims_91(spec):
        ck = []
        for v in VENUES:
            c = ctx(v)
            lo = [_delta_at(c, 1.1, X.T3M, m) for m in (0.5, 1.0, 1.5)]
            ck.append(chk(f"{v}: out-of-the-money call delta rises as volatility rises (0.5, 1, 1.5 x sigma)", True, bool(lo[0] < lo[1] < lo[2]), kind="bool"))
            at = [_delta_at(c, 1.0, X.T3M, m) for m in (0.5, 1.0, 1.5)]
            ck.append(chk(f"{v}: at-the-money call delta moves much less than the out-of-the-money delta over the same volatility range (spread, delta points: ATM {max(at)*100-min(at)*100:.1f}, OTM {max(lo)*100-min(lo)*100:.1f})", True, bool(max(at) - min(at) < max(lo) - min(lo)), kind="bool"))
        return ck
    out.append(fig_item("Figure 9-1", 30, "Call delta values as volatility changes (at-the-money and out-of-the-money calls)", dict(x="vol", y="delta", scale=100, series=[("at the money", dict(xr=1.0)), ("out of the money (X = 1.1 S)", dict(xr=1.1))]), claims_91, "call delta", "volatility (%)"))

    def claims_92(spec):
        ck = []
        for v in VENUES:
            c = ctx(v)
            itm, atm, otm = (_delta_at(c, xr, 1e-4, 1.0) for xr in (0.9, 1.0, 1.1))
            ck.append(chk(f"{v}: in-the-money call delta goes to 100 as time runs out", 1.0, itm, 1e-3)); ck.append(chk(f"{v}: out-of-the-money call delta goes to 0", 0.0, otm, 1e-3)); ck.append(chk(f"{v}: at-the-money call delta near 50 (delta x 100)", 50.0, atm * 100, 5.0))
        return ck
    out.append(fig_item("Figure 9-2", 31, "Call delta values as time passes (in-, at- and out-of-the-money calls)", dict(x="time", y="delta", scale=100, series=ser3()), claims_92, "call delta", "months to expiration"))

    def claims_93(spec):
        ck = []
        for v in VENUES:
            c = ctx(v)
            itm = [_delta_at(c, 1.1, X.T3M, m, "put") for m in (0.5, 1.0, 1.5)]
            ck.append(chk(f"{v}: in-the-money put delta moves toward -50 as volatility rises", True, bool(itm[0] < itm[1] < itm[2]), kind="bool"))
            at = [_delta_at(c, 1.0, X.T3M, m, "put") for m in (0.5, 1.0, 1.5)]
            ck.append(chk(f"{v}: at-the-money put delta moves much less than the in-the-money put delta over the same volatility range (spread: ATM {max(at)*100-min(at)*100:.1f}, ITM {max(itm)*100-min(itm)*100:.1f})", True, bool(max(at) - min(at) < max(itm) - min(itm)), kind="bool"))
        return ck
    out.append(fig_item("Figure 9-3", 32, "Put delta values as volatility changes (at-the-money and in-the-money puts)", dict(x="vol", y="delta", scale=100, series=[("at the money put", dict(xr=1.0, kind="put")), ("in the money put (X = 1.1 S)", dict(xr=1.1, kind="put"))]), claims_93, "put delta", "volatility (%)"))

    def claims_94(spec):
        ck = []
        for v in VENUES:
            c = ctx(v)
            otm, atm, itm = (_delta_at(c, xr, 1e-4, 1.0, "put") for xr in (0.9, 1.0, 1.1))
            ck.append(chk(f"{v}: in-the-money put delta goes to -100 as time runs out", -1.0, itm, 1e-3)); ck.append(chk(f"{v}: out-of-the-money put delta goes to 0", 0.0, otm, 1e-3))
        return ck
    out.append(fig_item("Figure 9-4", 33, "Put delta values as time passes (out-, at- and in-the-money puts)", dict(x="time", y="delta", scale=100, series=[("out of the money put (X = 0.9 S)", dict(xr=0.9, kind="put")), ("at the money put", dict(xr=1.0, kind="put")), ("in the money put (X = 1.1 S)", dict(xr=1.1, kind="put"))]), claims_94, "put delta", "months to expiration"))

    def claims_95(spec):
        ck = []
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = [float(G(S0, S0, t, sig, "gamma")) for t in (0.25, 0.1, 0.02)]
            ck.append(chk(f"{v}: the delta line gets steeper at the exercise price as time passes (gamma x S at 3 months, 1.2 months, 7 days)", True, bool(g[0] < g[1] < g[2]), kind="bool"))
            g2 = [float(G(S0, S0, 0.25, sig * m, "gamma")) for m in (1.5, 1.0, 0.5)]
            ck.append(chk(f"{v}: ...and as volatility declines (1.5, 1, 0.5 x sigma)", True, bool(g2[0] < g2[1] < g2[2]), kind="bool"))
        return ck
    out.append(fig_item("Figure 9-5", 34, "Call delta values against the underlying price as time passes or volatility declines", dict(x="S", y="delta", scale=100, series=[("3 months", dict(t=0.25)), ("1.2 months", dict(t=0.1)), ("7 days", dict(t=0.02))]), claims_95, "call delta", "underlying price (% of exercise price)"))

    out.append(fig_item("Figure 9-6", 35, "Vanna: the change of delta with volatility, against the option's delta (lower and higher volatility)", dict(x="delta", y="vanna", series=[("0.6 x sigma", dict(sm=0.6)), ("1.4 x sigma", dict(sm=1.4))]), None, "vanna (delta per 1.00 of vol)", "call delta (x 100)"))
    out.append(fig_item("Figure 9-7", 36, "Charm: the change of delta with time, against the option's delta (more and less time)", dict(x="delta", y="charm", series=[("more time (6 months)", dict(t=0.5)), ("less time (1.2 months)", dict(t=0.1))]), None, "charm (delta per year)", "call delta (x 100)"))
    out.append(fig_item("Figure 9-8", 37, "Theta (absolute value) as the underlying price changes, three exercise prices", dict(x="S", y="theta", scale=-1 / 365, series=ser3x()), None, "|theta| per day", "underlying price (% of S)"))
    out.append(fig_item("Figure 9-9", 38, "Theoretical value of an option as time passes (in-, at- and out-of-the-money)", dict(x="time", y="value", series=ser3()), None, "call value", "months to expiration"))
    out.append(fig_item("Figure 9-10", 39, "Theta (absolute value) as time passes, in-, at- and out-of-the-money", dict(x="time", y="theta", scale=-1 / 365, series=ser3()), None, "|theta| per day", "months to expiration"))
    out.append(fig_item("Figure 9-11", 40, "Theta (absolute value) as volatility changes: lower, at-the-money and higher exercise price", dict(x="vol", y="theta", scale=-1 / 365, series=ser3x()), None, "|theta| per day", "volatility (%)"))
    out.append(fig_item("Figure 9-12", 41, "Vega as the underlying price changes, three exercise prices", dict(x="S", y="vega", scale=0.01, series=ser3x()), None, "vega per vol point", "underlying price (% of S)"))
    out.append(fig_item("Figure 9-13", 42, "Theoretical value as volatility changes (in-, at- and out-of-the-money)", dict(x="vol", y="value", series=ser3()), None, "call value", "volatility (%)"))
    out.append(fig_item("Figure 9-14", 43, "Vega as volatility changes: at the money, lower and higher exercise price", dict(x="vol", y="vega", scale=0.01, series=[("ATM", dict(xr=1.0)), ("lower exercise (0.9 S)", dict(xr=0.9)), ("higher exercise (1.1 S)", dict(xr=1.1))]), None, "vega per vol point", "volatility (%)"))
    out.append(fig_item("Figure 9-15", 44, "Volga (vomma) against the option's delta, more and less time", dict(x="delta", y="volga", scale=0.0001, series=[("more time (6 months)", dict(t=0.5)), ("less time (1.2 months)", dict(t=0.1))]), None, "volga (vega change per vol point squared)", "call delta (x 100)"))
    out.append(fig_item("Figure 9-16", 45, "Vega as time passes: at the money, lower and higher exercise price", dict(x="time", y="vega", scale=0.01, series=[("ATM", dict(xr=1.0)), ("lower exercise (0.9 S)", dict(xr=0.9)), ("higher exercise (1.1 S)", dict(xr=1.1))]), None, "vega per vol point", "months to expiration"))
    out.append(fig_item("Figure 9-17", 46, "Vega decay against the option's delta, more and less time", dict(x="delta", y="vegadecay", series=[("more time (6 months)", dict(t=0.5)), ("less time (1.2 months)", dict(t=0.1))]), None, "vega decay (vega per year)", "call delta (x 100)"))
    out.append(fig_item("Figure 9-18", 47, "Gamma as the underlying price changes: lower, intermediate and higher exercise price", dict(x="S", y="gamma", scale=1.0, series=ser3x()), None, "gamma", "underlying price (% of S)"))
    out.append(fig_item("Figure 9-19", 48, "Speed against the option's delta (less time or lower volatility against more time or higher volatility)", dict(x="delta", y="speed", series=[("less time (1.2 months)", dict(t=0.1)), ("more time (6 months)", dict(t=0.5))]), None, "speed (gamma per unit of price)", "call delta (x 100)"))

    def claims_920(spec):
        ck = []
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            ga = [float(G(S0, S0, t, sig, "gamma")) for t in (0.5, 0.25, 0.05)]
            ck.append(chk(f"{v}: peak gamma at the exercise price rises as time to expiration falls", True, bool(ga[0] < ga[1] < ga[2]), kind="bool"))
        return ck
    out.append(fig_item("Figure 9-20", 49, "Gamma against the underlying price as time passes or volatility changes", dict(x="S", y="gamma", series=[("6 months", dict(t=0.5)), ("3 months", dict(t=0.25)), ("18 days", dict(t=0.05))]), claims_920, "gamma", "underlying price (% of exercise price)"))
    out.append(fig_item("Figure 9-21", 50, "Color (gamma decay) against the option's delta", dict(x="delta", y="color", series=[("more time (6 months)", dict(t=0.5)), ("less time (1.2 months)", dict(t=0.1))]), None, "color (gamma per year)", "call delta (x 100)"))
    out.append(fig_item("Figure 9-22", 51, "Zomma against the option's delta", dict(x="delta", y="zomma", series=[("more time (6 months)", dict(t=0.5)), ("less time (1.2 months)", dict(t=0.1))]), None, "zomma (gamma per 1.00 of vol)", "call delta (x 100)"))
    out.append(fig_item("Figure 9-23", 52, "Lambda (elasticity) of a call as time passes or volatility changes", dict(x="S", y="lambda", series=[("more time (6 months)", dict(t=0.5)), ("less time (1.2 months)", dict(t=0.1))]), None, "lambda", "underlying price (% of exercise price)"))
    out.append(fig_item("Figure 9-24", 53, "Lambda (elasticity) of a put as time passes or volatility changes", dict(x="S", y="lambda", series=[("more time (6 months)", dict(t=0.5, kind="put")), ("less time (1.2 months)", dict(t=0.1, kind="put"))]), None, "lambda", "underlying price (% of exercise price)"))
    out.append(fig_9_25())
    out.append(fig_9_26())
    return out


def _peak(name, kind="call", t=0.25, sm=1.0, r=0.0, b=None, c=None, absval=True):
    S0, sig = c["S"], c["sig"]
    S = np.linspace(0.3 * S0, 3.0 * S0, 54001)
    y = G(S, S0, t, sig * sm, name, kind, r, b)
    y = np.abs(y) if absval else y
    i = int(np.argmax(y))
    return float(S[i] / S0), float(y[i])


def fig_9_25():
    ck = []
    rows = []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        t = 0.25
        # delta
        dd = float(np.max(np.abs(G(np.array([0.5, 1.0, 2.0, 3.0]) * S0, S0, t, sig, "delta") - G(np.array([0.5, 1.0, 2.0, 3.0]) * S0, S0, t, sig, "delta", "put") - 1)))
        ck.append(chk(f"{v}: delta: call delta - put delta = 1 (largest gap)", 0.0, dd, 1e-12))
        ck.append(chk(f"{v}: delta: largest deep in the money (call delta at 3 S > at S)", True, bool(_delta_at(c, 1.0, t, 1.0, S=3 * S0) > _delta_at(c, 1.0, t, 1.0)), kind="bool"))
        # lambda
        l = lambda S, tt, m: float(G(S, S0, tt, sig * m, "lambda"))
        ck.append(chk(f"{v}: lambda: largest out of the money (0.9 S > S > 1.1 S in X terms: lambda at S=0.9X > at S=1.1X)", True, bool(l(0.9 * S0, t, 1) > l(1.1 * S0, t, 1)), kind="bool"))
        ck.append(chk(f"{v}: lambda: larger close to expiration (out-of-the-money, 1.2 months vs 6 months)", True, bool(l(0.95 * S0, 0.1, 1) > l(0.95 * S0, 0.5, 1)), kind="bool"))
        ck.append(chk(f"{v}: lambda: larger at low volatility (out-of-the-money, 0.5 vs 1.5 x sigma)", True, bool(l(0.95 * S0, t, 0.5) > l(0.95 * S0, t, 1.5)), kind="bool"))
        # gamma
        pk, _ = _peak("gamma", c=c)
        ck.append(chk(f"{v}: gamma: maximum at the exercise price (location of the peak, S/X; read 'at the money' as within 15%)", 1.0, pk, 0.15))
        ck.append(chk(f"{v}: gamma: larger close to expiration", True, bool(_peak("gamma", t=0.05, c=c)[1] > _peak("gamma", t=0.5, c=c)[1]), kind="bool"))
        ck.append(chk(f"{v}: gamma: larger at low volatility", True, bool(_peak("gamma", sm=0.5, c=c)[1] > _peak("gamma", sm=1.5, c=c)[1]), kind="bool"))
        # theta
        pk, _ = _peak("theta", c=c)
        ck.append(chk(f"{v}: theta: maximum (in absolute value) at the exercise price (peak location S/X; 'at the money' read as within 15%)", 1.0, pk, 0.15))
        ck.append(chk(f"{v}: theta: larger close to expiration", True, bool(_peak("theta", t=0.05, c=c)[1] > _peak("theta", t=0.5, c=c)[1]), kind="bool"))
        ck.append(chk(f"{v}: theta: larger at low volatility", True, bool(_peak("theta", sm=0.5, c=c)[1] > _peak("theta", sm=1.5, c=c)[1]), kind="bool"))
        # vega
        pk, _ = _peak("vega", c=c)
        ck.append(chk(f"{v}: vega: maximum at the exercise price (peak location S/X; 'at the money' read as within 15%)", 1.0, pk, 0.15))
        ck.append(chk(f"{v}: vega: larger long term (6 months vs 1.2 months)", True, bool(_peak("vega", t=0.5, c=c)[1] > _peak("vega", t=0.1, c=c)[1]), kind="bool"))
        # rho (r = 5%, b = r)
        r = 0.05
        rho = lambda S, tt: abs(float(G(S, S0, tt, sig, "rho", "call", r)))
        ck.append(chk(f"{v}: rho: largest deep in the money (|rho| at 1.5 S > at S, r = 5%)", True, bool(rho(1.5 * S0, 0.5) > rho(S0, 0.5)), kind="bool"))
        ck.append(chk(f"{v}: rho: larger long term (|rho| at S, 2 years > 3 months)", True, bool(rho(S0, 2.0) > rho(S0, 0.25)), kind="bool"))
        phi = lambda S, tt: abs(float(G(S, S0, tt, sig, "phi", "call", r)))
        ck.append(chk(f"{v}: rho-f (phi): largest deep in the money (at 1.5 S > at S)", True, bool(phi(1.5 * S0, 0.5) > phi(S0, 0.5)), kind="bool"))
        ck.append(chk(f"{v}: rho-f (phi): larger long term", True, bool(phi(S0, 2.0) > phi(S0, 0.25)), kind="bool"))
        # relations in the table's Math column
        gg = C.greeks(S0, S0, t, 0.0, sig)
        ck.append(chk(f"{v}: vega: dC/dsigma = dP/dsigma (largest gap)", 0.0, abs(float(gg["vega"][0] - gg["vega"][1])), 1e-12))
        rows.append([v, fmt(_peak("gamma", c=c)[0] * 100, 2), fmt(_peak("theta", c=c)[0] * 100, 2), fmt(_peak("vega", c=c)[0] * 100, 2)])
    return item("Figure 9-25", 54, "Traditional risk measures: what each is, and where it is largest (delta, lambda, gamma, theta, vega, rho, rho-f)", "data", table=chk_table(ck), table2=tbl(["Venue", "Gamma peak at S/X %", "|Theta| peak at S/X %", "Vega peak at S/X %"], rows),
                table2_title="Where the peaks fall (3 months, r = 0)", checks=ck,
                text="Each entry of the book's 'Maximized' column tested on the book's formulas on our two prices: the location of the maximum over the underlying price, and the direction of the change with time and volatility. A claim is true if the inequality holds on both venues.",
                note="Where a row says DIFFERS the computed behaviour is the opposite of the 'Maximized' column. (The book's theta maximum 'low volatility' compares absolute theta at 0.5 x and 1.5 x our volatility, 3 months, r = 0.)")


def _local_extrema_delta(name, t, sm, c, r=0.0):
    """Call deltas (x 100) at which the signed curve of `name` against the underlying price has an interior local maximum or minimum."""
    S0, sig = c["S"], c["sig"]
    S = np.linspace(0.3 * S0, 3.0 * S0, 54001)
    d = G(S, S0, t, sig * sm, "delta", "call", r) * 100
    y = G(S, S0, t, sig * sm, name, "call", r)
    g = np.sign(np.diff(y))
    idx = np.where(g[1:] * g[:-1] < 0)[0] + 1
    return [float(d[i]) for i in idx]


def fig_9_26():
    ck, rows = [], []
    printed = {"vanna": (17.5, 82.5), "charm": (17.5, 82.5), "speed": (17.5, 82.5), "volga": (10.0, 90.0), "vegadecay": (20.0, 80.0)}
    rng = {"vanna": ((15, 20), (80, 85)), "charm": ((15, 20), (80, 85)), "speed": ((15, 20), (80, 85)), "volga": ((10, 10), (90, 90)), "vegadecay": ((20, 20), (80, 80))}
    conds = [("3 months, our sigma", 0.25, 1.0), ("3 weeks, one quarter of our sigma (the book's 'close to expiration, low volatility')", 0.06, 0.25)]
    tol = 3.0
    for v in VENUES:
        c = ctx(v)
        for nm, (lo_r, hi_r) in rng.items():
            for cname, t, sm in conds:
                ex = _local_extrema_delta(nm, t, sm, c)
                lo_ok = any(lo_r[0] - tol <= e <= lo_r[1] + tol for e in ex)
                hi_ok = any(hi_r[0] - tol <= e <= hi_r[1] + tol for e in ex)
                rows.append([v, nm, f"{lo_r[0]}-{lo_r[1]}, {hi_r[0]}-{hi_r[1]}", cname, ", ".join(f"{e:.1f}" for e in ex) or "none"])
                ck.append(chk(f"{v}: {nm} [{cname}]: a local extremum within {tol:g} delta points of the printed {lo_r[0]}-{lo_r[1]}", True, bool(lo_ok), kind="bool"))
                ck.append(chk(f"{v}: {nm} [{cname}]: a local extremum within {tol:g} delta points of the printed {hi_r[0]}-{hi_r[1]}", True, bool(hi_ok), kind="bool"))
        mag = lambda nm, t, m: _peak(nm, t=t, sm=m, c=c)[1]
        ck.append(chk(f"{v}: charm: larger close to expiration (1.2 vs 6 months)", True, bool(mag("charm", 0.1, 1) > mag("charm", 0.5, 1)), kind="bool"))
        ck.append(chk(f"{v}: speed: larger at low volatility", True, bool(mag("speed", 0.25, 0.5) > mag("speed", 0.25, 1.5)), kind="bool"))
        ck.append(chk(f"{v}: speed: larger close to expiration", True, bool(mag("speed", 0.1, 1) > mag("speed", 0.5, 1)), kind="bool"))
        ck.append(chk(f"{v}: color: larger close to expiration", True, bool(mag("color", 0.1, 1) > mag("color", 0.5, 1)), kind="bool"))
        ck.append(chk(f"{v}: color: larger at low volatility", True, bool(mag("color", 0.25, 0.5) > mag("color", 0.25, 1.5)), kind="bool"))
        ck.append(chk(f"{v}: color: maximum at the money (peak S/X; read as within 15%)", 1.0, _peak("color", c=c)[0], 0.15))
        ck.append(chk(f"{v}: volga: larger long term", True, bool(mag("volga", 0.5, 1) > mag("volga", 0.1, 1)), kind="bool"))
        ck.append(chk(f"{v}: volga: larger at low volatility", True, bool(mag("volga", 0.25, 0.5) > mag("volga", 0.25, 1.5)), kind="bool"))
        ck.append(chk(f"{v}: vega decay: larger close to expiration", True, bool(mag("vegadecay", 0.1, 1) > mag("vegadecay", 0.5, 1)), kind="bool"))
        ck.append(chk(f"{v}: zomma: larger close to expiration", True, bool(mag("zomma", 0.1, 1) > mag("zomma", 0.5, 1)), kind="bool"))
        ck.append(chk(f"{v}: zomma: larger at low volatility", True, bool(mag("zomma", 0.25, 0.5) > mag("zomma", 0.25, 1.5)), kind="bool"))
        ck.append(chk(f"{v}: zomma: maximum at the money (peak S/X; read as within 15%)", 1.0, _peak("zomma", c=c)[0], 0.15))
    return item("Figure 9-26", 55, "Nontraditional higher-order risk measures: vanna, charm, speed, color, volga, vega decay, zomma and where each is largest", "data", table=chk_table(ck), table2=tbl(["Venue", "Measure", "Book: extremes at call delta", "Conditions", "Ours: local extremes at call delta"], rows),
                table2_title="Where each curve has its local extremes, in terms of the option's call delta (r = 0)", checks=ck,
                text="The book's 'Maximized' column: for the five measures that name delta values, every interior local extremum of the curve is found over a fine grid of underlying prices and read as a call delta; a claim passes when one lies within three delta points of the printed range. The 'close to expiration', 'low volatility', 'long term' and 'at the money' claims are tested as inequalities. The three-delta-point allowance and the 15% reading of 'at the money' are ours (the book prints ranges rounded to 5).",
                note="The location of an extremum moves with volatility and time (the table shows it at two sets of conditions); the book prints one range without saying which conditions.")


def all_items():
    return ch7() + ch9()
