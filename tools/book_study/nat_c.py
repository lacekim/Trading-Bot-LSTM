"""Natenberg write-up items: Chapter 11 (volatility spreads), Chapter 12 (bull and bear spreads), Chapter 17 (hedging with options).  New file.

The book's pictures print the legs (for example '+1 March 100 call, +1 March 100 put') but no prices, rates or times.  We use the first leg set printed in each figure with strikes at the same percentage of our last close
as the book's strike is of its centre (100 -> 100%, 90 -> 90%, 110 -> 110%), near expiry 2 months, far expiry 4 months, single-expiry strategies 3 months, r = b = 0, sigma = our one-year volatility. These are assumptions of ours.
"""
from __future__ import annotations

import numpy as np

import nat_core as C
import nat_ctx as X
from nat_ctx import item, chk, chk_table, ctx, tbl, fmt, VENUES, panel, sgn, pos_value, pos_greeks

T1, T2, T3 = 2 / 12, 3 / 12, 4 / 12


def fid(s):
    return s.replace(" ", "_").replace(".", "_").replace("-", "_")


def pos_figure(id_, pg, title, legs, near, extra_note="", mode="time", claims=None, text=None):
    """Value of the position (relative to its cost today) against the underlying: now, when time has passed (halfway to the nearest expiry, and at it), and with volatility lower by one third."""
    def draw(ax, v):
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        g = np.linspace(0.7 * S0, 1.3 * S0, 500)
        cost = float(pos_value(legs, np.array([S0]), S0, 0.0, sig)[0])
        if mode == "time":
            for dt, ls, lab in ((0.0, "-", "now"), (near / 2, "--", "half the time to the first expiry gone"), (near, ":", "first expiry")):
                ax.plot(g / S0 * 100, (pos_value(legs, g, S0, 0.0, sig, dt=dt) - cost) / S0 * 100, ls=ls, label=lab)
        else:
            for ds, ls, lab in ((0.0, "-", "now"), (-sig / 3, "--", "volatility down one third"), (sig / 3, ":", "volatility up one third")):
                ax.plot(g / S0 * 100, (pos_value(legs, g, S0, 0.0, sig, dsig=ds) - cost) / S0 * 100, ls=ls, label=lab)
        ax.axhline(0, lw=.4, color="0.6"); ax.axvline(100, lw=.4, ls=":", color="0.6")
        ax.set_xlabel("underlying price (% of last close)"); ax.set_ylabel("P&L vs today (% of S)"); ax.legend(fontsize=6)
    p = panel(draw, f"nat_{fid(id_)}")
    rows = []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        g = pos_greeks(legs, np.array([S0]), S0, 0.0, sig)
        cost = float(pos_value(legs, np.array([S0]), S0, 0.0, sig)[0])
        rows.append([v, fmt(cost / S0 * 100, 3), fmt(float(g["delta"][0]), 4), fmt(float(g["gamma"][0]) * S0, 4), fmt(float(g["theta"][0]) / 365 / S0 * 100, 5), fmt(float(g["vega"][0]) / 100 / S0 * 100, 5)])
    ck = claims() if claims else []
    legs_txt = "; ".join(f"{q:+d} {'underlying' if k == 'U' else f'{xr:.2f} S {k}'}" + ("" if k == "U" else f" ({t * 12:.0f} mo)") for q, k, xr, t in legs)
    return item(id_, pg, title, "formula", fig=p, table=tbl(["Venue", "Cost (% of S)", "Delta", "Gamma x S", "Theta per day (% of S)", "Vega per vol point (% of S)"], rows),
                table2=chk_table(ck) if ck else None, table2_title="Statements the book makes about this picture, tested" if ck else None,
                text=text or f"Legs (strikes as a multiple of our last close): {legs_txt}. Picture: the position's value relative to its cost today, against the underlying price. {extra_note}", checks=ck)


def L(*xs):
    return list(xs)


def ch11():
    out = []
    t = T2
    strat = {}
    strat["long straddle"] = L((1, "C", 1.0, t), (1, "P", 1.0, t)); strat["short straddle"] = [(-q, k, x, tt) for q, k, x, tt in strat["long straddle"]]
    strat["long strangle"] = L((1, "P", .9, t), (1, "C", 1.1, t)); strat["short strangle"] = [(-q, k, x, tt) for q, k, x, tt in strat["long strangle"]]
    strat["long butterfly"] = L((1, "C", .9, t), (-2, "C", 1.0, t), (1, "C", 1.1, t)); strat["short butterfly"] = [(-q, k, x, tt) for q, k, x, tt in strat["long butterfly"]]
    strat["long condor"] = L((1, "C", .9, t), (-1, "C", .95, t), (-1, "C", 1.05, t), (1, "C", 1.1, t)); strat["short condor"] = [(-q, k, x, tt) for q, k, x, tt in strat["long condor"]]
    order = [("Figure 11-1", 56, "Long straddle as time passes or volatility declines", "long straddle"), ("Figure 11-2", 57, "Short straddle as time passes or volatility declines", "short straddle"),
             ("Figure 11-3", 57, "Long strangle as time passes or volatility declines", "long strangle"), ("Figure 11-4", 58, "Short strangle as time passes or volatility declines", "short strangle"),
             ("Figure 11-5", 58, "Long butterfly as time passes or volatility declines", "long butterfly"), ("Figure 11-6", 59, "Short butterfly as time passes or volatility declines", "short butterfly"),
             ("Figure 11-7", 59, "Long condor as time passes or volatility declines", "long condor"), ("Figure 11-8", 60, "Short condor as time passes or volatility declines", "short condor")]
    for id_, pg, title, key in order:
        out.append(pos_figure(id_, pg, title, strat[key], t, mode="time"))
    # 11-9
    book = {"long straddle": "+-+", "short straddle": "-+-", "long strangle": "+-+", "short strangle": "-+-", "short butterfly": "+-+", "long butterfly": "-+-", "short condor": "+-+", "long condor": "-+-"}
    rows, ck = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for nm, bk in book.items():
            g = pos_greeks(strat[nm], np.array([S0]), S0, 0.0, sig)
            got = sgn(float(g["gamma"][0]), 1e-12) + sgn(float(g["theta"][0]), 1e-9) + sgn(float(g["vega"][0]), 1e-9)
            rows.append([v, nm, " / ".join(f"{a}{n}" for a, n in zip(bk, ("gamma", "theta", "vega"))), " / ".join(f"{a}{n}" for a, n in zip(got, ("gamma", "theta", "vega"))), "pass" if bk == got else "DIFFERS"])
            ck.append(chk(f"{v}: {nm}: gamma, theta, vega signs at the centre", bk, got, kind="text"))
    out.append(item("Figure 11-9", 61, "Symmetrical strategies: signs of gamma, theta and vega for the eight straddles, strangles, butterflies and condors", "data", table=tbl(["Venue", "Strategy", "Book prints", "Ours (at S = centre)", "Result"], rows), checks=ck,
                    text="Net gamma, theta and vega of the legs printed in Figures 11-1 to 11-8, at our last close, from the book's formulas (3 months, r = 0)."))
    # 11-10 ... 11-17
    rs = {}
    rs["call ratio buy more"] = L((3, "C", 1.05, t), (-1, "C", .95, t)); rs["put ratio buy more"] = L((4, "P", .9, t), (-1, "P", 1.0, t))
    rs["call ratio sell more"] = L((1, "C", .95, t), (-3, "C", 1.05, t)); rs["put ratio sell more"] = L((1, "P", 1.0, t), (-4, "P", .9, t))
    rs["long call tree"] = L((1, "C", .9, t), (-1, "C", .95, t), (-1, "C", 1.05, t)); rs["short call tree"] = [(-q, k, x, tt) for q, k, x, tt in rs["long call tree"]]
    rs["long put tree"] = L((1, "P", 1.1, t), (-1, "P", 1.05, t), (-1, "P", .95, t)); rs["short put tree"] = [(-q, k, x, tt) for q, k, x, tt in rs["long put tree"]]
    figs = [("Figure 11-10", 62, "Call ratio spread (buy more than sell) as time passes or volatility declines: +3 105 calls, -1 95 call", "call ratio buy more", ""),
            ("Figure 11-11", 62, "Put ratio spread (buy more than sell) as time passes or volatility declines: +4 90 puts, -1 100 put", "put ratio buy more", ""),
            ("Figure 11-12", 63, "Call ratio spread (sell more than buy) as time passes or volatility declines: +1 95 call, -3 105 calls", "call ratio sell more", ""),
            ("Figure 11-13", 63, "Put ratio spread (sell more than buy) as time passes or volatility declines: +1 100 put, -4 90 puts", "put ratio sell more", ""),
            ("Figure 11-14", 64, "Call ratio spread (sell more than buy) as time passes or volatility declines [title as printed; the legs +1 90, -1 95, -1 105 calls are a long call Christmas tree]", "long call tree", ""),
            ("Figure 11-15", 64, "Short call Christmas tree as time passes or volatility declines", "short call tree", ""),
            ("Figure 11-16", 65, "Long put Christmas tree as time passes or volatility declines", "long put tree", ""),
            ("Figure 11-17", 65, "Short put Christmas tree as time passes or volatility declines", "short put tree", "")]
    for id_, pg, title, key, nt in figs:
        out.append(pos_figure(id_, pg, title, rs[key], t, mode="time"))
    book = {"call ratio buy more": "+-+", "call ratio sell more": "-+-", "put ratio buy more": "+-+", "put ratio sell more": "-+-", "short call tree": "+-+", "long call tree": "-+-", "short put tree": "+-+", "long put tree": "-+-"}
    rows, ck = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for nm, bk in book.items():
            g = pos_greeks(rs[nm], np.array([S0]), S0, 0.0, sig)
            got = sgn(float(g["gamma"][0]), 1e-12) + sgn(float(g["theta"][0]), 1e-9) + sgn(float(g["vega"][0]), 1e-9)
            rows.append([v, nm, " / ".join(f"{a}{n}" for a, n in zip(bk, ("gamma", "theta", "vega"))), " / ".join(f"{a}{n}" for a, n in zip(got, ("gamma", "theta", "vega"))), "pass" if bk == got else "DIFFERS"])
            ck.append(chk(f"{v}: {nm}: gamma, theta, vega signs at our last close", bk, got, kind="text"))
    out.append(item("Figure 11-18", 66, "Nonsymmetrical strategies: signs of gamma, theta and vega for ratio spreads and Christmas trees", "data", table=tbl(["Venue", "Strategy", "Book prints", "Ours (S = last close)", "Result"], rows), checks=ck,
                    text="Net gamma, theta and vega of the legs printed in Figures 11-10 to 11-17 at our last close (strikes at the printed percentages of it, 3 months, r = 0). The sign of a non-symmetrical position's gamma, theta and vega depends on where the underlying sits between the strikes; the book prints one sign set per strategy.",
                    note="Where a row says DIFFERS the sign at our last close is the opposite of the printed one for these strikes; the pictures above show how the position changes with the underlying price."))
    # calendars and time butterfly
    cal = L((1, "C", 1.0, T3), (-1, "C", 1.0, T1))
    for id_, pg, title, legs, mode in (("Figure 11-19", 67, "Long calendar spread as time passes: +1 far call, -1 near call", cal, "time"), ("Figure 11-20", 67, "Short calendar spread as time passes: +1 near call, -1 far call", [(-q, k, x, tt) for q, k, x, tt in cal], "time"),
                                       ("Figure 11-21", 68, "Long calendar spread as volatility declines", cal, "vol"), ("Figure 11-22", 68, "Short calendar spread as volatility declines", [(-q, k, x, tt) for q, k, x, tt in cal], "vol")):
        def cl(legs=legs, id_=id_):
            ck = []
            for v in VENUES:
                c = ctx(v)
                S0, sig = c["S"], c["sig"]
                g = pos_greeks(legs, np.array([S0]), S0, 0.0, sig)
                longcal = legs[0][0] > 0
                ck.append(chk(f"{v}: {'long' if longcal else 'short'} calendar spread has {'positive' if longcal else 'negative'} vega at the strike", "+" if longcal else "-", sgn(float(g["vega"][0]), 1e-9), kind="text"))
                ck.append(chk(f"{v}: {'long' if longcal else 'short'} calendar spread has {'positive' if longcal else 'negative'} theta at the strike", "+" if longcal else "-", sgn(float(g["theta"][0]), 1e-9), kind="text"))
            return ck
        out.append(pos_figure(id_, pg, title, legs, T1, mode=mode, claims=cl, extra_note="Strike 100% of S; near expiry 2 months, far 4 months."))
    tb = L((-1, "C", 1.0, T1), (2, "C", 1.0, T2), (-1, "C", 1.0, T3))
    out.append(pos_figure("Figure 11-23", 69, "Long time butterfly as time passes: -1 near call, +2 middle calls, -1 far call", tb, T1, mode="time", extra_note="Expiries 2, 3 and 4 months."))
    out.append(pos_figure("Figure 11-24", 69, "Long time butterfly as volatility declines", tb, T1, mode="vol", extra_note="Expiries 2, 3 and 4 months."))
    return out


def ch12():
    out = []
    t = T2
    bull = L((1, "C", .95, t), (-1, "C", 1.05, t))
    bear = [(-q, k, x, tt) for q, k, x, tt in bull]

    def c1():
        ck = []
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.6 * S0, 1.4 * S0, 801)
            y = pos_value(bull, g, S0, 0.0, sig, dt=t)
            ck.append(chk(f"{v}: bull spread value at expiration rises from 0 to the strike difference and is flat beyond", True, bool(abs(y[0]) < 1e-9 and abs(y[-1] - 0.1 * S0) < 1e-9 and np.all(np.diff(y) >= -1e-12)), kind="bool"))
            ck.append(chk(f"{v}: delta of the bull spread is positive at our last close", "+", sgn(float(pos_greeks(bull, np.array([S0]), S0, 0.0, sig)["delta"][0])), kind="text"))
        return ck

    def c2():
        ck = []
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.6 * S0, 1.4 * S0, 801)
            y = pos_value(bear, g, S0, 0.0, sig, dt=t)
            ck.append(chk(f"{v}: bear spread value at expiration falls from 0 to minus the strike difference", True, bool(abs(y[0]) < 1e-9 and abs(y[-1] + 0.1 * S0) < 1e-9 and np.all(np.diff(y) <= 1e-12)), kind="bool"))
            ck.append(chk(f"{v}: delta of the bear spread is negative at our last close", "-", sgn(float(pos_greeks(bear, np.array([S0]), S0, 0.0, sig)["delta"][0])), kind="text"))
        return ck
    out.append(pos_figure("Figure 12-1", 70, "Bull spread: buy the lower exercise price, sell the higher (calls, 95% and 105% of S)", bull, t, claims=c1))
    out.append(pos_figure("Figure 12-2", 71, "Bear spread: buy the higher exercise price, sell the lower (calls, 95% and 105% of S)", bear, t, claims=c2))
    # 12-3 farther apart
    rows, ck = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        ds, dsb = [], []
        for w in (0.02, 0.05, 0.10, 0.20):
            lo, hi = 1 - w, 1 + w
            bl = L((1, "C", lo, t), (-1, "C", hi, t))
            ds.append(float(pos_greeks(bl, np.array([S0]), S0, 0.0, sig)["delta"][0]))
            dsb.append(-ds[-1])
            rows.append([v, f"+/-{w:.0%}", fmt(ds[-1], 4), fmt(-ds[-1], 4)])
        ck.append(chk(f"{v}: the wider the strikes, the more bullish the bull spread (delta rises with width, ATM-centred)", True, bool(ds[0] < ds[1] < ds[2] < ds[3]), kind="bool"))
        ck.append(chk(f"{v}: ...and the more bearish the bear spread", True, bool(dsb[0] > dsb[1] > dsb[2] > dsb[3]), kind="bool"))
    out.append(item("Figure 12-3", 71, "As the exercise prices move farther apart the spread takes on greater bullish or bearish characteristics", "formula", table=chk_table(ck), table2=tbl(["Venue", "Strikes (% of S)", "Bull spread delta", "Bear spread delta"], rows), table2_title="Delta of the call spread centred on our last close", checks=ck))
    # 12-4 .. 12-9
    def spread_curve(id_, pg, title, what, series, ylab):
        def draw(ax, v):
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.7 * S0, 1.3 * S0, 500)
            for (lab, dt, dsig), ls in zip(series, ("-", "--", ":")):
                sg = sig + dsig
                tt = t - dt
                if what == "value":
                    y = pos_value(bull, g, S0, 0.0, sig, dt=dt, dsig=dsig) / S0 * 100
                else:
                    y = pos_greeks([(q, k, xr, tt) for q, k, xr, _ in bull], g, S0, 0.0, sg)[what]
                    y = y * {"delta": 100, "gamma": S0, "theta": 1 / 365 / S0 * 100, "vega": 0.01 / S0 * 100}[what]
                ax.plot(g / S0 * 100, y, ls=ls, label=lab)
            ax.set_xlabel("underlying price (% of last close)"); ax.set_ylabel(ylab); ax.legend(fontsize=6)
        p = panel(draw, f"nat_{fid(id_)}")
        rows = []
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.7 * S0, 1.3 * S0, 2001)
            for lab, dt, dsig in series:
                tt = max(t - dt, 1e-4)
                y = pos_value(bull, g, S0, 0.0, sig, dt=dt, dsig=dsig) if what == "value" else pos_greeks([(q, k, xr, tt) for q, k, xr, _ in bull], g, S0, 0.0, sig + dsig)[what]
                rows.append([v, lab, fmt(float(g[np.argmax(y)] / S0 * 100), 2), fmt(float(y.max() / (S0 if what == 'value' else 1)), 5), fmt(float(g[np.argmin(y)] / S0 * 100), 2), fmt(float(y.min() / (S0 if what == 'value' else 1)), 5)])
        return item(id_, pg, title, "formula", fig=p, table=tbl(["Venue", "Series", "Price at max (% of S)", "max", "Price at min (% of S)", "min"], rows),
                    text="Bull spread of Figure 12-1 (calls at 95% and 105% of our last close, 3 months, r = 0, sigma ours). Series as labelled.")
    S = lambda lab, dt, ds: (lab, dt, ds)
    time3 = [S("3 months", 0.0, 0.0), S("1.5 months", t / 2, 0.0), S("expiration", t - 1e-4, 0.0)]
    vol3 = [S("sigma", 0.0, 0.0), S("sigma - one third", 0.0, "-"), S("sigma - two thirds", 0.0, "--")]
    out.append(spread_curve("Figure 12-4", 72, "Value of a bull spread as time passes or volatility declines", "value", time3, "% of S"))
    out.append(spread_curve("Figure 12-5", 72, "Delta of a bull spread as time passes or volatility declines", "delta", time3, "delta"))
    out.append(spread_curve("Figure 12-6", 73, "Gamma of a bull spread as time passes or volatility declines", "gamma", time3[:2], "gamma x S"))
    out.append(spread_curve("Figure 12-7", 73, "Vega of a bull spread as time passes or volatility declines", "vega", time3[:2], "vega per vol point (% of S)"))
    out.append(spread_curve("Figure 12-8", 74, "Theta of a bull spread as volatility declines", "theta", [S("sigma", 0.0, 0.0), S("sigma - 0.1", 0.0, -0.1)], "theta per day (% of S)"))
    out.append(spread_curve("Figure 12-9", 74, "Theta of a bull spread as time passes", "theta", time3[:2], "theta per day (% of S)"))
    return out


def ch17():
    out = []
    t = T2

    def bscalar(S0, sig, legs, S):
        return pos_value(legs, np.asarray(S, float), S0, 0.0, sig, dt=0.0)
    cases = [("Figure 17-1", 75, "Long an underlying position and long a protective put (the combined position is a synthetic long call)", [(1, "U", 0, t), (1, "P", 1.0, t)], [(1, "C", 1.0, t)], lambda S0: S0),
             ("Figure 17-2", 75, "Short an underlying position and long a protective call (a synthetic long put)", [(-1, "U", 0, t), (1, "C", 1.0, t)], [(1, "P", 1.0, t)], lambda S0: -S0),
             ("Figure 17-3", 76, "Long an underlying position and short a covered call (a synthetic short put)", [(1, "U", 0, t), (-1, "C", 1.0, t)], [(-1, "P", 1.0, t)], lambda S0: S0),
             ("Figure 17-4", 76, "Short an underlying position and short a covered put (a synthetic short call)", [(-1, "U", 0, t), (-1, "P", 1.0, t)], [(-1, "C", 1.0, t)], lambda S0: -S0),
             ("Figure 17-5", 77, "Long collar: long underlying, long a protective put (95%), short a covered call (105%) (a bull call spread)", [(1, "U", 0, t), (1, "P", .95, t), (-1, "C", 1.05, t)], [(1, "C", .95, t), (-1, "C", 1.05, t)], lambda S0: .95 * S0),
             ("Figure 17-6", 77, "Short collar: short underlying, long a protective call (105%), short a covered put (95%) (a bear put spread)", [(-1, "U", 0, t), (1, "C", 1.05, t), (-1, "P", .95, t)], [(1, "P", 1.05, t), (-1, "P", .95, t)], lambda S0: -1.05 * S0)]
    for id_, pg, title, legs, syn, off in cases:
        def cl(legs=legs, syn=syn, off=off):
            ck = []
            for v in VENUES:
                c = ctx(v)
                S0, sig = c["S"], c["sig"]
                g = np.linspace(0.5 * S0, 1.5 * S0, 801)
                a = pos_value(legs, g, S0, 0.0, sig, dt=t)
                b_ = pos_value(syn, g, S0, 0.0, sig, dt=t) + off(S0)
                ck.append(chk(f"{v}: payoff at expiration equals the synthetic position plus a constant (largest gap)", 0.0, float(np.max(np.abs(a - b_))), 1e-9))
                a0 = float(pos_value(legs, np.array([S0]), S0, 0.0, sig)[0])
                b0 = float(pos_value(syn, np.array([S0]), S0, 0.0, sig)[0]) + off(S0)
                ck.append(chk(f"{v}: ...and today's value too (r = 0, put-call parity), gap", 0.0, abs(a0 - b0), 1e-9))
            return ck
        out.append(pos_figure(id_, pg, title, legs, t, mode="time", claims=cl))
    # 17-7
    rows, ck = [], []
    strategies = [("long underlying", "sell a future or forward", [(1, "U", 0, t)], "no downside risk", "no upside profit potential", (0, 0), "fut"),
                  ("long underlying", "buy a protective put", [(1, "U", 0, t), (1, "P", 1.0, t)], "limited downside risk; unlimited upside profit potential", "cost of the option", (0, 1), None),
                  ("long underlying", "sell a covered call", [(1, "U", 0, t), (-1, "C", 1.0, t)], "partial downside protection equal to the price of the call", "unlimited downside risk; limited upside profit potential", (1, 0), None),
                  ("long underlying", "long collar", [(1, "U", 0, t), (1, "P", .95, t), (-1, "C", 1.05, t)], "limited downside risk", "limited upside profit potential", (0, 0), None),
                  ("short underlying", "buy a future or forward", [(-1, "U", 0, t)], "no upside risk", "no downside profit potential", (0, 0), "fut"),
                  ("short underlying", "buy a protective call", [(-1, "U", 0, t), (1, "C", 1.0, t)], "limited upside risk; unlimited downside profit potential", "cost of the option", (-1, 0), None),
                  ("short underlying", "sell a covered put", [(-1, "U", 0, t), (-1, "P", 1.0, t)], "partial upside protection equal to the price of the put", "unlimited upside risk; limited downside profit potential", (0, -1), None),
                  ("short underlying", "short collar", [(-1, "U", 0, t), (1, "C", 1.05, t), (-1, "P", .95, t)], "limited upside risk", "limited downside profit potential", (0, 0), None)]
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for pos, hedge, legs, adv, dis, (slo, shi), kind in strategies:
            if kind == "fut":                                                                  # the book's first hedge: offset with a future / forward at today's price -> the combined position has no price exposure
                lg = [(q, k, x, tt) for q, k, x, tt in legs] + [(-legs[0][0], "U", 0, t)]
                g = np.array([0.2 * S0, 5.0 * S0])
                y = pos_value(lg, g, S0, 0.0, sig, dt=t) - float(pos_value(lg, np.array([S0]), S0, 0.0, sig)[0])
                got = (0.0, 0.0) if abs(y[1] - y[0]) < 1e-9 else (None, None)
                cost = 0.0
            else:
                g = np.array([0.1 * S0, 0.11 * S0, 10 * S0, 11 * S0])
                y = pos_value(legs, g, S0, 0.0, sig, dt=t)
                got = (round(float((y[1] - y[0]) / (0.01 * S0)), 6) + 0.0, round(float((y[3] - y[2]) / (S0)), 6) + 0.0)
                cost = float(pos_value([l for l in legs if l[1] != "U"], np.array([S0]), S0, 0.0, sig)[0])
            ok = (got[0] == slo and got[1] == shi) if kind != "fut" else True
            rows.append([v, pos, hedge, adv, dis, f"{got[0]:+g}, {got[1]:+g}", fmt(cost / S0 * 100, 3), "pass" if ok else "DIFFERS"])
            ck.append(chk(f"{v}: {pos} / {hedge}: slope of the position at the far low and far high prices (book: {slo:+g} and {shi:+g})", f"{slo:+g}, {shi:+g}", f"{got[0]:+g}, {got[1]:+g}", kind="text"))
    out.append(item("Figure 17-7", 78, "Summary of basic hedging strategies: advantages and disadvantages", "data", table=tbl(["Venue", "Position", "Hedge", "Advantages (book)", "Disadvantages (book)", "Slope at far low, far high", "Option premium (% of S)", "Result"], rows), checks=ck,
                    text="Each row's claim is read off the payoff at expiration: slope 0 at the far end = no exposure that way, slope of the underlying (+1 long, -1 short) = unlimited exposure that way; 'cost of the option' is the premium paid today (3 months, r = 0, sigma ours).",
                    note="The first row of each position (selling or buying a future) is tested as: the combined position has no price exposure."))
    return out


def all_items():
    return ch11() + ch12() + ch17()
