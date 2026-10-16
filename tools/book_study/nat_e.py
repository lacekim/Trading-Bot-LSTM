"""Natenberg write-up items: Chapter 21 (position analysis), Chapter 23 (skewness, kurtosis, biggest moves), Chapter 24 (volatility surface).  New file.

Figure 21-9 and 21-20 print every input (price, time, volatility, rate, quantities), so those are reproduced with the printed inputs.  Figures 21-1, 21-7, 21-8 print the legs but no time, rate or volatility: same assumptions as the other chapters
(strikes at the book's percentage of our last close, 3 months, r = 0, sigma = our one-year volatility; the 14% / 18% / 22% of the book are read as sigma x 14/18, sigma, sigma x 22/18).
Time convention for Figure 21-9: calendar days / 365 (6 weeks = 42 days): this one reproduces every printed per-contract number; weeks/52 does not (90 call delta 90.8 instead of the printed 90.9).
"""
from __future__ import annotations

import numpy as np
from math import erfc, sqrt

import nat_core as C
import nat_ctx as X
from nat_ctx import item, chk, chk_table, ctx, tbl, fmt, VENUES, panel, sgn, pos_value, pos_greeks, T3M


def fid(s):
    return s.replace(" ", "_").replace(".", "_").replace("-", "_")


def not_testable(id_, page, title, why):
    return item(id_, page, title, "not testable", text=why)


# ------------------------------------------------------------------------------------------------------------------ 21-1 .. 21-8
LEGS_21_1 = [(10, "P", 0.95, T3M), (-10, "C", 1.05, T3M), (5, "U", 1.0, T3M)]


def _root(f, lo, hi, n=4000):
    g = np.linspace(lo, hi, n)
    y = np.array([f(x) for x in g])
    s = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    out = []
    for i in s:
        a, b = g[i], g[i + 1]
        fa = y[i]
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = f(m)
            if np.sign(fm) == np.sign(fa):
                a, fa = m, fm
            else:
                b = m
        out.append(0.5 * (a + b))
    return out


def ch21_a():
    out = []
    vols = (14 / 18, 1.0, 22 / 18)
    # Figures 21-1, 21-7, 21-8: value / delta / gamma against price for three volatilities
    def pic(kind):
        def draw(ax, v):
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.85 * S0, 1.15 * S0, 400)
            cost = float(pos_value(LEGS_21_1, np.array([S0]), S0, 0.0, sig)[0])
            for m, ls in zip(vols, ("--", "-", ":")):
                if kind == "value":
                    y = (pos_value(LEGS_21_1, g, S0, 0.0, sig * m) - cost) / S0 * 100
                else:
                    y = pos_greeks(LEGS_21_1, g, S0, 0.0, sig * m)[kind]
                    if kind == "gamma":
                        y = y * S0
                ax.plot(g / S0 * 100, y, ls=ls, label=f"volatility {sig * m * 100:.1f}%")
            ax.axhline(0, lw=.4, color="0.6"); ax.set_xlabel("underlying price (% of last close)")
            ax.set_ylabel({"value": "P&L (% of S)", "delta": "position delta", "gamma": "position gamma x S"}[kind]); ax.legend(fontsize=6)
        return panel(draw, f"nat_f21_{kind}")
    # claims
    ck, rows = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        d_ = lambda s, m=1.0: float(pos_greeks(LEGS_21_1, np.array([s]), S0, 0.0, sig * m)["delta"][0])
        g_ = lambda s, m=1.0: float(pos_greeks(LEGS_21_1, np.array([s]), S0, 0.0, sig * m)["gamma"][0])
        v_ = lambda s, m=1.0: float(pos_greeks(LEGS_21_1, np.array([s]), S0, 0.0, sig * m)["vega"][0])
        z = [r_ for r_ in _root(lambda s: g_(s), 0.85 * S0, 1.15 * S0)]
        zz = [r_ for m in (vols[0], vols[2]) for r_ in _root(lambda s, m=m: g_(s, m), 0.85 * S0, 1.15 * S0)]
        zs = sorted(z)
        zero = zs[0] if zs else float("nan")
        spread = max([abs(a - zero) for a in zz] + [0.0]) / S0 * 100
        rows.append([v, fmt(zero / S0 * 100, 3), fmt(d_(zero), 3), fmt(g_(zero) * S0, 5), fmt(v_(zero) / 100 / S0 * 100, 5), fmt(spread, 3)])
        ck.append(chk(f"{v}: gamma is zero at one price for all three volatilities (book: 99.60 for 14, 18, 22%); largest gap between the three zero-gamma prices, % of S (our tolerance 0.5)", 0.0, spread, 0.5))
        ck.append(chk(f"{v}: gamma positive below the zero-gamma price and negative above (21-8)", True, bool(g_(0.9 * zero) > 0 and g_(1.1 * zero) < 0), kind="bool"))
        ck.append(chk(f"{v}: at the zero-gamma price the position delta falls as volatility rises (21-7: +110, 0, -85 from 14%, 18%, 22%)", True, bool(d_(zero, vols[0]) > d_(zero) > d_(zero, vols[2])), kind="bool"))
        ck.append(chk(f"{v}: at the zero-gamma price vega is about 0, so the three value curves meet (21-1); |vega per vol point| as % of S under 0.05 (our tolerance)", True, bool(abs(v_(zero) / 100 / S0 * 100) < 0.05), kind="bool"))
        ck.append(chk(f"{v}: |gamma| is larger at low volatility (21-8: peaks 60 at 14%, 40 at 18%, 30 at 22%)", True, bool(max(abs(g_(s, vols[0])) for s in np.linspace(0.85 * S0, 1.15 * S0, 200)) > max(abs(g_(s)) for s in np.linspace(0.85 * S0, 1.15 * S0, 200)) > max(abs(g_(s, vols[2])) for s in np.linspace(0.85 * S0, 1.15 * S0, 200))), kind="bool"))
    p1, p7, p8 = pic("value"), pic("delta"), pic("gamma")
    tb = tbl(["Venue", "Zero-gamma price (% of S)", "Delta there", "Gamma x S there", "Vega per vol point there (% of S)", "Gap between the three zero-gamma prices (% of S)"], rows)
    legs_txt = "long 10 September 95 puts, short 10 September 105 calls, long 5 underlying contracts (strikes 95%, 105% of our last close; 3 months; r = 0)"
    base = f"Position of Figures 21-1, 21-7 and 21-8: {legs_txt}. The book prints no time or rate and its vols are 14, 18, 22%; ours are sigma x 14/18, sigma, sigma x 22/18."
    out.append(item("Figure 21-1", 123, "Position value as the underlying price and volatility change", "formula", fig=p1, table=tb, checks=ck, text=base))
    out.append(item("Figure 21-7", 126, "Position delta as the underlying price and volatility change", "formula", fig=p7, text=base))
    out.append(item("Figure 21-8", 126, "Position gamma as the underlying price and volatility change", "formula", fig=p8, table=chk_table(ck, 4), text=base + " Checks of the statements the book makes about Figures 21-1, 21-7 and 21-8 are in this table.", checks=[]))
    # 21-2: positive gamma = smile, negative gamma = frown; delta-neutral at the extreme
    ck2, rows2 = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for nm, legs in (("long straddle (positive gamma)", [(1, "C", 1.0, T3M), (1, "P", 1.0, T3M)]), ("short straddle (negative gamma)", [(-1, "C", 1.0, T3M), (-1, "P", 1.0, T3M)])):
            g = np.linspace(0.8 * S0, 1.2 * S0, 801)
            y = pos_value(legs, g, S0, 0.0, sig)
            dl = pos_greeks(legs, g, S0, 0.0, sig)["delta"]
            gm = pos_greeks(legs, g, S0, 0.0, sig)["gamma"]
            conv = bool(np.all(np.diff(y, 2) > -1e-9)) if gm[400] > 0 else bool(np.all(np.diff(y, 2) < 1e-9))
            neutral = float(g[np.argmin(abs(dl))])
            ext = float(g[np.argmin(y)] if gm[400] > 0 else g[np.argmax(y)])
            ck2.append(chk(f"{v}: {nm}: the value curve is a smile (curves up) / frown (curves down)", True, conv, kind="bool"))
            ck2.append(chk(f"{v}: {nm}: the delta-neutral price is the bottom of the smile / top of the frown; both prices as % of S, gap under 0.5 (our tolerance)", 0.0, abs(neutral - ext) / S0 * 100, 0.5))
            ck2.append(chk(f"{v}: {nm}: delta negative left of the neutral price, positive right of it (smile) / the reverse (frown)", True, bool((dl[100] < 0 < dl[700]) if gm[400] > 0 else (dl[100] > 0 > dl[700])), kind="bool"))
            rows2.append([v, nm, fmt(neutral / S0 * 100, 3), fmt(ext / S0 * 100, 3)])
    out.append(item("Figure 21-2", 123, "Positive and negative delta and gamma (a smile and a frown, delta neutral at the extreme)", "formula",
                    table=tbl(["Venue", "Position", "Delta-neutral price (% of S)", "Price at the extreme of the curve (% of S)"], rows2), table2=chk_table(ck2), table2_title="The picture's statements, tested on a long and a short straddle (the book draws no position for this picture; 3 months, r = 0)", checks=ck2,
                    text="The book draws two curves with no numbers. Tested here with a long straddle (positive gamma) and a short straddle (negative gamma): the value curve now, the position delta and gamma."))
    # 21-3, 21-4: time passes
    def timepic(id_, pg, title, sign):
        legs = [(sign, "C", 1.0, T3M), (sign, "P", 1.0, T3M)]
        ck_ = []
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.8 * S0, 1.2 * S0, 401)
            a = pos_value(legs, g, S0, 0.0, sig)
            b = pos_value(legs, g, S0, 0.0, sig, dt=T3M / 4)
            gm = float(pos_greeks(legs, np.array([S0]), S0, 0.0, sig)["gamma"][0]); th = float(pos_greeks(legs, np.array([S0]), S0, 0.0, sig)["theta"][0])
            at_m = float(b[200] - a[200])
            ck_.append(chk(f"{v}: gamma sign, theta sign", "+-" if sign > 0 else "-+", sgn(gm, 1e-12) + sgn(th, 1e-9), kind="text"))
            ck_.append(chk(f"{v}: with no movement in the underlying the position {'loses' if sign > 0 else 'gains'} value as time passes (graph shifted {'downward' if sign > 0 else 'upward'}) at the delta-neutral price", True, bool(at_m < 0) if sign > 0 else bool(at_m > 0), kind="bool"))
            ck_.append(chk(f"{v}: the shift is the same direction at every price on the grid (0.8 to 1.2 of S)", True, bool(np.all(b - a < 0)) if sign > 0 else bool(np.all(b - a > 0)), kind="bool"))
        return item(id_, pg, title, "formula", table=chk_table(ck_), checks=ck_, text=f"The book draws a curve with no numbers. Tested with a {'long' if sign > 0 else 'short'} straddle (3 months, r = 0): value now against value one quarter of the time to expiry later.")
    out.append(timepic("Figure 21-3", 124, "Positive gamma, negative theta position as time passes", 1))
    out.append(timepic("Figure 21-4", 124, "Negative gamma, positive theta position as time passes", -1))
    # 21-5, 21-6: volatility changes
    def volpic(id_, pg, title, sign):
        ck_ = []
        for lab, q_ in (("long straddle", 1), ("short straddle", -1)):
            pass
        for v in VENUES:
            c = ctx(v)
            S0, sig = c["S"], c["sig"]
            g = np.linspace(0.8 * S0, 1.2 * S0, 401)
            # sign x straddle has gamma sign; add an out-of-the-money overlay that flips vega: strangle vs straddle
            pos_vega = [(sign, "C", 1.0, T3M), (sign, "P", 1.0, T3M)]
            neg_vega = [(sign, "C", 1.0, T3M), (sign, "P", 1.0, T3M), (-sign * 2, "C", 1.3, T3M), (-sign * 2, "P", 0.7, T3M)]
            for nm, legs in (("vega positive", pos_vega if sign > 0 else [(-q, k, x, t) for q, k, x, t in pos_vega]), ):
                pass
            a = pos_value(pos_vega, g, S0, 0.0, sig)
            up = pos_value(pos_vega, g, S0, 0.0, sig, dsig=0.05 * sig)
            vg = float(pos_greeks(pos_vega, np.array([S0]), S0, 0.0, sig)["vega"][0])
            exp_up = (vg > 0)
            ck_.append(chk(f"{v}: {'long' if sign > 0 else 'short'} straddle: vega sign {'positive' if exp_up else 'negative'}; the position {'gains' if exp_up else 'loses'} value when volatility rises (graph shifted {'upward' if exp_up else 'downward'}), at the delta-neutral price", True,
                           bool(up[200] > a[200]) if exp_up else bool(up[200] < a[200]), kind="bool"))
            ck_.append(chk(f"{v}: the same direction at every price on the grid", True, bool(np.all(up > a)) if exp_up else bool(np.all(up < a)), kind="bool"))
        return item(id_, pg, title, "formula", table=chk_table(ck_), checks=ck_, text="The book draws curves with no numbers and says that, whatever the gamma, a positive-vega position gains when volatility rises and a negative-vega position gains when it falls. Tested with straddles (3 months, r = 0): value now against value with volatility 5% higher (relative).")
    out.append(volpic("Figure 21-5", 125, "Positive gamma position as volatility changes", 1))
    out.append(volpic("Figure 21-6", 125, "Negative gamma position as volatility changes", -1))
    return out


# ------------------------------------------------------------------------------------------------------------------ 21-9 .. 21-13
STRIKES_9 = [90, 95, 100, 105, 110]
CALLS_9 = [-4, -12, 14, -17, 12]
PUTS_9 = [13, -20, -8, 12, 8]
UND_9 = 13
S9, T9, SIG9, R9 = 101.25, 42 / 365, 0.27, 0.0


def pos9(S, sig=SIG9, dt=0.0, calls=CALLS_9, puts=PUTS_9, und=UND_9):
    legs = [(q, "C", x, T9) for q, x in zip(calls, STRIKES_9)] + [(q, "P", x, T9) for q, x in zip(puts, STRIKES_9)] + [(und, "U", 1.0, T9)]
    return legs


def tot9(S, sig=SIG9, who="all"):
    """delta (x100), gamma (x100), theta per day, vega per vol point: position totals per the book's table (contracts, no multiplier)."""
    S = np.asarray(S, float)
    tot = {k: np.zeros_like(S) for k in ("delta", "gamma", "theta", "vega")}
    for xs, cq, pq in zip(STRIKES_9, CALLS_9, PUTS_9):
        g = C.greeks(S, float(xs), T9, R9, sig, None)
        for k in tot:
            if who in ("all", "calls"):
                tot[k] = tot[k] + cq * g[k][0]
            if who in ("all", "puts"):
                tot[k] = tot[k] + pq * g[k][1]
    if who == "all":
        tot["delta"] = tot["delta"] + UND_9
    return {"delta": tot["delta"] * 100, "gamma": tot["gamma"] * 100, "theta": tot["theta"] / 365, "vega": tot["vega"] / 100}


def val9(S, sig=SIG9, dt=0.0):
    S = np.asarray(S, float)
    tot = np.zeros_like(S)
    for xs, cq, pq in zip(STRIKES_9, CALLS_9, PUTS_9):
        t = max(T9 - dt, 1e-9)
        c, p = C.bs(S, float(xs), t, R9, sig, None)
        tot = tot + cq * c + pq * p
    return tot + UND_9 * S


def ch21_b():
    out = []
    # ---- the printed table
    per_book = {  # strike: (call delta, put delta, gamma, theta, vega) per contract, as printed
        90: (90.9, -9.1, 1.77, -0.0181, 0.056), 95: (77.1, -22.9, 3.27, -0.0335, 0.104), 100: (57.2, -42.8, 4.23, -0.0433, 0.135), 105: (36.3, -63.7, 4.04, -0.0414, 0.129), 110: (19.5, -80.5, 2.97, -0.0304, 0.095)}
    ck, rows = [], []
    for xs in STRIKES_9:
        g = C.greeks(S9, float(xs), T9, R9, SIG9, None)
        b = per_book[xs]
        mine = (g["delta"][0] * 100, g["delta"][1] * 100, g["gamma"][0] * 100, g["theta"][0] / 365, g["vega"][0] / 100)
        names = ("call delta", "put delta", "gamma", "theta", "vega")
        tols = (0.05, 0.05, 0.005, 0.00005, 0.0005)
        for nm, bb, mm, tl in zip(names, b, mine, tols):
            ck.append(chk(f"{xs} {nm}", bb, mm, tl))
        rows.append([xs] + [fmt(bb, 4) + " / " + fmt(mm, 4) for bb, mm in zip(b, mine)])
    out.append(item("Figure 21-9 (per-contract values)", 127, "The case-study position: price 101.25, 6 weeks, volatility 27%, rate 0: delta, gamma, theta and vega of each call and put", "formula",
                    table=tbl(["Exercise", "Call delta (book / ours)", "Put delta", "Gamma", "Theta (abs.)", "Vega"], rows), table2=chk_table(ck, 4), table2_title="Each printed number (tolerance: half a unit of the last printed digit)",
                    checks=ck, text="Inputs as printed: S 101.25, 6 weeks (42 days / 365), volatility 27.00%, r 0. Theta is per calendar day, vega per volatility point, delta and gamma per 100 as printed."))
    # ---- positions
    book_pos = {
        "90C": (-4, -363.6, -7.08, 0.0724, -0.224), "95C": (-12, -925.2, -39.24, 0.4020, -1.248), "100C": (14, 800.8, 59.22, -0.6062, 1.890), "105C": (-17, -617.1, -68.68, 0.7038, -2.193), "110C": (12, 234.0, 35.64, -0.3648, 1.140),
        "90P": (13, -118.3, 23.01, -0.2353, 0.728), "95P": (-20, 458.0, -65.40, 0.6700, -2.080), "100P": (-8, 342.4, -33.84, 0.3464, -1.080), "105P": (12, -764.4, 48.48, -0.4968, 1.548), "110P": (8, -644.0, 23.76, -0.2432, 0.760)}
    ck2, rows2 = [], []
    for nm, (q, d, g_, th, vg) in book_pos.items():
        x = int(nm[:-1])
        kind = 0 if nm.endswith("C") else 1
        g = C.greeks(S9, float(x), T9, R9, SIG9, None)
        mine = (q * g["delta"][kind] * 100, q * g["gamma"][kind] * 100, q * g["theta"][kind] / 365, q * g["vega"][kind] / 100)
        tols = (abs(q) * 0.05 + 0.02, abs(q) * 0.005 + 0.002, abs(q) * 0.00005 + 0.00005, abs(q) * 0.0005 + 0.0005)
        for lab, bb, mm, tl in zip(("delta", "gamma", "theta", "vega"), (d, g_, th, vg), mine, tols):
            ck2.append(chk(f"{nm} ({q:+d}) position {lab}", bb, mm, tl))
        rows2.append([nm, f"{q:+d}"] + [fmt(bb, 4) + " / " + fmt(mm, 4) for bb, mm in zip((d, g_, th, vg), mine)])
    def rowtol(sel):
        return [sum(abs(q) * u for (nm_, (q, *_)) in book_pos.items() if sel(nm_)) + 0.02 * 0 for u in (0.05, 0.005, 0.00005, 0.0005)]
    tol_tot = {"calls": rowtol(lambda n_: n_.endswith("C")), "puts": rowtol(lambda n_: n_.endswith("P")), "all": rowtol(lambda n_: True)}
    tc, tp = tot9(S9, SIG9, "calls"), tot9(S9, SIG9, "puts")
    ta = tot9(S9, SIG9, "all")
    totals_book = {"calls": (-871.1, -20.14, 0.2072, -0.635), "puts": (-726.3, -3.99, 0.0411, -0.124), "all": (-297.4, -24.13, 0.2483, -0.759)}
    for lab, tt in (("calls", tc), ("puts", tp), ("all", ta)):
        for k, bb, tl in zip(("delta", "gamma", "theta", "vega"), totals_book[lab], tol_tot[lab]):
            ck2.append(chk(f"Totals ({lab}): {k} (tolerance = sum of the half-units of the book's rounding of the per-contract numbers x quantities)", bb, float(tt[k]), tl))
    out.append(item("Figure 21-9 (position)", 127, "The case-study position: each leg's delta, gamma, theta and vega times its quantity, and the totals", "formula",
                    table=tbl(["Leg", "Quantity", "Delta (book / ours)", "Gamma", "Theta", "Vega"], rows2), table2=chk_table(ck2, 4), table2_title="Every printed position number and total", checks=ck2,
                    text="Legs as printed: calls 90:-4, 95:-12, 100:+14, 105:-17, 110:+12; puts 90:+13, 95:-20, 100:-8, 105:+12, 110:+8; underlying +13 (1,300 deltas). The OCR of the printed 110C delta reads 4234.0; the page arithmetic (12 x 19.5 = 234.0) is used."))
    # ---- text statements
    ck3 = []
    d0, g0 = float(ta["delta"]), float(ta["gamma"])
    est = S9 - (297.4 / 24.13)
    ck3.append(chk("Delta-neutral estimate 101.25 - 297.4/24.13 = 88.93 (the book's arithmetic)", 88.93, est, 0.006))
    ck3.append(chk("Delta-neutral estimate from our totals", 88.93, S9 - abs(d0) / abs(g0), 0.15))
    ck3.append(chk("Breakeven volatility with edge 6.00: 27 + 6.00/0.759 = 34.90", 34.90, 27 + 6.00 / 0.759, 0.006))
    ck3.append(chk("Breakeven volatility with edge 8.00: 27 + 8.00/0.759 = 37.54", 37.54, 27 + 8.00 / 0.759, 0.006))
    ck3.append(chk("Breakeven volatility with vega -0.65: 27 + 6.00/0.65 = 36.23", 36.23, 27 + 6.00 / 0.65, 0.006))
    ck3.append(chk("Net contracts on the upside: -7 calls + 13 underlying = +6", 6, sum(CALLS_9) + UND_9, 0.0))
    ck3.append(chk("Net contracts on the downside: 13 underlying - 5 (net long 5 puts act like 5 short underlying) = +8", 8, UND_9 - sum(PUTS_9), 0.0))
    # extreme deltas as volatility rises and as volatility falls
    hi = max(float(tot9(S9, s)["delta"]) for s in np.exp(np.linspace(np.log(0.05), np.log(3.0), 120)))
    d_hi = float(tot9(S9, 3.0)["delta"])
    d_lo = float(tot9(S9, 0.002)["delta"])
    ck3.append(chk("As volatility falls (or time passes) delta tends to -900 (ITM calls to 100, ITM puts to -100): value at volatility 0.2%", -900, d_lo, 1.0))
    ck3.append(chk("As volatility rises delta becomes positive (the book's limit +700 if all deltas reached 50): largest position delta over volatilities 5%..300%", 700, hi, None, kind="num", ok=bool(hi > 0)))
    ck3.append(chk("As volatility rises the current delta of -297 becomes less negative and turns positive", True, bool(hi > 0 and float(tot9(S9, 0.27)["delta"]) < float(tot9(S9, 0.4)["delta"])), kind="bool"))
    # figure statements 21-10 .. 21-13 at 27%
    grid = np.linspace(80, 124.5, 8901)
    dl = tot9(grid)["delta"]; gm = tot9(grid)["gamma"]
    pl = val9(grid) - float(val9(np.array([S9]))[0])
    z_d = [r_ for r_ in _root(lambda s: float(tot9(s)["delta"]), 85, 100.5)]
    z_g = [r_ for r_ in _root(lambda s: float(tot9(s)["gamma"]), 101.25, 112)]
    z_dp = [r_ for r_ in _root(lambda s: float(tot9(s)["delta"]), 101.25, 120)]
    z_pl = [r_ for r_ in _root(lambda s: float(val9(np.array([s]))[0] - val9(np.array([S9]))[0]), 102, 140)]
    gmin = float(grid[np.argmin(np.where(grid < 101.25, gm, 1e9))])
    gmin_v = float(np.min(np.where(grid < 101.25, gm, 1e9)))
    sp = np.gradient(gm, grid)
    pos = grid[sp > 0]
    ck3.append(chk("Figure 21-10/21-11: on the downside the position delta is 0 (maximum profit) at about 95.00", 95.0, z_d[-1] if z_d else float("nan"), 0.5))
    ck3.append(chk("Figure 21-12: on the downside the gamma reaches its most negative value of about -80 ...", -80.0, gmin_v, 4.0))
    ck3.append(chk("... at a price of about 93", 93.0, gmin, 1.0))
    ck3.append(chk("Figure 21-12: the gamma becomes positive again at about 104", 104.0, z_g[0] if z_g else float("nan"), 0.5))
    ck3.append(chk("Figure 21-11: the delta becomes positive at about 112", 112.0, z_dp[0] if z_dp else float("nan"), 0.5))
    ck3.append(chk("Text: above about 124 the position begins to show a profit (position value above its value at 101.25)", 124.0, z_pl[0] if z_pl else float("nan"), 1.0))
    sp_lo = float(grid[(grid > 85) & (grid < 105)][np.argmax(sp[(grid > 85) & (grid < 105)] > 0)]) if np.any(sp[(grid > 85) & (grid < 105)] > 0) else float("nan")
    seg = grid[(grid > 100)]
    sp_hi = float(seg[np.argmax(sp[grid > 100] < 0)]) if np.any(sp[grid > 100] < 0) else float("nan")
    ck3.append(chk("Figure 21-12: speed is positive from about 93 ...", 93.0, sp_lo, 1.0))
    ck3.append(chk("... to about 114", 114.0, sp_hi, 1.0))
    # volatility dependence of speed
    s_at = 100.0
    spd = lambda sg: float(np.gradient(tot9(np.array([s_at - 0.01, s_at, s_at + 0.01]), sg)["gamma"], 0.01)[1])
    ck3.append(chk("Figure 21-12: lowering volatility makes the speed larger, raising it makes it smaller (at price 100: speed at 22%, 27%, 32%)", True, bool(spd(0.22) > spd(0.27) > spd(0.32)), kind="bool"))
    # Figure 21-13: value against volatility
    v0 = float(val9(np.array([S9]))[0])
    vg = np.linspace(0.17, 0.37, 201)
    w6 = np.array([val9(np.array([S9]), s)[0] - v0 for s in vg])
    w4 = np.array([val9(np.array([S9]), s, dt=14 / 365)[0] - val9(np.array([S9]), 0.27, dt=14 / 365)[0] for s in vg])
    w2 = np.array([val9(np.array([S9]), s, dt=28 / 365)[0] - val9(np.array([S9]), 0.27, dt=28 / 365)[0] for s in vg])
    cross = _root(lambda s: float(val9(np.array([S9]), s)[0] - v0 + 6.0), 0.27, 0.60)
    cv = lambda dt: (lambda f: float((f(0.28) - 2 * f(0.27) + f(0.26)) / 0.01 ** 2 / 10000))(lambda s: float(val9(np.array([S9]), s, dt=dt)[0]))
    v6, v4, v2 = cv(0.0), cv(14 / 365), cv(28 / 365)
    ck3.append(chk("Figure 21-13: the six-week curve crosses -6.00 (offsets an edge of +6.00) at a volatility of about 36 (%)", 36.0, (cross[0] * 100) if cross else float("nan"), 1.0))
    ck3.append(chk("The breakeven volatility is higher than the linear estimate 34.90 because volga is positive at six weeks", True, bool(cross and cross[0] * 100 > 34.90 and v6 > 0), kind="bool"))
    ck3.append(chk("Four weeks to expiry: approximately a straight line (|volga| below a quarter of the six-week volga; our tolerance)", True, bool(abs(v4) < 0.25 * abs(v6)), kind="bool"))
    ck3.append(chk("Two weeks to expiry: curves slightly downward (volga negative)", True, bool(v2 < 0), kind="bool"))
    ck3.append(chk("The position vega is -0.759 per volatility point (printed total) against ours", -0.759, float(ta["vega"]), 0.005))

    def draw10(ax, v=None):
        for sg, ls in ((0.22, ":"), (0.27, "-"), (0.32, "--")):
            ax.plot(grid, val9(grid, sg) - float(val9(np.array([S9]), sg)[0]), ls=ls, label=f"{sg*100:.0f}%")
        ax.axhline(0, lw=.4, color="0.6"); ax.axvline(S9, lw=.4, ls=":", color="0.6"); ax.set_xlabel("underlying price"); ax.set_ylabel("theoretical P&L"); ax.legend(fontsize=6)
    import matplotlib.pyplot as plt
    import writeup_core as W
    def one(draw, name, title):
        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        draw(ax)
        ax.set_title(title, fontsize=8); fig.tight_layout()
        return W.save(fig, name)
    def d11(ax):
        for sg, ls in ((0.22, ":"), (0.27, "-"), (0.32, "--")):
            ax.plot(grid, tot9(grid, sg)["delta"], ls=ls, label=f"{sg*100:.0f}%")
        ax.axhline(0, lw=.4, color="0.6"); ax.axvline(S9, lw=.4, ls=":", color="0.6"); ax.set_xlabel("underlying price"); ax.set_ylabel("position delta"); ax.legend(fontsize=6)
    def d12(ax):
        for sg, ls in ((0.22, ":"), (0.27, "-"), (0.32, "--")):
            ax.plot(grid, tot9(grid, sg)["gamma"], ls=ls, label=f"{sg*100:.0f}%")
        ax.axhline(0, lw=.4, color="0.6"); ax.axvline(S9, lw=.4, ls=":", color="0.6"); ax.set_xlabel("underlying price"); ax.set_ylabel("position gamma"); ax.legend(fontsize=6)
    def d13(ax):
        ax.plot(vg * 100, w6, label="6 weeks"); ax.plot(vg * 100, w4, "--", label="4 weeks"); ax.plot(vg * 100, w2, ":", label="2 weeks")
        ax.axhline(-6.0, lw=.4, color="0.6"); ax.set_xlabel("volatility (%)"); ax.set_ylabel("position value change"); ax.legend(fontsize=6)
    tb = chk_table(ck3, 3)
    base = "Case-study position of Figure 21-9 with the book's own inputs (S 101.25, 6 weeks, 27%, r 0); this is a calculation, so it is the same on both venues; the book gives no data that depends on a venue."
    out.append(item("Figure 21-10", 131, "Position value as the underlying price and volatility change (case-study position)", "formula", fig=one(draw10, "nat_f21_10", "Fig 21-10 (book inputs)"), table=tb, checks=ck3, text=base + " The statements of the text and of Figures 21-10 to 21-13 are tested in this table."))
    out.append(item("Figure 21-11", 131, "Position delta as the underlying price and volatility change (case-study position)", "formula", fig=one(d11, "nat_f21_11", "Fig 21-11 (book inputs)"), text=base))
    out.append(item("Figure 21-12", 132, "Position gamma as the underlying price and volatility change (case-study position)", "formula", fig=one(d12, "nat_f21_12", "Fig 21-12 (book inputs)"), text=base))
    out.append(item("Figure 21-13", 133, "Position value as volatility changes and time passes (case-study position)", "formula", fig=one(d13, "nat_f21_13", "Fig 21-13 (book inputs)"), text=base + f" The page prints the underlying price of the figure as 107.25 in its caption box; the text says 101.25 and 101.25 is used. Our curvature of the value in volatility (second difference per vol point squared) at 27%: six weeks {v6:+.5f}, four weeks {v4:+.5f}, two weeks {v2:+.5f}."))
    return out


# ------------------------------------------------------------------------------------------------------------------ 21-14 .. 21-20
def ch21_c():
    out = []
    for i, (pg, ttl) in enumerate([(134, "Position value as the underlying price and volatility change"), (135, "Position delta as the underlying price and volatility change"), (135, "Position gamma as the underlying price and volatility change"),
                                   (136, "Position vega as the underlying price and volatility change"), (136, "Position value as the underlying price changes and time passes")]):
        out.append(not_testable(f"Figure 21-{14 + i}", pg, ttl, "Not testable: the picture is a curve of a position whose legs, prices, volatility and time are not printed in this PDF."))
    out.append(not_testable("Figure 21-19", 137, "Risk sensitivities as the underlying price changes (a table of totals for eleven stock prices)", "Not testable: the table lists totals (theoretical P&L, delta, gamma, adjusted vega, theta, rho, vanna, charm, speed, color, volga, vegadecay, zomma) for a position whose legs are not printed in this PDF, and 'adjusted vega' is not defined in it."))
    # 21-20
    S, r, D, td = 34.38, 0.0495, 0.29, 10 / 52
    strikes = [27.5, 30, 32.5, 35, 37.5, 40, 42.5]
    pos = {"April": (28, 0.3427, [154, -324, 26, 212, -16, -62, -270], [94, 222, -154, -38, 122, -36, 92]),
           "June": (91, 0.3295, [0, 6, 184, -220, 172, 42, -50], [-206, 162, -50, -98, -4, 0, 14]),
           "August": (147, 0.3164, [0, 48, 0, -52, 16, 60, -144], [64, -92, 0, -40, -50, 0, 0])}
    book = {"April": (-280, 424, -25230, 2016, -2.51, 6.26, -7.50), "June": (134, -182, 12186, 52, -0.62, 0.36, 9.75), "August": (-72, -118, 6850, -1008, 1.25, -11.19, 10.45)}
    Sa = S - D * np.exp(r * td)
    ck, rows = [], []
    for nm, (days, sig, cs, ps) in pos.items():
        t = days / 365
        tot = np.zeros(5)
        for X_, cq, pq in zip(strikes, cs, ps):
            g = C.greeks(Sa, X_, t, r, sig, None)
            for k, name in enumerate(("delta", "gamma", "theta", "vega", "rho")):
                tot[k] += cq * g[name][0] + pq * g[name][1]
        mine = (sum(cs), sum(ps), tot[0] * 100, tot[1] * 100, tot[2] / 365, tot[3] / 100, tot[4] / 100)
        bk = book[nm]
        names = ("call contracts", "put contracts", "delta", "gamma", "theta", "vega (adjusted)", "rho")
        tols = (0.0, 0.0, 30.0, 3.0, 0.02, None, 0.05)
        for lab, bb, mm, tl in zip(names, bk, mine, tols):
            if lab == "vega (adjusted)":
                continue
            ck.append(chk(f"{nm}: {lab}", bb, mm, tl))
        rows.append([nm, f"{sum(cs):+d} / {bk[0]:+d}", f"{sum(ps):+d} / {bk[1]:+d}", f"{mine[2]:+,.0f} / {bk[2]:+,}", f"{mine[3]:+,.0f} / {bk[3]:+,}", f"{mine[4]:+.2f} / {bk[4]:+.2f}", f"{mine[6]:+.2f} / {bk[6]:+.2f}"])
    # diagnostic for the April line only: the printed totals if the 37.5 put were +244 (column then sums to the printed 424)
    t = 28 / 365
    dg = np.zeros(5)
    cs, ps = pos["April"][2], list(pos["April"][3]); ps[4] = 244
    for X_, cq, pq in zip(strikes, cs, ps):
        g = C.greeks(Sa, X_, t, r, pos["April"][1], None)
        for k, name in enumerate(("delta", "gamma", "theta", "vega", "rho")):
            dg[k] += cq * g[name][0] + pq * g[name][1]
    diag = f"With the 37.5 April put at +244 instead of +122 (not a printed input; shown only to locate the gap) the column sums to the printed 424 and our April line is delta {dg[0]*100:+,.0f}, gamma {dg[1]*100:+,.0f}, theta {dg[2]/365:+.2f}, rho {dg[4]/100:+.2f}: nearer to the printed April line (delta -25,230, gamma +2,016, theta -2.51, rho -7.50) but still not equal to it."
    out.append(item("Figure 21-20", 138, "The effect of a 2-for-1 stock split: the option position (strikes by month), its adjusted contract counts and its risk totals", "formula",
                    table=tbl(["Month", "Call contracts (ours from the printed rows / printed total)", "Put contracts", "Delta", "Gamma", "Theta", "Rho"], rows), table2=chk_table(ck, 3), table2_title="Each printed total against the book's formulas (tolerances: delta 30 shares, gamma 3, theta 0.02, rho 0.05; contracts exact)", checks=ck,
                    text=("Inputs as printed: share price 34.38, r 4.95%, expected dividend 0.29 in 10 weeks (the share price is reduced by the dividend compounded at r: S - D e^(r td)), April 4 weeks 34.27%, June 13 weeks 32.95%, August 21 weeks 31.64%; "
                          "position rows as printed; theta per calendar day and times 1 (one option), delta and gamma times 100, rho per 1% (these scalings reproduce the June and August lines). Time in calendar days / 365. The 'adjusted vega' column is not tested because the PDF does not define it. "
                          "The printed April put rows sum to 302 contracts while the printed April total is 424."), note=diag))
    return out


# ------------------------------------------------------------------------------------------------------------------ Chapter 23
def ch23():
    out = []
    import writeup_core as W
    import matplotlib.pyplot as plt
    from scipy import stats

    def draw(ax, v):
        c = ctx(v)
        x = c["ret"].values
        ax.hist(x, bins=120, density=True, alpha=.6, label="daily log returns")
        g = np.linspace(x.min(), x.max(), 400)
        ax.plot(g, stats.norm.pdf(g, x.mean(), x.std()), label="normal, same mean and st. dev.")
        ax.set_yscale("log"); ax.set_ylim(1e-3, None); ax.legend(fontsize=6); ax.set_xlabel("daily log return")
    p = panel(draw, "nat_f23_1")
    rows, ck = [], []
    for v in VENUES:
        x = ctx(v)["ret"].values
        m = C.moments(x)
        rows.append([v, f"{len(x):,}", fmt(m["skew"], 4), fmt(m["kurt"], 3), fmt(m["kurt"] - 3, 3)])
        ck.append(chk(f"{v}: kurtosis (m4/m2^2, book's normal = 3) of daily log returns exceeds 3 (more big and more small occurrences than a normal)", True, bool(m["kurt"] > 3), kind="bool"))
    out.append(item("Figure 23-1", 139, "Skewness: the degree to which one tail of a distribution is longer than the other", "data", fig=p, table=tbl(["Venue", "Daily returns", "Skewness (m3 / m2^1.5)", "Kurtosis (m4 / m2^2)", "Excess kurtosis"], rows), checks=ck[:0],
                    text="Appendix B definitions of the skewness and the kurtosis (nat_core.moments), on all daily log returns of each venue. Positive skewness: right tail longer."))
    out.append(item("Figure 23-2", 140, "Kurtosis: the degree to which a distribution has a taller peak and wider tails", "data", table=chk_table(ck), checks=ck,
                    text="The book's statement: a distribution with positive kurtosis has more big occurrences and more small occurrences than the normal. Kurtosis of our returns is in the table of Figure 23-1; the histogram there is drawn on a log scale against the normal curve."))
    # 23-3: the book's own table (probabilities from the normal distribution) and the same table on our data
    book = [("S&P 500", 1.31, 11.58, 8.84, None, 9.03, 6.89, 358e9), ("Crude oil", 2.25, 14.27, 6.34, 8.7e9, 10.80, 4.80, 1.26e6), ("Euro", 0.651, 3.52, 5.41, 31.7e6, 2.40, 3.69, 8.9e3), ("Bund", 0.364, 1.98, 5.40, 30e6, 1.50, 4.10, 48e3)]
    ck3 = []
    for nm, sd, up, upsd, upo, dn, dnsd, dno in book:
        ck3.append(chk(f"{nm}: up move in st. dev. = {up}/{sd}", upsd, up / sd, 0.006))
        ck3.append(chk(f"{nm}: down move in st. dev. = {dn}/{sd}", dnsd, dn / sd, 0.006))
        for lab, z, o in (("up", upsd, upo), ("down", dnsd, dno)):
            if o is None:
                continue
            odds = 1.0 / (0.5 * erfc(z / sqrt(2)))
            ck3.append(chk(f"{nm}: {lab} {z} st. dev., one-sided normal probability 1 chance in", o, odds, None, ok=bool(abs(np.log10(odds) - np.log10(o)) < 0.02)))
    rows = []
    for v in VENUES:
        x = ctx(v)["close"].pct_change().dropna()
        sd = float(x.std())
        up, dn = float(x.max()), float(x.min())
        for lab, mv in (("up", up), ("down", dn)):
            z = abs(mv) / sd
            rows.append([v, f"{sd*100:.3f}%", lab, f"{abs(mv)*100:.2f}%", f"{z:.2f}", f"1 in {1.0/(0.5*erfc(z/sqrt(2))):,.0f}" if z < 37 else "too unlikely to calculate"])
    out.append(item("Figure 23-3", 140, "Probabilities associated with the biggest up and down moves (normal distribution): the book's table, and the same table on our data", "data",
                    table=chk_table(ck3, 2), table2=tbl(["Venue", "One standard deviation (daily % change)", "Move", "Biggest move", "In st. dev.", "Normal probability"], rows), table2_title="The same table on our data: one st. dev. is the standard deviation of all daily simple returns (last close to last close) of the venue, the probability is the one-sided normal tail 1 - N(z)",
                    checks=ck3, text="First table: the book's printed rows checked with the book's arithmetic (move / st. dev., and the one-sided normal tail, 'within a factor of 1.05' because the book rounds the standard deviations). Second table: the same calculation on our data."))
    return out


def ch24():
    return [not_testable("Figure 24-1", 141, "FTSE 100 volatility surface, March 16, 2012", "Not testable: needs the implied volatilities of listed options by strike and expiry; we have none for either venue."),
            not_testable("Figure 24-2", 142, "(a) Declining skewness, (b) increasing skewness: which exercise prices to buy and sell in an investment skew and a demand skew", "Not testable: the rule is about how an implied-volatility skew is expected to change; we have no option quotes, so there is no skew to flatten or steepen."),
            not_testable("Figure 24-3", 143, "Rising and falling kurtosis: buy strangles when the skew is expected to become more curved, sell them when less curved", "Not testable: the rule is about how the curvature of an implied-volatility skew is expected to change; we have no option quotes.")]


def all_items():
    return ch21_a() + ch21_b() + ch21_c() + ch23() + ch24()
