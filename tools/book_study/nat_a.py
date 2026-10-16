"""Natenberg write-up items: Chapters 1-6 and Appendix B (settlement tables, parity graphs, theoretical value, distributions, volatility arithmetic).  New file."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import nat_core as C
import nat_ctx as X
import writeup_core as W
from nat_ctx import item, chk, chk_table, ctx, ctx_line, tbl, fmt, VENUES, SYM, save, panel


def fid(s):
    return s.replace(" ", "_").replace(".", "_")


# ----------------------------------------------------------------------------------------------------------------- Chapter 1
def stock_settlement(p, size=(1200, -500, 0, -700)):
    """Figure 1-1: buy 1,200 on day 1, sell 500 on day 2, no trade day 3, sell the remaining 700 on day 4 (the book's own trade sizes)."""
    pos, cost, real, rows = 0, None, 0.0, []
    for i, q in enumerate(size):
        if i == 0:
            cost = p[0]
        if q < 0:
            real += (p[i] - cost) * (-q)
        pos += q
        unreal = (p[i] - cost) * pos
        rows.append((i + 1, p[i], q, pos, real, unreal))
    return rows


def futures_settlement(p, trades=(-9, 0, 2, 4, 3), unit=1000, margin=3000):
    """Figure 1-2: sell 9 on day 1, no trade day 2, buy 2 day 3, buy 4 day 4, buy 3 day 5 (the book's trade sizes); variation = (today - yesterday) x position held overnight x unit."""
    pos, cum, rows = 0, 0.0, []
    for i, q in enumerate(trades):
        var = (p[i] - p[i - 1]) * pos * unit if i else 0.0
        cum += var
        pos += q
        rows.append((i + 1, p[i], q, pos, abs(pos) * margin, var, cum))
    return rows


def ch1():
    out = []
    pb = [53, 57, 51, 54]
    r = stock_settlement(pb)
    ck = [chk("Day 2 realised P&L", 2000, r[1][4], 0.5), chk("Day 2 unrealised P&L", 2800, r[1][5], 0.5), chk("Day 3 unrealised P&L", -1400, r[2][5], 0.5), chk("Day 4 cumulative realised P&L", 2700, r[3][4], 0.5)]
    rows = [[f"Day {x[0]}", x[1], x[2], x[3], fmt(x[4], 0), fmt(x[5], 0)] for x in r]
    t2 = {"cols": ["Venue", "Day-1..4 closes (our last four daily closes)", "Day 2 realised", "Day 2 unrealised", "Day 3 unrealised", "Day 4 cumulative realised"], "rows": []}
    for v in VENUES:
        c = ctx(v)
        p = [float(x) for x in c["close"].iloc[-4:]]
        q = stock_settlement(p)
        t2["rows"].append([f"{v} {c['sym']}", ", ".join(f"{x:,.2f}" for x in p), fmt(q[1][4], 0), fmt(q[1][5], 0), fmt(q[2][5], 0), fmt(q[3][4], 0)])
    out.append(item("Figure 1-1", 3, "Stock-type settlement: a four-day trade with its realised and unrealised P&L", "formula",
                    table=tbl(["Day", "Price", "Trade (shares)", "Position", "Cumulative realised", "Unrealised"], rows), table2=t2, table2_title="Same four trades (+1,200, -500, 0, -700) on the last four daily closes of each venue",
                    text="First table: the book's own prices (53, 57, 51, 54) and trade sizes. Checks against the printed P&L are below.", note="Bar-closing prices stand in for the book's day prices; the trade sizes are the book's.", checks=ck))
    out[-1]["table3"] = chk_table(ck, 0)
    pf = [75, 77, 74, 70, 80]
    q = futures_settlement(pf)
    ck = [chk("Day 2 variation", -18000, q[1][5], 0.5), chk("Day 3 variation", 9000 - (-18000) if False else 27000, q[2][5], 0.5), chk("Day 3 cumulative", 9000, q[2][6], 0.5), chk("Day 4 cumulative", 37000, q[3][6], 0.5), chk("Day 5 cumulative (closing)", 7000, q[4][6], 0.5),
          chk("Day 3 margin requirement", 21000, q[2][4], 0.5), chk("Day 4 margin requirement", 9000, q[3][4], 0.5)]
    t2 = {"cols": ["Venue", "Day-1..5 closes", "Day 5 cumulative P&L ($)"], "rows": []}
    for v in VENUES:
        c = ctx(v)
        p = [float(x) for x in c["close"].iloc[-5:]]
        t2["rows"].append([f"{v} {c['sym']}", ", ".join(f"{x:,.2f}" for x in p), fmt(futures_settlement(p)[4][6], 0)])
    out.append(item("Figure 1-2", 3, "Futures-type settlement: daily variation and margin on a nine-lot short", "formula",
                    table=tbl(["Day", "Futures price", "Trade", "Position", "Margin requirement ($)", "Variation ($)", "Cumulative ($)"], [[x[0], x[1], x[2], x[3], fmt(x[4], 0), fmt(x[5], 0), fmt(x[6], 0)] for x in q]),
                    table2=t2, table2_title="Same trades (-9, 0, +2, +4, +3; 1,000 units, margin 3,000) on the last five daily closes of each venue", text="First table: the book's prices (75, 77, 74, 70, 80). Checks are below.",
                    checks=ck))
    out[-1]["table3"] = chk_table(ck, 0)
    out.append(item("Figure 1-3", 4, "The clearing process (a flow chart of reports between buyer, seller, clearing firms and clearinghouse)", "not testable",
                    text="A flow chart of a process, with no number or formula in it. Nothing to run."))
    out.append(item("Figure 1-4", 4, "Costs and benefits of buying now for each kind of instrument (a table of words)", "not testable",
                    text="A table of words (interest, storage, insurance, dividends, coupons, convenience yield); no number or formula. Nothing to run."))
    return out


# ----------------------------------------------------------------------------------------------------------------- Chapters 3-4
def ch3():
    rows, ck = [], []
    for v in VENUES:
        c = ctx(v)
        S = c["S"]
        for k in (0.8, 0.9, 1.0, 1.1, 1.2):
            Xs = k * S
            cl = "ATM" if k == 1.0 else ("ITM" if Xs < S else "OTM")
            pu = "ATM" if k == 1.0 else ("ITM" if Xs > S else "OTM")
            ok = (cl == ("ATM" if k == 1.0 else ("ITM" if max(S - Xs, 0) > 0 else "OTM"))) and (pu == ("ATM" if k == 1.0 else ("ITM" if max(Xs - S, 0) > 0 else "OTM")))
            rows.append([v, f"{c['sym']} {Xs:,.2f} ({k:.0%} of S)", cl, pu, fmt(max(S - Xs, 0), 2), fmt(max(Xs - S, 0), 2)])
            ck.append(chk(f"{v} {k:.0%}: moneyness label agrees with intrinsic value", True, ok, kind="bool"))
    return [item("Figure 3-1", 5, "In-, at- and out-of-the-money calls and puts at five exercise prices around our last close", "data",
                 table=tbl(["Venue", "Exercise price", "Call", "Put", "Call intrinsic", "Put intrinsic"], rows),
                 text="A call is in the money below the current price and a put above it; the label is checked against the intrinsic value (all agree).", checks=ck)]


def _grid(S, lo=0.6, hi=1.4, n=801):
    return np.linspace(lo * S, hi * S, n)


def _slopes(f, g):
    y = f(g)
    s = np.diff(y) / np.diff(g)
    return float(s[0]), float(s[-1])


def pay(kind, q, Xs, prem=0.0):
    def f(S):
        S = np.asarray(S, float)
        if kind == "C":
            return q * (np.maximum(S - Xs, 0) - prem)
        if kind == "P":
            return q * (np.maximum(Xs - S, 0) - prem)
        return q * (S - Xs)
    return f


def ch4_basic():
    out = []
    specs = [("Figure 4-1", 6, "Parity graph: long a call", lambda Xs: [("long call", pay("C", 1, Xs))], (0.0, 1.0)),
             ("Figure 4-2", 6, "Parity graph: short a call", lambda Xs: [("short call", pay("C", -1, Xs))], (0.0, -1.0)),
             ("Figure 4-3", 7, "Parity graph: long a put", lambda Xs: [("long put", pay("P", 1, Xs))], (-1.0, 0.0)),
             ("Figure 4-4", 7, "Parity graph: short a put", lambda Xs: [("short put", pay("P", -1, Xs))], (1.0, 0.0)),
             ("Figure 4-6", 9, "Long and short an underlying contract", lambda Xs: [("long underlying", pay("U", 1, Xs)), ("short underlying", pay("U", -1, Xs))], None)]
    for id_, pg, title, legs, sl in specs:
        ck = []
        for v in VENUES:
            S = ctx(v)["S"]
            g = _grid(S)
            for name, f in legs(S):
                s0, s1 = _slopes(f, g)
                if sl is not None:
                    ck.append(chk(f"{v}: slope below the exercise price ({name})", sl[0], s0, 1e-9))
                    ck.append(chk(f"{v}: slope above the exercise price ({name})", sl[1], s1, 1e-9))
                else:
                    want = 1.0 if name.startswith("long") else -1.0
                    ck.append(chk(f"{v}: slope everywhere ({name})", want, s0, 1e-9))
                    ck.append(chk(f"{v}: slope everywhere ({name}), top end", want, s1, 1e-9))

        def draw(ax, v, legs=legs):
            S = ctx(v)["S"]
            g = _grid(S) / S * 100
            for name, f in legs(S):
                ax.plot(g, f(_grid(S)) / S * 100, label=name)
            ax.axhline(0, lw=.4, color="0.6"); ax.axvline(100, lw=.4, ls=":", color="0.6"); ax.set_xlabel("underlying price at expiration (% of last close)"); ax.set_ylabel("value at expiration (% of S)"); ax.legend(loc="best")
        p = panel(draw, f"nat_{fid(id_)}")
        out.append(item(id_, pg, title + " (exercise price = our last close; premium ignored, as in a parity graph)", "formula", fig=p, table=chk_table(ck, 4),
                        text="The picture of the book's parity graph on our two prices. The table reads the slopes off the computed curve and compares them with the book's slope marks (0 and +1 and so on).", checks=ck))
    # 4-5, 4-7, 4-8: the book's three combination pictures; identity tested on both venues
    combos = [("Figure 4-5", 8, "Long a call and long a put at the same exercise price, and the combined position", lambda S, Xs: pay("C", 1, Xs)(S) + pay("P", 1, Xs)(S), lambda S, Xs: np.abs(S - Xs), "= |S - X| (a V with slopes -1 and +1)"),
              ("Figure 4-7", 10, "Long two calls and short an underlying contract, and the combined position", lambda S, Xs: pay("C", 2, Xs)(S) + pay("U", -1, Xs)(S), lambda S, Xs: np.abs(S - Xs), "= |S - X| (the same V as Figure 4-5)"),
              ("Figure 4-8", 11, "Long a call and short a put at the same exercise price, and the combined position", lambda S, Xs: pay("C", 1, Xs)(S) + pay("P", -1, Xs)(S), lambda S, Xs: S - Xs, "= S - X (a long underlying position struck at X)")]
    for id_, pg, title, f, target, descr in combos:
        ck = []
        for v in VENUES:
            S0 = ctx(v)["S"]
            g = _grid(S0)
            ck.append(chk(f"{v}: largest gap between the combined curve and the book's stated result {descr}", 0.0, float(np.max(np.abs(f(g, S0) - target(g, S0)))), 1e-9))

        def draw(ax, v, f=f):
            S0 = ctx(v)["S"]
            g = _grid(S0)
            ax.plot(g / S0 * 100, f(g, S0) / S0 * 100, label="combined position")
            ax.axhline(0, lw=.4, color="0.6"); ax.set_xlabel("underlying price at expiration (% of last close)"); ax.set_ylabel("value at expiration (% of S)"); ax.legend()
        p = panel(draw, f"nat_{fid(id_)}")
        out.append(item(id_, pg, title, "formula", fig=p, table=chk_table(ck, 4), text=f"The book states the combined position {descr}; computed on a grid from 60% to 140% of each last close.", checks=ck))
    return out


POS49 = [(3, "C", 65), (2, "P", 65), (2, "C", 70), (-4, "P", 70), (-6, "C", 75), (3, "P", 75), (4, "C", 80), (-2, "P", 80), (-4, "U", 0)]
POS415 = [(3, "C", 65), (2, "P", 65), (2, "C", 70), (-4, "P", 70), (-5, "C", 75), (3, "P", 75), (3, "C", 80), (-2, "P", 80), (-4, "U", 0)]


def seg_slopes(pos, bps=(65, 70, 75, 80)):
    out = []
    pts = [bps[0] - 5] + [(a + b) / 2 for a, b in zip(bps[:-1], bps[1:])] + [bps[-1] + 5]

    def val(S):
        tot = 0.0
        for q, k, x in pos:
            tot += q * (max(S - x, 0) if k == "C" else max(x - S, 0) if k == "P" else S)
        return tot
    for s in pts:
        out.append(val(s + 0.5) - val(s - 0.5))
    return out


def ch4_tables():
    out = []
    rows = []
    for q, k, x in POS49:
        nm = f"{x} {'calls' if k == 'C' else 'puts'}" if k != "U" else "Underlying contracts"
        regs = []
        for lo, hi in ((0, 65), (65, 70), (70, 75), (75, 80), (80, 99)):
            m = (lo + hi) / 2
            regs.append(q * (1 if m > x else 0) if k == "C" else (-q * (1 if m < x else 0)) if k == "P" else q)
        rows.append([f"{q:+d} {nm}"] + [f"{v:+g}" if v else "0" for v in regs])
    tot = [sum(float(r[j]) for r in rows) for j in range(1, 6)]
    # per-region contract totals as the book prints them (-3, +2, 0, -3, -1 are slopes; totals are contracts)
    sl = seg_slopes(POS49)
    ck = [chk(f"Slope {n}", b, s, 1e-9) for n, b, s in zip(["below 65", "65-70", "70-75", "75-80", "above 80"], [-3, 2, 0, -3, -1], sl)]
    out.append(item("Figure 4-9", 12, "Sample parity graph: nine positions and the slope of the total in each region", "formula",
                    table=tbl(["Contract", "Below 65", "65-70", "70-75", "75-80", "Above 80"], rows + [["Total"] + [f"{t:+g}" if t else "0" for t in tot]]),
                    table2=chk_table(ck, 0), table2_title="Slopes: book prints -3, +2, 0, -3, -1", text="Each position's slope contribution in each region, summed. Book inputs; no market data enters a parity graph.", checks=ck))

    def pnl_be(legs, grid, tol=1e-9):
        f = lambda S: sum(q * (np.maximum(S - x, 0) - pr if k == "C" else np.maximum(x - S, 0) - pr if k == "P" else (S - pr)) for q, k, x, pr in legs)
        y = f(grid)
        be = []
        for i in range(len(grid) - 1):
            if y[i] == 0 or y[i] * y[i + 1] < 0:
                be.append(float(grid[i] - y[i] * (grid[i + 1] - grid[i]) / (y[i + 1] - y[i])) if y[i + 1] != y[i] else float(grid[i]))
        return f, be
    g = np.linspace(50, 150, 100001)
    # 4-10, 4-11, 4-12, 4-13
    f, be = pnl_be([(1, "C", 100, 3.50)], g)
    ck10 = [chk("Breakeven of long 100 call at 3.50", 103.50, be[0], 1e-3)]
    f, be = pnl_be([(-1, "P", 95, 2.25)], g)
    ck11 = [chk("Breakeven of short 95 put at 2.25", 92.75, be[0], 1e-3)]
    ck12 = []
    for k, pr, b in ((95, 6.25, 101.25), (100, 3.50, 103.50), (105, 1.75, 106.75)):
        _, be = pnl_be([(1, "C", k, pr)], g)
        ck12.append(chk(f"Breakeven of long {k} call at {pr}", b, be[0], 1e-3))
    ck13 = []
    for k, pr, b in ((95, 2.25, 92.75), (100, 4.50, 95.50), (105, 7.75, 97.25)):
        _, be = pnl_be([(1, "P", k, pr)], g)
        ck13.append(chk(f"Breakeven of long {k} put at {pr}", b, be[0], 1e-3))
    ck14 = []
    legs = [(1, "C", 95, 6.25), (-1, "C", 105, 1.75), (-2, "P", 105, 7.75), (-2, "U", 0, 98.0)]
    f, be = pnl_be(legs, g)
    ck14 += [chk("P&L at 95", -3.00, float(f(np.array([95.0]))[0]), 1e-9), chk("P&L at 105 (the peak)", 7.00, float(f(np.array([105.0]))[0]), 1e-9), chk("Lower breakeven", 98.00, be[0], 1e-3), chk("Upper breakeven", 108.50, be[1], 1e-3)]

    def own(venue, legs_rel):
        """Our-data analogue: the same trades with strikes at the same percentage of our last close as the book's strike is of its underlying, premiums from the book's own value formula (Figure 18-6), sigma and S ours, 3 months, r = 0."""
        c = ctx(venue)
        S, sig = c["S"], c["sig"]
        legs2 = []
        for q, k, xr, ref in legs_rel:
            Xs = xr * S
            pr = float(C.bs(S, Xs, X.T3M, 0.0, sig)[0 if k == "C" else 1]) if k != "U" else S
            legs2.append((q, k, Xs, pr))
        g2 = np.linspace(0.5 * S, 1.5 * S, 100001)
        f2, be2 = pnl_be(legs2, g2)
        return be2, legs2
    for id_, pg, title, legsb, cks, ref in (("Figure 4-10", 13, "Long a 100 call at a price of 3.50: expiration P&L", [(1, "C", 100, 3.50)], ck10, 100.0), ("Figure 4-11", 13, "Short a 95 put at a price of 2.25: expiration P&L", [(-1, "P", 95, 2.25)], ck11, 100.0),
                                           ("Figure 4-12", 14, "Long a 95 call at 6.25, a 100 call at 3.50, a 105 call at 1.75: three expiration P&L lines", None, ck12, 100.0),
                                           ("Figure 4-13", 14, "Long a 95 put at 2.25, a 100 put at 4.50, a 105 put at 7.75: three expiration P&L lines", None, ck13, 100.0),
                                           ("Figure 4-14", 15, "Sample profit and loss graph: +1 95 call, -1 105 call, -2 105 puts, -2 underlying at 98.00", legs, ck14, 98.0)):
        t2 = None
        if legsb is not None:
            rows = []
            for v in VENUES:
                rel = [(q, k, x / ref if k != "U" else 0, ref) for q, k, x, *_ in [(l[0], l[1], l[2]) for l in legsb]]
                be2, l2 = own(v, rel)
                rows.append([v, ", ".join(f"{q:+d} {k}{'' if k == 'U' else f' {x:,.2f}'} @ {pr:,.2f}" for q, k, x, pr in l2), ", ".join(f"{b:,.2f}" for b in be2)])
            t2 = tbl(["Venue", "Legs (strikes at the book's % of S, premium by Figure 18-6, 3 months, r = 0)", "Breakevens"], rows)
        out.append(item(id_, pg, title, "formula", table=chk_table(cks, 3), table2=t2, table2_title="Same trades on our data: strikes at the book's percentage of our last close", text="Book inputs: strikes and premiums as printed; breakevens found by solving the expiration P&L for zero.", checks=cks))
    sl = seg_slopes(POS415)
    ck = [chk(f"Slope {n} (book prints it)", b, s, 1e-9) for n, b, s in zip(["below 65", "65-70", "70-75", "75-80", "above 80"], [-3, 2, 0, -3, -1], sl)]
    out.append(item("Figure 4-15", 16, "Sample profit and loss graph (second parity example): slopes of the printed position", "formula", table=chk_table(ck, 0),
                    text="The figure prints the position (-4 underlying; +3/+2/-5/+3 calls and +2/-4/+3/-2 puts at 65/70/75/80) and its slopes. The per-option prices are not printed, so the P&L levels (-6.90, +3.10, -11.90, -13.40) and the breakevens (62.70, 68.45, 76.03) cannot be reproduced. Only the slopes can be.",
                    note="The printed position gives the slopes listed above; where one differs from the slope printed in the figure the row says DIFFERS (the figure's slope list is the same as Figure 4-9's, whose position has -6 75 calls and +4 80 calls).", checks=ck))
    return out


# ----------------------------------------------------------------------------------------------------------------- Chapters 5-6
def ch5():
    out = []
    out.append(item("Figure 5-1", 17, "Generating a theoretical value: the five inputs to a pricing model", "formula",
                    text="The model in Figure 18-6 (nat_core.bs) takes exactly these five inputs: exercise price, time to expiration, underlying price, interest rate, volatility. Each input's effect on the value is tested in Figure 7-1.", note="A box diagram of inputs; nothing numeric to run beyond Figure 7-1."))
    ck, rows = [], []
    for v in VENUES:
        c = ctx(v)
        S, sig = c["S"], c["sig"]
        g = C.greeks(S, S, X.T3M, 0.0, sig)
        for pos, nm, hedge, d in (("Buy call(s)", "Long", "Sell underlying", g["delta"][0]), ("Sell call(s)", "Short", "Buy underlying", -g["delta"][0]), ("Buy put(s)", "Short", "Buy underlying", g["delta"][1]), ("Sell put(s)", "Long", "Sell underlying", -g["delta"][1])):
            hedge_qty = -d
            net = d + hedge_qty
            exp_mkt = "Long" if d > 0 else "Short"
            rows.append([v, pos, nm, exp_mkt, hedge, f"{d:+.4f}", f"{hedge_qty:+.4f}", f"{net:+.1e}"])
            ck.append(chk(f"{v}: {pos}: market position (delta sign) = {nm}", nm, exp_mkt, kind="text"))
            ck.append(chk(f"{v}: {pos}: hedge side = {hedge.split()[0]}", hedge.split()[0], "Sell" if hedge_qty < 0 else "Buy", kind="text"))
    out.append(item("Figure 5-2", 17, "The four basic option positions, the market position each corresponds to, and the hedge", "data", table=chk_table(ck, 4), table2=tbl(["Venue", "Position", "Book: market", "Ours: market", "Book: hedge", "Delta", "Hedge qty", "Net delta"], rows),
                    table2_title="Delta of one at-the-money 3-month option (r = 0) and the hedge that offsets it", text="Each option position's delta sign gives the market position it behaves like; the hedge is the opposite trade in the underlying, and the net delta is zero.", checks=ck))
    return out


def ch6():
    out = []
    # 6-1 normal distribution: our standardised daily log returns against the normal curve
    def draw1(ax, v):
        c = ctx(v)
        z = ((c["ret"] - c["ret"].mean()) / c["ret"].std()).values
        ax.hist(z, bins=np.linspace(-6, 6, 61), density=True, color="0.7")
        xx = np.linspace(-6, 6, 400)
        ax.plot(xx, norm.pdf(xx), color="k", lw=1)
        ax.set_xlabel("daily log return, in standard deviations"); ax.set_ylabel("density")
    p = panel(draw1, "nat_Figure_6_1")
    rows = []
    for v in VENUES:
        c = ctx(v)
        z = ((c["ret"] - c["ret"].mean()) / c["ret"].std()).values
        rows.append([v, c["n"] - 1, fmt(np.mean(np.abs(z) < 1) * 100, 2), fmt(np.mean(np.abs(z) < 2) * 100, 2), fmt(np.mean(np.abs(z) < 3) * 100, 2), fmt(np.mean(np.abs(z) > 4) * 100, 3)])
    out.append(item("Figure 6-1", 18, "Normal distribution: the bell curve, shown against our standardised daily log returns", "data", fig=p,
                    table=tbl(["Venue", "Returns", "% within 1 sd", "% within 2 sd", "% within 3 sd", "% beyond 4 sd"], rows + [["normal curve", "", "68.27", "95.45", "99.73", "0.006"]]),
                    text="The book's figure is a pinball-board bell curve. Ours: every daily log return of the venue, standardised, against the normal density."))
    # 6-2 possible price distributions
    ck, rows = [], []

    def draw2(ax, v):
        c = ctx(v)
        S, sig = c["S"], c["sig"]
        xs = np.linspace(0.4 * S, 2.0 * S, 600)
        for m, ls in ((0.5, ":"), (1.0, "-"), (1.5, "--")):
            s = sig * m
            T = X.T3M
            pdf = norm.pdf((np.log(xs / S) + 0.5 * s * s * T) / (s * np.sqrt(T))) / (xs * s * np.sqrt(T))
            ax.plot(xs / S * 100, pdf * S, ls=ls, label=f"{m:g} x sigma")
        ax.axvline(100, lw=.4, color="0.6"); ax.set_xlabel("price at expiration (% of S), 3 months, r = 0"); ax.set_ylabel("density x S"); ax.legend()
    p = panel(draw2, "nat_Figure_6_2")
    for v in VENUES:
        c = ctx(v)
        S, sig = c["S"], c["sig"]
        vals = [float(C.bs(S, S, X.T3M, 0.0, sig * m)[0]) for m in (0.5, 1.0, 1.5)]
        rows.append([v] + [fmt(x, 2) for x in vals])
        ck.append(chk(f"{v}: call value rises with the width of the distribution (low < moderate < high)", True, bool(vals[0] < vals[1] < vals[2]), kind="bool"))
    out.append(item("Figure 6-2", 19, "Possible price distribution: low-, moderate- and high-volatility distributions around the same current price", "formula", fig=p,
                    table=chk_table(ck), table2=tbl(["Venue", "Call value, 0.5 x sigma", "1 x sigma", "1.5 x sigma"], rows), table2_title="At-the-money 3-month call value (Figure 18-6) for the three widths",
                    text="Lognormal densities of the price at expiration on our two prices at one half, one and one and a half times our measured volatility. The book's point is that a wider distribution is worth more to the owner of an at-the-money call.", checks=ck))
    # 6-3
    ck = []
    for k, bk in ((1, 68.3), (2, 95.4), (3, 99.7)):
        ck.append(chk(f"Share within {k} st. dev. of a normal curve (%)", bk, (2 * norm.cdf(k) - 1) * 100, 0.06))
    mean, sd = 7.5, 3.0
    for k, lo, hi in ((-2, 1.5, 0), (-1, 4.5, 0), (1, 10.5, 0), (2, 13.5, 0)):
        ck.append(chk(f"Mean {mean} {'+' if k > 0 else '-'} {abs(k)} x st. dev. {sd}", lo, mean + k * sd, 1e-9))
    rows = []
    for v in VENUES:
        c = ctx(v)
        z = ((c["ret"] - c["ret"].mean()) / c["ret"].std()).values
        rows.append([v, fmt(np.mean(np.abs(z) < 1) * 100, 2), fmt(np.mean(np.abs(z) < 2) * 100, 2), fmt(np.mean(np.abs(z) < 3) * 100, 2)])
    out.append(item("Figure 6-3", 20, "Pinball distribution with mean 7.50 and standard deviation 3.00: 68.3%, 95.4%, 99.7% within one, two, three standard deviations", "formula",
                    table=chk_table(ck, 2), table2=tbl(["Venue", "% within 1 sd", "% within 2 sd", "% within 3 sd"], rows), table2_title="The same three shares measured on our daily log returns (book: 68.3 / 95.4 / 99.7)", text="The percentages are the normal curve's; the band edges 1.50, 4.50, 10.50, 13.50 follow from mean 7.50 and st. dev. 3.00.", checks=ck))
    # 6-4 normal vs lognormal
    rows, ck = [], []

    def draw4(ax, v):
        c = ctx(v)
        s = c["close"]
        h = int(round(c["ann"] * X.T3M))
        lr = np.log(s).diff(h).dropna().values
        pr = np.exp(lr)
        ax.hist(pr, bins=60, density=True, color="0.75")
        ax.axvline(np.mean(pr), color="k", lw=1, label="mean"); ax.axvline(np.median(pr), color="0.4", ls="--", lw=1, label="median")
        ax.set_xlabel("price after 3 months / price now"); ax.legend()
    p = panel(draw4, "nat_Figure_6_4")
    for v in VENUES:
        c = ctx(v)
        h = int(round(c["ann"] * X.T3M))
        lr = np.log(c["close"]).diff(h).dropna().values
        pr = np.exp(lr)
        sk_p, sk_l = C.moments(pr)["skew"], C.moments(lr)["skew"]
        rows.append([v, h, fmt(sk_p, 3), fmt(sk_l, 3), fmt(np.mean(pr), 4), fmt(np.median(pr), 4)])
        ck.append(chk(f"{v}: mean of the 3-month price ratio is above its median (lognormal's long right tail)", True, bool(np.mean(pr) > np.median(pr)), kind="bool"))
        ck.append(chk(f"{v}: price ratio is more right-skewed than its log", True, bool(sk_p > sk_l), kind="bool"))
    out.append(item("Figure 6-4", 21, "Normal and lognormal distributions: the lognormal has a longer right tail and its mean sits to the right of its median", "data", fig=p,
                    table=tbl(["Venue", "Window (days)", "Skewness of price ratio", "Skewness of log ratio", "Mean ratio", "Median ratio"], rows), table2=chk_table(ck), table2_title="Checks of the book's two statements",
                    text="Overlapping 3-month windows over the whole history of each venue; price ratio against its log.", checks=ck))
    # 6-5 theoretical value and implied volatility
    tv = float(C.bs(98.50, 105, 0.25, 0.06, 0.25)[0])
    iv = C.implied_vol(3.60, 98.50, 105, 0.25, 0.06)
    ck = [chk("Theoretical value of the 105 call (S 98.50, r 6%, 3 months, vol 25%)", 2.94, tv, 0.005), chk("Implied volatility at a price of 3.60 (%)", 28.50, iv * 100, 0.005)]
    rows = []
    for v in VENUES:
        c = ctx(v)
        S, sig = c["S"], c["sig"]
        Xs = S * 105 / 98.50
        val = float(C.bs(S, Xs, 0.25, 0.06, sig)[0])
        back = C.implied_vol(val, S, Xs, 0.25, 0.06)
        rows.append([v, fmt(S, 2), fmt(Xs, 2), fmt(sig * 100, 2), fmt(val, 4), fmt(back * 100, 4)])
        ck.append(chk(f"{v}: implied volatility of our own theoretical value returns our sigma (%)", sig * 100, back * 100, 1e-6))
    out.append(item("Figure 6-5", 21, "Solving for the theoretical value (2.94) and for the implied volatility (28.50%) of a 105 call", "formula", table=chk_table(ck, 4),
                    table2=tbl(["Venue", "S", "X (105/98.50 of S)", "sigma %", "Theoretical value", "Implied vol back %"], rows), table2_title="The same call on our data (3 months, r = 6%, our sigma)",
                    text="Book inputs: stock 98.50, interest rate 6%, 3 months, volatility 25%, exercise price 105; and for the second case a price of 3.60. The implied volatility is found by searching for the volatility whose value equals the price.", checks=ck))
    return out


# ----------------------------------------------------------------------------------------------------------------- Appendix B
PIN = [0, 2, 2, 4, 5, 6, 9, 10, 11, 9, 7, 5, 3, 2, 1, 1]
B2P = [97.70, 99.50, 92.75, 95.85, 96.20, 102.45, 93.30, 91.15, 95.20, 102.80, 103.85]


def appendix_b():
    out = []
    x = np.repeat(np.arange(len(PIN)), PIN)
    m = C.moments(x)
    ck = [chk("Number of occurrences", 77, len(x), 0), chk("Total value", 577, x.sum(), 0), chk("Mean", 7.4935, m["mean"], 5e-5), chk("Sum of squared deviations x occurrences", 695.2440, float(np.sum((x - m["mean"]) ** 2)), 5e-3),
          chk("Population standard deviation", 3.0049, m["std_pop"], 5e-5), chk("Sample standard deviation", 3.0246, m["std_sample"], 5e-5)]
    mom = np.mean((x - 9.0) ** 2)
    out.append(item("Figure B-1", 161, "Mean and standard deviation of the pinball distribution in Figure 6-2 (77 occurrences over 16 troughs)", "formula", table=chk_table(ck, 4),
                    table2=tbl(["Trough", "Occurrences", "Total value", "Deviation", "Deviation squared", "Times occurrences"], [[i, PIN[i], i * PIN[i], fmt(i - m["mean"], 4), fmt((i - m["mean"]) ** 2, 4), fmt(PIN[i] * (i - m["mean"]) ** 2, 4)] for i in range(len(PIN))]),
                    table2_title="Our recomputation of the book's table from its occurrence counts", text="The counts per trough are those printed in the book's table (0, 2, 2, 4, 5, 6, 9, 10, 11, 9, 7, 5, 3, 2, 1, 1).", checks=ck))
    # skewness / kurtosis printed moments
    ck2 = [chk("Skewness = 1.1095 / 9.0291^1.5", 0.0409, 1.1095 / 9.0291 ** 1.5, 5e-5), chk("Kurtosis = 222.7640 / 9.0291^2 - 3", -0.2675, 222.7640 / 9.0291 ** 2 - 3, 5e-5)]
    rows = [["Book's printed moments m2, m3, m4", "9.0291, 1.1095, 222.7640", ""]]
    mm = C.moments(x)
    rows.append(["Moments of the Figure B-1 distribution (our computation)", f"{mm['m2']:.4f}, {mm['m3']:.4f}, {mm['m4']:.4f}", f"skew {mm['skew']:+.4f}, excess kurt {mm['kurt']-3:+.4f}"])
    out.append(item("Appendix B: moments, skewness and kurtosis", 160, "Skewness (+0.0409) and kurtosis (-0.2675) of the pinball distribution from m2, m3, m4", "formula", table=chk_table(ck2, 4), table2=tbl(["Item", "Moments", "Result"], rows),
                    table2_title="Printed moments against moments computed from the Figure B-1 counts",
                    text="The book prints m2 = 9.0291, m3 = 1.1095, m4 = 222.7640 and the two results. The printed values reproduce exactly from the printed moments; the table compares them with moments computed from the printed counts.",
                    note="If the computed moments differ from the printed ones, the book's printed moments are about a different distribution than Figure B-1's counts (the row above shows both).", checks=ck2))
    # B-2
    ck = [chk("Population st. dev., actual mean (%)", 37.62, C.vol_appendix_b(B2P, 7 / 365) * 100, 0.005), chk("Population st. dev., 0 mean (%)", 37.88, C.vol_appendix_b(B2P, 7 / 365, "zero") * 100, 0.005),
          chk("Estimated st. dev., actual mean (%)", 39.65, C.vol_appendix_b(B2P, 7 / 365, "actual", True) * 100, 0.005), chk("Estimated st. dev., 0 mean (%)", 39.93, C.vol_appendix_b(B2P, 7 / 365, "zero", True) * 100, 0.005)]
    rows = []
    for v in VENUES:
        c = ctx(v)
        p = c["close"].iloc[-11:].values
        dt = 1 / c["ann"]
        rows.append([v, ", ".join(f"{q:,.2f}" for q in p[:3]) + " ...", fmt(C.vol_appendix_b(p, dt) * 100, 2), fmt(C.vol_appendix_b(p, dt, "zero") * 100, 2), fmt(C.vol_appendix_b(p, dt, "actual", True) * 100, 2), fmt(C.vol_appendix_b(p, dt, "zero", True) * 100, 2)])
    out.append(item("Figure B-2", 162, "Volatility calculation for the stock option example in Figure 8-1: four estimates (37.62, 37.88, 39.65, 39.93%)", "formula", table=chk_table(ck, 2),
                    table2=tbl(["Venue", "Last 11 daily closes", "Pop. actual mean %", "Pop. 0 mean %", "Est. actual mean %", "Est. 0 mean %"], rows), table2_title="The same four estimates on our last 11 daily closes (one-day interval)",
                    text="Book inputs: eleven weekly prices (97.70 ... 103.85), interval 7/365 year. Volatility = sqrt(sum of squared deviations / n) / sqrt(t); the estimated (sample) form divides by n - 1; the zero-mean form drops the mean.", checks=ck))
    # rate-of-return formulas and normal-distribution formulas: text with check of identities
    c1 = C.rate_of_return(100.0, 0.05, 2.0, 4)
    ck = [chk("PV -> FV -> PV (simple, r 5%, 2 years)", 100.0, c1["simple"] / (1 + 0.05 * 2), 1e-9), chk("PV -> FV -> PV (continuous)", 100.0, c1["continuous"] * np.exp(-0.05 * 2), 1e-9),
          chk("t = ln(FV/PV)/r returns the time (continuous)", 2.0, np.log(c1["continuous"] / 100) / 0.05, 1e-12), chk("t = [ln(FV/PV)/ln(1 + r/n)]/n returns the time (compound)", 2.0, (np.log(c1["compound"] / 100) / np.log(1 + 0.05 / 4)) / 4, 1e-12),
          chk("r = (FV/PV - 1)/t returns the rate (simple)", 0.05, (c1["simple"] / 100 - 1) / 2, 1e-12)]
    out.append(item("Appendix B: rate-of-return calculations", 158, "Simple, compound and continuous interest: future value, present value, rate and time", "formula", table=chk_table(ck, 6),
                    text="The book prints formulas only (no numbers). The formulas are used as printed and the inverse formulas are tested to return the starting values (100 at 5% for 2 years, quarterly compounding).", checks=ck))
    # sigma range formulas applied to our data
    rows, ck = [], []
    for v in VENUES:
        c = ctx(v)
        S, sig, T = c["S"], c["sig"], X.T3M
        up, dn = S * np.exp(sig * np.sqrt(T)), S * np.exp(-sig * np.sqrt(T))
        h = int(round(c["ann"] * T))
        lr = np.log(c["close"]).diff(h).dropna()
        inside = float(np.mean(np.abs(lr.values) <= sig * np.sqrt(T)) * 100)
        rows.append([v, fmt(S, 2), fmt(dn, 2), fmt(up, 2), fmt(inside, 1), fmt(float(np.log(S * 1.1 / S) / (sig * np.sqrt(T))), 3)])
        ck.append(chk(f"{v}: number of st. devs to a price 10% above = ln(X/F)/(sigma sqrt t) consistent with F e^(n sigma sqrt t) = X", float(S * 1.1), float(S * np.exp(np.log(1.1 / 1) / (sig * np.sqrt(T)) * sig * np.sqrt(T))), 1e-6))
    out.append(item("Appendix B: volatility and price ranges", 159, "Price range of n standard deviations, F e^(+/- n sigma sqrt t), and the number of standard deviations to an exercise price", "data", table=tbl(["Venue", "S", "-1 sd price (3 months)", "+1 sd price", "% of 3-month windows within 1 sd", "St. devs to a price 10% above"], rows),
                    table2=chk_table(ck, 4), table2_title="Identity check", text="One-sd range uses our sigma and 3 months; the share of historical overlapping 3-month windows that stayed within the range is measured on the full history of the venue.", checks=ck))
    return out


def all_items():
    out = []
    for f in (ch1, ch3, ch4_basic, ch4_tables, ch5, ch6, appendix_b):
        out += f()
    return out
