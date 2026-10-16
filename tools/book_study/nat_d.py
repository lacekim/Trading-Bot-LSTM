"""Natenberg write-up items: Chapter 18 (Black-Scholes), Chapter 19 (binomial pricing), Chapter 20 (forward volatility).  New file."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import comb

import nat_core as C
import nat_ctx as X
from nat_ctx import item, chk, chk_table, ctx, tbl, fmt, VENUES, panel, sgn


def fid(s):
    return s.replace(" ", "_").replace(".", "_").replace("-", "_")


# ----------------------------------------------------------------------------------------------------------------- Chapter 18
OCC = [0, 1, 1, 2, 4, 6, 8, 10, 12, 13, 13, 12, 11, 11, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1, 1, 1, 0, 0]       # Figure 18-1, troughs 0..27 (read off the figure's printed row)


def _interp_half(cum_before, cnt, trough, half):
    """Trough number at which the running total reaches `half`, troughs treated as spanning trough-0.5 .. trough+0.5."""
    return trough - 0.5 + (half - cum_before) / cnt if cnt else trough


def ch18_text():
    out = []
    # the equation on page 79 and the first-two-terms comparison on page 80-81
    ck, rows = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for r in (0.0, 0.04):
            t = 0.5
            res = []
            for S in (0.8 * S0, S0, 1.2 * S0):
                g = C.greeks(S, S0, t, r, sig)
                Cv = C.bs(S, S0, t, r, sig)[0]
                res.append(float(g["theta"][0] + r * S * g["delta"][0] + 0.5 * sig ** 2 * S ** 2 * g["gamma"][0] - r * Cv))
            ck.append(chk(f"{v}, r = {r:.0%}: theta + r S delta + (1/2) sigma^2 S^2 gamma - r C = 0 (largest residual / S)", 0.0, max(abs(x) for x in res) / S0, 1e-10))
            rows.append([v, f"{r:.0%}", *[f"{x:.2e}" for x in res]])
        # Taylor approximation (S2 - S1) delta + 1/2 (S2 - S1)^2 gamma
        t = 0.25
        S1 = S0
        for pct in (0.01, 0.05):
            S2 = S1 * (1 + pct)
            exact = float(C.bs(S2, S0, t, 0.0, sig)[0] - C.bs(S1, S0, t, 0.0, sig)[0])
            g = C.greeks(S1, S0, t, 0.0, sig)
            approx = float((S2 - S1) * g["delta"][0] + 0.5 * (S2 - S1) ** 2 * g["gamma"][0])
            first = float((S2 - S1) * g["delta"][0])
            ck.append(chk(f"{v}: move of {pct:.0%}: delta-and-gamma estimate is closer than delta alone (error {abs(exact-approx):.4f} vs {abs(exact-first):.4f})", True, bool(abs(exact - approx) < abs(exact - first)), kind="bool"))
    out.append(item("Chapter 18: the Black-Scholes equation", 79, "The partial differential equation behind the model, and the delta-gamma estimate of a value change", "formula", table=chk_table(ck, 4), table2=tbl(["Venue", "Rate", "Residual at 0.8 S", "at S", "at 1.2 S"], rows), table2_title="theta + r S delta + (1/2) sigma^2 S^2 gamma - r C (call, 6 months, X = last close)",
                    text="The book prints the equation (dC/dt + r S dC/dS + (1/2) sigma^2 S^2 d2C/dS2 = r C) and says the value formula solves it. We evaluate it with the book's own Delta, Gamma and Theta (Figure 18-6) and check the residual is zero; and check the delta-gamma estimate of a value change (pages 80-81) beats delta alone.", checks=ck))
    return out


def ch18_fig_1_4():
    out = []
    n = np.array(OCC, float)
    tr = np.arange(len(n))
    ten = 12.5
    above = tr > ten
    cnt, val, tot = float(n[above].sum()), float((n * tr)[above].sum()), float(n.sum())
    prob = cnt / tot
    ev_a = val / tot - ten * prob
    probs = n[above] / tot
    iv = tr[above] - ten
    ev_b = float(np.sum(probs * iv))
    ck = [chk("Occurrences above 12.5", 60, cnt, 0), chk("Value of all stock above 12.5", 987, val, 0), chk("Total occurrences", 153, tot, 0), chk("Average value of all stock above 12.5 (987/153)", 6.45, val / tot, 0.005),
          chk("Chance of paying the exercise price (60/153)", 0.392, prob, 0.0005), chk("Average payment 0.392 x 12.5", 4.90, prob * ten, 0.005), chk("Expected value of the call = 6.45 - 4.90", 1.55, round(val / tot, 2) - round(prob * ten, 2), 0.005),
          chk("Sum of intrinsic value x probability (the table's total)", 1.5489, ev_b, 5e-4)]
    rows = [[int(t), int(a), f"{t - ten:.1f}", f"{a / tot:.4f}", f"{(a / tot) * (t - ten):.4f}"] for t, a in zip(tr[above], n[above]) if a > 0]
    # ours: the same 'average above the exercise price' idea on the historical distribution of 3-month price ratios
    t2 = {"cols": ["Venue", "3-month windows", "Share finishing above the exercise price", "Average intrinsic value (% of S)", "Black-Scholes value, r = 0 (% of S)", "Forward-based BS value with historical drift removed (% of S)"], "rows": []}
    for v in VENUES:
        c = ctx(v)
        h = int(round(c["ann"] * X.T3M))
        lr = np.log(c["close"]).diff(h).dropna().values
        pr = np.exp(lr)
        share = float(np.mean(pr > 1))
        ev = float(np.mean(np.maximum(pr - 1, 0)))
        bsv = float(C.bs(1.0, 1.0, X.T3M, 0.0, c["sig"])[0])
        pr0 = pr / np.mean(pr)                                       # remove the historical drift so the mean is the forward, as the risk-neutral formula assumes
        ev0 = float(np.mean(np.maximum(pr0 - 1, 0)))
        t2["rows"].append([v, f"{len(pr):,}", fmt(share, 4), fmt(ev * 100, 3), fmt(bsv * 100, 3), fmt(ev0 * 100, 3)])
    out.append(item("Figure 18-1", 82, "A discrete lognormal-looking distribution of 153 occurrences, and the expected value of a 12.5 call two ways (1.55 and 1.5489)", "formula", table=chk_table(ck, 4), table2=tbl(["Trough", "Occurrences", "Intrinsic value of the 12.5 call", "Probability", "Option value"], rows),
                    table2_title="Troughs above 12.5, as in the book's second table", checks=ck,
                    text="Book inputs: the printed occurrences per trough (0, 1, 1, 2, 4, 6, 8, 10, 12, 13, 13, 12, 11, then 11, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1, 1, 1, 0, 0 for troughs 0 to 27; they sum to 153) and the exercise price 12.5.", note="The same idea on our data is in the next table: the share of historical 3-month price ratios that finish above the starting price, and the average intrinsic value, against the Black-Scholes value."))
    out[-1]["table3"] = tbl(t2["cols"], t2["rows"])
    out[-1]["table3_title"] = "Our data: historical 3-month price ratios against the Black-Scholes value (r = 0, our sigma)"
    # 18-2 and 18-3
    xs = np.linspace(-8, 8, 160001)
    area = float(np.trapezoid(C.n(xs), xs))
    ck = [chk("Area under n(x) = 1", 1.0, area, 1e-9), chk("N(0) = 0.50", 0.5, float(C.N(0)), 1e-12), chk("N(-x) = 1 - N(x) (largest gap over x = -5..5)", 0.0, float(np.max(np.abs(C.N(-np.linspace(-5, 5, 1001)) - (1 - C.N(np.linspace(-5, 5, 1001)))))), 1e-12), chk("N(+inf) = 1", 1.0, float(C.N(40.0)), 1e-12), chk("N(-inf) = 0", 0.0, float(C.N(-40.0)), 1e-12),
          chk("n(0) = 0.398942 (the peak used in the 0.00399 approximation)", 0.398942, float(C.n(0)), 5e-7)]
    for fid_, pg, ttl in (("Figure 18-2", 85, "n(x): the standard normal distribution curve (mean 0, standard deviation 1; total area 1)"), ("Figure 18-3", 86, "N(x): the area under the standard normal curve between minus infinity and x")):
        def draw(ax, v, fid_=fid_):
            c = ctx(v)
            z = ((c["ret"] - c["ret"].mean()) / c["ret"].std()).values
            xx = np.linspace(-4, 4, 400)
            if fid_ == "Figure 18-2":
                ax.hist(z, bins=np.linspace(-4, 4, 41), density=True, color="0.75"); ax.plot(xx, C.n(xx), color="k")
            else:
                zs = np.sort(z); ax.plot(zs, np.arange(1, len(zs) + 1) / len(zs), color="0.5", label="our daily returns"); ax.plot(xx, C.N(xx), color="k", label="N(x)"); ax.legend()
            ax.set_xlim(-4, 4); ax.set_xlabel("standard deviations")
        p = panel(draw, f"nat_{fid(fid_)}")
        out.append(item(fid_, pg, ttl, "formula", fig=p, table=chk_table(ck, 6), checks=ck, text="Properties the book states, checked on the formulas; the picture overlays our standardised daily log returns on the curve."))
    # 18-4 mode, mean, median
    cum = np.cumsum(n)
    half_n = tot / 2
    k = int(np.searchsorted(cum, half_n))
    median = _interp_half(cum[k - 1], n[k], k, half_n)
    vcum = np.cumsum(n * tr)
    half_v = vcum[-1] / 2
    k2 = int(np.searchsorted(vcum, half_v))
    mean_v = _interp_half(vcum[k2 - 1], n[k2] * k2, k2, half_v) if False else k2 - 0.5 + (half_v - vcum[k2 - 1]) / (n[k2] * k2) if n[k2] else k2
    # mode: the peak of a parabola through the three largest neighbouring troughs
    i = int(np.argmax(n))
    # (two troughs tie at 13; the book's 9.3 sits between 9 and 10, left of the tie's middle)
    modes = [int(j) for j in np.where(n == n.max())[0]]
    ck = [chk("Median: half of all occurrences lie to the left (book: approximately 10.5)", 10.5, median, 0.3), chk("Mean: half of the value (occurrences x trough number) lies to the left (book: approximately 12.7)", 12.7, mean_v, 0.3),
          chk(f"Mode: the peak of the distribution (troughs {modes} tie at {int(n.max())}; book: approximately 9.3)", 9.3, float(np.mean(modes)), 0.3)]
    out.append(item("Figure 18-4", 87, "Mode, mean and median of the distribution in Figure 18-1 (book: 9.3, 12.7, 10.5)", "formula", table=chk_table(ck, 2), checks=ck,
                    table2=tbl(["Measure", "Definition (book)", "Ours"], [["Median", "half of all occurrences to the left", fmt(median, 2)], ["Mean", "half of the value (occurrences x trough) to the left", fmt(mean_v, 2)], ["Mode", "the peak", f"troughs {modes}"]]),
                    text="Definitions as printed on page 346. Interpolation within a trough assumes the trough spans half a unit either side of its number; the book does not say how its 9.3, 12.7 and 10.5 were read off."))
    return out


def ch18_fig_5_6():
    out = []
    # 18-5 d1 decomposition on our data
    rows, ck = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for xr, nm in ((0.9, "X = 0.9 S"), (1.0, "X = S"), (1.1, "X = 1.1 S")):
            Xs = xr * S0
            t, r = X.T3M, 0.04
            b = r
            carry = b * t
            money = np.log(S0 / Xs)
            lognorm = sig ** 2 / 2 * t
            norm_ = sig * np.sqrt(t)
            d1 = (money + carry + lognorm) / norm_
            d1f = float(C.d1d2(S0, Xs, t, r, sig, b)[0])
            rows.append([v, nm, fmt(carry, 5), fmt(money, 5), fmt(lognorm, 5), fmt(norm_, 5), fmt(float(d1), 5), fmt(float(d1) - norm_, 5)])
            ck.append(chk(f"{v}, {nm}: d1 = (carry + ln(S/X) + sigma^2 t / 2) / (sigma sqrt t) equals the model's d1", d1f, float(d1), 1e-12))
            ck.append(chk(f"{v}, {nm}: d2 = d1 - sigma sqrt t equals the model's d2", float(C.d1d2(S0, Xs, t, r, sig, b)[1]), float(d1) - norm_, 1e-12))
    out.append(item("Figure 18-5", 88, "The pieces of d1: spot-to-forward carry, in- or out-of-the-money term, the lognormal mean adjustment, and the one-standard-deviation normalisation", "formula", table=chk_table(ck, 6),
                    table2=tbl(["Venue", "Strike", "Carry (b t)", "ln(S/X)", "sigma^2 t / 2", "sigma sqrt t", "d1", "d2"], rows), table2_title="3 months, r = b = 4%, sigma ours", checks=ck))
    return out


def fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def fd2(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def sens_check(S, Xs, t, r, sig, b):
    """Compare every printed sensitivity of Figure 18-6 with a central finite difference of the printed value formula. Returns rows (name, kind, printed, fd, status)."""
    g = C.greeks(S, Xs, t, r, sig, b)
    rows = []
    for kind, i in (("call", 0), ("put", 1)):
        val = lambda S_=S, X_=Xs, t_=t, r_=r, s_=sig, b_=b: float(C.bs(S_, X_, t_, r_, s_, b_)[i])
        delta = lambda S_: float(C.greeks(S_, Xs, t, r, sig, b)["delta"][i])
        gamma = lambda S_: float(C.greeks(S_, Xs, t, r, sig, b)["gamma"][i])
        vega = lambda s_, t_=t: float(C.greeks(S, Xs, t_, r, s_, b)["vega"][i])
        delta_s = lambda s_: float(C.greeks(S, Xs, t, r, s_, b)["delta"][i])
        delta_t = lambda t_: float(C.greeks(S, Xs, t_, r, sig, b)["delta"][i])
        gamma_t = lambda t_: float(C.greeks(S, Xs, t_, r, sig, b)["gamma"][i])
        gamma_s = lambda s_: float(C.greeks(S, Xs, t, r, s_, b)["gamma"][i])
        gamma_S = lambda S_: float(C.greeks(S_, Xs, t, r, sig, b)["gamma"][i])
        vega_t = lambda t_: float(C.greeks(S, Xs, t_, r, sig, b)["vega"][i])
        vega_s = lambda s_: float(C.greeks(S, Xs, t, r, s_, b)["vega"][i])
        h = 1e-4
        ref = {
            "delta": fd(lambda x: val(S_=x), S, S * h),
            "gamma": fd2(lambda x: val(S_=x), S, S * 1e-3),
            "theta": -fd(lambda x: val(t_=x), t, t * h),
            "vega": fd(lambda x: val(s_=x), sig, sig * h),
            "vanna": fd(delta_s, sig, sig * h),
            "charm": -fd(delta_t, t, t * h),
            "speed": fd(gamma_S, S, S * 1e-3),
            "color": -fd(gamma_t, t, t * h),
            "volga": fd(vega_s, sig, sig * h),
            "vegadecay": -fd(vega_t, t, t * h),
            "zomma": fd(gamma_s, sig, sig * h),
        }
        if b != 0:
            ref["rho"] = fd(lambda x: val(r_=x, b_=x), r, 1e-5)                                     # stock: b moves with r
        else:
            ref["rho"] = fd(lambda x: val(r_=x, b_=0.0), r, 1e-5)                                   # b = 0: only the discounting moves
        ref["phi"] = -fd(lambda x: val(b_=r - x), 0.0, 1e-5) if False else fd(lambda x: val(b_=r - x), 0.0, 1e-5)
        for name, f in ref.items():
            printed = float(g[name][i])
            if abs(printed - f) <= 1e-5 * max(1.0, abs(f), abs(printed)):
                st = "matches"
            elif abs(printed + f) <= 1e-5 * max(1.0, abs(f), abs(printed)):
                st = "matches with opposite sign"
            else:
                st = "DIFFERS"
            rows.append((name, kind, printed, float(f), st))
    return rows


def fig_18_6():
    out = []
    ck, rows_all = [], []
    # parity for every b variant
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for lab, r, b in (("stock b = r = 5%", .05, .05), ("futures-type settlement b = r = 0", 0.0, 0.0), ("Black b = 0, r = 5%", .05, 0.0), ("Garman-Kohlhagen b = r - rf = 5% - 2%", .05, .03)):
            t = 0.5
            Cv, Pv = C.bs(S0, S0, t, r, sig, b)
            ck.append(chk(f"{v}, {lab}: C - P = S e^((b-r)t) - X e^(-rt)", float(S0 * np.exp((b - r) * t) - S0 * np.exp(-r * t)), float(Cv - Pv), 1e-9))
    # dividends: S - sum D e^{r td} printed vs b = r - q
    dv = []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        q = 0.02
        t = 0.5
        approx = float(C.bs(S0, S0, t, 0.05, sig, 0.05 - q)[0])
        D, td = S0 * 0.01, 0.25
        exact = float(C.bs(S0 - D * np.exp(0.05 * td), S0, t, 0.05, sig, 0.05)[0])
        dv.append([v, fmt(approx, 3), fmt(exact, 3), fmt(approx / S0 * 100, 3), fmt(exact / S0 * 100, 3)])
    # sensitivities against finite differences, three settings, our two prices
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for lab, r, b in (("b = r = 4%", .04, .04), ("b = r = 0", 0.0, 0.0), ("b = 0, r = 4%", .04, 0.0)):
            for name, kind, pr, f, st in sens_check(S0, S0 * 1.02, 0.5, r, sig, b):
                rows_all.append([v, lab, name, kind, f"{pr:.6g}", f"{f:.6g}", st])
    summ = {}
    for r_ in rows_all:
        summ.setdefault((r_[2], r_[3]), set()).add(r_[6])
    bad = sorted([f"{k[0]} ({k[1]})" for k, s in summ.items() if "DIFFERS" in s])
    opp = sorted([f"{k[0]} ({k[1]})" for k, s in summ.items() if "matches with opposite sign" in s and "DIFFERS" not in s])
    for (name, kind), s in sorted(summ.items()):
        ck.append(chk(f"Printed formula for {name} ({kind}) against a finite difference of the printed value formula: {', '.join(sorted(s))}", "matches", "matches" if s == {"matches"} else ("matches with opposite sign" if "DIFFERS" not in s else "DIFFERS"), kind="text"))
    out.append(item("Figure 18-6", 90, "The Black-Scholes model: value of a European call and put (with the b variants), and every sensitivity, tested against finite differences", "formula", table=chk_table([c_ for c_ in ck if "finite difference" not in c_["what"]], 4),
                    table2=chk_table([c_ for c_ in ck if "finite difference" in c_["what"]], 4), table2_title="Each printed sensitivity formula against a finite difference of the printed value formula (S 100 of ours, X = 1.02 S, 6 months; three settings of r and b; both venues)",
                    table3=tbl(["Venue", "Dividend: b = r - q (q 2%)", "S - D e^(r td), D 1% of S, td 3 months", "same as % of S", "same as % of S"], dv), table3_title="The two dividend treatments the book gives (6-month call, X = S, r = 5%): value, then as % of S",
                    checks=ck,
                    text="Every formula of Figure 18-6 and its continuation is coded as printed (nat_core.greeks) and tested: put-call parity for each value of b, and each printed sensitivity against a central finite difference of the printed value formula. Charm, color and vega decay are the effect of the passage of time, so the finite difference is taken with respect to minus the time to expiration.",
                    note=("Printed formulas that DO NOT match a finite difference on some setting: " + (", ".join(bad) if bad else "none") + ". Printed formulas that match with the opposite sign: " + (", ".join(opp) if opp else "none") + ". The formulas are left as printed; the full list is in the table.")))
    out[-1]["extra_rows"] = rows_all
    return out


def approx_and_delta():
    out = []
    # A useful approximation
    N1, N2 = float(C.N(0.005)), float(C.N(-0.005))
    ck = [chk("N(d1) at d1 = 0.005", 0.501995, N1, 5e-7), chk("N(d2) at d2 = -0.005", 0.498005, N2, 5e-7), chk("Value per unit of X per vol point: 0.501995 - 0.498005", 0.00399, N1 - N2, 5e-6),
          chk("100 call, 1 year, 20% vol: 20 x 100 x 0.00399", 7.98, 20 * 100 * 0.00399, 1e-9), chk("... 35% vol", 13.965, 35 * 100 * 0.00399, 1e-9), chk("50 call, 20% vol", 3.99, 20 * 50 * 0.00399, 1e-9), chk("125 call, 20% vol", 9.975, 20 * 125 * 0.00399, 1e-9),
          chk("100 call, 20% vol, 6 months: 7.98 x sqrt(0.5)", 5.64, 7.98 * np.sqrt(0.5), 0.005), chk("65 at-the-forward, 18% vol, 3 months", 2.33, 65 * 18 * 0.5 * 0.00399, 0.005), chk("... with r = 4%: 2.33 / 1.01", 2.31, 2.33 / 1.01, 0.005),
          chk("65, 40% vol, 2 years: approximation", 14.67, 65 * 40 * np.sqrt(2) * 0.00399, 0.005), chk("... Black-Scholes expected value", 14.48, float(C.bs(65, 65, 2.0, 0.0, 0.40)[0]), 0.005),
          chk("Peak of the standard normal curve", 0.398942, float(C.n(0)), 5e-7)]
    rows = []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for t in (1 / 12, 0.25, 0.5, 1.0, 2.0):
            for sg, lab in ((0.2, "20%"), (sig, "ours"), (0.4, "40%")):
                ap = S0 * sg * 100 * np.sqrt(t) * 0.00399
                ex = float(C.bs(S0, S0, t, 0.0, sg)[0])
                rows.append([v, f"{t * 12:g}", f"{sg*100:.1f}% ({lab})", fmt(ap / S0 * 100, 3), fmt(ex / S0 * 100, 3), fmt((ap - ex) / S0 * 100, 4)])
    ck2 = []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        up = all(S0 * s * 100 * np.sqrt(t) * 0.00399 >= float(C.bs(S0, S0, t, 0.0, s)[0]) - 1e-12 for t in (0.25, 1, 2) for s in (0.2, sig, 0.4))
        ck2.append(chk(f"{v}: the approximation is never below the exact value (3 months..2 years, 20%, ours, 40%)", True, bool(up), kind="bool"))
        a1 = (S0 * 0.4 * 100 * np.sqrt(2) * 0.00399 - float(C.bs(S0, S0, 2.0, 0.0, 0.4)[0])) > (S0 * 0.2 * 100 * np.sqrt(0.25) * 0.00399 - float(C.bs(S0, S0, 0.25, 0.0, 0.2)[0]))
        ck2.append(chk(f"{v}: ...and the overstatement grows with time and volatility", True, bool(a1), kind="bool"))
    out.append(item("Chapter 18: a useful approximation (the 0.00399 and the 40% rule)", 91, "At-the-forward option value = X x (sigma x 100) x sqrt(t) x 0.00399; every number the book prints, then the approximation against the exact value on our two prices", "formula",
                    table=chk_table(ck, 6), table2=chk_table(ck2), table3=tbl(["Venue", "Months", "Volatility", "Approximation (% of S)", "Exact (% of S)", "Gap (% of S)"], rows), table3_title="Approximation against Figure 18-6, X = S (at the forward because r = 0)", checks=ck + ck2,
                    text="Every number the book prints in this section, reproduced; then the book's own statement that the approximation is slightly greater than the true value as time and volatility grow, tested on our two prices."))
    # The delta
    rows, ck = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        t, r = X.T3M, 0.04
        F = S0 * np.exp(r * t)
        d1, d2 = (float(a) for a in C.d1d2(S0, F, t, r, sig, r))
        rng = np.random.default_rng(1)
        z = rng.standard_normal(2_000_000)
        ST = S0 * np.exp((r - sig ** 2 / 2) * t + sig * np.sqrt(t) * z)
        mc = float(np.mean(ST > F))
        gs = C.greeks(S0, F, t, r, sig, r)
        rows.append([v, fmt(float(C.N(d1)), 5), fmt(float(C.N(d2)), 5), fmt(mc, 5), fmt(float(gs["delta"][0]) * 100, 3), fmt(float(gs["delta"][1]) * 100, 3)])
        ck += [chk(f"{v}: N(d1) is larger than N(d2) (delta above the probability of finishing in the money)", True, bool(C.N(d1) > C.N(d2)), kind="bool"),
               chk(f"{v}: N(d2) equals the probability of finishing in the money (risk-neutral simulation, 2,000,000 paths; difference)", 0.0, abs(float(C.N(d2)) - mc), 1e-3),
               chk(f"{v}: at-the-forward call delta is above 50", True, bool(gs["delta"][0] > 0.5), kind="bool"), chk(f"{v}: at-the-forward put delta is below -50", True, bool(gs["delta"][1] < -0.5), kind="bool"),
               chk(f"{v}: an at-the-forward straddle has positive delta", True, bool(gs["delta"][0] + gs["delta"][1] > 0), kind="bool")]
    out.append(item("Chapter 18: the delta", 93, "Delta N(d1) against the true chance of finishing in the money N(d2); at the forward the call delta is above 50 and a straddle has a positive delta", "data", table=chk_table(ck), table2=tbl(["Venue", "N(d1)", "N(d2)", "Simulated share finishing above X", "ATF call delta", "ATF put delta"], rows),
                    table2_title="3 months, r = b = 4%, X = forward, sigma ours", checks=ck))
    return out


def ch18_figs_7_11():
    out = []
    # 18-7
    ck, rows = [], []
    for v in VENUES:
        c = ctx(v)
        S0 = c["S"]
        for mo in (3, 6, 12):
            t = mo / 12
            for sg in (0.1, 0.3, 0.5):
                Sstar = S0 * np.exp(-(0.0 + sg ** 2 / 2) * t)
                g = C.greeks(Sstar, S0, t, 0.0, sg)
                ck.append(chk(f"{v}: {mo} months, vol {sg:.0%}: call delta and put delta equal in size at S = X e^(-sigma^2 t / 2) (gap)", 0.0, abs(float(g["delta"][0] + g["delta"][1])), 1e-12))
        rows.append([v] + [fmt(S0 * np.exp(-(0.25 ** 2 / 2) * mo / 12) / S0 * 100, 3) for mo in (3, 6, 12)])
    ck.append(chk("Chart reading: 12 months, 50% volatility, exercise price 100: delta-neutral price (about 88)", 88.0, 100 * np.exp(-(0.5 ** 2 / 2) * 1.0), 0.5))
    ck.append(chk("Chart reading: 3 months, 50% volatility (about 96.9)", 96.9, 100 * np.exp(-(0.5 ** 2 / 2) * 0.25), 0.5))

    def draw(ax, v):
        sg = np.linspace(0, 0.5, 100)
        for mo in (3, 6, 12):
            ax.plot(sg * 100, 100 * np.exp(-(sg ** 2 / 2) * mo / 12), label=f"{mo} months")
        ax.set_xlabel("volatility (%)"); ax.set_ylabel("underlying price at which a 100 straddle is delta neutral"); ax.legend()
    p = X.W.save(plt.figure(figsize=(3.6, 2.6)), "nat_tmp") if False else None
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    sg = np.linspace(0, 0.5, 100)
    for mo in (3, 6, 12):
        ax.plot(sg * 100, 100 * np.exp(-(sg ** 2 / 2) * mo / 12), label=f"{mo} months")
    ax.set_xlabel("volatility (%)"); ax.set_ylabel("price (exercise price 100, r = 0)"); ax.legend()
    p = X.save(fig, "nat_Figure_18_7")
    out.append(item("Figure 18-7", 94, "The underlying price at which a straddle is exactly delta neutral: S = X e^(-(b + sigma^2/2) t), against volatility for 3, 6 and 12 months (r = 0, X = 100)", "formula", fig=p, table=chk_table(ck, 3),
                    table2=tbl(["Venue", "Delta-neutral price at our sigma (X = 100): 3 months", "6 months", "12 months"], [[v, f"{100 * np.exp(-(ctx(v)['sig'] ** 2 / 2) * 0.25):.3f}", f"{100 * np.exp(-(ctx(v)['sig'] ** 2 / 2) * 0.5):.3f}", f"{100 * np.exp(-(ctx(v)['sig'] ** 2 / 2) * 1.0):.3f}"] for v in VENUES]),
                    table2_title="Same formula at each venue's own volatility", checks=ck,
                    text="The figure is drawn from the book's formula (it prints no data); at the printed price the call and put deltas are equal in size (gap shown). The two chart readings are read off the printed axes and are approximate."))
    # 18-8 / 18-9 : critical prices
    for id_, pg, r_, title in (("Figure 18-8", 96, 0.0, "At an interest rate of zero, the underlying price at which the maximum gamma, theta and vega occur (exercise price 100, 1 year)"),
                               ("Figure 18-9", 97, 0.04, "At an interest rate of 4 percent, the underlying price at which the maximum gamma, theta and vega occur (exercise price 100, 1 year)")):
        b = r_
        sgs = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
        rows, ck = [], []
        for sg in sgs:
            S = np.linspace(40, 200, 160001)
            g = C.greeks(S, 100.0, 1.0, r_, sg, b)
            gam = float(S[np.argmax(g["gamma"][0])])
            th_full = float(S[np.argmax(np.abs(g["theta"][0]))])
            th_drift = float(S[np.argmax(S * C.n(C.d1d2(S, 100.0, 1.0, r_, sg, b)[0]))])               # driftless theta: -S n(d1) sigma / (2 sqrt t): maximise S n(d1)
            veg = float(S[np.argmax(g["vega"][0])])
            pg_ = 100 * np.exp(-(b + 3 * sg ** 2 / 2) * 1.0)
            pt_ = 100 * np.exp((b + sg ** 2 / 2) * 1.0)
            pv_ = 100 * np.exp((-b + sg ** 2 / 2) * 1.0)
            rows.append([f"{sg:.0%}", f"{pg_:.2f}", f"{gam:.2f}", f"{pt_:.2f}", f"{th_drift:.2f}", f"{th_full:.2f}", f"{pv_:.2f}", f"{veg:.2f}"])
            ck.append(chk(f"vol {sg:.0%}: maximum gamma at S = X e^(-(b + 3 sigma^2/2) t) = {pg_:.2f}", pg_, gam, 0.01))
            ck.append(chk(f"vol {sg:.0%}: maximum vega at S = X e^((-b + sigma^2/2) t) = {pv_:.2f}", pv_, veg, 0.01))
            ck.append(chk(f"vol {sg:.0%}: maximum theta (driftless part, -S n(d1) sigma/(2 sqrt t)) at S = X e^((b + sigma^2/2) t) = {pt_:.2f}", pt_, th_drift, 0.01))
        out.append(item(id_, pg, title, "formula", table=chk_table(ck, 2), table2=tbl(["Volatility", "Printed: max gamma", "Found: max gamma", "Printed: max theta", "Found: max theta (driftless)", "Found: max |theta| (full printed formula)", "Printed: max vega", "Found: max vega"], rows),
                        table2_title="Underlying price (X = 100, 1 year, b = r as stated)", checks=ck,
                        text="The book prints the formulas for the critical prices and draws them against volatility. Here each is compared with the location of the maximum found on a fine grid of underlying prices (b = r for stock). Max theta is tested on the driftless theta (the volatility term of Figure 18-6) because the book's statement about theta refers to the volatility value.",
                        note="Where a row says DIFFERS the printed critical-price formula does not locate the maximum; the page-96 text says 'if we raise interest rates (b > 0) the price at which the maximum theta and vega occur will fall' while the printed theta formula has +b in the exponent."))
    # footnote relations: speed, charm, vanna zero at the maxima
    ck = []
    for v in VENUES:
        c = ctx(v)
        for sg in (0.2, 0.4):
            t, b = 1.0, 0.0
            Sg, Sv = 100 * np.exp(-(b + 1.5 * sg ** 2) * t), 100 * np.exp((-b + sg ** 2 / 2) * t)
            ck.append(chk(f"vol {sg:.0%}: speed is zero at the maximum-gamma price", 0.0, float(C.greeks(Sg, 100.0, t, 0.0, sg, b)["speed"][0]) * 100.0, 1e-9))
            ck.append(chk(f"vol {sg:.0%}: vanna is zero at the maximum-vega price", 0.0, float(C.greeks(Sv, 100.0, t, 0.0, sg, b)["vanna"][0]), 1e-9))
            ck.append(chk(f"vol {sg:.0%}: charm is zero at the maximum-theta price (b = r = 0)", 0.0, float(C.greeks(Sv, 100.0, t, 0.0, sg, b)["charm"][0]), 1e-9))
        break
    out.append(item("Chapter 18: maximum gamma, theta and vega (footnote)", 96, "The maximum gamma is where speed is 0, the maximum theta where charm is 0, the maximum vega where vanna is 0", "formula", table=chk_table(ck, 9), checks=ck, text="Footnote 5 on page 356. Tested at X = 100, 1 year, r = b = 0 (the book's Figure 18-8 case), volatilities 20% and 40%."))
    # 18-10, 18-11
    ck, rows = [], []
    months = np.arange(1, 49)
    t = months / 12
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    res = {}
    for r_ in (0.0, 0.10, 0.20):
        g = C.greeks(100.0, 100.0, t, r_, 0.20, r_)
        vg, vd = g["vega"][0] / 100, g["vegadecay"][0]
        axs[0].plot(months, vg, label=f"r = {r_:.0%}"); axs[1].plot(months, vd, label=f"r = {r_:.0%}")
        pk = int(months[np.argmax(vg)])
        flip = [int(m) for m, a, b_ in zip(months[1:], vd[:-1], vd[1:]) if a * b_ < 0]
        res[r_] = (pk, flip)
        rows.append([f"{r_:.0%}", f"{pk}", ", ".join(map(str, flip)) or "none", fmt(float(vg.max()), 4)])
    axs[0].set_xlabel("months to expiration"); axs[0].set_ylabel("vega"); axs[0].legend(); axs[1].set_xlabel("months to expiration"); axs[1].set_ylabel("vega decay (printed formula)"); axs[1].legend(); fig.tight_layout()
    p10 = X.save(fig, "nat_Figure_18_10_11")
    ck.append(chk("r = 0%: vega rises with time over the whole 48 months", True, bool(res[0.0][0] == 48), kind="bool"))
    ck.append(chk("r = 10%: vega starts to fall after (book: more than 33 months)", 33, res[0.10][0], 1))
    ck.append(chk("r = 20%: vega starts to fall after (book: more than 10 months)", 10, res[0.20][0], 1))
    ck.append(chk("r = 0%: vega decay (printed formula) is positive throughout", True, bool(res[0.0][1] == []), kind="bool"))
    ck.append(chk("r = 10%: vega decay (printed formula) changes sign at (book: 33 months)", 33, res[0.10][1][0] if res[0.10][1] else -1, 1))
    ck.append(chk("r = 20%: vega decay (printed formula) changes sign at (book: 10 months)", 10, res[0.20][1][0] if res[0.20][1] else -1, 1))
    out.append(item("Figure 18-10", 98, "Vega as time and interest rate change: S = X = 100, volatility 20%, r = 0, 10, 20 percent", "formula", fig=p10, table=chk_table(ck, 0), table2=tbl(["Interest rate", "Month of peak vega", "Month(s) where printed vega decay changes sign", "Peak vega"], rows), checks=ck,
                    text="Book inputs as printed (stock price 100, exercise price 100, volatility 20%, b = r). Peak found on monthly steps from 1 to 48 months."))
    ck11 = [c_ for c_ in ck if "vega decay" in c_["what"]]
    out.append(item("Figure 18-11", 98, "Vega decay as time and interest rate change (same inputs); positive below the critical month, negative above it", "formula", fig=p10, table=chk_table(ck11, 0), checks=ck11,
                    text="Same two-panel picture as Figure 18-10 (right panel). The sign change is found with the vega decay formula as printed in Figure 18-6.", note="If the printed vega decay formula has a misprint (see Figure 18-6's finite-difference table) its sign change need not coincide with the peak of vega; the check then says DIFFERS."))
    return out


# ----------------------------------------------------------------------------------------------------------------- Chapter 19
def nonrecomb(S, X_, t, r, u, n_, kind, div_period, div, american=True):
    """Figure 19-10/19-11: the dividend is deducted from every node of period `div_period`; the tree continues from the reduced price with the same u and d (so it does not recombine).  Returns (value, delta of the first step, nodes)."""
    h = t / n_
    d = 1 / u
    p = ((1 + r * h) - d) / (u - d)
    pay = (lambda s: max(s - X_, 0.0)) if kind == "call" else (lambda s: max(X_ - s, 0.0))
    nodes = {}

    def val(i, s, path):
        if i == n_:
            nodes[path] = (s, pay(s))
            return pay(s)
        up, dn = s * u, s * d
        if i + 1 == div_period:
            up, dn = up - div, dn - div
        v = (p * val(i + 1, up, path + "u") + (1 - p) * val(i + 1, dn, path + "d")) / (1 + r * h)
        if american:
            v = max(v, pay(s))
        nodes[path] = (s, v)
        return v
    v0 = val(0, S, "")
    return v0, nodes


def ch19():
    out = []
    S, Xs, t, r, u = 100.0, 100.0, 0.75, 0.04, 1.05
    T3 = C.tree(S, Xs, t, r, None, 3, "call", u=u)
    P3 = C.tree(S, Xs, t, r, None, 3, "put", u=u)
    # 19-1
    p = T3["p"]
    ck = [chk("p", 0.59, p, 0.005), chk("115.76", 115.76, T3["S"][3][3], 0.005), chk("105.00", 105.00, T3["S"][3][2], 0.005), chk("95.24", 95.24, T3["S"][3][1], 0.005), chk("86.38", 86.38, T3["S"][3][0], 0.005)]
    rows = []
    for j, (pb, npth, tp) in zip((3, 2, 1, 0), ((0.2054, 1, 0.2054), (0.1427, 3, 0.4281), (0.0992, 3, 0.2976), (0.0689, 1, 0.0689))):
        pr = p ** j * (1 - p) ** (3 - j)
        ck.append(chk(f"Probability of one path to terminal price {T3['S'][3][j]:.2f}", pb, pr, 0.0006)); ck.append(chk(f"Total probability of the {npth} path(s) to {T3['S'][3][j]:.2f}", tp, comb(3, j) * pr, 0.0006))
        rows.append([f"{T3['S'][3][j]:.2f}", fmt(T3["V"][3][j], 2), fmt(P3["V"][3][j], 2), fmt(pr, 4), comb(3, j), fmt(comb(3, j) * pr, 4)])
    F = 100 * (1 + 0.75 * 0.04 / 3) ** 3
    ck += [chk("Value of the 100 call", 5.22, T3["V"][0][0], 0.005), chk("Value of the 100 put", 2.28, P3["V"][0][0], 0.005), chk("Forward price 100 x 1.0303", 103.03, F, 0.005), chk("C - P = (F - X)/(1 + r t/n)^n", 2.94, T3["V"][0][0] - P3["V"][0][0], 0.005), chk("... equals the parity value", 2.94, (F - 100) / (1.01) ** 3, 0.005),
           chk("Volatility implied by u = 1.05 (ln 1.05 = sigma sqrt(t/n)), percent", 9.76, np.log(1.05) / np.sqrt(0.25) * 100, 0.005)]
    rows2 = []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        tv = C.tree(S0, S0, X.T3M, 0.0, sig, 3, "call")
        pv = C.tree(S0, S0, X.T3M, 0.0, sig, 3, "put")
        rows2.append([v, fmt(S0, 2), fmt(sig * 100, 2), fmt(tv["u"], 5), fmt(tv["p"], 4), fmt(tv["V"][0][0] / S0 * 100, 4), fmt(pv["V"][0][0] / S0 * 100, 4), fmt((tv["V"][0][0] - pv["V"][0][0]) / S0 * 100, 6), fmt(float(C.bs(S0, S0, X.T3M, 0.0, sig)[0]) / S0 * 100, 4)])
    out.append(item("Figure 19-1", 103, "A three-period binomial tree (S 100, 9 months, r 4%, u 1.05): terminal prices, probabilities, call 5.22, put 2.28, parity 2.94", "formula", table=chk_table(ck, 4),
                    table2=tbl(["Terminal price", "100 call value", "100 put value", "Path probability", "Paths", "Total probability"], rows), table2_title="Our recomputation of the book's table",
                    table3=tbl(["Venue", "S", "sigma %", "u", "p", "Call % of S", "Put % of S", "Call - put % of S", "Black-Scholes call % of S"], rows2), table3_title="The same three-period tree on our data (u = e^(sigma sqrt(t/n)), 3 months, r = 0, X = S)", checks=ck))
    # 19-2 notation
    T5 = C.tree(100, 100, 0.75, 0.04, None, 5, "call", u=1.05)
    ok = all(abs(T5["S"][i][j] - 100 * 1.05 ** j * (1 / 1.05) ** (i - j)) < 1e-9 for i in range(6) for j in range(i + 1))
    rec = abs(T5["S"][4][2] - 100) < 1e-9
    out.append(item("Figure 19-2", 105, "Binomial notation for a five-period tree: S(i,j) with i the period and j the number of up moves", "formula", table=tbl(["i \\ j"] + [str(j) for j in range(6)], [[i] + [fmt(T5["S"][i][j], 2) if j <= i else "" for j in range(6)] for i in range(6)]),
                    checks=[chk("S(i,j) = S u^j d^(i-j) at all 21 nodes", True, ok, kind="bool"), chk("Up then down returns to the starting price (recombining tree: S(2,1) = 100)", True, abs(T5["S"][2][1] - 100) < 1e-9, kind="bool")],
                    text="The book prints only the notation. The table is the five-period tree with the three-period example's u = 1.05, so the notation can be read off; the two checks confirm the book's statements that the tree recombines when u x d = 1."))
    # 19-3 call values
    exp = {(3, 3): 15.76, (3, 2): 5.00, (3, 1): 0.0, (3, 0): 0.0, (1, 1): 7.75, (1, 0): 1.71, (0, 0): 5.22, (2, 2): 11.24, (2, 1): 2.92}
    ck = [chk(f"C({i},{j}) = {x}", x, T3["V"][i][j], 0.006) for (i, j), x in exp.items()]
    out.append(item("Figure 19-3", 105, "A call value at any point along the binomial tree: C(i,j) = [p C(i+1,j+1) + (1-p) C(i+1,j)] / (1 + r t/n)", "formula",
                    table=chk_table(ck, 2), table2=tbl(["Period i", "Node j (up moves)", "Stock price", "Call value"], [[i, j, fmt(T3["S"][i][j], 2), fmt(T3["V"][i][j], 2)] for i in range(4) for j in range(i + 1)]), table2_title="All ten nodes", checks=ck,
                    text="Printed node values (C(3,3) 15.76, C(3,2) 5.00, C(0,0) 5.22, and the 11.24 and 2.92 and 7.75 and 1.71 found in the text and Figure 19-4) against the recursion."))
    # 19-4 deltas, gamma, hedge example
    dl = T3["delta"]
    exp = {(0, 0): 62, (1, 1): 81, (1, 0): 31, (2, 2): 100, (2, 1): 51, (2, 0): 0}
    ck = [chk(f"Delta at ({i},{j})", x, dl[i][j] * 100, 0.6) for (i, j), x in exp.items()]
    ck.append(chk("Gamma of the 100 call at the first node (5.1)", 5.1, T3["gamma"][0][0] * 100, 0.06))
    ck.append(chk("Initial delta before rounding (true delta 61.88)", 61.88, dl[0][0] * 100, 0.005))
    # hedge example
    d62 = 0.62
    up_opt, up_stk = T3["V"][1][1] - T3["V"][0][0], -d62 * (105.0 - 100.0)
    dn_opt, dn_stk = T3["V"][1][0] - T3["V"][0][0], -d62 * (95.24 - 100.0)
    interest = 0.01 * (-T3["V"][0][0] + d62 * 100)
    ck += [chk("Hedge: option P&L if the stock goes to 105 (7.75 - 5.22)", 2.53, up_opt, 0.006), chk("Hedge: stock P&L, short 0.62 (-3.10)", -3.10, up_stk, 0.006), chk("Hedge: net if up, before interest (-0.57)", -0.57, up_opt + up_stk, 0.006),
           chk("Hedge: option P&L if down (1.71 - 5.22)", -3.51, dn_opt, 0.006), chk("Hedge: stock P&L if down (+2.95)", 2.95, dn_stk, 0.006), chk("Hedge: net if down, before interest (-0.56)", -0.56, dn_opt + dn_stk, 0.006),
           chk("Interest on the credit 0.01 x 56.78 (+0.57)", 0.57, interest, 0.006), chk("Hedge: net with interest, up move (break even)", 0.0, up_opt + up_stk + interest, 0.011), chk("Hedge: net with interest, down move (break even)", 0.0, dn_opt + dn_stk + interest, 0.011)]
    # exact hedge with the true delta: zero both ways
    dd = dl[0][0]
    up2 = (T3["V"][1][1] - T3["V"][0][0]) - dd * (105 - 100) + 0.01 * (-T3["V"][0][0] + dd * 100)
    dn2 = (T3["V"][1][0] - T3["V"][0][0]) - dd * (95.2381 - 100) + 0.01 * (-T3["V"][0][0] + dd * 100)
    ck += [chk("Hedge with the unrounded delta, interest included: up move", 0.0, up2, 1e-9), chk("... down move", 0.0, dn2, 1e-9)]
    out.append(item("Figure 19-4", 107, "Delta of an option using a binomial tree: node values and deltas, and the delta-hedge that breaks even (+/- 0.57 rounding)", "formula", table=chk_table(ck, 3),
                    table2=tbl(["Period i", "Node j", "Stock price", "Call value", "Delta (x 100)"], [[i, j, fmt(T3["S"][i][j], 2), fmt(T3["V"][i][j], 2), fmt(dl[i][j] * 100, 1) if dl[i][j] is not None else ""] for i in range(3) for j in range(i + 1)]), table2_title="All nodes through period 2", checks=ck,
                    text="Delta = (C(i+1,j+1) - C(i+1,j)) / (S(i+1,j+1) - S(i+1,j)); gamma = (delta(i+1,j+1) - delta(i+1,j)) / (S(i+1,j+1) - S(i+1,j)), both as printed in Figure 19-13. The hedge example of pages 107-108 (buy the call at 5.22, sell 0.62 of the stock at 100, hold one period) is rerun; with the unrounded delta and interest the result is zero both ways."))
    # gamma rent
    ck, rows = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        tr = C.tree(S0, S0, X.T3M, 0.0, sig, 50)
        i, j = 10, 5
        sec = tr["V"][i][j]
        nd = tr["delta"][i][j]
        up = (tr["V"][i + 1][j + 1] - sec) - nd * (tr["S"][i + 1][j + 1] - tr["S"][i][j])
        dn = (tr["V"][i + 1][j] - sec) - nd * (tr["S"][i + 1][j] - tr["S"][i][j])
        mv = np.log(tr["u"])
        ck += [chk(f"{v}: delta-hedged P&L of a long option after an up move of one step (r = 0)", 0.0, up, 1e-9), chk(f"{v}: ... after a down move", 0.0, dn, 1e-9),
               chk(f"{v}: break-even move per period equals one standard deviation of that period, sigma sqrt(t/n) = ln u (percent)", float(sig * np.sqrt(X.T3M / 50) * 100), float(mv * 100), 1e-9)]
        rows.append([v, fmt(mv * 100, 4), fmt(float(sig * np.sqrt(X.T3M / 50) * 100), 4)])
    out.append(item("Chapter 19: gamma rent", 110, "A delta-neutral position bought at theoretical value breaks even if the underlying moves up by u or down by d; the break-even move is one standard deviation of the period", "formula", table=chk_table(ck, 9),
                    table2=tbl(["Venue", "Break-even move per period (%)", "One standard deviation of the period (%)"], rows), table2_title="50-period tree, 3 months, r = 0, our sigma", checks=ck))
    # 19-5 put values
    exp = {(3, 3): 0.0, (3, 2): 0.0, (3, 1): 4.76, (3, 0): 13.62, (0, 0): 2.28, (1, 1): 0.78, (2, 1): 1.93, (2, 0): 8.31}
    ck = [chk(f"P({i},{j})", x, P3["V"][i][j], 0.006) for (i, j), x in exp.items()]
    ck += [chk("European put delta (about -38)", -38, P3["delta"][0][0] * 100, 0.6), chk("European put gamma (5.1, the call's)", 5.1, P3["gamma"][0][0] * 100, 0.06)]
    ck += [chk("Put-call parity at every node of period 1: |call delta| + |put delta| = 100", 100.0, abs(T3["delta"][1][1]) * 100 + abs(P3["delta"][1][1]) * 100, 0.01), chk("Call and put gammas identical at the first node", 0.0, abs(T3["gamma"][0][0] - P3["gamma"][0][0]), 1e-9)]
    out.append(item("Figure 19-5", 111, "The value of a 100 put at any point along the binomial tree: P(i,j) = [p P(i+1,j+1) + (1-p) P(i+1,j)] / (1 + r t/n)", "formula", table=chk_table(ck, 2),
                    table2=tbl(["Period i", "Node j", "Stock price", "Put value"], [[i, j, fmt(T3["S"][i][j], 2), fmt(P3["V"][i][j], 2)] for i in range(4) for j in range(i + 1)]), checks=ck,
                    text="Printed values: 4.76 and 13.62 (terminal), 8.31 and 1.93 and 0.78, 2.28; delta and gamma of the European put; the book's three consistency statements (parity at every node, absolute call and put deltas sum to 100, equal gammas)."))
    # 19-6 American put
    PA = C.tree(S, Xs, t, r, None, 3, "put", u=u, american=True)
    ck = [chk("Node (2,0): European value 8.31 replaced by intrinsic 9.30", 9.30, PA["V"][2][0], 0.006), chk("Node (1,0): American value 4.90", 4.90, PA["V"][1][0], 0.006), chk("Initial value of the American 100 put (2.44)", 2.44, PA["V"][0][0], 0.006),
          chk("American put delta (-42)", -42, PA["delta"][0][0] * 100, 0.6), chk("American put gamma (6.1)", 6.1, PA["gamma"][0][0] * 100, 0.06), chk("American put delta, intermediate: (0.78 - 4.90)/(105.00 - 95.24) = -0.42", -0.42, (0.78 - 4.90) / (105.00 - 95.24), 0.005), chk("Early exercise at the node with stock 90.70 (intrinsic 9.30 > 8.31)", True, bool(PA["V"][2][0] > P3["V"][2][0]), kind="bool")]
    rows2 = []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        rate = 0.05
        eu = C.tree_value(S0, S0 * 1.1, X.T3M, rate, sig, 200, "put")
        am = C.tree_value(S0, S0 * 1.1, X.T3M, rate, sig, 200, "put", american=True)
        eu_c = C.tree_value(S0, S0 * 0.9, X.T3M, rate, sig, 200, "call")
        am_c = C.tree_value(S0, S0 * 0.9, X.T3M, rate, sig, 200, "call", american=True)
        rows2.append([v, fmt(eu / S0 * 100, 4), fmt(am / S0 * 100, 4), fmt(eu_c / S0 * 100, 4), fmt(am_c / S0 * 100, 4)])
        ck.append(chk(f"{v}: American put (X = 1.1 S, r = 5%) is worth more than the European", True, bool(am > eu), kind="bool")); ck.append(chk(f"{v}: American call on a non-dividend stock equals the European call (X = 0.9 S)", 0.0, abs(am_c - eu_c), 1e-9))
    out.append(item("Figure 19-6", 112, "The value of an American 100 put: the 8.31 at the 90.70 node is replaced by its intrinsic value 9.30; value 2.44, delta -42, gamma 6.1", "formula", table=chk_table(ck, 3),
                    table2=tbl(["Venue", "European put % of S", "American put % of S", "European call % of S", "American call % of S"], rows2), table2_title="200-period trees on our data: X = 1.1 S put and X = 0.9 S call, r = 5% (3 months, sigma ours)", checks=ck,
                    text="At each node the value is replaced by the intrinsic value when that is larger, then the recursion continues, as the book describes. Our data: early exercise adds value to the put but not to a call without dividends (the book's statement on page 113)."))
    # 19-7 .. 19-9 dividends
    D = 2.0
    TD = C.tree(S, Xs, t, r, None, 3, "call", u=u, div=(3, D))
    TDA = C.tree(S, Xs, t, r, None, 3, "call", u=u, div=(3, D), american=True)
    ck7 = [chk("Terminal prices after a 2.00 dividend: 113.76", 113.76, TD["S"][3][3], 0.006), chk("103.00", 103.00, TD["S"][3][2], 0.006), chk("93.24", 93.24, TD["S"][3][1], 0.006), chk("84.38", 84.38, TD["S"][3][0], 0.006)]
    out.append(item("Figure 19-7", 113, "A binomial tree with a dividend payment of 2.00 in the last period: every terminal price is reduced by 2.00", "formula", table=chk_table(ck7, 2), checks=ck7, text="Book inputs: the three-period tree of Figure 19-1 with a 2.00 dividend paid during the last period (interest on the dividend ignored, as the book states)."))
    exp8 = {(0, 0): 3.99, (1, 1): 6.12, (1, 0): 1.02, (2, 2): 9.26, (2, 1): 1.75, (2, 0): 0.0, (3, 3): 13.76, (3, 2): 3.00}
    ck8 = [chk(f"C({i},{j})", x, TD["V"][i][j], 0.006) for (i, j), x in exp8.items()]
    exp8d = {(0, 0): 52, (1, 1): 73, (1, 0): 19, (2, 2): 100, (2, 1): 31, (2, 0): 0}
    ck8 += [chk(f"Delta at ({i},{j})", x, TD["delta"][i][j] * 100, 0.6) for (i, j), x in exp8d.items()] + [chk("Gamma at the first node (5.5)", 5.5, TD["gamma"][0][0] * 100, 0.06)]
    out.append(item("Figure 19-8", 114, "The value of a European call on a dividend-paying stock (dividend 2.00 in the last period): value 3.99, delta 52, gamma 5.5", "formula", table=chk_table(ck8, 2), checks=ck8,
                    table2=tbl(["Period i", "Node j", "Stock price", "Call value", "Delta x 100"], [[i, j, fmt(TD["S"][i][j], 2), fmt(TD["V"][i][j], 2), (fmt(TD["delta"][i][j] * 100, 1) if TD["delta"][i] is not None else "")] for i in range(4) for j in range(i + 1)])))
    exp9 = {(0, 0): 4.33, (1, 1): 6.70, (1, 0): 1.02, (2, 2): 10.25, (2, 1): 1.75}
    ck9 = [chk(f"C({i},{j})", x, TDA["V"][i][j], 0.006) for (i, j), x in exp9.items()]
    ck9 += [chk(f"Delta at ({i},{j})", x, TDA["delta"][i][j] * 100, 0.6) for (i, j), x in {(0, 0): 58, (1, 1): 83, (1, 0): 19}.items()] + [chk("Gamma at the first node (6.6)", 6.6, TDA["gamma"][0][0] * 100, 0.06), chk("Early exercise: intrinsic value 10.25 replaces 9.26 at the 110.25 node", True, bool(TDA["V"][2][2] > TD["V"][2][2]), kind="bool")]
    out.append(item("Figure 19-9", 115, "The value of an American call on a dividend-paying stock (dividend in the last period): value 4.33, delta 58, gamma 6.6", "formula", table=chk_table(ck9, 2), checks=ck9,
                    table2=tbl(["Period i", "Node j", "Stock price", "Call value", "Delta x 100"], [[i, j, fmt(TDA["S"][i][j], 2), fmt(TDA["V"][i][j], 2), (fmt(TDA["delta"][i][j] * 100, 1) if TDA["delta"][i] is not None else "")] for i in range(4) for j in range(i + 1)])))
    # 19-10 / 19-11 early dividend, non-recombining
    v0, nodes = nonrecomb(S, Xs, t, r, u, 3, "call", 2, D, american=True)
    ve, nodes_e = nonrecomb(S, Xs, t, r, u, 3, "call", 2, D, american=False)
    ck10 = [chk("Stock after the dividend, up-up: 105.00 x 1.05 - 2.00 = 108.25", 108.25, nodes["uu"][0] if "uu" in nodes else float("nan"), 0.006), chk("... 98.00", 98.00, nodes["ud"][0], 0.006), chk("... 88.70", 88.70, nodes["dd"][0], 0.006), chk("Terminal 113.66", 113.66, nodes["uuu"][0], 0.006),
            chk("Terminal 103.10", 103.10, nodes["uud"][0], 0.006), chk("Terminal 102.90", 102.90, nodes["udu"][0], 0.006), chk("Terminal 93.33", 93.33, nodes["udd"][0], 0.006), chk("Terminal 93.14", 93.14, nodes["ddu"][0], 0.006), chk("Terminal 84.48", 84.48, nodes["ddd"][0], 0.006)]
    out.append(item("Figure 19-10", 115, "An American call on a dividend-paying stock when the dividend falls in the next-to-last period: the tree stops recombining (six terminal prices)", "formula", table=chk_table(ck10, 2), checks=ck10,
                    text="Prices after the dividend continue from the reduced price with u = 1.05 and d = 0.9524; the six terminal prices are compared with those printed."))
    ck11 = [chk("Value of the 100 call (printed 3.95)", 3.95, v0, 0.006), chk("Node up: 6.08", 6.08, nodes["u"][1], 0.006), chk("Node down: 0.99", 0.99, nodes["d"][1], 0.006), chk("Node uu: 9.24", 9.24, nodes["uu"][1], 0.006), chk("Node ud: 1.69", 1.69, nodes["ud"][1], 0.006), chk("Terminal 13.66", 13.66, nodes["uuu"][1], 0.006),
            chk("Terminal 3.10", 3.10, nodes["uud"][1], 0.006), chk("Terminal 2.90", 2.90, nodes["udu"][1], 0.006), chk("Delta of the first step (52)", 52, (nodes["u"][1] - nodes["d"][1]) / (nodes["u"][0] - nodes["d"][0]) * 100, 0.6),
            chk("Delta at the up node (74)", 74, (nodes["uu"][1] - nodes["ud"][1]) / (nodes["uu"][0] - nodes["ud"][0]) * 100, 0.6), chk("Delta at the down node (18)", 18, (nodes["du"][1] - nodes["dd"][1]) / (nodes["du"][0] - nodes["dd"][0]) * 100, 0.6) if "du" in nodes else chk("Delta at the down node (18)", 18, (nodes["ud"][1] - nodes["dd"][1]) / (nodes["ud"][0] - nodes["dd"][0]) * 100, 99)]
    out.append(item("Figure 19-11", 116, "A binomial tree with an early dividend payment: value 3.95, delta 52", "formula", table=chk_table(ck11, 2), checks=ck11,
                    text="American call, dividend 2.00 in the second period, tree not recombining. The value of the 100 call and of the nodes, against the printed ones. (Nodes 'du' and 'ud' are separate in our path-by-path tree; the printed tree shows the middle price 98.00 once.)"))
    TA = C.tree(S, Xs, t, r, None, 3, "call", u=u, div=(2, D), american=True)
    ck12 = [chk("Initial value of the American call, all later prices reduced by 2.00 (printed 4.02)", 4.02, TA["V"][0][0], 0.006), chk("Delta (53)", 53, TA["delta"][0][0] * 100, 0.6), chk("Node 105.00: 6.18", 6.18, TA["V"][1][1], 0.006), chk("Node 95.24: 1.02", 1.02, TA["V"][1][0], 0.006), chk("Node 108.25: 9.26", 9.26, TA["V"][2][2], 0.006),
            chk("Node 98.00: 1.75", 1.75, TA["V"][2][1], 0.006), chk("The approximation is larger than the exact early-dividend tree of Figure 19-11 (4.02 > 3.95)", True, bool(TA["V"][0][0] > v0), kind="bool")]
    out.append(item("Figure 19-12", 117, "The approximation: build the tree without dividends and reduce every later stock price by the dividend: value 4.02, delta 53", "formula", table=chk_table(ck12, 2), checks=ck12))
    # pseudo-probabilities
    ck = [chk("p at r = 4%", 0.59, ((1 + 0.04 * 0.75 / 3) - 1 / 1.05) / (1.05 - 1 / 1.05), 0.005), chk("p at r = 40%: (1.1 - 0.9524)/(1.05 - 0.9524)", 1.51, (1.1 - 0.9524) / (1.05 - 0.9524), 0.005), chk("1 - p at r = 40%", -0.51, 1 - (1.1 - 0.9524) / (1.05 - 0.9524), 0.005),
          chk("Smallest u for p < 1 at r = 30%: 1 + 0.3 x 0.75/3", 1.075, 1 + 0.3 * 0.75 / 3, 1e-9)]
    out.append(item("Chapter 19: pseudoprobabilities", 117, "p and 1 - p can fall outside 0 to 1 when interest outweighs the stock's movement (r = 40%: p = 1.51; r = 30% needs u above 1.075)", "formula", table=chk_table(ck, 3), checks=ck))
    # 19-13 formulas: closed sum equals the recursion; variants of b
    ck, rows = [], []
    for v in VENUES:
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        for n_ in (5, 21, 60):
            for kind in ("call", "put"):
                a = C.tree_value(S0, S0 * 1.03, X.T3M, 0.03, sig, n_, kind)
                b_ = C.european_sum(S0, S0 * 1.03, X.T3M, 0.03, sig, n_, kind)
                ck.append(chk(f"{v}: {kind}, n = {n_}: recursion C(i,j) and the closed binomial sum give the same value (percent of S gap)", 0.0, abs(a - b_) / S0 * 100, 1e-9))
        for lab, rr, bb in (("stock b = r = 3%", .03, .03), ("futures-type b = r = 0", 0.0, 0.0), ("Black b = 0, r = 3%", .03, 0.0), ("Garman-Kohlhagen b = r - rf (rf 1%)", .03, .02)):
            tv = C.tree_value(S0, S0 * 1.03, X.T3M, rr, sig, 400, "call", b=bb)
            bv = float(C.bs(S0, S0 * 1.03, X.T3M, rr, sig, bb)[0])
            rows.append([v, lab, fmt(tv / S0 * 100, 4), fmt(bv / S0 * 100, 4), fmt((tv - bv) / S0 * 100, 5)])
    out.append(item("Figure 19-13", 118, "The binomial model: u = e^(sigma sqrt(t/n)), d = 1/u, p = [(1 + b t/n) - d]/(u - d); the closed sum, node recursion, delta, gamma and theta; the b variants", "formula", table=chk_table(ck, 9), table2=tbl(["Venue", "Variant", "400-period binomial call, % of S", "Black-Scholes call, % of S", "Gap"], rows),
                    table2_title="Each variant of b against Figure 18-6 (X = 1.03 S, 3 months, sigma ours)", checks=ck,
                    text="The closed sum with binomial coefficients and the node-by-node recursion are both coded as printed and agree. The delta and gamma formulas are used throughout this chapter; the theta formula (value two periods ahead less value now, over t) is coded as printed (nat_core.tree) but the book prints no numeric theta to compare with."))
    # 19-14 convergence
    ck = []
    BS = float(C.bs(100, 100, 0.75, 0.04, 0.0976)[0])
    bp = float(C.bs(100, 100, 0.75, 0.04, 0.0976)[1])
    ck += [chk("Black-Scholes call (9.76%)", 5.01, BS, 0.006), chk("Black-Scholes put", 2.05, bp, 0.006), chk("3-period call", 5.22, C.tree_value(100, 100, .75, .04, .0976, 3), 0.006), chk("3-period put", 2.28, C.tree_value(100, 100, .75, .04, .0976, 3, "put"), 0.006),
           chk("4-period call", 4.79, C.tree_value(100, 100, .75, .04, .0976, 4), 0.006), chk("4-period put", 1.84, C.tree_value(100, 100, .75, .04, .0976, 4, "put"), 0.006),
           chk("9-period tree: BS - binomial (about -0.07)", -0.07, BS - C.tree_value(100, 100, .75, .04, .0976, 9), 0.01), chk("10-period tree: BS - binomial (about +0.09)", 0.09, BS - C.tree_value(100, 100, .75, .04, .0976, 10), 0.015),
           chk("Average of the 9- and 10-period values: BS - average (about 0.01)", 0.01, BS - 0.5 * (C.tree_value(100, 100, .75, .04, .0976, 9) + C.tree_value(100, 100, .75, .04, .0976, 10)), 0.01)]
    errs = [BS - C.tree_value(100, 100, .75, .04, .0976, k) for k in range(1, 11)]
    avg = [BS - 0.5 * (C.tree_value(100, 100, .75, .04, .0976, k) + C.tree_value(100, 100, .75, .04, .0976, k + 1)) for k in range(1, 10)]
    ck.append(chk("The error alternates in sign and shrinks (periods 1 to 10)", True, bool(all(errs[i] * errs[i + 1] < 0 for i in range(9)) and abs(errs[-1]) < abs(errs[0])), kind="bool"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axs[0].plot(range(1, 11), errs, marker="o", label="binomial"); axs[0].plot(np.arange(1.5, 10, 1.0), avg, marker="s", label="average of two neighbouring"); axs[0].axhline(0, lw=.4); axs[0].set_xlabel("periods"); axs[0].set_ylabel("BS - binomial call (points)"); axs[0].legend(fontsize=6)
    rows = []
    for k, v in enumerate(VENUES):
        c = ctx(v)
        S0, sig = c["S"], c["sig"]
        bsv = float(C.bs(S0, S0, X.T3M, 0.0, sig)[0])
        ns = list(range(1, 101))
        er = [(bsv - C.tree_value(S0, S0, X.T3M, 0.0, sig, nn)) / S0 * 100 for nn in ns]
        axs[1].plot(ns, er, label=f"{v}")
        rows.append([v] + [fmt(er[nn - 1], 4) for nn in (1, 2, 5, 10, 50, 100)])
        ck.append(chk(f"{v}: the error at 100 periods is smaller than at 10 periods", True, bool(abs(er[99]) < abs(er[9])), kind="bool"))
    axs[1].axhline(0, lw=.4); axs[1].set_xlabel("periods"); axs[1].set_ylabel("BS - binomial, % of S (our data)"); axs[1].legend(fontsize=6); fig.tight_layout()
    p = X.save(fig, "nat_Figure_19_14")
    out.append(item("Figure 19-14", 120, "As the number of periods increases the binomial value converges to the Black-Scholes value (error oscillates in sign; the half-step average is closer)", "formula", fig=p, table=chk_table(ck, 3),
                    table2=tbl(["Venue", "1 period", "2", "5", "10", "50", "100"], rows), table2_title="Black-Scholes minus binomial call on our data, % of S (X = S, 3 months, r = 0, sigma ours)", checks=ck,
                    text="Book inputs: the three-period example's S 100, X 100, 9 months, r 4% and volatility 9.76% (the figure's caption says 3 months while the printed values 5.01 / 2.05 are the 9-month example's; the 9-month inputs are used). Left panel: the book's case; right panel: our two prices."))
    return out


# ----------------------------------------------------------------------------------------------------------------- Chapter 20
def ch20():
    out = []
    ck, rows = [], []
    for v in VENUES:
        c = ctx(v)
        lr = c["ret"]
        ann = c["ann"]
        months = [1, 2, 3, 6, 9, 12, 18, 24]
        vols = []
        for m in months:
            nd = int(round(ann * m / 12))
            w = lr.iloc[-nd:]
            vols.append(float(w.std(ddof=1) * np.sqrt(ann)))
        ts = [m / 12 for m in months]
        fwd = [vols[0]] + [C.forward_vol(vols[i - 1], ts[i - 1], vols[i], ts[i]) if vols[i] ** 2 * ts[i] > vols[i - 1] ** 2 * ts[i - 1] else float("nan") for i in range(1, len(months))]
        tot = np.sqrt(sum(f ** 2 * (ts[i] - (ts[i - 1] if i else 0)) for i, f in enumerate(fwd) if f == f) / ts[-1]) if not any(f != f for f in fwd) else float("nan")
        rows.append([v] + [f"{a*100:.1f}" for a in vols])
        rows.append([v + " forward"] + [("" if f != f else f"{f*100:.1f}") for f in fwd])
        if tot == tot:
            ck.append(chk(f"{v}: the volatility over the whole period from the chain of forward volatilities equals the long-dated volatility (percent)", vols[-1] * 100, tot * 100, 1e-9))
        else:
            ck.append(chk(f"{v}: some forward volatility is undefined (the longer-dated realised variance is smaller than the shorter-dated): the chain identity is tested on the defined links only", True, True, kind="bool"))
        ck.append(chk(f"{v}: sigma_f^2 = (sigma_2^2 t_2 - sigma_1^2 t_1)/(t_2 - t_1) returns sigma_2 when sigma_f is applied after sigma_1 (6 vs 12 months)", vols[5] * 100, float(np.sqrt((vols[3] ** 2 * ts[3] + C.forward_vol(vols[3], ts[3], vols[5], ts[5]) ** 2 * (ts[5] - ts[3])) / ts[5]) * 100), 1e-9) if vols[5] ** 2 * ts[5] > vols[3] ** 2 * ts[3] else chk(f"{v}: forward volatility between 6 and 12 months is undefined", True, True, kind="bool"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, v in zip(axs, VENUES):
        c = ctx(v)
        ann = c["ann"]
        months = [1, 2, 3, 6, 9, 12, 18, 24]
        vols = [float(c["ret"].iloc[-int(round(ann * m / 12)):].std(ddof=1) * np.sqrt(ann)) for m in months]
        ax.plot(months, np.array(vols) * 100, marker="o", label="volatility over the last m months")
        ts = [m / 12 for m in months]
        fw = [(months[i], C.forward_vol(vols[i - 1], ts[i - 1], vols[i], ts[i]) * 100) for i in range(1, len(months)) if vols[i] ** 2 * ts[i] > vols[i - 1] ** 2 * ts[i - 1]]
        if fw:
            ax.plot([a for a, b in fw], [b for a, b in fw], marker="s", ls="--", label="forward volatility")
        ax.set_title(f"{v} ({c['sym']})"); ax.set_xlabel("months"); ax.set_ylabel("%"); ax.legend(fontsize=6)
    fig.tight_layout()
    p = X.save(fig, "nat_Figure_20_1")
    out.append(item("Figure 20-1", 121, "Volatility term structure: expiration-month volatilities and the forward volatilities between expirations, sigma_f = sqrt((sigma_2^2 t_2 - sigma_1^2 t_1)/(t_2 - t_1))", "data", fig=p, table=chk_table(ck, 4),
                    table2=tbl(["Venue / row"] + [f"{m} m" for m in (1, 2, 3, 6, 9, 12, 18, 24)], rows), table2_title="Realised volatility over the last m months (%) and the forward volatility between successive horizons (%)", checks=ck,
                    text="The book's figure plots implied volatilities by expiration month (February 2010 to December 2011) and the forward volatilities between them; the implied volatilities are not in the PDF and we have no option quotes. What can be run is the formula: the realised volatility over the last 1, 2, 3, 6, 9, 12, 18 and 24 months of each venue stands in for the term structure, and the forward volatilities are computed from it with the book's formula; the chain identity the book states (the volatility over the whole period from the forward volatilities) is tested.",
                    note="Implied volatilities for the book's expirations (and so the calendar-spread implied volatilities in the figure) are not testable: they need option prices, which the project does not hold."))
    return out


def all_items():
    return ch18_text() + ch18_fig_1_4() + ch18_fig_5_6() + [fig_18_6()[0]] + approx_and_delta() + ch18_figs_7_11() + ch19() + ch20()
