"""Write-up items for Chapters 14 and 15 of the book (see writeup_core.py for the item format).  New file; reads finished runs, changes nothing."""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

import book_port_c as PC
import book_stats as BS
import writeup_core as W
from writeup_core import VENUES, SYM, item, tbl, fmt, save

CHECKS: list = []          # (what, ours, saved) rows used by the number-verification step


def meta(v):
    return W.run_json(v, "meta.json")


def _series(v, name, col):
    return pd.read_csv(W.RUNS[v] / name, index_col=0, parse_dates=True)[col]


def _bets(v):
    """Rebuilds the bet returns the way the bot did (book_bot_v2.stage_ch14): compounded daily net return between consecutive bet end-dates."""
    tpos, ret_d = _series(v, "ch14_target_position.csv", "target_position"), _series(v, "ch14_daily_return.csv", "net_return")
    equity = (1.0 + ret_d).cumprod()
    bets = PC.getBets(tpos)
    prev, vals = tpos.index[0], []
    for b in bets:
        seg = ret_d.loc[(ret_d.index > prev) & (ret_d.index <= b)] if prev != tpos.index[0] else ret_d.loc[ret_d.index <= b]
        vals.append(float((1.0 + seg).prod() - 1.0))
        prev = b
    return tpos, ret_d, equity, bets, pd.Series(vals, index=bets)


def heat(ax, X, Y, Z, xlabel, ylabel, title, levels=6, cmap="gray", cbar=True):
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto")
    try:
        cs = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.6, linestyles="--")
    except Exception:
        pass
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.05)


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch14():
    out = []
    data = {v: _bets(v) for v in VENUES}
    saved = {v: meta(v)["ch14"] for v in VENUES}

    # Snippet 14.1 / 14.2 ----------------------------------------------------------------------------------------------------------------------------
    rows = []
    for v in VENUES:
        tpos, ret_d, equity, bets, bret = data[v]
        hp = PC.getHoldingPeriod(tpos)
        CHECKS.append((f"{v} Snippet 14.1 bets", len(bets), saved[v]["bets_14_1"]))
        CHECKS.append((f"{v} Snippet 14.2 holding period", hp, saved[v]["avg_holding_period_days_14_2"]))
        rows.append([v, fmt(len(bets)), fmt(hp, 3), f"{tpos.index[0].date()} to {tpos.index[-1].date()}", fmt(int((tpos != 0).sum())), fmt(float(tpos.abs().max()), 4)])
    out.append(item("Snippet 14.1", "Timing of bets from the series of target positions (flat or flip ends a bet)", "data",
                    table=tbl(["Venue", "Bets", "Avg holding period (days, 14.2)", "Span", "Days with a position", "Max |position|"], rows),
                    text="Run on the walk-forward target position of the last 365 days (one value per UTC day; HRP weight x averaged bet size). book_port_c.getBets is the printed code."))
    out.append(item("Snippet 14.2", "Average holding period by the average-entry-time pairing algorithm", "data",
                    table=tbl(["Venue", "Holding period (days)"], [[v, fmt(PC.getHoldingPeriod(data[v][0]), 3)] for v in VENUES]),
                    text="book_port_c.getHoldingPeriod is the printed code (xrange -> range)."))

    # Equations 45-49 --------------------------------------------------------------------------------------------------------------------------------
    t45 = []
    for v in VENUES:
        tpos, ret_d, equity, bets, bret = data[v]
        T = len(ret_d)
        phi = float(equity.iloc[-1])
        years = T / 365.25
        R = phi ** (1.0 / years) - 1.0
        CHECKS.append((f"{v} Eq 49 annualised return %", 100 * R, saved[v]["annualised_return_pct_eq49"]))
        CHECKS.append((f"{v} Eq 48 gain %", 100 * (phi - 1), saved[v]["gain_pct"]))
        t45.append([v, fmt(T), fmt(float(ret_d.mean()) * 1e4, 3), fmt(float(ret_d.std()) * 1e4, 3), fmt(phi, 6), fmt(years, 4), f"{100 * R:.4f}"])
    out.append(item("Equation 45", "Return of the portfolio on day t: r = PnL / capital", "formula",
                    table=tbl(["Venue", "Days T", "Mean r (bp/day)", "Std r (bp/day)", "phi_T", "Years y", "Annualised R (%)"], t45),
                    text="r_t = pi_t / K_t.  The bot computes it as the HRP-weighted sum of the symbols' position x next-day return, minus costs, divided by the unit capital of 1 (long-only capital rule below).  The table lists the daily series' summary."))
    out.append(item("Equation 46", "PnL pi_t = sum_j [(dP + A) theta_(t-1) + d theta (P_t - P_(t-1))]", "formula",
                    text="Spot positions pay no coupon or dividend feed in our price series (A = 0 on Kraken; Alpaca prices are the venue's split-adjusted bars), so the PnL is position x price change plus the cost term of the rebalance trade.  Costs: Kraken 12 bp per side, Alpaca 10 bp per side."))
    out.append(item("Equation 47", "Capital K = sum |P theta| + max(0, sum P d theta)", "formula",
                    text="With a fully funded, unlevered book the capital is the gross position value; the bot normalises it to 1 and caps each symbol's weight by the HRP allocation, so the capital rule only scales r_t."))
    out.append(item("Equation 48", "Cumulative growth phi_T = prod (1 + r_t)", "data",
                    table=tbl(["Venue", "phi_T", "Gain (%)"], [[v, fmt(float(data[v][2].iloc[-1]), 6), fmt(100 * (float(data[v][2].iloc[-1]) - 1), 4)] for v in VENUES])))
    out.append(item("Equation 49", "Annualised return R = phi_T^(1/y) - 1, with y the years elapsed", "data",
                    table=tbl(["Venue", "y (days / 365.25)", "R (%)", "Saved run (%)"], [[v, fmt(len(data[v][1]) / 365.25, 4), fmt(100 * (float(data[v][2].iloc[-1]) ** (365.25 / len(data[v][1])) - 1), 4), fmt(saved[v]["annualised_return_pct_eq49"], 4)] for v in VENUES]),
                    note="The printed equation shows the exponent as -y; the bot uses 1/y (the usual annualisation)."))

    # Series 1-4, Equation 50-51 ---------------------------------------------------------------------------------------------------------------------
    s_rows, h_rows = [], []
    for v in VENUES:
        bret = data[v][4]
        rp, rn = bret[bret >= 0], bret[bret < 0]
        wp, wn = rp / rp.sum(), rn / rn.sum()
        def h(w):
            n = len(w)
            return (float((w ** 2).sum()) - 1.0 / n) / (1.0 - 1.0 / n) if n > 2 else float("nan")
        s_rows.append([v, fmt(len(rp)), fmt(float(rp.sum()), 5), fmt(len(rn)), fmt(float(rn.sum()), 5), fmt(float(wp.max()) if len(wp) else None, 4), fmt(float(wn.max()) if len(wn) else None, 4)])
        h_rows.append([v, fmt(h(wp), 5), fmt(h(wn), 5), fmt(saved[v]["hhi_positive_returns_14_3"], 5), fmt(saved[v]["hhi_negative_returns_14_3"], 5)])
        CHECKS.append((f"{v} Eq 50 h+", h(wp), saved[v]["hhi_positive_returns_14_3"]))
        CHECKS.append((f"{v} Eq 51 h-", h(wn), saved[v]["hhi_negative_returns_14_3"]))
    out.append(item("Series 1-4", "Positive and negative bet returns and their weights (r+, r-, w+, w-)", "data",
                    table=tbl(["Venue", "# r+", "sum r+", "# r-", "sum r-", "max w+", "max w-"], s_rows),
                    text="Bet returns: compounded daily net return over each bet of Snippet 14.1 (last 365 days)."))
    out.append(item("Equation 50", "Concentration of positive returns h+ = (sum w+^2 - n^-1)/(1 - n^-1)", "data",
                    table=tbl(["Venue", "h+ (Eq 50, recomputed)", "h- (Eq 51, recomputed)", "h+ saved run", "h- saved run"], h_rows)))
    out.append(item("Equation 51", "Concentration of negative returns h-", "data", text="Same table as Equation 50 (the two are printed together).  A value near 0 means the returns are spread evenly across bets; near 1 means a single bet carries them."))

    # Snippet 14.3 -----------------------------------------------------------------------------------------------------------------------------------
    rows = []
    for v in VENUES:
        bret = data[v][4]
        hp_, hn_, th_ = PC.getHHIs(bret)
        CHECKS.append((f"{v} Snippet 14.3 HHI bets/month", th_, saved[v]["hhi_bets_per_month_14_3"]))
        rows.append([v, fmt(hp_, 5), fmt(hn_, 5), fmt(th_, 5), fmt(saved[v]["hhi_positive_returns_14_3"], 5), fmt(saved[v]["hhi_negative_returns_14_3"], 5), fmt(saved[v]["hhi_bets_per_month_14_3"], 5)])
    out.append(item("Snippet 14.3", "HHI concentration of positive returns, negative returns and bets per month", "data",
                    table=tbl(["Venue", "HHI+ ", "HHI-", "HHI time", "saved +", "saved -", "saved time"], rows),
                    text="book_port_c.getHHI / getHHIs; the monthly grouping uses pandas' month-end frequency ('ME'; the book's TimeGrouper is retired in pandas 3)."))

    # Snippet 14.4 / Figure 14.1 --------------------------------------------------------------------------------------------------------------------
    rows = []
    for v in VENUES:
        equity = data[v][2]
        dd, tuw = PC.computeDD_TuW(equity)
        CHECKS.append((f"{v} Snippet 14.4 max DD %", 100 * float(dd.max()) if len(dd) else 0.0, saved[v]["drawdowns_14_4"]["max_pct"]))
        rows.append([v, fmt(len(dd)), fmt(100 * float(dd.max()) if len(dd) else None, 3), fmt(len(tuw)), fmt(float(tuw.max()) if len(tuw) else None, 4)])
    out.append(item("Snippet 14.4", "Sequence of drawdowns (DD) and time under water (TuW) from the high-water marks", "data",
                    table=tbl(["Venue", "Drawdowns", "Max DD (%)", "TuW values", "Max TuW (years)"], rows),
                    text="TuW needs at least two high-water marks that are each followed by a drawdown; a one-year curve with one such mark has none (empty cells)."))

    def draw(ax, v):
        eq = data[v][2]
        hwm = eq.expanding().max()
        ax.plot(eq.index, eq.values, color="0.15", lw=1, label="equity")
        ax.plot(hwm.index, hwm.values, color="0.6", lw=1, ls="--", label="high-water mark")
        ax.fill_between(eq.index, eq.values, hwm.values, color="0.8", label="drawdown")
        ax.legend(loc="best"); ax.tick_params(axis="x", rotation=30)
    out.append(item("Figure 14.1", "Drawdown and time under water on our walk-forward equity (last 365 days)", "data", fig=W.two_panel(draw, "f14_1")))

    # Equation 52 / Figure 14.2 ----------------------------------------------------------------------------------------------------------------------
    rows = []
    for v in VENUES:
        d = meta(v)["dsr"]
        for span in ("last_365_days", "whole_walk_forward_span"):
            m = d[span]
            p = BS.psr(m["sr_daily"], m["n_days"], m["skew"], m["kurt"], 0.0)
            CHECKS.append((f"{v} {span} PSR vs 0", p, m["psr_vs_0"]))
            rows.append([v, span.replace("_", " "), fmt(m["n_days"]), fmt(m["sr_daily"], 5), fmt(m["sr_daily"] * math.sqrt(365), 3), fmt(m["skew"], 4), fmt(m["kurt"], 3), fmt(p, 5)])
    out.append(item("Equation 52", "Probabilistic Sharpe ratio PSR[SR*] on our daily walk-forward returns (SR* = 0)", "data",
                    table=tbl(["Venue", "Span", "T (days)", "SR daily", "SR x sqrt(365)", "skew", "kurtosis", "PSR"], rows),
                    text="Z[ (SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2) ].  Recomputed with book_stats.psr and compared with the run's saved value."))
    T = np.arange(1, 32)
    sk = np.linspace(-1.5, 1.0, 51)
    X, Y = np.meshgrid(sk, T)
    Z = np.array([[BS.psr(1.5, t, s, 3.0, 1.0) for s in sk] for t in T])
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    heat(ax, X, Y, Z, "skew = g3", "T", "PSR for g4 = 3.0, SR = 1.5, SR* = 1.0")
    out.append(item("Figure 14.2", "PSR as a function of skewness and sample length (the book's parameters)", "formula", fig=save(fig, "f14_2"),
                    note="Closed form, so it matches the book's figure by construction (axes: skew -1.5..1.0, T 1..31, parameters read off the printed figure)."))

    # Equation 53 / Figure 14.3 ----------------------------------------------------------------------------------------------------------------------
    rows = []
    for v in VENUES:
        for span in ("last_365_days", "whole_walk_forward_span"):
            m = meta(v)["dsr"][span]
            sr_star = BS.expected_max_sharpe(m["trials"], m["var_sr_used"])
            CHECKS.append((f"{v} {span} SR*", sr_star, m["sr_star"]))
            CHECKS.append((f"{v} {span} DSR", BS.dsr(m["sr_daily"], m["n_days"], m["skew"], m["kurt"], m["trials"], m["var_sr_used"]), m["dsr"]))
            rows.append([v, span.replace("_", " "), fmt(m["trials"]), fmt(m["var_sr_used"], 6), fmt(sr_star, 5), fmt(m["dsr"], 6)])
    out.append(item("Equation 53", "Expected maximum Sharpe ratio of N unskilled trials (SR*) and the deflated Sharpe ratio", "data",
                    table=tbl(["Venue", "Span", "Trials N", "V[SR]", "SR*", "DSR"], rows),
                    text="N = 11, the number of candidates tried in the Chapter 9 search; V[SR] = the asymptotic variance of the Sharpe estimate (the variance across the five CPCV paths is saved in the run too and is smaller)."))
    Vs = np.linspace(0.1, 1.0, 46)
    Ns = np.arange(5, 31)
    X, Y = np.meshgrid(Vs, Ns)
    Z = np.array([[BS.expected_max_sharpe(int(n), float(vv)) for vv in Vs] for n in Ns])
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    heat(ax, X, Y, Z, "V[SR_n]", "N", "SR*")
    out.append(item("Figure 14.3", "SR* as a function of V[SR_n] and N", "formula", fig=save(fig, "f14_3"), note="Closed form (axes as printed: V 0.1..1.0, N 5..30)."))

    # Table 14.1 -------------------------------------------------------------------------------------------------------------------------------------
    book = [["Observed all 1s", "TN=FP=0", "= recall", "1", "[0,1]", "[0,1]"], ["Observed all 0s", "TP=FN=0", "[0,1]", "0", "NaN", "NaN"],
            ["Predicted all 1s", "TN=FN=0", "= precision", "[0,1]", "1", "[0,1]"], ["Predicted all 0s", "TP=FP=0", "[0,1]", "NaN", "0", "NaN"]]
    rows = []
    for v in VENUES:
        t = meta(v)["walk_forward"]["table_14_1"]
        rows.append([v, fmt(t["tp"]), fmt(t["fp"]), fmt(t["tn"]), fmt(t["fn"]), fmt(t.get("accuracy"), 4), fmt(t.get("precision"), 4), fmt(t.get("recall"), 4), fmt(t.get("f1"), 4)])
    out.append(item("Table 14.1", "The four degenerate cases of binary classification, and our walk-forward confusion counts", "data",
                    table=tbl(["Condition", "Collapse", "Accuracy", "Precision", "Recall", "F1"], book),
                    text=None, note="Our counts (none of the four degenerate cases occurs): " + "; ".join(
                        f"{r[0]}: TP {r[1]}, FP {r[2]}, TN {r[3]}, FN {r[4]}, accuracy {r[5]}, precision {r[6]}, recall {r[7]}, F1 {r[8]}" for r in rows)))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch15():
    out = []
    m15 = {v: meta(v)["ch15"] for v in VENUES}
    out.append(item("Equation 54", "t-value of a precision p under H0: p = 1/2, theta[p, n] = (2p - 1) sqrt(n) / (2 sqrt(p(1-p)))", "data",
                    table=tbl(["Venue", "Bets n", "Observed precision p", "theta (t-value, Eq 54)"], [[v, fmt(m15[v]["n_bets"]), fmt(m15[v]["observed_precision"], 5), fmt(m15[v]["t_value_of_precision_eq54"], 3)] for v in VENUES])))

    p = np.linspace(0.4, 0.6, 81)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for n, c in zip((0, 25, 50, 75, 100), ("0.0", "0.25", "0.4", "0.55", "0.7")):
        ax.plot(p, (2 * p - 1) / (2 * np.sqrt(p * (1 - p))) * math.sqrt(n), color=c, label=f"n={n}")
    ax.set_xlabel("precision p"); ax.set_ylabel("Sharpe ratio theta"); ax.legend()
    out.append(item("Figure 15.1", "Precision (x) against Sharpe ratio (y) for several bet frequencies n (Equation 54)", "formula", fig=save(fig, "f15_1")))

    np.random.seed(15)
    mean, std, sr = PC.sharpe_of_binomial_draws(1000000, 0.55)
    theory = (2 * 0.55 - 1) / (2 * math.sqrt(0.55 * 0.45))
    out.append(item("Snippet 15.1", "Sharpe ratio targeted as a function of the number of bets: 1,000,000 draws of +1 / -1 with p = 0.55", "formula",
                    table=tbl(["", "mean", "std", "mean/std"], [["printed code, seed 15", fmt(mean, 5), fmt(std, 5), fmt(sr, 5)], ["theory (2p-1)/(2 sqrt(p(1-p)))", fmt(0.1, 5), fmt(2 * math.sqrt(0.55 * 0.45), 5), fmt(theory, 5)]]),
                    note="The book's run is unseeded (its printed numbers differ draw to draw); the theory row is the exact value."))
    out.append(item("Equation 55", "Precision needed for a Sharpe target theta*: p = 1/2 (1 + sqrt(1 - n/(theta*^2 + n)))", "formula",
                    table=tbl(["theta*", "n", "p"], [[t, n, fmt(0.5 * (1 + math.sqrt(1 - n / (t ** 2 + n))), 5)] for t in (1.0, 2.0) for n in (52, 260)])))
    out.append(item("Equation 56", "Sharpe ratio with asymmetric payouts theta[p, n, pi-, pi+]", "data",
                    table=tbl(["Venue", "mean profit pi+", "mean loss pi-", "n", "p", "Sharpe (Eq 56)"],
                              [[v, fmt(m15[v]["mean_positive_return"], 5), fmt(m15[v]["mean_nonpositive_return"], 5), fmt(m15[v]["n_bets"]), fmt(m15[v]["observed_precision"], 5), fmt(m15[v]["targets"]["1.0"]["sharpe_of_observed_precision_eq56"], 3)] for v in VENUES]),
                    text="Per-bet unit returns of the walk-forward bets that started in the last 365 days (side x exit/entry - 1 - round-trip cost); n is the number of those bets in the year.  Saved by the run."))
    out.append(item("Equation 57", "Implied precision from the quadratic: p = (-b + sqrt(b^2 - 4ac)) / 2a", "data",
                    table=tbl(["Venue", "theta* = 1", "theta* = 2"], [[v, fmt(m15[v]["targets"]["1.0"]["implied_precision_15_3"], 5), fmt(m15[v]["targets"]["2.0"]["implied_precision_15_3"], 5)] for v in VENUES]),
                    text="Same numbers as Snippet 15.3 (it codes Equation 57)."))
    err = PC.factor_variance_check()
    out.append(item("Snippet 15.2", "Symbolic factorisation of the binary-payout variance (SymPy)", "code",
                    text=f"SymPy is not installed in this environment (pip is blocked), so factor(m2 - m1^2) = p(1-p)(u-d)^2 is checked numerically on a grid by book_port_c.factor_variance_check: largest absolute difference {err:.2e}."))
    rows = []
    for v in VENUES:
        for t in ("1.0", "2.0"):
            rows.append([v, t, fmt(m15[v]["targets"][t]["implied_precision_15_3"], 5), fmt(m15[v]["observed_precision"], 5), "yes" if m15[v]["observed_precision"] >= m15[v]["targets"][t]["implied_precision_15_3"] else "no"])
    out.append(item("Snippet 15.3", "Implied precision binHR(sl, pt, freq, tSR) for our bets", "data",
                    table=tbl(["Venue", "Target Sharpe", "Implied precision", "Observed precision", "Observed >= implied"], rows),
                    text="sl = mean non-positive bet return, pt = mean positive bet return, freq = bets in the year (the bot's own inputs, saved in ch15.json)."))

    pt = 0.01
    freqs = np.arange(10, 100, 2)
    sls = -np.linspace(0.001, 0.01, 40)
    X, Y = np.meshgrid(freqs, sls)
    Z = np.array([[PC.binHR(s, pt, f, 1.5) for f in freqs] for s in sls])
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    heat(ax, X, Y, Z, "frequency n", "stop loss", "implied precision, pi+ = 0.01, theta* = 1.5")
    out.append(item("Figure 15.2", "Heat-map of the implied precision as a function of n and pi-", "formula", fig=save(fig, "f15_2"),
                    note="The caption prints pi+ = 0.1; with pi+ = 0.1 the colour range does not match the printed figure (0.15 to 0.6), with pi+ = 0.01 it does (and Figure 15.3's 3 to 18), so 0.01 is used (assumption)."))
    rows = []
    for v in VENUES:
        for t in ("1.0", "2.0"):
            f_ = m15[v]["targets"][t]["implied_frequency_15_4_at_observed_precision"]
            rows.append([v, t, fmt(f_, 3) if f_ is not None else "none (extraneous root or no solution)"])
    out.append(item("Snippet 15.4", "Implied betting frequency binFreq(sl, pt, p, tSR) at our observed precision", "data",
                    table=tbl(["Venue", "Target Sharpe", "Bets per year needed"], rows),
                    text="The snippet returns nothing when the root is extraneous (the check binSR == tSR fails); at our precisions the target is not reachable."))
    ps = np.linspace(0.5, 0.59, 37)
    sls = -np.linspace(0.0014, 0.005, 36)
    X, Y = np.meshgrid(ps, sls)
    Z = np.array([[(lambda r: np.nan if r is None else r)(PC.binFreq(s, pt, q, 1.5)) for q in ps] for s in sls])
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    heat(ax, X, Y, Z, "precision p", "stop loss", "implied frequency, pi+ = 0.01, theta* = 1.5")
    out.append(item("Figure 15.3", "Implied frequency as a function of p and pi-", "formula", fig=save(fig, "f15_3"), note="Same pi+ assumption as Figure 15.2."))

    np.random.seed(0)
    mu1, mu2, s1, s2, pr1, nobs, tsr, freq = .05, -.1, .05, .1, .75, 2600, 2., 260
    ret = PC.mixGaussians(mu1, mu2, s1, s2, pr1, nobs)
    thres = PC.binHR(ret[ret <= 0].mean(), ret[ret > 0].mean(), freq, tsr)
    risk = PC.probFailure(ret, freq, tsr)
    rows = [["book's example (mu 0.05 / -0.10, sigma 0.05 / 0.10, 75% / 25%, 2600 obs, 260 bets/yr, SR 2), seed 0", fmt(thres, 5), fmt(risk, 5)]]
    for v in VENUES:
        rows.append([f"{v} walk-forward bets, target 2", fmt(m15[v]["targets"]["2.0"]["implied_precision_15_3"], 5), fmt(m15[v]["targets"]["2.0"]["prob_failure_15_5"], 5)])
    out.append(item("Snippet 15.5", "Strategy risk: probability that the precision falls below the implied threshold", "data",
                    table=tbl(["Case", "Threshold precision", "P[fail]"], rows),
                    text="book_port_c.mixGaussians / probFailure / main_15_5.  The printed risk uses the book's approximation ss.norm.cdf(thresP, p, p(1-p))."))
    return out


def build():
    items = []
    for f in (ch14, ch15):
        print("building", f.__name__, flush=True)
        items += f()
    return items


if __name__ == "__main__":
    its = build()
    json.dump(its, open(W.OUT / "items_ch14_15.json", "w"), indent=1, default=str)
    json.dump([[a, None if b is None else float(b), None if c is None else float(c)] for a, b, c in CHECKS], open(W.OUT / "checks_ch14_15.json", "w"), indent=1)
    bad = [c for c in CHECKS if c[1] is None or c[2] is None or not np.isclose(c[1], c[2], rtol=1e-6, atol=1e-9, equal_nan=True)]
    print(len(its), "items;", len(CHECKS), "checks;", len(bad), "mismatches", bad)
