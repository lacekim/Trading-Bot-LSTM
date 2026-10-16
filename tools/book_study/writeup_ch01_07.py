"""Write-up items for Chapters 1-7 of the book (see writeup_core.py for the item format).  New file."""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import book_port_a as PA
import book_port_b as PB
import book_port_e as E
import time_decay as TD
import writeup_core as W
from writeup_core import VENUES, SYM, item, tbl, fmt, save


def meta(v):
    return W.run_json(v, "meta.json")


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch1():
    out = []
    X = "X"
    book = [[1, 2, X, X, "", "", "", ""], [1, 3, X, X, "", "", "", ""], [1, 4, X, X, "", "", "", ""], [1, 5, X, X, "", X, "", ""], [2, 6, "", X, "", "", "", ""], [2, 7, "", X, "", "", X, X],
            [2, 8, "", X, "", "", X, ""], [2, 9, "", X, "", "", X, ""], [3, 10, "", X, "", "", X, ""], [3, 11, "", X, "", X, "", X], [3, 12, "", X, "", X, "", X], [3, 13, "", X, "", X, "", X],
            [3, 14, "", X, "", X, "", X], [3, 15, "", X, "", X, "", X], [3, 16, "", X, "", X, X, X], [4, 17, X, X, "", X, "", ""], [4, 18, X, X, "", X, "", ""], [4, 19, X, X, "", "", "", ""],
            [5, 20, "", X, X, X, "", ""], [5, 21, "", X, X, X, "", ""], [5, 22, "", X, X, X, "", ""]]
    where = {2: "build_dollar_bars.py, book_stats.cusum_events", 3: "book_events_v2.py, book_port_e.py", 4: "bar_events.attribution_weights_t, book_port_a.py, time_decay.py", 5: "fracdiff_core.py, book_port_e.py",
             6: "meta_model.py (bagged entropy forest)", 7: "book_stats.train_mask, book_port_b.py", 8: "book_ch8_ch9.py", 9: "book_ch8_ch9.py", 10: "bet_averaging.py, book_port_c.py", 11: "meta_eval.py", 12: "meta_eval.py (CPCV N=6, k=2)",
             13: "book_stats.py, book_otr_exit.py", 14: "book_stats.py, book_port_c.py", 15: "book_port_c.py", 16: "hrp_alloc.py, book_final.py", 17: "micro_features.py (SADF, CSW)", 18: "micro_features.py (entropy)",
             19: "book_micro.py (Corwin-Schultz, Roll, lambdas)", 20: "book_mp.py", 21: "book_port_d.py (Snippets 21.1-21.7)", 22: "no code (survey chapter)"}
    out.append(item("Table 1.1", "Overview of the challenges addressed by every chapter (the book's X marks), and where each chapter lives in our code", "text",
                    table=tbl(["Part", "Chapter", "Fin. data", "Software", "Hardware", "Math", "Meta-Strat", "Overfitting", "Where in our code"], [r + [where[r[1]]] for r in book]),
                    note="The X marks are the book's. The last column is ours."))
    ours = {1: "Not met as the book's row states it: our labels are side-free (getEvents with side=None, Snippets 3.3 and 3.5), so the forest learns side and size together.",
            2: "Done: MDI, MDA, SFI and orthogonal features on the 12-feature set.", 3: "Done: dollar bars (Section 2.2). Imbalance and run bars (Section 2.3): equations only in the book, no code, not computed.", 4: "Done: fractionally differentiated (FFD) feature, minimum d per symbol that passes the ADF test.",
            5: "Done: triple barrier, side-free labels (bin = sign of the return).", 6: "Not met, as row 1: side and size are learned together.", 7: "Done: average-uniqueness weights, sequential bootstrap, Snippet 3.8 dropLabels.",
            8: "Done: purging and embargo in every split (embargo 1%).", 9: "Done: walk-forward (cutoff 2024-01-01) and CPCV (N = 6, k = 2: 15 splits, 5 paths).", 10: "Done: Chapter 13 simulation, deflated Sharpe ratio with 11 trials."}
    pit = [[1, "Epistemological", "The Sisyphus paradigm", "The meta-strategy paradigm", "1"], [2, "Epistemological", "Research through backtesting", "Feature importance analysis", "8"], [3, "Data processing", "Chronological sampling", "The volume clock", "2"],
           [4, "Data processing", "Integer differentiation", "Fractional differentiation", "5"], [5, "Classification", "Fixed-time horizon labeling", "The triple-barrier method", "3"], [6, "Classification", "Learning side and size simultaneously", "Meta-labeling", "3"],
           [7, "Classification", "Weighting of non-IID samples", "Uniqueness weighting; sequential bootstrapping", "4"], [8, "Evaluation", "Cross-validation leakage", "Purging and embargoing", "7, 9"],
           [9, "Evaluation", "Walk-forward (historical) backtesting", "Combinatorial purged cross-validation", "11, 12"], [10, "Evaluation", "Backtest overfitting", "Backtesting on synthetic data; the deflated Sharpe ratio", "10-16"]]
    out.append(item("Table 1.2", "Common pitfalls in financial ML (the book's ten rows), with whether our bot avoids each", "text", table=tbl(["#", "Category", "Pitfall", "Solution", "Chapter", "Our bot"], [r + [ours[r[0]]] for r in pit]),
                    note="Rows 1 and 6 are not met as the book states them: our labels are side-free."))
    return out



# --------------------------------------------------------------------------------------------------------------------------------------------------
def _bars_per_day():
    rows = {}
    for v in VENUES:
        tot = {}
        for s in W.symbols(v):
            d = pd.read_csv(W.BARS[v] / "bars" / f"{s}.csv", usecols=["date", "n_rows"], keep_default_na=False, na_values=[""])
            d["date"] = pd.to_datetime(d["date"])
            g = d.groupby(d["date"].dt.year).agg(bars=("date", "size"), rows=("n_rows", "sum"), days=("date", lambda x: x.dt.floor("D").nunique()))
            for y, r in g.iterrows():
                a = tot.setdefault(y, [0, 0, 0])
                a[0] += r.bars; a[1] += r.rows; a[2] += r.days
        rows[v] = pd.DataFrame(tot, index=["bars", "rows", "days"]).T
    return rows


def ch2():
    out = []
    out.append(item("Table 2.1", "The four types of financial data, and which our bot uses", "text",
                    table=tbl(["Fundamental", "Market", "Analytics", "Alternative"], [["Not used", "Used: price and volume from Kraken 1-minute rows and Alpaca hourly bars, high / low for the reference features", "Not used", "Not used"]])))
    bpd = _bars_per_day()

    def draw(ax, v):
        d = bpd[v]
        ax.plot(d.index, d["bars"] / d["days"], color="0.15", label="dollar bars per symbol-day")
        ax.plot(d.index, d["rows"] / d["days"], color="0.6", ls=":", label="source rows per symbol-day")
        ax.set_yscale("log"); ax.legend(); ax.set_xlabel("Year")
    out.append(item("Figure 2.1", "Average daily frequency of bars (we use dollar bars only; tick and volume bars are not built)", "data", fig=W.two_panel(draw, "f2_1"),
                    note="The book compares tick, volume and dollar bars on one futures contract; our source rows are 1-minute (Kraken) and hourly (Alpaca) so tick and volume bars cannot be built from them."))
    # Section 2.3 (Equations 1-15): imbalance and run bars; Equations 16-18, Expressions 1-2 and Equations 19-21: the ETF trick (Section 2.4.1)
    rows = []
    for v in VENUES:
        bb = W.bars(v)
        d = np.sign(np.diff(bb.close))
        s = pd.Series(d).replace(0, np.nan).ffill()
        rows.append([v, SYM[v], fmt(len(d)), fmt(float((s > 0).mean()), 4), fmt(float((s < 0).mean()), 4), fmt(float((d == 0).mean()), 4)])
    NB = "Status: imbalance and run bars (Section 2.3) are not built.  The book defines b_t on ticks (trades with an aggressor side); our sources are Kraken 1-minute rows and Alpaca hourly bars, so the bot uses plain dollar bars (Section 2.2)."
    out.append(item("Equation 1", "The tick rule: b_t = b_(t-1) if dp_t = 0, else |dp_t| / dp_t", "data",
                    table=tbl(["Venue", "Symbol", "Dollar-bar changes", "share b = +1", "share b = -1", "share with dp = 0"], rows), text="Applied to the closes of our dollar bars (not to ticks); b_t is carried forward when the price does not change."))
    eq1 = {2: ("Tick imbalance theta_T = sum_(t=1)^T b_t", "A tick-imbalance bar closes when |theta_T| reaches its expected size."),
           3: ("Tick-imbalance bar length T* = argmin_T { |theta_T| >= E0[T] |2 P[b_t = 1] - 1| }", "E0[T] and P[b_t = 1] are exponentially weighted averages of earlier bars."),
           4: ("Volume / dollar imbalance theta_T = sum_(t=1)^T b_t v_t", "Same with each tick weighted by its volume (or dollar volume) v_t."),
           5: ("E0[theta_T] = E0[ sum over b_t = 1 of v_t ] - E0[ sum over b_t = -1 of v_t ] = E0[T] (P[b_t = 1] E0[v_t | b_t = 1] - P[b_t = -1] E0[v_t | b_t = -1])", "Expected volume imbalance."),
           6: ("v+ = P[b_t = 1] E0[v_t | b_t = 1],  v- = P[b_t = -1] E0[v_t | b_t = -1]", "Definitions of the buy and sell volume components."),
           7: ("E0[T]^-1 E0[ sum_t v_t ] = E0[v_t] = v+ + v-", "The two components add up to the unconditional expected volume."),
           8: ("E0[theta_T] = E0[T] (v+ - v-) = E0[T] (2 v+ - E0[v_t])", "Expected volume imbalance in terms of v+."),
           9: ("Volume-imbalance bar length T* = argmin_T { |theta_T| >= E0[T] |2 v+ - E0[v_t]| }", "The closing rule of a volume (or dollar) imbalance bar."),
           10: ("Tick runs theta_T = max{ sum over b_t = 1 of b_t, - sum over b_t = -1 of b_t }", "The length of the longest run of one sign."),
           11: ("E0[theta_T] = E0[T] max{ P[b_t = 1], 1 - P[b_t = 1] }", "Expected tick-run length."),
           12: ("Tick-run bar length T* = argmin_T { theta_T >= E0[T] max{ P[b_t = 1], 1 - P[b_t = 1] } }", "The closing rule of a tick-run bar."),
           13: ("Volume runs theta_T = max{ sum over b_t = 1 of b_t v_t, - sum over b_t = -1 of b_t v_t }", "Volume-weighted version of Equation 10."),
           14: ("E0[theta_T] = E0[T] max{ P[b_t = 1] E0[v_t | b_t = 1], (1 - P[b_t = 1]) E0[v_t | b_t = -1] }", "Expected volume run."),
           15: ("Volume-run bar length T* = argmin_T { theta_T >= E0[T] max{ P[b_t = 1] E0[v_t | b_t = 1], (1 - P[b_t = 1]) E0[v_t | b_t = -1] } }", "The closing rule of a volume (dollar) run bar.")}
    for n in range(2, 16):
        out.append(item(f"Equation {n}", eq1[n][0], "formula", text=eq1[n][1] + "  " + NB))
    out.append(item("Expression 1", "The expected volume-run term max{ P[b_t = 1] E0[v_t | b_t = 1], (1 - P[b_t = 1]) E0[v_t | b_t = -1] }", "formula", text="The factor inside Equations 14 and 15.  " + NB))
    etf = {16: "Holdings of the virtual ETF: h_(i,t) = omega_(i,t) K_t / (o_(i,t+1) phi_(i,t) sum_i |omega_(i,t)|) on rebalance bars t in B, h_(i,t-1) otherwise",
           17: "Price change of instrument i: delta_(i,t) = p_(i,t) - o_(i,t) if t-1 is in B, dp_(i,t) otherwise",
           18: "Value of the virtual ETF: K_t = K_(t-1) + sum_i h_(i,t-1) phi_(i,t) (delta_(i,t) + d_(i,t))"}
    for n in (16, 17, 18):
        out.append(item(f"Equation {n}", etf[n], "formula", text="The book's accounting for a virtual ETF built from a roll-adjusted futures basket.  We trade spot symbols, not a basket held as one instrument, so K_t is not used; the three rules at the end of Section 2.4.1 (Equations 19-21) are applied instead."))
    sums = []
    for v in VENUES:
        hw = pd.read_csv(W.RUNS[v] / "hrp_weights.csv", index_col=0)
        s = hw.sum(axis=1)
        sums.append([v, fmt(len(hw)), fmt(int(hw.notna().sum(axis=1).min())), fmt(int(hw.notna().sum(axis=1).max())), fmt(float(s.min()), 6), fmt(float(s.max()), 6)])
    out.append(item("Expression 2", "Normalised weights omega_(i,t) / sum_i |omega_(i,t)|", "data",
                    table=tbl(["Venue", "Rebalances", "Fewest symbols", "Most symbols", "Smallest sum of weights", "Largest sum of weights"], sums),
                    text="Our HRP weights are long-only and already sum to 1, so the normalisation is the identity."))
    rc = []
    for v in VENUES:
        r = meta(v)["rebalance_cost_effect"]
        for k, nm in (("last_365_days", "last 365 days"), ("whole_walk_forward_span", "whole walk-forward span")):
            rc.append([v, nm, fmt(r[k]["rebalances_charged"]), fmt(r[k]["gain_pct_before"], 4), fmt(r[k]["gain_pct_after"], 4), fmt(r[k]["total_charge_pct_of_capital"], 4)])
    out.append(item("Equation 19", "Rebalance costs c_t = sum_i (|h_(i,t-1)| p_(i,t) + |h_(i,t)| o_(i,t+1)) phi_(i,t) tau_i on every rebalance bar t in B", "data",
                    table=tbl(["Venue", "Span", "Rebalances charged", "Gain % before", "Gain % after", "Total charge (% of capital)"], rc),
                    text="B = the quarterly HRP rebalance dates; the charge is taken literally (the whole old allocation sold, the whole new one bought) at the venue's cost per side.  Numbers saved by the run (rebalance_cost_effect)."))
    out.append(item("Equation 20", "Bid-ask spread cost c~_t = sum_i |h_(i,t-1)| p_(i,t) phi_(i,t) tau_i of buying or selling one unit", "data",
                    table=tbl(["Venue", "Cost per side tau (bp)"], [["Kraken", fmt(W.COST["Kraken"], 1)], ["Alpaca", fmt(W.COST["Alpaca"], 1)]]),
                    text="The account already charges tau per side on every change of every symbol's averaged position, so the spread is inside the cost and nothing is added."))
    vc = []
    for v in VENUES:
        c = meta(v)["volume_capacity"]
        sn = c["share_not_fitting"]
        vc.append([v, fmt(c["n_trades"]), fmt(c["capacity_min"], 0), fmt(c["capacity_p01"], 0), fmt(c["capacity_median"], 0), f"{100 * sn['10000']:.2f}%", f"{100 * sn['100000']:.3f}%", f"{100 * sn['1000000']:.2f}%", f"{100 * sn['10000000']:.2f}%"])
    out.append(item("Equation 21", "Tradeable basket units v_t = min_i v_(i,t) / |h_(i,t-1)|: how much capital a trade can use before it exceeds its bar's volume", "data",
                    table=tbl(["Venue", "Trades", "Capacity min (USD)", "p01", "median", "share not fitting at 10k", "at 100k", "at 1M", "at 10M"], vc),
                    text="For each trade the capacity is p v / |de| (price x the bar's volume / the trade's change in exposure per unit of capital): the largest account for which the trade fits inside its own bar.  Computed over every walk-forward trade of the run."))
    # PCA weights, Equations 19-25 and Figure 2.2
    px = {}
    for v in VENUES:
        sy = W.symbols(v)
        s = {x: W.daily_close(v, x) for x in sy}
        df = pd.DataFrame(s)
        df = df[df.index >= df.index.max() - pd.Timedelta(days=730)]
        df = df[df.notna().mean(axis=1) > 0.6]                    # trading days
        df = df.loc[:, df.notna().mean() > 0.98]
        if v == "Alpaca":
            df = df.iloc[:, :30]
        px[v] = np.log(df).diff().dropna(how="any")

    def draw22(ax, v):
        r = px[v]
        cov = r.cov().values
        eVal, eVec = np.linalg.eigh(cov)
        idx = eVal.argsort()[::-1]
        eVal, eVec = eVal[idx], eVec[:, idx]
        ivp = 1 / np.diag(cov); ivp /= ivp.sum()
        w_pca = PA.pcaWeights(cov).ravel()
        def contrib(w):
            beta = eVec.T @ w
            c = beta ** 2 * eVal
            return c / c.sum()
        k = min(10, len(eVal))
        x = np.arange(1, k + 1)
        ax.bar(x - 0.2, contrib(ivp)[:k], 0.4, color="0.3", label="Inverse variance portfolio")
        ax.bar(x + 0.2, contrib(w_pca)[-k:][::-1][::-1] if False else contrib(w_pca)[:k], 0.4, color="0.7", label="PCA portfolio (all risk on the last component)")
        ax.set_xlabel("Principal component"); ax.set_ylabel("Risk contribution"); ax.legend()
        ax.set_title(f"{v}: {r.shape[1]} symbols, {len(r)} days (first 10 components shown)")
    f = W.two_panel(draw22, "f2_2", titles={v: "" for v in VENUES})
    out.append(item("Figure 2.2", "Risk contribution per principal component: inverse-variance versus PCA weights (Snippet 2.1)", "data", fig=f,
                    note="Book: 10 futures. Ours: daily log returns of the symbols with a full last two years of daily closes (all 27 Kraken symbols with data, the first 30 Alpaca symbols). With riskDist as printed, the PCA portfolio puts all risk on the last (smallest) component, which sits beyond the first 10 for these larger universes, so its bars are zero here; the IVP bars show the book's point."))
    pca = {22: ("Portfolio variance in the eigenvector basis: sigma^2 = omega' V omega = omega' W Lambda W' omega = beta' Lambda beta = (Lambda^(1/2) beta)' (Lambda^(1/2) beta)", "V = covariance matrix, W = eigenvectors, Lambda = eigenvalues, beta = W' omega."),
           23: ("sigma^2 = sum_(n=1)^N beta_n^2 Lambda_(n,n)", "The variance is a sum over the principal components."),
           24: ("Risk share of component n: R_n = beta_n^2 Lambda_(n,n) sigma^-2 = [W' omega]_n^2 Lambda_(n,n) sigma^-2", "The R_n sum to 1."),
           25: ("beta = { sigma sqrt(R_n / Lambda_(n,n)) } for n = 1..N", "Given the risk distribution R the weights are omega = W beta (Snippet 2.1, book_port_a.pcaWeights; Figure 2.2 on our symbols).")}
    for n in range(22, 26):
        out.append(item(f"Equation {n}", pca[n][0], "formula", text=pca[n][1]))
    out.append(item("Snippet 2.1", "PCA weights from a risk distribution", "code", text="book_port_a.pcaWeights; test_book_port_a.test_pca_weights_hit_the_risk_target_and_distribution passes."))
    out.append(item("Snippet 2.2", "Futures roll gaps", "code", text="book_port_a.rollGaps; tested on a hand example.  Not applicable to spot Kraken / Alpaca series (no rolls)."))
    out.append(item("Snippet 2.3", "Rolled futures series", "code", text="book_port_a.getRolledSeries / rolled_prices; tested on a hand example.  Not applicable to spot series."))
    # CUSUM
    cu = {"26": ("CUSUM recursion S_t = max{ 0, S_(t-1) + y_t - E_(t-1)[y_t] }", "One-sided filter that accumulates upward departures of y_t from its expectation."),
          "27": ("S_t >= h  if and only if  there is a tau in [1, t] with sum_(i=tau)^t (y_i - E_(i-1)[y_t]) >= h", "The trigger: the run of departures since some earlier time reaches the threshold h."),
          "28": ("Upward sum S+_t = max{ 0, S+_(t-1) + y_t - E_(t-1)[y_t] }, S+_0 = 0", "Symmetric version, upper side."),
          "29a": ("Downward sum S-_t = min{ 0, S-_(t-1) + y_t - E_(t-1)[y_t] }, S-_0 = 0", "Symmetric version, lower side."),
          "29b": ("S_t = max{ S+_t, -S-_t }: a bar is sampled when S_t >= h, then both sums reset to 0", "Snippet 2.4 (book_port_e.getTEvents) is this loop; the bot's vectorised form is book_stats.cusum_events.")}
    for n, (tt, tx) in cu.items():
        out.append(item(f"Equation {n}", tt, "formula", text=tx + ("  Figure 2.3 shows the events on our bars." if n == "29b" else "")))

    def draw23(ax, v):
        b = W.bars(v)
        ev = W.run_csv(v, "events.csv", usecols=["symbol", "pos", "t0"])
        e = ev[ev.symbol == SYM[v]]
        t_hi = e["t0"].iloc[len(e) // 2]
        lo = int(np.searchsorted(b.t, t_hi))
        hi = min(lo + 400, b.n)
        ax.plot(b.idx[lo:hi], b.close[lo:hi], color="0.5", lw=0.8, label="Price (dollar bars)")
        pos = e["pos"].to_numpy(); pos = pos[(pos >= lo) & (pos < hi)]
        ax.plot(b.idx[pos], b.close[pos], ".", color="0.1", ms=4, label="Sampled observations")
        ax.set_ylabel("Price"); ax.set_xlabel("Time"); ax.legend(); ax.tick_params(axis="x", rotation=30)
    out.append(item("Snippet 2.4", "The symmetric CUSUM filter", "code", text="book_port_e.getTEvents is the printed loop; book_stats.cusum_events is the vectorised version used in the bot and is tested against it (test_book_port_e.test_matches_project_cusum).  The threshold h is not given by the book; ours is a causal one-bar sigma x sqrt(bars per day)."))
    out.append(item("Figure 2.3", "CUSUM sampling of a price series (400 bars around the middle of each symbol's events)", "data", fig=W.two_panel(draw23, "f2_3")))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch3():
    out = []
    out.append(item("Equation 30", "Fixed-horizon label: -1, 0, 1 by comparing the return to a threshold tau", "formula", text="The book's label that Chapter 3 replaces with the triple barrier."))
    out.append(item("Equation 31", "Return from t_i0 to t_i0 + h", "formula", text="r = p(t1)/p(t0) - 1; used for ret in Snippets 3.5-3.7."))
    out.append(item("Snippet 3.1", "Daily volatility estimates", "code", text="book_vol_printed.daily_vol_printed is the printed code vectorised; test_book_vol_printed checks it against the verbatim port (book_port_e.getDailyVol) to 1e-12 on both venues' bars.  Earlier project code deviated from it (tie rule, hole mask); fixed in book_events_v2."))
    out.append(item("Snippet 3.2", "Triple-barrier labeling", "code", text="book_port_e.applyPtSlOnT1 (verbatim).  book_parity.py runs the verbatim snippets on our bars and our events: exit times and labels match 1,500 of 1,500 events per symbol checked (BTC and ETH, Kraken)."))

    def draw31(axs, v):
        ev = W.run_csv(v, "events.csv"); b = W.bars(v)
        e = ev[(ev.symbol == SYM[v]) & (ev.t0 > 1.5e9)]
        for ax, kind, lab in ((axs[0], 1, "(a) a horizontal barrier is hit first"), (axs[1], 0, "(b) the vertical barrier is hit first")):
            r = e[e["kind"] == kind].iloc[3]
            p0, p1 = int(r.pos), int(r.t1pos)
            lo, hi = max(0, p0 - 80), min(b.n, p1 + 40)
            ax.plot(b.idx[lo:hi], b.close[lo:hi], color="0.3", lw=0.8)
            c0 = b.close[p0]; up, dn = c0 * (1 + 2 * r.trgt), c0 * (1 - 2 * r.trgt)
            ax.plot([b.idx[p0], b.idx[p1]], [up, up], ls=":", color="0.2"); ax.plot([b.idx[p0], b.idx[p1]], [dn, dn], ls=":", color="0.2")
            ax.plot([b.idx[p1], b.idx[p1]], [dn, up], ls=":", color="0.2"); ax.plot([b.idx[p0], b.idx[p0]], [dn, up], ls=":", color="0.2")
            ax.set_xlabel(f"Time  {lab}"); ax.set_ylabel("Price"); ax.tick_params(axis="x", rotation=30)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for j, v in enumerate(VENUES):
        draw31([axs[0, j], axs[1, j]], v)
        axs[0, j].set_title(f"{v} ({SYM[v]})")
    fig.tight_layout()
    out.append(item("Figure 3.1", "Two configurations of the triple barrier: horizontal first, vertical first (barriers at 2 x daily vol, vertical at 4 days)", "data", fig=save(fig, "f3_1")))
    out.append(item("Snippet 3.3", "Triple-barrier events (getEvents)", "code", text="book_port_e.getEvents_3_3 / getEvents (verbatim)."))
    out.append(item("Snippet 3.4", "Vertical barrier (addVerticalBarrier)", "code", text="book_port_e.addVerticalBarrier (verbatim)."))
    out.append(item("Snippet 3.5", "Labeling for side and size (getBins)", "code", text="book_port_e.getBins_3_5 (verbatim)."))
    out.append(item("Snippet 3.6", "Expanded getEvents with side", "code", text="book_port_e.getEvents (side=None gives the side-free label used by the bot)."))
    out.append(item("Snippet 3.7", "Expanded getBins (meta-labeling)", "code", text="book_port_e.getBins; the bot uses case 1 (bin = sign of the return)."))

    def draw32(ax, v):
        t = meta(v)["walk_forward"]["table_14_1"]
        m = np.array([[t["tp"], t["fp"]], [t["fn"], t["tn"]]])
        ax.imshow(m, cmap="Greys", vmin=0, vmax=m.max() * 1.6)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{['TP','FP','FN','TN'][2*i+j]}\n{m[i,j]:,}", ha="center", va="center", color="0.1" if m[i, j] < m.max() * 0.6 else "w")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["predicted +", "predicted -"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["observed +", "observed -"])
    out.append(item("Figure 3.2", "The confusion matrix on our walk-forward test events", "data", fig=W.two_panel(draw32, "f3_2", w=6, h=2.6)))
    out.append(item("Snippet 3.8", "dropLabels", "code", text="book_port_a.dropLabels (verbatim); applied after the weights, with minPct = 0.05.  Dropped events: Kraken 49 of 64,028; Alpaca 723 of 352,728 (meta.json: events_before_dropLabels)."))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch4():
    out = []
    out.append(item("Snippet 4.1", "Number of concurrent events per bar", "code", text="book_port_e.mpNumCoEvents (verbatim); matches the bot's weights exactly on BTC (book_parity: uniqueness max|diff| 1e-12 over 1,200 events)."))
    out.append(item("Equation 32", "Average uniqueness of an event", "formula", text="u_i = sum(1_{t,i} / c_t) / sum(1_{t,i}); our events.csv column u."))

    def draw41(ax, v):
        u = W.run_csv(v, "events.csv", usecols=["u"])["u"]
        ax.hist(u, bins=40, color="0.35"); ax.set_xlabel("Uniqueness"); ax.set_ylabel("Events")
    out.append(item("Figure 4.1", "Histogram of uniqueness values of our events", "data", fig=W.two_panel(draw41, "f4_1", titles={v: v for v in VENUES})))
    out.append(item("Snippet 4.2", "Average uniqueness over each event's life", "code", text="book_port_e.getTW / mpSampleTW (verbatim)."))
    out.append(item("Equation 33", "Probability of drawing event j on the second draw: delta_j^(2) = u-bar_j^(2) (sum_(k=1)^I u-bar_k^(2))^-1", "formula", text="Sequential bootstrap (Snippets 4.3-4.5): the chance of drawing an event is proportional to its average uniqueness given the draws already made."))
    out.append(item("Snippet 4.3", "Indicator matrix", "code", text="book_port_a.getIndMatrix (verbatim)."))
    out.append(item("Snippet 4.4", "Average uniqueness from the indicator matrix", "code", text="book_port_a.getAvgUniqueness (verbatim)."))
    out.append(item("Snippet 4.5", "Sequential bootstrap", "code", text="book_port_a.seqBootstrap (verbatim); the bot's forest uses a faster implementation, checked against it."))
    ind = PA.getIndMatrix(range(6), pd.Series([2, 3, 5], index=[0, 2, 4]))
    out.append(item("Matrix 1", "Indicator matrix of the book's three-event example (Snippet 4.6's t1)", "formula", table=tbl(["bar"] + [f"event {c}" for c in ind.columns], [[i] + [int(x) for x in r] for i, r in ind.iterrows()])))
    out.append(item("Snippet 4.6", "Example of sequential bootstrap", "code", text="book_port_a.example_seq_bootstrap; test_book_port_a.test_seq_bootstrap_beats_standard_on_average_and_has_the_right_size passes."))
    out.append(item("Snippet 4.7", "Generating a random t1 series", "code", text="book_port_a.getRndT1 (verbatim)."))
    out.append(item("Snippet 4.8", "Uniqueness from standard and sequential bootstrap", "code", text="book_port_a.auxMC."))
    out.append(item("Snippet 4.9", "Multi-threaded Monte Carlo", "code", text="book_port_a.mainMC; book_mp.py runs it with processes."))
    # Fig 4.2: Monte Carlo
    rng_iters = 400
    np.random.seed(0)
    import random
    random.seed(0)
    std, seq = [], []
    for _ in range(rng_iters):
        r = PA.auxMC(10, 100, 5)
        std.append(r["stdU"]); seq.append(r["seqU"])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.hist(std, bins=24, color="0.75", alpha=0.9, label="Standard"); ax.hist(seq, bins=24, color="0.3", alpha=0.7, label="Sequential")
    ax.set_xlabel("Average uniqueness"); ax.set_ylabel("Frequency"); ax.legend()
    out.append(item("Figure 4.2", f"Monte Carlo experiment, standard vs sequential bootstrap (the book's own setup: 10 observations, 100 bars, maxH 5; {rng_iters} iterations, not 1e6)", "formula", fig=save(fig, "f4_2"),
                    table=tbl(["", "mean", "std"], [["standard", fmt(np.mean(std)), fmt(np.std(std))], ["sequential", fmt(np.mean(seq)), fmt(np.std(seq))]]),
                    note="Same inputs as the book, fewer iterations (a pure-Python run of 1e6 is not practical here), so the histogram is a sample of the book's, not identical."))
    out.append(item("Function 1", "Sample weight scaled to sum to the number of events", "formula", text="w_i = w~_i * I / sum(w~); events.csv column w (and w_raw before scaling)."))
    out.append(item("Snippet 4.10", "Sample weight by absolute return attribution", "code", text="book_port_e.mpSampleW / getSampleW (verbatim); matches the bot's w_raw up to the scaling (ratio constant 43.999 on BTC)."))
    out.append(item("Section 4.7", "Time decay", "text", text="Piecewise-linear decay by cumulative uniqueness (Snippet 4.11). The snippet's default c = 1 means no decay; the bot uses the default (any other value needs a refit per split and is refused by the code)."))
    out.append(item("Snippet 4.11", "Piecewise-linear decay factors", "code", text="book_port_e.getTimeDecay and time_decay.decay_factors (agree: test_time_decay)."))

    def draw43(ax, v):
        e = W.run_csv(v, "events.csv", usecols=["symbol", "t0", "u"])
        e = e[e.symbol == SYM[v]].sort_values("t0")
        for c, ls in ((1.0, ":"), (0.75, "--"), (0.5, "-."), (0.0, "-"), (-0.25, "-"), (-0.5, "-")):
            f = TD.decay_factors(e["u"].to_numpy(), e["t0"].to_numpy(), c)
            ax.plot(pd.to_datetime(e["t0"], unit="s"), f, ls=ls, color="0.2" if c >= 0 else "0.6", label=f"c={c}")
        ax.set_xlabel("Time"); ax.legend(ncol=2)
    out.append(item("Figure 4.3", "Piecewise-linear time-decay factors for c = 1, .75, .5, 0, -.25, -.5 on our events", "data", fig=W.two_panel(draw43, "f4_3")))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch5():
    out = []
    out.append(item("Section 5.4", "The method: fractional differentiation as a binomial series of the backshift operator", "text", text="X~_t = sum_k w_k X_(t-k) with w_0 = 1 and w_k = -w_(k-1) (d - k + 1) / k (Snippet 5.1; Figures 5.1 and 5.2).  Our feature uses the fixed-width window version (Snippet 5.3) with the smallest d, from Snippet 5.4's table, whose ADF statistic passes at 5%."))
    d = np.arange(0, 1.01, 0.01)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for dd, ls in zip((0, .25, .5, .75, 1.0), (":", "--", "-.", "-", "-")):
        w = PA.getWeights(dd, 6).ravel()
        axs[0].plot(range(6), w, ls=ls, color="0.2", label=f"d={dd}")
    for dd, ls in zip((1.0, 1.25, 1.5, 1.75, 2.0), (":", "--", "-.", "-", "-")):
        w = PA.getWeights(dd, 6).ravel()
        axs[1].plot(range(6), w, ls=ls, color="0.2", label=f"d={dd}")
    axs[0].legend(); axs[1].legend(); axs[0].set_xlabel("k"); axs[0].set_ylabel("weight"); axs[1].set_xlabel("k")
    p = save(fig, "f5_1_2")
    out.append(item("Figure 5.1", "Weights w_k as k grows, d in [0,1] (left panel)", "formula", fig=p, note="Figures 5.1 and 5.2 are closed-form (Snippet 5.1); same picture as the book. Left panel is Figure 5.1, right panel is Figure 5.2."))
    out.append(item("Figure 5.2", "Weights w_k as k grows, d in [1,2] (right panel of the figure above)", "formula", fig=p))
    out.append(item("Snippet 5.1", "Weighting function", "code", text="book_port_a.getWeights / fracdiff_core; test_book_port_e.test_ffd_weights_are_binomial passes."))
    out.append(item("Snippet 5.2", "Expanding-window fractional differentiation", "code", text="book_port_a.fracDiff (verbatim)."))
    # Fig 5.3, 5.4: with the per-symbol d
    dsel, ttab = {}, {}
    for v in VENUES:
        c = W.daily_close(v, SYM[v])
        t = E.minFFD_table(pd.DataFrame({"Close": c.values}, index=c.index))
        ttab[v] = t
        ok = t.index[t["adfStat"] < t["95% conf"]]
        dsel[v] = float(ok.min()) if len(ok) else 1.0

    def draw53(axs, v):
        s = np.log(W.daily_close(v, SYM[v])).to_frame("Close")
        a = PA.fracDiff(s, dsel[v], thres=1.0)
        b_ = PA.fracDiff(s, dsel[v], thres=0.01)
        for ax, r, lab in ((axs[0], a, "(a) no weight-loss control"), (axs[1], b_, "(b) expanding window, weight-loss control 0.01")):
            ax.plot(r.index, r["Close"], color="0.15", lw=0.6); ax.set_xlabel(f"{lab}"); ax.set_ylabel(f"FD(d={dsel[v]:.2f}) of log price")
            ax2 = ax.twinx(); ax2.plot(s.index, np.exp(s["Close"]), color="0.65", lw=0.6)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for j, v in enumerate(VENUES):
        draw53([axs[0, j], axs[1, j]], v); axs[0, j].set_title(f"{v} ({SYM[v]}, daily)")
    fig.tight_layout()
    out.append(item("Figure 5.3", "Fractional differentiation without weight-loss control (top) and with an expanding window (bottom)", "data", fig=save(fig, "f5_3"),
                    note="d = the smallest value in 0, 0.1, ..., 1 whose ADF statistic is below the 5% critical value (Table 5.1 below)."))
    out.append(item("Snippet 5.3", "Fixed-width window fracdiff (FFD)", "code", text="book_port_e.fracDiff_FFD / fracdiff_core; test_book_port_e.test_ffd_matches_project_frac_diff passes."))

    def draw54(ax, v):
        s = np.log(W.daily_close(v, SYM[v])).to_frame("Close")
        r = E.fracDiff_FFD(s, dsel[v], thres=1e-5)
        ax.plot(r.index, r["Close"], color="0.15", lw=0.6); ax.set_ylabel(f"FFD(d={dsel[v]:.2f}) of log price")
        ax2 = ax.twinx(); ax2.plot(s.index, np.exp(s["Close"]), color="0.65", lw=0.6)
    out.append(item("Figure 5.4", "FFD with a fixed-width window (thres 1e-5)", "data", fig=W.two_panel(draw54, "f5_4")))
    out.append(item("Snippet 5.4", "Finding the minimum d that passes the ADF test (plotMinFFD)", "code", text="book_port_e.minFFD_table (the body of the snippet); the adfuller is a port with constant and one lag (adfuller_c1), checked in test_book_port_e.test_adf_c1_properties."))
    tabs = {}
    for v in VENUES:
        sy = W.symbols(v)
        if v == "Alpaca":
            sy = sorted(sy, key=lambda x: -W.bars(v, x).n)[:40]
        rows = []
        cache = W.OUT / f"table_5_1_{v}.csv"
        if cache.exists():                                    # the loop below is slow (about 25 minutes per venue); reuse its saved output
            rows = pd.read_csv(cache, keep_default_na=False).values.tolist()
            sy = []
        for s in sy:
            c = W.daily_close(v, s)
            if len(c) < 400:
                continue
            try:
                t = E.minFFD_table(pd.DataFrame({"Close": c.values}, index=c.index))
            except Exception as ex:  # keep going, say so
                rows.append([s] + [f"error: {type(ex).__name__}"] + [""] * 10)
                continue
            rows.append([s] + [t.loc[x, "adfStat"] for x in sorted(t.index)] + [t["95% conf"].iloc[0]])
        tabs[v] = rows
        pd.DataFrame(rows, columns=["symbol"] + [f"d={x:.1f}" for x in np.linspace(0, 1, 11)] + ["5% critical"]).to_csv(W.OUT / f"table_5_1_{v}.csv", index=False)

    def draw55(ax, v):
        t = ttab[v]
        ax.plot(t.index, t["corr"], color="0.6", label="corr"); ax.set_ylim(0, 1.05); ax.set_xlabel("d")
        ax2 = ax.twinx(); ax2.plot(t.index, t["adfStat"], color="0.1", label="adfStat (right)"); ax2.axhline(t["95% conf"].iloc[0], color="0.3", ls=":")
        ax.legend(loc="lower left")
    out.append(item("Figure 5.5", "ADF statistic as a function of d (right axis) with the correlation to the original log series (left axis); dotted: 5% critical value", "data", fig=W.two_panel(draw55, "f5_5")))
    cols = ["symbol"] + [f"{x:.1f}" for x in np.linspace(0, 1, 11)] + ["5% crit"]
    for v in VENUES:
        rr = [[r[0]] + [fmt(x, 2) if not isinstance(x, str) else x for x in r[1:]] for r in tabs[v]]
        out.append(item("Table 5.1" if v == "Kraken" else "Table 5.1 (Alpaca)", f"ADF statistic on FFD(d), d = 0 to 1, {v} ({len(rr)} symbols{', the 40 with the most bars' if v == 'Alpaca' else ''})", "data", table=tbl(cols, rr),
                        note="Book: ~50 of the most liquid futures contracts (their data is not in the project). The same procedure on our daily closes."))
    out.append(item("Equation 34", "Mean squared error = bias^2 + variance + noise: E[(y_i - f^[x_i])^2] = (E[f^[x_i] - f[x_i]])^2 + V[f^[x_i]] + sigma_eps^2", "formula", text="The decomposition of the learning error that motivates ensembles (Section 6.2)."))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch6():
    out = []
    out.append(item("Section 6.3.1", "Variance reduction: bagging helps when the individual forecasts are not perfectly correlated", "text", text="The text prints (without an equation number) V[(1/N) sum_i phi_i] = sigma-bar^2 (rho-bar + (1 - rho-bar)/N), where rho-bar is the average correlation of the N forecasts; Figure 6.1 plots its square root."))
    out.append(item("Equation 35", "Accuracy of a bagging classifier: P[X > N/k] = 1 - P[X <= N/k] = 1 - sum_(i=0)^floor(N/k) C(N, i) p^i (1-p)^(N-i)", "formula", text="N independent classifiers, k classes, each correct with probability p; the ensemble is right when more than N/k of them vote for the right class (Snippet 6.1 evaluates it)."))
    N = np.arange(5, 31); rho = np.linspace(0, 1, 60)
    z = np.sqrt(rho[None, :] + (1 - rho[None, :]) / N[:, None])
    fig, ax = plt.subplots(figsize=(3.6, 2.8)); im = ax.imshow(z, origin="lower", extent=[0, 1, 5, 30], aspect="auto", cmap="Greys_r"); plt.colorbar(im)
    ax.set_xlabel("rho"); ax.set_ylabel("N"); ax.set_title("sigma = 1")
    out.append(item("Figure 6.1", "Standard deviation of the bagged prediction as a function of rho and N", "formula", fig=save(fig, "f6_1"), note="Closed form (Equation 35); same axes as the book."))
    out.append(item("Snippet 6.1", "Accuracy of the bagging classifier", "code", text="book_port_a.bagging_accuracy; test_book_port_a.test_bagging_accuracy_is_the_binomial_tail passes."))
    Ns = np.arange(1, 102, 2); ps = np.linspace(0.2, 0.8, 61)
    z = np.array([[PA.bagging_accuracy(int(n), p, 2.)[1] for p in ps] for n in Ns])
    fig, ax = plt.subplots(figsize=(3.8, 3.0)); im = ax.imshow(z, origin="lower", extent=[0.2, 0.8, 1, 101], aspect="auto", cmap="Greys_r", vmin=0, vmax=1); plt.colorbar(im)
    ax.set_xlabel("p"); ax.set_ylabel("N")
    for v, c in zip(VENUES, ("0.0", "0.4")):
        acc = meta(v)["walk_forward"]["side_accuracy"]
        ax.plot([acc], [100], "o", color="w" if v == "Kraken" else "0.5", ms=5, mec="k")
    out.append(item("Figure 6.2", "Accuracy of a bagging classifier as a function of the individual estimator's accuracy p and the number of estimators N, k = 2 (dots: our walk-forward side accuracy at N = 100)", "formula", fig=save(fig, "f6_2"),
                    table=tbl(["Venue", "single-model accuracy (walk-forward)", "bagged accuracy at N = 100, k = 2 (Snippet 6.1)"], [[v, fmt(meta(v)["walk_forward"]["side_accuracy"]), fmt(PA.bagging_accuracy(100, meta(v)["walk_forward"]["side_accuracy"], 2.)[1])] for v in VENUES]),
                    note="Snippet 6.1 assumes independent estimators; ours share data, so this is an upper bound on what bagging can add."))
    # AdaBoost flow diagram
    fig, ax = plt.subplots(figsize=(3.6, 4.4)); ax.axis("off")
    boxes = [(0.5, 0.95, "Start n = 1, w = 1, threshold A"), (0.5, 0.78, "n > N ?  (yes: final classifier = sum of w_n * classifier_n)"), (0.5, 0.60, "Sample with replacement using weights w to get training data X_n"),
             (0.5, 0.42, "Fit classifier on X_n, obtain accuracy w_n"), (0.5, 0.24, "Update weights: up if misclassified, down otherwise"), (0.5, 0.08, "accuracy >= A ?  (no: refit; yes: n += 1)")]
    for x, y, t in boxes:
        ax.text(x, y, t, ha="center", va="center", fontsize=7, bbox=dict(boxstyle="round", fc="0.9", ec="0.4"), wrap=True)
    for y0, y1 in ((0.91, 0.82), (0.74, 0.64), (0.56, 0.46), (0.38, 0.28), (0.20, 0.12)):
        ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="->", color="0.3"))
    out.append(item("Figure 6.3", "AdaBoost decision flow", "diagram", fig=save(fig, "f6_3"), note="A drawing in the book; redrawn from its steps. The bot does not use AdaBoost (it uses the bagged forest of Snippet 6.2)."))
    out.append(item("Snippet 6.2", "Three ways of setting up an RF", "code", text="book_port_a.three_rf_setups returns the three classifiers; the bot uses the second (bagged entropy trees, class_weight balanced, max_samples = average uniqueness, 100 trees, sequential bootstrap) via meta_model.py."))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch7():
    out = []
    folds = np.zeros((5, 5)); [folds.__setitem__((i, 4 - i), 1) for i in range(5)]
    fig, ax = plt.subplots(figsize=(4.2, 2.4)); ax.imshow(folds, cmap="Greys", vmin=0, vmax=1.6)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "Test" if folds[i, j] else "Train", ha="center", va="center", fontsize=7, color="w" if folds[i, j] else "0.2")
    ax.set_yticks(range(5)); ax.set_yticklabels([f"Fold {i+1}" for i in range(5)]); ax.set_xticks([])
    out.append(item("Figure 7.1", "Train/test splits in a 5-fold CV scheme", "diagram", fig=save(fig, "f7_1"), note="A diagram in the book; redrawn. Our splits are the 15 CPCV splits (Figure 12.1) and the walk-forward."))
    out.append(item("Snippet 7.1", "Purging observations in the training set (getTrainTimes)", "code", text="book_port_b.getTrainTimes (verbatim); test_book_port_b.test_get_train_times_three_overlap_cases passes."))

    def draw(ax, v, embargo=False):
        e = W.run_csv(v, "events.csv", usecols=["symbol", "t0", "t1"])
        e = e[e.symbol == SYM[v]].sort_values("t0")
        b = W.bars(v)
        ax.plot(b.idx, b.close, color="0.4", lw=0.5)
        t0m = pd.to_datetime(e["t0"], unit="s")
        lo, hi = t0m.iloc[int(len(e) * 0.55)], t0m.iloc[int(len(e) * 0.65)]
        ax.axvspan(lo, hi, color="0.8"); ax.axvline(lo, color="k", lw=0.6, ls=":"); ax.axvline(hi, color="k", lw=0.6, ls=":")
        ov = e[(pd.to_datetime(e["t0"], unit="s") < lo) & (pd.to_datetime(e["t1"], unit="s") >= lo)]
        if len(ov):
            ax.axvspan(pd.to_datetime(ov["t0"].min(), unit="s"), lo, color="0.55", alpha=0.7)
        if embargo:
            ax.axvspan(hi, hi + (hi - lo) * 0.1, color="0.5", alpha=0.7)
        ax.set_yscale("log"); ax.set_ylabel("Price")
    out.append(item("Figure 7.2", "Purging overlap in the training set: a test span (grey) and the training events whose labels reach into it (dark band)", "data",
                    fig=W.two_panel(lambda ax, v: draw(ax, v, False), "f7_2")))
    out.append(item("Snippet 7.2", "Embargo on training observations", "code", text="book_port_b.getEmbargoTimes / embargo_usage (verbatim); pctEmbargo = 0.01."))
    out.append(item("Figure 7.3", "Embargo of post-test training observations (the dark band after the test span)", "data", fig=W.two_panel(lambda ax, v: draw(ax, v, True), "f7_3")))
    out.append(item("Snippet 7.3", "Cross-validation class when observations overlap (PurgedKFold)", "code", text="book_port_b.PurgedKFold; test_book_port_b.test_purged_kfold_has_no_leak_and_covers_tests passes."))
    out.append(item("Snippet 7.4", "Using the PurgedKFold class (cvScore)", "code", text="book_port_b.cvScore; the bot's own CPCV applies the same purge and embargo in book_stats.train_mask, checked in verify_book_bot."))
    # purge stats table from the bot's split_metrics
    rows = []
    for v in VENUES:
        sm = W.run_csv(v, "split_metrics.csv")
        for _, r in sm.iterrows():
            rows.append([v, int(r["split"]), r["test_groups"], int(r["n_train"]), int(r["n_test"])])
    out.append(item("Snippet 7.4 (our splits)", "Training and test events per CPCV split after purging and the 1% embargo", "data", table=tbl(["Venue", "split", "test groups", "n_train", "n_test"], rows)))
    return out


def build():
    items = []
    for f in (ch1, ch2, ch3, ch4, ch5, ch6, ch7):
        print("building", f.__name__, flush=True)
        items += f()
    return items


if __name__ == "__main__":
    its = build()
    json.dump(its, open(W.OUT / "items_ch01_07.json", "w"), indent=1, default=str)
    print(len(its), "items")
