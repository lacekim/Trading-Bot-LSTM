"""Write-up items for Chapters 17, 18 and 19 of the book (structural breaks, entropy, microstructure).  New file; reads finished runs and bars, changes nothing."""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

import book_micro as BM
import book_port_c as PC
import book_port_e as PE
import meta_events as ME
import micro_features as MF
import writeup_core as W
from writeup_core import VENUES, SYM, item, tbl, fmt, save

_feat = {}


def daily(v):
    if v not in _feat:
        _feat[v] = ME.daily_features(W.bars(v).close_series())
    return _feat[v]


def logp(v):
    return np.log(W.bars(v).close_series().resample("1D").last().dropna())


def events(v, cols):
    return W.run_csv(v, "events.csv", usecols=["symbol"] + cols)


def sym_ev(v, col):
    e = events(v, [col])
    return e.loc[e["symbol"] == SYM[v], col].dropna()


def all_starts(p, minsl, lags=1):
    """t-statistic of the lagged level for every start point of one window (the loop of Snippets 17.1-17.4)."""
    p = np.asarray(p, float)
    dy = np.diff(p)
    n_rows = len(dy) - lags
    x = np.empty((n_rows, lags + 1))
    x[:, 0] = p[-n_rows - 1:-1]
    for k in range(1, lags + 1):
        x[:, k] = dy[lags - k: len(dy) - k]
    x = np.append(x, np.ones((n_rows, 1)), axis=1)
    y = dy[-n_rows:]
    out = []
    for start in range(0, n_rows + lags - minsl + 1):
        y_, x_ = y[start:], x[start:]
        xx = x_.T @ x_
        if abs(np.linalg.det(xx)) < 1e-14:
            out.append(np.nan); continue
        xi = np.linalg.inv(xx)
        b = xi @ (x_.T @ y_)
        e = y_ - x_ @ b
        var = (e @ e) / (x_.shape[0] - x_.shape[1]) * xi
        out.append(b[0] / math.sqrt(var[0, 0]))
    return np.array(out)


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch17():
    out = []
    dd = {v: daily(v)[1] for v in VENUES}
    lp = {v: logp(v) for v in VENUES}
    win = 100
    ev = {v: events(v, ["sadf", "csw"]) for v in VENUES}
    rows = [[v, fmt(len(ev[v])), fmt(float(ev[v]["csw"].quantile(.5)), 3), fmt(float(ev[v]["csw"].quantile(.95)), 3), fmt(float(ev[v]["sadf"].quantile(.5)), 3), fmt(float(ev[v]["sadf"].quantile(.95)), 3)] for v in VENUES]
    out.append(item("Section 17.3", "CUSUM tests: Brown-Durbin-Evans (recursive residuals) and Chu-Stinchcombe-White (levels)", "data",
                    table=tbl(["Venue", "Events", "CSW median", "CSW 95th pct", "SADF median", "SADF 95th pct"], rows),
                    text="Our bot uses the Chu-Stinchcombe-White test on levels (feature csw: S_t = sup over n of (y_t - y_n) / (sigma_t sqrt(t - n)), 100 hourly rows) and the SADF (Snippets 17.1-17.4).  The Brown-Durbin-Evans recursive-residual test needs a model with regressors x_t; the side-free bot has none, so it is not run.  Table: distribution of the two features at the events of the whole venue."))
    # Equations 59-62: Chow-type Dickey-Fuller on our last 250 days -------------------------------------------------------------------------------------
    rows = []
    for v in VENUES:
        y = lp[v].to_numpy()[-251:]
        dy, ylag = np.diff(y), y[:-1]
        T = len(dy)
        best, bt = -np.inf, None
        for tau in np.linspace(.1, .9, 41):
            k = int(tau * T)
            Dt = (np.arange(T) >= k).astype(float)
            xx = ylag * Dt
            if not np.any(xx):
                continue
            d = (xx @ dy) / (xx @ xx)
            e = dy - d * xx
            s2 = (e @ e) / (T - 1)
            t = d / math.sqrt(s2 / (xx @ xx))
            if t > best:
                best, bt = t, tau
        rows.append([v, fmt(T), fmt(best, 3), fmt(bt, 2)])
    out.append(item("Equation 59", "Chow-type Dickey-Fuller regression: dy_t = delta y_(t-1) D_t[tau*] + eps_t", "data",
                    table=tbl(["Venue", "T (days)", "SDFC = sup DFC_tau* (Eq 61)", "tau* at the supremum"], rows),
                    text="Run on the last 251 daily log closes of ETH (Kraken) and AAPL (Alpaca); tau* searched on 41 points in [0.1, 0.9] (tau0 = 0.1, a choice; the extract does not fix it)."))
    out.append(item("Equation 60", "Dickey-Fuller-Chow statistic DFC_tau* = delta-hat / sigma-hat_delta", "formula", text="The t-statistic of delta in Equation 59; computed inside the supremum above."))
    out.append(item("Equation 61", "SDFC = sup over tau* in [tau0, 1 - tau0] of DFC_tau*", "formula", text="Values in the table under Equation 59."))
    out.append(item("Equation 62", "Augmented Dickey-Fuller specification dy_t = alpha + beta y_(t-1) + sum_l gamma_l dy_(t-l) + eps_t", "formula",
                    text="Our SADF uses one lagged difference (L = 1), a constant, a trailing window of 100 daily log closes and a minimum sample of 20 (project settings in meta_events.py: SADF_WINDOW 100, SADF_MINSL 20, SADF_LAGS 1)."))

    # Figure 17.1 ----------------------------------------------------------------------------------------------------------------------------------------
    def draw(ax, v):
        s = dd[v]["sadf"]
        idx = pd.to_datetime(s.index.to_numpy() * 86400, unit="s")
        px = np.exp(lp[v])
        ax.plot(px.index, px.values, color="0.15", lw=0.8, label="close")
        ax.set_yscale("log")
        ax2 = ax.twinx()
        ax2.plot(idx, s.to_numpy(), color="0.6", lw=0.5, label="SADF")
        ax2.axhline(0, color="0.8", lw=0.4)
        ax.tick_params(axis="x", rotation=30)
    out.append(item("Figure 17.1", "Prices (left axis, log) and SADF (right axis) over time", "data", fig=W.two_panel(draw, "f17_1", w=7.6)))

    # Equation 63, 64, Table 17.1 -------------------------------------------------------------------------------------------------------------------------
    rows = []
    for v in VENUES:
        T = win
        e = all_starts(lp[v].to_numpy()[-T:], 20)
        rows.append([v, fmt(len(e)), fmt(float(np.nanmax(e)), 4)])
    out.append(item("Equation 63", "SADF_t = sup over t0 in [1, t - tau] of ADF_(t0,t) = beta-hat / sigma-hat_beta", "data",
                    table=tbl(["Venue", "Start points t0 in the last window (T = 100, tau = 20)", "SADF of the last window"], rows),
                    text="Each start point is one regression (Snippets 17.1-17.4); the saved SADF series is Figure 17.1's right axis."))
    T_, tau = 100, 20
    n_adf = (T_ - tau + 2) * (T_ - tau + 1) // 2
    out.append(item("Equation 64", "Number of ADF estimates needed over a sample, sum_(t=tau)^T (t - tau + 1) = (T - tau + 2)(T - tau + 1)/2", "formula",
                    text=f"Whole-sample SADF at T = {T_}, tau = {tau}: {n_adf:,} estimates per window ({T_ - tau + 1} for the last observation alone)."))
    N_ = 3
    fl = [("o1 = X'y", "(2T-1)N", (2 * T_ - 1) * N_), ("o2 = X'X", "(2T-1)N^2", (2 * T_ - 1) * N_ ** 2), ("o3 = o2^-1", "N^3 + N^2 + N", N_ ** 3 + N_ ** 2 + N_),
          ("o4 = o3 o1", "2N^2 - N", 2 * N_ ** 2 - N_), ("o5 = y - X o4", "T + (2N-1)T", T_ + (2 * N_ - 1) * T_), ("o6 = o5' o5", "2T - 1", 2 * T_ - 1),
          ("o7 = o3 o6 / (T-N)", "2 + N^2", 2 + N_ ** 2), ("o8 = o4[0,0] / sqrt(o7[0,0])", "1", 1)]
    tot = sum(x[2] for x in fl)
    out.append(item("Table 17.1", "FLOPs per ADF estimate (T = 100 rows, N = 3 regressors: level, one lagged difference, constant)", "formula",
                    table=tbl(["Operation", "FLOPs (printed formula)", "FLOPs at T = 100, N = 3"], [[a, b, fmt(c)] for a, b, c in fl] + [["Total per estimate", "", fmt(tot)], ["x estimates per SADF window (Eq 64)", "", fmt(n_adf)], ["= FLOPs per SADF value", "", fmt(tot * n_adf)]]),
                    note="Printed formulas as read from the extract (the layout of the table is garbled in the text layer); the arithmetic is ours."))

    # Figure 17.2 / 17.3 ----------------------------------------------------------------------------------------------------------------------------------
    def stats(v):
        s = dd[v]["sadf"].dropna()
        q = s.rolling(250, min_periods=100).quantile(.95)
        sd = s.rolling(250, min_periods=100).std()
        return s, q, sd
    def d2(ax, v):
        s, q, sd = stats(v)
        ax.scatter(s.to_numpy(), q.to_numpy(), s=2, color="0.3")
        ax.set_xlabel("SADF_t"); ax.set_ylabel("trailing 95% quantile of SADF")
    out.append(item("Figure 17.2", "SADF (x-axis) against a critical value (y-axis)", "data", fig=W.two_panel(d2, "f17_2"),
                    note="The extract prints the figure and its caption (SADF vs CADF) and no text defining CADF; ours plots SADF against its own trailing-250-day 95% quantile, which is a stand-in, not the book's CADF."))
    def d3(ax, v):
        s, q, sd = stats(v)
        z = (s - q) / sd
        idx = pd.to_datetime(s.index.to_numpy() * 86400, unit="s")
        ax.plot(idx, z.to_numpy(), color="0.2", lw=0.6)
        ax.axhline(0, color="0.7", lw=0.5); ax.tick_params(axis="x", rotation=30); ax.set_ylabel("(SADF - C) / spread")
    out.append(item("Figure 17.3", "(SADF_t - C_(t,q)) / C-dot_(t,q) over time", "data", fig=W.two_panel(d3, "f17_3"),
                    note="The extract does not define C_(t,q) or C-dot_(t,q); ours: C = trailing-250-day 95% quantile of SADF, divided by the trailing-250-day standard deviation (assumption)."))

    # Snippets 17.1 - 17.4 ---------------------------------------------------------------------------------------------------------------------------------
    rows = []
    for v in VENUES:
        x = lp[v].iloc[-win:]
        lit = PC.get_bsadf(x.to_frame("close"), 20, "c", 1)["gsadf"]
        mine = MF.sadf_literal(x.to_numpy(), 20, 1)
        fast = MF.sadf_windows(lp[v].to_numpy(), win, 20, 1)[-1]
        rows.append([v, fmt(lit, 8), fmt(mine, 8), fmt(float(fast), 8), f"{abs(lit - fast):.1e}"])
    t_par = tbl(["Venue", "Printed snippets (book_port_c.get_bsadf)", "meta literal (micro_features.sadf_literal)", "vectorised (feature in the bot)", "|printed - vectorised|"], rows)
    out.append(item("Snippet 17.1", "SADF's inner loop over start points", "code", table=t_par, text="book_port_c.get_bsadf; parity of the printed loop with the vectorised feature on the last 100 daily log closes of each venue's symbol."))
    out.append(item("Snippet 17.2", "Preparing the datasets (y and the lagged regressors)", "code", text="book_port_c.getYX; used by Snippet 17.1 above."))
    out.append(item("Snippet 17.3", "Applying lags to a dataframe", "code", text="book_port_c.lagDF; used by getYX."))
    out.append(item("Snippet 17.4", "Fitting the ADF specification (betas and their standard errors)", "code", text="book_port_c.getBetas; used by get_bsadf."))

    # Equation 65, 66 -------------------------------------------------------------------------------------------------------------------------------------
    rows = []
    for v in VENUES:
        e = all_starts(lp[v].to_numpy()[-win:], 20)
        t0 = np.arange(len(e))
        n = win - 1 - 1
        smt = np.nanmax(np.abs(e))
        smt5 = np.nanmax(np.abs(e) / (n - t0) ** 0.5)
        rows.append([v, fmt(float(np.nanmax(e)), 4), fmt(float(smt), 4), fmt(float(smt5), 4)])
    out.append(item("Equation 65", "Sub/super-martingale test SMT_t = sup |beta-hat| / sigma-hat_beta over start points", "data",
                    table=tbl(["Venue", "SADF (one-sided, Eq 63)", "SMT (Eq 65)", "SMT with phi = 0.5 (Eq 66)"], rows),
                    text="The absolute value lets the test flag both explosive up- and down-trends. Last window of 100 daily log closes."))
    out.append(item("Equation 66", "Same with a penalty on long windows: |beta-hat| / (sigma-hat_beta (t - t0)^phi)", "data", text="Third column of the table under Equation 65 (phi = 0.5 is our choice; the book leaves phi in [0, 1])."))
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------------
def _msg(v, n=100):
    x = logp(v).to_numpy()
    b = (np.diff(x) > 0).astype(int)[-n:]
    return "".join(map(str, b))


def ch18():
    import random
    out = []
    lp = {v: logp(v) for v in VENUES}
    msg = {v: _msg(v) for v in VENUES}
    e = {v: events(v, ["ent_plugin", "ent_konto"]) for v in VENUES}
    rows = [[v, fmt(float(e[v]["ent_plugin"].mean()), 4), fmt(float(e[v]["ent_konto"].mean()), 4), fmt(float(e[v]["ent_plugin"].std()), 4), fmt(float(e[v]["ent_konto"].std()), 4)] for v in VENUES]
    out.append(item("Section 18.2", "Shannon's entropy and redundancy R = 1 - H / log2|A|", "data",
                    table=tbl(["Venue", "mean plug-in entropy rate", "mean Kontoyiannis estimate", "std plug-in", "std Kontoyiannis"], rows),
                    text="Encoding (a declared choice; the condensed edition has no encoding section): bit = 1 if the daily log return is positive, else 0; message = last 100 days; plug-in word length 2; Kontoyiannis on the reversed message.  Values are the bot's two entropy features at the events of the whole venue (maximum for a fair coin is 1)."))
    rows = []
    for v in VENUES:
        h, pmf = PC.plugIn(msg[v], 2)
        rows.append([v, msg[v][-30:] + " (last 30 of 100 bits)", fmt(h, 5), fmt(1 - h, 5)])
    out.append(item("Equation 67", "Plug-in (maximum-likelihood) entropy estimator H-hat(n,w) = -sum p-hat log2 p-hat / w", "data",
                    table=tbl(["Venue", "Message", "entropy rate (w = 2)", "redundancy 1 - H"], rows)))
    rows = []
    for v in VENUES:
        _, pmf = PC.plugIn(msg[v], 2)
        h, _ = PC.plugIn(msg[v], 2)
        rows.append([v, fmt(h, 5), ", ".join(f"{k}:{x:.3f}" for k, x in sorted(pmf.items()))])
    out.append(item("Snippet 18.1", "Plug-in entropy estimator (plugIn, pmf1)", "code", table=tbl(["Venue", "entropy rate", "pmf of 2-bit words"], rows),
                    text="book_port_c.plugIn / pmf1 (verbatim; unit-tested)."))
    rows = []
    for v in VENUES:
        lib = PC.lempelZiv_lib(msg[v])
        rows.append([v, fmt(len(lib)), fmt(max(len(s) for s in lib))])
    out.append(item("Snippet 18.2", "A library built with the Lempel-Ziv algorithm", "code", table=tbl(["Venue", "Library size (distinct phrases)", "Longest phrase"], rows), text="book_port_c.lempelZiv_lib on the 100-bit messages."))
    out.append(item("Equation 68", "Matched length L_i^n = 1 + max{ l : x_i^(i+l) = x_j^(j+l) for some i - n <= j <= i - 1, l in [0, n] }", "formula", text="book_port_c.matchLength returns it (the snippet returns the matched length plus 1 and the matched substring)."))
    rows = []
    for v in VENUES:
        l, sub = PC.matchLength(msg[v], 50, 50)
        rows.append([v, fmt(l), sub])
    out.append(item("Snippet 18.3", "Length of the longest match found in the previous n characters", "code", table=tbl(["Venue", "L at i = 50, n = 50", "matched substring"], rows)))
    out.append(item("Equation 69", "Entropy rate as the limit lim L_i^n / log2(n) = 1 / H", "formula", text="Kontoyiannis et al.'s result; the basis of Equations 70 to 73."))
    out.append(item("Equation 70", "Fixed-window estimator H-hat(n,k) = [ (1/k) sum_i L_i^n / log2 n ]^-1", "formula", text="konto(msg, window) in Snippet 18.4 with a window."))
    out.append(item("Equation 71", "Expanding-window estimator H-hat(n) = [ (1/n) sum_(i=2)^n L_i^i / log2 i ]^-1", "formula", text="konto(msg) with window = None."))
    out.append(item("Equation 72", "Reciprocal form of Equation 70 (average of log2 n / L)", "formula", text="The snippet accumulates log2(window + 1) / l (the +1 avoids the Doeblin condition) and averages: this is the quantity in Equations 72 and 73."))
    out.append(item("Equation 73", "Reciprocal form of Equation 71", "formula", text="As above with the expanding window."))
    rows = []
    for m in ("101010" * 2, "101010" + "010101", msg["Kraken"], msg["Alpaca"]):
        r = PC.konto(m)
        rows.append([m if len(m) <= 14 else m[:14] + "...", fmt(len(m)), fmt(r["h"], 5), fmt(r["r"], 5)])
    out.append(item("Snippet 18.4", "Implementation of the Kontoyiannis algorithms (konto): the snippet's own examples and our two messages", "code",
                    table=tbl(["Message", "Length", "entropy rate h", "redundancy r"], rows), text="The first two rows are the snippet's main(): konto(msg*2) and konto(msg + msg[::-1]) with msg = '101010'."))
    rows = []
    for v in VENUES:
        sig = float(WP_ret(v).std())
        rows.append([v, fmt(sig, 6), fmt(0.5 * math.log(2 * math.pi * math.e * sig ** 2), 4)])
    out.append(item("Equation 74", "Differential entropy of a Gaussian: H = 1/2 log(2 pi e sigma^2)", "data", table=tbl(["Venue", "sigma of daily returns (symbol)", "H (nats)"], rows)))
    out.append(item("Equation 75", "Generalised mean M_q[x, p] = (sum p_i x_i^q)^(1/q)", "formula", text="Used with x = p to build the effective number below."))
    rng = np.random.RandomState(18)
    p = [rng.uniform(size=100) for _ in range(30)]
    p = [a / a.sum() for a in p]
    qs = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for a in p:
        ax.plot(qs, [-math.log(np.sum(a * a ** q) ** (1.0 / q)) for q in qs], color="0.5", lw=0.5)
    ax.set_xlabel("q"); ax.set_ylabel("log effective number")
    out.append(item("Equation 76", "Effective number 1 / M_q[p, p] = (sum p_i p_i^q)^(-1/q)", "formula", text="As q grows the effective number shrinks toward the reciprocal of the largest probability."))
    fp18_2 = save(fig, "f18_2")
    out.append(item("Figure 18.2", "Log effective numbers for a family of randomly generated p arrays (30 arrays of 100 uniform draws, normalised)", "formula", fig=fp18_2,
                    note="Seeded sample; the book's draws are unseeded."))
    # Figure 18.1 -------------------------------------------------------------------------------------------------------------------------------------------
    rs = random.Random(181)
    fig, axs = plt.subplots(2, 4, figsize=(7.8, 3.4))
    summary = []
    for j, A in enumerate((10, 7, 5, 2)):
        h1, h2 = [], []
        for _ in range(300):
            m = "".join(str(rs.randrange(A)) for _ in range(100))
            h1.append(PC.plugIn(m, 1)[0] / math.log2(A) if A > 1 else 0)
            h2.append(PC.konto(m)["h"] / math.log2(A))
        axs[0, j].hist(h1, bins=20, color="0.4"); axs[0, j].set_title(f"plug-in, {A} letters", fontsize=7)
        axs[1, j].hist(h2, bins=20, color="0.4"); axs[1, j].set_title(f"Kontoyiannis, {A} letters", fontsize=7)
        summary.append([A, fmt(float(np.mean(h1)), 4), fmt(float(np.std(h1)), 4), fmt(float(np.mean(h2)), 4), fmt(float(np.std(h2)), 4)])
    fig.tight_layout()
    out.append(item("Figure 18.1", "Distribution of entropy estimates on random messages of length 100 under 10, 7, 5 and 2 letter encodings", "formula", fig=save(fig, "f18_1"),
                    table=tbl(["Letters", "plug-in mean (/ log2 A)", "std", "Kontoyiannis mean (/ log2 A)", "std"], summary),
                    note="300 random messages per encoding (the book draws more, unseeded); values divided by log2(A) so that 1 is the maximum.  The book's histograms are on the raw scale."))
    conc = {v: meta_conc(v) for v in VENUES}
    out.append(item("Equation 77", "Risk contribution theta_i = [f_omega]_i^2 L_ii / sum_n [f_omega]_n^2 L_nn (PCA risk loadings of a portfolio)", "data",
                    table=tbl(["Venue", "Rebalances", "first H", "last H", "mean H"], [[v, fmt(conc[v]["rebalances"]), fmt(conc[v]["H_first"], 4), fmt(conc[v]["H_last"], 4), fmt(conc[v]["H_mean"], 4)] for v in VENUES]),
                    text="Computed for every HRP allocation of the run (saved in meta.json: concentration_eq77_78). f = V' w is an assumption (the extract does not define f)."))
    out.append(item("Equation 78", "Portfolio concentration H = 1 - exp(-sum theta_i log theta_i) / N", "data", text="Values in the table under Equation 77 (0 = every risk factor carries an equal share, 1 = one factor carries all risk)."))
    out.append(item("Equation 79", "PIN = alpha mu / (alpha mu + 2 epsilon)", "formula", text="Probability of information-based trading; needs buy and sell counts per volume bucket (see Section 19.5.1)."))
    out.append(item("Equation 80", "VPIN = alpha mu / V ~ E|2 V_tau^B - V| / V", "formula", text="Volume-synchronised PIN (see Section 19.5.2)."))
    return out


def WP_ret(v):
    import writeup_panel as WP
    return WP.panel(v)[SYM[v]].dropna()


def meta_conc(v):
    return W.run_json(v, "meta.json")["concentration_eq77_78"]


# --------------------------------------------------------------------------------------------------------------------------------------------------
def ch19():
    out = []
    refs = {}
    for v in VENUES:
        b = W.bars(v)
        refs[v] = b
    rows = []
    for v in VENUES:
        b = refs[v]
        bs = BM.tick_rule(b.ref_close)
        rows.append([v, fmt(len(bs)), fmt(float((bs > 0).mean()), 4), fmt(float((bs < 0).mean()), 4)])
    out.append(item("Equation 81", "Tick rule b_t = sign(dp_t), carried forward when dp_t = 0", "data",
                    table=tbl(["Venue", "Hourly rows", "share of +1", "share of -1"], rows), text="book_micro.tick_rule on the hourly reference series of the venue's symbol."))
    ev = {v: events(v, ["roll", "cs_spread", "bp_sigma", "kyle", "amihud", "hasbrouck"]) for v in VENUES}
    q = lambda v, c: fmt(float(ev[v][c].median()), 6)  # noqa: E731
    out.append(item("Equation 82", "Bid-ask bounce: sigma^2[dp] = 2c^2 + sigma_u^2", "data", text="Roll's model; the bot's roll feature uses it through Equation 83.",
                    table=tbl(["Venue", "median roll c (price units, events)"], [[v, q(v, "roll")] for v in VENUES])))
    out.append(item("Equation 83", "Roll estimator: cov[dp_t, dp_(t-1)] = -c^2, c = sqrt(max(0, -cov))", "data", text="Rolling 100 hourly rows (book_micro.roll_measure)."))
    out.append(item("Equation 84", "Parkinson-type moment E[(1/T) sum log(H_t/L_t)^2] = k1 sigma_HL^2", "formula", text="k1 = 4 log 2.  Used through Snippet 19.2 (beta)."))
    out.append(item("Equation 85", "E[(1/T) sum log(H_t/L_t)] = k2 sigma_HL, k2 = sqrt(8/pi)", "formula", text="Used in Snippet 19.2 (getSigma)."))
    out.append(item("Equation 86", "Corwin-Schultz spread S_t = 2 (e^alpha_t - 1) / (1 + e^alpha_t)", "data", text="Median over our events (feature cs_spread): " + "; ".join(f"{v} {q(v, 'cs_spread')}" for v in VENUES)))
    out.append(item("Equation 87", "alpha_t = (sqrt(2 beta_t) - sqrt(beta_t)) / (3 - 2 sqrt 2) - sqrt(gamma_t / (3 - 2 sqrt 2))", "formula", text="Printed with beta and gamma of Equations 88 and 89; Snippet 19.1 getAlpha."))
    out.append(item("Equation 88", "beta_t = E[ sum_(j=0)^1 (log (H_(t-j) / L_(t-j)))^2 ]", "formula", text="Snippet 19.1 getBeta (two-bar sum of squared log high/low, mean over sl bars)."))
    out.append(item("Equation 89", "gamma_t = (log (H_(t-1,t) / L_(t-1,t)))^2", "formula", text="Snippet 19.1 getGamma (high over two bars / low over two bars)."))
    rows = []
    for v in VENUES:
        b = refs[v]
        n = min(len(b.ref_t), 3000)
        df = pd.DataFrame({"High": b.ref_high[-n:], "Low": b.ref_low[-n:]}, index=pd.to_datetime(b.ref_t[-n:], unit="s"))
        lit = PE.corwinSchultz(df, 1)["Spread"]
        vec = MF.corwin_schultz(df["High"].to_numpy(), df["Low"].to_numpy(), 1)
        vs = pd.Series(vec, index=df.index).dropna()
        j = lit.index.intersection(range(len(vs)))
        a, c = lit.to_numpy(), vs.to_numpy()
        m = min(len(a), len(c))
        rows.append([v, fmt(n), fmt(m), f"{np.nanmax(np.abs(a[-m:] - c[-m:])):.2e}"])
    out.append(item("Snippet 19.1", "Implementation of the Corwin-Schultz algorithm", "code",
                    table=tbl(["Venue", "Hourly rows", "Compared", "max |printed - vectorised|"], rows),
                    text="book_port_e.corwinSchultz (printed) against micro_features.corwin_schultz (the bot's vectorised form) on the last 3000 hourly rows."))
    rows = []
    for v in VENUES:
        b = refs[v]
        n = min(len(b.ref_t), 3000)
        df = pd.DataFrame({"High": b.ref_high[-n:], "Low": b.ref_low[-n:]}, index=pd.to_datetime(b.ref_t[-n:], unit="s"))
        beta, gamma = PE.getBeta(df, 24), PE.getGamma(df)
        sg = PE.getSigma(beta, gamma)
        bp = BM.bp_sigma(b.ref_high, b.ref_low)
        a = sg.to_numpy()
        c = bp[-len(a):]
        rows.append([v, fmt(float(np.nanmedian(a)), 6), fmt(float(np.nanmedian(c)), 6), f"{np.nanmax(np.abs(a - c)):.2e}", fmt(float((a == 0).mean()), 3)])
    out.append(item("Snippet 19.2", "Estimating volatility for high-low prices (Becker-Parkinson)", "code",
                    table=tbl(["Venue", "median sigma (printed)", "median sigma (bot)", "max |difference|", "share of rows clipped to 0"], rows), text="book_port_e.getSigma on the last 3000 hourly rows against book_micro.bp_sigma.  The snippet sets negative estimates to 0; where more than half the rows are clipped the median is 0."))

    def hist(col, name):
        def draw(ax, v):
            x = ev[v][col].replace([np.inf, -np.inf], np.nan).dropna()
            lo, hi = x.quantile(.01), x.quantile(.99)
            ax.hist(x[(x >= lo) & (x <= hi)], bins=50, color="0.4")
            ax.set_xlabel(name); ax.set_ylabel("events")
        return draw
    d = lambda col: [[v, fmt(float(ev[v][col].median()), 8), fmt(float(ev[v][col].quantile(.05)), 8), fmt(float(ev[v][col].quantile(.95)), 8)] for v in VENUES]  # noqa: E731
    out.append(item("Figure 19.1", "Kyle's lambdas at our events (1st to 99th percentile)", "data", fig=W.two_panel(hist("kyle", "Kyle lambda"), "f19_1"),
                    note="The book computes them from trades inside each bar of E-mini S&P 500 futures; we hold no trades, so ours use hourly rows (declared in book_micro.py)."))
    out.append(item("Equation 90", "Kyle's lambda: dp_t = lambda (b_t V_t) + eps_t", "data", table=tbl(["Venue", "median", "5th pct", "95th pct"], d("kyle")), text="Regression through the origin over 100 hourly rows (book_micro.lambdas)."))
    out.append(item("Equation 91", "Amihud's lambda: |dlog p~_tau| = lambda sum (p_t V_t) + eps_tau", "data", table=tbl(["Venue", "median", "5th pct", "95th pct"], d("amihud"))))
    out.append(item("Figure 19.2", "Amihud's lambdas at our events", "data", fig=W.two_panel(hist("amihud", "Amihud lambda"), "f19_2")))
    out.append(item("Equation 92", "Hasbrouck's lambda: dlog p~_(i,tau) = lambda_i sum b_(i,t) sqrt(p_(i,t) V_(i,t)) + eps", "data", table=tbl(["Venue", "median", "5th pct", "95th pct"], d("hasbrouck"))))
    out.append(item("Figure 19.3", "Hasbrouck's lambdas at our events", "data", fig=W.two_panel(hist("hasbrouck", "Hasbrouck lambda"), "f19_3")))
    out.append(item("Section 19.5.1", "Probability of information-based trading (PIN)", "text",
                    text="PIN = alpha mu / (alpha mu + 2 epsilon), the probability that a trade is information-driven; it needs the numbers of buy and sell trades per period to fit alpha, mu, epsilon by maximum likelihood.  We hold neither trades nor a buy / sell split (only each row's total volume), so PIN is not estimated and is not a bot feature (the book's feature list for the bot uses Corwin-Schultz, Becker-Parkinson, Roll and the three lambdas)."))
    out.append(item("Section 19.5.2", "Volume-synchronised probability of informed trading (VPIN)", "text",
                    text="VPIN = E|2 V^B_tau - V| / V over volume buckets of equal size V.  Our dollar bars are such buckets, but each bar's buy share is unobservable without trades (the tick rule on a bar gives all-buy or all-sell), so VPIN is not computed; this item is covered in the text only."))
    return out


def build():
    items = []
    for f in (ch17, ch18, ch19):
        print("building", f.__name__, flush=True)
        items += f()
    return items


if __name__ == "__main__":
    its = build()
    json.dump(its, open(W.OUT / "items_ch17_19.json", "w"), indent=1, default=str)
    print(len(its), "items")
