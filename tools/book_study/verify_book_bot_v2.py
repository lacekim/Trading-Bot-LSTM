"""Independent re-derivation of the numbers in one book_bot.py run folder.  Shares NO code with book_bot.py, book_events.py, book_micro.py, book_final.py, book_otr_exit.py, bet_averaging.py, book_port_*.py, book_stats.py, bar_events.py,
meta_events.py or the bar loader: everything is recomputed from the raw bar / reference-hour csv files and the saved events, predictions and tables with its own code (constants copied here on purpose, so that a changed constant shows up as a difference).

Re-derived (each line is counted in the "numbers" total):
  events     CUSUM events (Snippet 2.4) of every symbol, one-bar sigma, bars per day, Snippet 3.1 daily volatility, the triple-barrier exit / kind / return / label of EVERY event (Snippets 3.2-3.7 with side=None), dropLabels (3.8), average uniqueness
             and return-attribution weights (4.1, 4.2, 4.10) for every symbol with no dropped event, time-decay factor (4.11)
  features   Roll, Kyle / Amihud / Hasbrouck lambda, Chu-Stinchcombe-White, Becker-Parkinson sigma and Corwin-Schultz spread, brute force from the reference-hour csv for a random sample of events (FFD, SADF and the two entropies are not re-derived)
  Ch.13      the OU fit (13.5-13.7) on the pre-cutoff long paths, the 20 x 20 simulated Sharpe mesh (same seed, own recursion), the chosen node, the 20 x 20 out-of-sample mesh and Definition 2
  model      walk-forward and CPCV AUC / accuracy, Table 14.1 counts, Ch.9 search table (mean of folds, chosen candidate)
  2.4.1      rebalance cost on every rebalance date and the per-trade volume capacity (ETF-trick items 1 and 3)
  account    bet size (10.1/10.3), averaging (10.2), OTR exits, per-bar position, day booking, HRP (own linkage / quasi-diagonalisation / bisection, every weight at every rebalance), every row of summary.csv
  Ch.14      daily target position, bets, holding period, HHI, drawdowns and time under water, annualised return (Eq. 49), PSR and DSR
  Ch.15      binHR, binFreq, binSR, probFailure from the per-bet returns; the t-value of precision (Eq.54); Eq.77-78 portfolio concentration of every HRP allocation (f = V'w assumed)
  10.4       every order ticket
Not re-derived (listed so nothing is hidden): FFD, SADF and entropy features; MDI / MDA / SFI / orthogonal-feature tables and the forest fits; the Snippet 13.2 verbatim-loop cross-check; the 100-tree forests behind the predictions.
Usage (from the project root): python tools/book_study/verify_book_bot.py <run folder> [--plant label|feature|weight|bet_size|cost|hrp|exit|ch13|ch14|ch15|dsr|order|rebal]
Exit code 1 when any number differs by more than 1e-9 relative (1e-9 absolute near zero)."""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.stats import norm, rankdata

DAY = 86400
SPAN, NDAY_W, STATE_TOL, CS_SL, W_ROLL, SL_BP, H_DAYS, PTSL, K_OTR = 336, 30 * DAY, 6 * 3600, 24, 100, 24, 4.0, 2.0, 100
MESH = [0.5 * i for i in range(1, 21)]
EULER = 0.5772156649015329


def epoch(s) -> np.ndarray:
    return pd.to_datetime(pd.Series(s)).to_numpy().astype("datetime64[s]").astype(np.int64)


def load_bars(bar_dir: Path, sym: str, break_s: int):
    b = pd.read_csv(bar_dir / "bars" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "start", "close"])
    b = b.dropna(subset=["close"])
    b = b[b["close"] > 0]
    t, st = epoch(b["date"]), epoch(b["start"])
    c = b["close"].to_numpy(float)
    o = np.argsort(t, kind="stable")
    t, st, c = t[o], st[o], c[o]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, st, c = t[keep], st[keep], c[keep]
    brk = (st[1:] - t[:-1]) > break_s
    seg = np.concatenate([[0], np.cumsum(brk)]).astype(np.int64)
    return t, c, seg, st


def load_vol(bar_dir: Path, sym: str) -> np.ndarray:
    """Volume of each bar, in the same order and with the same filters as load_bars."""
    b = pd.read_csv(bar_dir / "bars" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "close", "volume"])
    b = b.dropna(subset=["close"])
    b = b[b["close"] > 0]
    t = epoch(b["date"])
    o = np.argsort(t, kind="stable")
    keep = np.concatenate([[True], np.diff(t[o]) > 0])
    return b["volume"].to_numpy(float)[o][keep]


def load_ref(bar_dir: Path, sym: str):
    r = pd.read_csv(bar_dir / "ref_1h" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "high", "low", "close", "volume"])
    r = r.dropna(subset=["close"])
    r = r[r["close"] > 0]
    rt = epoch(r["date"]) + 3600
    o = np.argsort(rt, kind="stable")
    return rt[o], r["high"].to_numpy(float)[o], r["low"].to_numpy(float)[o], r["close"].to_numpy(float)[o], r["volume"].to_numpy(float)[o]


def bet_size(p: float, step: float, plant: bool) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    if p <= 0.5:
        return 0.0
    z = (p - 0.5) / math.sqrt(p * (1 - p))
    s = 2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))) - 1.0
    s = round(s / step) * step
    s = min(max(s, 0.0), 1.0)
    return s * 1.01 if plant else s


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------- brute-force features at one reference row ----------------
def _hl_terms(rh, rl):
    h, l = rh.copy(), rl.copy()
    bad = ~(h > 0) | ~(l > 0) | (h < l)
    h[bad], l[bad] = np.nan, np.nan
    return h, l


def brute_features(rh, rl, rc, rv, k):
    """All seven values at reference row k, from the definitions, with loops over the window."""
    W, sl = W_ROLL, SL_BP
    out = {}
    c = rc[:k + 1]
    nan = float("nan")
    b = np.zeros(k + 1)
    last = 0.0
    for i in range(1, k + 1):
        d = c[i] - c[i - 1]
        if d > 0:
            last = 1.0
        elif d < 0:
            last = -1.0
        b[i] = last
    dp = np.full(k + 1, np.nan)
    dp[1:] = c[1:] - c[:-1]
    dl = np.full(k + 1, np.nan)
    dl[1:] = np.log(c[1:]) - np.log(c[:-1])
    v = rv[:k + 1]
    dol = c * v
    if k - W + 1 >= 1:
        ix = np.arange(k - W + 1, k + 1)
        for nm, x, y in (("kyle", b[ix] * v[ix], dp[ix]), ("amihud", dol[ix], np.abs(dl[ix])), ("hasbrouck", b[ix] * np.sqrt(dol[ix]), dl[ix])):
            sxx = float((x * x).sum())
            out[nm] = float((x * y).sum() / sxx) if sxx > 0 and np.isfinite(x).all() and np.isfinite(y).all() else nan
    else:
        out.update(kyle=nan, amihud=nan, hasbrouck=nan)
    if k - W + 1 >= 2:
        ix = np.arange(k - W + 1, k + 1)
        cov = float(np.mean(dp[ix] * dp[ix - 1]) - np.mean(dp[ix]) * np.mean(dp[ix - 1]))
        out["roll"] = math.sqrt(max(0.0, -cov))
    else:
        out["roll"] = nan
    if k >= W and k > W + 1:
        y = np.log(c)
        dy2 = np.diff(y) ** 2
        sig = math.sqrt(dy2[:k].sum() / k)
        out["csw"] = max((y[k] - y[k - m]) / (sig * math.sqrt(m)) for m in range(1, W + 1))
    else:
        out["csw"] = nan
    h, l = _hl_terms(rh[:k + 1], rl[:k + 1])
    hl = np.log(h / l) ** 2
    den = 3 - 2 * 2 ** .5
    k2 = (8 / math.pi) ** .5
    if k >= sl and k - sl + 1 >= 1:
        beta = float(np.mean([hl[j] + hl[j - 1] for j in range(k - sl + 1, k + 1)]))
        gamma = math.log(max(h[k], h[k - 1]) / min(l[k], l[k - 1])) ** 2
        out["bp_sigma"] = max((2 ** -.5 - 1) * math.sqrt(beta) / (k2 * den) + math.sqrt(gamma / (k2 ** 2 * den)), 0.0)
    else:
        out["bp_sigma"] = nan
    if k >= CS_SL and k - CS_SL + 1 >= 1:
        beta = float(np.mean([hl[j] + hl[j - 1] for j in range(k - CS_SL + 1, k + 1)]))
        gamma = math.log(max(h[k], h[k - 1]) / min(l[k], l[k - 1])) ** 2
        alpha = max((2 ** .5 - 1) * math.sqrt(beta) / den - math.sqrt(gamma / den), 0.0)
        out["cs_spread"] = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
    else:
        out["cs_spread"] = nan
    return out


# ---------------- own HRP ----------------
def my_hrp(R: pd.DataFrame, plant: bool) -> dict:
    cols = list(R.columns)
    n = len(cols)
    if n < 3:
        return {c: 1.0 / n for c in cols}
    cov = R.cov().to_numpy()
    corr = R.corr().to_numpy()
    dist = np.sqrt((1.0 - corr) / 2.0)
    Z = sch.linkage(dist, "single").astype(int)          # the book's Snippet 16.4 passes this square matrix as is
    order = [int(Z[-1, 0]), int(Z[-1, 1])]
    while max(order) >= n:
        nxt = []
        for v in order:
            if v >= n:
                nxt += [int(Z[v - n, 0]), int(Z[v - n, 1])]
            else:
                nxt.append(v)
        order = nxt

    def cvar(items):
        sub = cov[np.ix_(items, items)]
        iv = 1.0 / np.diag(sub)
        iv = iv / iv.sum()
        return float(iv @ sub @ iv)

    w = np.ones(n)
    clusters = [order]
    while clusters:
        new = []
        for c in clusters:
            if len(c) > 1:
                h = len(c) // 2
                new += [c[:h], c[h:]]
        clusters = new
        for i in range(0, len(clusters), 2):
            a, b = clusters[i], clusters[i + 1]
            va, vb = cvar(a), cvar(b)
            alpha = 1.0 - va / (va + vb)
            for x in a:
                w[x] *= alpha
            for x in b:
                w[x] *= 1.0 - alpha
    out = {cols[i]: float(w[i]) for i in range(n)}
    if plant:
        out[cols[0]] *= 1.05
    return out


# ---------------- Chapter 14 / 15 / 10 helpers, own transcriptions ----------------
def my_bets(tp: np.ndarray) -> list:
    """Snippet 14.1 as printed: a bet ends where the position goes flat after a non-flat day, where it flips sign, and on the last day.  (The printed code also counts day 0 when the position is flat then: shift(1) is NaN there and NaN != 0.)"""
    n = len(tp)
    ends = set()
    for i in range(n):
        prev_nonzero = True if i == 0 else tp[i - 1] != 0
        if tp[i] == 0 and prev_nonzero:
            ends.add(i)
        if i >= 1 and tp[i] * tp[i - 1] < 0:
            ends.add(i)
    ends.add(n - 1)
    return sorted(ends)


def my_holding(tp: np.ndarray, plant: bool) -> float:
    n = len(tp)
    num = den = 0.0
    tE = 0.0
    for i in range(1, n):
        d = tp[i] - tp[i - 1]
        if d * tp[i - 1] >= 0:
            if tp[i] != 0:
                tE = (tE * tp[i - 1] + i * d) / tp[i]
        else:
            if tp[i] * tp[i - 1] < 0:
                num += (i - tE) * abs(tp[i - 1])
                den += abs(tp[i - 1])
                tE = float(i)
            else:
                num += (i - tE) * abs(d)
                den += abs(d)
    r = num / den if den > 0 else float("nan")
    return r * 1.01 if plant else r


def my_hhi(x: np.ndarray) -> float:
    n = len(x)
    if n <= 2:
        return float("nan")
    w = x / x.sum()
    h = float((w ** 2).sum())
    return (h - 1.0 / n) / (1.0 - 1.0 / n)


def my_dd_tuw(eq: np.ndarray):
    hwm = np.maximum.accumulate(eq)
    first_idx, mins = {}, {}
    for i, (h, e) in enumerate(zip(hwm, eq)):
        if h not in first_idx:
            first_idx[h] = i
            mins[h] = e
        else:
            mins[h] = min(mins[h], e)
    keep = sorted((first_idx[h], h) for h in first_idx if h > mins[h])
    dd = np.array([1.0 - mins[h] / h for _, h in keep])
    idx = np.array([i for i, _ in keep])
    tuw = np.diff(idx) / 365.25
    return dd, tuw


def b_sr(sl, pt, freq, p):
    return ((pt - sl) * p + sl) / ((pt - sl) * (p * (1 - p)) ** .5) * freq ** .5


def b_hr(sl, pt, freq, tsr):
    a = (freq + tsr ** 2) * (pt - sl) ** 2
    b = (2 * freq * sl - tsr ** 2 * (pt - sl)) * (pt - sl)
    c = freq * sl ** 2
    return (-b + (b ** 2 - 4 * a * c) ** .5) / (2. * a)


def b_freq(sl, pt, p, tsr):
    f = (tsr * (pt - sl)) ** 2 * p * (1 - p) / ((pt - sl) * p + sl) ** 2
    return f if np.isclose(b_sr(sl, pt, f, p), tsr) else float("nan")


def main(run: Path, plant: str | None) -> int:
    t_start = time.time()
    meta = json.loads((run / "meta.json").read_text())
    venue = meta["venue"]
    bar_dir = Path(meta["bar_dir"])
    man = json.loads((bar_dir / "manifest.json").read_text())
    break_s = int(man["break_hours"]) * 3600
    hcap = {"kraken": 600, "alpaca": 150}[venue]
    max_gap = {"kraken": 1, "alpaca": 5}[venue]
    step, seed, cutoff = meta["step"], meta["seed"], meta["cutoff"]
    cost_side = meta["cost_bp_per_side"] / 1e4
    pt, sl, K = float(meta["otr"]["pt"]), float(meta["otr"]["sl"]), int(meta["otr"]["K"])
    problems: list[str] = []
    checked = 0

    def cmp(name, mine, theirs, tol=1e-9):
        nonlocal checked
        checked += 1
        if mine is not None and theirs is None and isinstance(mine, float) and np.isnan(mine):
            return                                                    # the bot writes NaN as null
        if mine is None or theirs is None:
            if not (mine is None and theirs is None):
                problems.append(f"{name}: independent {mine!r} vs backtest {theirs!r}")
            return
        if np.isnan(mine) and np.isnan(theirs):
            return
        if np.isnan(mine) or np.isnan(theirs) or abs(mine - theirs) > tol * max(1.0, abs(mine)):
            problems.append(f"{name}: independent {mine!r} vs backtest {theirs!r}")

    ev = pd.read_csv(run / "events.csv", keep_default_na=False, na_values=[""], dtype={"symbol": str})
    sy = pd.read_csv(run / "symbols.csv", keep_default_na=False, na_values=[""], dtype={"symbol": str})
    syms_all = sorted(sy["symbol"])
    ev_syms = sorted(ev["symbol"].unique())
    bars = {s: load_bars(bar_dir, s, break_s) for s in syms_all}
    print(f"[{time.time() - t_start:5.0f}s] bars loaded: {len(bars)} symbols", flush=True)
    cutoff_ts = int(pd.Timestamp(cutoff).timestamp())
    cutoff_day = cutoff_ts // DAY
    N = len(ev)
    h_s = int(round(H_DAYS * DAY))

    # =============== 1. events ===============
    cmp("events kept", N, meta["events"])
    cmp("events cusum total", int(sy["events_total"].sum()), meta["events_cusum"])
    if (ev["bin_book"] == 0).any() or not (ev["bin"].to_numpy() == (ev["ret_side"].to_numpy() > 0)).all() or not (np.sign(ev["ret_side"].to_numpy()) == ev["bin_book"].to_numpy()).all():
        problems.append("labels: bin / bin_book do not follow the sign of ret_side, or a zero class survived dropLabels")
    cmp("events before dropLabels", int(sy["events_kept"].sum()), meta["events_before_dropLabels"])
    cmp("dropped by 3.8", meta["events_before_dropLabels"] - N, int(sy["zero_return_kept"].sum()))
    cmp("time decay const", 1.0, meta["time_decay_4_11"]["const"])
    cmp("time decay slope", 0.0, meta["time_decay_4_11"]["slope"])
    cmp("events pre cutoff", int((ev["t1"].to_numpy() < cutoff_ts).sum()), meta["events_pre_cutoff"])
    n_label_bad = n_edge = 0
    i0 = np.zeros(N, dtype=np.int64)
    sg_vec = np.zeros(N)
    cusum_total = 0
    cusum_missing = 0
    for s in syms_all:
        t, c, seg, st = bars[s]
        n = len(t)
        lc = np.log(c)
        r = np.concatenate([[np.nan], np.diff(lc)])
        r[1:][(st[1:] - t[:-1]) > break_s] = np.nan
        sigma = pd.Series(r).ewm(span=SPAN, min_periods=SPAN).std().to_numpy()
        w = np.minimum(NDAY_W, (t - t[0]).astype(np.float64))
        lo = np.searchsorted(t, t - w, side="right")
        with np.errstate(divide="ignore", invalid="ignore"):
            bpd = (np.arange(n) + 1 - lo) / (w / DAY)
        bpd[w <= 0] = np.nan
        thr = sigma * np.sqrt(bpd)
        sp = sn = 0.0
        evs = []
        for i in range(1, n):
            x = r[i] if np.isfinite(r[i]) else 0.0
            sp, sn = max(0.0, sp + x), min(0.0, sn + x)
            if not np.isfinite(thr[i]):
                continue
            if sn < -thr[i]:
                sn = 0.0
                evs.append(i)
            elif sp > thr[i]:
                sp = 0.0
                evs.append(i)
        evs = np.asarray(evs, dtype=np.int64)
        cusum_total += len(evs)
        # daily volatility, Snippet 3.1 exactly as printed: searchsorted(index - 1 day) -> the bar before that position, no hole rule, ewm(span=100) with no min_periods
        pos0 = np.searchsorted(t, t - DAY, side="left")
        kept = pos0 > 0
        rr = np.full(n, np.nan)
        rr[kept] = c[kept] / c[pos0[kept] - 1] - 1.0
        f0 = int(np.argmax(kept)) if kept.any() else n
        dvol = np.full(n, np.nan)
        if f0 < n:
            dvol[f0:] = pd.Series(rr[f0:]).ewm(span=100).std().to_numpy()
        g = ev[ev["symbol"] == s]
        if not len(g):
            continue
        ii = g.index.to_numpy()
        p, t0, t1 = g["pos"].to_numpy(np.int64), g["t0"].to_numpy(np.int64), g["t1"].to_numpy(np.int64)
        i0[ii] = p
        sg_vec[ii] = sigma[p]
        if (p >= n).any() or not (t[p] == t0).all() or not (np.searchsorted(t, t0) == p).all():
            problems.append(f"{s}: event positions do not sit on the bars by timestamp")
            continue
        miss = np.setdiff1d(p, evs)
        cusum_missing += len(miss)
        cmp(f"{s} sigma (all events)", float(np.nanmax(np.abs(sigma[p] - g["sigma"].to_numpy()) / g["sigma"].to_numpy())), 0.0)
        cmp(f"{s} trgt (all events)", float(np.nanmax(np.abs(dvol[p] - g["trgt"].to_numpy()) / g["trgt"].to_numpy())), 0.0)
        cmp(f"{s} log_trgt", float(np.nanmax(np.abs(np.log(g["trgt"].to_numpy()) - g["log_trgt"].to_numpy()))), 0.0)
        trg = g["trgt"].to_numpy()
        t1p, kind, retv = g["t1pos"].to_numpy(np.int64), g["kind"].to_numpy(np.int64), g["ret_side"].to_numpy()
        fac = 0.97 if plant == "label" else 1.0
        for q in range(len(g)):
            pp = int(p[q])
            iv = int(np.searchsorted(t, t0[q] + h_s, side="left"))
            L = min(iv - pp, hcap)
            R = c[pp + 1:pp + L + 1] / c[pp] - 1.0
            ptb, slb = PTSL * trg[q] * fac, -PTSL * trg[q]
            hp = np.flatnonzero(R > ptb)
            hs = np.flatnonzero(R < slb)
            jp = int(hp[0]) if len(hp) else L + 5
            js = int(hs[0]) if len(hs) else L + 5
            j = min(jp, js, L - 1)
            kd = 1 if (jp < js and jp < L) else (-1 if (js < jp and js < L) else 0)
            exp_t1 = pp + 1 + j
            if exp_t1 != t1p[q] or kd != kind[q] or abs(R[j] - retv[q]) > 1e-6 * max(1.0, abs(R[j])) or t[exp_t1] != t1[q]:
                # a razor-edge barrier decision (the bot casts the path to float32 before comparing) is not an error
                jb = int(t1p[q]) - pp - 1
                if 0 <= jb < L:
                    d1 = min(np.min(np.abs(R - ptb)), np.min(np.abs(R - slb)))
                    if d1 < 1e-6:
                        n_edge += 1
                        continue
                n_label_bad += 1
                if n_label_bad <= 5:
                    problems.append(f"{s} event pos {pp}: independent exit bar {exp_t1} kind {kd} ret {R[j]:.8g} vs saved {t1p[q]} {kind[q]} {retv[q]:.8g}")
        checked += 4 * len(g)
        # uniqueness and attribution weights where no event of the symbol was dropped by Snippet 3.8
        if int(sy.loc[sy["symbol"] == s, "zero_return_kept"].iloc[0]) == 0:
            diff = np.zeros(n + 1)
            for a_, b_ in zip(p, t1p):
                diff[a_] += 1.0
                diff[b_ + 1] -= 1.0
            conc = np.cumsum(diff)[:n]
            ret1 = np.concatenate([[0.0], np.diff(lc)])
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = np.where(conc > 0, 1.0 / conc, 0.0)
                contrib = np.where(conc > 0, ret1 / conc, 0.0)
            u_ = np.array([inv[a_:b_ + 1].mean() for a_, b_ in zip(p, t1p)])
            wr = np.array([abs(contrib[a_:b_ + 1].sum()) for a_, b_ in zip(p, t1p)])
            if plant == "weight":
                u_ = u_ * 1.01
            cmp(f"{s} uniqueness (max rel diff)", float(np.max(np.abs(u_ - g["u"].to_numpy()) / g["u"].to_numpy())), 0.0, 1e-9)
            cmp(f"{s} attribution weight (max abs diff)", float(np.max(np.abs(wr - g["w_raw"].to_numpy()))), 0.0, 1e-9)
    cmp("cusum events (own filter, all symbols)", cusum_total, meta["events_cusum"])
    cmp("events missing from the independent CUSUM set", cusum_missing, 0)
    cmp("labels disagreeing with the independent triple barrier", n_label_bad, 0)
    pos_w = ev["w_raw"].to_numpy() > 0
    ratio = (ev["w"].to_numpy()[pos_w] / ev["w_raw"].to_numpy()[pos_w])
    cmp("time-decay: w / w_raw constant (max rel spread, events with a positive weight)", float(np.max(np.abs(ratio / ratio[0] - 1.0))), 0.0)
    cmp("events with w_raw = 0 also have w = 0", int(((~pos_w) & (ev["w"].to_numpy() != 0)).sum()), 0)
    print(f"[{time.time() - t_start:5.0f}s] events checked ({n_edge} razor-edge barrier decisions tolerated, {n_label_bad} real label differences)", flush=True)

    # =============== 2. features brute force on a sample ===============
    rng = np.random.default_rng(12345)
    sample = rng.choice(N, size=min(300, N), replace=False)
    ref_cache = {}
    nf = 0
    for e in sample:
        s = ev.at[e, "symbol"]
        if s not in ref_cache:
            ref_cache[s] = load_ref(bar_dir, s)
        rt, rh, rl, rc, rv = ref_cache[s]
        k = int(np.searchsorted(rt, int(ev.at[e, "t0"]), side="right")) - 1
        if k < 0 or int(ev.at[e, "t0"]) - rt[k] > STATE_TOL:
            problems.append(f"event {e}: no reference row within the tolerance")
            continue
        bf = brute_features(rh, rl, rc, rv, k)
        for nm, v in bf.items():
            sv = float(ev.at[e, nm])
            if plant == "feature" and nm == "roll":
                v = v * 1.01
            if not np.isfinite(v):
                problems.append(f"event {e} {nm}: independent value is not finite, saved {sv!r}")
                continue
            cmp(f"event {e} {nm}", v, sv, 1e-8)
            nf += 1
    print(f"[{time.time() - t_start:5.0f}s] {nf} feature values re-derived", flush=True)

    # =============== 3. Chapter 13 ===============
    c13 = json.loads((run / "ch13.json").read_text())
    ymap = {s: bars[s] for s in ev_syms}
    sig_ev, u_ev = ev["sigma"].to_numpy(float), ev["u"].to_numpy(float)
    t0_ev, t1_ev = ev["t0"].to_numpy(np.int64), ev["t1"].to_numpy(np.int64)
    sym_ev = ev["symbol"].to_numpy()
    qrows, usable = [], np.zeros(N, dtype=bool)
    Q = np.full((N, K_OTR), np.nan, dtype=np.float32)
    for s in ev_syms:
        t, c, seg, _ = ymap[s]
        lc = np.log(c)
        ii = np.flatnonzero(sym_ev == s)
        pp = i0[ii]
        ok = pp + K_OTR < len(t)
        ii, pp = ii[ok], pp[ok]
        same = seg[pp + K_OTR] == seg[pp]
        ii, pp = ii[same], pp[same]
        Q[ii] = ((lc[pp[:, None] + np.arange(1, K_OTR + 1)[None, :]] - lc[pp][:, None]) / sig_ev[ii][:, None]).astype(np.float32)
        usable[ii] = True
    trm = usable & (t1_ev < cutoff_ts)
    tem = usable & (t0_ev >= cutoff_ts)
    cmp("Ch.13 fit events", int(trm.sum()), c13["fit_events"])
    cmp("Ch.13 test events", int(tem.sum()), c13["test_events"])
    qt = Q[trm].astype(np.float64)
    wt = u_ev[trm]
    X = np.concatenate([np.zeros((len(qt), 1)), qt[:, :-1]], axis=1).ravel()
    Y = qt.ravel()
    Wf = np.repeat(wt, K_OTR)
    sw = Wf.sum()
    mx, my = (Wf * X).sum() / sw, (Wf * Y).sum() / sw
    phi = float((Wf * (Y - my) * (X - mx)).sum() / (Wf * (X - mx) ** 2).sum())
    a_ = float(my - phi * mx)
    sg_fit = float(math.sqrt((Wf * (Y - a_ - phi * X) ** 2).sum() / sw))
    if plant == "ch13":
        phi *= 1.001
    cmp("Ch.13 phi", phi, c13["phi"])
    cmp("Ch.13 a", a_, c13["a"])
    cmp("Ch.13 sigma", sg_fit, c13["sigma_fit"])
    cmp("Ch.13 half-life (defined only for 0 < phi < 1; the bot writes null otherwise)", (-math.log(2.0) / math.log(phi)) if 0.0 < phi < 1.0 else None, c13["half_life_bars"])
    cmp("Ch.13 forecast", a_ / (1 - phi), c13["forecast_level_sigma"])
    del qt, X, Y, Wf

    def mesh_sharpe(P):
        n, k = P.shape
        runmax = np.maximum.accumulate(P, axis=1)
        runmin = np.minimum.accumulate(P, axis=1)
        out = np.empty((20, 20))
        rows = np.arange(n)
        fp = [(runmax <= lv).sum(axis=1) for lv in MESH]
        fs = [(runmin >= -lv).sum(axis=1) for lv in MESH]
        for i in range(20):
            for j in range(20):
                tau = np.minimum(np.minimum(fp[i], fs[j]), k - 1)
                x = P[rows, tau]
                sd = x.std()
                out[i, j] = x.mean() / sd if sd > 0 else np.nan
        return out

    rg = np.random.default_rng(seed + 13)
    eps = rg.standard_normal((100000, K_OTR)) * sg_fit
    P = np.empty_like(eps)
    pcur = np.zeros(100000)
    for k in range(K_OTR):
        pcur = a_ + phi * pcur + eps[:, k]
        P[:, k] = pcur
    del eps
    sh_sim = mesh_sharpe(P)
    del P
    saved_sim = np.loadtxt(run / "ch13_mesh_sharpe_simulated.csv", delimiter=",")
    cmp("Ch.13 simulated mesh (max abs diff, 400 nodes)", float(np.nanmax(np.abs(sh_sim - saved_sim))), 0.0, 1e-6)
    checked += 399
    i_, j_ = divmod(int(np.nanargmax(sh_sim)), 20)
    cmp("Ch.13 chosen pt", MESH[i_], c13["chosen_pt"])
    cmp("Ch.13 chosen sl", MESH[j_], c13["chosen_sl"])
    cmp("Ch.13 chosen pt (used by the account)", MESH[i_], pt)
    cmp("Ch.13 chosen sl (used by the account)", MESH[j_], sl)
    cmp("Ch.13 simulated Sharpe at chosen", float(sh_sim[i_, j_]), c13["simulated_sharpe_at_chosen"], 1e-7)
    cmp("Ch.13 simulated median", float(np.nanmedian(sh_sim)), c13["simulated_sharpe_median_over_mesh"], 1e-7)
    sh_oos = mesh_sharpe(Q[tem].astype(np.float64))
    saved_oos = np.loadtxt(run / "ch13_mesh_sharpe_oos.csv", delimiter=",")
    cmp("Ch.13 out-of-sample mesh (max abs diff, 400 nodes)", float(np.nanmax(np.abs(sh_oos - saved_oos))), 0.0, 1e-6)
    checked += 399
    d2 = c13["definition_2"]
    cmp("Ch.13 OOS Sharpe of chosen", float(sh_oos[i_, j_]), d2["oos_sharpe_chosen"], 1e-7)
    cmp("Ch.13 OOS median", float(np.nanmedian(sh_oos)), d2["oos_sharpe_mesh_median"], 1e-7)
    cmp("Ch.13 OOS best in hindsight", float(np.nanmax(sh_oos)), d2["oos_sharpe_mesh_best_in_hindsight"], 1e-7)
    if bool(sh_oos[i_, j_] < np.nanmedian(sh_oos)) != bool(d2["overfit"]):
        problems.append("Ch.13 Definition 2 flag differs")
    for x in c13["verbatim_loop_check"]:
        if x["sharpe_verbatim_loop"] is not None and x["sharpe_mesh"] is not None and abs(x["sharpe_verbatim_loop"] - x["sharpe_mesh"]) > 0.12:
            problems.append(f"Ch.13 verbatim loop vs mesh differ by more than sampling error at node {x['pt']}/{x['sl']}")
    del Q
    print(f"[{time.time() - t_start:5.0f}s] Ch.13 re-derived", flush=True)

    # =============== 4. model quality and Ch.9 table ===============
    y = (ev["ret_side"].to_numpy() > 0).astype(int)
    wf = pd.read_csv(run / "walk_forward_predictions.csv")
    cp = pd.read_csv(run / "cpcv_predictions.csv.gz")
    w_idx, w_p = wf["event"].to_numpy(np.int64), wf["p"].to_numpy(float)
    cmp("wf auc", auc(y[w_idx], w_p), meta["walk_forward"]["auc"])
    pred = w_p > .5
    yb = y[w_idx].astype(bool)
    tp_, fp_, tn_, fn_ = int((yb & pred).sum()), int((~yb & pred).sum()), int((~yb & ~pred).sum()), int((yb & ~pred).sum())
    T = meta["walk_forward"]["table_14_1"]
    for nm, v in (("tp", tp_), ("fp", fp_), ("tn", tn_), ("fn", fn_)):
        cmp(f"Table 14.1 {nm}", v, T[nm])
    cmp("Table 14.1 accuracy", (tp_ + tn_) / len(yb), T["accuracy"])
    cmp("Table 14.1 precision", tp_ / (tp_ + fp_), T["precision"])
    cmp("Table 14.1 recall", tp_ / (tp_ + fn_), T["recall"])
    pr_, rc_ = tp_ / (tp_ + fp_), tp_ / (tp_ + fn_)
    cmp("Table 14.1 F1", 2 * pr_ * rc_ / (pr_ + rc_), T["f1"])
    cmp("wf side accuracy", float((pred == yb).mean()), meta["walk_forward"]["side_accuracy"])
    if not (t0_ev[w_idx] >= cutoff_ts).all():
        problems.append("a walk-forward prediction is for an event that starts before the cutoff")
    paths = {int(k): g for k, g in cp.groupby("path")}
    for k, g in sorted(paths.items()):
        e_, p_ = g["event"].to_numpy(np.int64), g["p"].to_numpy(float)
        if len(np.unique(e_)) != len(e_):
            problems.append(f"cpcv path {k}: an event appears twice")
        cmp(f"cpcv path {k} auc", auc(y[e_], p_), meta["cpcv"]["auc_by_path"][k])
    R9 = pd.read_csv(run / "search.csv")
    tj = json.loads((run / "tuned_forest.json").read_text())
    for _, r_ in R9.iterrows():
        cmp(f"search candidate {int(r_['cand'])} mean of folds", float(np.mean(json.loads(r_["fold_scores"]))), float(r_["mean_neg_log_loss"]))
    cmp("search chosen = best mean", int(R9["mean_neg_log_loss"].idxmax()), int(tj["chosen_candidate"]))
    cmp("trials = number of search candidates", len(R9), meta["dsr"]["last_365_days"]["trials"])

    # =============== 5. account ===============
    sig_e = sig_ev
    sym_e = sym_ev
    exit_bar = {1.0: np.zeros(N, dtype=np.int64), -1.0: np.zeros(N, dtype=np.int64)}
    for sd in (1.0, -1.0):
        for s in ev_syms:
            ii = np.flatnonzero(sym_e == s)
            t, c, seg, _ = ymap[s]
            lc = np.log(c)
            for e in ii:
                a = int(i0[e])
                hi = min(a + K, len(t) - 1)
                seg_ok = seg[a + 1:hi + 1] == seg[a]
                m = int(seg_ok.argmin()) if not seg_ok.all() else len(seg_ok)
                if m == 0:
                    exit_bar[sd][e] = a
                    continue
                q = sd * (lc[a + 1:a + 1 + m] - lc[a]) / sig_e[e]
                if plant == "exit":
                    q = q * 0.97
                hit = np.flatnonzero((q >= pt) | (q <= -sl))
                exit_bar[sd][e] = a + 1 + int(hit[0]) if len(hit) else a + m
    first_bar_day = min(int(ymap[s][0][0] // DAY) for s in ev_syms)
    last_bar_day = max(int(ymap[s][0][-1] // DAY) for s in ev_syms)
    D0 = first_bar_day
    ND = last_bar_day - D0 + 1
    nS = len(syms_all)
    sidx = {s: i for i, s in enumerate(syms_all)}
    cs = cost_side * (1.1 if plant == "cost" else 1.0)

    caps_wf: list = []

    def nets(idx, sides, sizes, want_pos=False, want_cap=False):
        M = np.zeros((nS, ND))
        POS = np.zeros((nS, ND)) if want_pos else None
        for s in ev_syms:
            si = sidx[s]
            sel = np.flatnonzero(sym_e[idx] == s)
            if not len(sel):
                continue
            t, c, _, _ = ymap[s]
            n = len(t)
            tot, cnt = np.zeros(n), np.zeros(n)
            for j in sel:
                e = int(idx[j])
                a, b = int(i0[e]), int(exit_bar[float(sides[j])][e])
                tot[a:b] += sides[j] * sizes[j]
                cnt[a:b] += 1
            avg = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
            day = (t // DAY) - D0
            gross = avg[:-1] * (c[1:] / c[:-1] - 1.0)
            prev = np.concatenate([[0.0], avg[:-1]])
            cost = cs * np.abs(avg - prev)
            M[si] += np.bincount(day[1:], weights=gross, minlength=ND)
            M[si] -= np.bincount(day, weights=cost, minlength=ND)
            if want_cap:
                vol = load_vol(bar_dir, s)
                wv = Wm[si, day]
                e_ = wv * avg
                de = np.abs(np.diff(e_, prepend=0.0))
                absday = t // DAY
                mk = (de > 1e-12) & (absday >= cutoff_day) & (absday <= last_day)
                if mk.any():
                    caps_wf.append(c[mk] * vol[mk] / de[mk])
            if want_pos:
                lastidx = np.flatnonzero(np.diff(day, append=day[-1] + 1) != 0)
                pd_ = np.full(ND, np.nan)
                pd_[day[lastidx]] = avg[lastidx]
                POS[si] = pd.Series(pd_).ffill().fillna(0.0).to_numpy()
        return M, POS

    # daily returns for HRP over every loaded symbol
    lastc = {}
    for s in syms_all:
        t, c, _, _ = bars[s]
        lastc[s] = pd.Series(c, index=t // DAY).groupby(level=0).last()
    DC = pd.DataFrame(lastc).sort_index()
    DC = DC[DC.notna().mean(axis=1) >= .8]
    dd_ = DC.index.to_numpy()
    Rr = DC.pct_change(fill_method=None)
    okrow = np.zeros(len(DC), dtype=bool)
    okrow[1:] = np.diff(dd_) <= max_gap
    Rr = Rr[okrow]
    R_days = Rr.index.to_numpy()
    last_day, first_day_all = last_bar_day, first_bar_day
    first_event_day = int(t0_ev.min() // DAY)
    cmp("last_day", last_day, int(pd.Timestamp(meta["last_day"]).timestamp() // DAY))
    cmp("first_day", first_day_all, int(pd.Timestamp(meta["first_day"]).timestamp() // DAY))
    Wreb = int(meta["reb_window"])
    start0 = min(first_day_all, first_event_day) + Wreb

    def quarter(d):
        dt = datetime.fromtimestamp(d * DAY, timezone.utc)
        return dt.year * 4 + (dt.month - 1) // 3

    keys = [start0] + [d for d in range(start0 + 1, last_day + 1) if quarter(d) != quarter(d - 1)]
    cmp("HRP rebalances", len(keys), meta["hrp_rebalances"])
    sched = {}
    saved_w = pd.read_csv(run / "hrp_weights.csv", index_col=0, keep_default_na=False, na_values=[""])
    for r_ in keys:
        win = Rr[R_days < r_].tail(Wreb)
        if len(win) < Wreb * .8:
            sched[r_] = {}
            continue
        ok = [c for c in win.columns if win[c].notna().sum() >= .8 * len(win) and win[c].fillna(0.0).std() > 0]
        w = my_hrp(win[ok].fillna(0.0), plant == "hrp")
        tot = sum(w.values())
        sched[r_] = {k: v / tot for k, v in w.items()}
    lab = {pd.Timestamp(k * DAY, unit="s").strftime("%Y-%m-%d"): k for k in keys}
    nchk = 0
    for dstr, row in saved_w.iterrows():
        mine = sched[lab[dstr]]
        for s_ in saved_w.columns:
            v = row[s_]
            mv = mine.get(s_, float("nan"))
            if pd.isna(v):
                if s_ in mine:
                    problems.append(f"hrp weight {dstr} {s_}: saved table has no weight, checker has one")
                continue
            cmp(f"hrp weight {dstr} {s_}", mv, float(v))
            nchk += 1
    Wm = np.zeros((nS, ND))
    ks = sorted(sched)
    for q_, st_ in enumerate(ks):
        en = ks[q_ + 1] if q_ + 1 < len(ks) else last_day + 1
        for s in syms_all:
            Wm[sidx[s], st_ - D0:en - D0] = sched[st_].get(s, 0.0)

    def stats(r):
        eq, peak, mdd = 1.0, 1.0, 0.0
        for x in r:
            eq *= 1.0 + x
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1.0)
        m = sum(r) / len(r)
        sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
        return 100.0 * (eq - 1.0), 100.0 * mdd, (m / sd * math.sqrt(365)) if sd > 0 else float("nan")

    S = pd.read_csv(run / "summary.csv")
    side_w = np.where(w_p > .5, 1.0, -1.0)
    size_w = np.array([bet_size(float(max(q, 1 - q)), step, plant == "bet_size") for q in w_p])
    scopes = {"wf": (w_idx, side_w, size_w)}
    for k, g in sorted(paths.items()):
        e_, p_ = g["event"].to_numpy(np.int64), g["p"].to_numpy(float)
        scopes[f"cpcv{k}"] = (e_, np.where(p_ > .5, 1.0, -1.0), np.array([bet_size(float(max(q, 1 - q)), step, plant == "bet_size") for q in p_]))
    if len(S) != 2 * len(scopes):
        problems.append(f"summary.csv has {len(S)} rows, expected {2 * len(scopes)}")
    daily_by_scope, POS_wf = {}, None
    daily_before = {}
    for scope, (idx, sd, sz) in scopes.items():
        M, POS = nets(idx, sd, sz, want_pos=True, want_cap=(scope == "wf"))
        daily0 = (Wm * M).sum(axis=0)
        # section 2.4.1 item 1: c_t = tau * sum_i (|w_old,i| + |w_new,i|) |position_i at the end of the previous day| on every rebalance day
        cvec = np.zeros(ND)
        old = {}
        for st_ in sorted(sched):
            new = sched[st_]
            if D0 + 1 <= st_ <= last_day:
                cc = 0.0
                for s in syms_all:
                    pv = abs(float(POS[sidx[s], st_ - 1 - D0]))
                    if pv:
                        cc += (abs(old.get(s, 0.0)) + abs(new.get(s, 0.0))) * pv
                cvec[st_ - D0] = cost_side * cc * (0.9 if plant == "rebal" else 1.0)
            old = new
        daily_before[scope] = daily0
        daily = daily0 - cvec
        daily_by_scope[scope] = daily
        if scope == "wf":
            POS_wf = POS
        rows = S[S["scope"] == scope]
        if len(rows) != 2:
            problems.append(f"{scope}: {len(rows)} summary rows, expected 2")
            continue
        for _, row in rows.iterrows():
            lo = int(pd.Timestamp(row["start_day"]).timestamp() // DAY)
            exp_lo = last_day - 364 if row["span"] == "last_365_days" else (cutoff_day if row["span"] == "whole_walk_forward_span" else first_day_all + Wreb + 5)
            if lo != exp_lo:
                problems.append(f"{scope}/{row['span']}: start day {lo} vs expected {exp_lo}")
            g_, d_, sh_ = stats(daily[lo - D0:last_day - D0 + 1].tolist())
            nm = f"{scope}/{row['span']}"
            cmp(nm + "/gain_pct", g_, float(row["gain_pct"]))
            cmp(nm + "/max_drawdown_pct", d_, float(row["max_drawdown_pct"]))
            cmp(nm + "/sharpe", sh_, float(row["daily_sharpe_x_sqrt365"]))
    for sname, lo_ in (("last_365_days", last_day - 364), ("whole_walk_forward_span", cutoff_day)):
        m_ = meta["rebalance_cost_effect"][sname]
        cmp(f"2.4.1 rebalance cost {sname}: gain before", stats(daily_before["wf"][lo_ - D0:last_day - D0 + 1].tolist())[0], m_["gain_pct_before"])
        cmp(f"2.4.1 rebalance cost {sname}: gain after", stats(daily_by_scope["wf"][lo_ - D0:last_day - D0 + 1].tolist())[0], m_["gain_pct_after"])
        ch_ = (daily_before["wf"] - daily_by_scope["wf"])[lo_ - D0:last_day - D0 + 1]
        cmp(f"2.4.1 rebalance cost {sname}: total charge % of capital", 100.0 * float(ch_.sum()), m_["total_charge_pct_of_capital"])
        cmp(f"2.4.1 rebalance cost {sname}: rebalances charged", int((ch_ > 0).sum()), m_["rebalances_charged"])
    cap = np.concatenate(caps_wf) if caps_wf else np.zeros(0)
    cv = meta["volume_capacity"]
    cmp("2.4.1 volume: trades", len(cap), cv["n_trades"])
    if len(cap):
        cmp("2.4.1 volume: capacity min", float(cap.min()), cv["capacity_min"], 1e-9)
        cmp("2.4.1 volume: capacity p01", float(np.quantile(cap, .01)), cv["capacity_p01"], 1e-9)
        cmp("2.4.1 volume: capacity p05", float(np.quantile(cap, .05)), cv["capacity_p05"], 1e-9)
        cmp("2.4.1 volume: capacity median", float(np.median(cap)), cv["capacity_median"], 1e-9)
        for kk, vv in cv["share_not_fitting"].items():
            cmp(f"2.4.1 volume: share of trades not fitting K={kk}", float((cap < float(kk)).mean()), vv)
    print(f"[{time.time() - t_start:5.0f}s] account re-derived ({nchk} HRP weights)", flush=True)

    # =============== 6. PSR / DSR ===============
    def moments(r):
        a = np.asarray(r, float)
        n = len(a)
        m, s = a.mean(), a.std(ddof=1)
        z = (a - a.mean()) / a.std(ddof=0)
        return n, m / s, float((z ** 3).mean()), float((z ** 4).mean())

    ntr = len(R9)
    for sname, lo in (("last_365_days", last_day - 364), ("whole_walk_forward_span", cutoff_day)):
        n, sr, sk, ku = moments(daily_by_scope["wf"][lo - D0:last_day - D0 + 1])
        if sname == "last_365_days":
            psr_sr = [moments(daily_by_scope[k][last_day - 364 - D0:last_day - D0 + 1])[1] for k in daily_by_scope if k != "wf"]
            vp = float(np.var(psr_sr, ddof=1)) if len(psr_sr) > 1 else float("nan")
        else:
            vp = float("nan")
        va = (1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2) / max(n - 1.0, 1.0)
        v = float(np.nanmax([vp, va]))
        star = math.sqrt(v) * ((1.0 - EULER) * norm.ppf(1.0 - 1.0 / ntr) + EULER * norm.ppf(1.0 - 1.0 / (ntr * math.e)))
        if plant == "dsr":
            star *= 1.05
        psr0 = norm.cdf(sr * math.sqrt(n - 1.0) / math.sqrt(1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2))
        dsr = norm.cdf((sr - star) * math.sqrt(n - 1.0) / math.sqrt(1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2))
        m = meta["dsr"][sname]
        for nm, mine, theirs in (("n_days", n, m["n_days"]), ("sr_daily", sr, m["sr_daily"]), ("skew", sk, m["skew"]), ("kurt", ku, m["kurt"]), ("var_across_cpcv_paths", vp, m["var_across_cpcv_paths"]),
                                 ("var_asymptotic", va, m["var_asymptotic"]), ("sr_star", star, m["sr_star"]), ("psr_vs_0", psr0, m["psr_vs_0"]), ("dsr", dsr, m["dsr"])):
            cmp(f"dsr/{sname}/{nm}", mine, theirs, 1e-8)

    # =============== 7. Chapter 14 ===============
    lo = last_day - 364
    nd = last_day - lo + 1
    E_day = (Wm[:, lo - D0:] * POS_wf[:, lo - D0:]).sum(axis=0)
    r_year = daily_by_scope["wf"][lo - D0:]
    tpos_saved = pd.read_csv(run / "ch14_target_position.csv", index_col=0)
    rets_saved = pd.read_csv(run / "ch14_daily_return.csv", index_col=0)
    cmp("Ch.14 target position (max abs diff over the year)", float(np.max(np.abs(E_day - tpos_saved.iloc[:, 0].to_numpy()))), 0.0, 1e-9)
    cmp("Ch.14 daily return (max abs diff over the year)", float(np.max(np.abs(r_year - rets_saved.iloc[:, 0].to_numpy()))), 0.0, 1e-9)
    checked += 2 * nd - 2
    ch = meta["ch14"]
    eq = np.cumprod(1.0 + r_year)
    years = nd / 365.25
    cmp("Ch.14 gain %", 100.0 * (eq[-1] - 1.0), ch["gain_pct"])
    cmp("Ch.14 annualised return Eq.49 %", 100.0 * (eq[-1] ** (1.0 / years) - 1.0), ch["annualised_return_pct_eq49"])
    cmp("Ch.14 days with non-zero position", int((E_day != 0).sum()), ch["days_with_nonzero_position"])
    cmp("Ch.14 max |position|", float(np.abs(E_day).max()), ch["max_abs_position"])
    cmp("Ch.14 mean |position|", float(np.abs(E_day).mean()), ch["mean_abs_position"])
    bets = my_bets(E_day)
    cmp("Ch.14 bets (14.1)", len(bets), ch["bets_14_1"])
    hp = my_holding(E_day, plant == "ch14")
    cmp("Ch.14 holding period (14.2)", hp, ch["avg_holding_period_days_14_2"], 1e-9)
    prev = -1
    bret = []
    for b in bets:
        bret.append(float(np.prod(1.0 + r_year[prev + 1:b + 1]) - 1.0))
        prev = b
    bret = np.array(bret)
    cmp("Ch.14 HHI positive (14.3)", my_hhi(bret[bret >= 0]), ch["hhi_positive_returns_14_3"])
    cmp("Ch.14 HHI negative (14.3)", my_hhi(bret[bret < 0]), ch["hhi_negative_returns_14_3"])
    days_idx = pd.to_datetime((np.arange(lo, last_day + 1) * DAY), unit="s")
    mon = pd.Series(1, index=days_idx[bets]).groupby(pd.Grouper(freq="ME")).count().to_numpy(float)
    cmp("Ch.14 HHI bets per month (14.3)", my_hhi(mon), ch["hhi_bets_per_month_14_3"])
    dd, tuw = my_dd_tuw(eq)
    cmp("Ch.14 drawdown count (14.4)", len(dd), ch["drawdowns_14_4"]["count"])
    cmp("Ch.14 max drawdown % (14.4)", 100.0 * dd.max() if len(dd) else 0.0, ch["drawdowns_14_4"]["max_pct"])
    cmp("Ch.14 p95 drawdown % (14.4)", 100.0 * float(np.percentile(dd, 95)) if len(dd) else 0.0, ch["drawdowns_14_4"]["p95_pct"])
    cmp("Ch.14 time-under-water count (14.4)", len(tuw), ch["time_under_water_years_14_4"]["count"])
    cmp("Ch.14 time-under-water max", float(tuw.max()) if len(tuw) else None, ch["time_under_water_years_14_4"]["max"])
    cmp("Ch.14 time-under-water p95", float(np.percentile(tuw, 95)) if len(tuw) else None, ch["time_under_water_years_14_4"]["p95"])

    # =============== 8. Chapter 15 ===============
    lo_t = (last_day - 364) * DAY
    rets15 = []
    for k, e in enumerate(w_idx):
        if t0_ev[e] < lo_t:
            continue
        s = sym_e[e]
        c = ymap[s][1]
        b = int(exit_bar[float(side_w[k])][e])
        rets15.append(side_w[k] * (c[b] / c[int(i0[e])] - 1.0) - 2.0 * cost_side)
    r15 = np.array(rets15)
    c15 = meta["ch15"]
    cmp("Ch.15 bets", len(r15), c15["n_bets"])
    prec = float((r15 > 0).mean()) * (1.001 if plant == "ch15" else 1.0)
    cmp("Ch.15 t-value of precision (Eq.54)", (2 * prec - 1) / (2 * math.sqrt(prec * (1 - prec))) * math.sqrt(len(r15)), c15["t_value_of_precision_eq54"])
    rp, rn = float(r15[r15 > 0].mean()), float(r15[r15 <= 0].mean())
    cmp("Ch.15 observed precision", prec, c15["observed_precision"])
    cmp("Ch.15 mean positive", rp, c15["mean_positive_return"])
    cmp("Ch.15 mean non-positive", rn, c15["mean_nonpositive_return"])
    freq = float(len(r15))
    for tsr in (1.0, 2.0):
        tg = c15["targets"][str(tsr)]
        cmp(f"Ch.15 implied precision (15.3) SR {tsr}", b_hr(rn, rp, freq, tsr), tg["implied_precision_15_3"])
        fq = b_freq(rn, rp, prec, tsr)
        cmp(f"Ch.15 implied frequency (15.4) SR {tsr}", fq, tg["implied_frequency_15_4_at_observed_precision"])
        cmp(f"Ch.15 Sharpe at observed precision (Eq.56) SR {tsr}", b_sr(rn, rp, freq, prec), tg["sharpe_of_observed_precision_eq56"])
        thr = b_hr(rn, rp, freq, tsr)
        p_ = float((r15 > 0).mean())
        cmp(f"Ch.15 probability of failure (15.5) SR {tsr}", float(norm.cdf(thr, p_, p_ * (1 - p_))), tg["prob_failure_15_5"])

    # =============== 8b. Equations 77-78 ===============
    conc = pd.read_csv(run / "portfolio_concentration_eq77_78.csv")
    mine_h = []
    for st_ in sorted(sched):
        if not sched[st_]:
            continue
        cols = list(sched[st_].keys())
        win = Rr[R_days < st_].tail(Wreb)[cols].fillna(0.0)
        lam, V = np.linalg.eigh(win.cov().to_numpy())
        f = V.T @ np.array([sched[st_][c_] for c_ in cols])
        th = f ** 2 * lam
        th = th / th.sum()
        th = th[th > 0]
        mine_h.append((pd.Timestamp(st_ * DAY, unit="s").strftime("%Y-%m-%d"), len(cols), 1.0 - math.exp(-float((th * np.log(th)).sum())) / len(cols)))
    if [x[0] for x in mine_h] != conc["rebalance_day"].tolist():
        problems.append("Eq.77-78: rebalance days differ")
    else:
        for (d_, n_, h_), (_, row) in zip(mine_h, conc.iterrows()):
            cmp(f"Eq.78 H at {d_}", h_, float(row["H_eq78"]), 1e-8)
            cmp(f"Eq.78 N at {d_}", n_, int(row["n_assets"]))
        cmp("Eq.78 mean H (meta)", float(np.mean([x[2] for x in mine_h])), meta["concentration_eq77_78"]["H_mean"], 1e-8)

    # =============== 9. Snippet 10.4 ===============
    od = pd.read_csv(run / "orders_10_4.csv")
    trg_ev = ev["trgt"].to_numpy(float)
    exp_orders = [(int(e), float(side_w[k]), float(size_w[k])) for k, e in enumerate(w_idx) if t0_ev[e] >= lo_t]
    cmp("10.4 number of orders", len(exp_orders), len(od))
    cmp("10.4 number of orders (meta)", len(exp_orders), meta["ch10_orders"]["n_orders"])
    mstar, Qm = 0.95, 100
    worst_tp = worst_lp = 0.0
    knife = 0
    for (e, sd, sz), (_, row) in zip(exp_orders, od.iterrows()):
        if int(row["event"]) != e or int(row["side"]) != int(sd):
            problems.append(f"10.4 order for event {e}: event / side differ from the saved ticket")
            break
        x = PTSL * trg_ev[e]
        w = x * x * (mstar ** -2 - 1.0)
        raw = x / math.sqrt(w + x * x) * Qm                      # Snippet 10.4: int(betSize(w, f - mP) * maxPos), f - mP = x for a long order
        tpos = int(raw)
        bot_tp = abs(int(row["target_position_10_4"]))
        if abs(raw - round(raw)) < 1e-9 and abs(tpos - bot_tp) == 1 and plant != "order":
            knife += 1                                               # the calibration puts betSize * Q at exactly an integer (M* x Q = 95): int() truncation then depends on rounding noise
            tpos = bot_tp
        lp = sum((1.0 + x) - (j / Qm) * math.sqrt(w / (1.0 - (j / Qm) ** 2)) for j in range(1, tpos + 1)) / tpos if tpos else float("nan")
        if sd < 0:
            tpos, lp = -tpos, 2.0 - lp
        if plant == "order":
            tpos += 1
        worst_tp = max(worst_tp, abs(tpos - int(row["target_position_10_4"])))
        if not np.isnan(lp):
            worst_lp = max(worst_lp, abs((lp - 1.0) - float(row["limit_price_offset_from_entry"])))
        if abs(sz - float(row["size_10_3"])) > 1e-9:
            problems.append(f"10.4 order for event {e}: size_10_3 differs")
            break
    print(f"10.4: {knife} tickets sit on the int() knife edge (calibrated size x Q is an integer to 1e-9); either neighbour accepted, the bot's is used", flush=True)
    cmp("10.4 target positions (max abs diff)", worst_tp, 0.0)
    cmp("10.4 limit prices (max abs diff)", worst_lp, 0.0, 1e-8)
    checked += 2 * len(od)

    msg = f"{venue}: {checked} numbers re-derived independently ({nchk} HRP weights, {nf} feature values, {N:,} labels among them); {len(problems)} problems" + (f"  [planted: {plant}]" if plant else "") + f"  ({time.time() - t_start:.0f} s)"
    print(msg)
    for p in problems[:30]:
        print("  PROBLEM:", p)
    if not plant:
        (run / "checker_output.txt").write_text(msg + "\n" + "\n".join(problems) + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    pl = sys.argv[sys.argv.index("--plant") + 1] if "--plant" in sys.argv else None
    raise SystemExit(main(Path(sys.argv[1]), pl))
