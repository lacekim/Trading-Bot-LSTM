#!/usr/bin/env python
"""Independent checker for a run_meta_market.py output folder (stdlib only; imports only helper functions of verify_book_study.py).  It is verify_meta_pipeline.py
with the market-level features added: besides everything below, it rebuilds the equal-weight index from the raw candles and from it, with its own SADF and entropy code, the three
features mkt_sadf, mkt_ent_plugin and mkt_ent_konto of every event (last completed UTC day), checks the 14 original feature columns against the source run, the trial count,
and the comparison section of SUMMARY.md against the source run and this run.


From the raw candle files (each symbol cut at its recorded last bar) it rebuilds, with its own code, EVERY event the run should contain:
the CUSUM events, the one-bar sigma, the book's daily-volatility target, the trailing-return side, the triple-barrier exit (literal loop, strict
touches), the meta-label, uniqueness and return-attribution weights, the fractional-differentiation d (own FFD and ADF) and feature, the volatility
and trailing-return features, and the market state and distance -- then compares the set of events and every column with events.csv.
From events.csv alone it re-derives: the CPCV group edges, purge/embargo and the train / test counts of every split, path membership of every
prediction (each event once per path), AUC / log-loss / accuracy per split, bet sizes, the daily series of the primary and meta variants,
Sharpe, skew, kurtosis, PSR, the cross-path summary, the deflated-Sharpe table, the window lines, and the walk-forward counts and results.
Placebo rows are checked for what can be checked without numpy's random permutation (same events per day, same size multiset, internal consistency
between daily.csv and paths.csv, and window lines from daily.csv).
Not covered by this checker: the forest's internals, the sequential bootstrap and the MDA / SFI values (unit tests cover those; here only that the
files list every feature with finite numbers).
Exit code 0 = all checks pass.

    python tools/book_study/verify_meta_market.py runtime/book_meta_market/<timestamp> [source meta run folder]
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import sys
import time
from array import array
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import verify_book_study as V

HERE = Path(__file__).resolve().parent
SPAN = 336                       # one-bar sigma span used for the CUSUM threshold (same constant as the earlier studies)
NAN = math.nan
EULER = 0.5772156649015329
DECLARED_FEATURES = ["side", "ffd", "mom24n", "mom168n", "mom720n", "mom2160n", "log_trgt", "vol_ratio", "mkt_state", "mkt_dist",
                     "sadf", "ent_plugin", "ent_konto", "cs_spread", "mkt_sadf", "mkt_ent_plugin", "mkt_ent_konto"]
MARKET = ["mkt_sadf", "mkt_ent_plugin", "mkt_ent_konto"]
PRIOR_TRIALS = 100
DECLARED_MICRO = {"sadf": [100, 20, 1], "entropy": [100, 2], "cs_sl": 24}
SADF_W, SADF_MINSL, ENT_W, CS_SL = 100, 20, 100, 24
CRIT5 = (-2.86154, -2.8903, -4.234, -40.040)


def fin(x) -> bool:
    return x is not None and not math.isnan(x) and not math.isinf(x)


def close(a, b, rel=1e-6, ab=1e-9) -> bool:
    if not fin(a) and not fin(b):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if not fin(a) or not fin(b):
        return False
    return abs(a - b) <= max(ab, rel * max(abs(a), abs(b)))


# ---------------------------------------------------------------------------------------------------------------------
# fractional differentiation + ADF, own code
# ---------------------------------------------------------------------------------------------------------------------
def ffd_weights(d, thres):
    w = [1.0]
    k = 1
    while k < 100000:
        wk = -w[-1] / k * (d - k + 1)
        if abs(wk) < thres:
            break
        w.append(wk)
        k += 1
    return w[::-1]


def ffd_series(x, d, thres):
    w = ffd_weights(d, thres)
    width = len(w) - 1
    out = [NAN] * len(x)
    for t in range(width, len(x)):
        s = 0.0
        base = t - width
        for k in range(width + 1):
            s += w[k] * x[base + k]
        out[t] = s
    return out, width


def solve3(a, b):
    """Gauss-Jordan on a 3x3 system a x = b and the inverse of a; returns (x, inverse)."""
    n = 3
    m = [list(a[i]) + [1.0 if i == j else 0.0 for j in range(n)] + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        pv = m[c][c]
        m[c] = [v / pv for v in m[c]]
        for r in range(n):
            if r != c:
                f = m[r][c]
                if f:
                    m[r] = [vr - f * vc for vr, vc in zip(m[r], m[c])]
    return [m[i][2 * n] for i in range(n)], [m[i][n:2 * n] for i in range(n)]


def adf_stat(y):
    """t-statistic of the lagged level in dy_t = a + g*y_{t-1} + c*dy_{t-1} + e (one lagged difference, constant)."""
    dy = [y[i + 1] - y[i] for i in range(len(y) - 1)]
    lags = 1
    n = len(dy) - lags
    if n <= lags + 3:
        return NAN, 0
    rows = [(1.0, y[lags + i], dy[lags - 1 + i]) for i in range(n)]
    tgt = dy[lags:]
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for r, t in zip(rows, tgt):
        for i in range(3):
            xty[i] += r[i] * t
            for j in range(3):
                xtx[i][j] += r[i] * r[j]
    beta, inv = solve3(xtx, xty)
    ssr = sum((t - (beta[0] * r[0] + beta[1] * r[1] + beta[2] * r[2])) ** 2 for r, t in zip(rows, tgt))
    s2 = ssr / (n - 3)
    return beta[1] / math.sqrt(s2 * inv[1][1]), n


def crit5(nobs):
    t = float(nobs)
    return CRIT5[0] + CRIT5[1] / t + CRIT5[2] / t ** 2 + CRIT5[3] / t ** 3


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    return sab / math.sqrt(saa * sbb)


def min_d(x, thres):
    """Smallest d on 0, .05, ..., 1 whose FFD passes the ADF test at 5%; returns (d, stat, crit, corr, ffd series at that d)."""
    for i in range(21):
        d = round(0.05 * i, 2)
        f, width = ffd_series(x, d, thres)
        ok = [v for v in f if not math.isnan(v)]
        if len(ok) < 50:
            continue
        stat, nobs = adf_stat(ok)
        cr = crit5(nobs)
        if fin(stat) and stat < cr:
            corr = pearson(x[width:], ok)
            return d, stat, cr, corr, f
    f, _ = ffd_series(x, 1.0, thres)
    return 1.0, NAN, NAN, NAN, f


# ---------------------------------------------------------------------------------------------------------------------
# bet size, rank AUC, daily statistics
# ---------------------------------------------------------------------------------------------------------------------
def bet_size(p, step):
    p = min(max(p, 1e-9), 1 - 1e-9)
    if not p > 0.5:
        return 0.0
    z = (p - 0.5) / math.sqrt(p * (1 - p))
    s = 2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))) - 1.0
    if step > 0:
        s = round(s / step) * step
    return min(max(s, 0.0), 1.0)


def auc(y, p):
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return NAN
    return (sum(r for r, yy in zip(ranks, y) if yy) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def day_stats(x, events, t_eff_div):
    n, sr, sk, ku = V.moments(x)
    t_eff = n / t_eff_div
    return {"days": n, "sr_daily": sr, "sr_annual": sr * math.sqrt(365) if fin(sr) else NAN, "skew": sk, "kurt": ku, "t_eff": t_eff,
            "psr": V.psr(sr, t_eff, sk, ku) if fin(sr) else NAN}


def mean(x):
    return sum(x) / len(x) if x else NAN


def expected_max_sharpe(n, var):
    nd = NormalDist()
    if n < 2 or not fin(var) or var < 0:
        return NAN
    return math.sqrt(var) * ((1 - EULER) * nd.inv_cdf(1 - 1 / n) + EULER * nd.inv_cdf(1 - 1 / (n * math.e)))


# ---------------------------------------------------------------------------------------------------------------------
# Chapters 17-19, own code
# ---------------------------------------------------------------------------------------------------------------------
def load_raw(gmx_dir, tv_dir, symbol, timeframe="1h"):
    """(sorted epoch list, closes, highs, lows): GMX candles plus only TradingView candles older than the first GMX candle; closes > 0 only."""
    gmx, tv = {}, {}
    gpath = Path(gmx_dir) / f"gmx_arbitrum_{symbol}_{timeframe}.csv"
    if gpath.exists():
        for r in V.read(gpath):
            c = V.num(r.get("close", r.get("Close")))
            t = r.get("open_time", r.get("Date"))
            if t is not None and not math.isnan(c):
                gmx[V.ts_key(t)] = (c, V.num(r.get("high", r.get("High"))), V.num(r.get("low", r.get("Low"))))
    tpath = Path(tv_dir) / f"tradingview_{symbol}_{timeframe}.csv"
    if tpath.exists():
        for r in V.read(tpath):
            c = V.num(r.get("Close", r.get("close")))
            t = r.get("Date", r.get("time"))
            if t is not None and not math.isnan(c):
                tv[V.ts_key(t)] = (c, V.num(r.get("High", r.get("high"))), V.num(r.get("Low", r.get("low"))))
    first_gmx = min(gmx) if gmx else math.inf
    merged = {t: v for t, v in tv.items() if t < first_gmx}
    merged.update(gmx)
    keys = sorted(k for k in merged if merged[k][0] > 0)
    return keys, [merged[k][0] for k in keys], [merged[k][1] for k in keys], [merged[k][2] for k in keys]


def sadf_py(y, minsl=SADF_MINSL):
    """SADF of one window of log prices (constant, one lagged difference), by suffix sums of the normal equations; the window is shifted by its
    last value. Starts whose 3x3 matrix is singular (|det| < 1e-14) or whose variance is not positive are skipped."""
    W = len(y)
    c = y[-1]
    z = [v - c for v in y]
    n_rows = W - 2
    n_starts = n_rows + 1 - minsl + 1
    a00 = a01 = a02 = a11 = a12 = a22 = 0.0
    b0 = b1 = b2 = yy = 0.0
    best = None
    for s in range(n_rows - 1, -1, -1):
        t = s + 2                                  # row s is the observation y[t]
        lv, dl, tg = z[t - 1], z[t - 1] - z[t - 2], z[t] - z[t - 1]
        a00 += lv * lv; a01 += lv * dl; a02 += lv; a11 += dl * dl; a12 += dl; a22 += 1.0
        b0 += lv * tg; b1 += dl * tg; b2 += tg; yy += tg * tg
        if s <= n_starts - 1:
            c00, c01, c02 = a11 * a22 - a12 * a12, a02 * a12 - a01 * a22, a01 * a12 - a02 * a11
            det = a00 * c00 + a01 * c01 + a02 * c02
            if abs(det) < 1e-14:
                continue
            c11, c12, c22 = a00 * a22 - a02 * a02, a01 * a02 - a00 * a12, a00 * a11 - a01 * a01
            i00, i01, i02, i11, i12, i22 = c00 / det, c01 / det, c02 / det, c11 / det, c12 / det, c22 / det
            be0 = i00 * b0 + i01 * b1 + i02 * b2
            be1 = i01 * b0 + i11 * b1 + i12 * b2
            be2 = i02 * b0 + i12 * b1 + i22 * b2
            ssr = max(yy - (be0 * b0 + be1 * b1 + be2 * b2), 0.0)
            n_obs = n_rows - s
            var = ssr / (n_obs - 3) * i00
            if var <= 0 or not math.isfinite(var):
                continue
            st = be0 / math.sqrt(var)
            if math.isfinite(st) and (best is None or st > best):
                best = st
    return NAN if best is None else best


def plugin_py(bits, w=2):
    """Plug-in entropy rate of a 0/1 message: words of length w that end before the last character (Snippet 18.1)."""
    cnt = defaultdict(int)
    for i in range(w, len(bits)):
        cnt[bits[i - w:i]] += 1
    tot = float(len(bits) - w)
    return -sum((v / tot) * math.log2(v / tot) for v in cnt.values()) / w


def konto_py(msg):
    """Kontoyiannis estimate, expanding window: for i = 1 .. len/2 the longest prefix of msg[i:] (at most i characters) that starts somewhere in
    msg[0:i] (overlap allowed), found by comparing characters; h = mean of log2(i+1)/(match + 1)."""
    n2 = len(msg) // 2
    tot = 0.0
    for i in range(1, n2 + 1):
        best = 0
        for j in range(i):
            k = 0
            while k < i and i + k < len(msg) and msg[j + k] == msg[i + k]:
                k += 1
            if k > best:
                best = k
                if best == i:
                    break
        tot += math.log2(i + 1) / (best + 1)
    return tot / n2


def cs_py(hi, lo, sl):
    """Corwin-Schultz spread per bar (NaN where unavailable) from High / Low lists, Snippet 19.1."""
    n = len(hi)
    out = [NAN] * n
    ok = [hi[i] > 0 and lo[i] > 0 and hi[i] >= lo[i] for i in range(n)]
    hl = [math.log(hi[i] / lo[i]) ** 2 if ok[i] else NAN for i in range(n)]
    den = 3 - 2 * 2 ** 0.5
    b2 = [NAN] + [hl[i] + hl[i - 1] for i in range(1, n)]
    for t in range(sl, n):
        win = b2[t - sl + 1:t + 1]
        if any(math.isnan(v) for v in win) or not (ok[t] and ok[t - 1]):
            continue
        beta = sum(win) / sl
        g = math.log(max(hi[t], hi[t - 1]) / min(lo[t], lo[t - 1])) ** 2
        alpha = (2 ** 0.5 - 1) * math.sqrt(beta) / den - math.sqrt(g / den)
        if alpha < 0:
            alpha = 0.0
        out[t] = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
    return out


def main(folder: str, source_override: str | None = None) -> int:
    t_start = time.time()
    d = Path(folder)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    def say(msg):
        print(f"[{time.time() - t_start:5.0f}s] {msg}", flush=True)

    def report(stopped=""):
        for name, ok, detail in checks:
            print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
        n_ok = sum(1 for c in checks if c[1])
        if stopped:
            print(f"\nSTOPPED EARLY: {stopped}")
        print(f"\n{n_ok}/{len(checks)} checks passed  ({time.time() - t_start:.0f}s)")
        return 0 if (n_ok == len(checks) and not stopped) else 1

    summ = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    meta, params = summ["meta"], summ["params"]
    integrity = json.loads((d / "integrity.json").read_text(encoding="utf-8"))
    ev = V.read(d / "events.csv")
    symbols = V.read(d / "symbols.csv")
    fd_rows = V.read(d / "fracdiff.csv")
    split_rows = V.read(d / "split_metrics.csv")
    pred_rows = V.read(d / "cpcv_predictions.csv")
    wf_rows = V.read(d / "walk_forward_predictions.csv")
    daily_rows = V.read(d / "daily.csv")
    path_rows = V.read(d / "paths.csv")
    h, ptsl, cost = int(params["h"]), params["ptsl"], float(params["cost_frac"])
    step, N, K = float(params["step"]), int(params["n_groups"]), int(params["k"])
    embargo_pct = float(params["embargo_pct"])
    feats = params["features"]
    n_ev = len(ev)
    say(f"loaded {n_ev:,} events, {len(pred_rows):,} cpcv predictions")

    # ---- declared settings and run bookkeeping ----
    check("declared settings: vertical barrier 101 bars, +/-2 daily-vol barriers, FFD threshold 0.01, side = 30-day momentum, 24 bp round trip, the 17-feature list (14 per-event + 3 market), SADF window 100 / min 20 / 1 lag, entropy 100 days / word 2, Corwin-Schultz 24 bars",
          h == 100 and [float(x) for x in ptsl] == [2.0, 2.0] and float(params["ffd_thres"]) == 0.01 and params["side_signal"] == "mom720"
          and close(cost, 0.0024, 1e-12, 1e-12) and feats == DECLARED_FEATURES and params.get("micro") == DECLARED_MICRO and params.get("market_features") == MARKET,
          str({k: params.get(k) for k in ("h", "ptsl", "ffd_thres", "side_signal", "cost_frac", "features", "micro")}))
    check("models and calibration files untouched during the run", integrity.get("unchanged") is True)
    changed = []
    for name, md5 in meta["script_md5"].items():
        p = HERE / name
        if p.exists():
            got = hashlib.md5(p.read_bytes()).hexdigest()
            if got != md5:
                changed.append(name)
    check("the study scripts next to this checker are the ones that produced the run (md5 recorded in the run)", not changed, f"changed since the run: {changed}")
    check("window definitions: June 2026 = [2026-06-01, 2026-07-01) UTC and a sell-off window is present; each window starts before it ends",
          meta["windows"].get("june_2026_ai_correction", [None, None])[0].startswith("2026-06-01") and meta["windows"]["june_2026_ai_correction"][1].startswith("2026-07-01")
          and any(k_.startswith("sell_off") for k_ in meta["windows"]) and all(V.ts_key(v[0]) < V.ts_key(v[1]) for v in meta["windows"].values())
          and all(int(meta["window_epoch"][k_][0]) == int(V.ts_key(meta["windows"][k_][0])) and int(meta["window_epoch"][k_][1]) == int(V.ts_key(meta["windows"][k_][1])) for k_ in meta["windows"]),
          str(meta["windows"]))
    win_epoch = {k_: (int(v[0]), int(v[1])) for k_, v in meta["window_epoch"].items()}
    check("every window starts and ends on a UTC day boundary", all(a % 86400 == 0 and b % 86400 == 0 for a, b in win_epoch.values()), str(win_epoch))

    # ---- load raw candles, all symbols ----
    rows_by_sym = {r["symbol"]: r for r in symbols}
    sym_list = sorted(rows_by_sym)
    check("symbols.csv lists the symbols once each and the run used that many", len(sym_list) == len(symbols) == int(meta["symbols_used"]), f"{len(sym_list)} {len(symbols)} {meta['symbols_used']}")
    series = {}
    hl_raw = {}
    for s in sym_list:
        keys, closes, highs, lows = load_raw(meta["gmx_dir"], meta["tradingview_dir"], s)
        n_keep = bisect.bisect_right(keys, V.ts_key(rows_by_sym[s]["last"]))
        keys, closes, highs, lows = keys[:n_keep], closes[:n_keep], highs[:n_keep], lows[:n_keep]
        logc = array("d", [math.log(c) for c in closes])
        series[s] = (array("d", keys), logc)
        hl_raw[s] = (array("d", highs), array("d", lows))
    say(f"read raw candles for {len(series)} symbols")

    # market index (equal weight, hours with >= 10 symbols) and its 200-day mean
    hour_sum, hour_cnt = defaultdict(float), defaultdict(int)
    for s in sym_list:
        keys, logc = series[s]
        for i in range(1, len(keys)):
            hour_sum[keys[i]] += logc[i] - logc[i - 1]
            hour_cnt[keys[i]] += 1
    hours = sorted(t for t in hour_cnt if hour_cnt[t] >= 10)
    cum, level, csum = 0.0, [], [0.0]
    for t in hours:
        cum += hour_sum[t] / hour_cnt[t]
        v = math.exp(cum)
        level.append(v)
        csum.append(csum[-1] + v)

    # market features (independent): daily log of the index (last hourly level of each UTC day that has an index hour), own SADF / entropy code
    xm, dnum_m = [], []
    for t_, lv_ in zip(hours, level):
        dd_ = int(t_ // 86400)
        if dnum_m and dnum_m[-1] == dd_:
            xm[-1] = math.log(lv_)
        else:
            xm.append(math.log(lv_))
            dnum_m.append(dd_)
    dpos_m = {dd_: k_ for k_, dd_ in enumerate(dnum_m)}
    bits_m = "".join("1" if xm[k_] - xm[k_ - 1] > 0 else "0" for k_ in range(1, len(xm)))
    mkt_cache = {}

    def mkt_day(dd_):
        if dd_ in mkt_cache:
            return mkt_cache[dd_]
        k_ = dpos_m.get(dd_)
        if k_ is None:
            v_ = (NAN, NAN, NAN)
        else:
            sd_ = sadf_py(xm[k_ - SADF_W + 1:k_ + 1]) if k_ >= SADF_W - 1 else NAN
            if k_ >= ENT_W:
                msg_ = bits_m[k_ - ENT_W:k_]
                v_ = (sd_, plugin_py(msg_), konto_py(msg_[::-1]))
            else:
                v_ = (sd_, NAN, NAN)
        mkt_cache[dd_] = v_
        return v_

    def state_dist(t):
        i = bisect.bisect_left(hours, t)
        if i >= len(hours) or hours[i] != t:
            return -1, NAN
        lo = bisect.bisect_right(hours, t - 200 * 86400)
        n = i + 1 - lo
        if n < 3840:
            return -1, NAN
        ma = (csum[i + 1] - csum[lo]) / n
        return (1 if level[i] > ma else 0), math.log(level[i] / ma)

    # ---- rebuild every event ----
    exp_rows = []                        # dicts
    n_cusum = 0
    n_drop_side = n_drop_feat = 0
    ties = 0
    barrier_bad = []
    fd_exp = {}
    sym_info_bad = []
    uniq_cache = {}
    for s in sym_list:
        keys, logc = series[s]
        n = len(keys)
        rr = [NAN] + [logc[i] - logc[i - 1] for i in range(1, n)]
        sg = V.ewm_sigma(rr, SPAN)
        evp = V.cusum(rr, sg)
        kept = [p for p in evp if p + h + 1 < n and fin(sg[p]) and sg[p] > 0 and (keys[p + h + 1] - keys[p]) == (h + 1) * 3600]
        info = rows_by_sym[s]
        if not (n == int(info["bars"]) and len(evp) == int(info["events_total"]) and len(kept) == int(info["events_kept"])):
            sym_info_bad.append((s, n, info["bars"], len(evp), info["events_total"], len(kept), info["events_kept"]))
        n_cusum += len(kept)
        r24 = [NAN] * 24 + [math.exp(logc[i] - logc[i - 24]) - 1.0 for i in range(24, n)]
        trgt_all = V.ewm_sigma(r24, 100)
        slow = V.ewm_sigma(rr, 2016)
        # daily log close (last bar of each UTC day), min d and the feature series
        x, dnum = [], []
        for i in range(n):
            dd = int(keys[i] // 86400)
            if dnum and dnum[-1] == dd:
                x[-1] = logc[i]
            else:
                x.append(logc[i])
                dnum.append(dd)
        dsel, stat, cr, corr, f = min_d(x, 0.01)
        fd_exp[s] = (dsel, stat, cr, corr)
        fmap = {dd: v for dd, v in zip(dnum, f) if not math.isnan(v)}
        dpos = {dd: k for k, dd in enumerate(dnum)}
        csv_ = cs_py(list(hl_raw[s][0]), list(hl_raw[s][1]), CS_SL)
        bits_all = "".join("1" if x[k] - x[k - 1] > 0 else "0" for k in range(1, len(x)))      # bit of day k is bits_all[k-1]
        micro_cache = {}

        def micro_day(dd):
            """(sadf, plug-in, konto) at the daily index of UTC day number dd, computed on demand and only if its window exists."""
            if dd in micro_cache:
                return micro_cache[dd]
            k = dpos.get(dd)
            if k is None:
                v = (NAN, NAN, NAN)
            else:
                sd = sadf_py(x[k - SADF_W + 1:k + 1]) if k >= SADF_W - 1 else NAN
                if k >= ENT_W:
                    msg = bits_all[k - ENT_W:k]                       # returns of days k-ENT_W+1 .. k, oldest first
                    v = (sd, plugin_py(msg), konto_py(msg[::-1]))
                else:
                    v = (sd, NAN, NAN)
            micro_cache[dd] = v
            return v

        sub = []
        for p in kept:
            side = 0
            if p >= 720:
                dl = logc[p] - logc[p - 720]
                side = 1 if dl > 0 else -1 if dl < 0 else 0
            trgt = trgt_all[p]
            ok0 = side != 0 and fin(trgt) and trgt > 0
            if not ok0:
                n_drop_side += 1
                continue
            row = {"symbol": s, "pos": p, "t0": int(keys[p]), "sigma": sg[p], "trgt": trgt, "side": side, "log_trgt": math.log(trgt)}
            for L, nm in ((24, "mom24n"), (168, "mom168n"), (720, "mom720n"), (2160, "mom2160n")):
                row[nm] = (logc[p] - logc[p - L]) / (sg[p] * math.sqrt(L)) if p >= L else NAN
            row["vol_ratio"] = math.log(sg[p] / slow[p]) if fin(slow[p]) and slow[p] > 0 else NAN
            row["ffd"] = fmap.get(int(keys[p] // 86400) - 1, NAN)
            row["sadf"], row["ent_plugin"], row["ent_konto"] = micro_day(int(keys[p] // 86400) - 1)
            row["cs_spread"] = csv_[p]
            row["mkt_sadf"], row["mkt_ent_plugin"], row["mkt_ent_konto"] = mkt_day(int(keys[p] // 86400) - 1)
            st, dist = state_dist(float(keys[p]))
            row["mkt_state"] = float(st) if st >= 0 else NAN
            row["mkt_dist"] = dist
            # literal triple barrier, strict touches, simple returns times side
            pt, sl = ptsl[0] * trgt, -ptsl[1] * trgt
            Rs, jx, kd, rt = [], None, 0, None
            for t in range(1, h + 2):
                R = (math.exp(logc[p + t] - logc[p]) - 1.0) * side
                Rs.append(R)
                if jx is None:
                    if R > pt:
                        jx, kd, rt = t, 1, R
                    elif R < sl:
                        jx, kd, rt = t, -1, R
            if jx is None:
                jx, kd, rt = h + 1, 0, Rs[-1]
            row["_Rs"], row["_pt"], row["_sl"] = Rs, pt, sl
            row["exit_j"], row["kind"], row["ret_side"], row["t1pos"] = jx - 1, kd, rt, p + jx
            row["t1"] = int(keys[p]) + jx * 3600
            row["bin"] = 1 if rt > 0 else 0
            vals = [row[k_] for k_ in feats]
            if not (all(fin(v) for v in vals) and fin(rt)):
                n_drop_feat += 1
                continue
            sub.append(row)
        # uniqueness and return-attribution weights over the kept events of this symbol
        if sub:
            cdiff = [0.0] * (n + 2)
            for r in sub:
                cdiff[r["pos"]] += 1.0
                cdiff[r["t1pos"] + 1] -= 1.0
            conc, c = [0.0] * n, 0.0
            for i in range(n):
                c += cdiff[i]
                conc[i] = c
            inv = [1.0 / v if v > 0 else 0.0 for v in conc]
            contrib = [0.0] + [rr[i] / conc[i] if conc[i] > 0 else 0.0 for i in range(1, n)]
            pref_inv, pref_c = [0.0], [0.0]
            for i in range(n):
                pref_inv.append(pref_inv[-1] + inv[i])
                pref_c.append(pref_c[-1] + contrib[i])
            for r in sub:
                a, b = r["pos"], r["t1pos"]
                r["u"] = (pref_inv[b + 1] - pref_inv[a]) / (b - a + 1)
                r["w_raw"] = abs(pref_c[b + 1] - pref_c[a])
        exp_rows.extend(sub)
    say(f"rebuilt {len(exp_rows):,} events from raw candles")
    exp_rows.sort(key=lambda r: (r["t0"], r["symbol"]))
    tot_w = sum(r["w_raw"] for r in exp_rows)
    for r in exp_rows:
        r["w"] = r["w_raw"] * len(exp_rows) / tot_w
        r["net_ret"] = r["ret_side"] - cost

    info_ev = summ["events_info"]
    check("bar counts, CUSUM totals and kept (complete, gap-free window) events replayed from raw candles for every symbol", not sym_info_bad, str(sym_info_bad[:3]))
    check("event accounting: CUSUM events, dropped for no side / no volatility, dropped for a missing feature or state, kept",
          n_cusum == int(info_ev["events_cusum"]) and n_drop_side == int(info_ev["events_dropped_no_side_or_vol"]) and n_drop_feat == int(info_ev["events_dropped_missing_feature"])
          and len(exp_rows) == int(info_ev["events_kept"]) == n_ev == int(meta["events"]),
          f"cusum {n_cusum}/{info_ev['events_cusum']}, no side {n_drop_side}/{info_ev['events_dropped_no_side_or_vol']}, missing {n_drop_feat}/{info_ev['events_dropped_missing_feature']}, kept {len(exp_rows)}/{n_ev}")
    key_exp = [(r["symbol"], r["pos"]) for r in exp_rows]
    key_got = [(e["symbol"], int(e["pos"])) for e in ev]
    check("the set of events and their order (time, then symbol) equal the rebuilt ones", key_exp == key_got, f"{len(set(key_exp) ^ set(key_got))} differ")
    if set(key_exp) != set(key_got) or len(key_exp) != len(key_got):
        return report("events.csv does not contain the rebuilt set of events, so the event-indexed checks (splits, predictions, daily results) are meaningless")
    for fname, rows_x in (("cpcv_predictions.csv", pred_rows), ("walk_forward_predictions.csv", wf_rows)):
        if any(not (0 <= int(r["event"]) < n_ev) for r in rows_x):
            check(f"{fname} refers only to rows of events.csv", False, "an event index is outside events.csv")
            return report(f"{fname} refers to events that do not exist")

    # column by column, aligned on (symbol, pos)
    got_by_key = {(e["symbol"], int(e["pos"])): e for e in ev}
    int_cols = ("t0", "t1", "t1pos", "side", "exit_j", "kind", "bin")
    tol = {"sigma": (1e-6, 1e-12), "trgt": (1e-6, 1e-12), "log_trgt": (1e-6, 1e-9), "ret_side": (1e-6, 1e-7), "net_ret": (1e-6, 1e-7), "u": (1e-8, 1e-10),
           "w_raw": (1e-7, 1e-12), "w": (1e-7, 1e-9), "ffd": (1e-6, 1e-8), "mom24n": (1e-6, 1e-8), "mom168n": (1e-6, 1e-8), "mom720n": (1e-6, 1e-8),
           "mom2160n": (1e-6, 1e-8), "vol_ratio": (1e-5, 1e-8), "mkt_state": (0, 0), "mkt_dist": (1e-6, 1e-9),
           "sadf": (1e-6, 1e-7), "ent_plugin": (1e-9, 1e-12), "ent_konto": (1e-9, 1e-12), "cs_spread": (1e-6, 1e-10),
           "mkt_sadf": (1e-6, 1e-7), "mkt_ent_plugin": (1e-9, 1e-12), "mkt_ent_konto": (1e-9, 1e-12)}
    bad_col = defaultdict(list)
    barrier_tied = 0
    for r in exp_rows:
        e = got_by_key.get((r["symbol"], r["pos"]))
        if e is None:
            continue
        # barrier columns: tolerate float32 rounding of the stored path only when a barrier is touched within 1e-6 of the disputed bar
        stored = (int(float(e["exit_j"])), int(float(e["kind"])), float(e["ret_side"]))
        mine = (r["exit_j"], r["kind"], r["ret_side"])
        if stored[0] != mine[0] or stored[1] != mine[1] or abs(stored[2] - mine[2]) > 1e-6:
            hi_t = max(stored[0], mine[0]) + 1
            near = any(abs(R - r["_pt"]) < 1e-6 or abs(R - r["_sl"]) < 1e-6 for R in r["_Rs"][:hi_t])
            if near:
                barrier_tied += 1
                continue
            bad_col["barrier(exit_j,kind,ret_side)"].append((r["symbol"], r["pos"], stored, mine))
            continue
        for c in int_cols:
            if int(float(e[c])) != r[c]:
                bad_col[c].append((r["symbol"], r["pos"], e[c], r[c]))
        for c, (rel, ab) in tol.items():
            if c == "ret_side":
                continue
            if not close(V.num(e[c]), r[c], rel, ab):
                bad_col[c].append((r["symbol"], r["pos"], e[c], r[c]))
    check("every column of every event (sigma, target, side, barrier exit / kind / return, meta-label, t1, uniqueness, attribution weight, scaled weight, "
          "features including the three market features, state, distance, net return) equals the rebuilt value", not bad_col, str({k_: (len(v), v[:2]) for k_, v in bad_col.items()}))
    say(f"events compared; {barrier_tied} events with a barrier touch within 1e-6 tolerated as float32 ties")
    check("barrier ties tolerated (float32 rounding of the stored path) are a negligible share", barrier_tied <= max(5, n_ev // 2000), f"{barrier_tied} of {n_ev}")

    ret = [float(e["ret_side"]) for e in ev]
    binv = [int(float(e["bin"])) for e in ev]
    t0 = [int(float(e["t0"])) for e in ev]
    t1 = [int(float(e["t1"])) for e in ev]
    wt = [float(e["w"]) for e in ev]
    uu = [float(e["u"]) for e in ev]
    syms_ev = [e["symbol"] for e in ev]
    day_ev = [t // 86400 for t in t0]
    check("sorted by time then symbol; weights average 1; uniqueness in (0,1]; meta-label = 1 exactly when the side's return is positive; net = gross - cost",
          all((a, s1) <= (b, s2) for a, s1, b, s2 in zip(t0, syms_ev, t0[1:], syms_ev[1:])) and abs(mean(wt) - 1.0) < 1e-6 and all(0 < u <= 1 + 1e-12 for u in uu)
          and all(b == (1 if r > 0 else 0) for b, r in zip(binv, ret)) and all(close(float(e["net_ret"]), float(e["ret_side"]) - cost, 1e-9, 1e-12) for e in ev))

    # ---- fractional differentiation file ----
    fd_bad = []
    fd_by = {r["symbol"]: r for r in fd_rows}
    for s in sym_list:
        r = fd_by.get(s)
        if r is None:
            fd_bad.append((s, "missing"))
            continue
        dsel, stat, cr, corr = fd_exp[s]
        got = (float(r["d"]), float(r["adf"]), float(r["crit"]), float(r["corr"]))
        if not (close(got[0], dsel, 1e-12, 1e-12) and close(got[1], stat, 1e-4, 1e-6) and close(got[2], cr, 1e-6, 1e-9) and close(got[3], corr, 1e-5, 1e-8)):
            fd_bad.append((s, got, (dsel, stat, cr, corr)))
    check("fracdiff.csv: smallest d, ADF statistic, critical value and correlation recomputed (own FFD and ADF) for every symbol", not fd_bad and len(fd_rows) == len(sym_list), str(fd_bad[:3]))
    say("events, fracdiff checked")

    # ---- CPCV bookkeeping ----
    t_lo, t_hi = float(min(t0)), float(max(t0)) + 1.0
    edges = [t_lo + i * (t_hi - t_lo) / N for i in range(N + 1)]
    edges[-1] = t_hi
    grp = [min(max(bisect.bisect_right(edges, float(t)) - 1, 0), N - 1) for t in t0]
    embargo = embargo_pct * (t_hi - t_lo)
    combos = list(combinations(range(N), K))
    check("CPCV split list: every choice of K of N groups once, in order", [tuple(int(x) for x in r["test_groups"].split("-")) for r in split_rows] == combos and
          meta["n_splits"] == len(combos) and meta["n_paths"] == math.comb(N - 1, K - 1), f"{len(split_rows)} vs {len(combos)}")
    assign, counter = {}, [0] * N
    for si, tg in enumerate(combos):
        for g in tg:
            assign[(si, g)] = counter[g]
            counter[g] += 1
    n_paths = math.comb(N - 1, K - 1)
    check("CPCV edges recorded in the run equal the independent equal-duration edges and the embargo", all(close(a, b, 1e-12, 1e-6) for a, b in zip(meta["edges"], edges)) and close(meta["embargo_seconds"], embargo, 1e-12, 1e-6))
    bad_split = []
    test_sets, train_u_mean = [], []
    for si, tg in enumerate(combos):
        test = [i for i in range(n_ev) if grp[i] in tg]
        train = []
        for i in range(n_ev):
            if grp[i] in tg:
                continue
            ok = True
            for g in tg:
                lo, hi = edges[g], edges[g + 1]
                if (t0[i] <= hi and t1[i] >= lo) or (t0[i] > hi and t0[i] <= hi + embargo):
                    ok = False
                    break
            if ok:
                train.append(i)
        test_sets.append(test)
        train_u_mean.append(mean([uu[i] for i in train]))
        r = split_rows[si]
        if int(r["n_train"]) != len(train) or int(r["n_test"]) != len(test):
            bad_split.append((si, r["n_train"], len(train), r["n_test"], len(test)))
        if not close(float(r["avg_uniqueness"]), train_u_mean[-1], 1e-9, 1e-12):
            bad_split.append((si, "avg_uniqueness", r["avg_uniqueness"], train_u_mean[-1]))
        m_draws = min(max(1, round(train_u_mean[-1] * len(train))), int(params["forest"]["max_draws"]))
        if int(r["draws_per_tree"]) != m_draws:
            bad_split.append((si, "draws_per_tree", r["draws_per_tree"], m_draws))
    check("train and test counts of every split (purge of overlapping labels, embargo after each test group), average training uniqueness, draws per tree", not bad_split, str(bad_split[:3]))

    # predictions: each event once per path, in the right split / path
    by_split = defaultdict(list)
    seen_path = defaultdict(set)
    prob_bad = 0
    for r in pred_rows:
        si, ev_i, pa, p = int(r["split"]), int(r["event"]), int(r["path"]), float(r["p"])
        by_split[si].append((ev_i, pa, p))
        seen_path[ev_i].add(pa)
        if not (0.0 <= p <= 1.0):
            prob_bad += 1
    pred_bad = []
    for si, tg in enumerate(combos):
        rows_s = by_split.get(si, [])
        if sorted(e_ for e_, _, _ in rows_s) != test_sets[si]:
            pred_bad.append((si, "event set"))
            continue
        for e_, pa, _ in rows_s:
            if pa != assign[(si, grp[e_])]:
                pred_bad.append((si, e_, "path"))
                break
    check("cpcv_predictions.csv: every split predicts exactly its test events, each (split, group) sits on the right path, probabilities in [0,1]", not pred_bad and not prob_bad and len(pred_rows) == sum(len(t) for t in test_sets), str(pred_bad[:3]))
    check("each event is predicted exactly once on every one of the backtest paths", len(seen_path) == n_ev and all(v == set(range(n_paths)) for v in seen_path.values()) and len(pred_rows) == n_ev * n_paths)

    # per-split quality metrics
    mbad = []
    for si, tg in enumerate(combos):
        rows_s = by_split[si]
        y = [binv[e_] for e_, _, _ in rows_s]
        p = [q for _, _, q in rows_s]
        w = [wt[e_] for e_, _, _ in rows_s]
        pc = [min(max(q, 1e-6), 1 - 1e-6) for q in p]
        ll = -sum(wi * (yi * math.log(q) + (1 - yi) * math.log(1 - q)) for wi, yi, q in zip(w, y, pc)) / sum(w)
        want = {"auc": auc(y, p), "log_loss_w": ll, "base_rate": mean(y), "share_p_gt_half": mean([1.0 if q > 0.5 else 0.0 for q in p]),
                "acc": mean([1.0 if (q > 0.5) == bool(yy) else 0.0 for q, yy in zip(p, y)])}
        for k_, v in want.items():
            if not close(float(split_rows[si][k_]), v, 1e-6, 1e-9):
                mbad.append((si, k_, split_rows[si][k_], v))
    check("per-split AUC (own rank method), weighted log-loss, base rate, share with p > 0.5, accuracy recomputed from the predictions", not mbad, str(mbad[:3]))
    say("cpcv bookkeeping checked")

    # ---- evaluation ----
    t_eff_div = max(1.0, (h + 1) / 24.0)

    state_ev = [float(e["mkt_state"]) for e in ev]
    side_ev = [int(float(e["side"])) for e in ev]

    def evaluate(pred_by_path):
        """pred_by_path {path: [(event, p), ...]}  ->  per-path stats for primary / meta, daily dicts, window sums, summary and DSR."""
        out = {"paths": {}, "daily": {}}
        st_acc = defaultdict(lambda: [0.0, 0, 0.0])
        wst = defaultdict(int)
        wins = {(v, wn): [0.0, 0] for v in ("primary", "meta") for wn in win_epoch}
        for pa, lst in sorted(pred_by_path.items()):
            sizes = {e_: bet_size(p, step) for e_, p in lst}
            for vname in ("primary", "meta"):
                dd = defaultdict(lambda: [0.0, 0])
                tot = 0.0
                taken = 0
                szsum = 0.0
                for e_, _ in lst:
                    s = 1.0 if vname == "primary" else sizes[e_]
                    pnl = s * (ret[e_] - cost * (1.0 if s > 0 else 0.0)) * 1e4
                    cell = dd[day_ev[e_]]
                    cell[0] += pnl
                    cell[1] += 1
                    tot += pnl
                    sv = int(state_ev[e_])
                    for key in ((vname, sv), (vname, sv, side_ev[e_])):
                        a = st_acc[key]
                        a[0] += pnl
                        a[1] += 1
                        a[2] += s
                    if vname == "primary":
                        for wn, (lo, hi) in win_epoch.items():
                            if lo <= t0[e_] < hi:
                                wst[(wn, sv)] += 1
                    taken += 1 if s > 0 else 0
                    szsum += s
                    for wn, (lo, hi) in win_epoch.items():
                        if lo <= t0[e_] < hi:
                            wins[(vname, wn)][0] += pnl
                            wins[(vname, wn)][1] += 1
                days = sorted(dd)
                xs = [dd[k_][0] / dd[k_][1] for k_ in days]
                st = day_stats(xs, len(lst), t_eff_div)
                st.update({"events": len(lst), "mean_bp_per_candidate": tot / len(lst), "taken_share": taken / len(lst), "mean_size": szsum / len(lst)})
                out["paths"][(pa, vname)] = st
                out["daily"][(pa, vname)] = {k_: (dd[k_][0], dd[k_][1]) for k_ in days}
        out["win_sums"] = wins
        npth = len(pred_by_path)
        out["by_state"] = {"|".join([k_[0], f"state{k_[1]}"] + ([f"side{k_[2]:+d}"] if len(k_) == 3 else [])):
                           {"events_per_path": a[1] / npth, "mean_bp_per_candidate": a[0] / a[1] if a[1] else NAN, "mean_size": a[2] / a[1] if a[1] else NAN} for k_, a in st_acc.items()}
        out["window_state_events"] = {f"{wn}|state{sv}": n / npth for (wn, sv), n in wst.items()}
        return out

    pred_by_path = defaultdict(list)
    for r in pred_rows:
        pred_by_path[int(r["path"])].append((int(r["event"]), float(r["p"])))
    E = evaluate(pred_by_path)
    say("evaluated the CPCV paths")

    # daily.csv vs recomputed (primary, meta); placebo structure
    dgot = defaultdict(dict)
    for r in daily_rows:
        dgot[(int(r["path"]), r["variant"])][int(r["day"])] = (float(r["sum_bp"]), int(r["n"]))
    dbad = []
    for key, dd in E["daily"].items():
        g = dgot.get(key, {})
        if set(g) != set(dd) or any(g[k_][1] != dd[k_][1] or not close(g[k_][0], dd[k_][0], 1e-7, 1e-6) for k_ in dd if k_ in g):
            dbad.append(key)
    for pa in range(n_paths):
        g, pr = dgot.get((pa, "placebo"), {}), E["daily"][(pa, "primary")]
        if set(g) != set(pr) or any(g[k_][1] != pr[k_][1] for k_ in pr if k_ in g):
            dbad.append((pa, "placebo"))
    check("daily.csv: the primary and meta daily series (sum of bp, events) of every path recomputed; the placebo has the same days and event counts as the primary",
          not dbad and len(daily_rows) == 3 * sum(len(v) for (pa, v_), v in E["daily"].items() if v_ == "primary"), str(dbad[:3]))

    # paths.csv
    prow = {(int(r["path"]), r["variant"]): r for r in path_rows}
    pbad = []
    cmp_fields = ("events", "days", "mean_bp_per_candidate", "sr_daily", "sr_annual", "skew", "kurt", "t_eff", "psr", "taken_share", "mean_size")
    placebo_stats = {}
    for pa in range(n_paths):
        for vname in ("primary", "meta"):
            st, r = E["paths"][(pa, vname)], prow[(pa, vname)]
            for f_ in cmp_fields:
                fkey = "days" if f_ == "days" else f_
                want = st["days"] if f_ == "days" else st[f_]
                if not close(float(r[f_]), float(want), 1e-6, 1e-8):
                    pbad.append((pa, vname, f_, r[f_], want))
        # placebo: derived from its own daily rows; sizes multiset equals meta's
        g = dgot[(pa, "placebo")]
        xs = [g[k_][0] / g[k_][1] for k_ in sorted(g)]
        st = day_stats(xs, 0, t_eff_div)
        st["events"] = sum(c for _, c in g.values())
        st["mean_bp_per_candidate"] = sum(s_ for s_, _ in g.values()) / st["events"]
        m = E["paths"][(pa, "meta")]
        st["taken_share"], st["mean_size"] = m["taken_share"], m["mean_size"]      # the placebo reshuffles the meta sizes, so these two are the meta's
        placebo_stats[pa] = st
        r = prow[(pa, "placebo")]
        for f_, want in (("events", st["events"]), ("days", st["days"]), ("mean_bp_per_candidate", st["mean_bp_per_candidate"]), ("sr_daily", st["sr_daily"]), ("sr_annual", st["sr_annual"]),
                         ("skew", st["skew"]), ("kurt", st["kurt"]), ("psr", st["psr"]), ("taken_share", m["taken_share"]), ("mean_size", m["mean_size"])):
            if not close(float(r[f_]), float(want), 1e-6, 1e-8):
                pbad.append((pa, "placebo", f_, r[f_], want))
    check("paths.csv: every statistic of the primary and meta variants recomputed from the events; placebo statistics recomputed from daily.csv, with the meta sizes' taken share and mean size",
          not pbad and len(path_rows) == 3 * n_paths, str(pbad[:4]))

    def summarize(per_path):
        """per_path: {variant: [stat dict per path]} -> summary like the run's."""
        sm = {}
        for v, lst in per_path.items():
            sm[v] = {"paths": len(lst), "mean_bp_per_candidate": mean([s["mean_bp_per_candidate"] for s in lst]), "mean_sr_daily": mean([s["sr_daily"] for s in lst]),
                     "mean_sr_annual": mean([s["sr_annual"] for s in lst]), "paths_sr_negative": sum(1 for s in lst if s["sr_daily"] < 0), "mean_psr": mean([s["psr"] for s in lst]),
                     "taken_share": mean([s["taken_share"] for s in lst]), "mean_size": mean([s["mean_size"] for s in lst]), "events_per_path": mean([s["events"] for s in lst])}
        return sm

    def summary_checks(label, per_path, run_summ, n_p):
        sm = summarize(per_path)
        bad = []
        for v in ("primary", "meta", "placebo"):
            for f_, want in sm[v].items():
                if not close(float(run_summ[v][f_]), float(want), 1e-6, 1e-8):
                    bad.append((v, f_, run_summ[v][f_], want))
        sm_p, sm_m, sm_q = per_path["placebo"], per_path["meta"], per_path["primary"]
        diff_pl = [a["sr_daily"] - b["sr_daily"] for a, b in zip(sm_m, sm_p)]
        diff_pr = [a["sr_daily"] - b["sr_daily"] for a, b in zip(sm_m, sm_q)]
        mp, mq = run_summ["meta_minus_placebo"], run_summ["meta_minus_primary"]
        for f_, want in (("paths_sr_higher", sum(1 for x in diff_pl if x > 0)), ("paths", len(diff_pl)), ("mean_sr_daily_diff", mean(diff_pl)),
                         ("mean_bp_diff", mean([a["mean_bp_per_candidate"] - b["mean_bp_per_candidate"] for a, b in zip(sm_m, sm_p)]))):
            if not close(float(mp[f_]), float(want), 1e-6, 1e-8):
                bad.append(("meta_minus_placebo", f_, mp[f_], want))
        for f_, want in (("paths_sr_higher", sum(1 for x in diff_pr if x > 0)), ("paths", len(diff_pr)), ("mean_sr_daily_diff", mean(diff_pr))):
            if not close(float(mq[f_]), float(want), 1e-6, 1e-8):
                bad.append(("meta_minus_primary", f_, mq[f_], want))
        check(f"{label}: summary table (per-variant means over paths, paths with negative Sharpe, meta minus placebo / primary) recomputed", not bad, str(bad[:4]))

    per_path = {v: [E["paths"][(pa, v)] if v != "placebo" else placebo_stats[pa] for pa in range(n_paths)] for v in ("primary", "meta", "placebo")}
    summary_checks("CPCV", per_path, summ["eval"]["summary"], n_paths)

    def dsr_table(meta_stats, prior):
        srs = [s["sr_daily"] for s in meta_stats]
        sr_hat = mean(srs)
        var_paths = (sum((x - sr_hat) ** 2 for x in srs) / (len(srs) - 1)) if len(srs) > 1 else NAN
        t_eff, sk, ku = mean([s["t_eff"] for s in meta_stats]), mean([s["skew"] for s in meta_stats]), mean([s["kurt"] for s in meta_stats])
        var_asym = (1.0 - sk * sr_hat + (ku - 1.0) / 4.0 * sr_hat ** 2) / max(t_eff - 1.0, 1.0) if fin(sr_hat) else NAN
        cands = [v for v in (var_paths, var_asym) if fin(v)]
        var_sr = max(cands) if cands else NAN
        table = {}
        for n_tr in sorted({1, 10, int(prior), int(prior) + 1}):
            if n_tr <= 1:
                table[str(n_tr)] = (0.0, V.psr(sr_hat, t_eff, sk, ku, 0.0))
            else:
                star = expected_max_sharpe(n_tr, var_sr) if fin(var_sr) else NAN
                table[str(n_tr)] = (star, V.psr(sr_hat, t_eff, sk, ku, star) if fin(star) else NAN)
        return {"sr_daily_mean": sr_hat, "var_sr_used": var_sr, "var_across_paths": var_paths, "var_asymptotic": var_asym, "t_eff": t_eff, "skew": sk, "kurt": ku}, table

    def dsr_check(label, meta_stats, run_dsr):
        head, table = dsr_table(meta_stats, params["prior_trials"])
        bad = []
        for f_, want in head.items():
            if not close(float(run_dsr[f_]), float(want), 1e-6, 1e-9):
                bad.append((f_, run_dsr[f_], want))
        if set(run_dsr["by_trials"]) != set(table):
            bad.append(("trial counts", sorted(run_dsr["by_trials"]), sorted(table)))
        else:
            for k_, (star, dsr) in table.items():
                if not (close(float(run_dsr["by_trials"][k_]["sr_star"]), star, 1e-6, 1e-9) and close(float(run_dsr["by_trials"][k_]["dsr"]), dsr, 1e-6, 1e-9)):
                    bad.append((k_, run_dsr["by_trials"][k_], (star, dsr)))
        check(f"{label}: deflated Sharpe table (Sharpe variance = larger of across-path and asymptotic; trial counts 1, 10, prior, prior+1) recomputed with own normal quantiles", not bad, str(bad[:3]))

    dsr_check("CPCV", per_path["meta"], summ["eval"]["dsr"])
    check("deflated Sharpe trial counts include the earlier looks (100: 83 after the 14-feature run, +4 +1 three-algorithm study, +6 regime study, +6 fast-drop study) and the one more for this model (101)", int(params["prior_trials"]) == PRIOR_TRIALS and {"1", "10", str(int(params["prior_trials"])), str(int(params["prior_trials"]) + 1)} == set(summ["eval"]["dsr"]["by_trials"]),
          str(sorted(summ["eval"]["dsr"]["by_trials"])))

    # window lines
    def window_checks(label, E_, daily_rows_by_key, run_win, n_p, placebo_from_daily=True):
        bad = []
        for wn, (lo, hi) in win_epoch.items():
            dlo, dhi = lo // 86400, hi // 86400
            for v in ("primary", "meta", "placebo"):
                tot, cnt = 0.0, 0
                for pa in range(n_p):
                    g = daily_rows_by_key[(pa, v)]
                    for k_, (s_, c_) in g.items():
                        if dlo <= k_ < dhi:
                            tot += s_
                            cnt += c_
                row = run_win[f"{v}|{wn}"]
                want_mean = tot / cnt if cnt else NAN
                if int(row["events_total_over_paths"]) != cnt or not close(float(row["events_per_path"]), cnt / n_p, 1e-9, 1e-9) or not close(float(row["mean_bp_per_candidate"]), want_mean, 1e-6, 1e-7):
                    bad.append((wn, v, row, (cnt, want_mean)))
                if v in ("primary", "meta"):
                    s2, c2 = E_["win_sums"][(v, wn)]
                    if c2 != cnt or not close(s2, tot, 1e-7, 1e-5):
                        bad.append((wn, v, "events vs daily", (c2, s2), (cnt, tot)))
        check(f"{label}: window lines (June 2026, sell-off, deepest fall) for primary, meta and placebo, recomputed from events (primary, meta) and from the daily rows (all three)", not bad, str(bad[:3]))

    dg_all = dict(dgot)
    window_checks("CPCV", E, dg_all, summ["eval"]["windows"], n_paths)

    # ---- walk-forward ----
    cut = float(V.ts_key(params["cutoff"]))
    wf_set = sorted(i for i in range(n_ev) if t0[i] >= cut)
    wf_train = sum(1 for i in range(n_ev) if t1[i] < cut)
    wf_pred = [(int(r["event"]), float(r["p"])) for r in wf_rows]
    wmet = summ["walk_forward"]
    y = [binv[e_] for e_, _ in wf_pred]
    p = [q for _, q in wf_pred]
    w = [wt[e_] for e_, _ in wf_pred]
    pc = [min(max(q, 1e-6), 1 - 1e-6) for q in p]
    ll = -sum(wi * (yi * math.log(q) + (1 - yi) * math.log(1 - q)) for wi, yi, q in zip(w, y, pc)) / sum(w)
    wf_bad = []
    if sorted(e_ for e_, _ in wf_pred) != wf_set:
        wf_bad.append("event set")
    if int(wmet["n_train"]) != wf_train or int(wmet["n_test"]) != len(wf_set):
        wf_bad.append(("counts", wmet["n_train"], wf_train, wmet["n_test"], len(wf_set)))
    train_ev = [i for i in range(n_ev) if t1[i] < cut]
    if not close(float(wmet["avg_uniqueness"]), mean([uu[i] for i in train_ev]), 1e-9, 1e-12):
        wf_bad.append(("avg_uniqueness", wmet["avg_uniqueness"]))
    for k_, v in (("auc", auc(y, p)), ("log_loss_w", ll), ("base_rate", mean(y)), ("share_p_gt_half", mean([1.0 if q > 0.5 else 0.0 for q in p])),
                  ("acc", mean([1.0 if (q > 0.5) == bool(yy) else 0.0 for q, yy in zip(p, y)]))):
        if not close(float(wmet[k_]), v, 1e-6, 1e-9):
            wf_bad.append((k_, wmet[k_], v))
    check("walk-forward: training events are those whose labels end before the cutoff, test events start on or after it (none in both), counts, uniqueness, AUC, log-loss, base rate, accuracy recomputed",
          not wf_bad and max(t1[i] for i in train_ev) < cut <= min(t0[i] for i in wf_set), str(wf_bad[:3]))
    EW = evaluate({0: wf_pred})
    per_w = {"primary": [EW["paths"][(0, "primary")]], "meta": [EW["paths"][(0, "meta")]]}
    # the walk-forward placebo has no daily file: only its structure is checkable here (taken share and mean size equal the meta's)
    wsum = summ["walk_forward_eval"]["summary"]
    wbad = []
    sm = summarize(per_w)
    for v in ("primary", "meta"):
        for f_, want in sm[v].items():
            if not close(float(wsum[v][f_]), float(want), 1e-6, 1e-8):
                wbad.append((v, f_, wsum[v][f_], want))
    for f_ in ("taken_share", "mean_size"):
        if not close(float(wsum["placebo"][f_]), float(sm["meta"][f_]), 1e-9, 1e-12):
            wbad.append(("placebo", f_, wsum["placebo"][f_], sm["meta"][f_]))
    check("walk-forward: primary and meta summary rows recomputed; placebo has the meta's taken share and mean size", not wbad, str(wbad[:3]))
    dsr_check("walk-forward", per_w["meta"], summ["walk_forward_eval"]["dsr"])
    bad = []
    for wn, (lo, hi) in win_epoch.items():
        for v in ("primary", "meta"):
            s2, c2 = EW["win_sums"][(v, wn)]
            row = summ["walk_forward_eval"]["windows"][f"{v}|{wn}"]
            want = s2 / c2 if c2 else NAN
            if int(row["events_total_over_paths"]) != c2 or not close(float(row["mean_bp_per_candidate"]), want, 1e-6, 1e-7):
                bad.append((wn, v, row, (c2, want)))
        cnt_all = sum(1 for e_, _ in wf_pred if lo <= t0[e_] < hi)
        if int(summ["walk_forward_eval"]["windows"][f"placebo|{wn}"]["events_total_over_paths"]) != cnt_all:
            bad.append((wn, "placebo events"))
    check("walk-forward: window lines for primary and meta recomputed, placebo event counts per window", not bad, str(bad[:3]))

    def state_check(label, E_, run_eval, n_expected):
        bad = []
        got, want = run_eval["by_state"], E_["by_state"]
        if set(got) != set(want):
            bad.append(("keys", sorted(set(got) ^ set(want))))
        for k_ in want:
            if k_ in got:
                for f_ in ("events_per_path", "mean_bp_per_candidate", "mean_size"):
                    if not close(float(got[k_][f_]), float(want[k_][f_]), 1e-6, 1e-8):
                        bad.append((k_, f_, got[k_][f_], want[k_][f_]))
        gw, ww = run_eval["window_state_events"], E_["window_state_events"]
        for k_ in set(gw) | set(ww):
            if not close(float(gw.get(k_, 0)), float(ww.get(k_, 0)), 1e-9, 1e-9):
                bad.append(("window events", k_, gw.get(k_), ww.get(k_)))
        n_state_sum = sum(want[f"primary|state{sv}"]["events_per_path"] for sv in (0, 1))
        if not close(n_state_sum, n_expected, 1e-9, 1e-9):
            bad.append(("state events do not add up to the number of scored events", n_state_sum, n_expected))
        check(f"{label}: by-state table (events, mean bp for primary and meta, mean size, long / short split of the primary, events per window by state) recomputed", not bad, str(bad[:4]))

    state_check("CPCV", E, summ["eval"], n_ev)
    state_check("walk-forward", EW, summ["walk_forward_eval"], len(wf_set))

    # ---- aggregate numbers in the summary json and the rendered SUMMARY.md ----
    cp = summ["cpcv"]
    aucs = [float(r["auc"]) for r in split_rows]
    check("summary_numbers.json cpcv block equals the mean / min / max of the per-split metrics",
          close(cp["auc_mean"], mean(aucs), 1e-9, 1e-12) and close(cp["auc_min"], min(aucs), 1e-9, 1e-12) and close(cp["auc_max"], max(aucs), 1e-9, 1e-12)
          and close(cp["taken_mean"], mean([float(r["share_p_gt_half"]) for r in split_rows]), 1e-9, 1e-12) and close(cp["avg_u_mean"], mean(train_u_mean), 1e-9, 1e-12))
    fdr = [float(r["d"]) for r in fd_rows]
    fs = summ["fracdiff"]
    srt = sorted(fdr)
    med = (srt[len(srt) // 2] if len(srt) % 2 else 0.5 * (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]))
    check("summary_numbers.json fracdiff block (symbols, mean / median / min / max d, mean correlation)",
          fs["symbols"] == len(fdr) and close(fs["d_mean"], mean(fdr), 1e-9, 1e-12) and close(fs["d_median"], med, 1e-9, 1e-12) and fs["d_min"] == min(fdr) and fs["d_max"] == max(fdr)
          and close(fs["corr_mean"], mean([float(r["corr"]) for r in fd_rows]), 1e-9, 1e-12))
    md = (d / "SUMMARY.md").read_text(encoding="utf-8")
    miss = []
    for v in ("primary", "meta", "placebo"):
        s = summ["eval"]["summary"][v]
        line = next((ln for ln in md.splitlines() if ln.startswith(f"| {v} |")), "")
        for txt in (f"{s['mean_bp_per_candidate']:.1f}", f"{s['mean_sr_daily']:.4f}", f"{s['mean_sr_annual']:.2f}"):
            if txt not in line:
                miss.append((v, txt))
    for n_tr, v in summ["eval"]["dsr"]["by_trials"].items():
        line = next((ln for ln in md.splitlines() if ln.startswith(f"| {n_tr} |")), "")
        if f"{v['dsr']:.3f}" not in line:
            miss.append(("dsr", n_tr))
    check("SUMMARY.md shows the CPCV table and the deflated-Sharpe table exactly as in summary_numbers.json; deviations and both required windows are listed",
          not miss and "Deviations from the book" in md and "june_2026_ai_correction" in md and any("sell_off" in ln for ln in md.splitlines()), str(miss[:3]))

    # ---- importance files: structure only ----
    imp_bad = []
    for fn in ("mda.csv", "sfi.csv"):
        pth = d / fn
        if pth.exists() and pth.stat().st_size > 5:
            rows_i = V.read(pth)
            names = [list(r.values())[0] for r in rows_i]
            if sorted(names) != sorted(feats):
                imp_bad.append((fn, "features"))
            if not all(fin(float(r["mean"])) for r in rows_i):
                imp_bad.append((fn, "non-finite"))
    check("mda.csv / sfi.csv list every model feature with finite values when present (values themselves are covered by the unit tests, not here)", not imp_bad, str(imp_bad))


    # ---- market-run additions: source run, market columns, comparison section ----
    src_dir = Path(source_override) if source_override else Path(summ["market"]["source_folder"])
    if not (src_dir / "events.csv").exists():
        check("source meta-pipeline run is present", False, str(src_dir))
        return report("source run not found; pass its folder as the second argument")
    s_summ = json.loads((src_dir / "summary_numbers.json").read_text(encoding="utf-8"))
    check("the source run's events.csv and summary_numbers.json are the files this run used (md5), and the source run uses the declared 14 features",
          hashlib.md5((src_dir / "events.csv").read_bytes()).hexdigest() == summ["market"]["source_events_md5"]
          and hashlib.md5((src_dir / "summary_numbers.json").read_bytes()).hexdigest() == summ["market"]["source_summary_md5"] and s_summ["params"]["features"] == DECLARED_FEATURES[:14], str(s_summ["params"]["features"]))
    src_ev = V.read(src_dir / "events.csv")
    shared = [c for c in src_ev[0].keys()]
    same = len(src_ev) == n_ev and all(
        (a[c] == b[c]) or (fin(V.num(a[c])) and fin(V.num(b[c])) and close(V.num(a[c]), V.num(b[c]), 1e-11, 1e-14)) for a, b in zip(src_ev, ev) for c in shared)
    check("every column of the source run's events.csv (the 14 features, labels, weights, times) equals the same column of this run's events.csv row for row; the three market columns are the only addition",
          same and [c for c in ev[0].keys() if c not in shared] == MARKET, f"extra columns {[c for c in ev[0].keys() if c not in shared]}")
    check("the three market features are finite for every event (no event was dropped for them)", all(fin(V.num(e[c])) for e in ev for c in MARKET))
    pr = V.num
    def f_(x, nd):
        return "n/a" if x is None or not fin(float(x)) else f"{float(x):.{nd}f}"
    a_, b_ = s_summ, summ
    cmp_rows = []
    def add(name, x, y, nd):
        dd_ = (y - x) if (fin(x) and fin(y)) else NAN
        cmp_rows.append((name, f_(x, nd), f_(y, nd), f_(dd_, nd)))
    add("CPCV mean AUC", a_["cpcv"]["auc_mean"], b_["cpcv"]["auc_mean"], 4)
    add("CPCV meta: bp per candidate", a_["eval"]["summary"]["meta"]["mean_bp_per_candidate"], b_["eval"]["summary"]["meta"]["mean_bp_per_candidate"], 1)
    add("CPCV meta: daily Sharpe", a_["eval"]["summary"]["meta"]["mean_sr_daily"], b_["eval"]["summary"]["meta"]["mean_sr_daily"], 4)
    add("CPCV meta: annualised", a_["eval"]["summary"]["meta"]["mean_sr_annual"], b_["eval"]["summary"]["meta"]["mean_sr_annual"], 2)
    add("walk-forward AUC", a_["walk_forward"]["auc"], b_["walk_forward"]["auc"], 4)
    add("walk-forward meta: bp per candidate", a_["walk_forward_eval"]["summary"]["meta"]["mean_bp_per_candidate"], b_["walk_forward_eval"]["summary"]["meta"]["mean_bp_per_candidate"], 1)
    add("walk-forward meta: daily Sharpe", a_["walk_forward_eval"]["summary"]["meta"]["mean_sr_daily"], b_["walk_forward_eval"]["summary"]["meta"]["mean_sr_daily"], 4)
    for wn in meta["windows"]:
        add(f"CPCV meta bp per candidate, {wn}", a_["eval"]["windows"][f"meta|{wn}"]["mean_bp_per_candidate"], b_["eval"]["windows"][f"meta|{wn}"]["mean_bp_per_candidate"], 1)
    for wn in meta["windows"]:
        add(f"walk-forward meta bp per candidate, {wn}", a_["walk_forward_eval"]["windows"][f"meta|{wn}"]["mean_bp_per_candidate"], b_["walk_forward_eval"]["windows"][f"meta|{wn}"]["mean_bp_per_candidate"], 1)
    md2 = (d / "SUMMARY.md").read_text(encoding="utf-8")
    found2 = {tuple(c.strip() for c in ln.strip().strip("|").split("|")) for ln in md2.splitlines() if ln.startswith("|")}
    miss2 = [r_ for r_ in cmp_rows if r_ not in found2]
    # the summary's own numbers must be the ones computed from the run files: AUC mean from split_metrics.csv, walk-forward AUC
    auc_mean = mean([V.num(r["auc"]) for r in split_rows])
    miss2 += [("auc", auc_mean)] if not close(auc_mean, float(b_["cpcv"]["auc_mean"]), 1e-9, 1e-12) else []
    imp_rows = {}
    for fn in ("mda.csv", "sfi.csv"):
        if (d / fn).exists() and (d / fn).stat().st_size > 5:
            rows_i = V.read(d / fn)
            imp_rows[fn] = {list(r.values())[0]: float(r["mean"]) for r in rows_i}
    if "mda.csv" in imp_rows and "sfi.csv" in imp_rows:
        for fn, key in (("mda.csv", "mda"), ("sfi.csv", "sfi")):
            for f2, v2 in imp_rows[fn].items():
                if not close(v2, float(summ[key][f2]["mean"]), 1e-9, 1e-12):
                    miss2.append((fn, f2))
        order = sorted(imp_rows["mda.csv"], key=lambda f2: -imp_rows["mda.csv"][f2])
        for f2 in MARKET:
            if f2 not in order or f2 not in imp_rows["sfi.csv"]:
                miss2.append((f2, "missing from mda.csv / sfi.csv"))
                continue
            miss2 += [(f2, "rank row")] if (f2, str(order.index(f2) + 1), f_(imp_rows["mda.csv"][f2], 4), f_(imp_rows["sfi.csv"][f2], 4)) not in found2 else []
    check("SUMMARY.md comparison with the 14-feature run (AUC, meta bp and Sharpe, walk-forward, each window) equals the numbers of the source run's and this run's summaries, and the rank table of the market features equals mda.csv / sfi.csv",
          not miss2 and "Compared with the 14-feature run" in md2, str(miss2[:3]))


    # ---- SUMMARY.md main tables, cell by cell (the inherited check only looked for the numbers as substrings) ----
    def ff(x, nd):
        return "n/a" if x is None or not fin(float(x)) else f"{float(x):.{nd}f}"
    exp_rows_md = []
    for v_ in ("primary", "meta", "placebo"):
        s_ = summ["eval"]["summary"][v_]
        exp_rows_md.append((v_, ff(s_["mean_bp_per_candidate"], 1), ff(s_["mean_sr_daily"], 4), ff(s_["mean_sr_annual"], 2), f"{s_['paths_sr_negative']}/{s_['paths']}", ff(s_["mean_psr"], 2), ff(s_["taken_share"], 2), ff(s_["mean_size"], 3)))
        w_ = summ["walk_forward_eval"]["summary"][v_]
        exp_rows_md.append((v_, ff(w_["mean_bp_per_candidate"], 1), ff(w_["mean_sr_daily"], 4), ff(w_["mean_sr_annual"], 2), ff(w_["mean_psr"], 2), ff(w_["taken_share"], 2)))
    for n_tr, x_ in summ["eval"]["dsr"]["by_trials"].items():
        exp_rows_md.append((str(n_tr), ff(x_["sr_star"], 4), ff(x_["dsr"], 3)))
    for srcname, key_ in (("CPCV", "eval"), ("walk-forward", "walk_forward_eval")):
        for wn in meta["windows"]:
            ww = summ[key_]["windows"]
            exp_rows_md.append((wn, srcname, ff(ww[f"primary|{wn}"]["mean_bp_per_candidate"], 1), ff(ww[f"meta|{wn}"]["mean_bp_per_candidate"], 1), ff(ww[f"placebo|{wn}"]["mean_bp_per_candidate"], 1), ff(ww[f"primary|{wn}"]["events_per_path"], 0)))
        bs_ = summ[key_]["by_state"]
        for stv in (1, 0):
            pr_, me_ = bs_[f"primary|state{stv}"], bs_[f"meta|state{stv}"]
            lo_, sh_ = bs_[f"primary|state{stv}|side+1"], bs_[f"primary|state{stv}|side-1"]
            exp_rows_md.append((str(stv), f"{pr_['events_per_path']:.0f}", ff(pr_["mean_bp_per_candidate"], 1), ff(me_["mean_bp_per_candidate"], 1), ff(me_["mean_size"], 3),
                                f"{ff(lo_['mean_bp_per_candidate'], 1)} ({lo_['events_per_path']:.0f})", f"{ff(sh_['mean_bp_per_candidate'], 1)} ({sh_['events_per_path']:.0f})"))
    for f2 in feats:
        a2, b2 = summ["mda"].get(f2, {}), summ["sfi"].get(f2, {})
        exp_rows_md.append((f2, ff(a2.get("mean"), 4), ff(a2.get("std"), 4), ff(b2.get("mean"), 4), ff(b2.get("std"), 4)))
    cpl = summ["cpcv"]
    auc_line = (f"Out-of-sample AUC per split: mean {ff(cpl['auc_mean'], 3)}, min {ff(cpl['auc_min'], 3)}, max {ff(cpl['auc_max'], 3)} (0.5 = no information). "
                f"Share of events with p > 0.5 (taken): {ff(cpl['taken_mean'], 2)}. Average uniqueness of training events: {ff(cpl['avg_u_mean'], 3)}.")
    wfl = summ["walk_forward"]
    wf_line = f"Train events {wfl['n_train']:,}, test events {wfl['n_test']:,}, AUC {ff(wfl['auc'], 3)}, share with p > 0.5 {ff(wfl['share_p_gt_half'], 2)}."
    miss3 = [r_ for r_ in exp_rows_md if r_ not in found2]
    for ln_ in (auc_line, wf_line):
        if ln_ not in md2:
            miss3.append(("line", ln_[:60]))
    check("SUMMARY.md: every number of the CPCV table, deflated-Sharpe table, walk-forward table, window lines, by-state tables and the feature-importance table (all 17 features) equals summary_numbers.json cell by cell, and the AUC lines equal the split metrics",
          not miss3, str(miss3[:3]))

    return report()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    raise SystemExit(main(argv[0], argv[1] if len(argv) > 1 else None))
