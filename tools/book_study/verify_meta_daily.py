#!/usr/bin/env python
"""Independent checker for a run_meta_daily.py output folder (Alpaca daily US-equity bars; stdlib only; imports only helper functions of
verify_book_study.py and of the already tested verify_meta_venue.py: own FFD / ADF, SADF, entropy, bet size, AUC, normal quantiles).

From the raw Alpaca file (alpaca_daily.csv.gz, all symbols) it rebuilds, with its own code and without pandas:
  * the month-end universe: for every completed month-end the screen (adjusted close > 5, volume > 0, >= 253 bars, >= 40 valid dollar-volume days in the
    last 60 calendar rows, ranking by the trailing-60-day median dollar volume, ties by symbol, top N) and the eligible counts,
  * EVERY event the run should contain: the CUSUM events (threshold = the one-bar sigma), membership from the last month-end strictly before the signal day,
    the next-bar entry, the calendar-gap rule, the side from the project's momentum score, the book's daily-volatility target, the triple-barrier exit
    (literal loop, strict touches), the meta-label, uniqueness and return-attribution weights, the fractional-differentiation d and feature, the
    trailing-return and volatility features, SADF, both entropies, and the market state and distance -- then compares the set of events and every column
    with events.csv, and the per-symbol accounting with symbols.csv,
  * the three reporting windows (June 2026, the sell-off, the deepest 90-day fall of the market index) and their symbol coverage.
From events.csv alone it re-derives the CPCV group edges, purge / embargo and the train / test counts of every split, path membership of every
prediction, AUC / log-loss / accuracy per split, bet sizes, the daily series of the primary and meta variants, Sharpe (annualised with sqrt 252), skew,
kurtosis, PSR, the cross-path summary, the deflated-Sharpe table, the window lines, the by-state tables and the walk-forward counts and results.
Placebo rows are checked for what can be checked without numpy's random permutation (same events per day, same size multiset, internal consistency
between daily.csv and paths.csv, and window lines from daily.csv).
Not covered by this checker: the forest's internals, the sequential bootstrap and the MDA / SFI values (unit tests cover those; here only that the
files list every feature with finite numbers).
Exit code 0 = all checks pass; 1 = a check failed; 2 = a malformed file stopped the checker.

    python tools/book_study/verify_meta_daily.py runtime/book_meta_alpaca/<timestamp> [path to alpaca_daily.csv.gz if it moved]
"""
from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import sys
import time
from array import array
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path
from statistics import NormalDist, median

import verify_book_study as V
from verify_meta_venue import (adf_stat, auc, bet_size, close, crit5, expected_max_sharpe, ffd_series, fin, konto_py, mean, min_d,        # noqa: F401
                               plugin_py, sadf_py)

HERE = Path(__file__).resolve().parent
NAN = math.nan
EPOCH_ORD = date(1970, 1, 1).toordinal()
ANNUAL_DAYS = 252
# settings declared before the run (restated here independently of the study code)
SIGMA_FAST, SIGMA_SLOW, TRGT_SPAN = 14, 84, 100
LB12, LB6, SKIP = 252, 126, 21
LOOKBACKS = ((1, "mom1n"), (5, "mom5n"), (21, "mom21n"), (63, "mom63n"))
DV_WIN, DV_MIN, MIN_PRICE, MIN_HIST, MIN_ELIGIBLE_START = 60, 40, 5.0, 253, 200
MKT_DAYS, MKT_MIN_PERIODS, MKT_MIN_SYMBOLS = 200, 130, 10
SADF_W, SADF_MINSL, ENT_W = 100, 20, 100
DECLARED_FEATURES = ["side", "ffd", "mom1n", "mom5n", "mom21n", "mom63n", "log_trgt", "vol_ratio", "mkt_state", "mkt_dist", "sadf", "ent_plugin", "ent_konto"]
DECLARED_MICRO = {"sadf": [100, 20, 1], "entropy": [100, 2]}


def iso(day: int) -> str:
    return date.fromordinal(EPOCH_ORD + int(day)).isoformat()


def day_stats(x, events, t_eff_div):
    n, sr, sk, ku = V.moments(x)
    t_eff = n / t_eff_div
    return {"days": n, "sr_daily": sr, "sr_annual": sr * math.sqrt(ANNUAL_DAYS) if fin(sr) else NAN, "skew": sk, "kurt": ku, "t_eff": t_eff,
            "psr": V.psr(sr, t_eff, sk, ku) if fin(sr) else NAN}


def cusum_daily(r, sigma):
    """Snippet 2.4 with threshold = the one-bar sigma itself (one bar = one day); only the side that fires is reset; a NaN return counts as zero."""
    sp = sn = 0.0
    ev = []
    for i in range(1, len(r)):
        x = r[i] if not math.isnan(r[i]) else 0.0
        sp, sn = max(0.0, sp + x), min(0.0, sn + x)
        t = sigma[i]
        if math.isnan(t):
            continue
        if sn < -t:
            sn = 0.0
            ev.append(i)
        elif sp > t:
            sp = 0.0
            ev.append(i)
    return ev


def read_alpaca(path):
    """{symbol: (days array 'i', closes array 'd', volumes array 'd')} from the exported csv(.gz); a row needs a numeric close > 0 (as the study's loader);
    a repeated (symbol, day) keeps the LAST row. Also returns the number of repeated rows and of rows dropped for a bad close."""
    opener = gzip.open if str(path).endswith(".gz") else open
    per, cache = {}, {}
    n_rows = n_bad = 0
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {c: i for i, c in enumerate(head)}
        i_s, i_d, i_c, i_v = ix["symbol"], ix["date"], ix["close"], ix["volume"]
        for row in rd:
            n_rows += 1
            try:
                c = float(row[i_c])
            except ValueError:
                c = NAN
            if not (c > 0) or math.isinf(c):
                n_bad += 1
                continue
            ds = row[i_d]
            dd = cache.get(ds)
            if dd is None:
                dd = cache[ds] = int(V.ts_key(ds) // 86400)
            try:
                v = float(row[i_v])
            except ValueError:
                v = NAN
            e = per.get(row[i_s])
            if e is None:
                e = per[row[i_s]] = (array("i"), array("d"), array("d"))
            e[0].append(dd)
            e[1].append(c)
            e[2].append(v)
    dups = 0
    for s, (dl, cl, vl) in per.items():
        if all(dl[i] < dl[i + 1] for i in range(len(dl) - 1)):
            continue
        last = {}
        for k in range(len(dl)):
            if dl[k] in last:
                dups += 1
            last[dl[k]] = (cl[k], vl[k])
        ds_ = sorted(last)
        per[s] = (array("i", ds_), array("d", [last[d_][0] for d_ in ds_]), array("d", [last[d_][1] for d_ in ds_]))
    return per, dups, n_rows, n_bad


def rebuild_universe(per, cal_days, cal_idx, n_universe):
    """Completed month-ends (the month of the last calendar day is not complete): ({day: [members]}, {day: eligible count}, sorted list of days)."""
    by_month = {}
    for d_ in cal_days:
        dt = date.fromordinal(EPOCH_ORD + d_)
        by_month[(dt.year, dt.month)] = d_
    last_dt = date.fromordinal(EPOCH_ORD + cal_days[-1])
    ends = sorted(d_ for k, d_ in by_month.items() if k != (last_dt.year, last_dt.month))
    pos = {s: array("i", [cal_idx[d_] for d_ in e[0]]) for s, e in per.items()}
    uni, counts, started = {}, {}, False
    for m in ends:
        pm = cal_idx[m]
        cand = []
        for s, (dl, cl, vl) in per.items():
            ps = pos[s]
            j = bisect.bisect_right(ps, pm)
            if j < MIN_HIST or ps[j - 1] != pm:
                continue
            if not (cl[j - 1] > MIN_PRICE and vl[j - 1] > 0):
                continue
            lo = bisect.bisect_left(ps, pm - DV_WIN + 1)
            dv = [cl[k] * vl[k] for k in range(lo, j) if vl[k] > 0]
            if len(dv) < DV_MIN:
                continue
            cand.append((-median(dv), s))
        counts[m] = len(cand)
        started = started or counts[m] >= MIN_ELIGIBLE_START
        cand.sort()
        uni[m] = [s for _, s in cand[:n_universe]] if started else []
    return uni, counts, ends


def rebuild(csv_path, n_universe, h, ptsl, log):
    """Everything that depends only on the raw file and the declared settings."""
    per, dups, n_rows, n_bad = read_alpaca(csv_path)
    cal_days = sorted(set().union(*[set(e[0]) for e in per.values()]))
    cal_idx = {d_: i for i, d_ in enumerate(cal_days)}
    log(f"read {n_rows:,} rows, {len(per):,} symbols, {len(cal_days):,} trading days {iso(cal_days[0])} -> {iso(cal_days[-1])}")
    uni, counts, ends = rebuild_universe(per, cal_days, cal_idx, n_universe)
    members = sorted({s for v in uni.values() for s in v})
    log(f"rebuilt the universe: {len(ends)} completed month-ends, {len(members)} symbols were ever members")
    sets = [set(uni[m]) for m in ends]

    # market index: equal weight over the members' own consecutive-close log returns, days with >= 10 returns
    day_sum, day_cnt = defaultdict(float), defaultdict(int)
    logs = {}
    for s in members:
        dl, cl, _ = per[s]
        lc = array("d", [math.log(c) for c in cl])
        logs[s] = lc
        for i in range(1, len(dl)):
            day_sum[dl[i]] += lc[i] - lc[i - 1]
            day_cnt[dl[i]] += 1
    idx_days = sorted(d_ for d_ in day_cnt if day_cnt[d_] >= MKT_MIN_SYMBOLS)
    cum, level, csum = 0.0, [], [0.0]
    for d_ in idx_days:
        cum += day_sum[d_] / day_cnt[d_]
        v = math.exp(cum)
        level.append(v)
        csum.append(csum[-1] + v)

    def state_dist(day):
        i = bisect.bisect_left(idx_days, day)
        if i >= len(idx_days) or idx_days[i] != day:
            return -1, NAN
        lo = bisect.bisect_right(idx_days, day - MKT_DAYS)
        n = i + 1 - lo
        if n < MKT_MIN_PERIODS:
            return -1, NAN
        ma = (csum[i + 1] - csum[lo]) / n
        return (1 if level[i] > ma else 0), math.log(level[i] / ma)

    rows_out, sym_info, fd_exp = [], {}, {}
    n_kept0 = n_drop_side = n_drop_feat = 0
    for si, s in enumerate(members):
        dl, cl, _ = per[s]
        n = len(cl)
        lc = logs[s]
        rr = [NAN] + [lc[i] - lc[i - 1] for i in range(1, n)]
        sg = V.ewm_sigma(rr, SIGMA_FAST)
        evp = cusum_daily(rr, sg)
        ev_m = []
        for p in evp:
            k = bisect.bisect_left(ends, dl[p]) - 1
            if k >= 0 and s in sets[k]:
                ev_m.append(p)
        ev_end = [p for p in ev_m if p + h + 2 < n and fin(sg[p]) and sg[p] > 0]
        kept = [p for p in ev_end if cal_idx[dl[p + h + 2]] - cal_idx[dl[p]] == h + 2]
        sym_info[s] = {"bars": n, "first": iso(dl[0]), "last": iso(dl[-1]), "events_total": len(evp), "not_member": len(evp) - len(ev_m),
                       "dropped_end_or_sigma": len(ev_m) - len(ev_end), "dropped_gap": len(ev_end) - len(kept), "events_kept": len(kept)}
        n_kept0 += len(kept)
        if not kept:
            continue
        r1 = [NAN] + [cl[i] / cl[i - 1] - 1.0 for i in range(1, n)]
        trgt_all = V.ewm_sigma(r1, TRGT_SPAN)
        slow = V.ewm_sigma(rr, SIGMA_SLOW)
        x = list(lc)
        dsel, stat, cr, corr, f = min_d(x, 0.01)
        fd_exp[s] = (dsel, stat, cr, corr, n)
        bits = "".join("1" if x[k] - x[k - 1] > 0 else "0" for k in range(1, n))          # bit of day k is bits[k-1]
        sub = []
        for p in kept:                                   # p = signal bar; the trade is entered at the next bar's close
            side = 0
            if p >= LB12:
                a = cl[p - SKIP]
                score = 0.5 * (a / cl[p - LB12] - 1.0) + 0.5 * (a / cl[p - LB6] - 1.0)
                side = 1 if score > 0 else -1 if score < 0 else 0
            trgt = trgt_all[p]
            if not (side != 0 and fin(trgt) and trgt > 0):
                n_drop_side += 1
                continue
            pe = p + 1
            row = {"symbol": s, "pos": pe, "t_sig": int(dl[p]) * 86400, "t0": int(dl[pe]) * 86400, "sigma": sg[p], "trgt": trgt, "side": side, "log_trgt": math.log(trgt)}
            for L, nm in LOOKBACKS:
                row[nm] = (lc[p] - lc[p - L]) / (sg[p] * math.sqrt(L)) if p >= L else NAN
            row["vol_ratio"] = math.log(sg[p] / slow[p]) if fin(slow[p]) and slow[p] > 0 else NAN
            row["ffd"] = f[p]
            sd = sadf_py(x[p - SADF_W + 1:p + 1]) if p >= SADF_W - 1 else NAN
            if p >= ENT_W:
                msg = bits[p - ENT_W:p]                  # returns of days p-ENT_W+1 .. p, oldest first
                row["sadf"], row["ent_plugin"], row["ent_konto"] = sd, plugin_py(msg), konto_py(msg[::-1])
            else:
                row["sadf"], row["ent_plugin"], row["ent_konto"] = sd, NAN, NAN
            st, dist = state_dist(int(dl[p]))
            row["mkt_state"] = float(st) if st >= 0 else NAN
            row["mkt_dist"] = dist
            pt, sl = ptsl[0] * trgt, -ptsl[1] * trgt
            Rs, jx, kd, rt = [], None, 0, None
            for t in range(1, h + 2):
                R = (math.exp(lc[pe + t] - lc[pe]) - 1.0) * side
                Rs.append(R)
                if jx is None:
                    if R > pt:
                        jx, kd, rt = t, 1, R
                    elif R < sl:
                        jx, kd, rt = t, -1, R
            if jx is None:
                jx, kd, rt = h + 1, 0, Rs[-1]
            row["_Rs"], row["_pt"], row["_sl"] = Rs, pt, sl
            row["exit_j"], row["kind"], row["ret_side"], row["t1pos"] = jx - 1, kd, rt, pe + jx
            row["t1"] = int(dl[pe + jx]) * 86400
            row["bin"] = 1 if rt > 0 else 0
            vals = [row[k_] for k_ in DECLARED_FEATURES]
            if not (all(fin(v) for v in vals) and fin(rt)):
                n_drop_feat += 1
                continue
            sub.append(row)
        if sub:                                          # uniqueness and return-attribution weights over this symbol's kept events (bars pos .. t1pos)
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
                a_, b_ = r["pos"], r["t1pos"]
                r["u"] = (pref_inv[b_ + 1] - pref_inv[a_]) / (b_ - a_ + 1)
                r["w_raw"] = abs(pref_c[b_ + 1] - pref_c[a_])
        rows_out.extend(sub)
        if (si + 1) % 100 == 0:
            log(f"  events rebuilt for {si + 1}/{len(members)} symbols ({len(rows_out):,} so far)")
    rows_out.sort(key=lambda r: (r["t0"], r["symbol"]))
    tot_w = sum(r["w_raw"] for r in rows_out)
    for r in rows_out:
        r["w"] = r["w_raw"] * len(rows_out) / tot_w

    # deepest fall of the equal-weight index: peak at most 90 days before the trough, rolling maximum over the previous 91 observations
    fall = None
    if len(idx_days) >= 3:
        dd_ = [level[i] / max(level[max(0, i - 90):i + 1]) - 1.0 for i in range(len(level))]
        it = min(range(len(dd_)), key=lambda i: (dd_[i], i))
        win = [i for i in range(len(idx_days)) if idx_days[it] - 90 <= idx_days[i] <= idx_days[it]]
        ip = max(win, key=lambda i: (level[i], -i))
        fall = (idx_days[ip], idx_days[it], dd_[it] * 100.0)
    return {"dups": dups, "n_rows": n_rows, "n_bad": n_bad, "n_symbols_file": len(per), "cal_days": cal_days, "uni": uni, "counts": counts, "ends": ends, "members": members,
            "sym_info": sym_info, "fd_exp": fd_exp, "rows": rows_out, "n_kept0": n_kept0, "n_drop_side": n_drop_side, "n_drop_feat": n_drop_feat, "fall": fall,
            "member_days": {s: array("i", per[s][0]) for s in members}}


def main(folder: str, csv_override: str = "") -> int:
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
    uni_rows = V.read(d / "universe.csv")
    cnt_rows = V.read(d / "universe_counts.csv")
    skipped_rows = V.read(d / "skipped.csv")
    h, ptsl, cost = int(params["h"]), params["ptsl"], float(params["cost_frac"])
    step, N, K = float(params["step"]), int(params["n_groups"]), int(params["k"])
    embargo_pct = float(params["embargo_pct"])
    feats = params["features"]
    dp = params["daily"]
    n_ev = len(ev)
    say(f"loaded {n_ev:,} events, {len(pred_rows):,} cpcv predictions")

    # ---- declared settings and run bookkeeping ----
    check("declared settings: vertical barrier 5 bars after entry, +/-2 daily-vol barriers, FFD threshold 0.01, side = 12-1 / 6-1 momentum score, 20 bp round trip, the 13-feature list, SADF window 100 / min 20 / 1 lag, entropy 100 days / word 2",
          h == 4 and [float(x) for x in ptsl] == [2.0, 2.0] and float(params["ffd_thres"]) == 0.01 and params["side_signal"] == "mom12_1_6_1"
          and close(cost, 0.0020, 1e-12, 1e-12) and feats == DECLARED_FEATURES and params.get("micro") == DECLARED_MICRO,
          str({k: params.get(k) for k in ("h", "ptsl", "ffd_thres", "side_signal", "cost_frac", "features", "micro")}))
    check("declared universe, volatility, momentum and market-state settings (span 14 / 84 / 100, lookbacks 1 / 5 / 21 / 63, 252 / 126 / skip 21, price > 5, 253 bars, 40 of 60 dollar-volume days, 200 eligible names to start, 200-day index mean with 130 observations and 10 symbols)",
          dp["sigma_fast"] == SIGMA_FAST and dp["sigma_slow"] == SIGMA_SLOW and dp["trgt_span"] == TRGT_SPAN and [tuple(x) for x in dp["lookbacks"]] == list(LOOKBACKS)
          and (dp["lb12"], dp["lb6"], dp["skip"]) == (LB12, LB6, SKIP) and float(dp["min_price"]) == MIN_PRICE and dp["min_hist"] == MIN_HIST and (dp["dv_win"], dp["dv_min"]) == (DV_WIN, DV_MIN)
          and dp["min_eligible_start"] == MIN_ELIGIBLE_START and dp["mkt_window"] == f"{MKT_DAYS}D" and dp["mkt_min_periods"] == MKT_MIN_PERIODS and dp["mkt_min_symbols"] == MKT_MIN_SYMBOLS,
          str(dp))
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

    # ---- raw file: identity, then a full rebuild (cached only when VERIFY_DAILY_CACHE is set, which the planted-error tests use) ----
    csv_path = Path(csv_override) if csv_override else Path(meta["alpaca_csv"])
    csv_md5 = hashlib.md5(csv_path.read_bytes()).hexdigest() if csv_path.exists() else ""
    man_path = csv_path.parent.parent / "manifest.json"
    man_md5 = json.loads(man_path.read_text(encoding="utf-8")).get("alpaca", {}).get("csv_md5") if man_path.exists() else None
    check("the Alpaca file read here is the file the run used (md5 recorded in the run) and the file the export recorded (md5 in manifest.json next to the candle folder)",
          csv_path.exists() and csv_md5 == meta["alpaca_csv_md5"] and csv_md5 == man_md5, f"{csv_path} exists={csv_path.exists()} file {csv_md5} run {meta['alpaca_csv_md5']} manifest {man_md5}")
    n_univ = int(dp["n_universe"])
    cache_file = os.environ.get("VERIFY_DAILY_CACHE")
    key = f"{csv_md5}|{n_univ}|{h}|{ptsl}"
    R = None
    if cache_file and Path(cache_file).exists():
        with open(cache_file, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("key") == key:
            R = blob["R"]
            say("rebuild loaded from the test cache")
    if R is None:
        R = rebuild(csv_path, n_univ, h, [float(x) for x in ptsl], say)
        if cache_file:
            with open(cache_file, "wb") as fh:
                pickle.dump({"key": key, "R": R}, fh)
    say(f"rebuilt {len(R['rows']):,} events from the raw file")
    check("raw file: no repeated (symbol, day) rows; the rows without a positive close are the ones the study drops", R["dups"] == 0, f"{R['dups']} repeated rows, {R['n_bad']} bad-close rows of {R['n_rows']}")
    exp_rows, members, sym_info = R["rows"], R["members"], R["sym_info"]

    # ---- universe files ----
    cal_days, ends, uni, counts = R["cal_days"], R["ends"], R["uni"], R["counts"]
    got_uni = defaultdict(list)
    for r in uni_rows:
        got_uni[r["month_end"]].append((int(r["rank"]), r["symbol"]))
    uni_bad = []
    if sorted(got_uni) != sorted(iso(m) for m in ends if uni[m]):
        uni_bad.append(("month-ends with members", len(got_uni), sum(1 for m in ends if uni[m])))
    for m in ends:
        if not uni[m]:
            continue
        g = sorted(got_uni.get(iso(m), []))
        if [r_ for r_, _ in g] != list(range(1, len(g) + 1)) or [s_ for _, s_ in g] != uni[m]:
            uni_bad.append((iso(m), len(g), len(uni[m])))
    check("universe.csv: for every completed month-end the members and their order (trailing-60-day median dollar volume, ties by symbol) equal the universe rebuilt from the raw file", not uni_bad and len(uni_rows) == sum(len(v) for v in uni.values()), str(uni_bad[:3]))
    cnt_bad = []
    if [r["month_end"] for r in cnt_rows] != [iso(m) for m in ends]:
        cnt_bad.append("month-end list")
    else:
        for r, m in zip(cnt_rows, ends):
            if int(r["eligible"]) != counts[m] or int(r["members"]) != len(uni[m]):
                cnt_bad.append((r["month_end"], r["eligible"], counts[m], r["members"], len(uni[m])))
    first_m = next((m for m in ends if uni[m]), None)
    check("universe_counts.csv: eligible names and members per completed month-end equal the rebuilt counts; the universe starts at the first month-end with 200 eligible names; the month of the last date is not used",
          not cnt_bad and meta["universe_start"] == (iso(first_m) if first_m is not None else None) and int(meta["n_universe"]) == n_univ and int(meta["calendar_days"]) == len(cal_days)
          and max(ends) < cal_days[-1] and (date.fromordinal(EPOCH_ORD + max(ends)).month != date.fromordinal(EPOCH_ORD + cal_days[-1]).month), str(cnt_bad[:3]))

    # ---- symbols ----
    rows_by_sym = {r["symbol"]: r for r in symbols}
    sym_list = sorted(rows_by_sym)
    check("symbols.csv lists exactly the symbols that were ever members, once each, and the run used that many; nothing was skipped",
          sym_list == members and len(symbols) == len(members) == int(meta["symbols_used"]) and len(skipped_rows) == 0, f"{len(sym_list)} {len(members)} {meta['symbols_used']} skipped {len(skipped_rows)}")
    bad_acc = []
    for s in members:
        i, e = sym_info[s], rows_by_sym.get(s)
        if e is None or not (int(e["bars"]) == i["bars"] and e["first"][:10] == i["first"] and e["last"][:10] == i["last"]
                             and all(int(e[k_]) == i[k_] for k_ in ("events_total", "not_member", "dropped_end_or_sigma", "dropped_gap", "events_kept"))):
            bad_acc.append((s, e, i))
    check("per symbol: bars, first and last bar, CUSUM events, events outside the universe, events dropped for the file end or no volatility, events dropped for a missing calendar day on the path, events kept -- all replayed from the raw file",
          not bad_acc, str(bad_acc[:2]))
    info_ev = summ["events_info"]
    n_cusum, n_drop_side, n_drop_feat = R["n_kept0"], R["n_drop_side"], R["n_drop_feat"]
    check("event accounting: events kept after universe / end / gap, dropped for no side / no volatility, dropped for a missing feature or state, kept",
          n_cusum == int(info_ev["events_cusum"]) == sum(int(r["events_kept"]) for r in symbols) and n_drop_side == int(info_ev["events_dropped_no_side_or_vol"])
          and n_drop_feat == int(info_ev["events_dropped_missing_feature"]) and len(exp_rows) == int(info_ev["events_kept"]) == n_ev == int(meta["events"]),
          f"after gap {n_cusum}/{info_ev['events_cusum']}, no side {n_drop_side}/{info_ev['events_dropped_no_side_or_vol']}, missing {n_drop_feat}/{info_ev['events_dropped_missing_feature']}, kept {len(exp_rows)}/{n_ev}")
    key_exp = [(r["symbol"], r["pos"]) for r in exp_rows]
    key_got = [(e["symbol"], int(e["pos"])) for e in ev]
    check("the set of events and their order (entry time, then symbol) equal the rebuilt ones", key_exp == key_got, f"{len(set(key_exp) ^ set(key_got))} differ")
    if set(key_exp) != set(key_got) or len(key_exp) != len(key_got):
        return report("events.csv does not contain the rebuilt set of events, so the event-indexed checks (splits, predictions, daily results) are meaningless")
    for fname, rows_x in (("cpcv_predictions.csv", pred_rows), ("walk_forward_predictions.csv", wf_rows)):
        if any(not (0 <= int(r["event"]) < n_ev) for r in rows_x):
            check(f"{fname} refers only to rows of events.csv", False, "an event index is outside events.csv")
            return report(f"{fname} refers to events that do not exist")

    # column by column, aligned on (symbol, pos)
    got_by_key = {(e["symbol"], int(e["pos"])): e for e in ev}
    int_cols = ("t0", "t1", "t_sig", "t1pos", "side", "exit_j", "kind", "bin")
    tol = {"sigma": (1e-6, 1e-12), "trgt": (1e-6, 1e-12), "log_trgt": (1e-6, 1e-9), "ret_side": (1e-6, 1e-7), "net_ret": (1e-6, 1e-7), "u": (1e-8, 1e-10),
           "w_raw": (1e-7, 1e-12), "w": (1e-7, 1e-9), "ffd": (1e-6, 1e-8), "mom1n": (1e-6, 1e-8), "mom5n": (1e-6, 1e-8), "mom21n": (1e-6, 1e-8),
           "mom63n": (1e-6, 1e-8), "vol_ratio": (1e-5, 1e-8), "mkt_state": (0, 0), "mkt_dist": (1e-6, 1e-9),
           "sadf": (1e-6, 1e-7), "ent_plugin": (1e-9, 1e-12), "ent_konto": (1e-9, 1e-12)}
    bad_col = defaultdict(list)
    barrier_tied = 0
    for r in exp_rows:
        r["net_ret"] = r["ret_side"] - cost
        e = got_by_key.get((r["symbol"], r["pos"]))
        if e is None:
            continue
        stored = (int(float(e["exit_j"])), int(float(e["kind"])), float(e["ret_side"]))
        mine = (r["exit_j"], r["kind"], r["ret_side"])
        if stored[0] != mine[0] or stored[1] != mine[1] or abs(stored[2] - mine[2]) > 1e-6:
            hi_t = max(stored[0], mine[0]) + 1
            near = any(abs(R_ - r["_pt"]) < 1e-6 or abs(R_ - r["_sl"]) < 1e-6 for R_ in r["_Rs"][:hi_t])
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
    check("every column of every event (signal and entry time, sigma, target, side, barrier exit / kind / return, meta-label, t1, uniqueness, attribution weight, scaled weight, "
          "features, state, distance, net return) equals the rebuilt value", not bad_col, str({k_: (len(v), v[:2]) for k_, v in bad_col.items()}))
    say(f"events compared; {barrier_tied} events with a barrier touch within 1e-6 tolerated as float32 ties")
    check("barrier ties tolerated (float32 rounding of the stored path) are a negligible share", barrier_tied <= max(5, n_ev // 2000), f"{barrier_tied} of {n_ev}")
    check("every event is entered after its signal (t0 > t_sig, next trading day) and its label ends after entry (t1 > t0)",
          all(int(float(e["t0"])) > int(float(e["t_sig"])) and int(float(e["t1"])) > int(float(e["t0"])) for e in ev))

    ret = [V.num(e["ret_side"]) for e in ev]
    binv = [int(float(e["bin"])) for e in ev]
    t0 = [int(float(e["t0"])) for e in ev]
    t1 = [int(float(e["t1"])) for e in ev]
    wt = [V.num(e["w"]) for e in ev]
    uu = [V.num(e["u"]) for e in ev]
    syms_ev = [e["symbol"] for e in ev]
    day_ev = [t // 86400 for t in t0]
    check("sorted by time then symbol; weights average 1; uniqueness in (0,1]; meta-label = 1 exactly when the side's return is positive; net = gross - cost",
          all((a, s1) <= (b, s2) for a, s1, b, s2 in zip(t0, syms_ev, t0[1:], syms_ev[1:])) and abs(mean(wt) - 1.0) < 1e-6 and all(0 < u <= 1 + 1e-12 for u in uu)
          and all(b == (1 if r > 0 else 0) for b, r in zip(binv, ret)) and all(close(V.num(e["net_ret"]), V.num(e["ret_side"]) - cost, 1e-9, 1e-12) for e in ev))

    # ---- fractional differentiation file ----
    fd_bad = []
    fd_by = {r["symbol"]: r for r in fd_rows}
    with_events = sorted(s for s in members if sym_info[s]["events_kept"] > 0)
    for s in with_events:
        r = fd_by.get(s)
        if r is None:
            fd_bad.append((s, "missing"))
            continue
        dsel, stat, cr, corr, n_b = R["fd_exp"][s]
        got = (float(r["d"]), float(r["adf"]), float(r["crit"]), float(r["corr"]))
        if not (close(got[0], dsel, 1e-12, 1e-12) and close(got[1], stat, 1e-4, 1e-6) and close(got[2], cr, 1e-6, 1e-9) and close(got[3], corr, 1e-5, 1e-8) and int(r["days"]) == n_b):
            fd_bad.append((s, got, (dsel, stat, cr, corr, n_b)))
    check("fracdiff.csv: one row per symbol that has events; smallest d, ADF statistic, critical value, correlation and series length recomputed (own FFD and ADF) for every one",
          not fd_bad and len(fd_rows) == len(with_events) == len(fd_by), str(fd_bad[:3]))
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
    t_eff_div = max(1.0, float(h + 1))

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
    check("deflated Sharpe trial counts include the earlier looks (102: 101 before this run) and the one more for this run (look 103)", int(params["prior_trials"]) == 102 and {"1", "10", str(int(params["prior_trials"])), str(int(params["prior_trials"]) + 1)} == set(summ["eval"]["dsr"]["by_trials"]),
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

    # ---- SUMMARY.md main tables, cell by cell (the base checker only looked for the numbers as substrings) ----
    def ff(x, nd):
        return "n/a" if x is None or not fin(float(x)) else f"{float(x):.{nd}f}"
    found2 = {tuple(c.strip() for c in ln.strip().strip("|").split("|")) for ln in md.splitlines() if ln.startswith("|")}
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
        if ln_ not in md:
            miss3.append(("line", ln_[:60]))
    for pair_ in (f"- data: Alpaca daily bars, SIP feed, dividend- and split-adjusted", f"- round-trip cost (bp): {cost * 1e4:.1f}",
                  f"- universe: top {n_univ} by trailing-60-day median dollar volume at completed month-ends (price > 5)", f"- symbols used: {len(members)}", f"- events: {n_ev:,}"):
        if pair_ not in md.splitlines():
            miss3.append(("param line", pair_))
    check("SUMMARY.md: every number of the CPCV table, deflated-Sharpe table, walk-forward table, window lines, by-state tables and the feature-importance table equals summary_numbers.json cell by cell, the AUC lines equal the split metrics, and the data, universe, cost, symbol and event lines are shown",
          not miss3, str(miss3[:3]))


    # ---- shares, windows and coverage recomputed from the raw file ----
    share_ok = (close(float(info_ev["bin_share"]), mean([float(b) for b in binv]), 1e-9, 1e-12)
                and all(close(float(info_ev["kind_share"][str(k_)]), sum(1 for e in ev if int(float(e["kind"])) == k_) / n_ev, 1e-9, 1e-12) for k_ in (1, -1, 0)))
    check("events_info: share of events by barrier exit (profit-take, stop-loss, vertical) and the meta-label share equal the shares in events.csv", share_ok)
    fall = R["fall"]
    fall_name = "deepest_90d_fall_full_history"
    fall_ok = (fall is not None and fall_name in win_epoch and win_epoch[fall_name] == (fall[0] * 86400, (fall[1] + 1) * 86400) and f"{fall[2]:.1f}%" in meta["window_notes"].get(fall_name, ""))
    check("the third window is the deepest fall of the equal-weight index with its peak at most 90 days before its trough, recomputed from the raw file (peak day, trough day, percentage)",
          fall_ok, f"rebuilt {fall and (iso(fall[0]), iso(fall[1]), round(fall[2], 1))}, run {meta['window_epoch'].get(fall_name)} {meta['window_notes'].get(fall_name)}")
    cov = {}
    for wn, (lo, hi) in win_epoch.items():
        n_cov = 0
        for s in members:
            dd = R["member_days"][s]
            j = bisect.bisect_left(dd, lo // 86400)
            if j < len(dd) and dd[j] < hi // 86400:
                n_cov += 1
        cov[wn] = n_cov
    cov_sentence = (f"Candle coverage of the reporting windows (symbols with at least one bar inside, of {len(members)} used): " + "; ".join(f"{k_} {n_}" for k_, n_ in cov.items())
                    + ". A window with no entered events is reported n/a; a label needs 5 bars after entry, so events entered in the last 5 trading days of the file cannot be labelled.")
    check("window coverage (symbols with a bar inside each reporting window) recomputed from the raw file equals the run's record and is stated in SUMMARY.md",
          {k_: int(v) for k_, v in meta["window_symbol_coverage"].items()} == cov and any(ln.endswith(cov_sentence) for ln in md.splitlines()), str((cov, meta["window_symbol_coverage"])))
    last_day = cal_days[-1]
    check("no event is labelled beyond the last bar of the file (the last 5 trading days cannot be labelled): every label end t1 is on or before the last trading day",
          max(int(float(e["t1"])) for e in ev) <= last_day * 86400, f"{max(int(float(e['t1'])) for e in ev)} vs {last_day * 86400}")

    return report()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    try:
        raise SystemExit(main(argv[0], argv[1] if len(argv) > 1 else ""))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as exc:      # a malformed run folder is a failure of the run, not a pass
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
