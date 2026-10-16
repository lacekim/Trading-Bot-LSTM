#!/usr/bin/env python
"""Independent checker for a book-study output folder (stdlib only; imports none of the study code).

It re-derives, with its own code and from the raw candle files:
  1. cell bookkeeping: expected number of cells, group edges, purge/embargo training counts, test counts per state
  2. headline numbers in summary_numbers.json from oos_daily.csv (mean bp, daily Sharpe, PSR, window lines, path counts)
  3. a CUSUM replay of up to 3 symbols from raw candles (event positions must equal events_index.csv)
  4. every sampled event in samples.csv: price position, sigma, barrier exit (value, step, kind), basis points, cost, state
  5. rule flags (overfit = Sharpe below mesh median), break-even precision formula, models-untouched flag
Exit code 0 = all pass.

    python tools/book_study/verify_book_study.py runtime/book_study/<timestamp> [--symbols N]
"""
from __future__ import annotations

import bisect
import calendar
import csv
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import NormalDist


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def same(a: float, b: float, rel: float = 1e-6, ab: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= max(ab, rel * max(abs(a), abs(b)))


def ts_key(s) -> float:
    """UTC epoch seconds from a number (s or ms) or a 'YYYY-MM-DD HH:MM:SS' / ISO string."""
    try:
        v = float(s)
        return v / 1000.0 if abs(v) >= 1e11 else v
    except (TypeError, ValueError):
        pass
    s = str(s).strip().replace("T", " ").replace("Z", "")
    if "+" in s[10:]:
        s = s[: 10 + s[10:].index("+")]
    s = s.split(".")[0]
    fmt = "%Y-%m-%d %H:%M:%S" if len(s) > 10 else "%Y-%m-%d"
    return float(calendar.timegm(__import__("time").strptime(s, fmt)))


def load_raw_close(gmx_dir: str, tv_dir: str, symbol: str, timeframe: str = "1h"):
    """(sorted epoch list, closes): GMX candles plus only TradingView candles older than the first GMX candle; closes > 0 only."""
    gmx, tv = {}, {}
    gpath = Path(gmx_dir) / f"gmx_arbitrum_{symbol}_{timeframe}.csv"
    if gpath.exists():
        for r in read(gpath):
            c = num(r.get("close", r.get("Close")))
            t = r.get("open_time", r.get("Date"))
            if t is not None and not math.isnan(c):
                gmx[ts_key(t)] = c
    tpath = Path(tv_dir) / f"tradingview_{symbol}_{timeframe}.csv"
    if tpath.exists():
        for r in read(tpath):
            c = num(r.get("Close", r.get("close")))
            t = r.get("Date", r.get("time"))
            if t is not None and not math.isnan(c):
                tv[ts_key(t)] = c
    first_gmx = min(gmx) if gmx else math.inf
    merged = {t: c for t, c in tv.items() if t < first_gmx}
    merged.update(gmx)
    keys = sorted(k for k in merged if merged[k] > 0)
    return keys, [merged[k] for k in keys]


def ewm_sigma(r: list, span: int) -> list:
    """pandas Series.ewm(span, min_periods=span).std() with adjust=True, ignore_na=False, bias=False; r[0] is NaN."""
    alpha = 2.0 / (span + 1.0)
    d = 1.0 - alpha
    s0 = s1 = sxx = s2 = 0.0
    count = 0
    out = [math.nan] * len(r)
    for i, x in enumerate(r):
        s0 *= d
        s1 *= d
        sxx *= d
        s2 *= d * d
        if not math.isnan(x):
            s0 += 1.0
            s1 += x
            sxx += x * x
            s2 += 1.0
            count += 1
        if count >= span and s0 > 0:
            den = s0 * s0 - s2
            if den > 0:
                var = (sxx / s0 - (s1 / s0) ** 2) * (s0 * s0) / den
                out[i] = math.sqrt(var) if var > 0 else 0.0
    return out


def cusum(r: list, sigma: list) -> list:
    sp = sn = 0.0
    ev = []
    h24 = math.sqrt(24.0)
    for i in range(1, len(r)):
        x = r[i] if not math.isnan(r[i]) else 0.0
        sp, sn = max(0.0, sp + x), min(0.0, sn + x)
        t = sigma[i] * h24
        if math.isnan(t):
            continue
        if sn < -t:
            sn = 0.0
            ev.append(i)
        elif sp > t:
            sp = 0.0
            ev.append(i)
    return ev


def moments(x: list):
    n = len(x)
    if n < 4:
        return n, math.nan, math.nan, math.nan
    m = sum(x) / n
    v1 = sum((a - m) ** 2 for a in x) / (n - 1)
    s = math.sqrt(v1)
    if s <= 0:
        return n, math.nan, math.nan, math.nan
    s0 = math.sqrt(sum((a - m) ** 2 for a in x) / n)
    sk = sum(((a - m) / s0) ** 3 for a in x) / n
    ku = sum(((a - m) / s0) ** 4 for a in x) / n
    return n, m / s, sk, ku


def psr(sr, t, sk, ku, star=0.0):
    if any(math.isnan(v) for v in (sr, t, sk, ku)) or t <= 1:
        return math.nan
    var = 1.0 - sk * sr + (ku - 1.0) / 4.0 * sr * sr
    if var <= 0:
        return math.nan
    return NormalDist().cdf((sr - star) * math.sqrt(t - 1.0) / math.sqrt(var))


def literal_exit(logc: list, pos: int, sign: float, sigma: float, pt: float, sl: float, h: int):
    """Barrier exit for one event, Snippet-13.2 style: step t = 1..h+1, strict crossings, vertical barrier at t = h+1."""
    for t in range(1, h + 2):
        v = sign * (logc[pos + t] - logc[pos]) / sigma
        if v > pt:
            return v, t, 1
        if v < -sl:
            return v, t, -1
        if t == h + 1:
            return v, t, 0
    raise AssertionError


def main(folder: str, n_symbols: int) -> int:
    d = Path(folder)
    checks: list = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    summary = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    meta, run = summary["meta"], summary["run"]
    integrity = json.loads((d / "integrity.json").read_text(encoding="utf-8"))
    cells = read(d / "cells.csv")
    daily = read(d / "oos_daily.csv")
    samples = read(d / "samples.csv")
    events = read(d / "events_index.csv")
    symbols = read(d / "symbols.csv")
    cost_frac = meta["cost_frac"]

    # ---- 1. bookkeeping ----
    check("models and calibration files untouched during the run", integrity.get("unchanged") is True)
    n_groups, k = meta["n_groups"], meta["k"]
    splits = list(combinations(range(n_groups), k))
    check("number of splits = C(N,k)", len(splits) == meta["n_splits"], f"{len(splits)} vs {meta['n_splits']}")
    check("number of backtest paths = C(N-1,k-1)", math.comb(n_groups - 1, k - 1) == meta["n_paths"])
    sig_names = meta.get("signals", [])
    rows_per_split = (len(sig_names) * 3 + 1) if meta.get("only_signals") else (7 + 3 * len(sig_names))
    check("cells.csv has horizons x splits x 7 rows", len(cells) == len(meta["horizons"]) * len(splits) * rows_per_split,
          f"{len(cells)} vs {len(meta['horizons']) * len(splits) * rows_per_split}")
    edges = meta["edges"]
    t0s = [float(e["t0_epoch"]) for e in events]
    tmin, tmax = min(t0s), max(t0s) + 1.0
    check("group edges are N equal-duration groups between first and last event",
          len(edges) == n_groups + 1 and all(same(edges[i], tmin + (tmax - tmin) * i / n_groups, 1e-9, 1e-3) for i in range(n_groups + 1)))
    embargo = meta["embargo_seconds"]
    check("embargo = embargo_pct x span (stored value is 1% of the span unless overridden)", embargo >= 0 and embargo <= 0.05 * (tmax - tmin))
    grp = [int(e["group"]) for e in events]
    ev_sig = {nm: [int(float(e["sig_" + nm])) for e in events] for nm in sig_names}
    idx_state = [int(float(e["state"])) for e in events]
    wrong_group = 0
    for t, g in zip(t0s, grp):
        gg = min(max(bisect.bisect_right(edges, t) - 1, 0), n_groups - 1)
        wrong_group += (gg != g)
    check("every event's group matches the edges", wrong_group == 0, f"{wrong_group} wrong")

    # expected n_train / n_test per cell
    state_ok = {"all": lambda s: True, "above": lambda s: s == 1, "below": lambda s: s == 0}
    bad = []
    by_group_state = defaultdict(int)
    for g, s in zip(grp, idx_state):
        by_group_state[(g, s)] += 1
    cell_rows = [c for c in cells]
    for c in cell_rows:
        h = int(c["horizon"])
        test_groups = [int(x) for x in c["test_groups"].split("-")]
        sname = c["state"]
        sg = ev_sig.get(c["direction"])
        n_test = sum(1 for i_, (g, s) in enumerate(zip(grp, idx_state)) if g in test_groups and state_ok[sname](s) and (sg is None or sg[i_] != 0))
        if n_test != int(c["n_test"]):
            bad.append(("n_test", c["horizon"], c["split"], sname, n_test, c["n_test"]))
            continue
    check("n_test of every cell equals events in its test groups and state", not bad, str(bad[:3]))

    # purge/embargo: independent count for a sample of cells (cheap loops per split)
    checked_splits = {}
    bad = []
    seen = set()
    for c in cell_rows:
        key = (c["horizon"], c["split"], c["state"], c["direction"])
        if key in seen or c["direction"] not in ("long", "mom168", "ma200d", "placebo") or (c["direction"] == "placebo" and c["state"] != "all"):
            continue
        seen.add(key)
        h = int(c["horizon"])
        sp = int(c["split"])
        if sp not in (0, len(splits) // 2, len(splits) - 1):
            continue
        test_groups = splits[sp]
        n_train = 0
        sg = ev_sig.get(c["direction"])
        for i_, (t, g, s) in enumerate(zip(t0s, grp, idx_state)):
            if not state_ok[c["state"]](s) or g in test_groups or (sg is not None and sg[i_] == 0):
                continue
            t1 = t + (h + 1) * 3600.0
            ok = True
            for tg in test_groups:
                lo, hi = edges[tg], edges[tg + 1]
                if (t <= hi and t1 >= lo) or (t > hi and t <= hi + embargo):
                    ok = False
                    break
            n_train += ok
        if n_train != int(c["n_train"]):
            bad.append((key, n_train, c["n_train"]))
    check("n_train (purge + embargo) recomputed independently for first/middle/last splits", not bad, str(bad[:3]))

    # ---- 5. rule flags and formulas ----
    done = [c for c in cells if c["skipped"] != "True"]
    bad = [c for c in done if (c["overfit_net"] == "True") != (float(c["test_sharpe_net"]) < float(c["mesh_median_sharpe_net"]))
           or (c["overfit_gross"] == "True") != (float(c["test_sharpe_gross"]) < float(c["mesh_median_sharpe_gross"]))]
    check("Definition-2 overfit flags agree with the stored Sharpe ratios", not bad, str(len(bad)))
    bad = [c for c in done if not (0.5 <= float(c["pt"]) <= 10.0 and 0.5 <= float(c["sl"]) <= 10.0)]
    check("chosen rules lie on the 0.5..10 sigma mesh", not bad)
    bad = [c for c in done if not 0.0 < float(c["phi"]) < 1.2]
    check("fitted phi is in a plausible range (0, 1.2)", not bad, str([c["phi"] for c in bad[:3]]))
    bad = [c for c in done if c["direction"] == "placebo" and abs(float(c["test_mean_gross_bp"])) > 500]
    check("placebo test means are not absurd", not bad)

    # ---- 2. headline numbers from oos_daily.csv ----
    grouped = defaultdict(lambda: defaultdict(list))
    for r in daily:
        grouped[(int(r["horizon"]), r["state"], r["direction"])][int(r["path"])].append(r)
    bad, bad_events = [], []
    psr_bad = []
    for (h, s, dn), paths in grouped.items():
        row = summary["rows"].get(f"H{h}|{s}|{dn}")
        if row is None:
            bad.append((h, s, dn, "missing row"))
            continue
        sr_list, mean_net, mean_gross, psr_list, ev_per_path = [], [], [], [], []
        for p, rs in paths.items():
            rs.sort(key=lambda r: int(r["day"]))
            dnet = [float(r["sum_net_bp"]) / float(r["n"]) for r in rs]
            n_ev = sum(float(r["n"]) for r in rs)
            n, sr, sk, ku = moments(dnet)
            t_eff = n / max(1.0, (h + 1) / 24.0)
            sr_list.append(sr)
            psr_list.append(psr(sr, t_eff, sk, ku))
            mean_net.append(sum(float(r["sum_net_bp"]) for r in rs) / n_ev)
            mean_gross.append(sum(float(r["sum_gross_bp"]) for r in rs) / n_ev)
            ev_per_path.append(n_ev)
        ok = (len(paths) == row["paths"] and same(sum(mean_net) / len(mean_net), row["mean_net_bp"], 1e-6, 1e-6)
              and same(sum(mean_gross) / len(mean_gross), row["mean_gross_bp"], 1e-6, 1e-6)
              and same(sum(ev_per_path) / len(ev_per_path), row["events_per_path"], 1e-9, 1e-6))
        fin = [v for v in sr_list if not math.isnan(v)]
        if fin:
            ok = ok and same(sum(fin) / len(fin), row["sr_net_mean"], 1e-6, 1e-9) and (sum(1 for v in fin if v < 0) == row["paths_sr_net_negative"])
        pf = [v for v in psr_list if not math.isnan(v)]
        if pf and not same(sum(pf) / len(pf), row["psr_net_mean"], 1e-6, 1e-9):
            psr_bad.append((h, s, dn))
        if not ok:
            bad.append((h, s, dn))
    check("headline rows (paths, events, mean bp, daily Sharpe, paths with SR<0) recomputed from oos_daily.csv", not bad, str(bad[:4]))
    check("mean PSR per row recomputed from oos_daily.csv", not psr_bad, str(psr_bad[:4]))

    n_trials = summary["trials"]["n"]
    sr_trials = [summary["rows"][k_]["sr_net_mean"] for k_ in summary["rows"] if not k_.endswith("|placebo")
                 and not math.isnan(summary["rows"][k_]["sr_net_mean"])]
    check("trial count = number of non-placebo (horizon, state, direction) rows", n_trials == len(sr_trials), f"{n_trials} vs {len(sr_trials)}")
    if len(sr_trials) > 1:
        m = sum(sr_trials) / len(sr_trials)
        var = sum((v - m) ** 2 for v in sr_trials) / (len(sr_trials) - 1)
        check("variance of trial Sharpe ratios", same(var, summary["trials"]["var_sr_net"], 1e-6, 1e-12), f"{var} vs {summary['trials']['var_sr_net']}")
        nd = NormalDist()
        g = 0.5772156649015329
        emax = math.sqrt(var) * ((1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
        check("expected maximum Sharpe (Eq. 53) recomputed", same(emax, summary["trials"]["expected_max_sr"], 1e-6, 1e-9), f"{emax} vs {summary['trials']['expected_max_sr']}")

    # every backtest path covers every group once -> events per path equal across paths (for non-skipped splits)
    spread = []
    for (h, s, dn), paths in grouped.items():
        tot = {p: sum(float(r["n"]) for r in rs) for p, rs in paths.items()}
        if tot and max(tot.values()) != min(tot.values()):
            spread.append((h, s, dn, min(tot.values()), max(tot.values())))
    check("each backtest path contains the same events (every group appears once per path)", not spread, str(spread[:3]))

    # window lines
    win_bad = []
    for wname, (w0, w1) in summary["windows"].items():
        lo, hi = int(ts_key(w0) // 86400), int(ts_key(w1) // 86400)
        for (h, s, dn), paths in grouped.items():
            row = summary["rows"][f"H{h}|{s}|{dn}"]["windows"].get(wname)
            vals, ns = [], []
            for p, rs in paths.items():
                rr = [r for r in rs if lo <= int(r["day"]) < hi]
                if rr:
                    n_ev = sum(float(r["n"]) for r in rr)
                    vals.append(sum(float(r["sum_net_bp"]) for r in rr) / n_ev)
                    ns.append(n_ev)
            if not vals:
                if row is not None:
                    win_bad.append((wname, h, s, dn, "should be none"))
            elif row is None or not same(sum(vals) / len(vals), row["mean_net_bp"], 1e-6, 1e-6) or not same(sum(ns) / len(ns), row["events_per_path"], 1e-9, 1e-6):
                win_bad.append((wname, h, s, dn))
    check("window lines (June 2026, sell-off, deepest fall) recomputed from oos_daily.csv", not win_bad, str(win_bad[:4]))
    check("June 2026 and a sell-off window are both reported", "june_2026_ai_correction" in summary["windows"] and
          any(k_.startswith("sell_off") for k_ in summary["windows"]) , str(list(summary["windows"])))

    # ---- 3 & 4: raw candle checks ----
    gmx_dir, tv_dir = run["gmx_dir"], run["tv_dir"] if "tv_dir" in run else run["tradingview_dir"]
    sym_ev = defaultdict(list)
    for e in events:
        sym_ev[e["symbol"]].append(int(e["pos"]))
    rows_by_sym = {r["symbol"]: r for r in symbols}
    span = run["span"]
    h_max = max(meta["horizons"])
    cache = {}

    def series(sym):
        if sym not in cache:
            keys, closes = load_raw_close(gmx_dir, tv_dir, sym)
            # the candle files keep growing after a run: use only candles the run could see (its recorded last bar per symbol)
            cut = ts_key(rows_by_sym[sym]["last"])
            n_keep = bisect.bisect_right(keys, cut)
            keys, closes = keys[:n_keep], closes[:n_keep]
            logc = [math.log(c) for c in closes]
            r = [math.nan] + [logc[i] - logc[i - 1] for i in range(1, len(logc))]
            cache[sym] = (keys, logc, r)
        return cache[sym]

    # CUSUM replay for the symbols with the most events (bounded by n_symbols)
    pick = sorted(sym_ev, key=lambda s: -len(sym_ev[s]))[:n_symbols]
    for sym in pick:
        keys, logc, r = series(sym)
        sg = ewm_sigma(r, span)
        ev_all = cusum(r, sg)
        info = rows_by_sym[sym]
        ok_count = len(ev_all) == int(info["events_total"]) and len(keys) == int(info["bars"])
        kept = [p for p in ev_all if p + h_max + 1 < len(keys) and sg[p] > 0 and (keys[p + h_max + 1] - keys[p]) == (h_max + 1) * 3600]
        ok_kept = sorted(kept) == sorted(sym_ev[sym])
        check(f"{sym}: bar count and CUSUM event total replayed from raw candles", ok_count, f"{len(ev_all)} vs {info['events_total']}, bars {len(keys)} vs {info['bars']}")
        check(f"{sym}: kept events (complete, gap-free forward window) equal events_index.csv", ok_kept, f"{len(kept)} vs {len(sym_ev[sym])}")

    # market index for state checks
    need_syms = sorted(rows_by_sym)
    hour_sum, hour_cnt = defaultdict(float), defaultdict(int)
    for sym in need_syms:
        keys, logc, r = series(sym)
        for i in range(1, len(keys)):
            hour_sum[keys[i]] += r[i]
            hour_cnt[keys[i]] += 1
    hours = sorted(t for t in hour_cnt if hour_cnt[t] >= 10)
    cum, level, cumsum_idx = 0.0, [], [0.0]
    for t in hours:
        cum += hour_sum[t] / hour_cnt[t]
        v = math.exp(cum)
        level.append(v)
        cumsum_idx.append(cumsum_idx[-1] + v)

    def state_at(t):
        i = bisect.bisect_left(hours, t)
        if i >= len(hours) or hours[i] != t:
            return -1
        lo = bisect.bisect_right(hours, t - 200 * 86400)
        n = i + 1 - lo
        if n < 3840:
            return -1
        ma = (cumsum_idx[i + 1] - cumsum_idx[lo]) / n
        return 1 if level[i] > ma else 0

    ev_state = {(e["symbol"], int(e["t0_epoch"])): int(float(e["state"])) for e in events}
    st_bad = 0
    st_rng = list(ev_state.items())
    for (sym, t), s_stored in st_rng:
        if state_at(float(t)) != s_stored:
            st_bad += 1
    check("market state recomputed from raw candles for every event", st_bad == 0, f"{st_bad} of {len(st_rng)} differ")


    cum_cache = {}

    def expected_signal(name, sym, pos):
        """Independent sign of a momentum / trend rule at bar `pos` from the raw candles (0 = no signal)."""
        keys, logc, r = series(sym)
        if name == "ma200d":
            if sym not in cum_cache:
                cs = [0.0]
                for lc in logc:
                    cs.append(cs[-1] + math.exp(lc))
                cum_cache[sym] = cs
            cs = cum_cache[sym]
            t = keys[pos]
            lo = bisect.bisect_right(keys, t - 200 * 86400)
            n = pos + 1 - lo
            if n < 3840:
                return 0.0
            ma = (cs[pos + 1] - cs[lo]) / n
            c = math.exp(logc[pos])
            return 1.0 if c > ma else (-1.0 if c < ma else 0.0)
        look = int(name.replace("mom", ""))
        if pos < look:
            return 0.0
        d = logc[pos] - logc[pos - look]
        return 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)

    # sampled events
    bad_pos, bad_sig, bad_exit, bad_bp, bad_state = [], [], [], [], []
    sig_cache = {}
    for sm in samples:
        sym = sm["symbol"]
        keys, logc, r = series(sym)
        pos = int(sm["pos"])
        if int(keys[pos]) != int(sm["t0_epoch"]):
            bad_pos.append((sym, pos))
            continue
        if sym not in sig_cache:
            sig_cache[sym] = ewm_sigma(r, span)
        sg = sig_cache[sym][pos]
        if not same(sg, float(sm["sigma"]), 1e-7, 1e-12):
            bad_sig.append((sym, pos, sg, sm["sigma"]))
        h = int(sm["horizon"])
        sign = float(sm["sign"])
        v, t, kind = literal_exit(logc, pos, sign, float(sm["sigma"]), float(sm["pt"]), float(sm["sl"]), h)
        if not (same(v, float(sm["exit_value"]), 1e-4, 1e-5) and t == int(sm["exit_step"]) and kind == int(sm["exit_kind"])):
            bad_exit.append((sym, pos, v, sm["exit_value"], t, sm["exit_step"], kind, sm["exit_kind"]))
        gross = v * float(sm["sigma"]) * 1e4
        if not (same(gross, float(sm["gross_bp"]), 1e-4, 1e-3) and same(gross - cost_frac * 1e4, float(sm["net_bp"]), 1e-4, 1e-3)):
            bad_bp.append((sym, pos, gross, sm["gross_bp"], sm["net_bp"]))
        if sm["direction"] in sig_names:
            want_sign = expected_signal(sm["direction"], sym, pos)
            if want_sign != sign:
                bad_bp.append((sym, pos, "signal sign", sm["direction"], sign, want_sign))
        if sm["direction"] == "long" and sign != 1.0 or sm["direction"] == "short" and sign != -1.0 or abs(sign) != 1.0:
            bad_bp.append((sym, pos, "sign", sm["direction"], sign))
        want = sm["state"]
        s_ev = ev_state.get((sym, int(sm["t0_epoch"])))
        if s_ev is None or not state_ok[want](s_ev):
            bad_state.append((sym, pos, want, s_ev))
    check(f"{len(samples)} sampled events: price position matches the candle at that bar", not bad_pos, str(bad_pos[:3]))
    check("sampled events: sigma recomputed from raw candles (EWM std, span %d)" % span, not bad_sig, str(bad_sig[:3]))
    check("sampled events: barrier exit (value, step, kind) re-run with a literal loop", not bad_exit, str(bad_exit[:3]))
    check("sampled events: basis points, round-trip cost and direction sign", not bad_bp, str(bad_bp[:3]))
    check("sampled events: the event belongs to the state of its cell", not bad_state, str(bad_state[:3]))
    check("there are sampled events from at least 3 splits", len({sm["split"] for sm in samples}) >= 3 or not samples)

    width = max(len(c[0]) for c in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {'' if ok else detail}")
        failed += (not ok)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    ns = 3
    if "--symbols" in sys.argv:
        ns = int(sys.argv[sys.argv.index("--symbols") + 1])
        argv = [a for a in argv if a != str(ns)]
    if not argv:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(argv[0], ns))
