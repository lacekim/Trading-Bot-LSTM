#!/usr/bin/env python
"""Independent checker for a run_hrp.py output folder (stdlib only; imports neither hrp_alloc.py, run_hrp.py, bet_averaging.py, numpy, pandas nor scipy).

From the source run's files (events.csv, walk_forward_predictions.csv, summary_numbers.json) and the raw candle CSVs it re-derives, with its own code:
  * the settings (730-day window, 100 active days, cutoff, cost, bet-size step, windows, schemes) and the md5 of the four scripts recorded in hrp_extra.json;
  * eligibility of every symbol (first and last bar day, events before / after the cutoff, active days in the window) and the number of dropped post-cutoff events;
  * the estimation matrix (daily strategy result per symbol by BRUTE FORCE: for each bar a loop over the events active there), the sample covariance (cov_bp2.csv, every cell);
  * the weights: equal weight, inverse variance, HRP (own single linkage with the stated child-order rule, own quasi-diagonal order and recursive bisection; linkage.csv heights and
    hrp_order.csv too), and for the minimum-variance weights the constraints (long-only, sum 1), the first-order optimality conditions and that its variance is not above that of the
    other three schemes' weights;
  * scheme_stats.csv (effective number of names, largest weight, zero weights, in-sample sd from the estimation matrix, out-of-sample sd);
  * for the primary stream and for the meta stream of every source run (bet sizes from walk_forward_predictions.csv with its own formula): the daily portfolio series of each scheme
    (brute force again, weights rescaled over the symbols with a bar that day), every row of portfolio_daily.csv, and every column of metrics.csv (days, mean, sd, Sharpe, window means,
    calendar-year means and Sharpes);
  * every cell of the tables and the eligibility / leaf-order sentences in hrp_summary.md.
Exit code 0 = all checks pass.

    python tools/book_study/verify_hrp.py runtime/book_hrp/<timestamp> [--candle-dir <folder>] [--runs <run folder> ...]
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import verify_book_study as V
import verify_bet_averaging as VB

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAN = math.nan
DAY = 86400
EST_DAYS, MIN_ACTIVE = 730, 100
SCHEMES = ["equal_weight", "inverse_variance", "hrp", "min_variance"]


def close(a, b, rel=1e-7, ab=1e-9):
    return V.same(a, b, rel, ab)


def mean(v):
    return sum(v) / len(v) if v else NAN


def sd1(v):
    if len(v) < 2:
        return NAN
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def sharpe_annual(v):
    if len(v) < 4:
        return NAN
    s = sd1(v)
    return mean(v) / s * math.sqrt(365) if s > 0 else NAN


def brute_symbol(ts, cl, events_pos, events_t1pos, signals, cost_side):
    """Per day (UTC day number of the bars of this symbol): net result (gross - cost) and the active flag, by looping over events active at every bar."""
    n = len(cl)
    S, C = [0.0] * n, [0] * n
    for p_, q_, sg in zip(events_pos, events_t1pos, signals):
        for k in range(p_, q_):
            S[k] += sg
            C[k] += 1
    net, act, present = defaultdict(float), set(), set()
    prev = 0.0
    for k in range(n):
        a = S[k] / C[k] if C[k] > 0 else 0.0
        dk = int(ts[k] // DAY)
        present.add(dk)
        if k <= n - 2:
            d1 = int(ts[k + 1] // DAY)
            net[d1] += a * (cl[k + 1] / cl[k] - 1.0)
            if C[k] > 0:
                act.add(dk)
                act.add(d1)
        net[dk] -= cost_side * abs(a - prev)
        prev = a
    return net, act, present


def inverse_var(cov):
    iv = [1.0 / cov[i][i] for i in range(len(cov))]
    s = sum(iv)
    return [x / s for x in iv]


def corr_dist(cov):
    n = len(cov)
    sd = [math.sqrt(cov[i][i]) for i in range(n)]
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rho = 1.0 if i == j else min(1.0, max(-1.0, cov[i][j] / (sd[i] * sd[j])))
            d[i][j] = math.sqrt(max(0.0, (1.0 - rho) / 2.0))
    D = [[math.sqrt(sum((d[k][i] - d[k][j]) ** 2 for k in range(n))) for j in range(n)] for i in range(n)]
    return D


def linkage_single(D):
    """Merges (a, b, height, size) with a < b.  Rule: smallest linkage distance, ties by the smaller lower id then the smaller higher id; new ids n, n+1, ..."""
    n = len(D)
    clusters = {i: {i} for i in range(n)}
    nxt, out = n, []
    while len(clusters) > 1:
        best = None
        ids = sorted(clusters)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                h = min(D[i][j] for i in clusters[a] for j in clusters[b])
                if best is None or (h, a, b) < best:
                    best = (h, a, b)
        h, a, b = best
        clusters[nxt] = clusters.pop(a) | clusters.pop(b)
        out.append((a, b, h, len(clusters[nxt])))
        nxt += 1
    return out


def dendro_order(merges, n):
    if n == 1:
        return [0]
    kids = {n + i: (m[0], m[1]) for i, m in enumerate(merges)}

    def walk(c):
        return [c] if c < n else walk(kids[c][0]) + walk(kids[c][1])
    return walk(n + len(merges) - 1)


def sub_var(cov, items):
    iv = [1.0 / cov[i][i] for i in items]
    s = sum(iv)
    w = [x / s for x in iv]
    return sum(w[a] * w[b] * cov[items[a]][items[b]] for a in range(len(items)) for b in range(len(items)))


def hrp(cov):
    n = len(cov)
    if n == 1:
        return [1.0], [0], []
    mg = linkage_single(corr_dist(cov))
    order = dendro_order(mg, n)
    w = [1.0] * n
    stack = [order]
    while stack:
        cur = stack.pop()
        if len(cur) < 2:
            continue
        h = len(cur) // 2
        a, b = cur[:h], cur[h:]
        va, vb = sub_var(cov, a), sub_var(cov, b)
        al = 1.0 - va / (va + vb)
        for i in a:
            w[i] *= al
        for i in b:
            w[i] *= 1.0 - al
        stack += [a, b]
    s = sum(w)
    return [x / s for x in w], order, mg


def quad(cov, w):
    n = len(w)
    return sum(w[i] * cov[i][j] * w[j] for i in range(n) for j in range(n))


def kkt_gap(cov, w):
    n = len(w)
    g = [sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
    held = [i for i in range(n) if w[i] > 1e-9]
    lam = sum(w[i] * g[i] for i in held) / sum(w[i] for i in held)
    v = max([abs(g[i] - lam) for i in held] + [max(0.0, lam - g[i]) for i in range(n) if i not in held] + [0.0])
    return v / abs(mean(g))


def read_events(run):
    ev = V.read(run / "events.csv")
    cols = {"symbol": [r["symbol"] for r in ev]}
    for c in ("pos", "t1pos", "t0", "t1"):
        cols[c] = [int(float(r[c])) for r in ev]
    cols["side"] = [float(r["side"]) for r in ev]
    return cols


def year_of(day):
    return datetime.fromtimestamp(day * DAY, tz=timezone.utc).year


def main(folder: str, candle_override=None, runs_override=None) -> int:
    d = Path(folder)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    ex = json.loads((d / "hrp_extra.json").read_text(encoding="utf-8"))
    runs = [Path(r) for r in (runs_override or ex["source_runs"])]
    runs = [r if r.is_absolute() else ROOT / r for r in runs]
    summ = json.loads((runs[0] / "summary_numbers.json").read_text(encoding="utf-8"))
    p = summ["params"]
    cost_side, step = float(p["cost_frac"]) / 2.0, float(p["step"])
    cut = float(V.ts_key(p["cutoff"]))
    cutday = int(cut // DAY)
    w0 = cutday - EST_DAYS
    windows = {k: (int(v[0]), int(v[1])) for k, v in summ["meta"]["window_epoch"].items() if int(v[1]) > cut}
    cdir = Path(candle_override or ex["candle_dir"])
    if not cdir.is_absolute():
        cdir = ROOT / cdir
    E = read_events(runs[0])
    n_ev = len(E["symbol"])
    bad_md5 = [n for n, m in ex["script_md5"].items() if not (HERE / n).exists() or hashlib.md5((HERE / n).read_bytes()).hexdigest() != m]
    md5_same = all(hashlib.md5((r / "events.csv").read_bytes()).hexdigest() == ex["events_md5"] for r in runs)
    check("settings: 730-day window, 100 active days, cutoff, cost, bet-size step, the four schemes, the scored windows and the events md5 of every source run are the declared / recorded ones, and the four scripts are the recorded ones (md5)",
          ex["est_days"] == EST_DAYS and ex["min_active_days"] == MIN_ACTIVE and ex["cutoff"] == p["cutoff"] and close(ex["cost_frac"], float(p["cost_frac"]), 1e-12, 1e-12)
          and close(ex["step"], step, 1e-12, 1e-12) and ex["schemes"] == SCHEMES and {k: tuple(v) for k, v in ex["windows"].items()} == windows and ex["window_first_day"] == w0
          and ex["cutoff_day"] == cutday and md5_same and not bad_md5 and ex["n_events"] == n_ev, f"changed {bad_md5} md5_same {md5_same}")

    bars = {}
    for s in sorted(set(E["symbol"])):
        ts, cl = VB.read_candles(cdir, s)
        bars[s] = ([int(t) for t in ts], cl)
    # alignment of event bars with candles (entry / exit timestamps)
    mis = [i for i in range(n_ev) if bars[E["symbol"][i]][0][E["pos"][i]] != E["t0"][i] or bars[E["symbol"][i]][0][E["t1pos"][i]] != E["t1"][i]]
    check("every event's entry and exit bar timestamps equal the candle files' (so the candle folder is the run's)", not mis, str(mis[:3]))

    syms = sorted(bars)
    el = {r["symbol"]: r for r in V.read(d / "eligibility.csv")}
    est_cols, kept, bad_el = {}, [], []
    for s in syms:
        ts, cl = bars[s]
        first_day, last_day = ts[0] // DAY, ts[-1] // DAY
        pre = [i for i in range(n_ev) if E["symbol"][i] == s and E["t1"][i] < cut]
        n_post = sum(1 for i in range(n_ev) if E["symbol"][i] == s and E["t0"][i] >= cut)
        act_days, col = 0, [0.0] * EST_DAYS
        if pre:
            net, act, _ = brute_symbol(ts, cl, [E["pos"][i] for i in pre], [E["t1pos"][i] for i in pre], [E["side"][i] for i in pre], cost_side)
            act_days = sum(1 for x in act if w0 <= x < cutday)
            for x, v in net.items():
                if w0 <= x < cutday:
                    col[x - w0] = 1e4 * v
        ok = first_day <= w0 and last_day >= cutday - 1 and act_days >= MIN_ACTIVE and (sd1(col) > 0)
        r = el.get(s)
        if r is None or int(r["first_bar_day"]) != first_day or int(r["last_bar_day"]) != last_day or int(r["events_pre_cutoff"]) != len(pre) or int(r["events_post_cutoff"]) != n_post \
                or int(r["active_days_in_window"]) != act_days or int(r["eligible"]) != int(ok):
            bad_el.append(s)
        if ok:
            kept.append(s)
            est_cols[s] = col
    dropped = sum(1 for i in range(n_ev) if E["symbol"][i] not in est_cols and E["t0"][i] >= cut)
    check("eligibility.csv: first / last bar day, events before and after the cutoff, active days in the window and the eligible flag of every symbol, the eligible list and the dropped post-cutoff event count equal the recomputation",
          not bad_el and set(el) == set(syms) and ex["eligible"] == kept and ex["events_post_cutoff_dropped"] == dropped and ex["events_post_cutoff"] == sum(1 for i in range(n_ev) if E["t0"][i] >= cut), str(bad_el[:4]))
    n = len(kept)
    M = [est_cols[s] for s in kept]                                   # symbol-major, EST_DAYS values each
    mu = [mean(c) for c in M]
    cov = [[sum((M[i][t] - mu[i]) * (M[j][t] - mu[j]) for t in range(EST_DAYS)) / (EST_DAYS - 1) for j in range(n)] for i in range(n)]
    cf = V.read(d / "cov_bp2.csv")
    bad_cov = [(r["symbol"], kept[j]) for i, r in enumerate(cf) for j in range(n) if not close(V.num(r[kept[j]]), cov[i][j], 1e-7, 1e-7)]
    check("cov_bp2.csv: every cell equals the sample covariance (ddof 1) of the brute-force estimation matrix, symbols in alphabetical order", not bad_cov and [r["symbol"] for r in cf] == kept, str(bad_cov[:3]))

    wr = V.read(d / "weights.csv")
    W = {sch: [V.num(r[sch]) for r in wr] for sch in SCHEMES}
    w_hrp, order, mg = hrp(cov)
    exp_w = {"equal_weight": [1.0 / n] * n, "inverse_variance": inverse_var(cov), "hrp": w_hrp}
    bad_w = [(sch, kept[i], W[sch][i], exp_w[sch][i]) for sch in exp_w for i in range(n) if not close(W[sch][i], exp_w[sch][i], 1e-9, 1e-12)]
    check("weights.csv: equal-weight, inverse-variance and HRP weights equal the independent computation; every scheme is long-only with unit sum", not bad_w and [r["symbol"] for r in wr] == kept
          and all(abs(sum(W[s]) - 1.0) < 1e-9 and min(W[s]) >= -1e-12 for s in SCHEMES), str(bad_w[:3]))
    lk = V.read(d / "linkage.csv")
    ho = V.read(d / "hrp_order.csv")
    check("linkage.csv (children, heights, sizes) and hrp_order.csv equal the independent single-linkage and leaf order", len(lk) == len(mg) and all(int(r["left"]) == m[0] and int(r["right"]) == m[1] and close(V.num(r["height"]), m[2], 1e-9, 1e-12) and int(r["size"]) == m[3]
                                                                                                           for r, m in zip(lk, mg)) and [r["symbol"] for r in ho] == [kept[i] for i in order], str(order))
    wm = W["min_variance"]
    gap = kkt_gap(cov, wm)
    qv = quad(cov, wm)
    check("min-variance weights: long-only, unit sum, first-order optimality conditions hold (relative gap below 1e-4) and the variance is not above that of equal weight, inverse variance and HRP",
          min(wm) >= -1e-12 and abs(sum(wm) - 1) < 1e-9 and gap < 1e-4 and all(qv <= quad(cov, exp_w[k]) * (1 + 1e-9) for k in exp_w), f"gap {gap:.3g} var {qv:.6g}")

    # ---- evaluation streams
    d_hi = max(bars[s][0][-1] // DAY for s in kept)
    post_all = [i for i in range(n_ev) if E["t0"][i] >= cut]
    post_el = [i for i in post_all if E["symbol"][i] in est_cols]
    streams = [("primary", "primary", "", {i: 1.0 for i in post_el})]
    bad_cover = []
    for r in runs:
        rs = json.loads((r / "summary_numbers.json").read_text(encoding="utf-8"))
        wfp = V.read(r / "walk_forward_predictions.csv")
        if sorted(int(float(x["event"])) for x in wfp) != post_all:
            bad_cover.append(str(r))
        sz = {int(float(x["event"])): VB.bet_size(float(x["p"]), step) for x in wfp}
        streams.append((f"meta_{int(rs['params']['seed'])}", "meta", int(rs["params"]["seed"]), {i: sz[i] for i in post_el}))
    check("every source run's walk-forward predictions cover exactly the events starting on or after the cutoff", not bad_cover, str(bad_cover))

    mt = V.read(d / "metrics.csv")
    mtx = {(r["stream"], r["scheme"]): r for r in mt}
    pdaily = defaultdict(dict)
    for r in V.read(d / "portfolio_daily.csv"):
        pdaily[(r["stream"], r["scheme"])][int(r["day"])] = float(r["bp"])
    bad_m, bad_d, series = [], [], {}
    for sname, kind, seed, size in streams:
        per_net, per_pres = {}, {}
        for s in kept:
            ts, cl = bars[s]
            ids = [i for i in size if E["symbol"][i] == s]
            net, _, pres = brute_symbol(ts, cl, [E["pos"][i] for i in ids], [E["t1pos"][i] for i in ids], [E["side"][i] * size[i] for i in ids], cost_side)
            per_net[s], per_pres[s] = net, pres
        days_all = [x for x in range(cutday, d_hi + 1) if any(x in per_pres[s] for s in kept)]
        for si, sch in enumerate(SCHEMES):
            ser = []
            for x in days_all:
                num = sum(W[sch][j] * per_net[s].get(x, 0.0) for j, s in enumerate(kept) if x in per_pres[s])
                den = sum(W[sch][j] for j, s in enumerate(kept) if x in per_pres[s])
                ser.append(1e4 * num / den if den > 0 else 0.0)
            series[(sname, sch)] = (days_all, ser)
            got = pdaily.get((sname, sch), {})
            if sorted(got) != days_all or any(not close(got[x], v, 1e-7, 1e-7) for x, v in zip(days_all, ser)):
                bad_d.append((sname, sch))
            exp = {"n_days": len(ser), "mean_bp": mean(ser), "sd_bp": sd1(ser), "sr_annual": sharpe_annual(ser)}
            for wn, (lo, hi) in windows.items():
                sel = [v for x, v in zip(days_all, ser) if lo <= x * DAY < hi]
                exp[f"win_{wn}_bp"], exp[f"win_{wn}_days"] = (mean(sel) if sel else NAN), len(sel)
            for y in sorted({year_of(x) for x in days_all}):
                sel = [v for x, v in zip(days_all, ser) if year_of(x) == y]
                exp[f"year_{y}_days"], exp[f"year_{y}_mean_bp"], exp[f"year_{y}_sr_annual"] = len(sel), mean(sel), sharpe_annual(sel)
            r = mtx.get((sname, sch))
            if r is None:
                bad_m.append((sname, sch, "row missing"))
                continue
            for k, v in exp.items():
                if k not in r or not close(V.num(r[k]), float(v), 1e-7, 1e-7):
                    bad_m.append((sname, sch, k, r.get(k), v))
    check("portfolio_daily.csv: every day and value of every (stream, scheme) series equals the brute-force portfolio (weights rescaled over the symbols with a bar that day)", not bad_d and set(pdaily) == set(series), str(bad_d[:3]))
    check("metrics.csv: days, mean, sd, annualised Sharpe, window means and days and calendar-year days, means and Sharpes of every (stream, scheme) equal the recomputation; no extra rows",
          not bad_m and len(mt) == len(streams) * len(SCHEMES), str(bad_m[:3]))

    ss = {r["scheme"]: r for r in V.read(d / "scheme_stats.csv")}
    bad_s = []
    for si, sch in enumerate(SCHEMES):
        w = W[sch]
        isr = [sum(w[i] * M[i][t] for i in range(n)) for t in range(EST_DAYS)]
        exp = {"effective_n": 1.0 / sum(x * x for x in w), "max_weight": max(w), "n_zero": sum(1 for x in w if x == 0), "in_sample_sd_bp": sd1(isr), "oos_sd_bp_primary": float(mtx[("primary", sch)]["sd_bp"])}
        exp["oos_over_is"] = exp["oos_sd_bp_primary"] / exp["in_sample_sd_bp"]
        for k, v in exp.items():
            if not close(V.num(ss[sch][k]), float(v), 1e-7, 1e-9):
                bad_s.append((sch, k, ss[sch][k], v))
    check("scheme_stats.csv: effective number of names, largest weight, zero weights, in-sample sd (weights on the estimation matrix), out-of-sample sd and their ratio equal the recomputation", not bad_s and set(ss) == set(SCHEMES), str(bad_s[:3]))

    md = (d / "hrp_summary.md").read_text(encoding="utf-8")
    bad_t = []
    for i, s in enumerate(kept):
        m = re.search(rf"^\| {s} \| (\S+) \| (\S+) \| (\S+) \| (\S+) \|$", md, re.M)
        if not m or any(not close(float(m.group(k + 1)), W[SCHEMES[k]][i], 0, 6e-5) for k in range(4)):
            bad_t.append(("weights", s))
    for sch in SCHEMES:
        m = re.search(rf"^\| {sch} \| (\S+) \| (\S+) \| (\d+) \| (\S+) \| (\S+) \| (\S+) \|$", md, re.M)
        r = ss[sch]
        if not m or not close(float(m.group(1)), V.num(r["effective_n"]), 0, 6e-3) or not close(float(m.group(2)), V.num(r["max_weight"]), 0, 6e-4) or int(m.group(3)) != int(float(r["n_zero"])) \
                or not close(float(m.group(4)), V.num(r["in_sample_sd_bp"]), 0, 6e-3) or not close(float(m.group(5)), V.num(r["oos_sd_bp_primary"]), 0, 6e-3) or not close(float(m.group(6)), V.num(r["oos_over_is"]), 0, 6e-3):
            bad_t.append(("structure", sch))
    for s_ in syms:
        r = el[s_]
        m = re.search(rf"^\| {s_} \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (yes|no) \| (.*) \|$", md, re.M)
        if not m or [int(m.group(k)) for k in range(1, 6)] != [int(r["first_bar_day"]), int(r["last_bar_day"]), int(r["events_pre_cutoff"]), int(r["events_post_cutoff"]), int(r["active_days_in_window"])] \
                or (m.group(6) == "yes") != (r["eligible"] == "1") or m.group(7).strip() != (r["reason"] or "").strip():
            bad_t.append(("eligibility table", s_))
    m = re.search(r"Eligible symbols \((\d+)\): (.*?)\. Post-cutoff events of ineligible symbols are dropped: (\d+) of (\d+)\.", md)
    if not m or int(m.group(1)) != len(kept) or m.group(2) != ", ".join(kept) or int(m.group(3)) != dropped or int(m.group(4)) != ex["events_post_cutoff"]:
        bad_t.append(("eligible sentence",))
    m = re.search(r"HRP leaf order: (.*?)\.\n", md)
    if not m or m.group(1) != ", ".join(kept[i] for i in order):
        bad_t.append(("leaf order sentence",))
    m = re.search(r"the (\d+) UTC days before (\S+) \(primary stream.*?at least (\d+) active days", md, re.S)
    if not m or int(m.group(1)) != EST_DAYS or m.group(2) != p["cutoff"] or int(m.group(3)) != MIN_ACTIVE:
        bad_t.append(("intro sentence",))
    wn = list(windows)
    for sch in SCHEMES:
        r = mtx[("primary", sch)]
        m = re.search(rf"^\| {sch} \| (\d+) \| (\S+) \| (\S+) \| (\S+)" + r" \| (\S+)" * len(wn) + r" \|$", md, re.M)
        want = [V.num(r["mean_bp"]), V.num(r["sd_bp"]), V.num(r["sr_annual"])] + [V.num(r[f"win_{w}_bp"]) for w in wn]
        got = [m.group(k) for k in range(2, 5 + len(wn))] if m else None
        if not m or int(m.group(1)) != int(float(r["n_days"])) or any((got[k] == "n/a") != math.isnan(want[k]) or (got[k] != "n/a" and not close(float(got[k]), want[k], 0, 6e-3)) for k in range(len(want))):
            bad_t.append(("primary", sch))
        ys = sorted({int(c.split("_")[1]) for c in r if c.startswith("year_") and c.endswith("_mean_bp")})
        m = re.search(rf"^\| {sch} \|" + r" (\S+) \((\S+)\) \|" * len(ys) + "$", md, re.M)
        if not m:
            bad_t.append(("years", sch))
        else:
            for k, y in enumerate(ys):
                a, b = m.group(2 * k + 1), m.group(2 * k + 2)
                if not close(float(a), V.num(r[f"year_{y}_mean_bp"]), 0, 6e-3) or (b == "n/a") != math.isnan(V.num(r[f"year_{y}_sr_annual"])) or (b != "n/a" and not close(float(b), V.num(r[f"year_{y}_sr_annual"]), 0, 6e-3)):
                    bad_t.append(("years", sch, y))
    metas = [s for s in streams if s[1] == "meta"]
    cols = ["mean_bp", "sd_bp", "sr_annual"] + [f"win_{w}_bp" for w in wn]
    for sch in SCHEMES:
        m = re.search(rf"^\| {sch} \|" + r" (\S+) \((\S+)\) \|" * len(cols) + "$", md, re.M)
        if not m:
            bad_t.append(("meta", sch, "row"))
            continue
        for k, c in enumerate(cols):
            vals = [V.num(mtx[(s[0], sch)][c]) for s in metas]
            vals = [x for x in vals if not math.isnan(x)]
            a, b = m.group(2 * k + 1), m.group(2 * k + 2)
            if not vals:
                if a != "n/a":
                    bad_t.append(("meta", sch, c))
                continue
            if not close(float(a), mean(vals), 0, 6e-3) or (len(vals) > 1 and not close(float(b), sd1(vals), 0, 6e-3)) or (len(vals) < 2 and b != "n/a"):
                bad_t.append(("meta", sch, c))
    for sch in SCHEMES[1:]:
        m = re.search(rf"^\| {sch} \| (\S+) \| (\d+) of (\d+) \| (\S+) \| (\d+) of (\d+) \|" + r" (\S+) \((\d+) of (\d+)\) \|" * len(wn) + "$", md, re.M)
        if not m:
            bad_t.append(("diff", sch, "row"))
            continue
        for k, c in enumerate(["sr_annual", "mean_bp"] + [f"win_{w}_bp" for w in wn]):
            dl = [V.num(mtx[(s[0], sch)][c]) - V.num(mtx[(s[0], "equal_weight")][c]) for s in metas]
            dl = [x for x in dl if not math.isnan(x)]
            g = m.groups()
            cell = (g[0], g[1], g[2]) if k == 0 else (g[3], g[4], g[5]) if k == 1 else g[6 + 3 * (k - 2): 9 + 3 * (k - 2)]
            if not dl:
                if cell[0] != "n/a" or int(cell[1]) != 0:
                    bad_t.append(("diff", sch, c))
            elif not close(float(cell[0]), mean(dl), 0, 6e-4) or int(cell[1]) != sum(1 for x in dl if x > 0) or int(cell[2]) != len(dl):
                bad_t.append(("diff", sch, c))
    check("hrp_summary.md: every cell of the eligibility, weights, weight-structure, primary, calendar-year, meta (mean and sd over seeds) and scheme-minus-equal-weight tables equals the recomputation", not bad_t, str(bad_t[:3]))

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
    n_ok = sum(1 for c in checks if c[1])
    print(f"\n{n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    cd = runs = None
    if "--candle-dir" in argv:
        i = argv.index("--candle-dir")
        cd = argv[i + 1]
        del argv[i:i + 2]
    if "--runs" in argv:
        i = argv.index("--runs")
        runs = argv[i + 1:]
        argv = argv[:i]
    try:
        raise SystemExit(main(argv[0], cd, runs))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, FileNotFoundError, AttributeError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
