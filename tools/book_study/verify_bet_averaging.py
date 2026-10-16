#!/usr/bin/env python
"""Independent checker for a run_bet_averaging.py output folder (stdlib only; imports neither bet_averaging.py, run_bet_averaging.py, numpy nor pandas).

From the source run's files (events.csv, cpcv_predictions.csv, walk_forward_predictions.csv, paths.csv, summary_numbers.json) and the raw candle CSVs it re-derives:
  * the settings (seed, bet-size step, cost) and the md5 of the two scripts recorded in averaged_extra.json;
  * the bar alignment of every event (entry and exit timestamps and the side-signed return against the candles);
  * the meta bet size of every prediction from its probability (own formula) and that each path's placebo sizes are a permutation of its meta sizes;
  * that the per-event convention computed from those sizes reproduces the mean bp per candidate event of every path and variant in the source run's paths.csv
    (this ties the placebo sizes in averaged_sizes.csv to the run's own placebo);
  * the averaged result of every CPCV path and of the walk-forward fit for primary, meta and placebo by BRUTE FORCE: for every instrument and bar a loop over the events active
    there (no cumulative-sum trick), then gross, cost, daily booking, active instrument-days, daily bp, Sharpe, window means, compared with averaged_paths.csv, averaged_daily.csv
    (every row) and averaged_windows.csv;
  * every cell of averaged_summary.json and of the tables of averaged_summary.md.
Exit code 0 = all checks pass.

    python tools/book_study/verify_bet_averaging.py runtime/book_bet_avg/<name> [--run <source run folder>] [--candle-dir <folder>]
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import verify_book_study as V

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAN = math.nan
VARIANTS = ("primary", "meta", "placebo")
DAY = 86400


def close_to(a, b, rel=1e-7, ab=1e-9):
    return V.same(a, b, rel, ab)


def bet_size(p, step):
    p = min(max(p, 1e-9), 1 - 1e-9)
    z = (p - 0.5) / math.sqrt(p * (1 - p))
    s = (2.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) - 1.0) if p > 0.5 else 0.0
    if step > 0:
        s = round(s / step) * step
    return min(max(s, 0.0), 1.0)


def read_candles(candle_dir, symbol):
    """(epoch seconds list, closes list) with the loader's rules: close numeric and > 0; the last row wins for a repeated timestamp; sorted by time."""
    rows = {}
    for r in V.read(Path(candle_dir) / f"{symbol}_1h.csv"):
        c, t = V.num(r.get("close")), r.get("date")
        if t is not None and not math.isnan(c) and c > 0:
            rows[V.ts_key(t)] = c
    keys = sorted(rows)
    return keys, [rows[k] for k in keys]


def mean(v):
    return sum(v) / len(v) if v else NAN


def sample_sd(v):
    if len(v) < 2:
        return NAN
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def brute_force(events, bars, idx, sizes, cost_side, windows):
    """events: dict of lists; bars: {symbol: (ts, close)}; idx: event indices; sizes: sizes per idx (same order).  Returns dict of results."""
    by_sym = defaultdict(list)
    for j, e in enumerate(idx):
        by_sym[events["symbol"][e]].append(j)
    net, gross_d, cost_d = defaultdict(float), defaultdict(float), defaultdict(float)
    act = defaultdict(set)                      # day -> set of symbols
    for sym, js in by_sym.items():
        ts, cl = bars[sym]
        n = len(cl)
        S, C = [0.0] * n, [0] * n
        for j in js:
            e = idx[j]
            sig = events["side"][e] * sizes[j]
            for k in range(events["pos"][e], events["t1pos"][e]):
                S[k] += sig
                C[k] += 1
        prev = 0.0
        for k in range(n):
            a = S[k] / C[k] if C[k] > 0 else 0.0
            if k <= n - 2:
                g = a * (cl[k + 1] / cl[k] - 1.0)
                d1 = int(ts[k + 1] // DAY)
                gross_d[d1] += g
                net[d1] += g
                if C[k] > 0:
                    act[int(ts[k] // DAY)].add(sym)
                    act[d1].add(sym)
            c = cost_side * abs(a - prev)
            dk = int(ts[k] // DAY)
            cost_d[dk] += c
            net[dk] -= c
            prev = a
    days = sorted(d for d in act if len(act[d]) > 0)
    nact = {d: len(act[d]) for d in days}
    # days with activity only: net of such days (cost booked on a day without an active instrument cannot occur: see the rule), but keep totals over the active days
    bp = [1e4 * net[d] / nact[d] for d in days]
    tot_n = sum(nact.values())
    res = {"days": days, "bp": bp, "n_active": [nact[d] for d in days], "symbol_days": tot_n,
           "mean_bp": 1e4 * sum(net[d] for d in days) / tot_n, "gross_bp": 1e4 * sum(gross_d[d] for d in days) / tot_n, "cost_bp": 1e4 * sum(cost_d[d] for d in days) / tot_n,
           "net_by_day": {d: net[d] for d in days}}
    sr = NAN
    if len(bp) >= 4:
        m, sd = mean(bp), sample_sd(bp)
        sr = m / sd if sd > 0 else NAN
    res["sr_daily"], res["sr_annual"] = sr, (sr * math.sqrt(365) if not math.isnan(sr) else NAN)
    res["windows"] = {}
    for wn, (lo, hi) in windows.items():
        ds = [d for d in days if lo <= d * DAY < hi]
        n_ = sum(nact[d] for d in ds)
        res["windows"][wn] = (1e4 * sum(net[d] for d in ds) / n_ if n_ else NAN, n_)
    return res


def main(folder: str, run_override=None, candle_override=None) -> int:
    d = Path(folder)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    def report():
        for name, ok, detail in checks:
            print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
        n_ok = sum(1 for c in checks if c[1])
        print(f"\n{n_ok}/{len(checks)} checks passed")
        return 0 if n_ok == len(checks) else 1

    extra = json.loads((d / "averaged_extra.json").read_text(encoding="utf-8"))
    run = Path(run_override or extra["source_run"])
    if not run.is_absolute():
        run = ROOT / run
    summ = json.loads((run / "summary_numbers.json").read_text(encoding="utf-8"))
    params = summ["params"]
    step, seed, cost_frac = float(params["step"]), int(params["seed"]), float(params["cost_frac"])
    cost_side = cost_frac / 2.0
    candle_dir = Path(candle_override or extra["candle_dir"])
    if not candle_dir.is_absolute():
        candle_dir = ROOT / candle_dir

    ev_rows = V.read(run / "events.csv")
    events = {"symbol": [r["symbol"] for r in ev_rows], "pos": [int(float(r["pos"])) for r in ev_rows], "t1pos": [int(float(r["t1pos"])) for r in ev_rows],
              "t0": [int(float(r["t0"])) for r in ev_rows], "t1": [int(float(r["t1"])) for r in ev_rows], "side": [float(r["side"]) for r in ev_rows],
              "ret": [float(r["ret_side"]) for r in ev_rows]}
    n_ev = len(ev_rows)

    bad_md5 = [n for n, m in extra["script_md5"].items() if not (HERE / n).exists() or hashlib.md5((HERE / n).read_bytes()).hexdigest() != m]
    check("averaged_extra.json: seed, bet-size step, cost and event count equal the source run's, and the two scripts next to this checker are the ones recorded (md5)",
          extra["seed"] == seed and close_to(extra["step"], step, 1e-12, 1e-12) and close_to(extra["cost_frac"], cost_frac, 1e-12, 1e-12) and extra["events"] == n_ev
          and sorted(extra["script_md5"]) == ["bet_averaging.py", "run_bet_averaging.py"] and not bad_md5, f"changed or missing: {bad_md5}")

    bars = {s: read_candles(candle_dir, s) for s in sorted(set(events["symbol"]))}
    bad_al = []
    for i in range(n_ev):
        ts, cl = bars[events["symbol"][i]]
        p, q = events["pos"][i], events["t1pos"][i]
        r = events["side"][i] * (cl[q] / cl[p] - 1.0)
        if not (0 <= p < q < len(cl) and int(ts[p]) == events["t0"][i] and int(ts[q]) == events["t1"][i] and abs(r - events["ret"][i]) <= 1e-6):
            bad_al.append((i, events["symbol"][i]))
            if len(bad_al) > 5:
                break
    check("every event's entry / exit bar timestamps and side-signed return equal what the candle files give (bar indices line up)", not bad_al, str(bad_al[:3]))

    cp_rows, wf_rows = V.read(run / "cpcv_predictions.csv"), V.read(run / "walk_forward_predictions.csv")
    sz_rows = V.read(d / "averaged_sizes.csv")
    sz = defaultdict(list)                       # (scope, path) -> [(event, meta, placebo)]
    for r in sz_rows:
        sz[(r["scope"], int(r["path"]))].append((int(r["event"]), float(r["size_meta"]), float(r["size_placebo"])))
    pred = defaultdict(list)
    for scope, rows in (("cpcv", cp_rows), ("wf", wf_rows)):
        for r in rows:
            pred[(scope, int(r["path"]))].append((int(r["event"]), float(r["p"])))
    check("averaged_sizes.csv covers exactly the predictions of every CPCV path and of the walk-forward fit, event by event in the same order",
          set(sz) == set(pred) and all([e for e, _, _ in sz[k]] == [e for e, _ in pred[k]] for k in pred), f"{sorted(set(sz) ^ set(pred))[:5]}")
    bad_meta, bad_perm = [], []
    for k in pred:
        for (e, p), (e2, m, pl) in zip(pred[k], sz[k]):
            if not close_to(bet_size(p, step), m, 1e-12, 1e-12):
                bad_meta.append((k, e))
        if sorted(round(m, 12) for _, m, _ in sz[k]) != sorted(round(pl, 12) for _, _, pl in sz[k]):
            bad_perm.append(k)
    check("meta bet sizes recomputed from the probabilities (own formula and rounding to the step) equal averaged_sizes.csv", not bad_meta, str(bad_meta[:3]))
    check("in every path the placebo sizes are a permutation of the meta sizes", not bad_perm, str(bad_perm[:3]))

    paths_run = defaultdict(dict)
    for r in V.read(run / "paths.csv"):
        paths_run[int(r["path"])][r["variant"]] = float(r["mean_bp_per_candidate"])
    bad_ev = []
    for (scope, path), lst in sz.items():
        if scope != "cpcv":
            continue
        for v, col in (("primary", None), ("meta", 1), ("placebo", 2)):
            vals = [(1.0 if col is None else t[col]) for t in lst]
            got = mean([s * (events["ret"][t[0]] - cost_frac * (s > 0)) * 1e4 for s, t in zip(vals, lst)])
            if not close_to(got, paths_run[path][v], 1e-9, 1e-9):
                bad_ev.append((path, v, got, paths_run[path][v]))
    check("the per-event convention computed from these sizes reproduces the source run's mean bp per candidate event for every CPCV path and variant (primary, meta, placebo)", not bad_ev, str(bad_ev[:3]))

    windows = {k: (int(v[0]), int(v[1])) for k, v in summ["meta"]["window_epoch"].items()}
    res = {}
    for (scope, path), lst in sorted(sz.items()):
        idx = [t[0] for t in lst]
        for v, col in (("primary", None), ("meta", 1), ("placebo", 2)):
            res[(scope, path, v)] = brute_force(events, bars, idx, [1.0 if col is None else t[col] for t in lst], cost_side, windows)
    # event-convention cost per symbol-day
    ev_cost = {}
    for (scope, path, v), r in res.items():
        col = None if v == "primary" else (1 if v == "meta" else 2)
        ev_cost[(scope, path, v)] = 1e4 * cost_frac * sum((1.0 if col is None else t[col]) for t in sz[(scope, path)]) / r["symbol_days"]

    pr = {(r["scope"], int(r["path"]), r["variant"]): r for r in V.read(d / "averaged_paths.csv")}
    bad_p = []
    for k, r in res.items():
        g = pr.get(k)
        if g is None:
            bad_p.append((k, "missing"))
            continue
        pairs = [("days", len(r["days"])), ("symbol_days", r["symbol_days"]), ("mean_bp_per_symbol_day", r["mean_bp"]), ("gross_bp_per_symbol_day", r["gross_bp"]),
                 ("cost_bp_per_symbol_day", r["cost_bp"]), ("event_convention_cost_bp_per_symbol_day", ev_cost[k]), ("sr_daily", r["sr_daily"]), ("sr_annual", r["sr_annual"])]
        for col, exp in pairs:
            if not close_to(V.num(g[col]), float(exp) if not isinstance(exp, float) or not math.isnan(exp) else NAN, 1e-7, 1e-9):
                bad_p.append((k, col, g[col], exp))
    check("averaged_paths.csv: for every scope, path and variant the days, instrument-days, mean / gross / cost bp, the cost under the per-event rule and the Sharpe ratios equal the brute-force recomputation", not bad_p and set(pr) == set(res), str(bad_p[:3]))

    dr = defaultdict(dict)
    for r in V.read(d / "averaged_daily.csv"):
        dr[(r["scope"], int(r["path"]), r["variant"])][int(r["day"])] = (float(r["bp"]), int(r["n_active"]))
    bad_d = []
    for k, r in res.items():
        got = dr.get(k, {})
        if sorted(got) != r["days"]:
            bad_d.append((k, "days"))
            continue
        for dd, bp, na in zip(r["days"], r["bp"], r["n_active"]):
            if got[dd][1] != na or not close_to(got[dd][0], bp, 1e-7, 1e-7):
                bad_d.append((k, dd, got[dd], (bp, na)))
                break
    check("averaged_daily.csv: every day of every scope / path / variant (day set, bp, number of active instruments) equals the brute-force recomputation", not bad_d and set(dr) == set(res), str(bad_d[:3]))

    wr = {(r["scope"], int(r["path"]), r["variant"], r["window"]): r for r in V.read(d / "averaged_windows.csv")}
    bad_w = []
    for (scope, path, v), r in res.items():
        for wn, (bp, n_) in r["windows"].items():
            g = wr.get((scope, path, v, wn))
            if g is None or int(g["symbol_days"]) != n_ or not close_to(V.num(g["bp_per_symbol_day"]), bp, 1e-7, 1e-7):
                bad_w.append((scope, path, v, wn))
    check("averaged_windows.csv: June 2026, sell-off and deepest-fall window means (bp per active instrument-day) and instrument-day counts equal the brute-force recomputation",
          not bad_w and len(wr) == len(res) * len(windows), str(bad_w[:3]))

    # summary json and md
    S = json.loads((d / "averaged_summary.json").read_text(encoding="utf-8"))
    exp = {}
    for scope in ("cpcv", "wf"):
        plist = sorted({p for (s_, p, _) in res if s_ == scope})
        e = {}
        for v in VARIANTS:
            rs = [res[(scope, p, v)] for p in plist]
            e[v] = {"mean_bp_per_symbol_day": mean([r["mean_bp"] for r in rs]), "gross_bp_per_symbol_day": mean([r["gross_bp"] for r in rs]), "cost_bp_per_symbol_day": mean([r["cost_bp"] for r in rs]),
                    "event_convention_cost_bp_per_symbol_day": mean([ev_cost[(scope, p, v)] for p in plist]), "sr_daily": mean([r["sr_daily"] for r in rs]),
                    "sr_annual": mean([r["sr_annual"] for r in rs]), "paths": len(plist), "paths_sr_negative": sum(1 for r in rs if r["sr_daily"] < 0)}
        for name, other in (("meta_minus_placebo", "placebo"), ("meta_minus_primary", "primary")):
            sd_ = [res[(scope, p, "meta")]["sr_daily"] - res[(scope, p, other)]["sr_daily"] for p in plist]
            e[name] = {"paths_sr_higher": sum(1 for x in sd_ if x > 0), "paths": len(plist), "mean_sr_daily_diff": mean(sd_)}
            if other == "placebo":
                e[name]["mean_bp_diff"] = mean([res[(scope, p, "meta")]["mean_bp"] - res[(scope, p, "placebo")]["mean_bp"] for p in plist])
        w = {}
        for v in VARIANTS:
            for wn in windows:
                vals = [res[(scope, p, v)]["windows"][wn] for p in plist]
                good = [x[0] for x in vals if not math.isnan(x[0])]
                w[f"{v}|{wn}"] = {"mean_bp_per_symbol_day": mean(good) if len(good) == len(vals) else NAN, "symbol_days_per_path": mean([x[1] for x in vals])}
        e["windows"] = w
        exp[scope] = e
    bad_s = []

    def cmp(a, b, path):
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                bad_s.append((path, "keys"))
                return
            for k in b:
                cmp(a[k], b[k], path + "/" + k)
        else:
            av = NAN if a is None else float(a)
            if not close_to(av, float(b), 1e-7, 1e-9):
                bad_s.append((path, a, b))

    cmp(S, exp, "")
    check("averaged_summary.json: every cell (means over paths, paths with negative Sharpe, meta minus placebo / primary, window means) equals the recomputation from the brute-force results", not bad_s, str(bad_s[:3]))

    md = (d / "averaged_summary.md").read_text(encoding="utf-8")
    secs = md.split("\n## ")
    bad_m = []

    def f2(x):
        return NAN if x == "n/a" else float(x)

    for sec, scope in ((secs[1], "cpcv"), (secs[2], "wf")):
        for m in re.finditer(r"^\| (primary|meta|placebo) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\d+)/(\d+) \|$", sec, re.M):
            v, vals = m.group(1), [f2(m.group(i)) for i in range(2, 7)]
            e = exp[scope][v]
            ex = [e["mean_bp_per_symbol_day"], e["gross_bp_per_symbol_day"], e["cost_bp_per_symbol_day"], e["event_convention_cost_bp_per_symbol_day"], e["sr_annual"]]
            if not all(V.same(a, b, 0.0, 0.0051) for a, b in zip(vals, ex)) or int(m.group(7)) != e["paths_sr_negative"] or int(m.group(8)) != e["paths"]:
                bad_m.append((scope, v))
        n_rows = len(re.findall(r"^\| (primary|meta|placebo) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\d+)/(\d+) \|$", sec, re.M))
        if n_rows != 3:
            bad_m.append((scope, "variant rows", n_rows))
        for wn in windows:
            m = re.search(rf"^\| {wn} \| (\S+) \| (\S+) \| (\S+) \| (\S+) \|$", sec, re.M)
            if not m:
                bad_m.append((scope, wn, "row missing"))
                continue
            ex = [exp[scope]["windows"][f"{v}|{wn}"]["mean_bp_per_symbol_day"] for v in VARIANTS] + [exp[scope]["windows"][f"primary|{wn}"]["symbol_days_per_path"]]
            if not all(V.same(f2(m.group(i + 1)), x, 0.0, 0.0051 if i < 3 else 0.051) for i, x in enumerate(ex)):
                bad_m.append((scope, wn))
        e1, e2 = exp[scope]["meta_minus_placebo"], exp[scope]["meta_minus_primary"]
        m = re.search(r"Meta minus placebo: Sharpe higher on (\d+)/(\d+) paths, mean daily-Sharpe difference (\S+), mean bp difference (\S+)\. Meta minus primary: Sharpe higher on (\d+)/(\d+) paths, mean daily-Sharpe difference (\S+)\.", sec)
        if not m or (int(m.group(1)), int(m.group(2)), int(m.group(5)), int(m.group(6))) != (e1["paths_sr_higher"], e1["paths"], e2["paths_sr_higher"], e2["paths"]) \
                or not V.same(float(m.group(3)), e1["mean_sr_daily_diff"], 0.0, 0.00006) or not V.same(float(m.group(4)), e1["mean_bp_diff"], 0.0, 0.0051) or not V.same(float(m.group(7)), e2["mean_sr_daily_diff"], 0.0, 0.00006):
            bad_m.append((scope, "meta minus lines"))
    ev_ = summ["eval"]["summary"]
    for v in VARIANTS:
        m = re.search(rf"^\| {v} \| (\S+) \| (\S+) \|$", secs[3], re.M)
        if not m or not V.same(float(m.group(1)), ev_[v]["mean_bp_per_candidate"], 0.0, 0.0051) or not V.same(float(m.group(2)), ev_[v]["mean_sr_annual"], 0.0, 0.0051):
            bad_m.append(("comparison table", v))
    check("averaged_summary.md: every number of the averaged tables (both scopes), the window tables, the meta-minus lines and the per-event comparison table equals the recomputation / the source run (printed rounding)", not bad_m, str(bad_m[:3]))
    return report()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    ro = co = None
    for flag in ("--run", "--candle-dir"):
        if flag in argv:
            i_ = argv.index(flag)
            if flag == "--run":
                ro = argv[i_ + 1]
            else:
                co = argv[i_ + 1]
            del argv[i_:i_ + 2]
    try:
        raise SystemExit(main(argv[0], ro, co))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, FileNotFoundError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
