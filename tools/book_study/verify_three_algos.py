#!/usr/bin/env python
"""Independent checker for a run_three_algos.py output folder (stdlib only; imports helper functions of verify_book_study.py and
verify_meta_pipeline.py).

Inputs it trusts: the SOURCE meta-pipeline run named in the folder (its events.csv, symbols.csv, summary_numbers.json); that run is verified
separately by verify_meta_pipeline.py.  This checker confirms the source files are the ones the run used (md5), and then:
  * rebuilds from the raw candle files (each symbol cut at the source run's last bar) the three trailing returns of every event (20, 30, 40 days),
    the MOM side, the long return and the direction label, and compares them with events_ext.csv;
  * re-derives the CPCV group edges, purge / embargo, train and test counts of the direction forest AND the MOM forest for every split,
    the average training uniqueness and draws per tree, path membership of every prediction, per-split AUC / base rate / share above 1/2;
  * recomputes, with its own code, every variant (long_all, short_all, mom_all, mom_long_up, dir_model, mom_meta, gate) for every path: sides, bet
    sizes, daily series, Sharpe / skew / kurtosis / PSR, best-20-days share, the summary table, the comparisons, the deflated-Sharpe tables, the
    window lines (from events and from the daily rows), the by-state tables and the by-algorithm tables of the gate and of dir_model;
  * checks the two nulls only for what does not need numpy's random generator (same days and event counts as the other variants, the gate's
    sizes or sides preserved, statistics consistent with their own daily rows);
  * repeats the above for the walk-forward fit.
Not covered: the forests' internals (unit tests), and the nulls' random draws.
Exit code 0 = all checks pass.

    python tools/book_study/verify_three_algos.py runtime/book_three_algo/<timestamp> [source meta run folder]
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

import verify_book_study as V
import verify_meta_pipeline as VM

HERE = Path(__file__).resolve().parent
NAN = math.nan
fin, close, mean, bet_size, auc = VM.fin, VM.close, VM.mean, VM.bet_size, VM.auc
DECLARED_VARIANTS = ["long_all", "short_all", "mom_all", "mom_long_up", "dir_model", "mom_meta", "gate", "gate_shuffled_size", "gate_random_side"]
DECLARED_DSR = ["mom_long_up", "dir_model", "mom_meta", "gate"]
REAL = ["long_all", "short_all", "mom_all", "mom_long_up", "dir_model", "mom_meta", "gate"]
NULLS = ["gate_shuffled_size", "gate_random_side"]
DECLARED_DIR = ["ffd", "mom24n", "mom168n", "mom720n", "mom2160n", "log_trgt", "vol_ratio", "mkt_state", "mkt_dist", "sadf", "ent_plugin", "ent_konto", "cs_spread"]
DECLARED_MOM = ["side"] + DECLARED_DIR
LOOKBACKS = (480, 720, 960)
ALGOS = ("LONG", "SHORT", "MOM")
ALGO_CODE = {"LONG": 1, "SHORT": 2, "MOM": 3}


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
    src = Path(source_override) if source_override else Path(meta["source_folder"])
    ext = V.read(d / "events_ext.csv")
    split_rows = V.read(d / "split_metrics.csv")
    pred_rows = V.read(d / "cpcv_predictions.csv")
    wf_rows = V.read(d / "walk_forward_predictions.csv")
    daily_rows = V.read(d / "daily.csv")
    wdaily_rows = V.read(d / "walk_forward_daily.csv")
    path_rows = V.read(d / "paths.csv")
    wpath_rows = V.read(d / "walk_forward_paths.csv")
    h, cost = int(params["h"]), float(params["cost_frac"])
    step, N, K = float(params["step"]), int(params["n_groups"]), int(params["k"])
    embargo_pct = float(params["embargo_pct"])
    max_draws = int(params["forest"]["max_draws"])
    say(f"loaded run files: {len(ext):,} events, {len(pred_rows):,} cpcv predictions")

    # ---- declared settings and run bookkeeping ----
    check("declared settings: 101-bar vertical barrier, 24 bp round trip, 6 groups / 2 test groups, bet step 0.1, 100 trees with sequential bootstrap and 10,000 draws, cutoff 2024-01-01, 87 prior trials, "
          "lookbacks 480 / 720 / 960 bars, the 13- and 14-feature lists, the nine variants and the four deflated-Sharpe variants",
          h == 100 and close(cost, 0.0024, 1e-12, 1e-12) and N == 6 and K == 2 and step == 0.1 and params["forest"]["n_estimators"] == 100 and params["forest"]["bootstrap"] == "sequential"
          and max_draws == 10000 and params["cutoff"] == "2024-01-01" and int(params["prior_trials"]) == 87 and list(params["mom_lookbacks"]) == list(LOOKBACKS)
          and params["dir_features"] == DECLARED_DIR and params["mom_features"] == DECLARED_MOM and params["variants"] == DECLARED_VARIANTS and params["dsr_variants"] == DECLARED_DSR
          and close(embargo_pct, 0.01, 1e-12, 1e-12) and close(float(params["min_weight_fraction_leaf"]), 0.05, 1e-12, 1e-12),
          str({k_: params.get(k_) for k_ in ("h", "cost_frac", "n_groups", "k", "step", "forest", "cutoff", "prior_trials", "mom_lookbacks")}))
    check("models and calibration files untouched during the run", integrity.get("unchanged") is True)
    changed = []
    for name, md5 in meta["script_md5"].items():
        p = HERE / name
        if p.exists() and hashlib.md5(p.read_bytes()).hexdigest() != md5:
            changed.append(name)
    check("the study scripts next to this checker are the ones that produced the run (md5 recorded in the run)", not changed, f"changed since the run: {changed}")
    if not (src / "events.csv").exists():
        check("source meta-pipeline run is present", False, f"{src}")
        return report("source run not found; pass its folder as the second argument")
    src_md5 = hashlib.md5((src / "events.csv").read_bytes()).hexdigest()
    src_sum_md5 = hashlib.md5((src / "summary_numbers.json").read_bytes()).hexdigest()
    s_summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    check("the source run's events.csv and summary_numbers.json are the files this run used (md5), and the source run declares the same 14 features, 30-day side, 101-bar barrier, 24 bp cost",
          src_md5 == meta["source_events_md5"] and src_sum_md5 == meta["source_summary_md5"] and s_summ["params"]["features"] == DECLARED_MOM and s_summ["params"]["side_signal"] == "mom720"
          and int(s_summ["params"]["h"]) == 100 and close(float(s_summ["params"]["cost_frac"]), cost, 1e-12, 1e-12), f"{src_md5} {meta['source_events_md5']}")
    check("window definitions: June 2026 = [2026-06-01, 2026-07-01) UTC and a sell-off window is present; each window starts before it ends; the windows equal the source run's",
          meta["windows"].get("june_2026_ai_correction", [None, None])[0].startswith("2026-06-01") and meta["windows"]["june_2026_ai_correction"][1].startswith("2026-07-01")
          and any(k_.startswith("sell_off") for k_ in meta["windows"]) and all(V.ts_key(v[0]) < V.ts_key(v[1]) for v in meta["windows"].values())
          and all(int(meta["window_epoch"][k_][0]) == int(V.ts_key(meta["windows"][k_][0])) and int(meta["window_epoch"][k_][1]) == int(V.ts_key(meta["windows"][k_][1])) for k_ in meta["windows"])
          and meta["window_epoch"] == s_summ["meta"]["window_epoch"], str(meta["windows"]))
    win_epoch = {k_: (int(v[0]), int(v[1])) for k_, v in meta["window_epoch"].items()}
    check("every window starts and ends on a UTC day boundary", all(a % 86400 == 0 and b % 86400 == 0 for a, b in win_epoch.values()), str(win_epoch))

    # ---- source events and the extension file ----
    ev = V.read(src / "events.csv")
    n_ev = len(ev)
    sym_rows = V.read(src / "symbols.csv")
    if len(ext) != n_ev or n_ev != int(meta["events"]):
        check("events_ext.csv has one row per source event", False, f"{len(ext)} {n_ev} {meta['events']}")
        return report("event counts differ")
    aligned = all(a["symbol"] == b["symbol"] and int(float(a["pos"])) == int(float(b["pos"])) and int(float(a["t0"])) == int(float(b["t0"])) for a, b in zip(ext, ev))
    check("events_ext.csv holds the source events in the same order (symbol, bar position, entry time)", aligned)
    if not aligned:
        return report("events not aligned")
    sym_pos = defaultdict(list)
    for i, e in enumerate(ev):
        sym_pos[e["symbol"]].append(i)
    rows_by_sym = {r["symbol"]: r for r in sym_rows}
    bad_bars, bad_ret = [], []
    exp_ret = {L: [NAN] * n_ev for L in LOOKBACKS}
    for s in sorted(sym_pos):
        keys, closes, _hi, _lo = VM.load_raw(meta["gmx_dir"], meta["tradingview_dir"], s, meta["timeframe"])
        n_keep = bisect.bisect_right(keys, V.ts_key(rows_by_sym[s]["last"]))
        if n_keep != int(rows_by_sym[s]["bars"]):
            bad_bars.append((s, n_keep, rows_by_sym[s]["bars"]))
            continue
        logc = [math.log(c) for c in closes[:n_keep]]
        for i in sym_pos[s]:
            p = int(float(ev[i]["pos"]))
            for L in LOOKBACKS:
                if p >= L:
                    exp_ret[L][i] = logc[p] - logc[p - L]
    check("raw candles cut at the source run's last bar have exactly the bar counts of the source run", not bad_bars, str(bad_bars[:3]))
    say("rebuilt trailing returns from raw candles")
    side_ev = [int(float(e["side"])) for e in ev]
    ret_side = [float(e["ret_side"]) for e in ev]
    exp_mom = []
    for i in range(n_ev):
        r = [exp_ret[L][i] for L in LOOKBACKS]
        exp_mom.append(1 if all(x > 0 for x in r) else (-1 if all(x < 0 for x in r) else 0))      # NaN comparisons are False -> 0
    exp_rl = [side_ev[i] * ret_side[i] for i in range(n_ev)]
    for i, x in enumerate(ext):
        for L, col in zip(LOOKBACKS, ("ret480", "ret720", "ret960")):
            a, b = float(x[col]), exp_ret[L][i]
            if not close(a, b, 1e-9, 1e-11):
                bad_ret.append((i, col, a, b))
    check("events_ext.csv: the 20-, 30- and 40-day trailing log returns of every event recomputed from the raw candles", not bad_ret and not bad_bars, str(bad_ret[:3]))
    mom_side = [int(float(x["mom_side"])) for x in ext]
    check("events_ext.csv: MOM side (+1 if all three returns > 0, -1 if all < 0, else 0), long return = side x return, direction label = long return > 0",
          mom_side == exp_mom and all(close(float(x["ret_long"]), exp_rl[i], 1e-9, 1e-12) for i, x in enumerate(ext)) and all(int(float(x["y_dir"])) == (1 if exp_rl[i] > 0 else 0) for i, x in enumerate(ext)),
          str([(i, mom_side[i], exp_mom[i]) for i in range(n_ev) if mom_side[i] != exp_mom[i]][:3]))
    sign720 = all((side_ev[i] == (1 if exp_ret[720][i] > 0 else -1)) for i in range(n_ev) if fin(exp_ret[720][i]))
    check("the source side equals the sign of the 30-day return wherever the 30-day return exists; MOM events agree with that side",
          sign720 and all(mom_side[i] == 0 or mom_side[i] == side_ev[i] for i in range(n_ev)))
    n_mom = sum(1 for m in mom_side if m != 0)
    check("MOM event counts in the run (total, long, short) equal the recount", int(meta["mom_events"]) == n_mom and int(meta["mom_long"]) == sum(1 for m in mom_side if m == 1) and int(meta["mom_short"]) == sum(1 for m in mom_side if m == -1))

    ret_long = exp_rl
    ydir = [1 if x > 0 else 0 for x in ret_long]
    binv = [int(float(e["bin"])) for e in ev]
    t0 = [int(float(e["t0"])) for e in ev]
    t1 = [int(float(e["t1"])) for e in ev]
    uu = [float(e["u"]) for e in ev]
    state_ev = [float(e["mkt_state"]) for e in ev]
    day_ev = [t // 86400 for t in t0]
    active = [m != 0 for m in mom_side]
    check("source columns used here are sane: uniqueness in (0,1], entry before exit, state 0 or 1, meta-label = 1 exactly when the side's return is positive",
          all(0 < u <= 1 + 1e-12 for u in uu) and all(b > a for a, b in zip(t0, t1)) and all(s in (0.0, 1.0) for s in state_ev) and all(b == (1 if r > 0 else 0) for b, r in zip(binv, ret_side)))

    # ---- CPCV bookkeeping (both forests) ----
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

    def draws(avg_u, n):
        return min(max(1, round(avg_u * n)), max_draws)

    bad_split, test_sets = [], []
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
        train_m = [i for i in train if active[i]]
        test_sets.append(test)
        r = split_rows[si]
        au_d, au_m = mean([uu[i] for i in train]), mean([uu[i] for i in train_m])
        want = {"n_train_dir": len(train), "n_train_mom": len(train_m), "n_test": len(test), "n_test_mom": sum(1 for i in test if active[i]),
                "draws_dir": draws(au_d, len(train)), "draws_mom": draws(au_m, len(train_m))}
        for k_, v in want.items():
            if int(r[k_]) != v:
                bad_split.append((si, k_, r[k_], v))
        for k_, v in (("avg_u_dir", au_d), ("avg_u_mom", au_m)):
            if not close(float(r[k_]), v, 1e-9, 1e-12):
                bad_split.append((si, k_, r[k_], v))
    check("train and test counts of every split for both forests (purge of overlapping labels, embargo after each test group; MOM forest = training events with a MOM signal), average training uniqueness, draws per tree",
          not bad_split, str(bad_split[:3]))

    by_split = defaultdict(list)
    seen_path = defaultdict(set)
    prob_bad = 0
    for r in pred_rows:
        si, ev_i, pa = int(r["split"]), int(r["event"]), int(r["path"])
        pd_, pm_ = float(r["p_dir"]), float(r["p_mom"])
        by_split[si].append((ev_i, pa, pd_, pm_))
        seen_path[ev_i].add(pa)
        if not (0.0 <= pd_ <= 1.0 and 0.0 <= pm_ <= 1.0):
            prob_bad += 1
    pred_bad = []
    for si, tg in enumerate(combos):
        rows_s = by_split.get(si, [])
        if sorted(e_ for e_, _, _, _ in rows_s) != test_sets[si]:
            pred_bad.append((si, "event set"))
            continue
        for e_, pa, _, _ in rows_s:
            if pa != assign[(si, grp[e_])]:
                pred_bad.append((si, e_, "path"))
                break
    check("cpcv_predictions.csv: every split predicts exactly its test events, each (split, group) sits on the right path, both probabilities in [0,1]",
          not pred_bad and not prob_bad and len(pred_rows) == sum(len(t) for t in test_sets), str(pred_bad[:3]))
    check("each event is predicted exactly once on every one of the backtest paths", len(seen_path) == n_ev and all(v == set(range(n_paths)) for v in seen_path.values()) and len(pred_rows) == n_ev * n_paths)

    def split_quality(rows_s):
        yd = [ydir[e_] for e_, _, _, _ in rows_s]
        pdv = [q for _, _, q, _ in rows_s]
        act_rows = [(e_, pm) for e_, _, _, pm in rows_s if active[e_]]
        ym = [binv[e_] for e_, _ in act_rows]
        pmv = [pm for _, pm in act_rows]
        return {"auc_dir": auc(yd, pdv), "auc_mom": auc(ym, pmv) if ym else NAN, "base_dir": mean(yd), "base_mom": mean(ym),
                "share_dir_gt_half": mean([1.0 if q > 0.5 else 0.0 for q in pdv]), "share_mom_gt_half": mean([1.0 if q > 0.5 else 0.0 for q in pmv])}

    mbad = []
    for si in range(len(combos)):
        for k_, v in split_quality(by_split[si]).items():
            if not close(V.num(split_rows[si][k_]), v, 1e-6, 1e-9):
                mbad.append((si, k_, split_rows[si][k_], v))
    check("per-split AUC of both forests (own rank method), base rates and shares above 1/2 recomputed from the predictions", not mbad, str(mbad[:3]))
    say("cpcv bookkeeping checked")

    # ---- evaluation, own implementation ----
    t_eff_div = max(1.0, (h + 1) / 24.0)

    def own_variants(i, p_dir, p_mom):
        """{variant: (side, size, algo)} for the real variants of one event."""
        out = {}
        ms = mom_side[i]
        out["long_all"] = (1, 1.0, 1)
        out["short_all"] = (-1, 1.0, 2)
        out["mom_all"] = (ms, 1.0 if ms != 0 else 0.0, 3)
        out["mom_long_up"] = (1, 1.0 if (ms == 1 and state_ev[i] == 1.0) else 0.0, 3)
        ds = 1 if p_dir > 0.5 else -1
        out["dir_model"] = (ds, bet_size(max(p_dir, 1.0 - p_dir), step), 1 if ds > 0 else 2)
        out["mom_meta"] = (ms, bet_size(p_mom, step) if ms != 0 else 0.0, 3)
        best, side, algo = -math.inf, 1, 1
        for conf, sd, al in (((p_mom if ms != 0 else -math.inf), ms, 3), (p_dir, 1, 1), (1.0 - p_dir, -1, 2)):
            if conf > best:
                best, side, algo = conf, sd, al
        out["gate"] = (side, bet_size(best, step), algo)
        return out

    def best20(xs):
        tot = sum(xs)
        if not tot > 0 or not xs:
            return NAN
        return sum(sorted(xs)[-20:]) / tot

    def evaluate(pred_by_path):
        out = {"paths": {}, "daily": {}}
        st_acc = defaultdict(lambda: [0.0, 0, 0.0])
        wins = {(v, wn): [0.0, 0] for v in REAL for wn in win_epoch}
        algo_state = defaultdict(lambda: [0.0, 0])
        algo_win = defaultdict(lambda: [0.0, 0])
        cand_state = defaultdict(int)
        cand_win = defaultdict(int)
        for pa, lst in sorted(pred_by_path.items()):
            dd = {v: defaultdict(lambda: [0.0, 0]) for v in REAL}
            tot = {v: 0.0 for v in REAL}
            taken = {v: 0 for v in REAL}
            szsum = {v: 0.0 for v in REAL}
            for e_, pdir, pmom in lst:
                vs = own_variants(e_, pdir, pmom)
                sv = int(state_ev[e_])
                cand_state[sv] += 1
                for wn, (lo, hi) in win_epoch.items():
                    if lo <= t0[e_] < hi:
                        cand_win[wn] += 1
                for v in REAL:
                    side, size, al = vs[v]
                    pnl = size * (side * ret_long[e_] - cost * (1.0 if size > 0 else 0.0)) * 1e4
                    cell = dd[v][day_ev[e_]]
                    cell[0] += pnl
                    cell[1] += 1
                    tot[v] += pnl
                    taken[v] += 1 if size > 0 else 0
                    szsum[v] += size
                    a = st_acc[(v, sv)]
                    a[0] += pnl; a[1] += 1; a[2] += size
                    for wn, (lo, hi) in win_epoch.items():
                        if lo <= t0[e_] < hi:
                            wins[(v, wn)][0] += pnl
                            wins[(v, wn)][1] += 1
                    if v in ("gate", "dir_model") and size > 0:
                        b = algo_state[(v, al, sv)]
                        b[0] += pnl; b[1] += 1
                        for wn, (lo, hi) in win_epoch.items():
                            if lo <= t0[e_] < hi:
                                c = algo_win[(v, al, wn)]
                                c[0] += pnl; c[1] += 1
            for v in REAL:
                days = sorted(dd[v])
                xs = [dd[v][k_][0] / dd[v][k_][1] for k_ in days]
                st = VM.day_stats(xs, len(lst), t_eff_div)
                st.update({"events": len(lst), "mean_bp_per_candidate": tot[v] / len(lst), "taken_share": taken[v] / len(lst), "mean_size": szsum[v] / len(lst), "best20_share": best20(xs)})
                out["paths"][(pa, v)] = st
                out["daily"][(pa, v)] = {k_: (dd[v][k_][0], dd[v][k_][1]) for k_ in days}
        npth = len(pred_by_path)
        out["win_sums"] = wins
        out["by_state"] = {f"{v}|state{sv}": {"events_per_path": a[1] / npth, "mean_bp_per_candidate": a[0] / a[1] if a[1] else NAN, "mean_size": a[2] / a[1] if a[1] else NAN} for (v, sv), a in st_acc.items()}
        out["algo_state"] = {}
        out["algo_window"] = {}
        for v in ("gate", "dir_model"):
            for al in ALGOS:
                for sv in (0, 1):
                    tt, nt = algo_state[(v, ALGO_CODE[al], sv)] if (v, ALGO_CODE[al], sv) in algo_state else (0.0, 0)
                    out["algo_state"][f"{v}|{al}|state{sv}"] = {"trades_per_path": nt / npth, "bp_per_state_candidate": tt / cand_state[sv] if cand_state[sv] else NAN, "bp_per_trade": tt / nt if nt else NAN}
                for wn in win_epoch:
                    tt, nt = algo_win[(v, ALGO_CODE[al], wn)] if (v, ALGO_CODE[al], wn) in algo_win else (0.0, 0)
                    out["algo_window"][f"{v}|{al}|{wn}"] = {"trades_per_path": nt / npth, "bp_per_window_candidate": tt / cand_win[wn] if cand_win[wn] else NAN, "bp_per_trade": tt / nt if nt else NAN}
        out["cand_state"] = {f"state{sv}": c / npth for sv, c in cand_state.items()}
        out["cand_win"] = {wn: cand_win[wn] / npth for wn in win_epoch}
        return out

    pred_by_path = defaultdict(list)
    for r in pred_rows:
        pred_by_path[int(r["path"])].append((int(r["event"]), float(r["p_dir"]), float(r["p_mom"])))
    E = evaluate(pred_by_path)
    say("evaluated the CPCV paths")

    def load_daily(rows):
        g = defaultdict(dict)
        for r in rows:
            g[(int(r["path"]), r["variant"])][int(r["day"])] = (float(r["sum_bp"]), int(r["n"]))
        return g

    def stats_from_daily(g, n_events):
        xs = [g[k_][0] / g[k_][1] for k_ in sorted(g)]
        st = VM.day_stats(xs, n_events, t_eff_div)
        st["events"] = sum(c for _, c in g.values())
        st["mean_bp_per_candidate"] = sum(s_ for s_, _ in g.values()) / st["events"]
        st["best20_share"] = best20(xs)
        return st

    def daily_and_paths_checks(label, E_, drows, prows, n_p, n_expected):
        dg = load_daily(drows)
        dbad = []
        for key, dd in E_["daily"].items():
            g = dg.get(key, {})
            if set(g) != set(dd) or any(g[k_][1] != dd[k_][1] or not close(g[k_][0], dd[k_][0], 1e-7, 1e-6) for k_ in dd if k_ in g):
                dbad.append(key)
        for pa in range(n_p):
            ref = E_["daily"][(pa, "long_all")]
            for nv in NULLS:
                g = dg.get((pa, nv), {})
                if set(g) != set(ref) or any(g[k_][1] != ref[k_][1] for k_ in ref if k_ in g):
                    dbad.append((pa, nv))
        check(f"{label} daily file: the seven real variants' daily series (sum of bp, events) of every path recomputed; the two nulls have the same days and event counts as long_all",
              not dbad and len(drows) == len(DECLARED_VARIANTS) * sum(len(v) for (pa, v_), v in E_["daily"].items() if v_ == "long_all"), str(dbad[:3]))
        prow = {(int(r["path"]), r["variant"]): r for r in prows}
        pbad = []
        cmp_fields = ("events", "days", "mean_bp_per_candidate", "sr_daily", "sr_annual", "skew", "kurt", "t_eff", "psr", "taken_share", "mean_size", "best20_share")
        per_path = {v: [] for v in DECLARED_VARIANTS}
        for pa in range(n_p):
            for v in REAL:
                st, r = E_["paths"][(pa, v)], prow[(pa, v)]
                for f_ in cmp_fields:
                    want = st["days"] if f_ == "days" else st[f_]
                    if not close(V.num(r[f_]), float(want), 1e-6, 1e-8):
                        pbad.append((pa, v, f_, r[f_], want))
                per_path[v].append(st)
            gate_st = E_["paths"][(pa, "gate")]
            for nv in NULLS:
                st = stats_from_daily(dg[(pa, nv)], 0)
                st["taken_share"], st["mean_size"] = gate_st["taken_share"], gate_st["mean_size"]
                r = prow[(pa, nv)]
                for f_, want in (("events", st["events"]), ("days", st["days"]), ("mean_bp_per_candidate", st["mean_bp_per_candidate"]), ("sr_daily", st["sr_daily"]), ("sr_annual", st["sr_annual"]),
                                 ("skew", st["skew"]), ("kurt", st["kurt"]), ("psr", st["psr"]), ("taken_share", st["taken_share"]), ("mean_size", st["mean_size"]), ("best20_share", st["best20_share"])):
                    if not close(V.num(r[f_]), float(want), 1e-6, 1e-8):
                        pbad.append((pa, nv, f_, r[f_], want))
                st["events"] = int(r["events"])
                per_path[nv].append(st)
        check(f"{label} paths file: every statistic of the seven real variants recomputed from the events; the nulls' statistics recomputed from their daily rows, with the gate's taken share and mean size",
              not pbad and len(prows) == len(DECLARED_VARIANTS) * n_p and all(int(prow[(pa, v)]["events"]) == n_expected for pa in range(n_p) for v in DECLARED_VARIANTS), str(pbad[:4]))
        return dg, per_path

    dg, per_path = daily_and_paths_checks("CPCV", E, daily_rows, path_rows, n_paths, n_ev)

    def summarize(pp):
        sm = {}
        for v, lst in pp.items():
            b20 = [s["best20_share"] for s in lst if fin(s["best20_share"])]
            sm[v] = {"paths": len(lst), "mean_bp_per_candidate": mean([s["mean_bp_per_candidate"] for s in lst]), "mean_sr_daily": mean([s["sr_daily"] for s in lst]),
                     "mean_sr_annual": mean([s["sr_annual"] for s in lst]), "paths_sr_negative": sum(1 for s in lst if s["sr_daily"] < 0), "mean_psr": mean([s["psr"] for s in lst]),
                     "taken_share": mean([s["taken_share"] for s in lst]), "mean_size": mean([s["mean_size"] for s in lst]), "best20_share": mean(b20) if b20 else NAN}
        return sm

    COMPARES = (("gate", "gate_shuffled_size"), ("gate", "gate_random_side"), ("gate", "mom_all"), ("gate", "long_all"), ("gate", "dir_model"), ("gate", "mom_meta"), ("gate", "mom_long_up"),
                ("mom_meta", "mom_all"), ("dir_model", "long_all"))

    def summary_checks(label, pp, run_eval):
        sm = summarize(pp)
        bad = []
        for v in DECLARED_VARIANTS:
            for f_, want in sm[v].items():
                if not close(float(run_eval["summary"][v][f_]), float(want), 1e-6, 1e-8):
                    bad.append((v, f_, run_eval["summary"][v][f_], want))
        if set(run_eval["compare"]) != {f"{a}|{b}" for a, b in COMPARES}:
            bad.append(("compare keys", sorted(run_eval["compare"])))
        for a, b in COMPARES:
            c = run_eval["compare"].get(f"{a}|{b}")
            if c is None:
                continue
            dsr_ = [x["sr_daily"] - y["sr_daily"] for x, y in zip(pp[a], pp[b])]
            dbp = [x["mean_bp_per_candidate"] - y["mean_bp_per_candidate"] for x, y in zip(pp[a], pp[b])]
            for f_, want in (("paths_sr_higher", sum(1 for x in dsr_ if x > 0)), ("paths", len(dsr_)), ("mean_sr_daily_diff", mean(dsr_)), ("mean_bp_diff", mean(dbp))):
                if not close(float(c[f_]), float(want), 1e-6, 1e-8):
                    bad.append((a, b, f_, c[f_], want))
        check(f"{label}: summary table of the nine variants and the nine comparisons recomputed", not bad, str(bad[:4]))

    def dsr_table(stats, prior):
        srs = [s["sr_daily"] for s in stats]
        sr_hat = mean(srs)
        var_paths = (sum((x - sr_hat) ** 2 for x in srs) / (len(srs) - 1)) if len(srs) > 1 else NAN
        t_eff, sk, ku = mean([s["t_eff"] for s in stats]), mean([s["skew"] for s in stats]), mean([s["kurt"] for s in stats])
        var_asym = (1.0 - sk * sr_hat + (ku - 1.0) / 4.0 * sr_hat ** 2) / max(t_eff - 1.0, 1.0) if fin(sr_hat) else NAN
        cands = [v for v in (var_paths, var_asym) if fin(v)]
        var_sr = max(cands) if cands else NAN
        table = {}
        for n_tr in sorted({1, 10, int(prior), int(prior) + 1}):
            if n_tr <= 1:
                table[str(n_tr)] = (0.0, V.psr(sr_hat, t_eff, sk, ku, 0.0))
            else:
                star = VM.expected_max_sharpe(n_tr, var_sr) if fin(var_sr) else NAN
                table[str(n_tr)] = (star, V.psr(sr_hat, t_eff, sk, ku, star) if fin(star) else NAN)
        return {"sr_daily_mean": sr_hat, "var_sr_used": var_sr, "var_across_paths": var_paths, "var_asymptotic": var_asym, "t_eff": t_eff, "skew": sk, "kurt": ku}, table

    def dsr_checks(label, pp, run_eval):
        bad = []
        for v in DECLARED_DSR:
            head, table = dsr_table(pp[v], params["prior_trials"])
            rd = run_eval["dsr"][v]
            for f_, want in head.items():
                if not close(float(rd[f_]), float(want), 1e-6, 1e-9):
                    bad.append((v, f_, rd[f_], want))
            if set(rd["by_trials"]) != set(table) or set(table) != {"1", "10", "87", "88"}:
                bad.append((v, "trial counts", sorted(rd["by_trials"])))
            else:
                for k_, (star, dsr) in table.items():
                    if not (close(float(rd["by_trials"][k_]["sr_star"]), star, 1e-6, 1e-9) and close(float(rd["by_trials"][k_]["dsr"]), dsr, 1e-6, 1e-9)):
                        bad.append((v, k_, rd["by_trials"][k_], (star, dsr)))
        check(f"{label}: deflated-Sharpe tables of mom_long_up, dir_model, mom_meta and gate (N = 1, 10, 87, 88; variance = larger of across-path and asymptotic) recomputed with own normal quantiles", not bad, str(bad[:3]))

    summary_checks("CPCV", per_path, summ["eval"])
    dsr_checks("CPCV", per_path, summ["eval"])

    def window_checks(label, E_, dgot, run_eval, n_p):
        bad = []
        for wn, (lo, hi) in win_epoch.items():
            dlo, dhi = lo // 86400, hi // 86400
            if not close(float(run_eval["candidates_window"][wn]), E_["cand_win"][wn], 1e-9, 1e-9):
                bad.append((wn, "candidates", run_eval["candidates_window"][wn], E_["cand_win"][wn]))
            for v in DECLARED_VARIANTS:
                tot, cnt = 0.0, 0
                for pa in range(n_p):
                    for k_, (s_, c_) in dgot[(pa, v)].items():
                        if dlo <= k_ < dhi:
                            tot += s_
                            cnt += c_
                row = run_eval["windows"][f"{v}|{wn}"]
                want_mean = tot / cnt if cnt else NAN
                if not close(float(row["events_per_path"]), cnt / n_p, 1e-9, 1e-9) or not close(float(row["mean_bp_per_candidate"]), want_mean, 1e-6, 1e-7):
                    bad.append((wn, v, row, (cnt, want_mean)))
                if v in REAL:
                    s2, c2 = E_["win_sums"][(v, wn)]
                    if c2 != cnt or not close(s2, tot, 1e-7, 1e-5):
                        bad.append((wn, v, "events vs daily", (c2, s2), (cnt, tot)))
        check(f"{label}: window lines (June 2026, sell-off, deepest fall) for all nine variants, recomputed from the daily rows and, for the seven real variants, from the events", not bad, str(bad[:3]))

    window_checks("CPCV", E, dg, summ["eval"], n_paths)

    def state_checks(label, E_, run_eval, n_p, n_expected):
        bad = []
        for v in REAL:
            for sv in (0, 1):
                k_ = f"{v}|state{sv}"
                for f_ in ("events_per_path", "mean_bp_per_candidate", "mean_size"):
                    if not close(float(run_eval["by_state"][k_][f_]), float(E_["by_state"][k_][f_]), 1e-6, 1e-8):
                        bad.append((k_, f_, run_eval["by_state"][k_][f_], E_["by_state"][k_][f_]))
        for sv in (0, 1):
            for nv in NULLS:
                row, ref = run_eval["by_state"][f"{nv}|state{sv}"], run_eval["by_state"][f"gate|state{sv}"]
                if not close(float(row["events_per_path"]), float(ref["events_per_path"]), 1e-9, 1e-9):
                    bad.append((nv, sv, "events"))
            if not close(float(run_eval["by_state"][f"gate_random_side|state{sv}"]["mean_size"]), float(run_eval["by_state"][f"gate|state{sv}"]["mean_size"]), 1e-9, 1e-12):
                bad.append(("gate_random_side", sv, "mean size must equal the gate's"))
            if not close(float(run_eval["candidates_state"][f"state{sv}"]), E_["cand_state"][f"state{sv}"], 1e-9, 1e-9):
                bad.append(("candidates", sv))
        sz_tot = sum(float(run_eval["by_state"][f"gate_shuffled_size|state{sv}"]["events_per_path"]) * float(run_eval["by_state"][f"gate_shuffled_size|state{sv}"]["mean_size"]) for sv in (0, 1))
        if not close(sz_tot, float(run_eval["summary"]["gate_shuffled_size"]["mean_size"]) * n_expected, 1e-6, 1e-6):
            bad.append(("shuffled sizes by state do not add up to the total",))
        if not close(sum(E_["cand_state"].values()), n_expected, 1e-9, 1e-9):
            bad.append(("state candidates do not add up", sum(E_["cand_state"].values()), n_expected))
        for k_, want in E_["algo_state"].items():
            got = run_eval["algo_state"][k_]
            for f_ in ("trades_per_path", "bp_per_state_candidate", "bp_per_trade"):
                if not close(float(got[f_]), float(want[f_]), 1e-6, 1e-8):
                    bad.append((k_, f_, got[f_], want[f_]))
        for k_, want in E_["algo_window"].items():
            got = run_eval["algo_window"][k_]
            for f_ in ("trades_per_path", "bp_per_window_candidate", "bp_per_trade"):
                if not close(float(got[f_]), float(want[f_]), 1e-6, 1e-8):
                    bad.append((k_, f_, got[f_], want[f_]))
        if set(run_eval["algo_state"]) != set(E_["algo_state"]) or set(run_eval["algo_window"]) != set(E_["algo_window"]):
            bad.append(("algo table keys",))
        for sv in (0, 1):
            parts = sum(E_["algo_state"][f"gate|{al}|state{sv}"]["bp_per_state_candidate"] for al in ALGOS)
            if not close(parts, E_["by_state"][f"gate|state{sv}"]["mean_bp_per_candidate"], 1e-7, 1e-7):
                bad.append(("algorithms do not add up to the gate in state", sv))
        check(f"{label}: by-state table (all variants; nulls: event counts and sizes), candidate counts, and the gate's / dir_model's by-algorithm tables by state and window recomputed; algorithms add up to the gate", not bad, str(bad[:4]))

    state_checks("CPCV", E, summ["eval"], n_paths, n_ev)

    # ---- walk-forward ----
    cut = float(V.ts_key(params["cutoff"]))
    wf_set = sorted(i for i in range(n_ev) if t0[i] >= cut)
    train_ev = [i for i in range(n_ev) if t1[i] < cut]
    train_m = [i for i in train_ev if active[i]]
    wf_pred = [(int(r["event"]), float(r["p_dir"]), float(r["p_mom"])) for r in wf_rows]
    wmet = summ["walk_forward"]
    wf_bad = []
    if sorted(e_ for e_, _, _ in wf_pred) != wf_set:
        wf_bad.append("event set")
    au_d, au_m = mean([uu[i] for i in train_ev]), mean([uu[i] for i in train_m])
    want = {"n_train_dir": len(train_ev), "n_train_mom": len(train_m), "n_test": len(wf_set), "n_test_mom": sum(1 for i in wf_set if active[i]),
            "draws_dir": draws(au_d, len(train_ev)), "draws_mom": draws(au_m, len(train_m))}
    for k_, v in want.items():
        if int(wmet[k_]) != v:
            wf_bad.append((k_, wmet[k_], v))
    for k_, v in (("avg_u_dir", au_d), ("avg_u_mom", au_m)):
        if not close(float(wmet[k_]), v, 1e-9, 1e-12):
            wf_bad.append((k_, wmet[k_], v))
    rows_w = [(e_, 0, pdv, pmv) for e_, pdv, pmv in wf_pred]
    for k_, v in split_quality(rows_w).items():
        if not close(float(wmet[k_]), v, 1e-6, 1e-9):
            wf_bad.append((k_, wmet[k_], v))
    check("walk-forward: training events are those whose labels end before the cutoff, test events start on or after it (none in both); counts for both forests, uniqueness, draws, AUCs, base rates, shares recomputed",
          not wf_bad and max(t1[i] for i in train_ev) < cut <= min(t0[i] for i in wf_set), str(wf_bad[:3]))
    EW = evaluate({0: wf_pred})
    wdg, wper = daily_and_paths_checks("walk-forward", EW, wdaily_rows, wpath_rows, 1, len(wf_set))
    summary_checks("walk-forward", wper, summ["walk_forward_eval"])
    dsr_checks("walk-forward", wper, summ["walk_forward_eval"])
    window_checks("walk-forward", EW, wdg, summ["walk_forward_eval"], 1)
    state_checks("walk-forward", EW, summ["walk_forward_eval"], 1, len(wf_set))

    # ---- aggregate blocks and the rendered SUMMARY.md ----
    cp = summ["cpcv"]
    check("summary_numbers.json cpcv block equals the mean / min / max of the per-split metrics",
          all(close(cp[k_], v, 1e-9, 1e-12) for k_, v in (("auc_dir_mean", mean([float(r["auc_dir"]) for r in split_rows])), ("auc_dir_min", min(float(r["auc_dir"]) for r in split_rows)),
                                                           ("auc_dir_max", max(float(r["auc_dir"]) for r in split_rows)), ("auc_mom_mean", mean([float(r["auc_mom"]) for r in split_rows])),
                                                           ("auc_mom_min", min(float(r["auc_mom"]) for r in split_rows)), ("auc_mom_max", max(float(r["auc_mom"]) for r in split_rows)),
                                                           ("share_dir_mean", mean([float(r["share_dir_gt_half"]) for r in split_rows])), ("share_mom_mean", mean([float(r["share_mom_gt_half"]) for r in split_rows])))))
    md = (d / "SUMMARY.md").read_text(encoding="utf-8")
    lines = md.splitlines()

    def fm(x, nd):
        return "n/a" if x is None or not fin(float(x)) else f"{float(x):.{nd}f}"

    found = set()
    for ln in lines:
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) == 10 and cells[0] in DECLARED_VARIANTS:
                cells = cells[:1] + cells[2:]                               # the free-text 'what it is' column is not compared
            found.add(tuple(cells))
    expected = []
    for v in DECLARED_VARIANTS:
        s_ = summ["eval"]["summary"][v]
        expected.append((v, fm(s_["mean_bp_per_candidate"], 1), fm(s_["mean_sr_daily"], 4), fm(s_["mean_sr_annual"], 2), f"{s_['paths_sr_negative']}/{s_['paths']}", fm(s_["mean_psr"], 2),
                         fm(s_["taken_share"], 2), fm(s_["mean_size"], 3), fm(s_["best20_share"], 2)))
        w_ = summ["walk_forward_eval"]["summary"][v]
        expected.append((v, fm(w_["mean_bp_per_candidate"], 1), fm(w_["mean_sr_daily"], 4), fm(w_["mean_sr_annual"], 2), fm(w_["mean_psr"], 2), fm(w_["taken_share"], 2)))
    for v in DECLARED_DSR:
        for n_tr, x in summ["eval"]["dsr"][v]["by_trials"].items():
            expected.append((n_tr, fm(x["sr_star"], 4), fm(x["dsr"], 3)))
    for key in ("eval", "walk_forward_eval"):
        r_ = summ[key]
        for wn in meta["windows"]:
            expected.append((wn, fm(r_["candidates_window"][wn], 0)) + tuple(fm(r_["windows"][f"{v}|{wn}"]["mean_bp_per_candidate"], 1) for v in DECLARED_VARIANTS))
        for sv in (1, 0):
            expected.append((str(sv), fm(r_["candidates_state"][f"state{sv}"], 0)) + tuple(fm(r_["by_state"][f"{v}|state{sv}"]["mean_bp_per_candidate"], 1) for v in DECLARED_VARIANTS))
            for al in ALGOS:
                a_ = r_["algo_state"][f"gate|{al}|state{sv}"]
                expected.append((str(sv), al, fm(a_["trades_per_path"], 0), fm(a_["bp_per_state_candidate"], 1), fm(a_["bp_per_trade"], 1)))
        for wn in meta["windows"]:
            for al in ALGOS:
                a_ = r_["algo_window"][f"gate|{al}|{wn}"]
                expected.append((wn, al, fm(a_["trades_per_path"], 0), fm(a_["bp_per_window_candidate"], 1), fm(a_["bp_per_trade"], 1)))
    miss = [e_ for e_ in expected if e_ not in found]
    cp_line = f"F_dir mean {fm(cp['auc_dir_mean'], 3)}, min {fm(cp['auc_dir_min'], 3)}, max {fm(cp['auc_dir_max'], 3)}; F_mom mean {fm(cp['auc_mom_mean'], 3)}, min {fm(cp['auc_mom_min'], 3)}, max {fm(cp['auc_mom_max'], 3)}"
    if cp_line not in md:
        miss.append(("AUC line", cp_line))
    check("SUMMARY.md: every table row (variants, walk-forward, deflated Sharpe, windows, states, algorithms) equals the numbers in summary_numbers.json cell by cell; the AUC line matches; the design text, deviations and all three windows are listed",
          not miss and "DESIGN" in md and "Deviations from the book" in md and "june_2026_ai_correction" in md and any("sell_off" in ln for ln in lines), str(miss[:3]))
    return report()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    raise SystemExit(main(argv[0], argv[1] if len(argv) > 1 else None))
