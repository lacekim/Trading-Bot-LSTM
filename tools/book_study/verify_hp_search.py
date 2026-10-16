#!/usr/bin/env python
"""Independent checker for a run_hp_search.py output folder (stdlib only; imports neither the study code, numpy, pandas nor sklearn).

From the source run's events.csv (and its walk_forward_predictions.csv) it re-derives:
  * the declared settings (grid of 12 configs, search seed, seeds, criterion, folds, embargo, cutoff, bet-size step, cost) and the md5 of run_hp_search.py;
  * the search set (events whose labels end before the cutoff), the five purged folds with the embargo (own implementation of the rule: contiguous equal-count folds in start-time order,
    training events dropped when their label interval touches the test span or they start inside the embargo after it) and that the out-of-fold predictions of every config cover
    each fold's test events exactly;
  * for every config the pooled out-of-fold weighted log loss, AUC (rank formula), share of p > 0.5 and the fold mean / sd of the log loss, the two ranks, and that the chosen config
    is the lowest log loss (first on a tie);
  * the post-cutoff scoring: the test events of every (arm, seed), AUC, weighted log loss, and for the primary and meta variants the per-event convention of the pipeline (bet size from
    p, pnl, daily series, annualised Sharpe, mean bp, taken share, mean size, June 2026 and sell-off window means).  The placebo (a seeded shuffle) is NOT recomputed: the columns that
    involve it are taken as given;
  * that the default config at the source run's own seed reproduces the source run's walk_forward_predictions.csv (within 1e-6);
  * every cell of the tables in hp_summary.md.
The forests themselves are not refitted here.  Exit code 0 = all checks pass.

    python tools/book_study/verify_hp_search.py runtime/book_hp_search/<timestamp>
"""
from __future__ import annotations

import csv
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
MWFL = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
MAXF = ["sqrt", 0.5]
GRID = [(a, b) for a in MWFL for b in MAXF]
SEEDS = [20261017, 20261101, 20261102, 20261103, 20261104]
SEARCH_SEED, N_FOLDS, EMBARGO = 20261201, 5, 0.01
DAY = 86400


def close(a, b, rel=1e-6, ab=1e-9):
    return V.same(a, b, rel, ab)


def wll(y, p, w):
    s = 0.0
    for yi, pi, wi in zip(y, p, w):
        q = min(max(pi, 1e-6), 1 - 1e-6)
        s += wi * (yi * math.log(q) + (1 - yi) * math.log(1 - q))
    return -s / sum(w)


def auc(y, p):
    n1 = sum(y)
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return NAN
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
    sr = sum(ranks[k] for k in range(len(y)) if y[k] == 1)
    return (sr - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def mean(v):
    return sum(v) / len(v) if v else NAN


def sd(v):
    if len(v) < 2:
        return NAN
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def bet_size(p, step):
    p = min(max(p, 1e-9), 1 - 1e-9)
    z = (p - 0.5) / math.sqrt(p * (1 - p))
    s = (2.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) - 1.0) if p > 0.5 else 0.0
    if step > 0:
        s = round(s / step) * step
    return min(max(s, 0.0), 1.0)


def purged_folds(t0, t1, idx, n_splits, embargo_pct):
    """Own implementation on the events idx (list of global indices): contiguous equal-count folds in start-time order, array_split sizes (first n % k folds one larger)."""
    order = sorted(idx, key=lambda i: (t0[i], i))
    n = len(order)
    sizes = [n // n_splits + (1 if k < n % n_splits else 0) for k in range(n_splits)]
    t_min, t_max = float(min(t0[i] for i in idx)), float(max(t0[i] for i in idx)) + 1.0
    emb = embargo_pct * (t_max - t_min)
    out, a = [], 0
    for sz in sizes:
        te = order[a:a + sz]
        a += sz
        lo, hi = float(min(t0[i] for i in te)), float(max(t1[i] for i in te))
        tes = set(te)
        tr = [i for i in idx if i not in tes and not (t0[i] <= hi and t1[i] >= lo) and not (t0[i] > hi and t0[i] <= hi + emb)]
        out.append((sorted(tr), sorted(te)))
    return out


def per_event_eval(p_by_event, events, cut, step, cost_frac, windows):
    """Pipeline convention for the primary and meta variants on one walk-forward prediction set."""
    ev = sorted(p_by_event)
    ret = [events["ret"][e] for e in ev]
    t0 = [events["t0"][e] for e in ev]
    out = {}
    for v in ("primary", "meta"):
        s = [1.0 if v == "primary" else bet_size(p_by_event[e], step) for e in ev]
        pnl = [si * (r - cost_frac * (si > 0)) * 1e4 for si, r in zip(s, ret)]
        days = defaultdict(lambda: [0.0, 0])
        for t, x in zip(t0, pnl):
            days[t // DAY][0] += x
            days[t // DAY][1] += 1
        ser = [a / b for a, b in days.values()]
        m, sdv = mean(ser), sd(ser)
        sr = m / sdv if len(ser) >= 4 and sdv and sdv > 0 else NAN
        out[v] = {"sr_annual": sr * math.sqrt(365) if not math.isnan(sr) else NAN, "bp": mean(pnl), "taken": mean([1.0 if x > 0 else 0.0 for x in s]), "mean_size": mean(s),
                  "win": {wn: (mean([x for x, t in zip(pnl, t0) if lo <= t < hi]) if any(lo <= t < hi for t in t0) else NAN) for wn, (lo, hi) in windows.items()}}
    return out


def main(folder: str, allow_quick: bool = False) -> int:
    d = Path(folder)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    extra = json.loads((d / "hp_extra.json").read_text(encoding="utf-8"))
    run = Path(extra["source_run"])
    if not run.is_absolute():
        run = ROOT / run
    summ = json.loads((run / "summary_numbers.json").read_text(encoding="utf-8"))
    p = summ["params"]
    step, cost_frac = float(p["step"]), float(p["cost_frac"])
    cut = float(V.ts_key(p["cutoff"]))
    windows = {k: (int(v[0]), int(v[1])) for k, v in summ["meta"]["window_epoch"].items()}
    ev = V.read(run / "events.csv")
    events = {"t0": [int(float(r["t0"])) for r in ev], "t1": [int(float(r["t1"])) for r in ev], "y": [int(float(r["bin"])) for r in ev], "w": [float(r["w"]) for r in ev],
              "ret": [float(r["ret_side"]) for r in ev]}
    n_ev = len(ev)
    pre = [i for i in range(n_ev) if events["t1"][i] < cut]
    test_wf = [i for i in range(n_ev) if events["t0"][i] >= cut]

    bad_md5 = [n for n, m in extra["script_md5"].items() if not (HERE / n).exists() or hashlib.md5((HERE / n).read_bytes()).hexdigest() != m]
    declared_grid = [(g["min_weight_fraction_leaf"], g["max_features"]) for g in extra["grid"]]
    check("settings: the grid of 12 configs, search seed, seeds, criterion, 5 folds, 1% embargo, cutoff, step and cost are the declared ones (not a --quick run), and run_hp_search.py is the recorded script (md5)",
          (not extra["quick"] or allow_quick) and (declared_grid == GRID or extra["quick"]) and extra["search_seed"] == SEARCH_SEED and (extra["seeds"] == SEEDS or extra["quick"]) and extra["n_folds"] == N_FOLDS
          and extra["embargo_pct"] == EMBARGO and extra["cutoff"] == p["cutoff"] and close(extra["step"], step, 1e-12, 1e-12) and close(extra["cost_frac"], cost_frac, 1e-12, 1e-12)
          and extra["criterion"].startswith("pooled out-of-fold sample-weighted log loss") and extra["forest"] == {"n_estimators": 100, "bootstrap": "sequential", "max_draws": 10000}
          and not bad_md5, f"quick {extra['quick']} grid {declared_grid[:3]} changed {bad_md5}")
    check("search set = events whose labels end before the cutoff (count recorded), and the source run is a 14-feature run", extra["n_search_events"] == len(pre) and extra["n_events"] == n_ev and len(p["features"]) == 14, f"{extra['n_search_events']} vs {len(pre)}")

    folds = purged_folds(events["t0"], events["t1"], pre, N_FOLDS, EMBARGO)
    fr = V.read(d / "folds.csv")
    check("fold sizes (training events after purge and embargo, test events) of the five purged folds equal the independent rebuild",
          [(int(r["n_train"]), int(r["n_test"])) for r in fr] == [(len(tr), len(te)) for tr, te in folds], str([(int(r["n_train"]), int(r["n_test"])) for r in fr]))
    oof = defaultdict(lambda: defaultdict(list))
    for r in V.read(d / "oof_predictions.csv"):
        oof[int(r["config"])][int(r["fold"])].append((int(r["event"]), float(r["p"])))
    cfg_ids = [int(r["config"]) for r in extra["grid"]]
    bad_cov = [(c, k) for c in cfg_ids for k in range(N_FOLDS) if sorted(e for e, _ in oof[c][k]) != folds[k][1]]
    check("for every config the out-of-fold predictions of each fold cover exactly that fold's test events once, with probabilities in [0, 1]",
          not bad_cov and set(oof) == set(cfg_ids) and all(0.0 <= q <= 1.0 for c in oof for k in oof[c] for _, q in oof[c][k]), str(bad_cov[:3]))

    sc = {int(r["config"]): r for r in V.read(d / "config_scores.csv")}
    exp = {}
    for c in cfg_ids:
        allp = [(e, q) for k in range(N_FOLDS) for e, q in oof[c][k]]
        y, w, pp = [events["y"][e] for e, _ in allp], [events["w"][e] for e, _ in allp], [q for _, q in allp]
        pf = [wll([events["y"][e] for e, _ in oof[c][k]], [q for _, q in oof[c][k]], [events["w"][e] for e, _ in oof[c][k]]) for k in range(N_FOLDS)]
        exp[c] = {"log_loss_w": wll(y, pp, w), "auc": auc(y, pp), "share_p_gt_half": mean([1.0 if q > 0.5 else 0.0 for q in pp]), "fold_log_loss_mean": mean(pf), "fold_log_loss_sd": sd(pf), "n_oof": len(allp)}
    bad_sc = [(c, k, sc[c][k], v) for c in cfg_ids for k, v in exp[c].items() if not close(V.num(sc[c][k]), float(v), 1e-7, 1e-9)]
    check("config_scores.csv: pooled out-of-fold weighted log loss, AUC (rank formula), share p > 0.5, fold mean and sd of the log loss and the number of out-of-fold events equal the recomputation", not bad_sc and set(sc) == set(cfg_ids), str(bad_sc[:3]))
    ll_order = sorted(cfg_ids, key=lambda c: (exp[c]["log_loss_w"], c))
    auc_order = sorted(cfg_ids, key=lambda c: (-exp[c]["auc"], c))
    rk_ok = all(int(sc[c]["rank_log_loss"]) == ll_order.index(c) + 1 and int(sc[c]["rank_auc"]) == auc_order.index(c) + 1 for c in cfg_ids)
    dflt = next(i for i, (a, b) in zip(cfg_ids, declared_grid) if a == 0.05 and b == "sqrt")
    check("the ranks by log loss and by AUC are right, the chosen config is the lowest log loss (first on a tie), and the recorded default is the pipeline's setting (0.05, sqrt)",
          rk_ok and extra["chosen_config"] == ll_order[0] and extra["default_config"] == dflt, f"chosen {extra['chosen_config']} expected {ll_order[0]}")

    wp = defaultdict(dict)
    for r in V.read(d / "wf_predictions.csv"):
        wp[(r["arm"], int(r["seed"]))][int(r["event"])] = float(r["p"])
    arms = [("default", dflt)] + ([("chosen", extra["chosen_config"])] if extra["chosen_config"] != dflt else [])
    exp_keys = {(a, s) for a, _ in arms for s in extra["seeds"]}
    check("post-cutoff predictions exist for every (arm, seed) and cover exactly the events starting on or after the cutoff", set(wp) == exp_keys and all(sorted(wp[k]) == test_wf for k in wp), f"{sorted(set(wp) ^ exp_keys)[:3]}")
    wm = {(r["arm"], int(r["seed"])): r for r in V.read(d / "wf_metrics.csv")}
    bad_w = []
    for k in exp_keys:
        pe = wp.get(k, {})
        if not pe:
            continue
        evs = sorted(pe)
        y, w, pp = [events["y"][e] for e in evs], [events["w"][e] for e in evs], [pe[e] for e in evs]
        m = per_event_eval(pe, events, cut, step, cost_frac, windows)
        r = wm[k]
        chk = [("wf_auc", auc(y, pp)), ("wf_log_loss_w", wll(y, pp, w)), ("n_test", len(evs)), ("n_train", len(pre)), ("meta_sr_annual", m["meta"]["sr_annual"]), ("meta_bp", m["meta"]["bp"]),
               ("primary_sr_annual", m["primary"]["sr_annual"]), ("taken_share", m["meta"]["taken"]), ("mean_size", m["meta"]["mean_size"]),
               ("june_meta_bp", m["meta"]["win"]["june_2026_ai_correction"]), ("selloff_meta_bp", m["meta"]["win"]["sell_off_post_holdout"])]
        for col, e_ in chk:
            if not close(V.num(r[col]), float(e_), 1e-6, 1e-7):
                bad_w.append((k, col, r[col], e_))
    check("wf_metrics.csv: AUC, weighted log loss, event counts and, for the meta and primary variants, annualised Sharpe, mean bp, share taken, mean size and the June 2026 / sell-off window means equal the recomputation "
          "(placebo-based columns are not recomputed)", not bad_w and set(wm) == exp_keys, str(bad_w[:3]))
    src_seed = int(p["seed"])
    if not extra["quick"] and ("default", src_seed) in wp:
        srcp = {int(r["event"]): float(r["p"]) for r in V.read(run / "walk_forward_predictions.csv")}
        dd = wp[("default", src_seed)]
        ok = sorted(srcp) == sorted(dd) and max(abs(srcp[e] - dd[e]) for e in srcp) <= 1e-6
        check("the default config at the source run's own seed reproduces the source run's walk_forward_predictions.csv (max difference at most 1e-6) and the recorded gate says so",
              ok and extra["gate"].get("passed") is True, f"{extra['gate']}")
    else:
        check("gate against the source run's walk-forward predictions was applicable (not a --quick run and the source seed is among the seeds)", extra["quick"] and allow_quick)

    md = (d / "hp_summary.md").read_text(encoding="utf-8")
    bad_m = []
    for c in cfg_ids:
        m = re.search(rf"^\| {c}(?: \(default\))?(?: \(chosen\))? \| (\S+) \| (\S+) \| (\S+) \| (\d+) \| (\S+) \| (\d+) \| (\S+) \|$", md, re.M)
        if not m or not close(float(m.group(3)), exp[c]["log_loss_w"], 0, 6e-5) or int(m.group(4)) != ll_order.index(c) + 1 or not close(float(m.group(5)), exp[c]["auc"], 0, 6e-5) \
                or int(m.group(6)) != auc_order.index(c) + 1 or not close(float(m.group(7)), exp[c]["share_p_gt_half"], 0, 6e-4):
            bad_m.append(c)
    cols = ["wf_auc", "wf_log_loss_w", "meta_sr_annual", "meta_bp", "meta_minus_placebo_sr_daily", "taken_share", "june_meta_bp", "june_meta_minus_placebo_bp", "selloff_meta_bp", "selloff_meta_minus_placebo_bp"]
    for col in cols:
        m = re.search(rf"^\| {col} \| (\S+) \((\S+)\) \| (\S+)(?: \((\S+)\))? \| (\S+) \| (\S+) \| (\S+)(?: of (\d+))? \|$", md, re.M)
        if not m:
            bad_m.append((col, "row missing"))
            continue
        dv = [V.num(wm[("default", s)][col]) for s in extra["seeds"]]
        dv_f = [x for x in dv if not math.isnan(x)]
        if not close(float(m.group(1)), mean(dv_f), 0, 6e-5) or not close(float(m.group(2)), sd(dv_f), 0, 6e-5):
            bad_m.append((col, "default cells"))
        if ("chosen", extra["seeds"][0]) in wm:
            cv = [V.num(wm[("chosen", s)][col]) for s in extra["seeds"]]
            diff = [c_ - d_ for c_, d_ in zip(cv, dv) if not (math.isnan(c_) or math.isnan(d_))]
            if not close(float(m.group(3)), mean([c_ for c_ in cv if not math.isnan(c_)]), 0, 6e-5) or not close(float(m.group(5)), mean(diff), 0, 6e-5) or not close(float(m.group(6)), sd(diff), 0, 6e-5) \
                    or int(m.group(7)) != sum(1 for x in diff if x > 0) or int(m.group(8)) != len(diff):
                bad_m.append((col, "chosen cells"))
    check("hp_summary.md: every cell of the config table and of the post-cutoff table (means, sds, differences, counts) equals the recomputation from the checked files", not bad_m, str(bad_m[:3]))

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
    n_ok = sum(1 for c in checks if c[1])
    print(f"\n{n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    aq = "--allow-quick" in argv
    argv = [a for a in argv if a != "--allow-quick"]
    try:
        raise SystemExit(main(argv[0], aq))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, FileNotFoundError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
