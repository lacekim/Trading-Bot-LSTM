#!/usr/bin/env python
"""Independent checker for a run_regime_study.py output folder (stdlib only; imports helper functions of verify_book_study.py and
verify_meta_pipeline.py).

From the raw candle files (each symbol cut at the source run's last bar) it rebuilds, with its own code: the 20-, 30- and 40-day trailing returns of every
event and the MOM side; the equal-weight index (mean hourly log return of the symbols that have a return, hours with at least 10 symbols) and, for every
event, the index's lookback returns over 240, 360, 480, 720, 960, 1440 and 1920 hours.  From those and the source run's events it recomputes the three
MKT_DOWN definitions, every rule's side, every bet's result, the daily series, the whole-history statistics (Sharpe, skew, kurtosis, PSR, best-20-days share),
the before / after 2024-01-01 halves, each calendar year, the three windows (results, trades and the share of events with MKT_DOWN), the 200-day-state x
MKT_DOWN cells, the long / short split of every rule and the deflated-Sharpe tables, and compares them with the run's files, including every number
in SUMMARY.md cell by cell.  The source run's own events are trusted (they are checked by verify_meta_pipeline.py); the checker confirms the md5 of the
files the run used.
Exit code 0 = all checks pass.

    python tools/book_study/verify_regime_study.py runtime/book_regime/<timestamp> [source meta run folder]
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import verify_book_study as V
import verify_meta_pipeline as VM

HERE = Path(__file__).resolve().parent
NAN = math.nan
fin, close, mean = VM.fin, VM.close, VM.mean
BASELINES = ["long_all", "short_all", "mom_all", "mom_long_up"]
NEW = ["short_down", "short_down_mom", "long_up_notdown", "regime_rule", "regime_rule_tight", "regime_rule_wide"]
VARIANTS = BASELINES + NEW
SETS = {"base": (480, 720, 960), "tight": (240, 360, 480), "wide": (960, 1440, 1920)}
ALL_L = sorted({L for v in SETS.values() for L in v})
MOM_L = (480, 720, 960)
CUT = float(V.ts_key("2024-01-01"))


def fm(x, nd):
    return "n/a" if x is None or not fin(float(x)) else f"{float(x):.{nd}f}"


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
    meta, params, res = summ["meta"], summ["params"], summ["result"]
    integrity = json.loads((d / "integrity.json").read_text(encoding="utf-8"))
    src = Path(source_override) if source_override else Path(meta["source_folder"])
    out_ev = V.read(d / "events_regime.csv")
    daily_rows = V.read(d / "daily.csv")
    h, cost = int(params["h"]), float(params["cost_frac"])
    say(f"loaded run files: {len(out_ev):,} events")

    check("declared settings: 101-bar barrier, 24 bp round trip, cutoff 2024-01-01, stock lookbacks 480 / 720 / 960, index lookback sets base 480 / 720 / 960, tight 240 / 360 / 480, wide 960 / 1440 / 1920, ten variants, trials 88 -> 94",
          h == 100 and close(cost, 0.0024, 1e-12, 1e-12) and params["cutoff"] == "2024-01-01" and list(params["mom_lookbacks"]) == list(MOM_L)
          and {k: tuple(v) for k, v in params["lookback_sets"].items()} == SETS and params["variants"] == VARIANTS and params["new_variants"] == NEW
          and int(params["prior_trials_before"]) == 88 and int(params["trials_after"]) == 94, str(params))
    check("models and calibration files untouched during the run (integrity.json: before equals after, flag true, and summary_numbers.json agrees)",
          integrity.get("unchanged") is True and integrity.get("before") == integrity.get("after") and summ.get("integrity_unchanged") is True, str(integrity)[:300])
    changed = []
    for name, md5 in meta["script_md5"].items():
        p = HERE / name
        if p.exists() and hashlib.md5(p.read_bytes()).hexdigest() != md5:
            changed.append(name)
    check("the study scripts next to this checker are the ones that produced the run (md5 recorded in the run)", not changed, f"changed since the run: {changed}")
    if not (src / "events.csv").exists():
        check("source meta-pipeline run is present", False, str(src))
        return report("source run not found; pass its folder as the second argument")
    s_summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    check("the source run's events.csv and summary_numbers.json are the files this run used (md5), and the source run's window definitions are the ones used",
          hashlib.md5((src / "events.csv").read_bytes()).hexdigest() == meta["source_events_md5"] and hashlib.md5((src / "summary_numbers.json").read_bytes()).hexdigest() == meta["source_summary_md5"]
          and meta["window_epoch"] == s_summ["meta"]["window_epoch"] and s_summ["params"]["side_signal"] == "mom720" and int(s_summ["params"]["h"]) == 100 and close(float(s_summ["params"]["cost_frac"]), cost, 1e-12, 1e-12))
    win_epoch = {k_: (int(v[0]), int(v[1])) for k_, v in meta["window_epoch"].items()}
    check("windows: June 2026 = [2026-06-01, 2026-07-01) UTC, a sell-off window is present, every window starts and ends on a UTC day boundary",
          meta["windows"]["june_2026_ai_correction"][0].startswith("2026-06-01") and meta["windows"]["june_2026_ai_correction"][1].startswith("2026-07-01") and any(k_.startswith("sell_off") for k_ in win_epoch)
          and all(a % 86400 == 0 and b % 86400 == 0 and a < b for a, b in win_epoch.values()), str(win_epoch))

    ev = V.read(src / "events.csv")
    n_ev = len(ev)
    aligned = len(out_ev) == n_ev == int(meta["events"]) and all(a["symbol"] == b["symbol"] and int(float(a["pos"])) == int(float(b["pos"])) and int(float(a["t0"])) == int(float(b["t0"])) for a, b in zip(out_ev, ev))
    check("events_regime.csv holds the source events in the same order (symbol, bar position, entry time)", aligned)
    if not aligned:
        return report("events not aligned")
    sym_rows = V.read(src / "symbols.csv")
    rows_by_sym = {r["symbol"]: r for r in sym_rows}
    sym_pos = defaultdict(list)
    for i, e in enumerate(ev):
        sym_pos[e["symbol"]].append(i)

    # ---- raw candles: symbol trailing returns and the equal-weight index ----
    exp_ret = {L: [NAN] * n_ev for L in MOM_L}
    hour_sum, hour_cnt = defaultdict(float), defaultdict(int)
    bad_bars = []
    for s in sorted(rows_by_sym):
        keys, closes, _h, _l = VM.load_raw(meta["gmx_dir"], meta["tradingview_dir"], s, meta["timeframe"])
        n_keep = bisect.bisect_right(keys, V.ts_key(rows_by_sym[s]["last"]))
        if n_keep != int(rows_by_sym[s]["bars"]):
            bad_bars.append((s, n_keep, rows_by_sym[s]["bars"]))
            continue
        logc = [math.log(c) for c in closes[:n_keep]]
        for i in range(1, n_keep):
            hour_sum[keys[i]] += logc[i] - logc[i - 1]
            hour_cnt[keys[i]] += 1
        for i in sym_pos.get(s, []):
            p = int(float(ev[i]["pos"]))
            for L in MOM_L:
                if p >= L:
                    exp_ret[L][i] = logc[p] - logc[p - L]
    check("raw candles cut at the source run's last bar have exactly the bar counts of the source run", not bad_bars, str(bad_bars[:3]))
    say("read raw candles")
    hours = sorted(t for t in hour_cnt if hour_cnt[t] >= 10)
    lvl, cum = [], 0.0
    for t in hours:
        cum += hour_sum[t] / hour_cnt[t]
        lvl.append(cum)                                           # log of the index level
    t0 = [int(float(e["t0"])) for e in ev]
    exp_idx = {L: [NAN] * n_ev for L in ALL_L}
    for i, t in enumerate(t0):
        now = bisect.bisect_right(hours, t) - 1
        if now < 0:
            continue
        for L in ALL_L:
            past = bisect.bisect_right(hours, t - L * 3600) - 1
            if past >= 0:
                exp_idx[L][i] = lvl[now] - lvl[past]
    bad_i = []
    for L in ALL_L:
        col = f"idxret{L}"
        for i, r in enumerate(out_ev):
            if not close(float(r[col]) if r[col] != "" else NAN, exp_idx[L][i], 1e-8, 1e-10):
                bad_i.append((i, col, r[col], exp_idx[L][i]))
                break
    check("events_regime.csv: the equal-weight index rebuilt from the raw candles gives every event's index lookback returns (240, 360, 480, 720, 960, 1440, 1920 hours)", not bad_i, str(bad_i[:3]))
    side_ev = [int(float(e["side"])) for e in ev]
    ret_side = [float(e["ret_side"]) for e in ev]
    state = [float(e["mkt_state"]) for e in ev]
    mom = [(1 if all(exp_ret[L][i] > 0 for L in MOM_L) else (-1 if all(exp_ret[L][i] < 0 for L in MOM_L) else 0)) for i in range(n_ev)]
    ret_long = [side_ev[i] * ret_side[i] for i in range(n_ev)]
    check("events_regime.csv: MOM side (20-, 30-, 40-day returns all > 0 / all < 0), long return = side x return, and the 200-day state equal the independent values",
          [int(float(r["mom_side"])) for r in out_ev] == mom and all(close(float(r["ret_long"]), ret_long[i], 1e-9, 1e-12) for i, r in enumerate(out_ev)) and [float(r["mkt_state"]) for r in out_ev] == state
          and all(s in (0.0, 1.0) for s in state))

    # ---- rules ----
    down = {k: [all(fin(exp_idx[L][i]) and exp_idx[L][i] < 0 for L in Ls) for i in range(n_ev)] for k, Ls in SETS.items()}

    def side_of(v, i):
        m, s = mom[i], state[i] == 1.0
        dA = down["base"][i]
        if v == "long_all":
            return 1
        if v == "short_all":
            return -1
        if v == "mom_all":
            return m
        if v == "mom_long_up":
            return 1 if (m == 1 and s) else 0
        if v == "short_down":
            return -1 if dA else 0
        if v == "short_down_mom":
            return -1 if (dA and m == -1) else 0
        if v == "long_up_notdown":
            return 1 if (m == 1 and s and not dA) else 0
        dd = down[{"regime_rule": "base", "regime_rule_tight": "tight", "regime_rule_wide": "wide"}[v]][i]
        if m == 1 and s and not dd:
            return 1
        if m == -1 and dd:
            return -1
        return 0

    day = [t // 86400 for t in t0]
    year = [time.gmtime(t).tm_year for t in t0]
    t_eff_div = max(1.0, (h + 1) / 24.0)

    def best20(xs):
        tot = sum(xs)
        return NAN if (not tot > 0 or not xs) else sum(sorted(xs)[-20:]) / tot

    def stats(idx_list, pnl):
        dd = defaultdict(lambda: [0.0, 0])
        for i in idx_list:
            c = dd[day[i]]
            c[0] += pnl[i]
            c[1] += 1
        xs = [dd[k_][0] / dd[k_][1] for k_ in sorted(dd)]
        st = VM.day_stats(xs, len(idx_list), t_eff_div)
        st.update({"events": len(idx_list), "mean_bp_per_candidate": (sum(pnl[i] for i in idx_list) / len(idx_list)) if idx_list else NAN, "best20_share": best20(xs)})
        return st, {k_: tuple(v) for k_, v in dd.items()}

    all_idx = list(range(n_ev))
    exp = {"stats": {}, "daily": {}, "windows": {}, "halves": {}, "years": {}, "cells": {}, "side_split": {}}
    for v in VARIANTS:
        sd = [side_of(v, i) for i in range(n_ev)]
        pnl = [(sd[i] * ret_long[i] - cost) * 1e4 if sd[i] != 0 else 0.0 for i in range(n_ev)]
        st, dd = stats(all_idx, pnl)
        st["taken_share"] = sum(1 for x in sd if x != 0) / n_ev
        exp["stats"][v], exp["daily"][v] = st, dd
        for wn, (lo, hi) in win_epoch.items():
            m = [i for i in all_idx if lo <= t0[i] < hi]
            exp["windows"][f"{v}|{wn}"] = {"events": len(m), "trades": sum(1 for i in m if sd[i] != 0), "mean_bp_per_candidate": (sum(pnl[i] for i in m) / len(m)) if m else NAN}
        for hn, m in (("before", [i for i in all_idx if t0[i] < CUT]), ("after", [i for i in all_idx if t0[i] >= CUT])):
            sth, _ = stats(m, pnl)
            sth["taken_share"] = sum(1 for i in m if sd[i] != 0) / len(m)
            exp["halves"][f"{v}|{hn}"] = sth
        for y in sorted(set(year)):
            m = [i for i in all_idx if year[i] == y]
            exp["years"][f"{v}|{y}"] = {"events": len(m), "trades": sum(1 for i in m if sd[i] != 0), "mean_bp_per_candidate": sum(pnl[i] for i in m) / len(m)}
        for s_ in (0, 1):
            for dn in (0, 1):
                m = [i for i in all_idx if state[i] == s_ and down["base"][i] == bool(dn)]
                exp["cells"][f"{v}|state{s_}|down{dn}"] = {"events": len(m), "trades": sum(1 for i in m if sd[i] != 0), "mean_bp_per_candidate": (sum(pnl[i] for i in m) / len(m)) if m else NAN}
        for sgn, nm in ((1, "long"), (-1, "short")):
            m = [i for i in all_idx if sd[i] == sgn]
            exp["side_split"][f"{v}|{nm}"] = {"trades": len(m), "bp_per_trade": (sum(pnl[i] for i in m) / len(m)) if m else NAN}
    say("recomputed every rule")

    cmp_fields = ("events", "days", "mean_bp_per_candidate", "sr_daily", "sr_annual", "skew", "kurt", "t_eff", "psr", "best20_share")
    bad = []
    for v in VARIANTS:
        got, want = res["stats"][v], exp["stats"][v]
        for f_ in cmp_fields + ("taken_share",):
            if not close(float(got[f_]), float(want[f_]), 1e-6, 1e-8):
                bad.append((v, f_, got[f_], want[f_]))
    check("whole-history statistics of the ten rules (events, days, bp per candidate, Sharpe, skew, kurtosis, effective sample, PSR, best-20-days share, share taken) recomputed", not bad, str(bad[:4]))
    bad = []
    for key, want in exp["halves"].items():
        got = res["halves"][key]
        for f_ in cmp_fields + ("taken_share",):
            if not close(float(got[f_]), float(want[f_]), 1e-6, 1e-8):
                bad.append((key, f_, got[f_], want[f_]))
    check("before / after 2024-01-01 statistics of the ten rules recomputed from the events", not bad and set(res["halves"]) == set(exp["halves"]), str(bad[:4]))
    bad = []
    for key, want in exp["years"].items():
        got = res["years"][key]
        if got["events"] != want["events"] or got["trades"] != want["trades"] or not close(float(got["mean_bp_per_candidate"]), want["mean_bp_per_candidate"], 1e-6, 1e-8):
            bad.append((key, got, want))
    check("calendar-year table of the ten rules recomputed", not bad and set(res["years"]) == set(exp["years"]), str(bad[:3]))
    bad = []
    for key, want in exp["windows"].items():
        got = res["windows"][key]
        if got["events"] != want["events"] or got["trades"] != want["trades"] or not close(float(got["mean_bp_per_candidate"]), want["mean_bp_per_candidate"], 1e-6, 1e-8):
            bad.append((key, got, want))
    ds = {}
    for wn, (lo, hi) in win_epoch.items():
        m = [i for i in all_idx if lo <= t0[i] < hi]
        ds[wn] = {k_: (sum(1 for i in m if down[k_][i]) / len(m)) if m else NAN for k_ in SETS}
        g = res["window_down_share"][wn]
        if g["events"] != len(m) or not all(close(float(g[k_]), ds[wn][k_], 1e-9, 1e-12) for k_ in SETS):
            bad.append((wn, "MKT_DOWN share", g, ds[wn]))
    check("window lines (June 2026, sell-off, deepest fall): result, trades and event count of every rule, and the share of events with MKT_DOWN under the three definitions, recomputed", not bad and set(res["windows"]) == set(exp["windows"]), str(bad[:3]))
    bad = []
    for key, want in exp["cells"].items():
        got = res["cells"][key]
        if got["events"] != want["events"] or got["trades"] != want["trades"] or not close(float(got["mean_bp_per_candidate"]), want["mean_bp_per_candidate"], 1e-6, 1e-8):
            bad.append((key, got, want))
    for key, want in exp["side_split"].items():
        got = res["side_split"][key]
        if got["trades"] != want["trades"] or not close(float(got["bp_per_trade"]), want["bp_per_trade"], 1e-6, 1e-8):
            bad.append((key, got, want))
    ce = {f"state{s_}|down{dn}": sum(1 for i in all_idx if state[i] == s_ and down["base"][i] == bool(dn)) for s_ in (0, 1) for dn in (0, 1)}
    check("200-day state x MKT_DOWN cells (events, bp per candidate), the long / short split of every rule, cell event counts and the MKT_DOWN shares of the whole history recomputed",
          not bad and res["cell_events"] == ce and all(close(float(res["down_share"][k_]), sum(down[k_]) / n_ev, 1e-9, 1e-12) for k_ in SETS), str(bad[:3]))
    # daily file
    dgot = defaultdict(dict)
    for r in daily_rows:
        dgot[r["variant"]][int(r["day"])] = (float(r["sum_bp"]), int(r["n"]))
    bad = [v for v in VARIANTS if set(dgot[v]) != set(exp["daily"][v]) or any(dgot[v][k_][1] != exp["daily"][v][k_][1] or not close(dgot[v][k_][0], exp["daily"][v][k_][0], 1e-7, 1e-6) for k_ in exp["daily"][v])]
    check("daily.csv: the daily series (sum of bp, events) of all ten rules recomputed", not bad and len(daily_rows) == sum(len(exp["daily"][v]) for v in VARIANTS), str(bad[:3]))

    # deflated Sharpe
    bad = []
    for v in NEW:
        st = exp["stats"][v]
        sr, t_eff, sk, ku = st["sr_daily"], st["t_eff"], st["skew"], st["kurt"]
        va = (1.0 - sk * sr + (ku - 1.0) / 4.0 * sr * sr) / max(t_eff - 1.0, 1.0) if fin(sr) else NAN
        rd = res["dsr"][v]
        if set(rd["by_trials"]) != {"1", "10", "88", "94"} or not close(float(rd["var_sr_used"]), va, 1e-6, 1e-9):
            bad.append((v, "head", rd["var_sr_used"], va))
            continue
        for n_tr in (1, 10, 88, 94):
            star = 0.0 if n_tr <= 1 else (VM.expected_max_sharpe(n_tr, va) if fin(va) and va >= 0 else NAN)
            dsr = V.psr(sr, t_eff, sk, ku, star) if fin(star) else NAN
            g = rd["by_trials"][str(n_tr)]
            if not (close(float(g["sr_star"]), star, 1e-6, 1e-9) and close(float(g["dsr"]), dsr, 1e-6, 1e-9)):
                bad.append((v, n_tr, g, (star, dsr)))
    check("deflated-Sharpe tables of the six new rules (N = 1, 10, 88, 94; variance = asymptotic sampling variance, one path) recomputed with own normal quantiles", not bad, str(bad[:3]))

    # ---- SUMMARY.md cell by cell ----
    md = (d / "SUMMARY.md").read_text(encoding="utf-8")
    lines = md.splitlines()
    found = set()
    for ln in lines:
        if ln.startswith("|"):
            found.add(tuple(c.strip() for c in ln.strip().strip("|").split("|")))
    expected = []
    for v in VARIANTS:
        s_ = res["stats"][v]
        expected.append((v, fm(s_["mean_bp_per_candidate"], 1), fm(s_["taken_share"], 2), fm(s_["sr_daily"], 4), fm(s_["sr_annual"], 2), fm(s_["psr"], 2), str(s_["days"]), fm(s_["best20_share"], 2)))
        a, b = res["halves"][f"{v}|before"], res["halves"][f"{v}|after"]
        expected.append((v, fm(a["mean_bp_per_candidate"], 1), fm(a["sr_daily"], 4), fm(a["sr_annual"], 2), fm(b["mean_bp_per_candidate"], 1), fm(b["sr_daily"], 4), fm(b["sr_annual"], 2)))
        l_, s2 = res["side_split"][f"{v}|long"], res["side_split"][f"{v}|short"]
        expected.append((v, fm(l_["bp_per_trade"], 1), str(l_["trades"]), fm(s2["bp_per_trade"], 1), str(s2["trades"])))
    for v in NEW:
        for n_tr, x in res["dsr"][v]["by_trials"].items():
            expected.append((n_tr, fm(x["sr_star"], 4), fm(x["dsr"], 3)))
    for wn in meta["windows"]:
        expected.append((wn, str(res["window_down_share"][wn]["events"])) + tuple(fm(res["windows"][f"{v}|{wn}"]["mean_bp_per_candidate"], 1) for v in VARIANTS))
        expected.append((wn,) + tuple(str(res["windows"][f"{v}|{wn}"]["trades"]) for v in VARIANTS))
        w = res["window_down_share"][wn]
        expected.append((wn, fm(w["base"], 3), fm(w["tight"], 3), fm(w["wide"], 3)))
    for s_ in (1, 0):
        for dn in (0, 1):
            expected.append((str(s_), str(dn), str(res["cell_events"][f"state{s_}|down{dn}"])) + tuple(fm(res["cells"][f"{v}|state{s_}|down{dn}"]["mean_bp_per_candidate"], 1) for v in VARIANTS))
    for y in sorted({k.split("|")[1] for k in res["years"]}):
        expected.append((y, str(res["years"][f"{VARIANTS[0]}|{y}"]["events"])) + tuple(fm(res["years"][f"{v}|{y}"]["mean_bp_per_candidate"], 1) for v in VARIANTS))
    miss = [e_ for e_ in expected if e_ not in found]
    head_exp = [f"- symbols: {len(set(e['symbol'] for e in ev))}", f"- events: {n_ev:,}", f"- round-trip cost (bp): {cost * 1e4:.1f}",
                "- index lookbacks (hours): " + " | ".join(f"{k_}: {v_}" for k_, v_ in SETS.items()), "- trials before / after this study: 88 / 94"]
    for hl in head_exp:
        if hl not in lines:
            miss.append(("header line", hl))
    ps = summ["params_shown"]
    if not (int(ps["symbols"]) == len(set(e["symbol"] for e in ev)) and ps["events"] == f"{n_ev:,}" and ps["round-trip cost (bp)"] == f"{cost * 1e4:.1f}"):
        miss.append(("params_shown", ps))
    ds_line = f"MKT_DOWN is true for {res['down_share']['base']:.1%} of events (tight {res['down_share']['tight']:.1%}, wide {res['down_share']['wide']:.1%})."
    if ds_line not in md:
        miss.append(("down share line", ds_line))
    check("SUMMARY.md: every table row (whole history, halves, long / short split, deflated Sharpe, windows, trades, MKT_DOWN shares, cells, years) equals the numbers in summary_numbers.json cell by cell; design text and deviations are present",
          not miss and "DESIGN" in md and "Deviations from the book" in md and "june_2026_ai_correction" in md and any("sell_off" in ln for ln in lines), str(miss[:3]))
    return report()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    raise SystemExit(main(argv[0], argv[1] if len(argv) > 1 else None))
