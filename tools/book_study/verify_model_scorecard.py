#!/usr/bin/env python
"""Independent checker for a model-scorecard output folder (stdlib only; imports only the helper functions of verify_book_study.py).

Re-derives from scored_events.csv and the raw candle files: the flagged / not-flagged split, every group mean and lift in
summary_numbers.json, the day-cluster bootstrap interval (own RNG, approximate match), the window lines, PSR / expected-maximum Sharpe,
segment labels against each symbol's holdout bounds, the stop/profit rules against the control run, and - for a sample of events -
the price position, sigma and barrier exit from the raw candles. Exit code 0 = all pass.

    python tools/book_study/verify_model_scorecard.py runtime/book_study_models/<timestamp> [--sample N]
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import verify_book_study as V


def main(folder: str, n_sample: int) -> int:
    d = Path(folder)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    summ = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    meta, rows, trials = summ["meta"], summ["rows"], summ["trials"]
    ev = V.read(d / "scored_events.csv")
    syms = V.read(d / "symbols.csv")
    align = V.read(d / "alignment_check.csv") if (d / "alignment_check.csv").exists() and (d / "alignment_check.csv").stat().st_size > 5 else []
    integrity = json.loads((d / "integrity.json").read_text(encoding="utf-8"))
    cost_frac = meta["cost_frac"]
    horizons = meta["horizons"]

    check("models and calibration files untouched during the run", integrity.get("unchanged") is True)
    check("alignment check against the replay: at least one PASS and no FAIL", any(a["status"] == "PASS" for a in align) and not any(a["status"] == "FAIL" for a in align),
          str([a for a in align if a["status"] != "PASS"][:3]))

    # ---- event table internal consistency ----
    sel_bad = sum(1 for e in ev if (V.num(e["p"]) >= V.num(e["thr"])) != (e["selected"] == "True"))
    check("flagged <=> probability >= threshold, for every event", sel_bad == 0, f"{sel_bad} wrong")
    net_bad = sum(1 for e in ev if not V.same(V.num(e["net_bp"]), V.num(e["gross_bp"]) - cost_frac * 1e4, 1e-7, 1e-6))
    check("net = gross - round-trip cost, for every event", net_bad == 0, f"{net_bad} wrong")
    by = defaultdict(list)
    for e in ev:
        by[(e["direction"], int(e["horizon"]), e["segment"])].append(e)
    check("one group per (direction, horizon, segment) and the same events across horizons",
          len(by) == len(rows) and len({len(v) for (dd, h, s), v in by.items() if (dd, s) == ("long", "holdout")}) <= 1)

    # symbols.csv vs events
    per_sym = defaultdict(lambda: [0, 0, 0])
    h0 = horizons[0]
    for e in ev:
        if int(e["horizon"]) == h0:
            k = (e["direction"], e["symbol"])
            per_sym[k][0] += 1
            per_sym[k][1] += (e["segment"] == "holdout")
            per_sym[k][2] += (e["selected"] == "True")
    bad = [(s["direction"], s["symbol"]) for s in syms
           if per_sym[(s["direction"], s["symbol"])] != [int(s["events_scored"]), int(s["events_holdout"]), int(s["flagged"])]]
    check("events per symbol / segment / flagged equal symbols.csv", not bad, str(bad[:3]))
    hb = {(s["direction"], s["symbol"]): (V.ts_key(s["holdout_start"]), V.ts_key(s["holdout_end"])) for s in syms}
    seg_bad = 0
    for e in ev:
        hs, he = hb[(e["direction"], e["symbol"])]
        t = float(e["t0_epoch"])
        seg_bad += not (t >= hs and ((e["segment"] == "holdout") == (t <= he)))
    check("every event lies inside its model's out-of-sample span and carries the right segment label", seg_bad == 0, f"{seg_bad} wrong")

    # ---- rows recomputed ----
    def mean(xs):
        return sum(xs) / len(xs) if xs else math.nan

    def boot(days, x, sel, n_boot, seed):
        u = sorted(set(days))
        ix = {dv: i for i, dv in enumerate(u)}
        ss, sn, cs, cn = [0.0] * len(u), [0] * len(u), [0.0] * len(u), [0] * len(u)
        for dv, xv, sv in zip(days, x, sel):
            i = ix[dv]
            if sv:
                ss[i] += xv
                sn[i] += 1
            else:
                cs[i] += xv
                cn[i] += 1
        rnd = random.Random(seed)
        draws = []
        for _ in range(n_boot):
            a = b = c = e_ = 0
            fa = fc = 0.0
            for _j in range(len(u)):
                i = rnd.randrange(len(u))
                fa += ss[i]; a += sn[i]; fc += cs[i]; c += cn[i]
            if a and c:
                draws.append(fa / a - fc / c)
        draws.sort()
        return draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]

    row_bad, ci_bad, win_bad = [], [], []
    daily_sr = []
    for r in rows:
        g = by[(r["direction"], r["horizon"], r["segment"])]
        sel = [e["selected"] == "True" for e in g]
        days = [int(float(e["t0_epoch"]) // 86400) for e in g]
        cols = {k: [V.num(e[k]) for e in g] for k in ("gross_bp", "net_bp", "hold_bp", "hold1_bp")}
        ok = r["n_population"] == len(g) and r["n_selected"] == sum(sel)
        for name, m in (("selected", sel), ("complement", [not s for s in sel]), ("population", [True] * len(g))):
            if any(m):
                for key, col in (("gross", "gross_bp"), ("net", "net_bp"), ("hold", "hold_bp"), ("hold1", "hold1_bp")):
                    ok = ok and V.same(mean([v for v, mm in zip(cols[col], m) if mm]), r[f"{name}_{key}_bp"], 1e-6, 1e-6)
        for key, col in (("gross", "gross_bp"), ("hold", "hold_bp"), ("hold1", "hold1_bp")):
            lift = mean([v for v, s in zip(cols[col], sel) if s]) - mean([v for v, s in zip(cols[col], sel) if not s])
            ok = ok and V.same(lift, r[f"lift_{key}_bp"], 1e-6, 1e-6)
            lo, hi = boot(days, cols[col], sel, 600, 7)
            width = max(hi - lo, 1e-9)
            if not (r[f"lift_{key}_lo"] <= r[f"lift_{key}_bp"] <= r[f"lift_{key}_hi"]) or abs(lo - r[f"lift_{key}_lo"]) > 0.3 * width + 0.5 or abs(hi - r[f"lift_{key}_hi"]) > 0.3 * width + 0.5:
                ci_bad.append((r["direction"], r["horizon"], r["segment"], key, (round(lo, 1), round(hi, 1)), (round(r[f"lift_{key}_lo"], 1), round(r[f"lift_{key}_hi"], 1))))
        if not ok:
            row_bad.append((r["direction"], r["horizon"], r["segment"]))
        # windows
        for wname, (w0, w1) in meta["windows"].items():
            lo_d, hi_d = int(V.ts_key(w0) // 86400), int(V.ts_key(w1) // 86400)
            inw = [lo_d <= dv < hi_d for dv in days]
            w = r["windows"][wname]
            exp_sel = [v for v, i, s in zip(cols["net_bp"], inw, sel) if i and s]
            exp_pop = [v for v, i in zip(cols["net_bp"], inw) if i]
            good = w["n_population"] == sum(inw) and w["n_selected"] == len(exp_sel)
            good = good and ((not exp_sel and w["selected_net_bp"] is None) or (exp_sel and V.same(mean(exp_sel), w["selected_net_bp"], 1e-6, 1e-6)))
            good = good and ((not exp_pop and w["population_net_bp"] is None) or (exp_pop and V.same(mean(exp_pop), w["population_net_bp"], 1e-6, 1e-6)))
            if not good:
                win_bad.append((r["direction"], r["horizon"], r["segment"], wname))
        # daily series of the flagged events
        if any(sel):
            dd = defaultdict(list)
            for dv, v, s in zip(days, cols["net_bp"], sel):
                if s:
                    dd[dv].append(v)
            series = [mean(v) for _, v in sorted(dd.items())]
            n, sr, sk, ku = V.moments(series)
            t_eff = n / max(1.0, (r["horizon"] + 1) / 24.0)
            p_ok = V.same(V.psr(sr, t_eff, sk, ku), r.get("psr_net", math.nan), 1e-6, 1e-9) if not math.isnan(sr) else True
            if not (p_ok and (math.isnan(sr) or V.same(sr, r["daily_sr_net"], 1e-6, 1e-9))):
                row_bad.append(("psr/sr", r["direction"], r["horizon"], r["segment"]))
            if not math.isnan(sr):
                daily_sr.append(sr)
    check("group sizes, group means and lift points recomputed from scored_events.csv", not row_bad, str(row_bad[:4]))
    check("bootstrap intervals bracket the lift and agree with an independent day-cluster bootstrap", not ci_bad, str(ci_bad[:3]))
    check("window lines recomputed", not win_bad, str(win_bad[:3]))
    check("trial count equals rows with a daily Sharpe", trials["n"] == len(daily_sr), f"{trials['n']} vs {len(daily_sr)}")
    if len(daily_sr) > 1:
        m = sum(daily_sr) / len(daily_sr)
        var = sum((x - m) ** 2 for x in daily_sr) / (len(daily_sr) - 1)
        nd, g = NormalDist(), 0.5772156649015329
        n_t = len(daily_sr)
        emax = math.sqrt(var) * ((1 - g) * nd.inv_cdf(1 - 1 / n_t) + g * nd.inv_cdf(1 - 1 / (n_t * math.e)))
        check("variance of trial Sharpe ratios and expected maximum Sharpe", V.same(var, trials["var_sr_net"], 1e-6, 1e-12) and V.same(emax, trials["expected_max_sr"], 1e-6, 1e-9),
              f"{var} {emax} vs {trials}")

    # ---- rules equal the control run's ----
    ctrl = Path(meta["control_run_used_for_rules"]) / "summary_numbers.json"
    if ctrl.exists():
        cj = json.loads(ctrl.read_text(encoding="utf-8"))
        bad = [(r["direction"], r["horizon"]) for r in rows if [float(x) for x in cj["rows"][f"H{r['horizon']}|all|{r['direction']}"]["modal_rule"]] != r["rule"]]
        check("stop/profit rules equal the control run's modal rules", not bad, str(bad))
    else:
        check("stop/profit rules equal the control run's modal rules", False, f"{ctrl} not found")

    # ---- raw candle checks on a sample ----
    gmx_dir, tv_dir, span = meta["gmx_dir"], meta["tradingview_dir"], meta["sigma_span"]
    cache, sig_cache = {}, {}

    def series(sym):
        if sym not in cache:
            keys, closes = V.load_raw_close(gmx_dir, tv_dir, sym)
            logc = [math.log(c) for c in closes]
            r = [math.nan] + [logc[i] - logc[i - 1] for i in range(1, len(logc))]
            cache[sym] = (keys, logc, r)
        return cache[sym]

    rnd = random.Random(5)
    pick = rnd.sample(ev, min(n_sample, len(ev)))
    bad_pos, bad_sig, bad_exit, bad_hold = [], [], [], []
    for e in pick:
        sym, pos, h = e["symbol"], int(e["pos"]), int(e["horizon"])
        keys, logc, r = series(sym)
        if int(keys[pos]) != int(float(e["t0_epoch"])):
            bad_pos.append((sym, pos))
            continue
        if sym not in sig_cache:
            sig_cache[sym] = V.ewm_sigma(r, span)
        sg = sig_cache[sym][pos]
        if not V.same(sg, float(e["sigma"]), 1e-7, 1e-12):
            bad_sig.append((sym, pos, sg, e["sigma"]))
        sign = 1.0 if e["direction"] == "long" else -1.0
        v, t, kind = V.literal_exit(logc, pos, sign, float(e["sigma"]), float(e["pt"]), float(e["sl"]), h)
        if not (V.same(v, float(e["exit_value"]), 1e-4, 1e-5) and t == int(e["exit_step"]) and kind == int(e["exit_kind"])):
            bad_exit.append((sym, pos, v, e["exit_value"], t, e["exit_step"], kind, e["exit_kind"]))
        hold = sign * (logc[pos + h + 1] - logc[pos]) * 1e4
        hold1 = sign * (logc[pos + 1] - logc[pos]) * 1e4
        if not (V.same(hold, float(e["hold_bp"]), 1e-4, 1e-3) and V.same(hold1, float(e["hold1_bp"]), 1e-4, 1e-3)
                and V.same(v * float(e["sigma"]) * 1e4, float(e["gross_bp"]), 1e-4, 1e-3)):
            bad_hold.append((sym, pos, hold, e["hold_bp"], hold1, e["hold1_bp"]))
    check(f"{len(pick)} sampled events: price position matches the candle at that bar", not bad_pos, str(bad_pos[:3]))
    check("sampled events: sigma recomputed from raw candles", not bad_sig, str(bad_sig[:3]))
    check("sampled events: barrier exit re-run with a literal loop", not bad_exit, str(bad_exit[:3]))
    check("sampled events: gross, hold-H and hold-1 basis points", not bad_hold, str(bad_hold[:3]))

    width = max(len(c[0]) for c in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {'' if ok else detail}")
        failed += (not ok)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:]]
    n = 400
    if "--sample" in argv:
        n = int(argv[argv.index("--sample") + 1])
        del argv[argv.index("--sample"):argv.index("--sample") + 2]
    if not argv:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(argv[0], n))
