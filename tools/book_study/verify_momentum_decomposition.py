#!/usr/bin/env python
"""Independent checker for a run_momentum_decomposition.py output folder (stdlib only; imports only helper functions of verify_book_study.py).

Recomputes from events.csv, with its own code: every row of summary_numbers.json (counts, pooled / leg / always-long / winners-minus-losers means,
net, best-20-days and top-5-symbols shares, symbols-positive share), the by-year table and the window lines; the bootstrap intervals (own RNG,
fewer draws, approximate match) for a sample of rows; and from the raw candle files (cut at each symbol's recorded last bar): the CUSUM events,
sigma, the trailing-return signs, the forward returns of every event of the checked symbols, and the market state of every event.
Exit code 0 = all checks pass.

    python tools/book_study/verify_momentum_decomposition.py runtime/book_study_decomposition/<timestamp> [--symbols N] [--ci-rows N]
"""
from __future__ import annotations

import bisect
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import verify_book_study as V


def mean(x):
    return sum(x) / len(x) if x else math.nan


def stats(r, s):
    up = [a for a, b in zip(r, s) if b > 0]
    dn = [a for a, b in zip(r, s) if b < 0]
    pooled = mean([a * b for a, b in zip(r, s)])
    return {"pooled": pooled, "long": mean(up), "short": -mean(dn) if dn else math.nan, "al": mean(r),
            "spread": (mean(up) - mean(dn)) if up and dn else math.nan}


def main(folder: str, n_symbols: int, ci_rows: int) -> int:
    d = Path(folder)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    summ = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    meta, rows, by_year, win_rows = summ["meta"], summ["rows"], summ["by_year"], summ["window_rows"]
    integrity = json.loads((d / "integrity.json").read_text(encoding="utf-8"))
    ev = V.read(d / "events.csv")
    symbols = V.read(d / "symbols.csv")
    cost_bp = meta["cost_frac"] * 1e4
    lookbacks, horizons = meta["lookbacks"], meta["horizons"]

    check("models and calibration files untouched during the run", integrity.get("unchanged") is True)
    check("events.csv length equals meta.events and equals the sum of kept events in symbols.csv",
          len(ev) == meta["events"] == sum(int(r["events_kept"]) for r in symbols), f"{len(ev)} / {meta['events']} / {sum(int(r['events_kept']) for r in symbols)}")
    t0 = [int(float(e["t0_epoch"])) for e in ev]
    check("events are sorted by time", all(a <= b for a, b in zip(t0, t0[1:])))
    state = [int(float(e["state"])) for e in ev]
    sym = [e["symbol"] for e in ev]
    year = [time.gmtime(t).tm_year for t in t0]
    day = [t // 86400 for t in t0]
    sig = {L: [int(float(e[f"sig_mom{L}"])) for e in ev] for L in lookbacks}
    ret = {h: [float(e[f"ret_h{h}_bp"]) for e in ev] for h in horizons}
    state_ok = {"all": lambda s: True, "above": lambda s: s == 1, "below": lambda s: s == 0}

    # ---- 1. every row ----
    expected_keys = set()
    bad_rows, bad_fields = [], []
    row_by_key = {(r["lookback_bars"], r["horizon"], r["state"]): r for r in rows}
    for L in lookbacks:
        for h in horizons:
            for sname, ok in state_ok.items():
                idx = [i for i in range(len(ev)) if ok(state[i]) and sig[L][i] != 0]
                key = (L, h, sname)
                if len(idx) < 50:
                    if key in row_by_key:
                        bad_rows.append((key, "should be absent"))
                    continue
                expected_keys.add(key)
                r = row_by_key.get(key)
                if r is None:
                    bad_rows.append((key, "missing"))
                    continue
                rr = [ret[h][i] for i in idx]
                ss = [sig[L][i] for i in idx]
                st = stats(rr, ss)
                pooled_list = [a * b for a, b in zip(rr, ss)]
                per_day = defaultdict(float)
                per_sym = defaultdict(float)
                per_sym_n = defaultdict(int)
                for i, p in zip(idx, pooled_list):
                    per_day[day[i]] += p
                    per_sym[sym[i]] += p
                    per_sym_n[sym[i]] += 1
                tot = sum(per_day.values())
                best20 = sum(sorted(per_day.values(), reverse=True)[:20])
                top5 = sum(sorted(per_sym.values(), reverse=True)[:5])
                want = {"events": len(idx), "share_long_signal": sum(1 for x in ss if x > 0) / len(ss), "days": len(per_day), "symbols": len(per_sym),
                        "pooled_gross_bp": st["pooled"], "pooled_net_bp": st["pooled"] - cost_bp, "long_leg_bp": st["long"], "short_leg_bp": st["short"],
                        "always_long_bp": st["al"], "spread_bp": st["spread"],
                        "best20_days_share_of_total": best20 / tot if tot > 0 else math.nan,
                        "top5_symbols_share_of_total": top5 / tot if tot > 0 else math.nan,
                        "symbols_positive_share": sum(1 for k_ in per_sym if per_sym[k_] / per_sym_n[k_] > 0) / len(per_sym),
                        "total_gross_bp_events": tot, "block_days": math.ceil((h + 1) / 24.0),
                        "blocks": len({day[i] // math.ceil((h + 1) / 24.0) for i in idx})}
                for f_, v in want.items():
                    if not V.same(float(r[f_]), float(v), 1e-6, 1e-6):
                        bad_fields.append((key, f_, r[f_], v))
    check("rows present exactly where an (lookback, horizon, state) group has at least 50 signalled events", not bad_rows and len(rows) == len(expected_keys), str(bad_rows[:3]))
    check("every row's counts, means, legs, spread, net, concentration shares recomputed from events.csv", not bad_fields, f"{len(bad_fields)} differ; first: {bad_fields[:3]}")
    check("net = pooled gross minus the round-trip cost in every row", all(V.same(r["pooled_net_bp"], r["pooled_gross_bp"] - cost_bp, 1e-9, 1e-9) for r in rows))
    check("intervals contain their point estimates", all(r["pooled_lo"] <= r["pooled_gross_bp"] <= r["pooled_hi"] and r["always_long_lo"] <= r["always_long_bp"] <= r["always_long_hi"]
                                                        and (math.isnan(r["spread_bp"]) or r["spread_lo"] <= r["spread_bp"] <= r["spread_hi"]) for r in rows))

    # ---- 2. identity: pooled = long-weighted legs; spread = long mean minus short-side mean ----
    ident_bad = []
    for r in rows:
        w = r["share_long_signal"]
        if not V.same(r["pooled_gross_bp"], w * r["long_leg_bp"] + (1 - w) * r["short_leg_bp"], 1e-6, 1e-6):
            ident_bad.append((r["lookback_bars"], r["horizon"], r["state"]))
    check("pooled gross equals the long-share-weighted average of the two legs", not ident_bad, str(ident_bad[:3]))

    # ---- 3. by-year ----
    bad_y = []
    n_y_expected = 0
    for L in lookbacks:
        for h in horizons:
            for sname, ok in state_ok.items():
                if (L, h, sname) not in expected_keys:
                    continue
                for yv in sorted({year[i] for i in range(len(ev)) if ok(state[i]) and sig[L][i] != 0}):
                    idx = [i for i in range(len(ev)) if ok(state[i]) and sig[L][i] != 0 and year[i] == yv]
                    n_y_expected += 1
                    st = stats([ret[h][i] for i in idx], [sig[L][i] for i in idx])
                    got = [b for b in by_year if b["lookback_bars"] == L and b["horizon"] == h and b["state"] == sname and b["year"] == yv]
                    if len(got) != 1:
                        bad_y.append((L, h, sname, yv, "row count", len(got)))
                        continue
                    g = got[0]
                    for f_, v in (("events", len(idx)), ("pooled_gross_bp", st["pooled"]), ("long_leg_bp", st["long"]), ("short_leg_bp", st["short"]),
                                  ("always_long_bp", st["al"]), ("spread_bp", st["spread"])):
                        if not V.same(float(g[f_]), float(v), 1e-6, 1e-6):
                            bad_y.append((L, h, sname, yv, f_, g[f_], v))
    check("by-year table recomputed (all lookbacks, horizons, states, years)", not bad_y and len(by_year) == n_y_expected, f"{len(by_year)} vs {n_y_expected}; {bad_y[:3]}")

    # ---- 4. windows ----
    check("June 2026 window is the calendar month and a sell-off window is reported",
          meta["windows"].get("june_2026_ai_correction", [None])[0].startswith("2026-06-01") and meta["windows"]["june_2026_ai_correction"][1].startswith("2026-07-01")
          and any(k_.startswith("sell_off") for k_ in meta["windows"]), str(meta["windows"]))
    bad_w = []
    for wname, (w0, w1) in meta["windows"].items():
        lo_t, hi_t = V.ts_key(w0), V.ts_key(w1)
        for L in lookbacks:
            for h in horizons:
                idx = [i for i in range(len(ev)) if lo_t <= t0[i] < hi_t and sig[L][i] != 0]
                got = [x for x in win_rows if x["lookback_bars"] == L and x["horizon"] == h and x["window"] == wname]
                if len(got) != 1 or got[0]["events"] != len(idx):
                    bad_w.append((wname, L, h, "events", len(got), len(idx)))
                    continue
                if idx:
                    st = stats([ret[h][i] for i in idx], [sig[L][i] for i in idx])
                    g = got[0]
                    for f_, v in (("pooled_gross_bp", st["pooled"]), ("pooled_net_bp", st["pooled"] - cost_bp), ("long_leg_bp", st["long"]),
                                  ("short_leg_bp", st["short"]), ("always_long_bp", st["al"]), ("spread_bp", st["spread"]), ("days", len({day[i] for i in idx}))):
                        if not V.same(float(g[f_]), float(v), 1e-6, 1e-6):
                            bad_w.append((wname, L, h, f_, g[f_], v))
    check("window lines (June 2026, sell-off, deepest fall) recomputed from events.csv", not bad_w, str(bad_w[:4]))

    # ---- 5. bootstrap intervals, own RNG ----
    def boot(vals, blocks, nb, rng, mode):
        ub = sorted(set(blocks))
        pos = {b: i for i, b in enumerate(ub)}
        sa, na, sb, nb_ = [0.0] * len(ub), [0] * len(ub), [0.0] * len(ub), [0] * len(ub)
        for v, b, side in zip(vals, blocks, mode):
            j = pos[b]
            if side:
                sa[j] += v
                na[j] += 1
            else:
                sb[j] += v
                nb_[j] += 1
        out = []
        for _ in range(nb):
            ia = ib = 0.0
            ca = cb = 0
            for _k in range(len(ub)):
                j = rng.randrange(len(ub))
                ia += sa[j]
                ca += na[j]
                ib += sb[j]
                cb += nb_[j]
            out.append(ia / ca if ca else math.nan)
        return sorted(x for x in out if not math.isnan(x))

    rng = random.Random(20261013)
    sample = [r for r in rows if r["state"] == "above" and r["horizon"] == max(horizons)][:max(1, ci_rows // 2)] + \
             [r for r in rows if r["state"] == "all" and r["horizon"] == min(horizons)][:max(1, ci_rows - ci_rows // 2)]
    ci_bad = []
    for r in sample:
        L, h, sname = r["lookback_bars"], r["horizon"], r["state"]
        bd = math.ceil((h + 1) / 24.0)
        idx = [i for i in range(len(ev)) if state_ok[sname](state[i]) and sig[L][i] != 0]
        vals = [ret[h][i] * sig[L][i] for i in idx]
        blocks = [day[i] // bd for i in idx]
        dist = boot(vals, blocks, 600, rng, [True] * len(idx))
        lo, hi = dist[int(0.025 * len(dist))], dist[int(0.975 * len(dist)) - 1]
        width = r["pooled_hi"] - r["pooled_lo"]
        if abs(lo - r["pooled_lo"]) > 0.25 * width or abs(hi - r["pooled_hi"]) > 0.25 * width:
            ci_bad.append((L, h, sname, "pooled", (round(lo, 1), round(hi, 1)), (round(r["pooled_lo"], 1), round(r["pooled_hi"], 1))))
    check(f"pooled-mean intervals match an independent bootstrap on {len(sample)} sampled rows (within a quarter of the width)", not ci_bad, str(ci_bad[:3]))

    # ---- 6. raw candles ----
    gmx_dir, tv_dir = meta["gmx_dir"], meta["tradingview_dir"]
    rows_by_sym = {r["symbol"]: r for r in symbols}
    span = meta["span"]
    h_max = max(horizons)
    cache = {}

    def series(s):
        if s not in cache:
            keys, closes = V.load_raw_close(gmx_dir, tv_dir, s)
            cut = V.ts_key(rows_by_sym[s]["last"])
            n_keep = bisect.bisect_right(keys, cut)
            keys, closes = keys[:n_keep], closes[:n_keep]
            logc = [math.log(c) for c in closes]
            rr = [math.nan] + [logc[i] - logc[i - 1] for i in range(1, len(logc))]
            cache[s] = (keys, logc, rr)
        return cache[s]

    ev_by_sym = defaultdict(list)
    for i in range(len(ev)):
        ev_by_sym[sym[i]].append(i)
    pick = sorted(ev_by_sym, key=lambda s: -len(ev_by_sym[s]))[:n_symbols]
    for s in pick:
        keys, logc, rr = series(s)
        sg = V.ewm_sigma(rr, span)
        all_ev = V.cusum(rr, sg)
        info = rows_by_sym[s]
        kept = sorted(p for p in all_ev if p + h_max + 1 < len(keys) and sg[p] > 0 and (keys[p + h_max + 1] - keys[p]) == (h_max + 1) * 3600)
        got = sorted(int(float(ev[i]["pos"])) for i in ev_by_sym[s])
        check(f"{s}: bar count and CUSUM total replayed from raw candles", len(keys) == int(info["bars"]) and len(all_ev) == int(info["events_total"]),
              f"{len(keys)} vs {info['bars']}, {len(all_ev)} vs {info['events_total']}")
        check(f"{s}: kept events equal events.csv", kept == got, f"{len(kept)} vs {len(got)}")
        bad = defaultdict(list)
        for i in ev_by_sym[s]:
            p = int(float(ev[i]["pos"]))
            if p >= len(keys):
                bad["pos"].append(i)
                continue
            if int(keys[p]) != t0[i]:
                bad["time"].append(i)
            if not V.same(sg[p], float(ev[i]["sigma"]), 1e-6, 1e-12):
                bad["sigma"].append(i)
            for L in lookbacks:
                want = 0 if p < L else (1 if logc[p] - logc[p - L] > 0 else -1 if logc[p] - logc[p - L] < 0 else 0)
                if want != sig[L][i]:
                    bad[f"mom{L}"].append(i)
            for h in horizons:
                want = (logc[p + h + 1] - logc[p]) * 1e4
                if not V.same(want, ret[h][i], 1e-4, 1e-3):
                    bad[f"ret{h}"].append(i)
        check(f"{s}: time, sigma, every trailing-return sign and every forward return recomputed from raw candles for all {len(ev_by_sym[s])} events",
              not bad, str({k_: len(v) for k_, v in bad.items()}))

    # market state for every event
    hour_sum, hour_cnt = defaultdict(float), defaultdict(int)
    for s in sorted(rows_by_sym):
        keys, logc, rr = series(s)
        for i in range(1, len(keys)):
            hour_sum[keys[i]] += rr[i]
            hour_cnt[keys[i]] += 1
    hours = sorted(t for t in hour_cnt if hour_cnt[t] >= 10)
    cum, level, csum = 0.0, [], [0.0]
    for t in hours:
        cum += hour_sum[t] / hour_cnt[t]
        v = math.exp(cum)
        level.append(v)
        csum.append(csum[-1] + v)

    def state_at(t):
        i = bisect.bisect_left(hours, t)
        if i >= len(hours) or hours[i] != t:
            return -1
        lo = bisect.bisect_right(hours, t - 200 * 86400)
        n = i + 1 - lo
        if n < 3840:
            return -1
        return 1 if level[i] > (csum[i + 1] - csum[lo]) / n else 0

    st_bad = sum(1 for i in range(len(ev)) if state_at(float(t0[i])) != state[i])
    check("market state recomputed from raw candles for every event", st_bad == 0, f"{st_bad} of {len(ev)} differ")

    # ---- report ----
    width = max(len(c[0]) for c in checks)
    n_fail = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
        n_fail += (not ok)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    ns, cr = 3, 6
    if "--symbols" in argv:
        i = argv.index("--symbols"); ns = int(argv[i + 1]); del argv[i:i + 2]
    if "--ci-rows" in argv:
        i = argv.index("--ci-rows"); cr = int(argv[i + 1]); del argv[i:i + 2]
    if not argv:
        raise SystemExit(__doc__)
    raise SystemExit(main(argv[0], ns, cr))
