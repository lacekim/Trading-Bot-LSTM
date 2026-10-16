"""Builds verify_meta_bars.py from verify_meta_venue.py: the downstream checks (CPCV bookkeeping, predictions, metrics, evaluation, deflated Sharpe, windows, walk-forward, SUMMARY.md cells) are kept
byte for byte; the three places that read hourly candles are replaced by the dollar-bar versions below.  Run once; the result is the file that is delivered and used."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "verify_meta_venue.py").read_text(encoding="utf-8")


def cut(text, start_marker, end_marker, new, include_end=False):
    a = text.index(start_marker)
    b = text.index(end_marker, a)
    if text.count(start_marker) != 1 or text.count(end_marker) < 1:
        raise SystemExit(f"anchor problem: {start_marker!r}")
    if include_end:
        b += len(end_marker)
    return text[:a] + new + text[b:]


DOC = '''#!/usr/bin/env python
"""Independent checker for a run_meta_bars.py output folder (DOLLAR BARS of one venue; stdlib only; imports only helper functions of verify_book_study.py).
It is verify_meta_venue.py with the three places that read hourly candles replaced; everything that starts from events.csv (CPCV bookkeeping, predictions, metrics, evaluation,
deflated Sharpe, windows, walk-forward, SUMMARY.md cells) is the same code.

From the dollar-bar files (bars/<SYM>.csv and ref_1h/<SYM>.csv of the run's bar folder) it rebuilds, with its own plain-loop code, EVERY event the run should contain:
the CUSUM events (threshold = one-bar sigma x sqrt of the trailing bars per calendar day), point-in-time membership (Alpaca), the vertical-barrier window (time limit, bar cap, data holes),
the book's daily-volatility target (Snippet 3.1 on irregular bars), the trailing 720-bar side, the triple-barrier exit (literal loop, strict touches), the meta-label, uniqueness and
return-attribution weights, the fractional-differentiation d and feature, the volatility and trailing-return features (missing across a data hole), the Corwin-Schultz spread from the hourly
reference series, and the market state and distance from the hourly reference closes -- then compares the set of events and every column with events.csv.
Optional  --micro-sample N : the SADF and entropy columns (slow plain loops) are recomputed in full for about N evenly spaced events; for the others only that their window exists
is used (the finite/missing pattern is still compared for every event).  Without it all events are recomputed in full.
Exit code 0 = all checks pass.

    python tools/book_study/verify_meta_bars.py runtime/book_meta_<venue>_dollar/<timestamp> [--prior-trials N] [--micro-sample N]
"""
'''
out = src
# 1. docstring
a = out.index('"""', 0)
b = out.index('"""', a + 3) + 3
out = '#!/usr/bin/env python\n' + DOC.split('\n', 1)[1] + out[b + 1:] if False else DOC + out[b + 1:]

# 2. constants for the sample option
out = out.replace("EXTRA_CHECKS = None\n", "EXTRA_CHECKS = None\nMICRO_SAMPLE = 0                  # 0 = recompute SADF / entropy for every event\n", 1)

# 3. declared settings
SETTINGS_OLD = out[out.index('    check(f"declared settings: vertical barrier 101 bars'):out.index('    check("models and calibration files untouched during the run"')]
SETTINGS_NEW = '''    check(f"declared settings: vertical barrier {params['h_days']:g} days with a cap of {params['hcap']} bars, +/-2 daily-vol barriers, FFD threshold 0.01, side = 720-bar momentum, round-trip cost = 2 x cost per side, the {len(DECLARED_FEATURES) + len(EXTRA_FEATURES)}-feature list, SADF window 100 / min 20 / 1 lag, entropy 100 days / word 2, Corwin-Schultz 24 rows, bar-clock constants",
          h == int(round(float(params["h_days"]) * 24)) - 1 and [float(x) for x in ptsl] == [2.0, 2.0] and float(params["ffd_thres"]) == 0.01 and params["side_signal"] == "mom720"
          and close(cost, 2.0 * float(meta["cost_bp_per_side"]) / 1e4, 1e-12, 1e-12) and feats == DECLARED_FEATURES + EXTRA_FEATURES and params.get("micro") == DECLARED_MICRO
          and params["bars"]["span"] == SPAN and params["bars"]["slow_span"] == 2016 and params["bars"]["mkt_min_history_days"] == 160 and params["bars"]["state_tolerance_hours"] == 6
          and meta["timeframe"] == "dollar",
          str({k: params.get(k) for k in ("h", "h_days", "hcap", "ptsl", "ffd_thres", "side_signal", "cost_frac", "features", "micro", "bars")}))
'''
out = out.replace(SETTINGS_OLD, SETTINGS_NEW)

# 4. the data block
BLOCK = r'''    # ---- load the dollar bars of every symbol (own reader) ----
    import csv as _csv
    bars_dir = Path(meta["candle_dir"])
    bman = json.loads((bars_dir / "manifest.json").read_text(encoding="utf-8"))
    break_s = int(bman["break_hours"]) * 3600
    H_DAYS, HCAP = float(params["h_days"]), int(params["hcap"])
    H_S = int(round(H_DAYS * 86400))
    DAY = 86400
    rows_by_sym = {r["symbol"]: r for r in symbols}
    sym_list = sorted(rows_by_sym)
    check("symbols.csv lists the symbols once each and the run used that many", len(sym_list) == len(symbols) == int(meta["symbols_used"]), f"{len(sym_list)} {len(symbols)} {meta['symbols_used']}")
    check("the bar manifest is the one the run recorded (md5), and every bar / reference CSV used has the md5 written by the builder",
          hashlib.md5((bars_dir / "manifest.json").read_bytes()).hexdigest() == meta["bars_manifest_md5"]
          and all(hashlib.md5((bars_dir / "bars" / f"{s}.csv").read_bytes()).hexdigest() == bman["symbols"][s]["bars_md5"]
                  and hashlib.md5((bars_dir / "ref_1h" / f"{s}.csv").read_bytes()).hexdigest() == bman["symbols"][s]["ref_md5"] for s in sym_list),
          str(meta["bars_manifest_md5"]))

    def to_epoch(txt):
        return int(V.ts_key(txt))

    def read_cols(path, names):
        with open(path, newline="", encoding="utf-8") as fh:
            rd = _csv.reader(fh)
            head = next(rd)
            ix = [head.index(n) for n in names]
            cols = [[] for _ in names]
            for row in rd:
                for k_, j_ in enumerate(ix):
                    cols[k_].append(row[j_])
        return cols

    # point-in-time membership (Alpaca): the last month-end strictly before the event's UTC date
    member_sets, member_ends = {}, []
    if meta.get("universe"):
        up = Path(meta["universe"])
        up = up if up.is_absolute() else HERE.parents[1] / up
        for r_ in V.read(up):
            member_sets.setdefault(int(V.ts_key(r_["month_end"])), set()).add(r_["symbol"])
        member_ends = sorted(member_sets)

    def is_member(sym, t):
        if not member_ends:
            return True
        day0 = (int(t) // DAY) * DAY
        k = bisect.bisect_left(member_ends, day0) - 1
        return k >= 0 and sym in member_sets[member_ends[k]]

    series, ref, seg_of = {}, {}, {}
    for s in sym_list:
        tt, st, cl = read_cols(bars_dir / "bars" / f"{s}.csv", ["date", "start", "close"])
        keys = array("d", [to_epoch(x) for x in tt])
        starts = array("d", [to_epoch(x) for x in st])
        logc = array("d", [math.log(float(x)) for x in cl])
        series[s] = (keys, logc)
        n_ = len(keys)
        seg = array("l", [0]) * n_
        for i in range(1, n_):
            seg[i] = seg[i - 1] + (1 if starts[i] - keys[i - 1] > break_s else 0)
        seg_of[s] = seg
        rd_, rh_, rl_, rc_ = read_cols(bars_dir / "ref_1h" / f"{s}.csv", ["date", "high", "low", "close"])
        ref[s] = (array("d", [to_epoch(x) + 3600 for x in rd_]), [float(x) for x in rh_], [float(x) for x in rl_], array("d", [math.log(float(x)) for x in rc_]))
    say(f"read the dollar bars of {len(series)} symbols ({sum(len(v[0]) for v in series.values()):,} bars)")

    # market index (equal weight of hourly reference closes, hours with >= 10 symbols) and its 200-day mean
    hour_sum, hour_cnt = defaultdict(float), defaultdict(int)
    for s in sym_list:
        rk, _, _, rlc = ref[s]
        for i in range(1, len(rk)):
            hour_sum[rk[i]] += rlc[i] - rlc[i - 1]
            hour_cnt[rk[i]] += 1
    hours = sorted(t for t in hour_cnt if hour_cnt[t] >= 10)
    cum, level, csum = 0.0, [], [0.0]
    for t in hours:
        cum += hour_sum[t] / hour_cnt[t]
        v = math.exp(cum)
        level.append(v)
        csum.append(csum[-1] + v)

    def state_dist(t):
        i = bisect.bisect_right(hours, t) - 1
        if i < 0 or t - hours[i] > 6 * 3600:
            return -1, NAN
        if hours[i] - hours[0] < 160 * DAY:
            return -1, NAN
        lo = bisect.bisect_right(hours, hours[i] - 200 * DAY)
        n = i + 1 - lo
        ma = (csum[i + 1] - csum[lo]) / n
        return (1 if level[i] > ma else 0), math.log(level[i] / ma)

    # ---- rebuild every event ----
    exp_rows = []                        # dicts
    n_cusum = 0
    n_drop_side = n_drop_feat = n_drop_extra = 0
    ties = 0
    barrier_bad = []
    fd_exp = {}
    sym_info_bad = []
    uniq_cache = {}
    micro_checked = micro_full_n = 0
    n_events_guess = int(meta["events"])
    micro_step = max(1, n_events_guess // MICRO_SAMPLE) if MICRO_SAMPLE else 1
    ev_counter = 0
    for s in sym_list:
        keys, logc = series[s]
        seg = seg_of[s]
        n = len(keys)
        rr = [NAN] + [(logc[i] - logc[i - 1]) if seg[i] == seg[i - 1] else NAN for i in range(1, n)]
        sg = V.ewm_sigma(rr, SPAN)
        # bars per calendar day over the trailing min(30 days, history), counting bars with a stamp in (t - W, t]
        thr = [NAN] * n
        for i in range(1, n):
            w = min(30 * DAY, keys[i] - keys[0])
            if w <= 0 or not fin(sg[i]):
                continue
            lo = bisect.bisect_right(keys, keys[i] - w)
            thr[i] = sg[i] * math.sqrt((i + 1 - lo) / (w / DAY))
        evp = []
        sp = sn = 0.0
        for i in range(1, n):
            x = rr[i] if not math.isnan(rr[i]) else 0.0
            sp, sn = max(0.0, sp + x), min(0.0, sn + x)
            t_ = thr[i]
            if math.isnan(t_):
                continue
            if sn < -t_:
                sn = 0.0
                evp.append(i)
            elif sp > t_:
                sp = 0.0
                evp.append(i)
        mem = [p for p in evp if is_member(s, keys[p])]
        n_not_member = len(evp) - len(mem)
        n_sig = n_win = 0
        kept, kept_L, kept_cap = [], {}, {}
        for p in mem:
            if not (fin(sg[p]) and sg[p] > 0):
                n_sig += 1
                continue
            iv = bisect.bisect_left(keys, keys[p] + H_S)
            L_full = iv - p
            L = min(L_full, HCAP)
            if iv >= n or L < 1 or seg[min(p + L, n - 1)] != seg[p]:
                n_win += 1
                continue
            kept.append(p)
            kept_L[p] = L
            kept_cap[p] = 1 if L_full > HCAP else 0
        info = rows_by_sym[s]
        if not (n == int(info["bars"]) and len(evp) == int(info["events_total"]) and len(kept) == int(info["events_kept"]) and n_not_member == int(info["not_member"])
                and n_sig == int(info["dropped_sigma"]) and n_win == int(info["dropped_window"]) and sum(kept_cap.values()) == int(info["capped"])):
            sym_info_bad.append((s, n, info["bars"], len(evp), info["events_total"], len(kept), info["events_kept"], n_not_member, info["not_member"], n_sig, n_win, info["dropped_sigma"], info["dropped_window"]))
        n_cusum += len(kept)
        # the book's daily volatility (Snippet 3.1) on irregular bars: return from the last bar at or before t - 1 day; missing when that bar is older than the hole limit
        r1 = [NAN] * n
        for i in range(n):
            j = bisect.bisect_right(keys, keys[i] - DAY) - 1
            if j >= 0 and keys[i] - keys[j] <= break_s:
                r1[i] = math.exp(logc[i] - logc[j]) - 1.0
        trgt_all = V.ewm_sigma(r1, 100)
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
        rk_, rh_, rl_, _ = ref[s]
        csv_ = cs_py(rh_, rl_, CS_SL)
        bits_all = "".join("1" if x[k] - x[k - 1] > 0 else "0" for k in range(1, len(x)))      # bit of day k is bits_all[k-1]
        micro_cache = {}

        def micro_day(dd, full):
            """(sadf, plug-in, konto) at the daily index of UTC day number dd, computed on demand and only if its window exists.  full=False: NaN pattern only (values 0.0)."""
            key = (dd, full)
            if key in micro_cache:
                return micro_cache[key]
            k = dpos.get(dd)
            if k is None:
                v = (NAN, NAN, NAN)
            elif not full:
                v = (0.0 if k >= SADF_W - 1 else NAN, 0.0 if k >= ENT_W else NAN, 0.0 if k >= ENT_W else NAN)
            else:
                sd = sadf_py(x[k - SADF_W + 1:k + 1]) if k >= SADF_W - 1 else NAN
                if k >= ENT_W:
                    msg = bits_all[k - ENT_W:k]                       # returns of days k-ENT_W+1 .. k, oldest first
                    v = (sd, plugin_py(msg), konto_py(msg[::-1]))
                else:
                    v = (sd, NAN, NAN)
            micro_cache[key] = v
            return v

        extra_fn = EXTRA_PREP(meta["candle_dir"], s, keys) if EXTRA_PREP is not None else None
        sub = []
        for p in kept:
            side = 0
            if p >= 720 and seg[p - 720] == seg[p]:
                dl = logc[p] - logc[p - 720]
                side = 1 if dl > 0 else -1 if dl < 0 else 0
            trgt = trgt_all[p]
            ok0 = side != 0 and fin(trgt) and trgt > 0
            if not ok0:
                n_drop_side += 1
                continue
            row = {"symbol": s, "pos": p, "t0": int(keys[p]), "sigma": sg[p], "trgt": trgt, "side": side, "log_trgt": math.log(trgt), "n_fwd": kept_L[p], "capped": kept_cap[p]}
            for L, nm in ((24, "mom24n"), (168, "mom168n"), (720, "mom720n"), (2160, "mom2160n")):
                row[nm] = (logc[p] - logc[p - L]) / (sg[p] * math.sqrt(L)) if (p >= L and seg[p - L] == seg[p]) else NAN
            row["vol_ratio"] = math.log(sg[p] / slow[p]) if fin(slow[p]) and slow[p] > 0 else NAN
            row["ffd"] = fmap.get(int(keys[p] // 86400) - 1, NAN)
            ev_counter += 1
            full = (MICRO_SAMPLE == 0) or (ev_counter % micro_step == 0)
            row["sadf"], row["ent_plugin"], row["ent_konto"] = micro_day(int(keys[p] // 86400) - 1, full)
            row["_micro_full"] = full
            kk = bisect.bisect_right(rk_, keys[p]) - 1
            row["cs_spread"] = csv_[kk] if kk >= 0 and keys[p] - rk_[kk] <= 6 * 3600 else NAN
            st, dist = state_dist(float(keys[p]))
            row["mkt_state"] = float(st) if st >= 0 else NAN
            row["mkt_dist"] = dist
            # literal triple barrier, strict touches, simple returns times side, vertical barrier at the last followed bar
            pt, sl = ptsl[0] * trgt, -ptsl[1] * trgt
            Rs, jx, kd, rt = [], None, 0, None
            Lp = kept_L[p]
            for t in range(1, Lp + 1):
                R = (math.exp(logc[p + t] - logc[p]) - 1.0) * side
                Rs.append(R)
                if jx is None:
                    if R > pt:
                        jx, kd, rt = t, 1, R
                    elif R < sl:
                        jx, kd, rt = t, -1, R
            if jx is None:
                jx, kd, rt = Lp, 0, Rs[-1]
            row["_Rs"], row["_pt"], row["_sl"] = Rs, pt, sl
            row["exit_j"], row["kind"], row["ret_side"], row["t1pos"] = jx - 1, kd, rt, p + jx
            row["t1"] = int(keys[p + jx])
            row["bin"] = 1 if rt > 0 else 0
            vals = [row[k_] for k_ in feats if k_ not in EXTRA_FEATURES]
            if not (all(fin(v) for v in vals) and fin(rt)):
                n_drop_feat += 1
                continue
            if extra_fn is not None:
                row.update(extra_fn(p))
                if not all(fin(row[k_]) for k_ in EXTRA_FEATURES):
                    n_drop_extra += 1
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
            contrib = [0.0] + [(logc[i] - logc[i - 1]) / conc[i] if conc[i] > 0 else 0.0 for i in range(1, n)]
            pref_inv, pref_c = [0.0], [0.0]
            for i in range(n):
                pref_inv.append(pref_inv[-1] + inv[i])
                pref_c.append(pref_c[-1] + contrib[i])
            for r in sub:
                a, b = r["pos"], r["t1pos"]
                r["u"] = (pref_inv[b + 1] - pref_inv[a]) / (b - a + 1)
                r["w_raw"] = abs(pref_c[b + 1] - pref_c[a])
        exp_rows.extend(sub)
        say(f"{s}: {n:,} bars, {len(sub):,} events")
    say(f"rebuilt {len(exp_rows):,} events from the dollar bars")
'''
out = cut(out, "    # ---- load raw candles, all symbols ----", '    say(f"rebuilt {len(exp_rows):,} events from raw candles")\n', BLOCK, include_end=True)

# 5. columns compared
out = out.replace('int_cols = ("t0", "t1", "t1pos", "side", "exit_j", "kind", "bin")', 'int_cols = ("t0", "t1", "t1pos", "side", "exit_j", "kind", "bin", "n_fwd", "capped")', 1)
# micro columns: values compared only for rows recomputed in full; the finite pattern for all
out = out.replace('''        for c, (rel, ab) in tol.items():
            if c == "ret_side":
                continue
            if not close(V.num(e[c]), r[c], rel, ab):''', '''        for c, (rel, ab) in tol.items():
            if c == "ret_side":
                continue
            if c in ("sadf", "ent_plugin", "ent_konto") and not r.get("_micro_full", True):
                if not (fin(V.num(e[c])) == fin(r[c])):
                    bad_col[c].append((r["symbol"], r["pos"], e[c], "pattern"))
                continue
            if not close(V.num(e[c]), r[c], rel, ab):''', 1)

# 6. event accounting labels
out = out.replace('"bar counts, CUSUM totals and kept (complete, gap-free window) events replayed from raw candles for every symbol"',
                  '"bar counts, CUSUM totals, membership, sigma and window drops and kept events (time-limit window without a data hole) replayed from the dollar bars for every symbol"', 1)

# 7. trial-count label
out = out.replace("({EXPECTED_PRIOR}; 101 for the first Kraken run, 100 before the market-features run plus that run) and the one more for this run", "({EXPECTED_PRIOR}, given on the command line) and the one more for this run", 1)

# 8. tail
TAIL = r'''    # ---- venue-specific: the bar files are the built ones, and the window coverage is what the bars say ----
    cov = {}
    for wn, (lo, hi) in win_epoch.items():
        cov[wn] = sum(1 for s in sym_list if any(lo <= k < hi for k in series[s][0]))
    cov_sentence = (f"Bar coverage of the reporting windows (symbols with at least one bar inside, of {len(sym_list)} used): " + "; ".join(f"{k_} {n_}" for k_, n_ in cov.items())
                    + f". The market-state feature needs at least 10 symbols with an hourly return in the same hour, so events cannot be built where fewer than 10 symbols have rows; a window with no entered events is reported n/a.")
    check("window coverage (symbols with a bar inside each reporting window) recomputed from the bars equals the run's record and is stated in SUMMARY.md",
          {k_: int(v) for k_, v in meta["window_symbol_coverage"].items()} == cov and any(ln.endswith(cov_sentence) for ln in md.splitlines()), str((cov, meta["window_symbol_coverage"])))
    check("the run is for the bar folder's venue and says so in SUMMARY.md (dollar bars, not hourly candles)", meta["venue"] == bman["venue"] and "DOLLAR BARS" in md, meta["venue"])

    return report()
'''
a = out.index("    # ---- venue-specific: the candle files are the exported ones")
b = out.index("    return report()\n", a) + len("    return report()\n")
out = out[:a] + TAIL + out[b:]

# 9. command line
out = out.replace('''    if "--prior-trials" in argv:''', '''    if "--micro-sample" in argv:
        i_ = argv.index("--micro-sample")
        MICRO_SAMPLE = int(argv[i_ + 1])
        del argv[i_:i_ + 2]
    if "--prior-trials" in argv:''', 1)
(HERE / "verify_meta_bars.py").write_text(out, encoding="utf-8")
print("written", len(out.splitlines()), "lines")
