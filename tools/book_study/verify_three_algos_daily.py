#!/usr/bin/env python
"""Independent checker for a run_three_algos_daily.py output folder (stdlib only).

It runs the existing checker verify_three_algos.py with its HOURLY-specific parts replaced (each replaced region is found between two markers that must each occur exactly once in that file, or this
checker stops; the md5 of that file is printed):
  * the declared lists and settings (5-bar vertical barrier h=4, 20 bp round trip, 12 / 13 features, lookbacks 252 / 126 with skip 21, prior trials 120);
  * the rebuild of the event extension: from the raw Alpaca file (alpaca_daily.csv.gz, read here with gzip and csv), the bar before every entry bar must be the event's signal bar (t_sig); the 12-1
    month return C[t-21] / C[t-252] - 1 and the 6-1 month return C[t-21] / C[t-126] - 1 are recomputed, then the MOM side (+1 both > 0, -1 both < 0, else 0), ret_long and y_dir;
    the source side must equal the sign of 0.5 x (12-1) + 0.5 x (6-1);
  * the effective-sample divisor (holding period of 5 days, not hours / 24), the annualisation (sqrt 252, not sqrt 365) and the deflated-Sharpe trial set (1, 10, 120, 121).
Everything else that checker does is unchanged: CPCV splits and purge / embargo for both forests, every variant, daily series, Sharpe / PSR / deflated Sharpe, windows, states, by-algorithm tables,
walk-forward, every SUMMARY.md table row.  Then, on top:
  * the source run is the Alpaca daily run (venue, timeframe), its closes file has the md5 the source run recorded, and the scripts that built the source run are the recorded versions;
  * the venue section of SUMMARY.md: MOM count and the statement that MOM side equals the source side on every MOM event, the window table (events entered inside each window, first and last day)
    recomputed from events_ext.csv, and the trial statement (prior trials 120; the gate is the 121st).
Exit code 0 = all checks pass.

    python tools/book_study/verify_three_algos_daily.py runtime/book_three_algo_alpaca/<timestamp> [source run folder]
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import verify_book_study as V  # noqa: E402
import verify_meta_pipeline as VM  # noqa: E402

PRIOR = 120
H_DAYS = 4
COST = 0.0020
LB12, LB6, SKIP = 252, 126, 21
NAN = math.nan


def alpaca_raw(path, wanted):
    """{symbol: (sorted epoch list, closes)} from the Alpaca daily file: closes > 0 only, one row per (symbol, date) with the last one kept, ordered by date."""
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            s = r["symbol"]
            if s not in wanted:
                continue
            try:
                c = float(r["close"])
            except ValueError:
                continue
            if not c > 0 or c != c:
                continue
            rows.setdefault(s, {})[r["date"]] = c                 # a later row for the same date replaces the earlier one
    out = {}
    for s, d in rows.items():
        ks = sorted(d)
        out[s] = ([V.ts_key(k) for k in ks], [d[k] for k in ks])
    return out


def day_stats_daily(x, events, t_eff_div):
    n, sr, sk, ku = V.moments(x)
    t_eff = n / t_eff_div
    return {"days": n, "sr_daily": sr, "sr_annual": sr * math.sqrt(252) if VM.fin(sr) else NAN, "skew": sk, "kurt": ku, "t_eff": t_eff, "psr": V.psr(sr, t_eff, sk, ku) if VM.fin(sr) else NAN}


R1_NEW = '''DECLARED_DIR = ["ffd", "mom1n", "mom5n", "mom21n", "mom63n", "log_trgt", "vol_ratio", "mkt_state", "mkt_dist", "sadf", "ent_plugin", "ent_konto"]
DECLARED_MOM = ["side"] + DECLARED_DIR
LOOKBACKS = (252, 126)
SKIP = 21
'''
R2_NEW = '''    check("declared settings: h=4 (5-bar vertical barrier), 20 bp round trip, 6 groups / 2 test groups, bet step 0.1, 100 trees with sequential bootstrap and 10,000 draws, cutoff 2024-01-01, 120 prior trials, "
          "lookbacks 252 / 126 days with skip 21, the 12- and 13-feature lists, the nine variants and the four deflated-Sharpe variants",
          h == 4 and close(cost, 0.0020, 1e-12, 1e-12) and N == 6 and K == 2 and step == 0.1 and params["forest"]["n_estimators"] == 100 and params["forest"]["bootstrap"] == "sequential"
          and max_draws == 10000 and params["cutoff"] == "2024-01-01" and int(params["prior_trials"]) == 120 and list(params["mom_lookbacks"]) == list(LOOKBACKS) and int(params["mom_skip"]) == 21
          and params["dir_features"] == DECLARED_DIR and params["mom_features"] == DECLARED_MOM and params["variants"] == DECLARED_VARIANTS and params["dsr_variants"] == DECLARED_DSR
          and close(embargo_pct, 0.01, 1e-12, 1e-12) and close(float(params["min_weight_fraction_leaf"]), 0.05, 1e-12, 1e-12),
          str({k_: params.get(k_) for k_ in ("h", "cost_frac", "n_groups", "k", "step", "forest", "cutoff", "prior_trials", "mom_lookbacks", "mom_skip")}))
'''
R3_NEW = '''    check("the source run's events.csv and summary_numbers.json are the files this run used (md5), and the source run declares the same 13 features, the 12-1 / 6-1 momentum side, h=4 and the same cost",
          src_md5 == meta["source_events_md5"] and src_sum_md5 == meta["source_summary_md5"] and s_summ["params"]["features"] == DECLARED_MOM and s_summ["params"]["side_signal"] == "mom12_1_6_1"
          and int(s_summ["params"]["h"]) == 4 and close(float(s_summ["params"]["cost_frac"]), cost, 1e-12, 1e-12), f"{src_md5} {meta['source_events_md5']}")
'''
R4_NEW = '''    bad_bars, bad_ret = [], []
    exp_ret = {L: [NAN] * n_ev for L in LOOKBACKS}
    raw = ALPACA_RAW(CSV_OVERRIDE or meta["alpaca_csv"], set(sym_pos))
    for s in sorted(sym_pos):
        keys, closes = raw.get(s, ([], []))
        n_keep = bisect.bisect_right(keys, V.ts_key(rows_by_sym[s]["last"]))
        if n_keep != int(rows_by_sym[s]["bars"]):
            bad_bars.append((s, n_keep, rows_by_sym[s]["bars"]))
            continue
        for i in sym_pos[s]:
            sb = int(float(ev[i]["pos"])) - 1                                   # the signal bar: the bar before the entry bar
            if sb < 0 or sb >= n_keep or int(keys[sb]) != int(float(ev[i]["t_sig"])):
                bad_bars.append((s, i, "the bar before the entry bar is not t_sig"))
                continue
            if sb >= LOOKBACKS[0]:
                a = closes[sb - SKIP]
                for L in LOOKBACKS:
                    exp_ret[L][i] = a / closes[sb - L] - 1.0
    check("raw closes cut at the source run's last bar have exactly the bar counts of the source run, and the bar before every entry bar is the event's signal bar (t_sig)", not bad_bars, str(bad_bars[:3]))
    say("rebuilt trailing returns from raw closes")
    side_ev = [int(float(e["side"])) for e in ev]
    ret_side = [float(e["ret_side"]) for e in ev]
    exp_mom = []
    for i in range(n_ev):
        r = [exp_ret[L][i] for L in LOOKBACKS]
        exp_mom.append(1 if all(x > 0 for x in r) else (-1 if all(x < 0 for x in r) else 0))      # NaN comparisons are False -> 0
    exp_rl = [side_ev[i] * ret_side[i] for i in range(n_ev)]
    for i, x in enumerate(ext):
        for L, col in zip(LOOKBACKS, ("ret12_1", "ret6_1")):
            a, b = float(x[col]), exp_ret[L][i]
            if not close(a, b, 1e-9, 1e-11):
                bad_ret.append((i, col, a, b))
    check("events_ext.csv: the 12-1 and 6-1 month returns of every event recomputed from the raw closes", not bad_ret and not bad_bars, str(bad_ret[:3]))
    mom_side = [int(float(x["mom_side"])) for x in ext]
    check("events_ext.csv: MOM side (+1 if both returns > 0, -1 if both < 0, else 0), long return = side x return, direction label = long return > 0",
          mom_side == exp_mom and all(close(float(x["ret_long"]), exp_rl[i], 1e-9, 1e-12) for i, x in enumerate(ext)) and all(int(float(x["y_dir"])) == (1 if exp_rl[i] > 0 else 0) for i, x in enumerate(ext)),
          str([(i, mom_side[i], exp_mom[i]) for i in range(n_ev) if mom_side[i] != exp_mom[i]][:3]))
    score_sign = all((side_ev[i] == (1 if 0.5 * exp_ret[LOOKBACKS[0]][i] + 0.5 * exp_ret[LOOKBACKS[1]][i] > 0 else -1)) for i in range(n_ev) if fin(exp_ret[LOOKBACKS[0]][i]) and fin(exp_ret[LOOKBACKS[1]][i]))
    check("the source side equals the sign of 0.5 x (12-1 return) + 0.5 x (6-1 return) wherever both exist; MOM events agree with that side",
          score_sign and all(mom_side[i] == 0 or mom_side[i] == side_ev[i] for i in range(n_ev)))
    n_mom = sum(1 for m in mom_side if m != 0)
    check("MOM event counts in the run (total, long, short) equal the recount", int(meta["mom_events"]) == n_mom and int(meta["mom_long"]) == sum(1 for m in mom_side if m == 1) and int(meta["mom_short"]) == sum(1 for m in mom_side if m == -1))

'''
# (old start marker, old end marker (kept), new text); the region from the start marker up to the end marker is replaced
REGIONS = [
    ("DECLARED_DIR = [", "ALGOS = (", R1_NEW),
    ('    check("declared settings:', '    check("models and calibration files untouched', R2_NEW),
    ('    check("the source run\'s events.csv and summary_numbers.json are the files', '    check("window definitions:', R3_NEW),
    ("    bad_bars, bad_ret = [], []\n", "    ret_long = exp_rl\n", R4_NEW),
]
SUBSTITUTIONS = [
    ("t_eff_div = max(1.0, (h + 1) / 24.0)", "t_eff_div = max(1.0, float(h + 1))"),
    ('{"1", "10", "87", "88"}', f'{{"1", "10", "{PRIOR}", "{PRIOR + 1}"}}'),
    ("(N = 1, 10, 87, 88;", f"(N = 1, 10, {PRIOR}, {PRIOR + 1};"),
]


def load_patched(csv_override=None):
    path = HERE / "verify_three_algos.py"
    src = path.read_text(encoding="utf-8")
    for a, b, new in REGIONS:
        if src.count(a) != 1 or src.count(b) != 1 or src.index(a) >= src.index(b):
            raise SystemExit(f"verify_three_algos.py has changed: markers {a!r} / {b!r} occur {src.count(a)} / {src.count(b)} times or are out of order; stopping")
        src = src[: src.index(a)] + new + src[src.index(b):]
    for old, new in SUBSTITUTIONS:
        if src.count(old) != 1:
            raise SystemExit(f"verify_three_algos.py has changed: the text {old!r} occurs {src.count(old)} times, expected once; stopping")
        src = src.replace(old, new)
    mod = types.ModuleType("verify_three_algos_daily_patched")
    mod.__file__ = str(path)
    mod.ALPACA_RAW = alpaca_raw
    mod.CSV_OVERRIDE = csv_override
    sys.modules["verify_three_algos_daily_patched"] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod, hashlib.md5(path.read_bytes()).hexdigest()


def main(folder: str, source_override=None, csv_override=None) -> int:
    d = Path(folder)
    mod, base_md5 = load_patched(csv_override)
    VM.day_stats = day_stats_daily                                          # verify_three_algos calls VM.day_stats(xs, n_events, t_eff_div)
    rc_base = mod.main(folder, source_override)
    print("\n--- Alpaca daily additions ---")
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    summ = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    meta, params = summ["meta"], summ["params"]
    src = Path(source_override) if source_override else Path(meta["source_folder"])
    s_summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    s_meta = s_summ["meta"]
    csv_path = Path(csv_override or meta["alpaca_csv"])
    csv_md5 = hashlib.md5(csv_path.read_bytes()).hexdigest() if csv_path.exists() else None
    check("the source run is an Alpaca daily run, this run records its closes file, and that file's md5 equals the one the source run recorded and the one this run recorded",
          s_meta.get("venue") == "alpaca" and s_meta.get("timeframe") == "1d" and meta.get("venue") == "alpaca" and meta.get("timeframe") == "1d"
          and csv_md5 is not None and csv_md5 == s_meta["alpaca_csv_md5"] == meta["alpaca_csv_md5"], str((s_meta.get("venue"), s_meta.get("timeframe"), csv_md5, s_meta.get("alpaca_csv_md5"), meta.get("alpaca_csv_md5"))))
    bad = [n for n in ("equity_events.py", "meta_events.py", "meta_model.py", "book_stats.py") if hashlib.md5((HERE / n).read_bytes()).hexdigest() != s_meta["script_md5"][n]]
    check("equity_events.py, meta_events.py, meta_model.py and book_stats.py next to this checker are the versions that built the source run (md5 recorded in the source run)", not bad, str(bad))
    check("three_algo.py is the one the study code was built from (md5 recorded as the base of three_algo_daily.py), and the audited checker it was patched from is printed",
          hashlib.md5((HERE / "three_algo.py").read_bytes()).hexdigest() == meta["three_algo_md5_base"] and bool(base_md5), f"verify_three_algos.py md5 {base_md5}")
    md = (d / "SUMMARY.md").read_text(encoding="utf-8")
    lines = md.splitlines()
    ext = V.read(d / "events_ext.csv")
    n_mom = sum(1 for x in ext if int(float(x["mom_side"])) != 0)
    ev = V.read(src / "events.csv")
    same_side = all(int(float(x["mom_side"])) == int(float(e["side"])) for x, e in zip(ext, ev) if int(float(x["mom_side"])) != 0)
    t0 = [int(float(x["t0"])) for x in ext]
    miss = []
    for wn, (lo, hi) in meta["window_epoch"].items():
        sel = [t for t in t0 if int(lo) <= t < int(hi)]
        import datetime as _dt
        day = lambda t: _dt.datetime.fromtimestamp(t, _dt.timezone.utc).date().isoformat()          # noqa: E731
        row = f"| {wn} | {len(sel)} | {day(min(sel)) if sel else 'n/a'} | {day(max(sel)) if sel else 'n/a'} |"
        if row not in lines:
            miss.append(row)
    sentence = f"MOM events {n_mom:,} of {len(ext):,} ({n_mom / len(ext):.4f}); on every one the MOM side equals the source side: {same_side}."
    stated = f"prior trials {PRIOR}; the gate is the {PRIOR + 1}st" in md and "## Venue: Alpaca daily bars" in md and sentence in md
    check("SUMMARY.md venue section: the MOM count and 'MOM side equals the source side on every MOM event' (True, recomputed), the window rows (events entered inside, first and last day) recomputed from "
          "events_ext.csv, and the trial statement (120, gate 121st)", not miss and stated and same_side, str((miss[:2], sentence)))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
    n_ok = sum(1 for c in checks if c[1])
    print(f"\n{n_ok}/{len(checks)} Alpaca additions passed; base checker exit code {rc_base}")
    return 0 if (rc_base == 0 and n_ok == len(checks)) else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    try:
        raise SystemExit(main(argv[0], argv[1] if len(argv) > 1 else None, argv[2] if len(argv) > 2 else None))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, FileNotFoundError, AttributeError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
