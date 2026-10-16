#!/usr/bin/env python
"""Independent checker for a run_three_algos_bars.py output folder (stdlib only).

It runs the existing checker verify_three_algos.py UNCHANGED, loaded with exact textual substitutions (listed in SUBSTITUTIONS; each must occur exactly once or this checker stops) and with its
raw-candle reader pointed at the dollar-bar files (own reader, below):
  * the declared round-trip cost 0.0024 -> 2 x --cost-bp-per-side (Kraken 12, Alpaca 10);
  * the declared trial count 87 -> N (and the deflated-Sharpe trial set 1, 10, 87, 88 -> 1, 10, N, N+1), with the two labels that mention them; the declared hold h == 100 -> h == 95 (4 days);
  * the one rule that differs from the candle versions: a 480 / 720 / 960-bar trailing return whose first bar lies in an earlier data segment than the event's bar is missing.
Everything else that checker does is unchanged: events_ext.csv rebuilt from the raw bars, CPCV splits and purge / embargo for both forests, every variant, daily series, Sharpe / PSR / deflated
Sharpe, windows, states, by-algorithm tables, walk-forward, every SUMMARY.md table row.
Then, on top:
  * the source run is a dollar-bar run of the same venue, the bar folder is the one the source run recorded, and the bar manifest md5 equals the one in venue_extra.json;
  * the hole statistics and the calendar-day spans of the lookbacks recomputed from the raw bars equal venue_extra.json, and the table in SUMMARY.md equals them;
  * the recorded trial count, and the md5 of run_three_algos_bars.py and bar_loader.py.
Exit code 0 = all checks pass.

    python tools/book_study/verify_three_algos_bars.py runtime/book_three_algo_<venue>_bars/<timestamp> --prior-trials N [--cost-bp-per-side 12] [--source <source run folder>]
"""
from __future__ import annotations

import bisect
import csv
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

PRIOR = None
COST_BP_PER_SIDE = 12.0       # Kraken; Alpaca: pass --cost-bp-per-side 10
H_EXPECTED = 95
SEGS = {}
ENDS = {}


def substitutions(prior):
    cost = 2 * COST_BP_PER_SIDE / 1e4
    return [
        ("close(cost, 0.0024, 1e-12, 1e-12)", f"close(cost, {cost!r}, 1e-12, 1e-12)"),
        ('int(params["prior_trials"]) == 87', f'int(params["prior_trials"]) == {prior}'),
        ("87 prior trials", f"{prior} prior trials"),
        ('{"1", "10", "87", "88"}', f'{{"1", "10", "{prior}", "{prior + 1}"}}'),
        ("(N = 1, 10, 87, 88;", f"(N = 1, 10, {prior}, {prior + 1};"),
        ("h == 100 and close(cost", f"h == {H_EXPECTED} and close(cost"),
        ('int(s_summ["params"]["h"]) == 100 and', f'int(s_summ["params"]["h"]) == {H_EXPECTED} and'),
        ("                if p >= L:\n                    exp_ret[L][i] = logc[p] - logc[p - L]",
         "                if p >= L and SEGS[s][p] == SEGS[s][p - L]:\n                    exp_ret[L][i] = logc[p] - logc[p - L]"),
    ]


def load_patched(prior):
    path = HERE / "verify_three_algos.py"
    src = path.read_text(encoding="utf-8")
    for old, new in substitutions(prior):
        if src.count(old) != 1:
            raise SystemExit(f"verify_three_algos.py has changed: the text {old!r} occurs {src.count(old)} times, expected once; stopping")
        src = src.replace(old, new)
    mod = types.ModuleType("verify_three_algos_patched")
    mod.__file__ = str(path)
    mod.__dict__["SEGS"] = SEGS
    sys.modules["verify_three_algos_patched"] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod, hashlib.md5(path.read_bytes()).hexdigest()


def read_bars(bar_dir, symbol):
    """(sorted bar-end epochs, closes, starts) from bars/<SYM>.csv: close > 0 only, first row wins for a repeated end time (the builder never writes one)."""
    bar_dir = Path(bar_dir)
    rows = {}
    with open(bar_dir / "bars" / f"{symbol}.csv", newline="") as f:
        for r in csv.DictReader(f):
            c = V.num(r.get("close"))
            if r["date"] and not math.isnan(c) and c > 0:
                k = V.ts_key(r["date"])
                if k not in rows:
                    rows[k] = (c, V.ts_key(r["start"]))
    keys = sorted(rows)
    return keys, [rows[k][0] for k in keys], [rows[k][1] for k in keys]


def segments_of(keys, starts, break_s):
    seg, cur = [0], 0
    for i in range(1, len(keys)):
        if starts[i] - keys[i - 1] > break_s:
            cur += 1
        seg.append(cur)
    return seg


def bar_raw(bar_dir, _unused, symbol, timeframe="dollar"):
    man = json.loads((Path(bar_dir) / "manifest.json").read_text(encoding="utf-8"))
    keys, closes, starts = read_bars(bar_dir, symbol)
    SEGS[symbol] = segments_of(keys, starts, int(man["break_hours"]) * 3600)
    ENDS[symbol] = keys
    return keys, closes, [math.nan] * len(keys), [math.nan] * len(keys)


def percentile(xs, q):
    """Linear interpolation between order statistics (the usual default)."""
    if not xs:
        return math.nan
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def parse_args(argv):
    global COST_BP_PER_SIDE
    folder, prior, source = None, None, None
    i = 0
    while i < len(argv):
        if argv[i] == "--cost-bp-per-side":
            COST_BP_PER_SIDE = float(argv[i + 1])
            i += 2
        elif argv[i] == "--prior-trials":
            prior = int(argv[i + 1])
            i += 2
        elif argv[i] == "--source":
            source = argv[i + 1]
            i += 2
        else:
            folder = argv[i]
            i += 1
    if folder is None or prior is None:
        raise SystemExit(__doc__)
    return folder, prior, source


def main(folder, prior, source_override=None) -> int:
    global PRIOR
    PRIOR = prior
    d = Path(folder)
    mod, _ = load_patched(prior)
    VM.load_raw = bar_raw                                              # verify_three_algos calls VM.load_raw(meta["gmx_dir"], meta["tradingview_dir"], symbol, meta["timeframe"])
    rc_base = mod.main(folder, source_override)
    print("\n--- dollar-bar additions ---")
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    summ = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    meta, params = summ["meta"], summ["params"]
    ve = json.loads((d / "venue_extra.json").read_text(encoding="utf-8"))
    src = Path(source_override) if source_override else Path(meta["source_folder"])
    s_summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    bdir = Path(ve["bar_dir"])
    venue = s_summ["meta"].get("venue")
    check("the source run is a dollar-bar run, the run's bar folder (both slots) is the folder the source run recorded and the one named in venue_extra.json, the venue matches and the prior trial count is the declared one",
          s_summ["meta"].get("timeframe") == "dollar" and meta["gmx_dir"] == meta["tradingview_dir"] == str(bdir) and str(Path(s_summ["meta"]["candle_dir"]).resolve()) == str(bdir.resolve())
          and ve["venue"] == venue and int(ve["prior_trials"]) == prior == int(params["prior_trials"]), str((meta["gmx_dir"], s_summ["meta"]["candle_dir"], ve["prior_trials"])))
    man = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    check("the bar manifest md5 equals the one recorded in venue_extra.json and in the source run, and the manifest is for the same venue",
          hashlib.md5((bdir / "manifest.json").read_bytes()).hexdigest() == ve["bars_manifest_md5"] == s_summ["meta"]["bars_manifest_md5"] and man["venue"] == venue and ve["break_hours"] == man["break_hours"])
    syms = sorted(r["symbol"] for r in V.read(src / "symbols.csv"))
    bad = []
    for s in syms:
        got = hashlib.md5((bdir / "bars" / f"{s}.csv").read_bytes()).hexdigest()
        if man["symbols"][s]["bars_md5"] != got:
            bad.append((s, got))
    check("every bar CSV used has the md5 written by the builder", not bad, str(bad[:2]))
    # hole statistics and spans, own code
    ev = V.read(src / "events.csv")
    cut = float(V.ts_key(params["cutoff"]))
    by_sym = {}
    for e in ev:
        by_sym.setdefault(e["symbol"], []).append(e)
    Ls = (480, 720, 960)
    n_with = {L: 0 for L in Ls}
    dropped = {L: 0 for L in Ls}
    days = {L: [] for L in Ls}
    days_post = {L: [] for L in Ls}
    for s, es in by_sym.items():
        keys = ENDS[s] if s in ENDS else bar_raw(bdir, None, s)[0]
        seg = SEGS[s]
        for e in es:
            p = int(float(e["pos"]))
            post = float(e["t0"]) >= cut
            for L in Ls:
                if p >= L:
                    n_with[L] += 1
                    if seg[p] != seg[p - L]:
                        dropped[L] += 1
                    else:
                        dd = (keys[p] - keys[p - L]) / 86400.0
                        days[L].append(dd)
                        if post:
                            days_post[L].append(dd)
    hs, sp = ve["hole_stats"], ve["span_days"]
    bad_h, bad_sp = [], []
    for L in Ls:
        h = hs[str(L)]
        if (h["events_with_lookback"], h["dropped_for_hole"]) != (n_with[L], dropped[L]):
            bad_h.append((L, (n_with[L], dropped[L]), h))
        r = sp[str(L)]
        want = {"events": len(days[L]), "median_days": percentile(days[L], 50), "p10_days": percentile(days[L], 10), "p90_days": percentile(days[L], 90),
                "events_post_cutoff": len(days_post[L]), "median_days_post_cutoff": percentile(days_post[L], 50), "p10_days_post_cutoff": percentile(days_post[L], 10),
                "p90_days_post_cutoff": percentile(days_post[L], 90)}
        for k, v in want.items():
            a = r[k]
            ok = (a == v) if isinstance(v, int) else (abs(a - v) <= 1e-9 * max(1.0, abs(v)) or (math.isnan(a) and math.isnan(v)))
            if not ok:
                bad_sp.append((L, k, a, v))
    check("hole statistics (events with the lookback, dropped because it reaches across a data hole) recomputed from the raw bars equal venue_extra.json", not bad_h, str(bad_h[:2]))
    check("calendar-day spans of the 480 / 720 / 960-bar lookbacks (count, median, 10th and 90th percentile, overall and after the cutoff) recomputed from the raw bars equal venue_extra.json", not bad_sp, str(bad_sp[:3]))
    md = (d / "SUMMARY.md").read_text(encoding="utf-8")
    lines = set(md.splitlines())
    miss = []
    for L in Ls:
        r, h = sp[str(L)], hs[str(L)]
        row = (f"| {L} | {r['events']} | {r['median_days']:.1f} [{r['p10_days']:.1f}, {r['p90_days']:.1f}] | {r['events_post_cutoff']} | "
               f"{r['median_days_post_cutoff']:.1f} [{r['p10_days_post_cutoff']:.1f}, {r['p90_days_post_cutoff']:.1f}] | {h['dropped_for_hole']} of {h['events_with_lookback']} |")
        if row not in lines:
            miss.append((L, row))
    stated = (f"prior trials {prior}; the gate is the {prior + 1}th" in md and f"## Venue: {venue.capitalize()} dollar bars" in md
              and f"prior trials {prior}, the gate the {prior + 1}th, N = 1, 10, {prior}, {prior + 1}." in md and "describe that version." in md)
    check("SUMMARY.md venue section: the span / hole table rows equal the recomputation and the trial count is stated", not miss and stated, str(miss[:2]))
    chg = [n for n, m in ve["script_md5"].items() if hashlib.md5((HERE / n).read_bytes()).hexdigest() != m]
    check("run_three_algos_bars.py and bar_loader.py are the recorded ones (md5)", not chg, str(chg))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
    n_ok = sum(1 for c in checks if c[1])
    print(f"\n{n_ok}/{len(checks)} dollar-bar additions passed; base checker exit code {rc_base}")
    return 0 if (rc_base == 0 and n_ok == len(checks)) else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    f, p, s = parse_args(argv)
    try:
        raise SystemExit(main(f, p, s))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, FileNotFoundError, AttributeError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
