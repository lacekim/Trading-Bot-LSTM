#!/usr/bin/env python
"""Independent checker for a run_three_algos_venue.py output folder (stdlib only).

It runs the existing checker verify_three_algos.py UNCHANGED, loaded with four exact textual substitutions (listed in SUBSTITUTIONS; each must occur exactly once or this checker stops)
and with its raw-candle reader pointed at the Kraken candle files (verify_meta_venue.load_raw):
  * the declared trial count 87 -> 115 (and the deflated-Sharpe trial set 1, 10, 87, 88 -> 1, 10, 115, 116), with the two labels that mention them.
Everything else that checker does is unchanged: events_ext.csv rebuilt from the raw candles, CPCV splits and purge / embargo for both forests, every variant, daily series, Sharpe / PSR /
deflated Sharpe, windows, states, by-algorithm tables, walk-forward, every SUMMARY.md table row.
Then, on top:
  * the source run is a Kraken run, the candle folder in the run is the one the source run recorded, and every candle CSV equals the export manifest (md5);
  * the span diagnostic of venue_extra.json (share of events whose 480 / 720 / 960-bar lookback spans a different number of days than 20 / 30 / 40 by more than 10%, overall and after the
    cutoff) recomputed from the raw candles, and the table in SUMMARY.md equals it;
  * the recorded trial count (115), and the md5 of run_three_algos_venue.py and venue_loader.py.
Exit code 0 = all checks pass.

    python tools/book_study/verify_three_algos_venue.py runtime/book_three_algo_kraken/<timestamp> [source run folder]
"""
from __future__ import annotations

import bisect
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
import verify_meta_venue as VV  # noqa: E402

PRIOR = 115
SUBSTITUTIONS = [
    ('int(params["prior_trials"]) == 87', f'int(params["prior_trials"]) == {PRIOR}'),
    ("87 prior trials", f"{PRIOR} prior trials"),
    ('{"1", "10", "87", "88"}', f'{{"1", "10", "{PRIOR}", "{PRIOR + 1}"}}'),
    ("(N = 1, 10, 87, 88;", f"(N = 1, 10, {PRIOR}, {PRIOR + 1};"),
]


def load_patched():
    path = HERE / "verify_three_algos.py"
    src = path.read_text(encoding="utf-8")
    for old, new in SUBSTITUTIONS:
        if src.count(old) != 1:
            raise SystemExit(f"verify_three_algos.py has changed: the text {old!r} occurs {src.count(old)} times, expected once; stopping")
        src = src.replace(old, new)
    mod = types.ModuleType("verify_three_algos_patched")
    mod.__file__ = str(path)
    sys.modules["verify_three_algos_patched"] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod, hashlib.md5(path.read_bytes()).hexdigest()


def kraken_raw(gmx_dir, _tv_dir, symbol, timeframe="1h"):
    return VV.load_raw(gmx_dir, symbol, timeframe)


def main(folder: str, source_override=None) -> int:
    d = Path(folder)
    mod, _ = load_patched()
    VM.load_raw = kraken_raw                                           # verify_three_algos calls VM.load_raw(meta["gmx_dir"], meta["tradingview_dir"], symbol, timeframe)
    rc_base = mod.main(folder, source_override)
    print("\n--- Kraken additions ---")
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    summ = json.loads((d / "summary_numbers.json").read_text(encoding="utf-8"))
    meta, params = summ["meta"], summ["params"]
    ve = json.loads((d / "venue_extra.json").read_text(encoding="utf-8"))
    src = Path(source_override) if source_override else Path(meta["source_folder"])
    s_summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    cdir = Path(ve["candle_dir"])
    check("the source run is a Kraken run, the run's candle folder (both slots) is the folder the source run recorded and the one named in venue_extra.json, and the prior trial count is the declared 115",
          s_summ["meta"].get("venue") == "kraken" and meta["gmx_dir"] == meta["tradingview_dir"] == str(cdir) and str(Path(s_summ["meta"]["candle_dir"]).resolve()) == str(cdir.resolve())
          and ve["venue"] == "kraken" and int(ve["prior_trials"]) == PRIOR == int(params["prior_trials"]), str((meta["gmx_dir"], s_summ["meta"]["candle_dir"], ve["prior_trials"])))
    syms = sorted(r["symbol"] for r in V.read(src / "symbols.csv"))
    man_path = cdir.parent / "manifest.json"
    bad = []
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8")).get("kraken", {})
        for s in syms:
            got = hashlib.md5((cdir / f"{s}_1h.csv").read_bytes()).hexdigest()
            if s not in man or man[s].get("csv_md5") != got:
                bad.append((s, got, man.get(s, {}).get("csv_md5")))
    check("every candle CSV used equals the file recorded by the export (md5 in manifest.json next to the candle folder)", man_path.exists() and not bad, f"{man_path} exists={man_path.exists()} {bad[:2]}")
    # span diagnostic from the raw candles
    ev = V.read(src / "events.csv")
    sym_rows = {r["symbol"]: r for r in V.read(src / "symbols.csv")}
    cut = float(V.ts_key(params["cutoff"]))
    tol = float(ve["span_tolerance"])
    cnt = {L: [0, 0, 0, 0] for L in (480, 720, 960)}
    by_sym = {}
    for e in ev:
        by_sym.setdefault(e["symbol"], []).append(e)
    for s, es in by_sym.items():
        keys, _c, _h, _l = VV.load_raw(cdir, s)
        n_keep = bisect.bisect_right(keys, V.ts_key(sym_rows[s]["last"]))
        keys = keys[:n_keep]
        for e in es:
            p = int(float(e["pos"]))
            post = float(e["t0"]) >= cut
            for L in cnt:
                if p >= L:
                    off = abs((keys[p] - keys[p - L]) / 3600.0 / L - 1.0) > tol
                    c = cnt[L]
                    c[0] += 1
                    c[1] += off
                    if post:
                        c[2] += 1
                        c[3] += off
    sp = ve["span_diagnostic"]
    bad_sp = []
    for L, c in cnt.items():
        r = sp[str(L)]
        if [r["events_with_lookback"], r["events_span_off"], r["events_post_cutoff"], r["events_post_cutoff_span_off"]] != c:
            bad_sp.append((L, c, r))
    check("span diagnostic (events whose lookback spans more than 10% away from 20 / 30 / 40 days, overall and after the cutoff) recomputed from the raw candles equals venue_extra.json", not bad_sp, str(bad_sp[:2]))
    md = (d / "SUMMARY.md").read_text(encoding="utf-8")
    miss = []
    for L, c in cnt.items():
        row = f"| {L} | {c[0]} | {c[1]} | {c[1] / c[0]:.4f} | {c[2]} | {c[3] / c[2]:.4f} |" if c[0] and c[2] else None
        if row is None or row not in md.splitlines():
            miss.append((L, row))
    stated = (f"prior trials {PRIOR}; the gate is the {PRIOR + 1}th" in md and "## Venue: Kraken hourly candles" in md
              and f"prior trials {PRIOR}, the gate the {PRIOR + 1}th, N = 1, 10, {PRIOR}, {PRIOR + 1}." in md and "describe that version." in md)
    check("SUMMARY.md venue section: the span table rows equal the recomputation and the trial count (115, gate 116th) is stated", not miss and stated, str(miss[:2]))
    chg = [n for n, m in ve["script_md5"].items() if hashlib.md5((HERE / n).read_bytes()).hexdigest() != m]
    check("run_three_algos_venue.py and venue_loader.py are the recorded ones (md5)", not chg, str(chg))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      {detail}"))
    n_ok = sum(1 for c in checks if c[1])
    print(f"\n{n_ok}/{len(checks)} Kraken additions passed; base checker exit code {rc_base}")
    return 0 if (rc_base == 0 and n_ok == len(checks)) else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(__doc__)
    try:
        raise SystemExit(main(argv[0], argv[1] if len(argv) > 1 else None))
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, FileNotFoundError, AttributeError) as exc:
        print(f"CHECKER STOPPED ON A MALFORMED FILE: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
