#!/usr/bin/env python
"""Every chapter's code from the book (the verbatim ports in book_mp.py, book_port_a.py ... book_port_e.py) applied to one venue's DOLLAR BARS (read-only diagnostic: no model of the project is changed, no order path is touched).

    python tools/book_study/run_book_ports_bars.py --source runtime/book_meta_kraken_bars/<timestamp> [--quick] [--only ch2,ch3,...]

The source run (run_meta_bars.py) supplies the events, labels, weights and predictions; the bars come from the bar folder it names.  Output: <out>/<timestamp>/{SUMMARY.md, summary_numbers.json, run.log, tables/*.csv}.
Every figure in SUMMARY.md is produced by this script; the numbers in summary_numbers.json are the ones the SUMMARY prints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ports_common as PCm  # noqa: E402
import ports_sections_a as SA  # noqa: E402
import ports_sections_b as SB  # noqa: E402
import ports_sections_c as SC  # noqa: E402

FULL = {"seed": 20261023, "threads": 2, "cv": 5,
        "ch2_days": 730,
        "chain_symbols": 3, "chain_smallest": True,
        "ch4_symbols": 3, "ch4_events": 150, "ch4_seq_events": 90, "ch4_reps": 8, "ch4_mc_bars": 1000, "ch4_mc_obs_cap": 40, "ch4_mc_iters": 300,
        "ch5_symbols": 12, "ch5_days": 1500,
        "model_sample": 6000, "n_estimators": 200, "n_estimators_fi": 100,
        "synth_features": 20, "synth_informative": 5, "synth_redundant": 5, "synth_samples": 2000,
        "hp_sample": 2500, "hp_random_iter": 25,
        "ch10_symbols": 40,
        "ch13_compare_iter": 1500, "ch13_ref_paths": 100000, "ch13_steps": 100, "ch13_est_iter": 400,
        "ch15_binom_n": 1000000,
        "ch16_days": 1100, "ch16_slength": 260, "ch16_rebal": 22, "ch16_mc_iters": 1000,
        "ch17_events_per_symbol": 40, "ch18_lz_messages": 300,
        "ch20_rows": 1000, "ch20_cols": 4000,
        "ch21_assets": 3, "ch21_horizon": 3, "ch21_days": 360,
        "ch13_main_iter": 100, "ch4_example_reps": 1000, "cusum_bars": 20000}
QUICK = dict(FULL, ch2_days=300, chain_symbols=1, ch4_symbols=1, ch4_events=60, ch4_seq_events=30, ch4_reps=2, ch4_mc_iters=20, ch4_mc_obs_cap=12, ch5_symbols=2, ch5_days=500,
             model_sample=1200, n_estimators=30, n_estimators_fi=20, synth_samples=600, hp_sample=600, hp_random_iter=4, ch10_symbols=4,
             ch13_compare_iter=100, ch13_ref_paths=5000, ch13_est_iter=30, ch15_binom_n=20000, ch16_days=500, ch16_mc_iters=20, ch17_events_per_symbol=3, ch18_lz_messages=20,
             ch20_rows=300, ch20_cols=300, ch21_days=90, ch21_horizon=2, ch13_main_iter=2, ch4_example_reps=50, cusum_bars=3000)

ORDER = ["ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "ch9", "ch10", "ch13", "ch14", "ch15", "ch16", "ch17_19", "ch20", "ch21"]
TITLES = {"ch2": "Chapter 2 - financial data structures (Snippets 2.1-2.3)", "ch3": "Chapter 3 - labeling (Snippets 2.4, 3.1-3.8)", "ch4": "Chapter 4 - sample weights (Snippets 4.1-4.11)",
          "ch5": "Chapter 5 - fractionally differentiated features (Snippets 5.1-5.4)", "ch6": "Chapter 6 - ensemble methods (Snippets 6.1-6.2)", "ch7": "Chapter 7 - cross-validation in finance (Snippets 7.1-7.4)",
          "ch8": "Chapter 8 - feature importance (Snippets 8.2-8.10)", "ch9": "Chapter 9 - hyper-parameter tuning (Snippets 9.1-9.4)", "ch10": "Chapter 10 - bet sizing (Snippets 10.1-10.4)",
          "ch13": "Chapter 13 - backtesting on synthetic data (Snippets 13.1-13.2)", "ch14": "Chapter 14 - backtest statistics (Snippets 14.1-14.4)", "ch15": "Chapter 15 - understanding strategy risk (Snippets 15.1-15.5)",
          "ch16": "Chapter 16 - machine-learning asset allocation (Snippets 16.1-16.5)", "ch17_19": "Chapters 17-19 - structural breaks, entropy, microstructure (Snippets 17.1-17.4, 18.1-18.4, 19.1-19.2)",
          "ch20": "Chapter 20 - multiprocessing (Snippets 20.1-20.14)", "ch21": "Chapter 21 - brute force and quantum computers (Snippets 21.1-21.7)"}


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--bars", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of sections: " + ",".join(ORDER))
    ap.add_argument("--subset", type=int, default=None, help="number of symbols in the symbol subset (default: all for <= 40 symbols, else 40)")
    a = ap.parse_args()
    src = Path(a.source)
    src = (src if src.is_absolute() else ROOT / src).resolve()
    summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    if summ["meta"].get("timeframe") != "dollar":
        raise SystemExit("the source run is not a dollar-bar run")
    venue = summ["meta"]["venue"]
    bd = Path(a.bars) if a.bars else Path(summ["meta"]["candle_dir"])
    bd = (bd if bd.is_absolute() else ROOT / bd).resolve()
    cfg = dict(QUICK if a.quick else FULL)
    subset_n = a.subset or 40
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(a.out or f"runtime/book_ports_{venue}_bars")
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    out = out_root / ts
    (out / "tables").mkdir(parents=True)
    log = PCm.Log(out / "run.log")
    log(f"venue {venue}; source {src}; bars {bd}; quick={a.quick}")
    V = PCm.Venue(venue, src, bd, subset_n)
    log(f"events {len(V.events):,}; symbols {len(V.by_events)}; subset {len(V.subset)}: {', '.join(V.subset[:8])}{'...' if len(V.subset) > 8 else ''}")
    only = [s.strip() for s in a.only.split(",")] if a.only else list(ORDER)
    results, mds, errors, secs = {}, {}, {}, {}
    fns = {"ch2": SA.ch2, "ch3": SA.ch3, "ch4": SA.ch4, "ch5": SA.ch5, "ch6": SA.ch6, "ch7": SB.ch7, "ch8": SB.ch8, "ch9": SB.ch9, "ch10": SB.ch10, "ch13": SC.ch13, "ch14": SC.ch14, "ch15": SC.ch15,
           "ch16": SC.ch16, "ch17_19": SC.ch17_18_19, "ch20": SC.ch20, "ch21": SC.ch21}
    for name in ORDER:
        if name not in only:
            continue
        t0 = time.time()
        log(f"== {name} ==")
        try:
            r = fns[name](V, cfg, log)
            if name == "ch8":
                res, md, tabs = r
                for m, df in tabs.items():
                    df.to_csv(out / "tables" / f"ch8_{m.lower()}_importance.csv", float_format="%.10g")
            elif name == "ch17_19":
                res, md17, md18, md19 = r
                results["ch17"], results["ch18"], results["ch19"] = {k: res[k] for k in ("event_features_recomputed_by_book_code",)}, {k: res[k] for k in ("event_features_recomputed_by_book_code", "lempel_ziv_library")}, res["corwin_schultz"]
                mds["ch17"], mds["ch18"], mds["ch19"] = md17, md18, md19
                secs[name] = time.time() - t0
                continue
            else:
                res, md = r
            results[name], mds[name] = res, md
        except Exception as exc:                                              # noqa: BLE001
            tb = traceback.format_exc()
            errors[name] = tb
            log(f"!! {name} failed: {exc}\n{tb}")
        secs[name] = time.time() - t0
        log(f"   {name} done in {secs[name]:.0f}s")
    meta = {"venue": venue, "run_utc": ts, "source_run": str(src), "bar_dir": str(bd), "bars_manifest_md5": md5_file(bd / "manifest.json"), "quick": a.quick, "config": cfg, "subset": V.subset,
            "events": int(len(V.events)), "seconds": secs, "errors": list(errors),
            "script_md5": {n: md5_file(HERE / n) for n in ("run_book_ports_bars.py", "ports_common.py", "ports_sections_a.py", "ports_sections_b.py", "ports_sections_c.py", "book_mp.py", "book_port_a.py", "book_port_b.py",
                                                            "book_port_c.py", "book_port_d.py", "book_port_e.py")}}
    (out / "summary_numbers.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    L = [f"# The book's code, chapter by chapter, on {venue.capitalize()} dollar bars", "",
         f"Run {ts} UTC. Events, labels, weights and predictions: {src.name} ({len(V.events):,} events, {len(V.by_events)} symbols); bars: {bd}. Symbol subset for the studies that need bar-level series: {len(V.subset)} symbols "
         f"({'all' if len(V.subset) == len(V.by_events) else 'the ' + str(len(V.subset)) + ' with the most events'}). {'QUICK RUN: sizes cut for a smoke test, numbers are not results. ' if a.quick else ''}"
         "The code is the verbatim port of each snippet (book_*.py; changes from the printed text are listed in each file's header). Run-size reductions against the printed examples are stated where they apply.", ""]
    for name in ORDER:
        if name == "ch17_19":
            for sub, title in (("ch17", "Chapter 17 - structural breaks (Snippets 17.1-17.4)"), ("ch18", "Chapter 18 - entropy features (Snippets 18.1-18.4)"), ("ch19", "Chapter 19 - microstructural features (Snippets 19.1-19.2)")):
                if sub in mds:
                    L += [f"## {title}", ""] + mds[sub] + [""]
            continue
        if name in mds:
            L += [f"## {TITLES[name]}", ""] + mds[name] + [""]
        elif name in errors:
            L += [f"## {TITLES[name]}", "", "**This section failed; no numbers.** Error:", "", "```", errors[name].strip().splitlines()[-1], "```", ""]
    (out / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    log(f"written {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
