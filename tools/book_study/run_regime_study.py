#!/usr/bin/env python
"""Market-regime rule for the short side and the matching long filter (READ-ONLY diagnostic; fixed rules, nothing fitted).  Design: see the
docstring of regime_algo.py (printed in SUMMARY.md).  Takes a finished and separately verified meta-pipeline run as its event source (as
run_three_algos.py does), adds the trailing returns of each symbol and of the equal-weight index, and scores the rules.  It loads no model, changes
no threshold, calibration file, model file or execution code, and writes only to runtime/book_regime/<UTC timestamp>/.

    python tools/book_study/run_regime_study.py --source runtime/book_meta/<folder of a verified meta run>
    python tools/book_study/verify_regime_study.py runtime/book_regime/<timestamp> runtime/book_meta/<same folder>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import book_study_core as C
import candle_loader as CL
import regime_algo as R
import run_book_study as RB
import three_algo as T

CUTOFF = "2024-01-01"


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def design_text() -> str:
    doc = R.__doc__
    return doc[doc.index("DESIGN"):].strip()


def render(summ) -> str:
    m, res = summ["meta"], summ["result"]
    V = R.VARIANTS
    L = ["# Market-regime short rule and matching long filter (read-only diagnostic, fixed rules)", ""]
    L.append(f"Run (UTC): {m['run_utc']}. Models/calibration untouched: {summ['integrity_unchanged']}. Source meta run: {m['source_folder']} (events.csv md5 {m['source_events_md5']}).")
    L.append("")
    L += ["```", design_text(), "```", ""]
    for k, v in summ["params_shown"].items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## Whole history (every event scored once; bp per candidate event net of 24 bp where a bet is taken)")
    L.append("")
    L.append("| rule | mean bp per candidate | share taken | daily Sharpe | annualised | PSR | days | best-20-days share of profit |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v in V:
        s = res["stats"][v]
        L.append(f"| {v} | {fmt(s['mean_bp_per_candidate'], 1)} | {fmt(s['taken_share'], 2)} | {fmt(s['sr_daily'], 4)} | {fmt(s['sr_annual'], 2)} | {fmt(s['psr'], 2)} | {s['days']} | {fmt(s['best20_share'], 2)} |")
    L.append("")
    L.append(f"MKT_DOWN is true for {res['down_share']['base']:.1%} of events (tight {res['down_share']['tight']:.1%}, wide {res['down_share']['wide']:.1%}).")
    L.append("")
    for v in R.NEW_VARIANTS:
        d = res["dsr"][v]
        L.append(f"Deflated Sharpe of {v} (daily Sharpe {fmt(d['sr_daily'], 4)}, effective sample {fmt(d['t_eff'], 0)} days, Sharpe variance used {fmt(d['var_sr_used'], 6)}):")
        L.append("")
        L.append("| trials N | SR* (per day) | DSR |")
        L.append("|---|---|---|")
        for n_tr, x in d["by_trials"].items():
            L.append(f"| {n_tr} | {fmt(x['sr_star'], 4)} | {fmt(x['dsr'], 3)} |")
        L.append("")
    L.append(f"## Before and after {CUTOFF} (nothing is fitted; this shows stability, not out-of-sample performance)")
    L.append("")
    L.append("| rule | before: bp per candidate | before: daily Sharpe | before: annualised | after: bp per candidate | after: daily Sharpe | after: annualised |")
    L.append("|---|---|---|---|---|---|---|")
    for v in V:
        a, b = res["halves"][f"{v}|before"], res["halves"][f"{v}|after"]
        L.append(f"| {v} | {fmt(a['mean_bp_per_candidate'], 1)} | {fmt(a['sr_daily'], 4)} | {fmt(a['sr_annual'], 2)} | {fmt(b['mean_bp_per_candidate'], 1)} | {fmt(b['sr_daily'], 4)} | {fmt(b['sr_annual'], 2)} |")
    L.append("")
    L.append("## Reporting windows (events entered inside; mean bp per candidate event)")
    L.append("")
    L.append("| window | events | " + " | ".join(V) + " |")
    L.append("|---|---|" + "---|" * len(V))
    for wn in m["windows"]:
        L.append(f"| {wn} | {res['window_down_share'][wn]['events']} | " + " | ".join(fmt(res["windows"][f"{v}|{wn}"]["mean_bp_per_candidate"], 1) for v in V) + " |")
    L.append("")
    L.append("Trades taken inside each window:")
    L.append("")
    L.append("| window | " + " | ".join(V) + " |")
    L.append("|---|" + "---|" * len(V))
    for wn in m["windows"]:
        L.append(f"| {wn} | " + " | ".join(str(res["windows"][f"{v}|{wn}"]["trades"]) for v in V) + " |")
    L.append("")
    L.append("Share of the window's events with MKT_DOWN true (base / tight / wide definitions):")
    L.append("")
    L.append("| window | base | tight | wide |")
    L.append("|---|---|---|---|")
    for wn in m["windows"]:
        w = res["window_down_share"][wn]
        L.append(f"| {wn} | {fmt(w['base'], 3)} | {fmt(w['tight'], 3)} | {fmt(w['wide'], 3)} |")
    L.append("")
    L.append("Window lines carry no intervals (a window is a few correlated days).")
    L.append("")
    L.append("## By 200-day state and MKT_DOWN (mean bp per candidate event; events in the cell)")
    L.append("")
    L.append("| state | MKT_DOWN | events | " + " | ".join(V) + " |")
    L.append("|---|---|---|" + "---|" * len(V))
    for s_ in (1, 0):
        for dn in (0, 1):
            L.append(f"| {s_} | {dn} | {res['cell_events'][f'state{s_}|down{dn}']} | " + " | ".join(fmt(res["cells"][f"{v}|state{s_}|down{dn}"]["mean_bp_per_candidate"], 1) for v in V) + " |")
    L.append("")
    L.append("## Long and short trades of each rule (bp per trade, net of cost; trades)")
    L.append("")
    L.append("| rule | long: bp per trade | long: trades | short: bp per trade | short: trades |")
    L.append("|---|---|---|---|---|")
    for v in V:
        a, b = res["side_split"][f"{v}|long"], res["side_split"][f"{v}|short"]
        L.append(f"| {v} | {fmt(a['bp_per_trade'], 1)} | {a['trades']} | {fmt(b['bp_per_trade'], 1)} | {b['trades']} |")
    L.append("")
    L.append("## By calendar year (mean bp per candidate event; events in the year)")
    L.append("")
    years = sorted({k.split("|")[1] for k in res["years"]})
    L.append("| year | events | " + " | ".join(V) + " |")
    L.append("|---|---|" + "---|" * len(V))
    for y in years:
        L.append(f"| {y} | {res['years'][f'{V[0]}|{y}']['events']} | " + " | ".join(fmt(res["years"][f"{v}|{y}"]["mean_bp_per_candidate"], 1) for v in V) + " |")
    L.append("")
    L.append("## Deviations from the book and limits")
    L.append("")
    for i, n in enumerate(R.DEVIATIONS, 1):
        L.append(f"{i}. {n}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder of a finished run_meta_pipeline.py run")
    ap.add_argument("--out", default="runtime/book_regime")
    ap.add_argument("--candle-dirs", nargs=2, default=None, metavar=("GMX_DIR", "TV_DIR"))
    args = ap.parse_args()
    src = Path(args.source)
    if not src.is_absolute():
        src = ROOT / src
    started = datetime.now(timezone.utc)
    out_dir = ROOT / args.out / started.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=False)
    before = RB.integrity_snapshot()
    log_file = open(out_dir / "run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    s_summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    s_meta, s_params = s_summ["meta"], s_summ["params"]
    log(f"source meta run {src} ({s_meta['events']:,} events, run {s_meta['run_utc']})")
    df = pd.read_csv(src / "events.csv")
    if len(df) != int(s_meta["events"]):
        raise SystemExit(f"events.csv has {len(df)} rows, the source run says {s_meta['events']}")
    sym_rows = pd.read_csv(src / "symbols.csv")
    gmx_dir, tv_dir = (Path(args.candle_dirs[0]), Path(args.candle_dirs[1])) if args.candle_dirs else (Path(s_meta["gmx_dir"]), Path(s_meta["tradingview_dir"]))
    closes_all, _ = CL.load_closes(gmx_dir, tv_dir, sorted(sym_rows["symbol"].astype(str)), s_meta["timeframe"], 0)
    closes = {}
    for _, r in sym_rows.iterrows():
        s = str(r["symbol"])
        c = closes_all[s]
        c = c[c.index <= pd.Timestamp(r["last"])]
        if len(c) != int(r["bars"]):
            raise SystemExit(f"{s}: {len(c)} candles up to {r['last']}, the source run used {int(r['bars'])}; the candle files changed, stopping")
        closes[s] = c
    log(f"loaded {len(closes)} symbols, cut at the source run's last bars")
    df = T.extend_events(df, closes)
    t0 = df["t0"].to_numpy(np.int64)
    state, idx, _ma = C.market_state(closes, t0)
    if not np.array_equal(state.astype(float), df["mkt_state"].to_numpy(np.float64)):
        raise SystemExit("the 200-day state rebuilt here differs from the source run's mkt_state; stopping")
    rets = R.index_lookback_returns(idx, t0)
    down = {k: R.mkt_down(rets, v) for k, v in R.LOOKBACK_SETS.items()}
    log("MKT_DOWN share: " + ", ".join(f"{k} {v.mean():.3f}" for k, v in down.items()))
    out = df[["symbol", "pos", "t0", "ret_long", "mkt_state", "mom_side"]].copy()
    for L_ in R.ALL_LOOKBACKS:
        out[f"idxret{L_}"] = rets[L_]
    out.to_csv(out_dir / "events_regime.csv", index=False, float_format="%.12g")
    win_t = {k_: (int(v[0]), int(v[1])) for k_, v in s_meta["window_epoch"].items()}
    cost_frac, h = float(s_params["cost_frac"]), int(s_params["h"])
    cut = int(pd.Timestamp(CUTOFF).timestamp())
    res, daily = R.evaluate_rules(df, down, cost_frac, h, win_t, cut)
    daily.to_csv(out_dir / "daily.csv", index=False)
    after = RB.integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    params = {"h": h, "cost_frac": cost_frac, "cutoff": CUTOFF, "lookback_sets": {k: list(v) for k, v in R.LOOKBACK_SETS.items()}, "variants": list(R.VARIANTS),
              "new_variants": list(R.NEW_VARIANTS), "prior_trials_before": R.PRIOR_TRIALS_BEFORE, "trials_after": R.TRIALS_AFTER, "mom_lookbacks": list(T.MOM_LOOKBACKS)}
    meta = {"run_utc": started.isoformat(timespec="seconds"), "source_folder": str(src), "source_events_md5": RB.md5_file(src / "events.csv"),
            "source_summary_md5": RB.md5_file(src / "summary_numbers.json"), "events": int(len(df)), "windows": s_meta["windows"], "window_epoch": s_meta["window_epoch"],
            "window_notes": s_meta["window_notes"], "gmx_dir": str(gmx_dir.resolve()), "tradingview_dir": str(tv_dir.resolve()), "timeframe": s_meta["timeframe"],
            "script_md5": {n: RB.md5_file(HERE / n) for n in ("book_stats.py", "book_study_core.py", "candle_loader.py", "three_algo.py", "regime_algo.py", "run_regime_study.py", "meta_events.py", "meta_model.py", "meta_eval.py")}}
    summ = {"meta": meta, "params": params, "integrity_unchanged": before == after, "result": res,
            "params_shown": {"symbols": len(closes), "events": f"{len(df):,}", "round-trip cost (bp)": f"{cost_frac * 1e4:.1f}", "index lookbacks (hours)": " | ".join(f"{k}: {v}" for k, v in R.LOOKBACK_SETS.items()),
                             "trials before / after this study": f"{R.PRIOR_TRIALS_BEFORE} / {R.TRIALS_AFTER}", "windows": " | ".join(f"{k_}: {v}" for k_, v in s_meta["window_notes"].items())}}
    (out_dir / "summary_numbers.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    (out_dir / "SUMMARY.md").write_text(render(summ), encoding="utf-8")
    log(f"done in {time.time() - started.timestamp():.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
