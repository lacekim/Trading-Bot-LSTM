#!/usr/bin/env python
"""Three algorithms (long, short, momentum) and a forest gate that picks one per event (READ-ONLY diagnostic).  Design: see the docstring of
three_algo.py (printed in SUMMARY.md).  It takes a finished meta-pipeline run as input (its events.csv; that run is checked separately by
verify_meta_pipeline.py), adds three trailing returns per event from the candles, fits the forests with combinatorial purged CV and one
walk-forward fit, and scores the variants.  It loads no model, changes no threshold, calibration file, model file or execution code, and writes only
to runtime/book_three_algo/<UTC timestamp>/.

Run in the `javier` conda env, from the project root:
    python tools/book_study/run_three_algos.py --source runtime/book_meta/<folder of a verified meta run>
    python tools/book_study/verify_three_algos.py runtime/book_three_algo/<timestamp>
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

import candle_loader as CL
import meta_events as E
import run_book_study as RB
import three_algo as T

SEED = 20261014          # fixed before any result of this module was seen
VARIANT_TEXT = {
    "long_all": "LONG at every event, size 1", "short_all": "SHORT at every event, size 1", "mom_all": "MOM side on MOM events, size 1",
    "mom_long_up": "fixed rule: MOM long while the index is above its 200-day mean", "dir_model": "side and size from F_dir alone",
    "mom_meta": "MOM events sized by F_mom", "gate": "the gate (best of LONG / SHORT / MOM by confidence)",
    "gate_shuffled_size": "null: gate sides, sizes shuffled among all events", "gate_random_side": "null: gate sizes, coin-flip sides"}


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def design_text() -> str:
    doc = T.__doc__
    a = doc.index("DESIGN")
    return doc[a:].strip()


def render(summ) -> str:
    m = summ["meta"]
    L = ["# Three algorithms (long, short, momentum) and a gate that picks one (read-only diagnostic)", ""]
    L.append(f"Run (UTC): {m['run_utc']}. Models/calibration untouched: {summ['integrity_unchanged']}. Source meta run: {m['source_folder']} (events.csv md5 {m['source_events_md5']}).")
    L.append("")
    L.append("```")
    L.append(design_text())
    L.append("```")
    L.append("")
    for k, v in summ["params_shown"].items():
        L.append(f"- {k}: {v}")
    L.append("")
    cp = summ["cpcv"]
    L.append(f"## Combinatorial purged CV: {m['n_groups']} groups, {m['k']} test groups -> {m['n_splits']} splits, {m['n_paths']} backtest paths")
    L.append("")
    L.append(f"Events {m['events']:,}; MOM events {m['mom_events']:,} ({m['mom_long']:,} long, {m['mom_short']:,} short). Out-of-sample AUC per split (0.5 = no information): "
             f"F_dir mean {fmt(cp['auc_dir_mean'])}, min {fmt(cp['auc_dir_min'])}, max {fmt(cp['auc_dir_max'])}; F_mom mean {fmt(cp['auc_mom_mean'])}, min {fmt(cp['auc_mom_min'])}, max {fmt(cp['auc_mom_max'])}. "
             f"Share of events with p_dir > 0.5: {fmt(cp['share_dir_mean'], 2)}; share of MOM events with p_mom > 0.5: {fmt(cp['share_mom_mean'], 2)}.")
    L.append("")
    L.append("| variant | what it is | mean bp per candidate event | daily Sharpe | annualised | paths with Sharpe < 0 | mean PSR | share taken | mean size | best-20-days share of profit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for v in T.VARIANTS:
        s = summ["eval"]["summary"][v]
        L.append(f"| {v} | {VARIANT_TEXT[v]} | {fmt(s['mean_bp_per_candidate'], 1)} | {fmt(s['mean_sr_daily'], 4)} | {fmt(s['mean_sr_annual'], 2)} | {s['paths_sr_negative']}/{s['paths']} | "
                 f"{fmt(s['mean_psr'], 2)} | {fmt(s['taken_share'], 2)} | {fmt(s['mean_size'], 3)} | {fmt(s['best20_share'], 2)} |")
    L.append("")
    L.append("Candidate events are all events (MOM-only variants therefore show the gain per unit of capital offered, not per trade). The fixed-rule variants (long_all, short_all, mom_all, mom_long_up) have no fitted part, so their paths are identical.")
    L.append("")
    L.append("Gate against the others (Sharpe higher on how many of the paths; mean difference in daily Sharpe; mean difference in bp per candidate):")
    L.append("")
    L.append("| comparison | paths higher | mean daily-Sharpe difference | mean bp difference |")
    L.append("|---|---|---|---|")
    for k_, c in summ["eval"]["compare"].items():
        L.append(f"| {k_.replace('|', ' minus ')} | {c['paths_sr_higher']}/{c['paths']} | {fmt(c['mean_sr_daily_diff'], 4)} | {fmt(c['mean_bp_diff'], 2)} |")
    L.append("")
    for v in T.DSR_VARIANTS:
        d = summ["eval"]["dsr"][v]
        L.append(f"Deflated Sharpe of {v} (mean daily Sharpe {fmt(d['sr_daily_mean'], 4)}, effective sample {fmt(d['t_eff'], 0)} days, Sharpe variance used {fmt(d['var_sr_used'], 6)}):")
        L.append("")
        L.append("| trials N | SR* (per day) | DSR |")
        L.append("|---|---|---|")
        for n_tr, x in d["by_trials"].items():
            L.append(f"| {n_tr} | {fmt(x['sr_star'], 4)} | {fmt(x['dsr'], 3)} |")
        L.append("")
    wf = summ["walk_forward"]
    L.append(f"## Walk-forward check (one fit, labels ending before {wf['cutoff']}, tested on later events)")
    L.append("")
    L.append(f"Train events {wf['n_train_dir']:,} (MOM {wf['n_train_mom']:,}), test events {wf['n_test']:,} (MOM {wf['n_test_mom']:,}); AUC F_dir {fmt(wf['auc_dir'])}, F_mom {fmt(wf['auc_mom'])}.")
    L.append("")
    L.append("| variant | mean bp per candidate event | daily Sharpe | annualised | PSR | share taken |")
    L.append("|---|---|---|---|---|---|")
    for v in T.VARIANTS:
        s = summ["walk_forward_eval"]["summary"][v]
        L.append(f"| {v} | {fmt(s['mean_bp_per_candidate'], 1)} | {fmt(s['mean_sr_daily'], 4)} | {fmt(s['mean_sr_annual'], 2)} | {fmt(s['mean_psr'], 2)} | {fmt(s['taken_share'], 2)} |")
    L.append("")
    for src, key in (("CPCV", "eval"), ("walk-forward", "walk_forward_eval")):
        L.append(f"## Reporting windows, {src} (events entered inside; mean bp per candidate event)")
        L.append("")
        L.append("| window | events per path | " + " | ".join(T.VARIANTS) + " |")
        L.append("|---|---|" + "---|" * len(T.VARIANTS))
        for wn in m["windows"]:
            w = summ[key]["windows"]
            L.append(f"| {wn} | {summ[key]['candidates_window'][wn]:.0f} | " + " | ".join(fmt(w[f"{v}|{wn}"]["mean_bp_per_candidate"], 1) for v in T.VARIANTS) + " |")
        L.append("")
    L.append("Window lines carry no intervals (a window is a few correlated days).")
    L.append("")
    for src, key in (("CPCV", "eval"), ("walk-forward", "walk_forward_eval")):
        L.append(f"## By market state, {src} (state 1 = equal-weight index above its own 200-day mean at the event; 0 = below; mean bp per candidate event)")
        L.append("")
        L.append("| state | events per path | " + " | ".join(T.VARIANTS) + " |")
        L.append("|---|---|" + "---|" * len(T.VARIANTS))
        for stv in (1, 0):
            bs = summ[key]["by_state"]
            L.append(f"| {stv} | {summ[key]['candidates_state'][f'state{stv}']:.0f} | " + " | ".join(fmt(bs[f"{v}|state{stv}"]["mean_bp_per_candidate"], 1) for v in T.VARIANTS) + " |")
        L.append("")
    for src, key in (("CPCV", "eval"), ("walk-forward", "walk_forward_eval")):
        L.append(f"## Which algorithm the gate used, and what it earned, {src}")
        L.append("")
        L.append("By state: trades per path; contribution in bp per candidate event of that state (the three rows of a state add up to the gate's number); bp per trade.")
        L.append("")
        L.append("| state | algorithm | trades per path | bp per state candidate | bp per trade |")
        L.append("|---|---|---|---|---|")
        for stv in (1, 0):
            for al in ("LONG", "SHORT", "MOM"):
                a = summ[key]["algo_state"][f"gate|{al}|state{stv}"]
                L.append(f"| {stv} | {al} | {a['trades_per_path']:.0f} | {fmt(a['bp_per_state_candidate'], 1)} | {fmt(a['bp_per_trade'], 1)} |")
        L.append("")
        L.append("By window:")
        L.append("")
        L.append("| window | algorithm | trades per path | bp per window candidate | bp per trade |")
        L.append("|---|---|---|---|---|")
        for wn in m["windows"]:
            for al in ("LONG", "SHORT", "MOM"):
                a = summ[key]["algo_window"][f"gate|{al}|{wn}"]
                L.append(f"| {wn} | {al} | {a['trades_per_path']:.0f} | {fmt(a['bp_per_window_candidate'], 1)} | {fmt(a['bp_per_trade'], 1)} |")
        L.append("")
    L.append("## Deviations from the book and limits")
    L.append("")
    for i, n in enumerate(T.DEVIATIONS, 1):
        L.append(f"{i}. {n}")
    L.append("")
    return "\n".join(L)


def newest_meta_run() -> Path:
    runs = sorted(p for p in (ROOT / "runtime" / "book_meta").glob("*") if (p / "events.csv").exists())
    if not runs:
        raise SystemExit("no meta-pipeline run found under runtime/book_meta; pass --source")
    return runs[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="folder of a finished run_meta_pipeline.py run (default: the newest under runtime/book_meta)")
    ap.add_argument("--out", default="runtime/book_three_algo")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-draws", type=int, default=10000)
    ap.add_argument("--bootstrap", choices=("sequential", "standard"), default="sequential")
    ap.add_argument("--embargo-pct", type=float, default=0.01)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--cutoff", default="2024-01-01")
    ap.add_argument("--prior-trials", type=int, default=T.PRIOR_TRIALS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--candle-dirs", nargs=2, default=None, metavar=("GMX_DIR", "TV_DIR"), help="read candles here instead of the folders recorded in the source run")
    args = ap.parse_args()
    if args.quick:
        args.n_groups, args.n_estimators, args.max_draws = 4, 10, 3000
    src = Path(args.source) if args.source else newest_meta_run()
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
    if s_params["features"] != E.FEATURES or s_params["side_signal"] != E.SIDE_SIGNAL:
        raise SystemExit("the source run's features / side signal differ from this code's; stopping")
    log(f"source meta run {src} ({s_meta['events']:,} events, run {s_meta['run_utc']})")
    df = pd.read_csv(src / "events.csv")
    if len(df) != int(s_meta["events"]):
        raise SystemExit(f"events.csv has {len(df)} rows, the source run says {s_meta['events']}")
    sym_rows = pd.read_csv(src / "symbols.csv")
    gmx_dir, tv_dir = (Path(args.candle_dirs[0]), Path(args.candle_dirs[1])) if args.candle_dirs else (Path(s_meta["gmx_dir"]), Path(s_meta["tradingview_dir"]))
    names = sorted(sym_rows["symbol"].astype(str))
    closes_all, skipped = CL.load_closes(gmx_dir, tv_dir, names, s_meta["timeframe"], 0)
    closes = {}
    for _, r in sym_rows.iterrows():
        s = str(r["symbol"])
        c = closes_all[s]
        c = c[c.index <= pd.Timestamp(r["last"])]            # the source run's last bar; later candles did not exist for it
        if len(c) != int(r["bars"]):
            raise SystemExit(f"{s}: {len(c)} candles up to {r['last']}, the source run used {int(r['bars'])}; the candle files changed, stopping")
        closes[s] = c
    log(f"loaded {len(closes)} symbols, cut at the source run's last bars")
    df = T.extend_events(df, closes)
    n_mom = int((df["mom_side"] != 0).sum())
    log(f"events {len(df):,}; MOM events {n_mom:,} (long {int((df['mom_side'] == 1).sum()):,}, short {int((df['mom_side'] == -1).sum()):,}); y_dir share {df['y_dir'].mean():.3f}")
    ext_cols = ["symbol", "pos", "t0", "ret480", "ret720", "ret960", "mom_side", "ret_long", "y_dir"]
    df[ext_cols].to_csv(out_dir / "events_ext.csv", index=False, float_format="%.12g")

    win_t = {k_: (int(v[0]), int(v[1])) for k_, v in s_meta["window_epoch"].items()}
    windows_txt = s_meta["window_notes"]
    cost_frac, h = float(s_params["cost_frac"]), int(s_params["h"])
    forest_kw = dict(n_estimators=args.n_estimators, bootstrap=args.bootstrap, max_draws=args.max_draws)

    log("CPCV ...")
    preds, smet, cmeta = T.run_cpcv_gate(df, args.n_groups, args.k, args.embargo_pct, forest_kw, args.seed, log)
    ev = T.evaluate_three(df, preds, cost_frac, args.step, h, args.seed, win_t, args.prior_trials)
    log("walk-forward ...")
    wpred, wmet = T.run_walk_forward_gate(df, args.cutoff, forest_kw, args.seed + 500, log)
    wev = T.evaluate_three(df, wpred, cost_frac, args.step, h, args.seed + 7, win_t, args.prior_trials)

    preds.to_csv(out_dir / "cpcv_predictions.csv", index=False, float_format="%.12g")
    smet.to_csv(out_dir / "split_metrics.csv", index=False)
    wpred.to_csv(out_dir / "walk_forward_predictions.csv", index=False, float_format="%.12g")
    ev["daily"].to_csv(out_dir / "daily.csv", index=False)
    wev["daily"].to_csv(out_dir / "walk_forward_daily.csv", index=False)
    pd.DataFrame(ev["paths"]).to_csv(out_dir / "paths.csv", index=False)
    pd.DataFrame(wev["paths"]).to_csv(out_dir / "walk_forward_paths.csv", index=False)
    after = RB.integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    params = {"h": h, "cost_frac": cost_frac, "n_groups": args.n_groups, "k": args.k, "embargo_pct": args.embargo_pct, "step": args.step, "forest": forest_kw,
              "seed": args.seed, "cutoff": args.cutoff, "prior_trials": args.prior_trials, "dir_features": T.DIR_FEATURES, "mom_features": T.MOM_FEATURES,
              "mom_lookbacks": list(T.MOM_LOOKBACKS), "min_weight_fraction_leaf": 0.05, "variants": list(T.VARIANTS), "dsr_variants": list(T.DSR_VARIANTS)}
    meta = {**cmeta, "run_utc": started.isoformat(timespec="seconds"), "source_folder": str(src), "source_events_md5": RB.md5_file(src / "events.csv"),
            "source_summary_md5": RB.md5_file(src / "summary_numbers.json"), "events": int(len(df)), "mom_events": n_mom, "mom_long": int((df["mom_side"] == 1).sum()),
            "mom_short": int((df["mom_side"] == -1).sum()), "windows": s_meta["windows"], "window_epoch": s_meta["window_epoch"], "window_notes": windows_txt,
            "gmx_dir": str(gmx_dir.resolve()), "tradingview_dir": str(tv_dir.resolve()), "timeframe": s_meta["timeframe"],
            "script_md5": {n: RB.md5_file(HERE / n) for n in ("book_stats.py", "meta_events.py", "meta_model.py", "meta_eval.py", "candle_loader.py", "three_algo.py", "run_three_algos.py")}}
    summ = {"meta": meta, "params": params,
            "params_shown": {"symbols": len(closes), "round-trip cost (bp)": f"{cost_frac * 1e4:.1f}", "barriers": "as the source run (+/- 2.0 x daily vol, vertical 101 bars)",
                             "forest": f"{args.n_estimators} trees, {args.bootstrap} bootstrap, max {args.max_draws:,} draws per tree", "bet-size step": args.step, "seed": args.seed,
                             "prior trials for the deflated Sharpe": args.prior_trials, "windows": " | ".join(f"{k_}: {v}" for k_, v in windows_txt.items())},
            "integrity_unchanged": before == after,
            "cpcv": {"auc_dir_mean": float(smet["auc_dir"].mean()), "auc_dir_min": float(smet["auc_dir"].min()), "auc_dir_max": float(smet["auc_dir"].max()),
                     "auc_mom_mean": float(smet["auc_mom"].mean()), "auc_mom_min": float(smet["auc_mom"].min()), "auc_mom_max": float(smet["auc_mom"].max()),
                     "share_dir_mean": float(smet["share_dir_gt_half"].mean()), "share_mom_mean": float(smet["share_mom_gt_half"].mean())},
            "eval": {k_: v for k_, v in ev.items() if k_ not in ("daily", "paths")}, "walk_forward": wmet,
            "walk_forward_eval": {k_: v for k_, v in wev.items() if k_ not in ("daily", "paths")}}
    (out_dir / "summary_numbers.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    md = render(summ)
    (out_dir / "SUMMARY.md").write_text(md, encoding="utf-8")
    log(f"done in {time.time() - started.timestamp():.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
