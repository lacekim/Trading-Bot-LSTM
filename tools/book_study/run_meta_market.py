#!/usr/bin/env python
"""Meta-labeling pipeline with the market-level Chapter 17-18 features added (READ-ONLY diagnostic).  Everything is the verified pipeline of
run_meta_pipeline.py (same events, labels, weights, forest, sequential bootstrap, combinatorial purged CV, walk-forward, MDA / SFI, bet sizing,
deflated Sharpe, same seed) with three more features: the SADF (Ch.17) and the plug-in and Kontoyiannis entropies (Ch.18) of the equal-weight index, computed with
the declared per-symbol settings (market_features.py).  It reads a finished and separately verified meta-pipeline run as its event source, loads no model, changes no
threshold, calibration file, model file or execution code, and writes only to runtime/book_meta_market/<UTC timestamp>/.

    python tools/book_study/run_meta_market.py --source runtime/book_meta/<folder of a verified meta run>
    python tools/book_study/verify_meta_market.py runtime/book_meta_market/<timestamp> runtime/book_meta/<same folder>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
import market_features as MKF
import meta_eval as ME
import meta_events as E
import meta_model as MM
import run_book_study as RB
import run_meta_pipeline as RM

SEED = 20261013                  # the seed of the 14-feature run: the only difference between the two runs is the three features
PRIOR_TRIALS = 100               # earlier looks: 83 after the 14-feature run + 4 + 1 (three-algorithm study) + 6 + 6 (regime and fast-drop studies) = 100
EXTRA_DEVIATIONS = [
    "Market-level features (this run): SADF and the two entropies are computed on the equal-weight index of the 200-day state (daily log of its last hourly level of each UTC day; the index starts when 10 symbols exist), with exactly the per-symbol settings; each event takes the last completed UTC day. The book applies these tools to a price series; applying them to a market index is this study's choice, and the three features are not tuned.",
    "The comparison with the 14-feature run is one run against one run with the same seed, events and folds; a difference in AUC or Sharpe of a few thousandths is inside what a different seed would move, and the 14-feature run is itself one of the looks counted in the deflated Sharpe.",
]


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(summ, src_summ) -> str:
    md = RM.render(summ)
    md = md.replace("# Meta-labeling pipeline built to the book's construction chapters (read-only diagnostic)",
                    "# Meta-labeling pipeline with market-level Chapter 17-18 features (read-only diagnostic)", 1)
    m = summ["meta"]
    a, b = src_summ, summ
    L = ["## Compared with the 14-feature run (same events, folds and seed; only the three market features differ)", ""]
    L.append(f"Source run: {summ['market']['source_folder']} (events.csv md5 {summ['market']['source_events_md5']}). Added features: {', '.join(MKF.MARKET_FEATURES)}.")
    L.append("")
    L.append("| quantity | 14 features | 17 features | difference |")
    L.append("|---|---|---|---|")

    def row(name, x, y, nd):
        d = (y - x) if (x is not None and y is not None and np.isfinite(x) and np.isfinite(y)) else float("nan")
        L.append(f"| {name} | {fmt(x, nd)} | {fmt(y, nd)} | {fmt(d, nd)} |")

    row("CPCV mean AUC", a["cpcv"]["auc_mean"], b["cpcv"]["auc_mean"], 4)
    row("CPCV meta: bp per candidate", a["eval"]["summary"]["meta"]["mean_bp_per_candidate"], b["eval"]["summary"]["meta"]["mean_bp_per_candidate"], 1)
    row("CPCV meta: daily Sharpe", a["eval"]["summary"]["meta"]["mean_sr_daily"], b["eval"]["summary"]["meta"]["mean_sr_daily"], 4)
    row("CPCV meta: annualised", a["eval"]["summary"]["meta"]["mean_sr_annual"], b["eval"]["summary"]["meta"]["mean_sr_annual"], 2)
    row("walk-forward AUC", a["walk_forward"]["auc"], b["walk_forward"]["auc"], 4)
    row("walk-forward meta: bp per candidate", a["walk_forward_eval"]["summary"]["meta"]["mean_bp_per_candidate"], b["walk_forward_eval"]["summary"]["meta"]["mean_bp_per_candidate"], 1)
    row("walk-forward meta: daily Sharpe", a["walk_forward_eval"]["summary"]["meta"]["mean_sr_daily"], b["walk_forward_eval"]["summary"]["meta"]["mean_sr_daily"], 4)
    for wn in m["windows"]:
        row(f"CPCV meta bp per candidate, {wn}", a["eval"]["windows"][f"meta|{wn}"]["mean_bp_per_candidate"], b["eval"]["windows"][f"meta|{wn}"]["mean_bp_per_candidate"], 1)
    for wn in m["windows"]:
        row(f"walk-forward meta bp per candidate, {wn}", a["walk_forward_eval"]["windows"][f"meta|{wn}"]["mean_bp_per_candidate"], b["walk_forward_eval"]["windows"][f"meta|{wn}"]["mean_bp_per_candidate"], 1)
    L.append("")
    L.append("MDA rank of the market features among the 17 (1 = most used; the 14-feature run had no such rows):")
    L.append("")
    L.append("| feature | MDA rank | MDA mean | SFI neg-log-loss |")
    L.append("|---|---|---|---|")
    rank = {f_: i + 1 for i, f_ in enumerate(sorted(E.FEATURES, key=lambda f_: -summ["mda"][f_]["mean"]))} if summ["mda"] else {}
    for f_ in MKF.MARKET_FEATURES:
        L.append(f"| {f_} | {rank.get(f_, 'n/a')} | {fmt(summ['mda'].get(f_, {}).get('mean'), 4)} | {fmt(summ['sfi'].get(f_, {}).get('mean'), 4)} |")
    L.append("")
    cut = md.index("## Deviations from the book and limits")
    return md[:cut] + "\n".join(L) + "\n" + md[cut:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder of a finished and verified run_meta_pipeline.py run")
    ap.add_argument("--out", default="runtime/book_meta_market")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-draws", type=int, default=10000)
    ap.add_argument("--bootstrap", choices=("sequential", "standard"), default="sequential")
    ap.add_argument("--embargo-pct", type=float, default=0.01)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--cutoff", default="2024-01-01")
    ap.add_argument("--prior-trials", type=int, default=PRIOR_TRIALS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--candle-dirs", nargs=2, default=None, metavar=("GMX_DIR", "TV_DIR"))
    args = ap.parse_args()
    if args.quick:
        args.n_groups, args.n_estimators, args.max_draws = 4, 10, 3000
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
    if s_params["features"] != MKF.E.FEATURES[:14] and s_params["features"] != E.FEATURES:
        raise SystemExit("the source run does not use the 14-feature list")
    h, cost_frac = int(s_params["h"]), float(s_params["cost_frac"])
    df = pd.read_csv(src / "events.csv")
    if len(df) != int(s_meta["events"]):
        raise SystemExit(f"events.csv has {len(df)} rows, the source run says {s_meta['events']}")
    log(f"source meta run {src} ({len(df):,} events, run {s_meta['run_utc']})")
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
    t0 = df["t0"].to_numpy(np.int64)
    state, idx, _ma = C.market_state(closes, t0)
    if not np.array_equal(state.astype(float), df["mkt_state"].to_numpy(np.float64)):
        raise SystemExit("the 200-day state rebuilt here differs from the source run's mkt_state; stopping")
    df = MKF.attach(df, idx)
    nan_ct = {f_: int(df[f_].isna().sum()) for f_ in MKF.MARKET_FEATURES}
    if any(nan_ct.values()):
        raise SystemExit(f"events without a market feature: {nan_ct}; events would have to be dropped and the weights recomputed, which this runner does not do")
    MKF.use_market_features()
    log("market features attached: " + ", ".join(f"{f_} mean {df[f_].mean():.4f}" for f_ in MKF.MARKET_FEATURES))
    ME.DEVIATIONS.extend(EXTRA_DEVIATIONS)
    forest_kw = dict(n_estimators=args.n_estimators, bootstrap=args.bootstrap, max_draws=args.max_draws)
    win_t = {k_: (int(v[0]), int(v[1])) for k_, v in s_meta["window_epoch"].items()}

    log("CPCV ...")
    preds, smet, cmeta = ME.run_cpcv(df, args.n_groups, args.k, args.embargo_pct, forest_kw, args.seed, log)
    ev = ME.evaluate(df, preds, cost_frac, args.step, h, args.seed, win_t, args.prior_trials)
    log("walk-forward ...")
    wpred, wmet = ME.run_walk_forward(df, args.cutoff, forest_kw, args.seed + 500, log)
    wev = ME.evaluate(df, wpred, cost_frac, args.step, h, args.seed + 7, win_t, args.prior_trials)

    mda_d, sfi_d, mda_base = {}, {}, float("nan")
    if not args.skip_importance:
        log("MDA / SFI (purged 5-fold) ...")
        data = ME.prepare(df)
        X, y, w, sym, starts, ends, t0f, t1f, u = data
        avg_u_fn = lambda tr: float(u[tr].mean())                                         # noqa: E731
        mk = lambda: MM.BaggedForest(seed=args.seed + 900, **{**forest_kw, "n_estimators": max(10, args.n_estimators // 2)})
        mda, mda_base = MM.mda_importance(X, y, w, sym, starts, ends, avg_u_fn, t0f, t1f, E.FEATURES, 5, args.embargo_pct, mk, args.seed)
        mk2 = lambda: MM.BaggedForest(seed=args.seed + 901, **{**forest_kw, "bootstrap": "standard", "n_estimators": max(10, args.n_estimators // 2)})
        sfi = MM.sfi_importance(X, y, w, sym, starts, ends, avg_u_fn, t0f, t1f, E.FEATURES, 5, args.embargo_pct, mk2)
        mda_d, sfi_d = mda.to_dict("index"), sfi.to_dict("index")
        mda.to_csv(out_dir / "mda.csv")
        sfi.to_csv(out_dir / "sfi.csv")

    df.to_csv(out_dir / "events.csv", index=False, float_format="%.12g")
    preds.to_csv(out_dir / "cpcv_predictions.csv", index=False, float_format="%.12g")
    smet.to_csv(out_dir / "split_metrics.csv", index=False)
    wpred.to_csv(out_dir / "walk_forward_predictions.csv", index=False, float_format="%.12g")
    ev["daily"].to_csv(out_dir / "daily.csv", index=False)
    pd.DataFrame(ev["paths"]).to_csv(out_dir / "paths.csv", index=False)
    for fn in ("symbols.csv", "fracdiff.csv", "skipped.csv"):
        shutil.copyfile(src / fn, out_dir / fn)
    after = RB.integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    params = {**s_params, "n_groups": args.n_groups, "k": args.k, "embargo_pct": args.embargo_pct, "step": args.step, "forest": forest_kw, "seed": args.seed,
              "cutoff": args.cutoff, "prior_trials": args.prior_trials, "features": list(E.FEATURES), "market_features": list(MKF.MARKET_FEATURES)}
    meta = {**s_meta, **cmeta, "run_utc": started.isoformat(timespec="seconds"), "events": int(len(df)), "gmx_dir": str(gmx_dir.resolve()), "tradingview_dir": str(tv_dir.resolve()),
            "script_md5": {n: RB.md5_file(HERE / n) for n in ("book_stats.py", "book_study_core.py", "fracdiff_core.py", "micro_features.py", "candle_loader.py", "meta_events.py", "meta_model.py", "meta_eval.py",
                                                              "run_meta_pipeline.py", "market_features.py", "run_meta_market.py")}}
    summ = {"meta": meta, "params": params, "events_info": s_summ["events_info"],
            "market": {"source_folder": str(src), "source_events_md5": RB.md5_file(src / "events.csv"), "source_summary_md5": RB.md5_file(src / "summary_numbers.json")},
            "params_shown": {"symbols used": len(closes), "events": f"{len(df):,}", "round-trip cost (bp)": f"{cost_frac * 1e4:.1f}",
                             "barriers": f"+/- {s_params['ptsl'][0]} x daily vol, vertical {h + 1} bars", "forest": f"{args.n_estimators} trees, {args.bootstrap} bootstrap, max {args.max_draws:,} draws per tree",
                             "bet-size step": args.step, "seed": args.seed, "features": f"{len(E.FEATURES)} (14 per-event + {', '.join(MKF.MARKET_FEATURES)})",
                             "trials before / after this run": f"{args.prior_trials} / {args.prior_trials + 1}", "windows": " | ".join(f"{k_}: {v}" for k_, v in s_meta["window_notes"].items())},
            "integrity_unchanged": before == after,
            "cpcv": {"auc_mean": float(smet["auc"].mean()), "auc_min": float(smet["auc"].min()), "auc_max": float(smet["auc"].max()),
                     "taken_mean": float(smet["share_p_gt_half"].mean()), "avg_u_mean": float(smet["avg_uniqueness"].mean())},
            "eval": {k_: v for k_, v in ev.items() if k_ not in ("daily",)}, "walk_forward": wmet,
            "walk_forward_eval": {k_: v for k_, v in wev.items() if k_ not in ("daily",)},
            "mda": mda_d, "sfi": sfi_d, "mda_baseline": mda_base, "fracdiff": s_summ["fracdiff"]}
    (out_dir / "summary_numbers.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    md = render(summ, s_summ)
    (out_dir / "SUMMARY.md").write_text(md, encoding="utf-8")
    log(f"done in {time.time() - started.timestamp():.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
