#!/usr/bin/env python
"""Meta-labeling pipeline built to the book's construction chapters (READ-ONLY diagnostic): CUSUM events (Ch.2), triple-barrier meta-labels (Ch.3),
uniqueness / return-attribution weights and sequential bootstrap (Ch.4), fractional-differentiation feature (Ch.5), a bagged forest (Ch.6),
purged / embargoed CV (Ch.7), MDA and SFI importance (Ch.8), bet sizing (Ch.10), combinatorial purged CV, PSR and deflated Sharpe (Ch.12, 14).
Deviations from the book are listed in SUMMARY.md and meta_eval.DEVIATIONS.

Same pipeline as run_meta_pipeline.py, run on ONE venue's hourly candles (exported by export_venue_candles.py) instead of GMX + TradingView.
It loads no model, changes no threshold, calibration file, model file or execution code, and writes only to runtime/book_meta_<venue>/<UTC timestamp>/.

Run in the `javier` conda env, from the project root:
    python tools/book_study/run_meta_venue.py --venue kraken --candle-dir runtime/venue_candles/kraken_1h --sell-off 2026-09-27 2026-10-08
    python tools/book_study/verify_meta_venue.py runtime/book_meta_kraken/<timestamp>
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
import meta_eval as ME
import meta_events as E
import meta_model as MM
import run_book_study as RB
import venue_loader as VL

SEED = 20261013          # fixed before any result was seen
PRIOR_TRIALS = 101       # earlier looks in this research (all on GMX / TradingView candles): 100 before the market-features run, plus that run. This run is look 102 and the data are new.
COST_BP_PER_SIDE = 12.0  # trading_bot_v4/signals/ma200_asset_qualifier.py: ENTRY_COST = EXIT_COST = 12 / 10000 (Kraken) -> 24 bp round trip


def venue_deviations(venue: str, prior_trials: int):
    """The shared deviation list with the entries that talk about GMX / TradingView candles replaced by statements true for this venue's files."""
    out = list(ME.DEVIATIONS)

    def swap(prefix, new):
        hits = [i for i, t in enumerate(out) if t.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit(f"deviation entry {prefix!r} matched {len(hits)} times")
        out[hits[0]] = new

    swap("Time bars, not volume or dollar bars", f"Time bars, not volume or dollar bars (Ch.2): hourly candles of {venue} are used as they are; volume bars were not built in this run.")
    swap("Not built (need volume or trade data", "Not built in this run: Kyle, Amihud and Hasbrouck lambdas, VPIN, PIN, volume-based structural-break features. Also not built: HRP allocation (Ch.16), the Chapter 20-22 material.")
    swap("DSR trial count:", f"DSR trial count: there is one model here, but earlier diagnostics in this research (on GMX / TradingView candles) looked at many momentum lookbacks, holds, states and feature sets; "
                              f"the DSR is shown for several trial counts, including the one that counts those earlier looks ({prior_trials}, this run being look {prior_trials + 1}). The 30-day momentum side was chosen on those earlier candles.")
    return out


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(summ) -> str:
    m = summ["meta"]
    L = ["# Meta-labeling pipeline built to the book's construction chapters (read-only diagnostic)", ""]
    L.append(f"Run (UTC): {m['run_utc']}. Models/calibration untouched: {summ['integrity_unchanged']}.")
    L.append("")
    L.append("Primary side: sign of the trailing 30-day return at each CUSUM event (in-sample choice, declared). Meta-label: did the side's trade end in profit "
             "at the first barrier touch. The model sees fractionally differentiated price, normalised trailing returns, volatility, the market state and the side; "
             "it outputs a probability that becomes a bet size. 'primary' takes every event with size 1, 'meta' sizes by the model, 'placebo' shuffles the meta sizes "
             "among the same events (what the sizing distribution alone earns).")
    L.append("")
    for k, v in summ["params_shown"].items():
        L.append(f"- {k}: {v}")
    L.append("")
    cp = summ["cpcv"]
    L.append(f"## Combinatorial purged CV: {m['n_groups']} groups, {m['k']} test groups -> {m['n_splits']} splits, {m['n_paths']} backtest paths")
    L.append("")
    L.append(f"Out-of-sample AUC per split: mean {fmt(cp['auc_mean'])}, min {fmt(cp['auc_min'])}, max {fmt(cp['auc_max'])} (0.5 = no information). "
             f"Share of events with p > 0.5 (taken): {fmt(cp['taken_mean'], 2)}. Average uniqueness of training events: {fmt(cp['avg_u_mean'])}.")
    L.append("")
    L.append("| variant | mean bp per candidate event | daily Sharpe (mean over paths) | annualised | paths with Sharpe < 0 | mean PSR | share of events taken | mean size |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v in ("primary", "meta", "placebo"):
        s = summ["eval"]["summary"][v]
        L.append(f"| {v} | {fmt(s['mean_bp_per_candidate'], 1)} | {fmt(s['mean_sr_daily'], 4)} | {fmt(s['mean_sr_annual'], 2)} | {s['paths_sr_negative']}/{s['paths']} | {fmt(s['mean_psr'], 2)} | {fmt(s['taken_share'], 2)} | {fmt(s['mean_size'], 3)} |")
    mp, mq = summ["eval"]["summary"]["meta_minus_placebo"], summ["eval"]["summary"]["meta_minus_primary"]
    L.append("")
    L.append(f"Meta minus placebo (does the model's ranking add anything beyond the size distribution): Sharpe higher on {mp['paths_sr_higher']}/{mp['paths']} paths, "
             f"mean daily-Sharpe difference {fmt(mp['mean_sr_daily_diff'], 4)}, mean bp difference {fmt(mp['mean_bp_diff'], 2)}. "
             f"Meta minus primary: Sharpe higher on {mq['paths_sr_higher']}/{mq['paths']} paths, mean difference {fmt(mq['mean_sr_daily_diff'], 4)}.")
    L.append("")
    d = summ["eval"]["dsr"]
    L.append("Deflated Sharpe of the meta variant (probability the true Sharpe exceeds the best of N unskilled trials; N counts earlier looks at this history):")
    L.append("")
    L.append("| trials N | SR* (per day) | DSR |")
    L.append("|---|---|---|")
    for n_tr, v in d["by_trials"].items():
        L.append(f"| {n_tr} | {fmt(v['sr_star'], 4)} | {fmt(v['dsr'], 3)} |")
    L.append("")
    L.append(f"(mean daily Sharpe {fmt(d['sr_daily_mean'], 4)}, effective sample {fmt(d['t_eff'], 0)} days, Sharpe variance used {fmt(d['var_sr_used'], 6)}: the larger of the variance across paths and the asymptotic sampling variance.)")
    L.append("")
    wf = summ["walk_forward"]
    L.append(f"## Walk-forward check (single fit, labels ending before {wf['cutoff']}, tested on later events)")
    L.append("")
    L.append(f"Train events {wf['n_train']:,}, test events {wf['n_test']:,}, AUC {fmt(wf['auc'])}, share with p > 0.5 {fmt(wf['share_p_gt_half'], 2)}.")
    L.append("")
    L.append("| variant | mean bp per candidate event | daily Sharpe | annualised | PSR | share taken |")
    L.append("|---|---|---|---|---|---|")
    for v in ("primary", "meta", "placebo"):
        s = summ["walk_forward_eval"]["summary"][v]
        L.append(f"| {v} | {fmt(s['mean_bp_per_candidate'], 1)} | {fmt(s['mean_sr_daily'], 4)} | {fmt(s['mean_sr_annual'], 2)} | {fmt(s['mean_psr'], 2)} | {fmt(s['taken_share'], 2)} |")
    L.append("")
    L.append("## Reporting windows (events entered inside; mean bp per candidate event; averaged over backtest paths / the walk-forward fit)")
    L.append("")
    L.append("| window | source | primary | meta | placebo | events per path |")
    L.append("|---|---|---|---|---|---|")
    for src, key in (("CPCV", "eval"), ("walk-forward", "walk_forward_eval")):
        for wn in m["windows"]:
            w = summ[key]["windows"]
            L.append(f"| {wn} | {src} | {fmt(w[f'primary|{wn}']['mean_bp_per_candidate'], 1)} | {fmt(w[f'meta|{wn}']['mean_bp_per_candidate'], 1)} | "
                     f"{fmt(w[f'placebo|{wn}']['mean_bp_per_candidate'], 1)} | {fmt(w[f'primary|{wn}']['events_per_path'], 0)} |")
    L.append("")
    L.append("Window lines carry no intervals (a window is a few correlated days).")
    L.append("")
    L.append("## By market state (state 1 = equal-weight index above its own 200-day mean at the event; 0 = below)")
    L.append("")
    L.append("Events entered in each window, by state (per path): " + "; ".join(f"{k_.replace('|state', ' state ')}: {v:.0f}" for k_, v in sorted(summ["eval"]["window_state_events"].items())))
    L.append("")
    for src, key in (("CPCV", "eval"), ("walk-forward", "walk_forward_eval")):
        bs = summ[key]["by_state"]
        L.append(f"{src}: mean bp per candidate event (net of cost where a bet is taken), pooled over {'paths' if key == 'eval' else 'the fit'}.")
        L.append("")
        L.append("| state | events/path | primary | meta | meta mean size | primary, side +1 (long) | primary, side -1 (short) |")
        L.append("|---|---|---|---|---|---|---|")
        for stv in (1, 0):
            pr, me = bs[f"primary|state{stv}"], bs[f"meta|state{stv}"]
            lo_, sh_ = bs[f"primary|state{stv}|side+1"], bs[f"primary|state{stv}|side-1"]
            L.append(f"| {stv} | {pr['events_per_path']:.0f} | {fmt(pr['mean_bp_per_candidate'], 1)} | {fmt(me['mean_bp_per_candidate'], 1)} | {fmt(me['mean_size'], 3)} | "
                     f"{fmt(lo_['mean_bp_per_candidate'], 1)} ({lo_['events_per_path']:.0f}) | {fmt(sh_['mean_bp_per_candidate'], 1)} ({sh_['events_per_path']:.0f}) |")
        L.append("")
    L.append("")
    L.append("## Feature importance (purged 5-fold)")
    L.append("")
    L.append("| feature | MDA mean (higher = more used) | MDA s.e. | SFI neg-log-loss (higher = better; 0.693 = no information) | SFI s.e. |")
    L.append("|---|---|---|---|---|")
    for f_ in E.FEATURES:
        a, b = summ["mda"].get(f_, {}), summ["sfi"].get(f_, {})
        L.append(f"| {f_} | {fmt(a.get('mean'), 4)} | {fmt(a.get('std'), 4)} | {fmt(b.get('mean'), 4)} | {fmt(b.get('std'), 4)} |")
    L.append("")
    L.append(f"MDA baseline score (neg-log-loss, sample-weighted): {fmt(summ['mda_baseline'], 4)}. 'No information' for balanced classes is -0.6931.")
    L.append("")
    fd = summ["fracdiff"]
    L.append(f"## Fractional differentiation: d per symbol, mean {fmt(fd['d_mean'], 2)}, median {fmt(fd['d_median'], 2)}, min {fmt(fd['d_min'], 2)}, max {fmt(fd['d_max'], 2)} "
             f"({fd['symbols']} symbols; correlation of the differentiated series with log price, mean {fmt(fd['corr_mean'], 3)}).")
    L.append("")
    L.append("## Deviations from the book and limits")
    L.append("")
    for i, n in enumerate(ME.DEVIATIONS, 1):
        L.append(f"{i}. {n}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, help="label of the venue the candles come from, e.g. kraken")
    ap.add_argument("--candle-dir", required=True, help="folder with <SYMBOL>_1h.csv files")
    ap.add_argument("--cost-bp-per-side", type=float, default=COST_BP_PER_SIDE)
    ap.add_argument("--out", default=None, help="default runtime/book_meta_<venue>")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-draws", type=int, default=10000)
    ap.add_argument("--bootstrap", choices=("sequential", "standard"), default="sequential")
    ap.add_argument("--embargo-pct", type=float, default=0.01)
    ap.add_argument("--step", type=float, default=0.1, help="bet-size discretisation (Snippet 10.3)")
    ap.add_argument("--cutoff", default="2024-01-01", help="walk-forward cutoff (UTC)")
    ap.add_argument("--prior-trials", type=int, default=PRIOR_TRIALS)
    ap.add_argument("--min-years", type=float, default=0.5)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--sell-off", nargs=2, default=None, metavar=("START", "END"))
    args = ap.parse_args()

    if args.quick:
        args.n_groups, args.n_estimators, args.max_draws = 4, 10, 3000
    h, ptsl = 100, (2.0, 2.0)
    forest_kw = dict(n_estimators=args.n_estimators, bootstrap=args.bootstrap, max_draws=args.max_draws)
    started = datetime.now(timezone.utc)
    out_dir = ROOT / (args.out or f"runtime/book_meta_{args.venue}") / started.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=False)
    before = RB.integrity_snapshot()
    log_file = open(out_dir / "run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    min_bars = int(args.min_years * 365 * 24)
    candle_dir = Path(args.candle_dir)
    if not candle_dir.is_absolute():
        candle_dir = ROOT / candle_dir
    names = VL.list_symbols(candle_dir, "1h")
    timeframe = "1h"
    fee_bps, slip_bps = float(args.cost_bp_per_side), 0.0
    cost_frac = 2.0 * (fee_bps + slip_bps) / 1e4
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        names = [n for n in names if n.upper() in wanted]
    if args.limit:
        names = names[: args.limit]
    log(f"candidate symbols {len(names)}; round-trip cost {cost_frac * 1e4:.1f} bp")

    closes, skipped = VL.load_closes(candle_dir, names, timeframe, min_bars)
    if len(closes) < 10:
        raise SystemExit(f"only {len(closes)} usable symbols; need at least 10")
    first, last = min(c.index[0] for c in closes.values()), max(c.index[-1] for c in closes.values())
    log(f"loaded {len(closes)} symbols; earliest candle {first}, latest {last}")

    ohlc, _skip = VL.load_ohlc(candle_dir, sorted(closes), timeframe, 0)
    for s_ in closes:
        # the High / Low frames must sit on exactly the bars of the close series the rest of the pipeline uses
        if s_ not in ohlc or not ohlc[s_].index.equals(closes[s_][closes[s_] > 0].index) or not np.allclose(ohlc[s_]["close"].to_numpy(), closes[s_][closes[s_] > 0].to_numpy().astype(float), rtol=1e-9, atol=0.0):
            raise SystemExit(f"{s_}: High/Low candles do not line up with the close series used by the loader; stopping")
    df, einfo = E.build_meta_events(closes, ohlc, h=h, ptsl=ptsl, cost_frac=cost_frac, log=log)
    df = df.sort_values(["t0", "symbol"], kind="stable").reset_index(drop=True)
    log(f"events kept {len(df):,}")

    # windows
    windows, notes = {}, {}
    windows["june_2026_ai_correction"] = C.JUNE_2026
    notes["june_2026_ai_correction"] = "calendar month 2026-06-01 .. 2026-06-30"
    if args.sell_off:
        s0, s1, src = pd.Timestamp(args.sell_off[0]), pd.Timestamp(args.sell_off[1]), "command line"
    else:
        found = RB.latest_baseline_sell_off()
        s0, s1, src = found if found else (None, None, None)
    if s0 is not None:
        windows["sell_off_post_holdout"] = RB.day_window(s0, s1)
        notes["sell_off_post_holdout"] = f"{s0} -> {s1} (whole UTC days), from {src}"
    idx_series = C.market_state(closes, df["t0"].to_numpy()[:1])[1]
    fall = RB.deepest_fall(idx_series)
    if fall:
        windows["deepest_90d_fall_full_history"] = RB.day_window(fall[0], fall[1])
        notes["deepest_90d_fall_full_history"] = f"{fall[0]} -> {fall[1]}, {fall[2]:.1f}% on the equal-weight index (whole UTC days)"
    win_t = {k_: (int(pd.Timestamp(v[0]).timestamp()), int(pd.Timestamp(v[1]).timestamp())) for k_, v in windows.items()}
    for k_, v in notes.items():
        log(f"window {k_}: {v}")

    cover = {k_: int(sum(1 for c_ in closes.values() if ((c_.index >= pd.Timestamp(v[0])) & (c_.index < pd.Timestamp(v[1]))).any())) for k_, v in windows.items()}
    for k_, n_ in cover.items():
        log(f"window {k_}: {n_} of {len(closes)} symbols have at least one bar inside")
    ME.DEVIATIONS = venue_deviations(args.venue, int(args.prior_trials)) + [
        f"Candle coverage of the reporting windows (symbols with at least one bar inside, of {len(closes)} used): " + "; ".join(f"{k_} {n_}" for k_, n_ in cover.items())
        + ". The market-state feature needs at least 10 symbols with a return in the same hour, so events cannot be built where fewer than 10 symbols have bars; a window with no entered events is reported n/a."]

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
        X, y, w, sym, starts, ends, t0, t1, u = data
        avg_u_fn = lambda tr: float(u[tr].mean())                                         # noqa: E731
        mk = lambda: MM.BaggedForest(seed=args.seed + 900, **{**forest_kw, "n_estimators": max(10, args.n_estimators // 2)})
        mda, mda_base = MM.mda_importance(X, y, w, sym, starts, ends, avg_u_fn, t0, t1, E.FEATURES, 5, args.embargo_pct, mk, args.seed)
        mk2 = lambda: MM.BaggedForest(seed=args.seed + 901, **{**forest_kw, "bootstrap": "standard", "n_estimators": max(10, args.n_estimators // 2)})
        sfi = MM.sfi_importance(X, y, w, sym, starts, ends, avg_u_fn, t0, t1, E.FEATURES, 5, args.embargo_pct, mk2)
        mda_d, sfi_d = mda.to_dict("index"), sfi.to_dict("index")
        mda.to_csv(out_dir / "mda.csv")
        sfi.to_csv(out_dir / "sfi.csv")

    # ---- write ----
    df.to_csv(out_dir / "events.csv", index=False, float_format="%.12g")
    preds.to_csv(out_dir / "cpcv_predictions.csv", index=False, float_format="%.12g")
    smet.to_csv(out_dir / "split_metrics.csv", index=False)
    wpred.to_csv(out_dir / "walk_forward_predictions.csv", index=False, float_format="%.12g")
    ev["daily"].to_csv(out_dir / "daily.csv", index=False)
    pd.DataFrame(ev["paths"]).to_csv(out_dir / "paths.csv", index=False)
    pd.DataFrame(einfo["per_symbol"]).to_csv(out_dir / "symbols.csv", index=False)
    fdf = pd.DataFrame(einfo["symbols"])
    fdf.to_csv(out_dir / "fracdiff.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False)
    after = RB.integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    params = {"h": h, "ptsl": list(ptsl), "cost_frac": cost_frac, "n_groups": args.n_groups, "k": args.k, "embargo_pct": args.embargo_pct, "step": args.step,
              "forest": forest_kw, "seed": args.seed, "cutoff": args.cutoff, "prior_trials": args.prior_trials, "features": E.FEATURES,
              "ffd_thres": E.FFD_THRES, "side_signal": E.SIDE_SIGNAL,
              "micro": {"sadf": [E.SADF_WINDOW, E.SADF_MINSL, E.SADF_LAGS], "entropy": [E.ENT_WINDOW, E.ENT_WORD], "cs_sl": E.CS_SL}, "min_weight_fraction_leaf": 0.05}
    meta = {**cmeta, "run_utc": started.isoformat(timespec="seconds"), "timeframe": timeframe, "symbols_used": len(closes), "events": int(len(df)),
            "earliest_candle": str(first), "latest_candle": str(last), "windows": {k_: [str(v[0]), str(v[1])] for k_, v in windows.items()},
            "window_epoch": win_t, "window_notes": notes, "venue": args.venue, "window_symbol_coverage": cover, "candle_dir": str(candle_dir.resolve()), "cost_bp_per_side": float(args.cost_bp_per_side),
            "script_md5": {n: RB.md5_file(HERE / n) for n in ("book_stats.py", "book_study_core.py", "fracdiff_core.py", "micro_features.py", "venue_loader.py", "meta_events.py", "meta_model.py", "meta_eval.py", "run_meta_venue.py")}}
    summ = {"meta": meta, "params": params, "events_info": {k_: v for k_, v in einfo.items() if k_ not in ("symbols", "per_symbol")},
            "params_shown": {"symbols used": len(closes), "events": f"{len(df):,}", "venue": args.venue, "round-trip cost (bp)": f"{cost_frac * 1e4:.1f}",
                             "barriers": f"+/- {ptsl[0]} x daily vol, vertical {h + 1} bars", "forest": f"{args.n_estimators} trees, {args.bootstrap} bootstrap, max {args.max_draws:,} draws per tree",
                             "bet-size step": args.step, "seed": args.seed, "windows": " | ".join(f"{k_}: {v}" for k_, v in notes.items())},
            "integrity_unchanged": before == after,
            "cpcv": {"auc_mean": float(smet["auc"].mean()), "auc_min": float(smet["auc"].min()), "auc_max": float(smet["auc"].max()),
                     "taken_mean": float(smet["share_p_gt_half"].mean()), "avg_u_mean": float(smet["avg_uniqueness"].mean())},
            "eval": {k_: v for k_, v in ev.items() if k_ not in ("daily",)}, "walk_forward": wmet,
            "walk_forward_eval": {k_: v for k_, v in wev.items() if k_ not in ("daily",)},
            "mda": mda_d, "sfi": sfi_d, "mda_baseline": mda_base,
            "fracdiff": {"symbols": int(len(fdf)), "d_mean": float(fdf["d"].mean()), "d_median": float(fdf["d"].median()), "d_min": float(fdf["d"].min()),
                         "d_max": float(fdf["d"].max()), "corr_mean": float(fdf["corr"].mean())}}
    (out_dir / "summary_numbers.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    md = render(summ)
    (out_dir / "SUMMARY.md").write_text(md, encoding="utf-8")
    log(f"done in {time.time() - started.timestamp():.0f}s -> {out_dir}")
    print("\n" + md[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
