#!/usr/bin/env python
"""Meta-labeling pipeline (the book's construction chapters, READ-ONLY diagnostic) on DOLLAR BARS (Chapter 2's volume clock) of ONE venue.

Same pipeline as run_meta_venue.py - CUSUM events (Ch.2), triple-barrier meta-labels (Ch.3), uniqueness / attribution weights and sequential bootstrap (Ch.4), fractional differentiation (Ch.5),
bagged forest (Ch.6), purged CV (Ch.7), MDA / SFI (Ch.8), bet sizing (Ch.10), combinatorial purged CV, PSR and deflated Sharpe (Ch.12, 14) - with the event construction replaced by
bar_events.py (which does not assume one bar = one hour; see its docstring).  Everything downstream of the event table (meta_model.py, meta_eval.py) is the unchanged audited code.
It loads no model, changes no threshold, calibration file, model file or execution code, and writes only to runtime/book_meta_<venue>_dollar/<UTC timestamp>/.

Kraken : python tools/book_study/run_meta_bars.py --venue kraken --bars runtime/dollar_bars/kraken --cost-bp-per-side 12 --prior-trials 121 --sell-off 2026-09-27 2026-10-08
Alpaca : python tools/book_study/run_meta_bars.py --venue alpaca --bars runtime/dollar_bars/alpaca --cost-bp-per-side 10 --prior-trials <n> --universe runtime/book_meta_alpaca/20261012T023148Z/universe.csv --sell-off 2026-09-27 2026-10-08
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

import bar_events as BE
import bar_loader as BL
import book_study_core as C
import meta_eval as ME
import meta_events as E
import meta_model as MM
import run_book_study as RB
import run_meta_venue as RMV

SEED = 20261022          # fixed before any result of this module was seen
H_DAYS = BE.H_DAYS
HCAP = {"kraken": 600, "alpaca": 150}


def membership(universe_csv):
    """eligible(symbol, t0_epoch_array) -> bool: the symbol was a member at the last month-end strictly before the event's UTC date (the daily Alpaca runs' rule)."""
    u = pd.read_csv(universe_csv, keep_default_na=False, na_values=[""], dtype={"symbol": str})
    sets = {}
    for me, g in u.groupby("month_end"):
        sets[pd.Timestamp(me)] = set(g["symbol"])
    ends = sorted(sets)
    ends_ns = np.array(ends, dtype="datetime64[ns]")
    sets_l = [sets[m] for m in ends]

    def eligible(sym, t0):
        days = pd.DatetimeIndex(pd.to_datetime(np.asarray(t0, np.int64), unit="s")).floor("D").to_numpy().astype("datetime64[ns]")
        k = np.searchsorted(ends_ns, days, side="left") - 1
        return np.array([kk >= 0 and sym in sets_l[kk] for kk in k], dtype=bool)
    return eligible, ends[0] if ends else None


def bar_deviations(venue: str, prior_trials: int, h_days: float, hcap: int, extra: list):
    out = list(ME.DEVIATIONS)

    def swap(prefix, new):
        hits = [i for i, t in enumerate(out) if t.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit(f"deviation entry {prefix!r} matched {len(hits)} times")
        out[hits[0]] = new

    swap("Time bars, not volume or dollar bars", f"Dollar bars (Ch.2), not tick-built bars: {venue} rows are the finest data held (Kraken: 1-minute rows; Alpaca: 1-hour rows) and a row is never split, so a bar is a whole number of rows. "
         "A bar closes when about one average clock-hour of traded value has accumulated (trailing 30-day mean, project's own choice).")
    swap("The book builds bars from ticks;", "Events are the book's symmetric CUSUM filter on the dollar bars (threshold: causal one-bar sigma x sqrt(bars per calendar day)); see bar_events.py for each place the earlier hourly assumptions were replaced.")
    swap("Primary side = sign of the trailing 30-day", "Primary side = sign of the trailing 720-BAR return at each CUSUM event (30 days of average trading volume; on dollar bars not exactly 30 days), in every market state. "
         "The 720 was chosen after seeing the hourly history in earlier diagnostics (in-sample); the state is a model feature, not a filter.")
    swap("Barriers (Ch.3):", f"Barriers (Ch.3): simple-return path times side; horizontal barriers at 2 x the book's daily-volatility estimate (Snippet 3.1 as printed: return from the last bar at or before one calendar day earlier, span 100); "
         f"vertical barrier as in Snippet 3.4 a TIME limit of {h_days:g} calendar days (the first bar ending at or after it), with the path followed for at most {hcap} bars. Fixed before any result was seen; no search over them.")
    swap("Corwin-Schultz spread (Snippet 19.1)", "Corwin-Schultz spread (Snippet 19.1) on hourly High / Low of the reference series built from the same rows (equal time intervals), 24-row rolling mean, read at the last completed hour; negative alphas set to 0 as in the snippet. "
         "Snippet 19.2 is not used.")
    swap("Not built (need volume or trade data", "Not built in this run: Kyle, Amihud and Hasbrouck lambdas, VPIN, PIN, volume-based structural-break features (they need trade-level data). Also not built: the Chapter 20-22 material.")
    swap("DSR trial count:", f"DSR trial count: one model here, but earlier looks at this research (hourly candles of Kraken and Alpaca and the daily Alpaca bars) tried many lookbacks, holds, states and feature sets; the DSR is shown for several trial counts, "
         f"including the one that counts those earlier looks ({prior_trials}, this run being look {prior_trials + 1}). The 720-bar side was chosen on those earlier candles.")
    return out + extra


def render(summ, devs):
    md = RMV.render(summ)
    md = md.replace("Primary side: sign of the trailing 30-day return at each CUSUM event (in-sample choice, declared).",
                    "Primary side: sign of the trailing 720-bar return at each CUSUM event (30 days of average trading volume; in-sample choice, declared).")
    head = ["", "**Bars: DOLLAR BARS (volume clock), not hourly candles.** " + summ["params_shown"]["bars"], ""]
    i = md.index("\n", md.index("Run (UTC)"))
    return md[:i + 1] + "\n".join(head) + md[i + 1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=("kraken", "alpaca"))
    ap.add_argument("--bars", required=True, help="folder written by build_dollar_bars.py (bars/, ref_1h/, manifest.json)")
    ap.add_argument("--cost-bp-per-side", type=float, required=True)
    ap.add_argument("--prior-trials", type=int, required=True)
    ap.add_argument("--universe", default=None, help="Alpaca: universe.csv (month_end, rank, symbol) of the daily Alpaca run; events only for members of the last month-end before the event date")
    ap.add_argument("--out", default=None, help="default runtime/book_meta_<venue>_dollar")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-draws", type=int, default=10000)
    ap.add_argument("--bootstrap", choices=("sequential", "standard"), default="sequential")
    ap.add_argument("--embargo-pct", type=float, default=0.01)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--cutoff", default="2024-01-01")
    ap.add_argument("--h-days", type=float, default=H_DAYS)
    ap.add_argument("--hcap", type=int, default=None)
    ap.add_argument("--min-years", type=float, default=0.5)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--sell-off", nargs=2, default=None, metavar=("START", "END"))
    args = ap.parse_args()
    if args.quick:
        args.n_groups, args.n_estimators, args.max_draws = 4, 10, 3000
    hcap = args.hcap or HCAP[args.venue]
    h_eq = int(round(args.h_days * 24)) - 1            # meta_eval's (h + 1) / 24 is the horizon in days; here the horizon is a time, so h_eq + 1 = hours
    ptsl = (2.0, 2.0)
    forest_kw = dict(n_estimators=args.n_estimators, bootstrap=args.bootstrap, max_draws=args.max_draws)
    started = datetime.now(timezone.utc)
    out_dir = ROOT / (args.out or f"runtime/book_meta_{args.venue}_dollar") / started.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=False)
    before = RB.integrity_snapshot()
    log_file = open(out_dir / "run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    bars_dir = Path(args.bars)
    bars_dir = bars_dir if bars_dir.is_absolute() else ROOT / bars_dir
    man = BL.read_manifest(bars_dir)
    if man["venue"] != args.venue:
        raise SystemExit(f"the bar folder is for {man['venue']!r}, not {args.venue!r}")
    break_s = int(man["break_hours"]) * 3600
    cost_frac = 2.0 * float(args.cost_bp_per_side) / 1e4
    names = BL.list_symbols(bars_dir)
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        names = [n for n in names if n.upper() in wanted]
    if args.limit:
        names = names[: args.limit]
    log(f"candidate symbols {len(names)}; round-trip cost {cost_frac * 1e4:.1f} bp; vertical barrier {args.h_days:g} days (path cap {hcap} bars)")
    bars, skipped = BL.load_all(bars_dir, names, args.min_years, log)
    if len(bars) < 10:
        raise SystemExit(f"only {len(bars)} usable symbols; need at least 10")
    first, last = min(b.idx[0] for b in bars.values()), max(b.idx[-1] for b in bars.values())
    log(f"loaded {len(bars)} symbols ({sum(b.n for b in bars.values()):,} bars); earliest bar {first}, latest {last}")

    eligible, uni_start = (None, None)
    if args.universe:
        eligible, uni_start = membership(args.universe if Path(args.universe).is_absolute() else ROOT / args.universe)
        log(f"universe rule: member at the last month-end before the event date ({args.universe}; first month-end {uni_start})")
    df, einfo = BE.build_meta_events_bars(bars, break_s, h_days=args.h_days, hcap=hcap, ptsl=ptsl, cost_frac=cost_frac, eligible=eligible, log=log)
    df = df.sort_values(["t0", "symbol"], kind="stable").reset_index(drop=True)
    log(f"events kept {len(df):,}")

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
    state_t, above, dist, idx_series, ma = BE.market_state_ref(bars)
    fall = RB.deepest_fall(idx_series)
    if fall:
        windows["deepest_90d_fall_full_history"] = RB.day_window(fall[0], fall[1])
        notes["deepest_90d_fall_full_history"] = f"{fall[0]} -> {fall[1]}, {fall[2]:.1f}% on the equal-weight index (whole UTC days)"
    win_t = {k_: (int(pd.Timestamp(v[0]).timestamp()), int(pd.Timestamp(v[1]).timestamp())) for k_, v in windows.items()}
    for k_, v in notes.items():
        log(f"window {k_}: {v}")
    cover = {k_: int(sum(1 for b in bars.values() if ((b.t >= win_t[k_][0]) & (b.t < win_t[k_][1])).any())) for k_ in windows}
    for k_, n_ in cover.items():
        log(f"window {k_}: {n_} of {len(bars)} symbols have at least one bar inside")
    sc = df.groupby("symbol").size()
    extra = [f"Bar coverage of the reporting windows (symbols with at least one bar inside, of {len(bars)} used): " + "; ".join(f"{k_} {n_}" for k_, n_ in cover.items())
             + ". The market-state feature needs at least 10 symbols with an hourly return in the same hour, so events cannot be built where fewer than 10 symbols have rows; a window with no entered events is reported n/a.",
             f"Data holes: where the source has a hole longer than {man['break_hours']} hours (Kraken's official files and the trade files do not cover 2025-04-01 to about 2025-07-07 for most coins) no bar spans the hole, no event's path crosses it and no lookback reaches across it.",
             "Sources: Kraken = official 1-minute files to 2025-03-31, then 1-minute rows built from trades (freqtrade trade files, then the refreshed trade files from 2026-01-01); BTC and ICX end 2026-01-05 / 2026-01-04 (not refreshed). "
             "Alpaca = SIP hourly bars, split- and dividend-adjusted, pre-market, regular and after-hours included; the finest data downloaded is one hour, so most Alpaca bars are single hours and the volume clock only merges quiet hours."]
    if args.universe:
        extra.append("Alpaca universe: an event is kept only if its symbol was in the top 200 by trailing dollar volume at the last month-end before the event's UTC date (universe.csv of the daily Alpaca run); the symbols are those that were ever members, so survivorship is as in the daily run.")
    devs = bar_deviations(args.venue, int(args.prior_trials), args.h_days, hcap, extra)
    ME.DEVIATIONS = devs

    log("CPCV ...")
    preds, smet, cmeta = ME.run_cpcv(df, args.n_groups, args.k, args.embargo_pct, forest_kw, args.seed, log)
    ev = ME.evaluate(df, preds, cost_frac, args.step, h_eq, args.seed, win_t, args.prior_trials)
    log("walk-forward ...")
    wpred, wmet = ME.run_walk_forward(df, args.cutoff, forest_kw, args.seed + 500, log)
    wev = ME.evaluate(df, wpred, cost_frac, args.step, h_eq, args.seed + 7, win_t, args.prior_trials)

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
    params = {"h": h_eq, "h_days": args.h_days, "hcap": hcap, "ptsl": list(ptsl), "cost_frac": cost_frac, "n_groups": args.n_groups, "k": args.k, "embargo_pct": args.embargo_pct, "step": args.step,
              "forest": forest_kw, "seed": args.seed, "cutoff": args.cutoff, "prior_trials": args.prior_trials, "features": E.FEATURES, "ffd_thres": E.FFD_THRES, "side_signal": E.SIDE_SIGNAL,
              "micro": {"sadf": [E.SADF_WINDOW, E.SADF_MINSL, E.SADF_LAGS], "entropy": [E.ENT_WINDOW, E.ENT_WORD], "cs_sl": E.CS_SL}, "min_weight_fraction_leaf": 0.05,
              "bars": {"window_days": man["window_days"], "min_history_days": man["min_history_days"], "break_hours": man["break_hours"], "row_seconds": man["row_seconds"],
                       "span": BE.SPAN, "slow_span": BE.SLOW_SPAN, "mkt_min_history_days": BE.MKT_MIN_HISTORY_S // 86400, "state_tolerance_hours": BE.STATE_TOL_S // 3600}}
    md5s = {n: RB.md5_file(HERE / n) for n in ("book_stats.py", "book_study_core.py", "fracdiff_core.py", "micro_features.py", "meta_events.py", "meta_model.py", "meta_eval.py",
                                               "bar_loader.py", "bar_events.py", "build_dollar_bars.py", "run_meta_venue.py", "run_meta_bars.py")}
    meta = {**cmeta, "run_utc": started.isoformat(timespec="seconds"), "timeframe": "dollar", "symbols_used": len(bars), "events": int(len(df)),
            "earliest_candle": str(first), "latest_candle": str(last), "windows": {k_: [str(v[0]), str(v[1])] for k_, v in windows.items()},
            "window_epoch": win_t, "window_notes": notes, "venue": args.venue, "window_symbol_coverage": cover, "candle_dir": str(bars_dir.resolve()),
            "bars_manifest_md5": RB.md5_file(bars_dir / "manifest.json"), "cost_bp_per_side": float(args.cost_bp_per_side), "script_md5": md5s,
            "universe": args.universe}
    bars_text = (f"{len(bars)} symbols, {sum(b.n for b in bars.values()):,} bars from build_dollar_bars.py (manifest md5 {meta['bars_manifest_md5']}); a bar closes when ~one average clock-hour of traded value "
                 f"(trailing 30 days) has accumulated; stamped with its end.")
    summ = {"meta": meta, "params": params, "events_info": {k_: v for k_, v in einfo.items() if k_ not in ("symbols", "per_symbol")},
            "params_shown": {"bars": bars_text, "symbols used": len(bars), "events": f"{len(df):,}", "venue": args.venue, "round-trip cost (bp)": f"{cost_frac * 1e4:.1f}",
                             "barriers": f"+/- {ptsl[0]} x daily vol, vertical {args.h_days:g} days (cap {hcap} bars)", "forest": f"{args.n_estimators} trees, {args.bootstrap} bootstrap, max {args.max_draws:,} draws per tree",
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
    md = render(summ, devs)
    (out_dir / "SUMMARY.md").write_text(md, encoding="utf-8")
    log(f"done in {time.time() - started.timestamp():.0f}s -> {out_dir}")
    print("\n" + md[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
