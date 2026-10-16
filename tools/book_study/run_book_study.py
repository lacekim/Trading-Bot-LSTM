#!/usr/bin/env python
"""Book-procedure study (READ-ONLY diagnostic): always-long and always-short controls, measured by market state.

Follows the book's procedures (Advances in Financial Machine Learning, condensed edition): CUSUM event sampling (Ch.2),
causal volatility units (Ch.3), uniqueness weights (Ch.4), purged + embargoed combinatorial cross-validation (Ch.7, 12),
optimal-trading-rule search on a 20 x 20 stop/profit mesh from a fitted Ornstein-Uhlenbeck process (Ch.13, with its
Definition 2 overfit test), probabilistic and deflated Sharpe ratios (Ch.14).  The places where it cannot follow the book
exactly are listed in BOOK_NOTES.md and at the bottom of SUMMARY.md.

State = the equal-weight index of all symbols above / below its own 200-day average (known at the event bar).
Directions = always long, always short, plus a random-sign placebo that must come out at zero before costs.

It loads no model, changes no threshold, calibration file, model file or execution code, and writes only to
runtime/book_study/<UTC timestamp>/.

Run in the `javier` conda env, from the project root:
    python tools/book_study/run_book_study.py            # full: N=10, k=2, H = 8, 24, 100, 100,000 simulated paths
    python tools/book_study/run_book_study.py --quick    # smoke: N=6, k=2, H = 8, 24, 20,000 simulated paths
    python tools/book_study/run_book_study.py --signals mom168 mom720 mom2160 ma200d --only-signals    # momentum / trend rules, H = 24, 100
"""
from __future__ import annotations

import argparse
import hashlib
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

import book_stats as B
import book_study_core as C

SEED = 20261010           # fixed before any result was seen


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def integrity_snapshot() -> dict:
    snap = {}
    for direction in ("long", "short"):
        root = ROOT / "models" / direction
        files = sorted(p for p in root.glob("*") if p.is_file())
        cal = root / "calibration.json"
        snap[direction] = {"files": len(files), "newest_mtime": max((p.stat().st_mtime for p in files), default=0.0),
                           "calibration_md5": md5_file(cal) if cal.exists() else None}
    return snap


def deepest_fall(index: pd.Series, max_days: int = 90):
    """(peak time, trough time, drawdown %) of the deepest fall of a daily-resampled index whose peak is at most
    max_days before its trough (an unrestricted peak-to-trough on a five-year history is just 'the whole history')."""
    d = index.resample("1D").last().dropna()
    if len(d) < 3:
        return None
    peak = d.rolling(max_days + 1, min_periods=1).max()
    dd = d / peak - 1.0
    trough = dd.idxmin()
    window = d.loc[trough - pd.Timedelta(days=max_days): trough]
    return window.idxmax(), trough, float(dd.min() * 100.0)


def latest_baseline_sell_off():
    base = ROOT / "runtime" / "post_holdout_baseline"
    for p in sorted(base.glob("*"), reverse=True):
        f = p / "summary_numbers.json"
        if f.exists():
            w = json.loads(f.read_text(encoding="utf-8")).get("windows", {}).get("post_holdout")
            if w:
                return pd.Timestamp(w["start"]), pd.Timestamp(w["end"]), str(f)
    return None


def day_window(start: pd.Timestamp, end_inclusive: pd.Timestamp):
    """Whole UTC days that contain [start, end_inclusive]: [floor(start), floor(end)+1 day)."""
    return start.floor("D"), end_inclusive.floor("D") + pd.Timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/book_study")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-groups", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--horizons", type=int, nargs="+", default=None)
    ap.add_argument("--n-paths", type=int, default=None, help="simulated paths per rule search")
    ap.add_argument("--embargo-pct", type=float, default=0.01)
    ap.add_argument("--min-years", type=float, default=0.5)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--min-test", type=int, default=100)
    ap.add_argument("--n-sample", type=int, default=40, help="events per sampled split/cell saved for the independent checker")
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--signals", nargs="+", default=[], choices=list(C.SIGNALS),
                    help="also test these plain momentum / trend rules (sign of the trailing return, or price vs its own 200-day mean)")
    ap.add_argument("--only-signals", action="store_true",
                    help="skip the always-long / always-short rows; run only the named signals plus the placebo (trial count = signal rows)")
    ap.add_argument("--sell-off", nargs=2, default=None, metavar=("START", "END"),
                    help="override the sell-off window (UTC); default: the newest plain-rule baseline's window")
    args = ap.parse_args()

    n_groups = args.n_groups or (6 if args.quick else 10)
    k = args.k or 2
    if args.only_signals and not args.signals:
        raise SystemExit("--only-signals needs --signals")
    horizons = args.horizons or ([24] if (args.only_signals and args.quick) else [24, 100] if args.only_signals
                                 else [8, 24] if args.quick else [8, 24, 100])
    n_paths = args.n_paths or (20000 if args.quick else 100000)

    from trading_bot_v4.config_v4 import V4Config as Config
    from trading_bot_v4.execution.gmx_adapter import list_gmx_symbols, load_gmx_ohlc

    timeframe = str(getattr(Config, "TIMEFRAME", "1h"))
    fee_bps = float(Config.PAPER_FEE_BPS)
    slip_bps = float(Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS)
    cost_frac = 2.0 * (fee_bps + slip_bps) / 1e4          # round trip
    started = datetime.now(timezone.utc)
    out_dir = ROOT / args.out / started.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=False)
    before = integrity_snapshot()
    log_file = open(out_dir / "run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    # ---- symbols: those with TradingView history on disk and a GMX file ----
    tv_dir = Path(Config.TRADINGVIEW_HISTORY_DIR)
    tv_symbols = sorted(p.stem.removeprefix("tradingview_").removesuffix(f"_{timeframe}")
                        for p in tv_dir.glob(f"tradingview_*_{timeframe}.csv"))
    gmx_symbols = set(list_gmx_symbols(timeframe))
    names = [s for s in tv_symbols if s in gmx_symbols]
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        names = [n for n in names if n.upper() in wanted]
    if args.limit:
        names = names[: args.limit]
    log(f"TradingView files {len(tv_symbols)}, GMX files {len(gmx_symbols)}, intersection used {len(names)}")

    closes, skipped = {}, []
    min_bars = int(args.min_years * 365 * 24)
    for i, sym in enumerate(names, 1):
        try:
            df = load_gmx_ohlc(sym, timeframe)
            if df is None or df.empty:
                skipped.append({"symbol": sym, "reason": "empty"})
                continue
            c = df["Close"].astype(float)
            c = c[c > 0]
            if len(c) < min_bars:
                skipped.append({"symbol": sym, "reason": f"fewer than {min_bars} bars ({len(c)})"})
                continue
            closes[sym] = c
        except Exception as exc:                                # noqa: BLE001
            skipped.append({"symbol": sym, "reason": f"error: {exc!r}"})
            log(f"  {sym} ERROR {exc!r}")
    if len(closes) < 10:
        raise SystemExit(f"only {len(closes)} usable symbols; need at least 10 for the market index")
    first = min(c.index[0] for c in closes.values())
    last = max(c.index[-1] for c in closes.values())
    log(f"loaded {len(closes)} symbols; earliest candle {first}, latest {last}")

    h_max = max(horizons)
    log("building CUSUM events ...")
    ev = C.build_events(closes, h_max, log=lambda m: None, signals=tuple(args.signals))
    state, index, ma = C.market_state(closes, ev["t0"])
    n_ev = len(ev["t0"])
    log(f"events kept {n_ev:,}; state counts: above {(state == 1).sum():,}, below {(state == 0).sum():,}, unavailable {(state == -1).sum():,}")

    # ---- windows (whole UTC days; entries that happened inside) ----
    windows, window_notes = {}, {}
    windows["june_2026_ai_correction"] = C.JUNE_2026
    window_notes["june_2026_ai_correction"] = "calendar month 2026-06-01 .. 2026-06-30"
    if args.sell_off:
        s0, s1 = pd.Timestamp(args.sell_off[0]), pd.Timestamp(args.sell_off[1])
        src = "command line"
    else:
        found = latest_baseline_sell_off()
        s0, s1, src = (found if found else (None, None, None))
    if s0 is not None:
        windows["sell_off_post_holdout"] = day_window(s0, s1)
        window_notes["sell_off_post_holdout"] = f"{s0} -> {s1} (whole UTC days), from {src}"
    else:
        log("no sell-off window available (no baseline summary found and none given); only June 2026 is reported")
    full_fall = deepest_fall(index)
    if full_fall:
        windows["deepest_90d_fall_full_history"] = day_window(full_fall[0], full_fall[1])
        window_notes["deepest_90d_fall_full_history"] = (
            f"{full_fall[0]} -> {full_fall[1]}, {full_fall[2]:.1f}% on this study's own equal-weight index, peak at most 90 days before trough (whole UTC days)")
    for name, w in windows.items():
        log(f"window {name}: {w[0]} .. {w[1]}  ({window_notes[name]})")

    params = dict(n_groups=n_groups, k=k, horizons=horizons, n_paths=n_paths, embargo_pct=args.embargo_pct, seed=args.seed,
                  cost_frac=cost_frac, min_train=args.min_train, min_test=args.min_test, n_sample=args.n_sample,
                  signals=args.signals, only_signals=args.only_signals)
    log(f"params {params}")
    result = C.run_study(ev, state, params, log=log)

    # ---- write ----
    result["cells"].to_csv(out_dir / "cells.csv", index=False)
    result["daily"].to_csv(out_dir / "oos_daily.csv", index=False)
    result["samples"].to_csv(out_dir / "samples.csv", index=False)
    result["events_index"].to_csv(out_dir / "events_index.csv", index=False)
    sym_df = pd.DataFrame(ev["per_symbol"])
    sym_df.to_csv(out_dir / "symbols.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False)
    pd.DataFrame({"equal_weight_index": index, "ma_200d": ma}).resample("1D").last().to_csv(out_dir / "equal_weight_index_daily.csv")

    summary = C.summarize(result, windows)
    summary["window_notes"] = window_notes
    summary["run"] = {"run_utc": started.isoformat(timespec="seconds"), "timeframe": timeframe, "fee_bps_per_side": fee_bps,
                      "slippage_plus_impact_bps_per_side": slip_bps, "cost_round_trip_bp": cost_frac * 1e4,
                      "symbols_used": len(closes), "symbols_skipped": len(skipped), "earliest_candle": str(first),
                      "latest_candle": str(last), "seed": args.seed, "span": C.SPAN,
                      "tradingview_dir": str(tv_dir.resolve()), "gmx_dir": str(Path(Config.GMX_OHLC_DIR).resolve()),
                      "state_counts": {"above": int((state == 1).sum()), "below": int((state == 0).sum()),
                                       "unavailable": int((state == -1).sum())},
                      "script_md5": {n: md5_file(HERE / n) for n in ("book_stats.py", "book_study_core.py", "run_book_study.py")}}
    after = integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    (out_dir / "summary_numbers.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md = C.render_markdown(summary, sym_df, {"runtime_seconds": time.time() - started.timestamp()})
    head = [f"Run (UTC): {started.isoformat(timespec='seconds')}. Models/calibration untouched: {before == after}.",
            f"Window definitions: " + " | ".join(f"{k_}: {v}" for k_, v in window_notes.items()), ""]
    lines = md.split("\n")
    md = "\n".join(lines[:2] + head + lines[2:])
    (out_dir / "SUMMARY.md").write_text(md, encoding="utf-8")
    log(f"done in {time.time() - started.timestamp():.0f}s -> {out_dir}")
    print("\n" + md[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
