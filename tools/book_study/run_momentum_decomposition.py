#!/usr/bin/env python
"""Rule-free decomposition of the momentum signals (READ-ONLY diagnostic).

No stop, no profit target, no simulated rule: every CUSUM event (same events as run_book_study.py) is held for H+1 bars.  For each trailing-return
lookback it reports what the signal earned (pooled), what each leg earned, what simply being long earned on the same events (survivor drift), and
winners-minus-losers (drift cancels), by market state, by calendar year, and inside the June 2026 / sell-off / deepest-fall windows.

It loads no model, changes no threshold, calibration file, model file or execution code, and writes only to runtime/book_study_decomposition/<UTC timestamp>/.

Run in the `javier` conda env, from the project root:
    python tools/book_study/run_momentum_decomposition.py
    python tools/book_study/verify_momentum_decomposition.py runtime/book_study_decomposition/<timestamp>
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
import momentum_decomp_core as M
import run_book_study as RB

SEED = 20261012           # fixed before any result was seen

NOTES = [
    "Events are CUSUM events, not a rebalanced portfolio: the signal is read only on event bars, so this is momentum conditional on a volatility event.",
    "The signal is the plain sign of the trailing log return, not the volatility-scaled sum of the published time-series-momentum rule.",
    "Survivorship: the symbols are those that still have a GMX file and a long TradingView history today. Coins that died are absent, which flatters any long-biased read. "
    "'Always-long' shows how large that drift is; only the winners-minus-losers column removes it.",
    "Overlap: with a hold of 101 bars, events of one symbol within 4 days share price moves. Intervals therefore resample blocks of ceil((H+1)/24) whole UTC days, not single days; they are still approximate.",
    "Trials: lookbacks of 7, 20, 30, 40 and 90 days (plus the 200-day price-versus-average rule in an earlier run) have now been looked at on this history, two horizons each, three states. "
    "A single interval excluding zero here is therefore not a confirmed finding; nothing in this report is corrected for the number of looks.",
    "Windows are whole UTC days and carry no interval: a window is a few correlated days. Read them as descriptions, not tests.",
    "Nothing here selects a rule, a threshold or a symbol, and nothing here is connected to any order path.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/book_study_decomposition")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-years", type=float, default=0.5)
    ap.add_argument("--sell-off", nargs=2, default=None, metavar=("START", "END"))
    args = ap.parse_args()

    from trading_bot_v4.config_v4 import V4Config as Config
    from trading_bot_v4.execution.gmx_adapter import list_gmx_symbols, load_gmx_ohlc

    lookbacks, horizons = M.LOOKBACKS, M.HORIZONS
    timeframe = str(getattr(Config, "TIMEFRAME", "1h"))
    fee_bps = float(Config.PAPER_FEE_BPS)
    slip_bps = float(Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS)
    cost_frac = 2.0 * (fee_bps + slip_bps) / 1e4
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

    tv_dir = Path(Config.TRADINGVIEW_HISTORY_DIR)
    tv_symbols = sorted(p.stem.removeprefix("tradingview_").removesuffix(f"_{timeframe}") for p in tv_dir.glob(f"tradingview_*_{timeframe}.csv"))
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
    for sym in names:
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

    sig_names = tuple(f"mom{L}" for L in lookbacks)
    ev = C.build_events(closes, max(horizons), log=lambda m: None, signals=sig_names)
    state, index, ma = C.market_state(closes, ev["t0"])
    n_ev = len(ev["t0"])
    log(f"events kept {n_ev:,}; state counts: above {(state == 1).sum():,}, below {(state == 0).sum():,}, unavailable {(state == -1).sum():,}")

    windows, window_notes = {"june_2026_ai_correction": C.JUNE_2026}, {"june_2026_ai_correction": "calendar month 2026-06-01 .. 2026-06-30"}
    if args.sell_off:
        s0, s1, src = pd.Timestamp(args.sell_off[0]), pd.Timestamp(args.sell_off[1]), "command line"
    else:
        found = RB.latest_baseline_sell_off()
        s0, s1, src = found if found else (None, None, None)
    if s0 is not None:
        windows["sell_off_post_holdout"] = RB.day_window(s0, s1)
        window_notes["sell_off_post_holdout"] = f"{s0} -> {s1} (whole UTC days), from {src}"
    else:
        log("no sell-off window available; only June 2026 is reported")
    fall = RB.deepest_fall(index)
    if fall:
        windows["deepest_90d_fall_full_history"] = RB.day_window(fall[0], fall[1])
        window_notes["deepest_90d_fall_full_history"] = f"{fall[0]} -> {fall[1]}, {fall[2]:.1f}% on this study's own equal-weight index (whole UTC days)"
    for name, w in windows.items():
        log(f"window {name}: {w[0]} .. {w[1]}  ({window_notes[name]})")

    log("decomposing ...")
    rows, by_year, win_rows = M.decompose(ev, state, cost_frac, args.n_boot, args.seed, lookbacks, horizons, windows)

    # ---- write ----
    syms = ev["symbols"]
    cols = {"symbol": [syms[i] for i in ev["sym"]], "pos": ev["pos"], "t0_epoch": ev["t0"], "sigma": ev["sigma"], "state": state}
    for nm in sig_names:
        cols["sig_" + nm] = ev["sig_" + nm]
    for h in horizons:
        cols[f"ret_h{h}_bp"] = ev["q"][:, h].astype(np.float64) * ev["sigma"] * 1e4
    pd.DataFrame(cols).to_csv(out_dir / "events.csv", index=False, float_format="%.10g")
    pd.DataFrame(ev["per_symbol"]).to_csv(out_dir / "symbols.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "rows.csv", index=False)
    pd.DataFrame(by_year).to_csv(out_dir / "by_year.csv", index=False)
    pd.DataFrame(win_rows).to_csv(out_dir / "window_rows.csv", index=False)
    after = RB.integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    meta = {"run_utc": started.isoformat(timespec="seconds"), "timeframe": timeframe, "cost_round_trip_bp": cost_frac * 1e4, "cost_frac": cost_frac,
            "lookbacks": list(lookbacks), "horizons": list(horizons), "n_boot": args.n_boot, "seed": args.seed, "span": C.SPAN,
            "symbols_used": len(closes), "symbols_skipped": len(skipped), "events": n_ev, "earliest_candle": str(first), "latest_candle": str(last),
            "windows": {k_: [str(v[0]), str(v[1])] for k_, v in windows.items()}, "window_notes": window_notes,
            "tradingview_dir": str(tv_dir.resolve()), "gmx_dir": str(Path(Config.GMX_OHLC_DIR).resolve()),
            "script_md5": {n: RB.md5_file(HERE / n) for n in ("book_stats.py", "book_study_core.py", "momentum_decomp_core.py", "run_momentum_decomposition.py")}}
    (out_dir / "summary_numbers.json").write_text(json.dumps({"meta": meta, "rows": rows, "by_year": by_year, "window_rows": win_rows, "notes": NOTES},
                                                           indent=2, default=str), encoding="utf-8")
    shown = {"run (UTC)": meta["run_utc"], "models/calibration untouched": before == after, "symbols used": len(closes), "events": f"{n_ev:,}",
             "round-trip cost (bp)": f"{cost_frac * 1e4:.1f}", "bootstrap draws / seed": f"{args.n_boot} / {args.seed}",
             "windows": " | ".join(f"{k_}: {v}" for k_, v in window_notes.items())}
    md = M.render(rows, by_year, win_rows, shown, NOTES)
    (out_dir / "SUMMARY.md").write_text(md, encoding="utf-8")
    log(f"done in {time.time() - started.timestamp():.0f}s -> {out_dir}")
    print("\n" + md[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
