#!/usr/bin/env python
"""Plain-rule baseline (READ-ONLY diagnostic): the bar the models have to beat.

Runs three predeclared, parameter-free rules through the project's OWN simulator
(simulate_production_symbol: same ATR stop/target, max hold, sizing, fee 5 bp, slippage+impact 7 bp per side)
over the SAME symbols and the SAME post-holdout candles as replay_post_holdout.py:

  ma200d  LONG when Close > its 200-day average, SHORT when below   (primary rule)
  ma200h  same with a 200-hour average                               (variant)
  always  a signal on every candle                                   (control: exits + costs alone)

Segments (each a fresh simulation, 100,000 USD start per symbol):
  post_holdout  candles strictly after each symbol's promotion holdout (identical to the model replay)
  june_2026     the calendar month 2026-06-01 .. 2026-06-30 (the AI-correction window; rules have no
                training period, so any period is out of sample for them)
  full_history  every candle with a usable average (opt-in: --full-history; slow)

It does NOT change thresholds, calibration files, model files or execution code, loads no model, and
promotes nothing. Rules are not promotion evidence; this is the yardstick for the models and for the
situation table. Three rules were declared up front, so at most 3 trials.

Run in the `javier` conda env:
    python tools/post_holdout_replay/run_baseline.py
Options: --replay-dir <folder> (default: newest runtime/post_holdout_replay/*), --no-replay-compare,
         --full-history, --rules ..., --directions ..., --symbols ..., --limit N
Output:  runtime/post_holdout_baseline/<UTC timestamp>/
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

import baseline_rules as B
import replay_stats as S


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
        snap[direction] = {
            "files": len(files),
            "newest_mtime": max((p.stat().st_mtime for p in files), default=0.0),
            "calibration_md5": md5_file(cal) if cal.exists() else None,
        }
    return snap


def fmt_pf(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return "inf" if np.isinf(x) else f"{x:.2f}"


def fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}%"


def latest_replay_dir():
    base = ROOT / "runtime" / "post_holdout_replay"
    cands = sorted(p for p in base.glob("*") if (p / "replay_per_symbol.csv").exists() and (p / "summary_numbers.json").exists())
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/post_holdout_baseline")
    ap.add_argument("--directions", nargs="+", default=["long", "short"], choices=["long", "short"])
    ap.add_argument("--rules", nargs="+", default=list(B.RULES), choices=list(B.RULES))
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full-history", action="store_true", help="also simulate the whole usable history (slow)")
    ap.add_argument("--replay-dir", default=None, help="model replay folder to compare against (default: newest)")
    ap.add_argument("--no-replay-compare", action="store_true")
    ap.add_argument("--no-align-to-replay", action="store_true",
                    help="use every candle currently on disk instead of stopping each symbol where the model replay stopped")
    args = ap.parse_args()

    from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol
    from trading_bot_v4.config_v4 import V4Config as Config
    from trading_bot_v4.ml import per_asset_trainer as T
    from trading_bot_v4.utils.model_cache import per_asset_model_path, per_asset_scaler_path

    timeframe = str(getattr(Config, "TIMEFRAME", "1h"))
    slip_bps = float(Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS)
    fee_bps = float(Config.PAPER_FEE_BPS)
    started = datetime.now(timezone.utc)
    out_dir = ROOT / args.out / started.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=False)
    before = integrity_snapshot()

    replay_dir = None
    if not args.no_replay_compare:
        replay_dir = Path(args.replay_dir).resolve() if args.replay_dir else latest_replay_dir()
        if replay_dir is not None and not (replay_dir / "replay_per_symbol.csv").exists():
            print(f"Replay folder {replay_dir} has no replay_per_symbol.csv; comparison skipped.", flush=True)
            replay_dir = None

    # The candle files keep growing. To compare like with like, stop each symbol-direction at the last candle
    # the model replay scored (pairs the replay did not score use the latest end seen anywhere in it).
    end_cut: dict = {}
    global_end = None
    if replay_dir is not None and not args.no_align_to_replay:
        rep0 = pd.read_csv(replay_dir / "replay_per_symbol.csv")
        end_cut = {(r.direction, r.symbol): pd.Timestamp(r.replay_end) for r in rep0.itertuples()}
        global_end = max(end_cut.values()) if end_cut else None
    aligned = bool(end_cut)

    segments = ["post_holdout", "june_2026"] + (["full_history"] if args.full_history else [])

    def simulate(ep: pd.DataFrame, sig: np.ndarray, direction: str, symbol: str):
        signals = T._build_signals_frame(ep.reset_index(drop=True), sig.astype(float), 0.5, direction)
        return simulate_production_symbol(
            signals, symbol, T.SIMULATION_STARTING_CAPITAL, entry_eligible=True,
            fee_bps=fee_bps, slippage_bps=slip_bps,
        )

    rows, trade_frames, skipped = [], [], []
    closes: dict[str, pd.Series] = {}
    holdout_ends = []

    for direction in args.directions:
        calibration = json.loads((ROOT / "models" / direction / "calibration.json").read_text(encoding="utf-8"))
        per_symbol = calibration.get("per_symbol", {})
        names = sorted(per_symbol)
        if args.symbols:
            wanted = {s.upper() for s in args.symbols}
            names = [n for n in names if n.upper() in wanted]
        if args.limit:
            names = names[: args.limit]
        print(f"[{direction}] {len(names)} symbols", flush=True)

        for i, symbol in enumerate(names, start=1):
            entry = per_symbol[symbol]
            tag = f"{direction}/{symbol}"
            bounds = S.parse_holdout_bounds(entry)
            if not entry.get("trained") or bounds is None:
                skipped.append({"direction": direction, "symbol": symbol,
                                "reason": "not trained" if not entry.get("trained") else "no holdout windows stored"})
                continue
            mp, sp = per_asset_model_path(direction, symbol, 1), per_asset_scaler_path(direction, symbol, 1)
            if not (mp.exists() and sp.exists()):   # same universe as the model replay; no model is loaded
                skipped.append({"direction": direction, "symbol": symbol, "reason": "model/scaler file missing"})
                continue
            try:
                df = T._load_symbol_frame(symbol, timeframe, direction, 1)
                seq = int(Config.SEQUENCE_LENGTH)
                if df is not None and aligned:
                    cut = end_cut.get((direction, symbol), global_end)
                    # keep every row up to the replay's last scored candle plus the one row after it
                    # (the trainer's _endpoints() drops the final row), exactly as the replay's data ended
                    k = int((pd.to_datetime(df["timestamp"]) <= cut).sum())
                    df = df.iloc[: min(len(df), k + 1)].reset_index(drop=True)
                if df is None or len(df) <= seq + 1:
                    skipped.append({"direction": direction, "symbol": symbol, "reason": "no usable data"})
                    continue
                ep = T._endpoints(df)
                ts = pd.to_datetime(ep["timestamp"])
                close_series = pd.Series(df["Close"].to_numpy(float), index=pd.to_datetime(df["timestamp"]))
                closes.setdefault(symbol, close_series)
                methodology = "current" if "holdout_trades" in entry else "legacy"
                h_start, h_end = bounds
                holdout_ends.append(h_end)

                slices = {}
                after = np.flatnonzero((ts > h_end).to_numpy())
                if len(after) >= 2:
                    slices["post_holdout"] = (int(after[0]), len(ep))
                else:
                    skipped.append({"direction": direction, "symbol": symbol, "reason": "no candles after holdout"})
                js = B.window_slice(ts, *B.JUNE_2026, min_len=48)
                if js is not None:
                    slices["june_2026"] = js
                if args.full_history:
                    slices["full_history"] = (0, len(ep))

                for rule in args.rules:
                    avg = B.average_for_rule(close_series, rule).reindex(pd.DatetimeIndex(ts)).to_numpy()
                    sig_all = B.signal_array(ep["Close"].to_numpy(float), avg, direction, rule)
                    for segment, (lo, hi) in slices.items():
                        ep_seg = ep.iloc[lo:hi].reset_index(drop=True)
                        if segment == "post_holdout" and not (pd.to_datetime(ep_seg["timestamp"]) > h_end).all():
                            raise RuntimeError("post-holdout window overlaps the holdout")
                        sig = sig_all[lo:hi]
                        summary, trades = simulate(ep_seg, sig, direction, symbol)
                        if len(trades):
                            trades = trades.copy()
                            trades["segment"], trades["rule"], trades["methodology"] = segment, rule, methodology
                            trade_frames.append(trades)
                        rows.append({
                            "segment": segment, "rule": rule, "direction": direction, "symbol": symbol,
                            "methodology": methodology, "holdout_end": str(h_end),
                            "window_start": str(ep_seg["timestamp"].iloc[0]), "window_end": str(ep_seg["timestamp"].iloc[-1]),
                            "candles": len(ep_seg), "valid_average_bars": B.valid_average_bars(avg[lo:hi], rule),
                            "signal_bars": int(sig.sum()),
                            "trades": summary["trades"], "win_rate_pct": summary["win_rate_pct"],
                            "return_pct": summary["return_pct"], "profit_factor": summary["profit_factor"],
                            "max_drawdown_pct": summary["max_drawdown_pct"],
                            "price_return_pct": (float(ep_seg["Close"].iloc[-1]) / float(ep_seg["Close"].iloc[0]) - 1.0) * 100.0,
                            "bars_in_market_pct": B.bars_in_market_pct(trades, len(ep_seg)),
                        })
                        if segment == "post_holdout" and rule == B.PRIMARY_RULE:
                            print(f"  [{i}/{len(names)}] {tag} {rule} trades={summary['trades']} "
                                  f"PF={fmt_pf(summary['profit_factor'])}", flush=True)
            except Exception as exc:
                skipped.append({"direction": direction, "symbol": symbol, "reason": f"error: {exc!r}"})
                print(f"  {tag} ERROR {exc!r}", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "baseline_per_symbol.partial.csv", index=False)

    per_symbol_df = pd.DataFrame(rows)
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    per_symbol_df.to_csv(out_dir / "baseline_per_symbol.csv", index=False)
    trades_df.to_csv(out_dir / "baseline_trades.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False)
    (out_dir / "baseline_per_symbol.partial.csv").unlink(missing_ok=True)
    after_snap = integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps(
        {"before": before, "after": after_snap, "unchanged": before == after_snap}, indent=2), encoding="utf-8")

    # ---- market windows (from the data): identical construction to the model replay ----
    windows = {}
    index_ph = pd.Series(dtype=float)
    if closes and holdout_ends:
        index_ph = S.equal_weight_index(closes, max(holdout_ends), min_symbols=min(10, len(closes)))
        if not index_ph.empty:
            w = S.max_drawdown_window(index_ph)
            windows["post_holdout"] = {"name": "sell_off", "start": str(w[0]), "end": str(w[1]), "drawdown_pct": w[2],
                                       "index_start": str(index_ph.index[0]), "index_end": str(index_ph.index[-1]),
                                       "index_return_pct": float((index_ph.iloc[-1] - 1) * 100.0)}
            index_ph.resample("1D").last().rename("equal_weight_index").to_csv(out_dir / "equal_weight_index_daily.csv")
    if args.full_history and closes:
        first = min(c.index[0] for c in closes.values())
        index_full = S.equal_weight_index(closes, first, min_symbols=min(10, len(closes)))
        if not index_full.empty:
            w = S.max_drawdown_window(index_full)
            windows["full_history"] = {"name": "deepest_fall", "start": str(w[0]), "end": str(w[1]), "drawdown_pct": w[2],
                                       "index_start": str(index_full.index[0]), "index_end": str(index_full.index[-1]),
                                       "index_return_pct": float((index_full.iloc[-1] - 1) * 100.0)}

    # ---- pooled numbers ----
    numbers: dict = {}
    if not per_symbol_df.empty:
        for segment in segments:
            seg_rows = per_symbol_df[per_symbol_df["segment"] == segment]
            if seg_rows.empty:
                continue
            numbers[segment] = {}
            for rule in args.rules:
                numbers[segment][rule] = {}
                for direction in args.directions:
                    d_rows = seg_rows[(seg_rows["rule"] == rule) & (seg_rows["direction"] == direction)]
                    if d_rows.empty:
                        continue
                    if len(trades_df):
                        t = trades_df[(trades_df["segment"] == segment) & (trades_df["rule"] == rule)
                                      & (trades_df["direction"].str.upper() == direction.upper())]
                    else:
                        t = trades_df
                    groups = {"all": t}
                    if segment in windows and len(t):
                        wv = windows[segment]
                        groups[f"entered_during_{wv['name']}"] = B.entries_in(t, pd.Timestamp(wv["start"]), pd.Timestamp(wv["end"]))
                    if segment != "june_2026":
                        groups["entered_june_2026"] = B.entries_in_half_open(t, *B.JUNE_2026) if len(t) else t
                    numbers[segment][rule][direction] = {g: S.pooled_stats(f) for g, f in groups.items()}
                    numbers[segment][rule][direction]["_symbols"] = {
                        "scored": int(len(d_rows)), "with_trades": int((d_rows["trades"] > 0).sum()),
                        "with_signals": int((d_rows["signal_bars"] > 0).sum()),
                        "pf_above_1_with_5plus_trades": int(((d_rows["profit_factor"] > 1.0) & (d_rows["trades"] >= 5)).sum()),
                        "median_price_return_pct": float(d_rows["price_return_pct"].median()),
                        "median_bars_in_market_pct": float(d_rows["bars_in_market_pct"].median()),
                    }

    # ---- comparison with the model replay (post-holdout only) ----
    model_ref, comparison = {}, pd.DataFrame()
    replay_summary_path = replay_dir / "summary_numbers.json" if replay_dir else None
    if replay_dir is not None and "post_holdout" in numbers:
        rep = pd.read_csv(replay_dir / "replay_per_symbol.csv")
        rep_sum = json.loads(replay_summary_path.read_text(encoding="utf-8")) if replay_summary_path.exists() else {}
        for direction in args.directions:
            if direction in rep_sum:
                model_ref[direction] = {g: rep_sum[direction][g] for g in rep_sum[direction] if not g.startswith("_")}
        base = per_symbol_df[per_symbol_df["segment"] == "post_holdout"]
        comp = rep[["direction", "symbol", "methodology", "promoted", "replay_start", "replay_end", "replay_candles",
                    "trades", "profit_factor", "return_pct"]].rename(columns={
            "trades": "model_trades", "profit_factor": "model_pf", "return_pct": "model_return_pct"})
        for rule in args.rules:
            r = base[base["rule"] == rule][["direction", "symbol", "trades", "profit_factor", "return_pct"]].rename(columns={
                "trades": f"{rule}_trades", "profit_factor": f"{rule}_pf", "return_pct": f"{rule}_return_pct"})
            comp = comp.merge(r, on=["direction", "symbol"], how="outer")
        comparison = comp
        comparison.to_csv(out_dir / "comparison_with_model.csv", index=False)

    meta = {
        "run_utc": started.isoformat(timespec="seconds"), "timeframe": timeframe,
        "fee_bps": fee_bps, "slippage_plus_impact_bps": slip_bps,
        "gmx_dir": str(Path(Config.GMX_OHLC_DIR).resolve()), "tradingview_dir": str(Path(Config.TRADINGVIEW_HISTORY_DIR).resolve()),
        "rules": args.rules, "primary_rule": B.PRIMARY_RULE, "ma_days": B.MA_DAYS, "ma_days_min_bars": B.MA_DAYS_MIN_BARS,
        "ma_hours": B.MA_HOURS, "june_2026": [str(B.JUNE_2026[0]), str(B.JUNE_2026[1])],
        "segments": segments, "replay_dir": str(replay_dir) if replay_dir else None,
        "aligned_to_replay_end": aligned,
    }
    (out_dir / "summary_numbers.json").write_text(json.dumps(
        {"meta": meta, "windows": windows, "numbers": numbers, "model_reference": model_ref}, indent=2, default=str), encoding="utf-8")

    # ---- SUMMARY.md ----
    errors = [r for r in skipped if str(r.get("reason", "")).startswith("error")]
    lines = [
        "# Plain-rule baseline (read-only diagnostic)", "",
        f"Run (UTC): {started.isoformat(timespec='seconds')}  |  timeframe {timeframe}  |  costs: fee {fee_bps:g} bp + slippage/impact {slip_bps:g} bp per side",
        f"Scored {len(per_symbol_df)} symbol-direction-rule-segment rows; skipped {len(skipped)} ({len(errors)} because of errors; see skipped.csv).",
        f"Models/calibration untouched: {before == after_snap}.  Model replay compared against: {replay_dir if replay_dir else 'none'}"
        f"{' (each symbol stopped at the last candle the replay scored)' if aligned else ''}.", "",
        "Rules (declared before running): ma200d = price vs its 200-DAY average (primary), ma200h = vs 200-HOUR average, "
        "always = a signal on every candle (control). LONG above the average, SHORT below. No parameters were fitted, "
        "so every period is out of sample for the rules. This is a yardstick, not promotion evidence: the post-holdout period "
        "is one short path, and trade counts differ a lot between the rules and the models.", "",
    ]
    for segment, w in windows.items():
        lines += [f"{segment}: equal-weight index of the scored symbols {fmt_pct(w['index_return_pct'])} from {w['index_start']} to "
                  f"{w['index_end']}. Deepest fall ('{w['name']}', from the data): {w['start']} -> {w['end']}, {w['drawdown_pct']:.1f}%.", ""]
    for segment, rules_block in numbers.items():
        lines += [f"## Segment: {segment}", ""]
        lines += ["| rule | direction | group | trades | win rate | pooled PF | net P&L (USD, summed over symbols) |",
                  "|---|---|---|---|---|---|---|"]
        for rule, dirs in rules_block.items():
            for direction, groups in dirs.items():
                for g, s in groups.items():
                    if g.startswith("_"):
                        continue
                    lines.append(f"| {rule} | {direction} | {g} | {s['trades']} | {fmt_pct(s['win_rate_pct'])} | {fmt_pf(s['profit_factor'])} | {s['net_pnl']:.0f} |")
        if segment == "post_holdout" and model_ref:
            for direction, groups in model_ref.items():
                for g, s in groups.items():
                    lines.append(f"| MODEL (replay) | {direction} | {g} | {s['trades']} | {fmt_pct(s['win_rate_pct'])} | {fmt_pf(s['profit_factor'])} | {s['net_pnl']:.0f} |")
        lines.append("")
        lines.append("Per-rule breadth (scored symbols / with trades / PF>1 with 5+ trades / median share of candles in a position / median buy-and-hold return of the period):")
        for rule, dirs in rules_block.items():
            for direction, groups in dirs.items():
                m = groups["_symbols"]
                lines.append(f"- {rule} {direction}: {m['scored']} / {m['with_trades']} / {m['pf_above_1_with_5plus_trades']} / "
                             f"{m['median_bars_in_market_pct']:.0f}% / {fmt_pct(m['median_price_return_pct'])}")
        lines.append("")
    if "post_holdout" in numbers and holdout_ends and B.JUNE_2026[1] <= min(holdout_ends):
        lines += ["June 2026 ends before the earliest holdout end among the scored symbols, so the post_holdout segment has no June entries "
                  "by construction; the june_2026 segment is the separate fresh run for that month.", ""]
    lines += ["Files: baseline_per_symbol.csv, baseline_trades.csv, skipped.csv, integrity.json, summary_numbers.json, "
              "comparison_with_model.csv (when a replay folder was given), equal_weight_index_daily.csv. Run verify_baseline.py on this folder next.", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone in {time.time() - started.timestamp():.0f}s. Output: {out_dir}", flush=True)
    print("\n".join(lines[:6]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
