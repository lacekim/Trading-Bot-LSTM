#!/usr/bin/env python
"""Post-holdout out-of-sample replay (READ-ONLY diagnostic).

Replays the saved per-asset LONG/SHORT models over the candles that came AFTER each
model's promotion holdout ended, using the trainer's own data loading, feature code,
scaler, signal frame and simulator (simulate_production_symbol with the same costs).

It does NOT change thresholds, calibration files, model files or execution code, and it
does NOT promote anything. Every symbol is scored at its calibrated threshold, promoted
or not, purely to see what the models would have done. Results are a diagnostic, not
promotion evidence (see SUMMARY.md for the caveats).

Stage 1 (self-test, the validator): re-score each current-schema symbol's STORED
holdout window and compare trades / profit factor / return with what the trainer
saved. If the replay pipeline cannot reproduce those, its output is flagged UNVERIFIED.

Run from anywhere, in the `javier` conda env:
    python tools/post_holdout_replay/replay_post_holdout.py
Options: --directions long short | --symbols BTC ETH | --limit 5 | --skip-selftest
Output:  runtime/post_holdout_replay/<UTC timestamp>/
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
os.chdir(ROOT)  # model paths in utils/model_cache.py are relative to the project root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/post_holdout_replay")
    ap.add_argument("--directions", nargs="+", default=["long", "short"], choices=["long", "short"])
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N symbols per direction (trial run)")
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()

    # Heavy imports only after chdir/sys.path so they resolve like the trainer's do.
    import tensorflow as tf
    from train_model import DataHandler
    from trading_bot_v4.backtesting.production_backtest import simulate_production_symbol
    from trading_bot_v4.config_v4 import V4Config as Config
    from trading_bot_v4.ml import per_asset_trainer as T
    from trading_bot_v4.utils.model_cache import ModelScalerCache, per_asset_model_path, per_asset_scaler_path

    timeframe = str(getattr(Config, "TIMEFRAME", "1h"))
    seq = int(Config.SEQUENCE_LENGTH)
    started = datetime.now(timezone.utc)
    out_dir = ROOT / args.out / started.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=False)
    before = integrity_snapshot()

    def simulate(ep: pd.DataFrame, probs: np.ndarray, thr: float, direction: str, symbol: str):
        signals = T._build_signals_frame(ep.reset_index(drop=True), probs, thr, direction)
        return simulate_production_symbol(
            signals, symbol, T.SIMULATION_STARTING_CAPITAL, entry_eligible=True,
            fee_bps=Config.PAPER_FEE_BPS,
            slippage_bps=Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS,
        )

    selftest_rows, symbol_rows, trade_frames, skipped = [], [], [], []
    closes: dict[str, pd.Series] = {}
    holdout_ends = []

    for direction in args.directions:
        calibration = json.loads((ROOT / "models" / direction / "calibration.json").read_text(encoding="utf-8"))
        per_symbol = calibration.get("per_symbol", {})
        promoted_map = {k.upper(): v for k, v in calibration.get("promoted_symbols", {}).items()}
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
            if not (mp.exists() and sp.exists()):
                skipped.append({"direction": direction, "symbol": symbol, "reason": "model/scaler file missing"})
                continue
            try:
                df = T._load_symbol_frame(symbol, timeframe, direction, 1)
                if df is None or len(df) <= seq + 1:
                    skipped.append({"direction": direction, "symbol": symbol, "reason": "no usable data"})
                    continue
                model, scaler = ModelScalerCache(mp, sp).load()
                handler = DataHandler()
                handler.scaler = scaler
                scaled = handler.normalize_data(df[Config.FEATURE_COLUMNS].to_numpy(np.float32), fit=False)
                ep = T._endpoints(df)  # same alignment as the trainer: endpoint k <-> scaled[k:k+seq]
                if len(ep) != len(scaled) - seq:
                    raise RuntimeError(f"alignment: {len(ep)} endpoints vs {len(scaled) - seq} sequences")
                ts = pd.to_datetime(ep["timestamp"])
                closes.setdefault(symbol, pd.Series(df["Close"].to_numpy(float), index=pd.to_datetime(df["timestamp"])))

                def predict(lo: int, hi: int) -> np.ndarray:
                    X = np.stack([scaled[k:k + seq] for k in range(lo, hi)]).astype(np.float32)
                    return model.predict(X, batch_size=1024, verbose=0).reshape(-1)

                methodology = "current" if "holdout_trades" in entry else "legacy"
                thr = float(entry["threshold"])
                h_start, h_end = bounds
                holdout_ends.append(h_end)
                selftest_status = "NA"

                # ---- Stage 1: self-test against the numbers the trainer stored ----
                if methodology == "current" and not args.skip_selftest:
                    idx = np.flatnonzero(((ts >= h_start) & (ts <= h_end)).to_numpy())
                    expected_n = int(sum(int(w.get("samples", 0)) for w in entry["windows"]))
                    if len(idx):
                        lo, hi = int(idx[0]), int(idx[-1]) + 1
                        p = predict(lo, hi)
                        summary, _ = simulate(ep.iloc[lo:hi], p, thr, direction, symbol)
                        cmp = S.compare_selftest(entry, summary)
                        endpoints_ok = (len(idx) == expected_n) and (hi - lo == len(idx))
                        passed = cmp["passed"] and endpoints_ok
                        selftest_status = "PASS" if passed else "FAIL"
                        selftest_rows.append({
                            "direction": direction, "symbol": symbol, "status": selftest_status,
                            "endpoints_found": len(idx), "endpoints_stored": expected_n,
                            "stored_trades": entry["holdout_trades"], "replay_trades": summary["trades"],
                            "stored_pf": entry["holdout_profit_factor"], "replay_pf": summary["profit_factor"],
                            "stored_return_pct": entry["holdout_return_pct"], "replay_return_pct": summary["return_pct"],
                        })
                    else:
                        selftest_status = "FAIL"
                        selftest_rows.append({"direction": direction, "symbol": symbol, "status": "FAIL",
                                              "endpoints_found": 0, "endpoints_stored": expected_n})

                # ---- Stage 2: replay strictly after the holdout ended ----
                after = np.flatnonzero((ts > h_end).to_numpy())
                if len(after) < 2:
                    skipped.append({"direction": direction, "symbol": symbol, "reason": "no candles after holdout"})
                    continue
                lo, hi = int(after[0]), len(ep)
                p = predict(lo, hi)
                ep_after = ep.iloc[lo:hi].reset_index(drop=True)
                if not (pd.to_datetime(ep_after["timestamp"]) > h_end).all():
                    raise RuntimeError("replay window overlaps holdout")
                summary, trades = simulate(ep_after, p, thr, direction, symbol)
                fret = ep_after["future_return"].to_numpy(float)       # next-candle return (fraction)
                sret = fret if direction == "long" else -fret         # return in the trade's direction
                if len(trades):
                    trades = trades.copy()
                    trades["methodology"] = methodology
                    trades["promoted"] = symbol.upper() in promoted_map
                    trade_frames.append(trades)
                symbol_rows.append({
                    "direction": direction, "symbol": symbol, "methodology": methodology,
                    "promoted": symbol.upper() in promoted_map, "threshold": thr,
                    "applied_threshold_in_production": float(promoted_map.get(symbol.upper(), 1.0)),
                    "holdout_end": str(h_end), "replay_start": str(ep_after["timestamp"].iloc[0]),
                    "replay_end": str(ep_after["timestamp"].iloc[-1]), "replay_candles": len(ep_after),
                    "signals_at_calibrated_threshold": int((p >= thr).sum()),
                    "trades": summary["trades"], "win_rate_pct": summary["win_rate_pct"],
                    "return_pct": summary["return_pct"], "profit_factor": summary["profit_factor"],
                    "max_drawdown_pct": summary["max_drawdown_pct"],
                    "price_return_pct": (float(ep_after["Close"].iloc[-1]) / float(ep_after["Close"].iloc[0]) - 1.0) * 100.0,
                    "prob_mean": float(np.mean(p)), "prob_p95": float(np.percentile(p, 95)), "prob_max": float(np.max(p)),
                    # Does the probability track the DIRECTION of the next candle, or only the SIZE of the move?
                    "rho_signed": S.spearman(p, sret), "rho_abs": S.spearman(p, np.abs(fret)),
                    "top_decile_edge_pct": S.top_decile_edge(p, sret) * 100.0,
                    "base_mean_ret_pct": float(np.mean(sret)) * 100.0,
                    "selftest": selftest_status,
                })
                print(f"  [{i}/{len(names)}] {tag} {methodology} trades={summary['trades']} "
                      f"PF={fmt_pf(summary['profit_factor'])} selftest={selftest_status}", flush=True)
            except Exception as exc:  # keep going; record the failure
                skipped.append({"direction": direction, "symbol": symbol, "reason": f"error: {exc!r}"})
                print(f"  {tag} ERROR {exc!r}", flush=True)
            finally:
                try:
                    del model
                except NameError:
                    pass
                tf.keras.backend.clear_session()

        pd.DataFrame(symbol_rows).to_csv(out_dir / "replay_per_symbol.partial.csv", index=False)

    per_symbol_df = pd.DataFrame(symbol_rows)
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    selftest_df = pd.DataFrame(selftest_rows)
    per_symbol_df.to_csv(out_dir / "replay_per_symbol.csv", index=False)
    trades_df.to_csv(out_dir / "replay_trades.csv", index=False)
    selftest_df.to_csv(out_dir / "selftest.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False)
    (out_dir / "replay_per_symbol.partial.csv").unlink(missing_ok=True)

    after_snap = integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps(
        {"before": before, "after": after_snap, "unchanged": before == after_snap}, indent=2), encoding="utf-8")

    # ---- market context + sell-off window (from the data, not assumed) ----
    index = pd.Series(dtype=float)
    window = None
    if closes and holdout_ends:
        common_start = max(holdout_ends)
        index = S.equal_weight_index(closes, common_start, min_symbols=min(10, len(closes)))
        window = S.max_drawdown_window(index) if not index.empty else None
        if not index.empty:
            index.resample("1D").last().rename("equal_weight_index").to_csv(out_dir / "equal_weight_index_daily.csv")

    numbers: dict = {}
    if not per_symbol_df.empty:
        for direction in args.directions:
            d_sym = per_symbol_df[per_symbol_df["direction"] == direction]
            d_tr = trades_df[trades_df["direction"].str.upper() == direction.upper()] if len(trades_df) else trades_df
            groups = {
                "all": d_tr,
                "current_schema": d_tr[d_tr["methodology"] == "current"] if len(d_tr) else d_tr,
                "legacy_schema": d_tr[d_tr["methodology"] == "legacy"] if len(d_tr) else d_tr,
                "promoted": d_tr[d_tr["promoted"]] if len(d_tr) else d_tr,
            }
            if window is not None and len(d_tr):
                et = pd.to_datetime(d_tr["entry_time"])
                groups["entered_during_sell_off"] = d_tr[(et >= window[0]) & (et <= window[1])]
            numbers[direction] = {g: S.pooled_stats(f) for g, f in groups.items()}
            numbers[direction]["_symbols"] = {
                "scored": int(len(d_sym)), "with_trades": int((d_sym["trades"] > 0).sum()),
                "pf_above_1_with_5plus_trades": int(((d_sym["profit_factor"] > 1.0) & (d_sym["trades"] >= 5)).sum()),
                "median_price_return_pct": float(d_sym["price_return_pct"].median()),
                "median_rho_signed": float(d_sym["rho_signed"].median()),
                "median_rho_abs": float(d_sym["rho_abs"].median()),
                "symbols_rho_signed_positive": int((d_sym["rho_signed"] > 0).sum()),
                "symbols_tracking_size_more_than_direction": int((d_sym["rho_abs"] > d_sym["rho_signed"] + 0.02).sum()),
                "median_top_decile_edge_pct": float(d_sym["top_decile_edge_pct"].median()),
            }
    (out_dir / "summary_numbers.json").write_text(json.dumps(numbers, indent=2, default=str), encoding="utf-8")

    # ---- SUMMARY.md ----
    st_pass = int((selftest_df["status"] == "PASS").sum()) if len(selftest_df) else 0
    st_total = int(len(selftest_df))
    errors = [r for r in skipped if str(r.get("reason", "")).startswith("error")]
    verified = ((st_total > 0 and st_pass == st_total) or args.skip_selftest) and not errors and not per_symbol_df.empty
    lines = [
        "# Post-holdout replay (read-only diagnostic)", "",
        f"Run (UTC): {started.isoformat(timespec='seconds')}  |  timeframe {timeframe}  |  sequence length {seq}",
        f"Verification: self-test {st_pass}/{st_total} current-schema symbol-directions reproduced their stored "
        f"holdout numbers. Status: {'VERIFIED' if verified else 'UNVERIFIED - do not rely on these numbers'}.",
        f"Scored {len(per_symbol_df)} symbol-directions; skipped {len(skipped)} ({len(errors)} because of errors; see skipped.csv).",
        f"Models/calibration untouched: {before == after_snap}.", "",
        "Guardrails: every symbol is scored at its CALIBRATED threshold (promoted or not). Production applies 1.0 to "
        "non-promoted symbols, so none of these trades would have happened. This is not promotion evidence: thresholds "
        "for legacy-schema symbols come from the pre-fix methodology, ~240 models are scored with no multiple-testing "
        "correction, and one post-holdout period is a single path.", "",
    ]
    if window is not None:
        lines += [f"Equal-weight index of the replayed symbols: {fmt_pct((index.iloc[-1] - 1) * 100)} from "
                  f"{index.index[0]} to {index.index[-1]}. Deepest fall (the 'sell-off' used below, taken from the data): "
                  f"{window[0]} -> {window[1]}, {window[2]:.1f}%.", ""]
    if window is None:
        lines += ["Sell-off window: not computed (no overlapping post-holdout data across the scored symbols).", ""]
    for direction, groups in numbers.items():
        meta = groups["_symbols"]
        lines += [f"## {direction.upper()} model", "",
                  f"Symbols scored: {meta['scored']}; with at least one trade: {meta['with_trades']}; "
                  f"PF>1 with 5+ trades: {meta['pf_above_1_with_5plus_trades']}; "
                  f"median buy-and-hold price return over each replay: {fmt_pct(meta['median_price_return_pct'])}.", "",
                  f"Direction vs size of move (next-candle, replay period): median rank correlation of the probability with the "
                  f"signed return {meta['median_rho_signed']:.3f} and with the absolute return {meta['median_rho_abs']:.3f}; "
                  f"{meta['symbols_rho_signed_positive']} of {meta['scored']} symbols have a positive signed correlation; "
                  f"{meta['symbols_tracking_size_more_than_direction']} track the size of the move clearly more than its direction; "
                  f"median edge of the top-10%-probability candles over the average candle: {meta['median_top_decile_edge_pct']:.3f}% "
                  "per candle, before costs. If the absolute correlation is larger than the signed one, the model is mostly a "
                  "volatility detector.", "",
                  "| group | trades | win rate | pooled PF | net P&L (USD, summed over symbols) |", "|---|---|---|---|---|"]
        for g, s in groups.items():
            if g.startswith("_"):
                continue
            lines.append(f"| {g} | {s['trades']} | {fmt_pct(s['win_rate_pct'])} | {fmt_pf(s['profit_factor'])} | {s['net_pnl']:.0f} |")
        lines.append("")
    lines += ["Files: replay_per_symbol.csv, replay_trades.csv, selftest.csv, skipped.csv, integrity.json, "
              "summary_numbers.json, equal_weight_index_daily.csv. Run verify_replay.py on this folder next.", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone in {time.time() - started.timestamp():.0f}s. Output: {out_dir}", flush=True)
    print("\n".join(lines[:6]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
