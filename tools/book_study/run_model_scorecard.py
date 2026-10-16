#!/usr/bin/env python
"""Trained long / short models scored to the book's yardstick (READ-ONLY diagnostic).

For every symbol with a trained per-asset model it takes the CUSUM events (the same event definition as run_book_study.py) that fall
in the model's own out-of-sample span - its promotion holdout and the bars after it - reads the model's probability at each event
bar, and compares events the model FLAGGED (probability >= its calibrated threshold) with events it did not flag, same symbols and
period, using the stop/profit rule the control study chose (no model output was used to pick it) plus rule-free holds of H bars and
of 1 bar (the models' own target horizon).  See model_scorecard_core.py for the exact statistics, SUMMARY.md for the limits.

It changes no threshold, calibration file, model file or execution code, promotes nothing, and writes only to
runtime/book_study_models/<UTC timestamp>/.

Run in the `javier` conda env, from the project root:
    python tools/book_study/run_model_scorecard.py
Options: --book-dir <control run folder> (default: newest runtime/book_study run without --only-signals),
         --replay-dir <post-holdout replay folder> (alignment check; default: newest), --symbols BTC ETH, --limit N
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
sys.path.insert(0, str(ROOT / "tools" / "post_holdout_replay"))

import numpy as np
import pandas as pd

import book_study_core as C
import model_scorecard_core as M
import run_book_study as RB

SEED = 20261011        # fixed before any result was seen


def newest_control_run(base: Path):
    for p in sorted(base.glob("*"), reverse=True):
        f = p / "summary_numbers.json"
        if f.exists():
            s = json.loads(f.read_text(encoding="utf-8"))
            if not s.get("meta", {}).get("only_signals") and "H8|all|long" in s.get("rows", {}):
                return p, s
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/book_study_models")
    ap.add_argument("--book-dir", default=None)
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--no-align-check", action="store_true")
    ap.add_argument("--horizons", type=int, nargs="+", default=[8, 24])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-years", type=float, default=0.5)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sell-off", nargs=2, default=None, metavar=("START", "END"))
    args = ap.parse_args()

    import tensorflow as tf
    from train_model import DataHandler
    from trading_bot_v4.config_v4 import V4Config as Config
    from trading_bot_v4.execution.gmx_adapter import list_gmx_symbols, load_gmx_ohlc
    from trading_bot_v4.ml import per_asset_trainer as T
    from trading_bot_v4.utils.model_cache import ModelScalerCache, per_asset_model_path, per_asset_scaler_path
    import replay_stats as S      # tools/post_holdout_replay (same bounds parser as the replay)

    timeframe = str(getattr(Config, "TIMEFRAME", "1h"))
    seq = int(Config.SEQUENCE_LENGTH)
    cost_frac = 2.0 * (float(Config.PAPER_FEE_BPS) + float(Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS)) / 1e4
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

    # ---- rules from the control study ----
    if args.book_dir:
        book_dir = Path(args.book_dir).resolve()
        book = json.loads((book_dir / "summary_numbers.json").read_text(encoding="utf-8"))
    else:
        book_dir, book = newest_control_run(ROOT / "runtime" / "book_study")
    if book is None:
        raise SystemExit("no control run found: run tools/book_study/run_book_study.py first (it supplies the stop/profit rules)")
    rules = {}
    for d in ("long", "short"):
        for h in args.horizons:
            r = book["rows"].get(f"H{h}|all|{d}")
            if r is None or not r.get("modal_rule"):
                raise SystemExit(f"control run {book_dir} has no modal rule for H{h} {d}; run it with --horizons including {h}")
            rules[(d, h)] = (float(r["modal_rule"][0]), float(r["modal_rule"][1]))
    log(f"rules from control run {book_dir.name}: " + ", ".join(f"{d}/H{h}={v}" for (d, h), v in rules.items()))

    # ---- universe: symbols with a trained model in either direction and a GMX file ----
    cal = {d: json.loads((ROOT / "models" / d / "calibration.json").read_text(encoding="utf-8")) for d in ("long", "short")}
    model_syms = sorted({s for d in cal for s, e in cal[d].get("per_symbol", {}).items() if e.get("trained")})
    gmx = set(list_gmx_symbols(timeframe))
    names = [s for s in model_syms if s in gmx]
    if args.symbols:
        want = {s.upper() for s in args.symbols}
        names = [n for n in names if n.upper() in want]
    if args.limit:
        names = names[: args.limit]
    closes, skipped = {}, []
    min_bars = int(args.min_years * 365 * 24)
    for sym in names:
        try:
            df = load_gmx_ohlc(sym, timeframe)
            c = df["Close"].astype(float)
            c = c[c > 0]
            if len(c) < min_bars:
                skipped.append({"direction": "both", "symbol": sym, "reason": f"fewer than {min_bars} bars ({len(c)})"})
                continue
            closes[sym] = c
        except Exception as exc:                                # noqa: BLE001
            skipped.append({"direction": "both", "symbol": sym, "reason": f"error: {exc!r}"})
    if len(closes) < 10:
        raise SystemExit(f"only {len(closes)} usable symbols")
    log(f"{len(closes)} symbols with data; model symbols {len(model_syms)}")
    h_max = max(args.horizons)
    ev = C.build_events(closes, h_max, log=lambda m: None)
    sym_id = {s: i for i, s in enumerate(ev["symbols"])}
    log(f"events {len(ev['t0']):,}")
    key_to_idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ev["sym"], ev["t0"]))}

    # ---- model probabilities at event bars ----
    table_parts, sym_rows, align_rows = [], [], []
    replay_dir = None
    if not args.no_align_check:
        base = ROOT / "runtime" / "post_holdout_replay"
        cands = sorted(p for p in base.glob("*") if (p / "replay_per_symbol.csv").exists())
        replay_dir = Path(args.replay_dir).resolve() if args.replay_dir else (cands[-1] if cands else None)
    replay_df = pd.read_csv(replay_dir / "replay_per_symbol.csv") if replay_dir is not None else None
    n_align = {"long": 0, "short": 0}
    holdout_ends, holdout_starts = [], []

    for direction in ("long", "short"):
        per_symbol = cal[direction].get("per_symbol", {})
        promoted = {k.upper() for k in cal[direction].get("promoted_symbols", {})}
        for sym in [s for s in names if s in closes and s in per_symbol]:
            entry = per_symbol[sym]
            tag = f"{direction}/{sym}"
            bounds = S.parse_holdout_bounds(entry)
            mp, sp = per_asset_model_path(direction, sym, 1), per_asset_scaler_path(direction, sym, 1)
            if not entry.get("trained") or bounds is None or not (mp.exists() and sp.exists()):
                skipped.append({"direction": direction, "symbol": sym, "reason": "not trained / no holdout windows / model file missing"})
                continue
            try:
                df = T._load_symbol_frame(sym, timeframe, direction, 1)
                if df is None or len(df) <= seq + 1:
                    skipped.append({"direction": direction, "symbol": sym, "reason": "no usable data"})
                    continue
                model, scaler = ModelScalerCache(mp, sp).load()
                handler = DataHandler()
                handler.scaler = scaler
                scaled = handler.normalize_data(df[Config.FEATURE_COLUMNS].to_numpy(np.float32), fit=False)
                ep = T._endpoints(df)
                if len(ep) != len(scaled) - seq:
                    raise RuntimeError(f"alignment: {len(ep)} endpoints vs {len(scaled) - seq} sequences")
                ts = pd.to_datetime(ep["timestamp"])
                ts_sec = C.to_seconds(pd.DatetimeIndex(ts))
                h_start, h_end = bounds
                holdout_starts.append(h_start)
                holdout_ends.append(h_end)
                hs, he = int(h_start.timestamp()), int(h_end.timestamp())

                def predict(ks: np.ndarray) -> np.ndarray:
                    out = []
                    for b0 in range(0, len(ks), 2048):
                        kk = ks[b0:b0 + 2048]
                        X = np.stack([scaled[k:k + seq] for k in kk]).astype(np.float32)
                        out.append(model.predict(X, batch_size=1024, verbose=0).reshape(-1))
                    return np.concatenate(out) if out else np.empty(0)

                sid = sym_id[sym]
                m = (ev["sym"] == sid) & (ev["t0"] >= hs)
                eidx = np.flatnonzero(m)
                pos_k = np.searchsorted(ts_sec, ev["t0"][eidx])
                ok = (pos_k < len(ts_sec)) & (ts_sec[np.minimum(pos_k, len(ts_sec) - 1)] == ev["t0"][eidx])
                eidx, ks = eidx[ok], pos_k[ok]
                p = predict(ks)
                seg = np.where(ev["t0"][eidx] <= he, "holdout", "post_holdout")
                thr = float(entry["threshold"])
                table_parts.append(pd.DataFrame({"eidx": eidx, "direction": direction, "p": p, "thr": thr, "segment": seg,
                                                 "promoted": sym.upper() in promoted}))
                sym_rows.append({"direction": direction, "symbol": sym, "threshold": thr, "promoted": sym.upper() in promoted,
                                 "holdout_start": str(h_start), "holdout_end": str(h_end), "events_scored": int(len(eidx)),
                                 "events_holdout": int((seg == "holdout").sum()), "events_post_holdout": int((seg == "post_holdout").sum()),
                                 "flagged": int((p >= thr).sum())})

                # alignment check: reproduce the replay's own per-symbol probability statistics on the very same bars
                if replay_df is not None and n_align[direction] < 6:
                    rr = replay_df[(replay_df.direction == direction) & (replay_df.symbol == sym)]
                    if len(rr):
                        r0 = rr.iloc[0]
                        sel = np.flatnonzero(((ts >= pd.Timestamp(r0.replay_start)) & (ts <= pd.Timestamp(r0.replay_end))).to_numpy())
                        if len(sel) == int(r0.replay_candles):
                            pp = predict(sel)
                            ok_a = (abs(pp.mean() - r0.prob_mean) < 1e-4 and abs(pp.max() - r0.prob_max) < 1e-4
                                    and int((pp >= thr).sum()) == int(r0.signals_at_calibrated_threshold))
                            align_rows.append({"direction": direction, "symbol": sym, "bars": len(sel), "status": "PASS" if ok_a else "FAIL",
                                               "prob_mean": float(pp.mean()), "stored_prob_mean": float(r0.prob_mean),
                                               "prob_max": float(pp.max()), "stored_prob_max": float(r0.prob_max),
                                               "signals": int((pp >= thr).sum()), "stored_signals": int(r0.signals_at_calibrated_threshold)})
                        else:
                            align_rows.append({"direction": direction, "symbol": sym, "bars": len(sel), "status": "SKIP (bar count differs)",
                                               "stored_bars": int(r0.replay_candles)})
                        n_align[direction] += 1
                log(f"  {tag}: events scored {len(eidx)}, flagged {(p >= thr).sum()}")
            except Exception as exc:                            # noqa: BLE001
                skipped.append({"direction": direction, "symbol": sym, "reason": f"error: {exc!r}"})
                log(f"  {tag} ERROR {exc!r}")
            finally:
                try:
                    del model
                except NameError:
                    pass
                tf.keras.backend.clear_session()

    table = pd.concat(table_parts, ignore_index=True) if table_parts else pd.DataFrame()
    if table.empty:
        raise SystemExit("no model probabilities were produced")

    # ---- windows ----
    windows = {"june_2026_ai_correction": C.JUNE_2026}
    if args.sell_off:
        s0, s1 = pd.Timestamp(args.sell_off[0]), pd.Timestamp(args.sell_off[1])
        src = "command line"
    else:
        found = RB.latest_baseline_sell_off()
        s0, s1, src = found if found else (None, None, None)
    if s0 is not None:
        windows["sell_off_post_holdout"] = RB.day_window(s0, s1)
    log("windows: " + "; ".join(f"{k}: {v[0]}..{v[1]}" for k, v in windows.items()))

    rows, trials, events = M.score(ev, table, rules, cost_frac, windows, args.seed, tuple(args.horizons), args.n_boot)

    # ---- write ----
    events.to_csv(out_dir / "scored_events.csv", index=False)
    pd.DataFrame(sym_rows).to_csv(out_dir / "symbols.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False)
    pd.DataFrame(align_rows).to_csv(out_dir / "alignment_check.csv", index=False)
    after = RB.integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    n_pass = sum(1 for r in align_rows if r["status"] == "PASS")
    n_fail = sum(1 for r in align_rows if r["status"] == "FAIL")
    promoted_n = {d: len(cal[d].get("promoted_symbols", {})) for d in cal}
    meta = {
        "run_utc": started.isoformat(timespec="seconds"), "timeframe": timeframe, "round_trip_cost_bp": round(cost_frac * 1e4, 2),
        "control_run_used_for_rules": str(book_dir), "models_untouched": before == after,
        "alignment_check_vs_replay": f"{n_pass} PASS, {n_fail} FAIL, {len(align_rows) - n_pass - n_fail} skipped" if align_rows else "not run",
        "symbols_with_events": len({r['symbol'] for r in sym_rows}), "events_scored_long": int((table.direction == 'long').sum()),
        "events_scored_short": int((table.direction == 'short').sum()), "flagged_long": int(((table.direction == 'long') & (table.p >= table.thr)).sum()),
        "flagged_short": int(((table.direction == 'short') & (table.p >= table.thr)).sum()),
        "promoted_symbols_in_calibration": promoted_n, "seed": args.seed, "bootstrap_draws": args.n_boot,
        "gmx_dir": str(Path(Config.GMX_OHLC_DIR).resolve()), "tradingview_dir": str(Path(Config.TRADINGVIEW_HISTORY_DIR).resolve()),
        "sigma_span": C.SPAN, "horizons": list(args.horizons), "cost_frac": cost_frac,
        "windows": {k: [str(v[0]), str(v[1])] for k, v in windows.items()},
    }
    notes = [
        "The models were trained once on a fixed chronological split, so the book's combinatorial cross-validation (a model refitted per split) is not "
        "available. Only each model's holdout and post-holdout bars are out of sample for its weights and threshold; everything before is training, "
        "validation or calibration data and is not scored.",
        "The holdout segment was seen by the promotion gate (and by earlier replays); the post-holdout segment is the clean one but is only about two months.",
        "June 2026 is scored only where an event falls inside a model's out-of-sample span (symbols with a long holdout). The window table shows how many events that is.",
        "No symbol is promoted in either direction (see promoted_symbols_in_calibration), so production applies a threshold of 1.0 and none of these signals would trade. Thresholds here are the stored calibrated ones, as in the replay.",
        "The stop/profit rule is the control study's modal rule for the plain 'all' row of the same direction and horizon. It was fitted without any model output. The control's own rule effect (a cut-losers/ride-winners rule earns on trending paths whatever the sign) is why the comparison is flagged versus NOT flagged, not flagged versus zero.",
        "The models' target is the next candle. The horizon-1 hold is their native test; H = 8 and 24 are the control study's horizons.",
        "Eight rows (2 directions x 2 horizons x 2 segments) are examined; the deflated Sharpe uses the book's variance across those rows. Sample sizes are small, so read the intervals, not the point values.",
        "Events are the same CUSUM events as the control study, rebuilt from the candle files as they are now; entry at the event bar's close; barriers on bar closes (deviations 2, 3, 5, 12 of SUMMARY.md in the control run apply).",
    ]
    md = M.render(rows, trials, meta, notes)
    (out_dir / "SUMMARY.md").write_text(md, encoding="utf-8")
    (out_dir / "summary_numbers.json").write_text(json.dumps({"meta": meta, "trials": trials, "rows": rows, "rules": {f"{d}|H{h}": v for (d, h), v in rules.items()}},
                                                           indent=2, default=str), encoding="utf-8")
    log(f"done in {time.time() - started.timestamp():.0f}s -> {out_dir}")
    print("\n" + md[:5000])
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
