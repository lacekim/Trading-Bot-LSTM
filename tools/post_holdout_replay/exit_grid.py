#!/usr/bin/env python
"""Exit-rule grid test (READ-ONLY diagnostic).

Question: with NO entry signal at all (an always-in LONG and an always-in SHORT control), does any combination of
stop (x ATR), target (x ATR) and maximum hold produce a positive result after the simulator's costs - and if some
cell does in one period, does it still do so in a later one?  This separates two things:
  * cost drag      (every trade pays ~24 bp round trip; fewer, longer trades pay less in total), from
  * missing edge   (no stop/target choice can create a positive pre-cost expectation by itself).

Grid: stop {0.5, 0.75, 1.5, 3, 6} x target {0.5, 0.75, 1.5, 3, 6} x max hold {8, 24, 96} candles = 75 cells (plus the
simulator's current default if it is not in the grid). Every cell is run through the project's own
simulate_production_symbol (same sizing, fees, slippage/impact); only the exit policy is swapped IN MEMORY for the
duration of this process (the files are never edited).

Three disjoint periods per symbol (declared before running, never re-cut afterwards):
  select        2026-01-01 .. 2026-05-31   the cell is CHOSEN here (highest mean net bp per trade, >= 1000 trades)
  june_2026     2026-06-01 .. 2026-06-30   the AI-correction month; scored once
  post_holdout  after each symbol's promotion holdout, stopping where the model replay stopped; scored once
(the gap 2026-07-01 .. holdout end is not used, so the periods do not touch).

Self-test: the cell equal to the current default must reproduce the saved baseline run (rule 'always') exactly -
same trade counts per symbol and same pooled profit factor in june_2026 and post_holdout. Otherwise UNVERIFIED.

It does not change thresholds, calibration, model files or execution code, loads no model, promotes nothing.

    python tools/post_holdout_replay/exit_grid.py                # all symbols, uses up to 5 worker processes
Options: --workers N | --limit N | --symbols ... | --directions ... | --baseline-dir | --replay-dir | --no-align-to-replay
Output:  runtime/post_holdout_exit_grid/<UTC timestamp>/
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
import exit_grid_stats as G
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
        snap[direction] = {"files": len(files), "newest_mtime": max((p.stat().st_mtime for p in files), default=0.0),
                           "calibration_md5": md5_file(cal) if cal.exists() else None}
    return snap


def fmt_pf(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return "inf" if np.isinf(x) else f"{x:.2f}"


def fmt_bp(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.1f}"


def work(task: dict) -> dict:
    """One symbol-direction: every cell x every period. Runs in a worker process (or inline with --workers 1)."""
    from trading_bot_v4.backtesting import production_backtest as PB
    from trading_bot_v4.config_v4 import V4Config as Config
    from trading_bot_v4.execution.persistent_policy import PersistentPolicy
    from trading_bot_v4.ml import per_asset_trainer as T

    direction, symbol = task["direction"], task["symbol"]
    out = {"direction": direction, "symbol": symbol, "rows": [], "skipped": None}
    try:
        timeframe = str(getattr(Config, "TIMEFRAME", "1h"))
        df = T._load_symbol_frame(symbol, timeframe, direction, 1)
        if df is not None and task["cut"] is not None:
            cut = pd.Timestamp(task["cut"])
            k = int((pd.to_datetime(df["timestamp"]) <= cut).sum())
            df = df.iloc[: min(len(df), k + 1)].reset_index(drop=True)   # same truncation as run_baseline.py
        if df is None or len(df) <= int(Config.SEQUENCE_LENGTH) + 1:
            out["skipped"] = "no usable data"
            return out
        ep = T._endpoints(df)
        ts = pd.to_datetime(ep["timestamp"])
        h_end = pd.Timestamp(task["holdout_end"])
        slices = {}
        for period, window in (("select", G.SELECT_WINDOW), ("june_2026", G.JUNE_WINDOW)):
            sl = B.window_slice(ts, *window, min_len=48)
            if sl is not None:
                slices[period] = sl
        after = np.flatnonzero((ts > h_end).to_numpy())
        if len(after) >= 2:
            slices["post_holdout"] = (int(after[0]), len(ep))
        slip = float(Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS)
        fee = float(Config.PAPER_FEE_BPS)
        for cell in task["cells"]:
            stop, target, hold = cell
            policy = PersistentPolicy(False, 0.5, float(stop), float(target), int(hold), True)
            PB.policy_for = lambda _symbol, _p=policy: _p        # in-memory only; the module file is never edited
            for period, (lo, hi) in slices.items():
                seg = ep.iloc[lo:hi].reset_index(drop=True)
                if period == "post_holdout" and not (pd.to_datetime(seg["timestamp"]) > h_end).all():
                    raise RuntimeError("post-holdout window overlaps the holdout")
                signals = T._build_signals_frame(seg, np.ones(len(seg)), 0.5, direction)
                _, trades = PB.simulate_production_symbol(signals, symbol, T.SIMULATION_STARTING_CAPITAL,
                                                          entry_eligible=True, fee_bps=fee, slippage_bps=slip)
                row = {"direction": direction, "symbol": symbol, "period": period, "stop_atr": stop, "target_atr": target,
                       "max_hold": hold, "window_start": str(seg["timestamp"].iloc[0]), "window_end": str(seg["timestamp"].iloc[-1]),
                       "candles": len(seg), "holdout_end": str(h_end)}
                row.update(G.aggregate_trades(trades, slip))
                out["rows"].append(row)
    except Exception as exc:
        out["skipped"] = f"error: {exc!r}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/post_holdout_exit_grid")
    ap.add_argument("--directions", nargs="+", default=["long", "short"], choices=["long", "short"])
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(5, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--baseline-dir", default=None, help="saved baseline run for the self-test (default: newest)")
    ap.add_argument("--replay-dir", default=None, help="model replay folder that fixes each symbol's end candle (default: newest)")
    ap.add_argument("--no-align-to-replay", action="store_true")
    ap.add_argument("--quick", action="store_true", help="trial run: 2 x 2 x 2 grid (8 cells) instead of 75")
    ap.add_argument("--min-select-trades", type=int, default=1000)
    args = ap.parse_args()

    from trading_bot_v4.config_v4 import V4Config as Config
    from trading_bot_v4.execution.persistent_policy import policy_for
    from trading_bot_v4.utils.model_cache import per_asset_model_path, per_asset_scaler_path

    started = datetime.now(timezone.utc)
    out_dir = ROOT / args.out / started.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=False)
    before = integrity_snapshot()

    current = policy_for("")
    default_cell = (float(current.stop_atr), float(current.target_atr), int(current.max_hold_candles))
    cells = G.make_cells((0.75, 3.0), (1.5, 6.0), (8, 96)) if args.quick else G.make_cells()
    if default_cell not in cells:
        cells.append(default_cell)

    def newest(base: str, needed: str):
        d = ROOT / "runtime" / base
        c = sorted(p for p in d.glob("*") if (p / needed).exists())
        return c[-1] if c else None

    replay_dir = Path(args.replay_dir).resolve() if args.replay_dir else newest("post_holdout_replay", "replay_per_symbol.csv")
    baseline_dir = Path(args.baseline_dir).resolve() if args.baseline_dir else newest("post_holdout_baseline", "baseline_per_symbol.csv")
    end_cut, global_end = {}, None
    if replay_dir is not None and not args.no_align_to_replay:
        rep0 = pd.read_csv(replay_dir / "replay_per_symbol.csv")
        end_cut = {(r.direction, r.symbol): pd.Timestamp(r.replay_end) for r in rep0.itertuples()}
        global_end = max(end_cut.values()) if end_cut else None
    aligned = bool(end_cut)

    tasks, skipped = [], []
    for direction in args.directions:
        calibration = json.loads((ROOT / "models" / direction / "calibration.json").read_text(encoding="utf-8"))
        per_symbol = calibration.get("per_symbol", {})
        names = sorted(per_symbol)
        if args.symbols:
            wanted = {s.upper() for s in args.symbols}
            names = [n for n in names if n.upper() in wanted]
        if args.limit:
            names = names[: args.limit]
        for symbol in names:
            entry = per_symbol[symbol]
            bounds = S.parse_holdout_bounds(entry)
            if not entry.get("trained") or bounds is None:
                skipped.append({"direction": direction, "symbol": symbol,
                                "reason": "not trained" if not entry.get("trained") else "no holdout windows stored"})
                continue
            if not (per_asset_model_path(direction, symbol, 1).exists() and per_asset_scaler_path(direction, symbol, 1).exists()):
                skipped.append({"direction": direction, "symbol": symbol, "reason": "model/scaler file missing"})
                continue
            cut = end_cut.get((direction, symbol), global_end) if aligned else None
            tasks.append({"direction": direction, "symbol": symbol, "holdout_end": str(bounds[1]),
                          "cut": str(cut) if cut is not None else None, "cells": cells})

    print(f"{len(tasks)} symbol-directions x {len(cells)} cells x 3 periods; workers={args.workers}; default cell (stop, target, hold) = {default_cell}", flush=True)
    rows = []
    t0 = time.time()
    done = 0

    def handle(res: dict):
        nonlocal done
        done += 1
        rows.extend(res["rows"])
        if res["skipped"]:
            skipped.append({"direction": res["direction"], "symbol": res["symbol"], "reason": res["skipped"]})
        el = time.time() - t0
        print(f"  [{done}/{len(tasks)}] {res['direction']}/{res['symbol']} {'SKIPPED ' + res['skipped'] if res['skipped'] else 'ok'}"
              f"  elapsed {el / 60:.1f} min, est. total {el / done * len(tasks) / 60:.1f} min", flush=True)
        if done % 20 == 0:
            pd.DataFrame(rows).to_csv(out_dir / "exit_grid_per_symbol.partial.csv", index=False)

    if args.workers <= 1:
        for task in tasks:
            handle(work(task))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            for fut in as_completed([pool.submit(work, t) for t in tasks]):
                handle(fut.result())

    per_symbol_df = pd.DataFrame(rows)
    per_symbol_df.to_csv(out_dir / "exit_grid_per_symbol.csv", index=False)
    (out_dir / "exit_grid_per_symbol.partial.csv").unlink(missing_ok=True)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False)
    summary = G.summarize(per_symbol_df)
    summary.to_csv(out_dir / "cell_summary.csv", index=False)
    after_snap = integrity_snapshot()
    (out_dir / "integrity.json").write_text(json.dumps({"before": before, "after": after_snap, "unchanged": before == after_snap}, indent=2), encoding="utf-8")

    # ---- self-test: the default cell must reproduce the saved baseline run (rule 'always') ----
    selftest = []
    if baseline_dir is not None and not per_symbol_df.empty:
        bs = pd.read_csv(baseline_dir / "baseline_per_symbol.csv")
        bs = bs[bs["rule"] == "always"]
        cur = per_symbol_df[(per_symbol_df["stop_atr"] == default_cell[0]) & (per_symbol_df["target_atr"] == default_cell[1])
                            & (per_symbol_df["max_hold"] == default_cell[2])]
        for period, seg in (("june_2026", "june_2026"), ("post_holdout", "post_holdout")):
            for direction in args.directions:
                a = cur[(cur["period"] == period) & (cur["direction"] == direction)].set_index("symbol")
                b = bs[(bs["segment"] == seg) & (bs["direction"] == direction)].set_index("symbol")
                common = sorted(set(a.index) & set(b.index))
                mism = [s for s in common if int(a.loc[s, "trades"]) != int(b.loc[s, "trades"])]
                ga, gl = a.loc[common, "gross_win"].sum(), a.loc[common, "gross_loss"].sum()
                pf_a = ga / gl if gl else float("nan")
                bt_path = baseline_dir / "baseline_trades.csv"
                pf_b = float("nan")
                if bt_path.exists():
                    bt = pd.read_csv(bt_path, usecols=["segment", "rule", "direction", "symbol", "net_pnl"])
                    bt = bt[(bt["segment"] == seg) & (bt["rule"] == "always") & (bt["direction"].str.lower() == direction) & (bt["symbol"].isin(common))]
                    w, l = bt.loc[bt["net_pnl"] > 0, "net_pnl"].sum(), -bt.loc[bt["net_pnl"] < 0, "net_pnl"].sum()
                    pf_b = w / l if l else float("nan")
                ok = bool(common) and not mism and np.isclose(pf_a, pf_b, rtol=1e-9, atol=0.0)
                selftest.append({"period": period, "direction": direction, "symbols_compared": len(common),
                                 "symbols_with_different_trade_count": len(mism), "grid_pf": pf_a, "baseline_pf": pf_b,
                                 "status": "PASS" if ok else "FAIL"})
    pd.DataFrame(selftest).to_csv(out_dir / "selftest.csv", index=False)
    st_pass = sum(1 for r in selftest if r["status"] == "PASS")
    errors = [r for r in skipped if str(r.get("reason", "")).startswith("error")]
    verified = bool(selftest) and st_pass == len(selftest) and not errors and not per_symbol_df.empty

    # ---- analysis ----
    numbers = {"directions": {}}
    for direction in args.directions:
        d = {}
        chosen = G.select_cell(summary, direction, "select", args.min_select_trades) if not summary.empty else None
        d["selected_cell"] = [float(x) for x in chosen] if chosen else None
        d["default_cell"] = list(default_cell)
        d["cells_in_grid"] = len(cells)
        d["rank_agreement_select_vs_june"] = G.rank_agreement(summary, direction, "select", "june_2026") if not summary.empty else float("nan")
        d["rank_agreement_select_vs_post"] = G.rank_agreement(summary, direction, "select", "post_holdout") if not summary.empty else float("nan")
        d["rank_agreement_june_vs_post"] = G.rank_agreement(summary, direction, "june_2026", "post_holdout") if not summary.empty else float("nan")
        for period in G.PERIODS:
            t = G.cell_table(summary, direction, period) if not summary.empty else pd.DataFrame()
            t = t[t["trades"] >= 200] if len(t) else t
            if t is None or t.empty:
                d[period] = None
                continue
            best_net, best_pre = t["mean_net_bp"].idxmax(), t["mean_pre_cost_bp"].idxmax()
            row = lambda c: {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in t.loc[c, ["trades", "mean_pre_cost_bp", "mean_cost_bp", "mean_net_bp", "profit_factor", "win_rate_pct"]].items()}
            d[period] = {
                "cells_scored": int(len(t)), "cells_with_positive_mean_net_bp": int((t["mean_net_bp"] > 0).sum()),
                "best_cell_by_net": [float(x) for x in best_net], "best_by_net": row(best_net),
                "best_cell_by_pre_cost": [float(x) for x in best_pre], "best_by_pre_cost": row(best_pre),
                "mean_pre_cost_bp_over_cells": float(t["mean_pre_cost_bp"].mean()),
                "default": row(default_cell) if default_cell in t.index else None,
                "selected": row(chosen) if chosen and chosen in t.index else None,
                "selected_percentile_among_cells": G.percentile_rank(summary, direction, period, chosen) if chosen else None,
            }
        numbers["directions"][direction] = d
    meta = {"run_utc": started.isoformat(timespec="seconds"), "grid": {"stops": sorted({c[0] for c in cells}), "targets": sorted({c[1] for c in cells}), "holds": sorted({c[2] for c in cells})},
            "quick": bool(args.quick),
            "default_cell": list(default_cell), "periods": {"select": [str(x) for x in G.SELECT_WINDOW], "june_2026": [str(x) for x in G.JUNE_WINDOW],
                                                             "post_holdout": "after each symbol's holdout end, stopped at the model replay's last candle"},
            "slippage_plus_impact_bps": float(Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS), "fee_bps": float(Config.PAPER_FEE_BPS),
            "min_select_trades": args.min_select_trades, "replay_dir": str(replay_dir) if replay_dir else None,
            "baseline_dir": str(baseline_dir) if baseline_dir else None, "aligned_to_replay_end": aligned, "verified": verified}
    (out_dir / "summary_numbers.json").write_text(json.dumps({"meta": meta, "numbers": numbers}, indent=2, default=str), encoding="utf-8")

    # ---- SUMMARY.md ----
    L = ["# Exit-rule grid on an always-in control (read-only diagnostic)", "",
         f"Run (UTC): {started.isoformat(timespec='seconds')}  |  {len(cells)} cells (stop x target x max hold)  |  costs: fee {meta['fee_bps']:g} bp + slippage/impact {meta['slippage_plus_impact_bps']:g} bp per side",
         f"Self-test (current default cell reproduces the saved baseline, rule 'always'): {st_pass}/{len(selftest)} comparisons PASS. Status: "
         f"{'VERIFIED' if verified else 'UNVERIFIED - do not rely on these numbers'}.",
         f"Symbol-directions run: {len(tasks)}; skipped {len(skipped)} ({len(errors)} errors). Models/calibration untouched: {before == after_snap}.", "",
         "Control = a signal on every candle, so any pre-cost edge comes from the exit rule and the market alone. Metrics are basis points per trade "
         "(position size changes with the stop, so USD figures are not comparable across cells). Trades from different symbols overlap in time and "
         "share the market's moves, so they are not independent; there is no significance test here. "
         "The cell is chosen on `select` only; june_2026 and post_holdout are scored once.", ""]
    for direction, d in numbers["directions"].items():
        L += [f"## {direction.upper()} control", "",
              f"Selected on `select`: stop/target/hold = {d['selected_cell']}; current default = {d['default_cell']}. "
              f"Rank agreement across the {d['cells_in_grid']} cells (Spearman of mean net bp): select vs june {fmt_bp(d['rank_agreement_select_vs_june'])}, "
              f"select vs post-holdout {fmt_bp(d['rank_agreement_select_vs_post'])}, june vs post-holdout {fmt_bp(d['rank_agreement_june_vs_post'])} "
              "(near 0 means the grid ranking in one period says nothing about the next).", "",
              "| period | cells | cells with net bp > 0 | mean pre-cost bp over cells | default: pre-cost / cost / net bp (PF) | selected: pre-cost / net bp (PF) | selected beats this % of cells | best cell by pre-cost bp: cell -> pre-cost / net bp |",
              "|---|---|---|---|---|---|---|---|"]
        for period in G.PERIODS:
            p = d.get(period)
            if not p:
                L.append(f"| {period} | no data | | | | | | |")
                continue
            df_, sel, bp_ = p["default"], p["selected"], p["best_by_pre_cost"]
            L.append(f"| {period} | {p['cells_scored']} | {p['cells_with_positive_mean_net_bp']} | {fmt_bp(p['mean_pre_cost_bp_over_cells'])} | "
                     + (f"{fmt_bp(df_['mean_pre_cost_bp'])} / {df_['mean_cost_bp']:.1f} / {fmt_bp(df_['mean_net_bp'])} ({fmt_pf(df_['profit_factor'])})" if df_ else "n/a")
                     + " | " + (f"{fmt_bp(sel['mean_pre_cost_bp'])} / {fmt_bp(sel['mean_net_bp'])} ({fmt_pf(sel['profit_factor'])})" if sel else "n/a")
                     + f" | {('%.0f%%' % p['selected_percentile_among_cells']) if p['selected_percentile_among_cells'] is not None and not np.isnan(p['selected_percentile_among_cells']) else 'n/a'} | "
                     + f"{p['best_cell_by_pre_cost']} -> {fmt_bp(bp_['mean_pre_cost_bp'])} / {fmt_bp(bp_['mean_net_bp'])} |")
        L.append("")
    L += ["Files: exit_grid_per_symbol.csv, cell_summary.csv, selftest.csv, skipped.csv, integrity.json, summary_numbers.json. "
          "Run verify_exit_grid.py on this folder next.", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Output: {out_dir}", flush=True)
    print("\n".join(L[:6]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
