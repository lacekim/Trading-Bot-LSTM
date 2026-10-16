#!/usr/bin/env python
"""Three algorithms (long, short, momentum) and the forest gate, run on ALPACA DAILY bars (read-only diagnostic).

Same study as run_three_algos.py (forests, gate, variants, nulls, CPCV, walk-forward, windows, states, tables), applied to the event table of a finished Alpaca daily meta run
(run_meta_daily.py).  The study code is three_algo.py executed with three recorded substitutions (three_algo_daily.py); the MOM rule is the project's 12-1 / 6-1 month agreement.
The new pieces here: reading the closes from alpaca_daily.csv.gz (md5 checked against the source run), the daily MOM columns, the declared trial count, and a venue section.

Trial count: the running count is 116 after the Kraken three-algorithm study; the four choices made here add 4 -> 120 prior trials, the gate is the 121st; deflated Sharpe at N = 1, 10, 120, 121.

    python tools/book_study/run_three_algos_daily.py --source "$(ls -d runtime/book_meta_alpaca/*/ | tail -1)"
    python tools/book_study/verify_three_algos_daily.py "$(ls -d runtime/book_three_algo_alpaca/*/ | tail -1)"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import equity_events as E  # noqa: E402
import run_book_study as RB  # noqa: E402
import run_three_algos as RT  # noqa: E402       (render() and fmt(); it reads its module-level T, which is set to the daily module below)
import three_algo_daily as T  # noqa: E402

SEED = 20261018          # fixed before any result of this module was seen
PRIOR_TRIALS_DAILY = 120
RT.T = T


def load_source_closes(csv_path: Path, sym_rows: pd.DataFrame) -> dict:
    """{symbol: close Series} exactly as the source run built them (closes > 0, one row per date), cut at the source run's last bar; the bar counts must match symbols.csv."""
    names = set(sym_rows["symbol"].astype(str))
    raw = E.load_alpaca(csv_path)
    raw = raw[raw["symbol"].isin(names)]
    close_df, _ = E.pivots(raw)
    closes = {}
    for _, r in sym_rows.iterrows():
        s = str(r["symbol"])
        c = close_df[s].dropna()
        c = c[c.index <= pd.Timestamp(r["last"])]
        if len(c) != int(r["bars"]):
            raise SystemExit(f"{s}: {len(c)} bars up to {r['last']}, the source run used {int(r['bars'])}; the file changed, stopping")
        closes[s] = c
    return closes


def venue_section(src: Path, csv_path: Path, csv_md5: str, df: pd.DataFrame, windows: dict, prior: int) -> list:
    mom = df["mom_side"].to_numpy()
    side = df["side"].to_numpy()
    n_mom = int((mom != 0).sum())
    L = ["", "## Venue: Alpaca daily bars", "",
         f"Source events: {src}. Closes: {csv_path} (md5 {csv_md5}). Study code: three_algo.py with the three substitutions listed in three_algo_daily.py (annualisation sqrt(252), effective sample = days / 5, daily features); trial count: prior trials {prior}; the gate is the {prior + 1}st.",
         "", f"MOM events {n_mom:,} of {len(df):,} ({n_mom / len(df):.4f}); on every one the MOM side equals the source side: {bool(np.all(mom[mom != 0] == side[mom != 0]))}. "
         "MOM is therefore the 'both returns agree' subset of the primary strategy, not an independent signal.", "",
         "Events entered inside each reporting window (the event table holds labelled events only):", "", "| window | events entered inside | first event day | last event day |", "|---|---|---|---|"]
    t0 = df["t0"].to_numpy(np.int64)
    for wn, (lo, hi) in windows.items():
        m = (t0 >= lo) & (t0 < hi)
        first = pd.Timestamp(int(t0[m].min()), unit="s").date() if m.any() else "n/a"
        last = pd.Timestamp(int(t0[m].max()), unit="s").date() if m.any() else "n/a"
        L.append(f"| {wn} | {int(m.sum())} | {first} | {last} |")
    L += ["", "A window with no events is reported n/a, not filled in: a label needs 5 bars after entry, so events entered in the last 5 trading days of the file are never labelled. "
          "Limits of the source run apply: survivorship in the symbol list (optimistic for long momentum), top-200 universe, no high / low, shorts without borrow cost. One seed: the seed spread is not estimated.", ""]
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="timestamp folder of a finished run_meta_daily.py run (default: the newest under runtime/book_meta_alpaca)")
    ap.add_argument("--alpaca-csv", default=None, help="alpaca_daily.csv.gz (default: the file recorded in the source run, if it exists)")
    ap.add_argument("--out", default="runtime/book_three_algo_alpaca")
    ap.add_argument("--quick", action="store_true", help="smoke test only (4 groups, 10 trees)")
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-draws", type=int, default=10000)
    ap.add_argument("--bootstrap", choices=("sequential", "standard"), default="sequential")
    ap.add_argument("--embargo-pct", type=float, default=0.01)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--cutoff", default="2024-01-01")
    ap.add_argument("--prior-trials", type=int, default=PRIOR_TRIALS_DAILY)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.quick:
        args.n_groups, args.n_estimators, args.max_draws = 4, 10, 3000
    if args.source:
        src = Path(args.source)
    else:
        runs = sorted(p for p in (ROOT / "runtime" / "book_meta_alpaca").glob("*") if (p / "events.csv").exists())
        if not runs:
            raise SystemExit("no Alpaca daily meta run found under runtime/book_meta_alpaca; pass --source")
        src = runs[-1]
    src = (src if src.is_absolute() else ROOT / src).resolve()
    out_root = Path(args.out)
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    started = datetime.now(timezone.utc)
    out_dir = out_root / started.strftime("%Y%m%dT%H%M%SZ")
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
    if s_meta.get("venue") != "alpaca" or s_meta.get("timeframe") != "1d":
        raise SystemExit(f"the source run's venue / timeframe are {s_meta.get('venue')!r} / {s_meta.get('timeframe')!r}, not 'alpaca' / '1d'")
    if s_params["features"] != E.FEATURES or s_params["side_signal"] != E.SIDE_SIGNAL:
        raise SystemExit("the source run's features / side signal differ from equity_events.py's; stopping")
    csv_path = Path(args.alpaca_csv) if args.alpaca_csv else Path(s_meta["alpaca_csv"])
    csv_path = (csv_path if csv_path.is_absolute() else ROOT / csv_path).resolve()
    if not csv_path.exists():
        raise SystemExit(f"the closes file {csv_path} does not exist; pass --alpaca-csv")
    csv_md5 = RB.md5_file(csv_path)
    if csv_md5 != s_meta["alpaca_csv_md5"]:
        raise SystemExit(f"{csv_path} has md5 {csv_md5}, the source run used {s_meta['alpaca_csv_md5']}; stopping")
    for name in ("equity_events.py", "meta_events.py", "meta_model.py", "book_stats.py"):
        if RB.md5_file(HERE / name) != s_meta["script_md5"][name]:
            raise SystemExit(f"{name} next to this script is not the version that produced the source run; stopping")
    log(f"source meta run {src} ({s_meta['events']:,} events, run {s_meta['run_utc']}); closes {csv_path}")
    df = pd.read_csv(src / "events.csv")
    if len(df) != int(s_meta["events"]):
        raise SystemExit(f"events.csv has {len(df)} rows, the source run says {s_meta['events']}")
    sym_rows = pd.read_csv(src / "symbols.csv", keep_default_na=False, na_values=[""])
    sym_rows["symbol"] = sym_rows["symbol"].astype(str)
    closes = load_source_closes(csv_path, sym_rows)
    log(f"loaded {len(closes)} symbols, bar counts equal the source run's")
    df = T.extend_events(df, closes)
    n_mom = int((df["mom_side"] != 0).sum())
    log(f"events {len(df):,}; MOM events {n_mom:,} (long {int((df['mom_side'] == 1).sum()):,}, short {int((df['mom_side'] == -1).sum()):,}); y_dir share {df['y_dir'].mean():.3f}")
    ext_cols = ["symbol", "pos", "t_sig", "t0", *T.MOM_COLUMNS, "mom_side", "ret_long", "y_dir"]
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
              "mom_lookbacks": list(T.MOM_LOOKBACKS), "mom_skip": T.SKIP, "min_weight_fraction_leaf": 0.05, "variants": list(T.VARIANTS), "dsr_variants": list(T.DSR_VARIANTS)}
    script_names = ("book_stats.py", "equity_events.py", "meta_events.py", "meta_model.py", "meta_eval.py", "three_algo.py", "three_algo_daily.py", "run_three_algos.py", "run_three_algos_daily.py")
    meta = {**cmeta, "run_utc": started.isoformat(timespec="seconds"), "source_folder": str(src), "source_events_md5": RB.md5_file(src / "events.csv"),
            "source_summary_md5": RB.md5_file(src / "summary_numbers.json"), "events": int(len(df)), "mom_events": n_mom, "mom_long": int((df["mom_side"] == 1).sum()),
            "mom_short": int((df["mom_side"] == -1).sum()), "windows": s_meta["windows"], "window_epoch": s_meta["window_epoch"], "window_notes": windows_txt,
            "venue": "alpaca", "alpaca_csv": str(csv_path), "alpaca_csv_md5": csv_md5, "timeframe": "1d", "three_algo_md5_base": T.BASE_MD5,
            "script_md5": {n: RB.md5_file(HERE / n) for n in script_names}}
    summ = {"meta": meta, "params": params,
            "params_shown": {"symbols": len(closes), "round-trip cost (bp)": f"{cost_frac * 1e4:.1f}", "barriers": "as the source run (+/- 2.0 x daily vol, vertical barrier 5 bars after entry)",
                             "forest": f"{args.n_estimators} trees, {args.bootstrap} bootstrap, max {args.max_draws:,} draws per tree", "bet-size step": args.step, "seed": args.seed,
                             "prior trials for the deflated Sharpe": args.prior_trials, "windows": " | ".join(f"{k_}: {v}" for k_, v in windows_txt.items())},
            "integrity_unchanged": before == after,
            "cpcv": {"auc_dir_mean": float(smet["auc_dir"].mean()), "auc_dir_min": float(smet["auc_dir"].min()), "auc_dir_max": float(smet["auc_dir"].max()),
                     "auc_mom_mean": float(smet["auc_mom"].mean()), "auc_mom_min": float(smet["auc_mom"].min()), "auc_mom_max": float(smet["auc_mom"].max()),
                     "share_dir_mean": float(smet["share_dir_gt_half"].mean()), "share_mom_mean": float(smet["share_mom_gt_half"].mean())},
            "eval": {k_: v for k_, v in ev.items() if k_ not in ("daily", "paths")}, "walk_forward": wmet,
            "walk_forward_eval": {k_: v for k_, v in wev.items() if k_ not in ("daily", "paths")}}
    (out_dir / "summary_numbers.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    md = RT.render(summ) + "\n".join(venue_section(src, csv_path, csv_md5, df, win_t, args.prior_trials))
    (out_dir / "SUMMARY.md").write_text(md, encoding="utf-8")
    log(f"done in {time.time() - started.timestamp():.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
