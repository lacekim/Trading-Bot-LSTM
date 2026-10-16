#!/usr/bin/env python
"""Ch.10 bar-by-bar averaging of active bets, applied to the predictions of an EXISTING run (read-only: no model is fitted, nothing in the run folder is touched).

For a run folder written by run_meta_venue.py / run_meta_volume.py / run_meta_decay.py it
  1. reads events.csv, cpcv_predictions.csv, walk_forward_predictions.csv, summary_numbers.json, paths.csv and the candle files named in the run (or --candle-dir);
  2. checks that every event's bar indices match the candle files (timestamp of the entry and exit bar, side-signed return equal to ret_side within 1e-6), and that the
     per-event convention recomputed here (same bet sizes, same placebo shuffle) reproduces the mean bp per candidate event of every path and variant in paths.csv;
  3. evaluates primary / meta / placebo under the averaging rule of bet_averaging.py for each CPCV backtest path and for the walk-forward fit.
No new model is fitted, so this adds no look to the trial count.

    python tools/book_study/run_bet_averaging.py --run runtime/book_meta_kraken_v2/20261013T022218Z
    python tools/book_study/verify_bet_averaging.py runtime/book_bet_avg/<name>

Output folder (default runtime/book_bet_avg/<parent folder>__<timestamp folder>): averaged_paths.csv, averaged_daily.csv, averaged_windows.csv, averaged_sizes.csv (the meta and
placebo bet sizes used), averaged_summary.json, averaged_summary.md, averaged_extra.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import bet_averaging as BA  # noqa: E402
import book_stats as B  # noqa: E402
import meta_model as MM  # noqa: E402
import run_book_study as RB  # noqa: E402
import venue_loader as VL  # noqa: E402

VARIANTS = ("primary", "meta", "placebo")


def load_inputs(run: Path, candle_dir: Path | None):
    summ = json.loads((run / "summary_numbers.json").read_text(encoding="utf-8"))
    ev = pd.read_csv(run / "events.csv")
    cd = Path(candle_dir) if candle_dir else Path(summ["meta"]["candle_dir"])
    if not cd.is_absolute():
        cd = ROOT / cd
    closes, _ = VL.load_closes(cd, sorted(ev["symbol"].unique()), summ["meta"].get("timeframe", "1h"), 0)
    missing = sorted(set(ev["symbol"]) - set(closes))
    if missing:
        raise SystemExit(f"candle files missing for {missing} in {cd}")
    return summ, ev, closes, cd


def check_alignment(ev: pd.DataFrame, closes: dict) -> None:
    """Entry / exit bar timestamps and the side-signed return of every event must agree with the candle files."""
    for sym, g in ev.groupby("symbol"):
        c = closes[sym]
        cl, ts = c.to_numpy(dtype=np.float64), (c.index.astype("datetime64[s]").astype(np.int64))
        pos, t1p = g["pos"].to_numpy(np.int64), g["t1pos"].to_numpy(np.int64)
        if pos.min() < 0 or t1p.max() >= len(cl):
            raise SystemExit(f"{sym}: event bar index outside the candle series")
        if not (ts[pos] == g["t0"].to_numpy(np.int64)).all() or not (ts[t1p] == g["t1"].to_numpy(np.int64)).all():
            raise SystemExit(f"{sym}: event timestamps do not match the candle bars (wrong candle folder?)")
        r = g["side"].to_numpy(np.float64) * (cl[t1p] / cl[pos] - 1.0)
        if np.abs(r - g["ret_side"].to_numpy(np.float64)).max() > 1e-6:
            raise SystemExit(f"{sym}: side-signed return recomputed from the candles differs from ret_side")


def sizes_by_path(preds: pd.DataFrame, step: float, seed: int):
    """[(path, event index array, p, meta size, placebo size)] reproducing meta_eval.evaluate: rng = default_rng(seed + 99), one permutation per path in path order."""
    rng = np.random.default_rng(seed + 99)
    out = []
    for path, g in preds.groupby("path"):
        idx = g["event"].to_numpy(np.int64)
        size = MM.bet_size(g["p"].to_numpy(np.float64), step)
        out.append((int(path), idx, size, rng.permutation(size)))
    return out


def event_convention_check(ev, preds, step, seed, cost_frac, paths_csv: pd.DataFrame, label: str) -> None:
    ret = ev["ret_side"].to_numpy(np.float64)
    for path, idx, size, plc in sizes_by_path(preds, step, seed):
        for v, s in (("primary", np.ones(len(idx))), ("meta", size), ("placebo", plc)):
            got = float((s * (ret[idx] - cost_frac * (s > 0)) * 1e4).mean())
            row = paths_csv[(paths_csv["path"] == path) & (paths_csv["variant"] == v)]
            if len(row) != 1 or abs(float(row["mean_bp_per_candidate"].iloc[0]) - got) > 1e-9 * max(1.0, abs(got)):
                raise SystemExit(f"{label}: per-event mean bp of path {path} {v} not reproduced ({got} vs {row['mean_bp_per_candidate'].tolist()}); the inputs are not the run's")


def evaluate_scope(ev, closes, scope, plist, cost_side, cost_frac, windows):
    """plist: output of sizes_by_path.  Returns (paths rows, daily rows, window rows)."""
    side = ev["side"].to_numpy(np.float64)
    sym_arr = ev["symbol"].to_numpy()
    pos_all, t1_all = ev["pos"].to_numpy(np.int64), ev["t1pos"].to_numpy(np.int64)
    ts_cache = {s: c.index.astype("datetime64[s]").astype(np.int64) for s, c in closes.items()}
    cl_cache = {s: c.to_numpy(dtype=np.float64) for s, c in closes.items()}
    prow, drow, wrow = [], [], []
    for path, idx, size, plc in plist:
        by_sym = {}
        for j, e in enumerate(idx):
            by_sym.setdefault(sym_arr[e], []).append(j)
        for v, s in (("primary", np.ones(len(idx))), ("meta", size), ("placebo", plc)):
            parts = []
            for sym, js in by_sym.items():
                js = np.asarray(js)
                e = idx[js]
                parts.append(BA.symbol_daily(cl_cache[sym], ts_cache[sym], pos_all[e], t1_all[e], side[e] * s[js], cost_side))
            comb = BA.combine_days(parts)
            days, bp, n_act = BA.daily_bp(comb)
            mom = B.sharpe_moments(bp)
            sd = float(n_act.sum())
            prow.append({"scope": scope, "path": path, "variant": v, "days": int(len(days)), "symbol_days": int(sd),
                         "mean_bp_per_symbol_day": 1e4 * float(comb["net"].sum()) / sd, "gross_bp_per_symbol_day": 1e4 * float(comb["gross"].sum()) / sd,
                         "cost_bp_per_symbol_day": 1e4 * float(comb["cost"].sum()) / sd,
                         "event_convention_cost_bp_per_symbol_day": 1e4 * cost_frac * float(s.sum()) / sd,
                         "sr_daily": mom["sr"], "sr_annual": mom["sr"] * math.sqrt(365) if np.isfinite(mom["sr"]) else float("nan")})
            for d_, b_, n_ in zip(days, bp, n_act):
                drow.append({"scope": scope, "path": path, "variant": v, "day": int(d_), "bp": float(b_), "n_active": int(n_)})
            for wn, (lo, hi) in windows.items():
                b_, n_ = BA.window_bp(comb, lo, hi)
                wrow.append({"scope": scope, "path": path, "variant": v, "window": wn, "bp_per_symbol_day": b_, "symbol_days": n_})
    return prow, drow, wrow


def summarise(paths: pd.DataFrame, wins: pd.DataFrame) -> dict:
    out = {}
    for scope, g in paths.groupby("scope"):
        d = {}
        for v, gv in g.groupby("variant"):
            d[v] = {k: float(gv[k].mean()) for k in ("mean_bp_per_symbol_day", "gross_bp_per_symbol_day", "cost_bp_per_symbol_day", "event_convention_cost_bp_per_symbol_day", "sr_daily", "sr_annual")}
            d[v]["paths"] = int(len(gv))
            d[v]["paths_sr_negative"] = int((gv["sr_daily"] < 0).sum())
        piv = g.pivot(index="path", columns="variant", values="sr_daily")
        pivb = g.pivot(index="path", columns="variant", values="mean_bp_per_symbol_day")
        d["meta_minus_placebo"] = {"paths_sr_higher": int((piv["meta"] > piv["placebo"]).sum()), "paths": int(len(piv)),
                                   "mean_sr_daily_diff": float((piv["meta"] - piv["placebo"]).mean()), "mean_bp_diff": float((pivb["meta"] - pivb["placebo"]).mean())}
        d["meta_minus_primary"] = {"paths_sr_higher": int((piv["meta"] > piv["primary"]).sum()), "paths": int(len(piv)), "mean_sr_daily_diff": float((piv["meta"] - piv["primary"]).mean())}
        w = {}
        for (v, wn), gw in wins[wins["scope"] == scope].groupby(["variant", "window"]):
            w[f"{v}|{wn}"] = {"mean_bp_per_symbol_day": float(gw["bp_per_symbol_day"].mean()), "symbol_days_per_path": float(gw["symbol_days"].mean())}
        d["windows"] = w
        out[scope] = d
    return out


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def render(summ_run: dict, S: dict, extra: dict) -> str:
    ev_ = summ_run["eval"]["summary"]
    L = ["# Ch.10 bar-by-bar averaging of active bets, applied to an existing run (read-only diagnostic)", "",
         f"Run: {extra['source_run']}. Seed {extra['seed']}, bet-size step {extra['step']}, cost {extra['cost_bp_per_side']} bp per side. No model was fitted.", "",
         "Rule (see bet_averaging.py): position of an instrument at a bar = mean signal (side x size) of its events active at that bar, zeros included; result per UTC day = sum over instruments of "
         "(position x bar return - cost x change in position) divided by the number of instruments with a candidate event active that day. Not comparable in level with the per-event numbers.", ""]
    for scope, title in (("cpcv", "Combinatorial purged CV paths"), ("wf", "Walk-forward fit")):
        d = S[scope]
        L += [f"## {title} - averaged convention", "", "| variant | mean bp per active instrument-day | of which gross | cost | cost under the per-event rule | annualised Sharpe | paths with Sharpe < 0 |", "|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            x = d[v]
            L.append(f"| {v} | {fmt(x['mean_bp_per_symbol_day'])} | {fmt(x['gross_bp_per_symbol_day'])} | {fmt(x['cost_bp_per_symbol_day'])} | {fmt(x['event_convention_cost_bp_per_symbol_day'])} | {fmt(x['sr_annual'])} | {x['paths_sr_negative']}/{x['paths']} |")
        mp, mq = d["meta_minus_placebo"], d["meta_minus_primary"]
        L += ["", f"Meta minus placebo: Sharpe higher on {mp['paths_sr_higher']}/{mp['paths']} paths, mean daily-Sharpe difference {fmt(mp['mean_sr_daily_diff'], 4)}, mean bp difference {fmt(mp['mean_bp_diff'])}. "
                  f"Meta minus primary: Sharpe higher on {mq['paths_sr_higher']}/{mq['paths']} paths, mean daily-Sharpe difference {fmt(mq['mean_sr_daily_diff'], 4)}.", ""]
        L += ["Reporting windows (mean bp per active instrument-day, averaged over paths):", "", "| window | primary | meta | placebo | instrument-days per path |", "|---|---|---|---|---|"]
        for wn in extra["windows"]:
            r = [d["windows"].get(f"{v}|{wn}", {}) for v in VARIANTS]
            L.append(f"| {wn} | " + " | ".join(fmt(x.get("mean_bp_per_symbol_day")) for x in r) + f" | {fmt(r[0].get('symbol_days_per_path'), 1)} |")
        L.append("")
    L += ["## For comparison: the run's own per-event convention (CPCV, from its summary_numbers.json)", "", "| variant | mean bp per candidate event | annualised Sharpe |", "|---|---|---|"]
    for v in VARIANTS:
        L.append(f"| {v} | {fmt(ev_[v]['mean_bp_per_candidate'])} | {fmt(ev_[v]['mean_sr_annual'])} |")
    L.append("")
    return "\n".join(L)


def run_one(run: Path, out: Path, candle_dir) -> Path:
    summ, ev, closes, cd = load_inputs(run, candle_dir)
    p = summ["params"]
    step, seed, cost_frac = float(p["step"]), int(p["seed"]), float(p["cost_frac"])
    cost_side = cost_frac / 2.0
    check_alignment(ev, closes)
    paths_csv = pd.read_csv(run / "paths.csv")
    cp, wfp = pd.read_csv(run / "cpcv_predictions.csv"), pd.read_csv(run / "walk_forward_predictions.csv")
    event_convention_check(ev, cp, step, seed, cost_frac, paths_csv, "CPCV")
    windows = {k: (int(v[0]), int(v[1])) for k, v in summ["meta"]["window_epoch"].items()}
    # walk-forward: meta_eval.evaluate was called with seed + 7, so its placebo stream is default_rng(seed + 7 + 99)
    pl1, pl2 = sizes_by_path(cp, step, seed), sizes_by_path(wfp, step, seed + 7)
    p1 = evaluate_scope(ev, closes, "cpcv", pl1, cost_side, cost_frac, windows)
    p2 = evaluate_scope(ev, closes, "wf", pl2, cost_side, cost_frac, windows)
    sizes = pd.concat([pd.DataFrame({"scope": sc, "path": path, "event": idx, "size_meta": size, "size_placebo": plc})
                       for sc, pl in (("cpcv", pl1), ("wf", pl2)) for path, idx, size, plc in pl], ignore_index=True)
    paths = pd.DataFrame(p1[0] + p2[0])
    daily = pd.DataFrame(p1[1] + p2[1])
    wins = pd.DataFrame(p1[2] + p2[2])
    S = summarise(paths, wins)
    out.mkdir(parents=True, exist_ok=False)
    paths.to_csv(out / "averaged_paths.csv", index=False, float_format="%.12g")
    daily.to_csv(out / "averaged_daily.csv", index=False, float_format="%.12g")
    wins.to_csv(out / "averaged_windows.csv", index=False, float_format="%.12g")
    sizes.to_csv(out / "averaged_sizes.csv", index=False, float_format="%.12g")
    (out / "averaged_summary.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    extra = {"source_run": str(run), "candle_dir": str(cd), "seed": seed, "step": step, "cost_frac": cost_frac, "cost_bp_per_side": round(cost_side * 1e4, 9), "events": int(len(ev)),
             "windows": {k: list(v) for k, v in windows.items()}, "script_md5": {n: RB.md5_file(HERE / n) for n in ("bet_averaging.py", "run_bet_averaging.py")}}
    (out / "averaged_extra.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    (out / "averaged_summary.md").write_text(render(summ, S, extra), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run folder (the timestamp folder) of run_meta_venue.py / run_meta_volume.py / run_meta_decay.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--candle-dir", default=None, help="default: the candle folder recorded in the run")
    a = ap.parse_args()
    run = Path(a.run)
    if not run.is_absolute():
        run = ROOT / run
    run = run.resolve()
    out = Path(a.out) if a.out else ROOT / "runtime" / "book_bet_avg" / f"{run.parent.name}__{run.name}"
    if not out.is_absolute():
        out = ROOT / out
    res = run_one(run, out, a.candle_dir)
    print((res / "averaged_summary.md").read_text(encoding="utf-8"))
    print(f"written to {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
