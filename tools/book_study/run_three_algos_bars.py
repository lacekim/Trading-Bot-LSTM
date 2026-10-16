#!/usr/bin/env python
"""The three algorithms (long, short, momentum), the learned-side forest (the book's first labeling mode: no side given, side AND size learned) and the forest gate, run on DOLLAR BARS (read-only diagnostic).

This is run_three_algos.py / three_algo.py UNCHANGED (same algorithms, forests, gate, variants, nulls, CPCV, walk-forward, windows, evaluation; their md5 is recorded), pointed at a finished
run_meta_bars.py run (its events.csv, built on dollar bars) instead of a candle-based one.  The only additions:
  * the "candle" reader is the bar reader (bar_loader.load_one): a series per symbol of bar-end closes, so that events.csv 'pos' indexes it;
  * the three momentum lookbacks stay 480 / 720 / 960 BARS (as in every earlier run: the lookback is counted in bars), and a lookback that reaches back across a data hole (a break longer than the
    venue's limit; see build_dollar_bars.py) is missing (NaN, so no MOM side), the same rule as the bar pipeline's own trailing returns;
  * a diagnostic of how many calendar days 480 / 720 / 960 bars span (median, 10th / 90th percentile) per venue, overall and after the cutoff, and how many lookbacks were dropped for a hole.

    python tools/book_study/run_three_algos_bars.py --source runtime/book_meta_kraken_bars/<timestamp> --prior-trials 121
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bar_loader as BL  # noqa: E402
import run_three_algos as RT  # noqa: E402
import three_algo as T  # noqa: E402

BARS = {}            # symbol -> bar_loader.Bars (filled by the reader; read by the patched extend_events)
HOLE_STATS = {}


class _BarCandles:
    """Stands in for candle_loader: the two folder arguments are the same bar folder; the second is ignored."""

    @staticmethod
    def load_closes(venue_dir, _unused_dir, names, timeframe="dollar", min_bars=0):
        man = BL.read_manifest(venue_dir)
        break_s = int(man["break_hours"]) * 3600
        out = {}
        for s in names:
            b = BL.load_one(venue_dir, s, break_s)
            BARS[s] = b
            out[s] = b.close_series()
        return out, []


_orig_extend = T.extend_events


def extend_events_bars(df: pd.DataFrame, closes: dict) -> pd.DataFrame:
    """The unchanged extend_events, then: a lookback whose first bar sits in an earlier data segment than the event's bar is missing; the MOM side is recomputed from what remains."""
    out = _orig_extend(df, closes)
    cols = {L: out[f"ret{L}"].to_numpy(np.float64).copy() for L in T.MOM_LOOKBACKS}
    HOLE_STATS.clear()
    for sym, grp in out.groupby("symbol", sort=False):
        seg = BARS[sym].seg
        ix = grp.index.to_numpy()
        p = grp["pos"].to_numpy(np.int64)
        for L in T.MOM_LOOKBACKS:
            ok = p >= L
            across = np.zeros(len(p), bool)
            across[ok] = seg[p[ok]] != seg[p[ok] - L]
            if across.any():
                cols[L][ix[across]] = np.nan
            h = HOLE_STATS.setdefault(str(L), {"events_with_lookback": 0, "dropped_for_hole": 0})
            h["events_with_lookback"] += int(ok.sum())
            h["dropped_for_hole"] += int(across.sum())
    for L in T.MOM_LOOKBACKS:
        out[f"ret{L}"] = cols[L]
    out["mom_side"] = T.mom_side_from(out["ret480"], out["ret720"], out["ret960"])
    return out


def span_days(df: pd.DataFrame, cutoff: str) -> dict:
    """Calendar days spanned by L bars ending at each event (bars at pos - L .. pos, same segment): median and 10th / 90th percentile, all events and events at or after the cutoff."""
    cut = float(pd.Timestamp(cutoff).timestamp())
    res = {}
    for L in T.MOM_LOOKBACKS:
        allv, post = [], []
        for sym, g in df.groupby("symbol", sort=False):
            b = BARS[sym]
            p = g["pos"].to_numpy(np.int64)
            t0 = g["t0"].to_numpy(np.float64)
            ok = (p >= L)
            ok[ok] = b.seg[p[ok]] == b.seg[p[ok] - L]
            d = (b.t[p[ok]] - b.t[p[ok] - L]) / 86400.0
            allv.append(d)
            post.append(d[t0[ok] >= cut])
        a, q = np.concatenate(allv), np.concatenate(post)
        pc = lambda x, z: float(np.percentile(x, z)) if len(x) else float("nan")      # noqa: E731
        res[str(L)] = {"events": int(len(a)), "median_days": pc(a, 50), "p10_days": pc(a, 10), "p90_days": pc(a, 90),
                       "events_post_cutoff": int(len(q)), "median_days_post_cutoff": pc(q, 50), "p10_days_post_cutoff": pc(q, 10), "p90_days_post_cutoff": pc(q, 90)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="timestamp folder of a run_meta_bars.py run (events.csv, symbols.csv, summary_numbers.json)")
    ap.add_argument("--out", default=None, help="default runtime/book_three_algo_<venue>_bars")
    ap.add_argument("--quick", action="store_true", help="smoke test only (4 groups, 10 trees)")
    ap.add_argument("--seed", type=int, default=RT.SEED)
    ap.add_argument("--prior-trials", type=int, required=True, help="earlier looks counted by the deflated Sharpe (the gate is the next one)")
    ap.add_argument("--bars", default=None, help="default: the bar folder recorded in the source run")
    a = ap.parse_args()
    src = Path(a.source)
    src = (src if src.is_absolute() else ROOT / src).resolve()
    s_summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    if s_summ["meta"].get("timeframe") != "dollar":
        raise SystemExit(f"the source run's timeframe is {s_summ['meta'].get('timeframe')!r}, not 'dollar'")
    venue = s_summ["meta"]["venue"]
    bd = Path(a.bars) if a.bars else Path(s_summ["meta"]["candle_dir"])
    bd = (bd if bd.is_absolute() else ROOT / bd).resolve()
    out_root = Path(a.out or f"runtime/book_three_algo_{venue}_bars")
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    before = {p.name for p in out_root.glob("*")} if out_root.exists() else set()
    RT.CL = _BarCandles
    T.extend_events = extend_events_bars
    RT.T.extend_events = extend_events_bars
    argv = ["run_three_algos.py", "--source", str(src), "--out", str(out_root.relative_to(ROOT)) if out_root.is_relative_to(ROOT) else str(out_root), "--seed", str(a.seed),
            "--prior-trials", str(a.prior_trials), "--candle-dirs", str(bd), str(bd)]
    if a.quick:
        argv.append("--quick")
    sys.argv = argv
    rc = RT.main()
    new = sorted(p for p in out_root.glob("*") if p.name not in before and (p / "summary_numbers.json").exists())
    if rc != 0 or not new:
        return rc or 1
    out = new[-1]
    df = pd.read_csv(src / "events.csv")
    spans = span_days(df, s_summ["params"]["cutoff"])
    man = BL.read_manifest(bd)
    extra = {"venue": venue, "bar_dir": str(bd), "bars_manifest_md5": RT.RB.md5_file(bd / "manifest.json"), "source_run": str(src), "prior_trials": a.prior_trials,
             "hole_rule": "a momentum lookback whose first bar lies in an earlier data segment than the event's bar is missing (no MOM side)", "hole_stats": HOLE_STATS, "span_days": spans,
             "break_hours": int(man["break_hours"]), "script_md5": {n: RT.RB.md5_file(HERE / n) for n in ("run_three_algos_bars.py", "bar_loader.py")}}
    (out / "venue_extra.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    L = ["", f"## Venue: {venue.capitalize()} dollar bars", "",
         f"Source events: {src} (dollar-bar run). Bars: {bd}. The study code is unchanged from the candle versions (run_three_algos.py / three_algo.py); only the reader, the hole rule and the trial count differ "
         f"(prior trials {a.prior_trials}; the gate is the {a.prior_trials + 1}th).",
         "The lookbacks 480 / 720 / 960 are counted in BARS, as in every earlier run. A bar holds about one average clock-hour of traded value, so their span in calendar days depends on activity. "
         "Calendar days spanned by the lookbacks (events whose lookback stays inside one data segment): median [10th, 90th percentile], all events | at or after the cutoff:", "",
         "| lookback (bars) | events | median days [p10, p90] | events after cutoff | median days [p10, p90] after cutoff | dropped for a data hole (of events with the lookback) |", "|---|---|---|---|---|---|"]
    for k, v in spans.items():
        hs = HOLE_STATS[k]
        L.append(f"| {k} | {v['events']} | {v['median_days']:.1f} [{v['p10_days']:.1f}, {v['p90_days']:.1f}] | {v['events_post_cutoff']} | "
                 f"{v['median_days_post_cutoff']:.1f} [{v['p10_days_post_cutoff']:.1f}, {v['p90_days_post_cutoff']:.1f}] | {hs['dropped_for_hole']} of {hs['events_with_lookback']} |")
    L += ["", "The DESIGN block at the top of this file is printed unchanged from the earlier version, so its trial sentence and its event-count remark describe that version. "
          f"This run's values are the parameter list and tables above: prior trials {a.prior_trials}, the gate the {a.prior_trials + 1}th, N = 1, 10, {a.prior_trials}, {a.prior_trials + 1}.",
          "The deviations list above is printed unchanged too; its sentence that the momentum lookbacks are 20 / 30 / 40 days of hourly bars does not hold on dollar bars (see the table). "
          "The momentum lookbacks and the 200-day market state were chosen on earlier history, not on these bars. One seed: the seed spread is not estimated. "
          "The limits of the source bar run apply (see its SUMMARY.md: survivorship in the symbol list, data holes, shorts without borrow cost).", ""]
    with open(out / "SUMMARY.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"venue additions written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
