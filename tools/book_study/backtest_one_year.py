"""One-year percentage-gain backtest of the book-built meta-labeling system, one venue per run (new file; nothing audited is edited).

What is traded (declared):
  * candidates: the venue's CUSUM events on its dollar bars (triple barrier 2/2, 4-day vertical barrier), side = sign of the trailing 720-bar return;
  * sizes: the meta model's bet size (Ch.10, step 0.1) from the WALK-FORWARD predictions (every prediction is out of sample: model trained only on events that ended before the
    holdout cutoff 2024-01-01 and then rolled forward);
  * positions: bar-by-bar averaging of the overlapping bets per instrument (bet_averaging.py, already audited), costs charged on every change of position
    (Kraken 12 bp per side, Alpaca 10 bp per side);
  * capital (headline, 'slot' convention): the account is cut into N_ref equal slots, N_ref = the median number of instruments with an active candidate on the venue's active days
    (fixed once from the candidates, not from any result).  On a day with n active instruments the account's return is (sum of their net results) / max(n, N_ref): fully invested when
    n >= N_ref, partly in cash when fewer.  No leverage.  The meta size is a fraction of the instrument's slot; the unused part is cash.
    Compounded: equity_t = equity_{t-1} * (1 + r_t).
  * second convention ('per_active', the one of the earlier averaged-bet evaluation): the whole account is split over the n active instruments of the day, r_t = mean net result.  It is shown
    because it is comparable with the earlier numbers, but on the few days with n = 1..3 (stray weekend bars of equities) it puts the whole account into one or two names.
  * variants: primary (every candidate, size 1), meta (model sizes), placebo (the same sizes shuffled among the same events, seed+7+99 stream as in the pipeline).
  * window: the last 365 UTC days ending on the last UTC day that has an active instrument; a second window (the whole walk-forward span) is given for context.
  * benchmark (context only): equal-weight daily-rebalanced buy and hold of the same instruments, from the last bar close of each UTC day, no return formed across a data hole.
Usage: python backtest_one_year.py --source <meta run folder> [--bars <dollar bar folder>] --out <out folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ports_common as PCm  # noqa: E402
import run_bet_averaging as RBA  # noqa: E402

DAY = 86400
VARIANTS = ("primary", "meta", "placebo")


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curve(bp_by_day: pd.Series, nact_by_day: pd.Series, days: np.ndarray, n_ref: int | None) -> pd.DataFrame:
    """Equity over every calendar day in `days` (inactive days = 0 return).  n_ref None: per_active convention; else slot convention."""
    bp = bp_by_day.reindex(days).fillna(0.0).to_numpy()
    n = nact_by_day.reindex(days).fillna(0).to_numpy(np.float64)
    r = (bp if n_ref is None else bp * n / np.maximum(n, n_ref)) / 1e4
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate(([1.0], eq)))[1:]
    return pd.DataFrame({"day": days, "ret": r, "equity": eq, "drawdown": eq / peak - 1.0})


def stats(c: pd.DataFrame) -> dict:
    r = c["ret"].to_numpy()
    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    return {"gain_pct": 100.0 * (float(c["equity"].iloc[-1]) - 1.0), "max_drawdown_pct": 100.0 * float(c["drawdown"].min()),
            "daily_sharpe_x_sqrt365": (float(r.mean()) / sd * math.sqrt(365)) if sd and np.isfinite(sd) and sd > 0 else float("nan"),
            "calendar_days": int(len(r)), "days_with_return_nonzero": int((r != 0).sum()), "best_day_pct": 100.0 * float(r.max()), "worst_day_pct": 100.0 * float(r.min())}


def monthly(c: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(c["day"].to_numpy() * DAY, unit="s")
    g = pd.DataFrame({"m": d.strftime("%Y-%m"), "f": 1.0 + c["ret"].to_numpy()}).groupby("m")["f"].prod()
    return (100.0 * (g - 1.0)).rename("gain_pct")


def benchmark(bars: dict, max_gap_days: int, lo_day: int, hi_day: int, syms: list) -> np.ndarray:
    dc = PCm.trading_days(PCm.daily_closes({s: bars[s] for s in syms}), .8)       # drop days on which fewer than 80% of the instruments have a bar (stray weekend bars)
    r = PCm.returns_within_gap(dc, max_gap_days, "simple")
    day = (r.index.to_numpy().astype("datetime64[s]").astype(np.int64)) // DAY
    m = r.mean(axis=1, skipna=True).fillna(0.0).to_numpy()
    s = pd.Series(m, index=day)
    return s.reindex(np.arange(lo_day, hi_day + 1)).fillna(0.0).to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--bars", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.source).resolve()
    summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    venue = summ["meta"]["venue"]
    bd = Path(a.bars) if a.bars else Path(summ["meta"]["candle_dir"])
    V = PCm.Venue(venue, src, bd, 40)
    ev = V.events
    p = summ["params"]
    step, seed, cost_frac = float(p["step"]), int(p["seed"]), float(p["cost_frac"])
    allsyms = sorted(ev["symbol"].unique())
    bars = V.bars(allsyms)
    closes = {s: pd.Series(b.close, index=b.idx) for s, b in bars.items()}
    RBA.check_alignment(ev, closes)
    windows = {k: (int(v[0]), int(v[1])) for k, v in summ["meta"]["window_epoch"].items()}
    wfp = pd.read_csv(src / "walk_forward_predictions.csv")
    plist = RBA.sizes_by_path(wfp, step, seed + 7)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = {}
    for label, cs in (("net", cost_frac / 2.0), ("before_costs", 0.0)):
        _, drow, _ = RBA.evaluate_scope(ev, closes, "wf", plist, cs, cost_frac, windows)
        res[label] = pd.DataFrame(drow)
    dn = res["net"]
    if dn["path"].nunique() != 1:
        raise SystemExit("walk-forward has more than one path; this script expects one")
    last_day = int(dn["day"].max())
    first_day = int(dn["day"].min())
    n_ref = int(np.median(dn[dn["variant"] == "primary"]["n_active"].to_numpy()))
    spans = {"last_365_days": (last_day - 364, last_day), "whole_walk_forward_span": (first_day, last_day)}
    # instruments that traded in each window (for the benchmark)
    evd = pd.DataFrame({"s": ev["symbol"].to_numpy(), "d0": ev["t0"].to_numpy() // DAY, "d1": ev["t1"].to_numpy() // DAY})
    rows, curves, mrows = [], [], {}
    for sname, (lo, hi) in spans.items():
        days = np.arange(lo, hi + 1)
        syms = sorted(evd[(evd["d1"] >= lo) & (evd["d0"] <= hi)]["s"].unique())
        bm = benchmark(bars, V.max_gap_days, lo, hi, syms)
        bc = pd.DataFrame({"day": days, "ret": bm})
        bc["equity"] = np.cumprod(1.0 + bc["ret"].to_numpy())
        bc["drawdown"] = bc["equity"] / np.maximum.accumulate(bc["equity"].to_numpy()) - 1.0
        for conv, cost_label, d in [(cv, cl, dd) for cv in ("slot", "per_active") for cl, dd in res.items()]:
            for v in VARIANTS:
                g = d[d["variant"] == v].set_index("day")["bp"]
                nact_s = d[d["variant"] == v].set_index("day")["n_active"]
                c = curve(g, nact_s, days, n_ref if conv == "slot" else None)
                st = stats(c)
                nact = nact_s.reindex(days).fillna(0)
                st.update({"span": sname, "convention": conv, "costs": cost_label, "variant": v, "start_day": pd.Timestamp(lo * DAY, unit="s").strftime("%Y-%m-%d"),
                           "end_day": pd.Timestamp(hi * DAY, unit="s").strftime("%Y-%m-%d"), "mean_active_instruments": float(nact[nact > 0].mean()), "active_days": int((nact > 0).sum())})
                rows.append(st)
                cc = c.assign(span=sname, convention=conv, costs=cost_label, variant=v)
                curves.append(cc)
                if cost_label == "net" and conv == "slot":
                    mrows[(sname, v)] = monthly(c)
        bs = stats(bc)
        bs.update({"span": sname, "convention": "-", "costs": "none (buy and hold)", "variant": "equal_weight_buy_and_hold", "start_day": pd.Timestamp(lo * DAY, unit="s").strftime("%Y-%m-%d"),
                   "end_day": pd.Timestamp(hi * DAY, unit="s").strftime("%Y-%m-%d"), "instruments": len(syms)})
        rows.append(bs)
        curves.append(bc.assign(span=sname, convention="-", costs="none", variant="equal_weight_buy_and_hold"))
        mrows[(sname, "benchmark")] = monthly(bc)
    S = pd.DataFrame(rows)
    C = pd.concat(curves, ignore_index=True)
    S.to_csv(out / "summary.csv", index=False, float_format="%.10g")
    C.to_csv(out / "equity_curves.csv", index=False, float_format="%.12g")
    M = pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in mrows.items()})
    M.to_csv(out / "monthly_gain_pct.csv", float_format="%.10g")
    meta = {"venue": venue, "source_run": str(src), "bar_dir": str(bd), "seed": seed, "step": step, "cost_frac_round_trip": cost_frac, "cost_bp_per_side": cost_frac / 2.0 * 1e4,
            "n_ref_slots": n_ref, "events": int(len(ev)), "symbols": len(allsyms), "walk_forward_rows": int(len(wfp)), "first_day": pd.Timestamp(first_day * DAY, unit="s").strftime("%Y-%m-%d"),
            "last_day": pd.Timestamp(last_day * DAY, unit="s").strftime("%Y-%m-%d"),
            "script_md5": {n: md5_file(HERE / n) for n in ("backtest_one_year.py", "bet_averaging.py", "run_bet_averaging.py", "ports_common.py")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # short markdown, numbers straight from the frames
    L = [f"# One-year backtest, {venue} ({meta['first_day']} .. {meta['last_day']} out-of-sample)", "",
         f"N_ref = {n_ref} slots. Costs {meta['cost_bp_per_side']:.0f} bp per side. Candidates {meta['events']:,} events on {meta['symbols']} instruments. Equity starts at 100% and compounds daily; inactive days earn 0.", ""]
    for sname in spans:
        x = S[S["span"] == sname]
        L += [f"## {sname}: {x['start_day'].iloc[0]} to {x['end_day'].iloc[0]} ({int(x['calendar_days'].iloc[0])} days)", "",
              "| variant | convention | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) | active days | mean active instruments |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for _, r in x.iterrows():
            L.append(f"| {r['variant']} | {r['convention']} | {r['costs']} | {r['gain_pct']:.2f} | {r['max_drawdown_pct']:.2f} | {r['daily_sharpe_x_sqrt365']:.2f} | "
                     f"{'' if pd.isna(r.get('active_days')) else int(r['active_days'])} | {'' if pd.isna(r.get('mean_active_instruments')) else format(r['mean_active_instruments'], '.1f')} |")
        L.append("")
    (out / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
