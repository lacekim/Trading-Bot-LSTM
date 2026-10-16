"""Independent re-derivation of backtest_one_year.py from the raw files (events.csv, walk_forward_predictions.csv, bar csv files).
Shares NO code with the backtest, the averaging module or the bar loader: its own bet-size function (math.erf), its own per-bar position loop (slice adds), its own day booking, its own
compounding loop in plain Python, its own benchmark.  Events are located on the bars by their start / end TIMESTAMPS, not by the stored bar numbers.
Usage: python verify_backtest_one_year.py <backtest out folder>      Exit code 1 when any number differs by more than 1e-9 relative (or 1e-9 absolute near zero).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400


def epoch(s: pd.Series) -> np.ndarray:
    return np.array([int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()) for x in s], dtype=np.int64)


def bet_size(p: float, step: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    if p <= 0.5:
        return 0.0
    z = (p - 0.5) / math.sqrt(p * (1 - p))
    s = 2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))) - 1.0
    s = round(s / step) * step                      # python round = half to even, as numpy.round
    return min(max(s, 0.0), 1.0)


def load_bars(bar_dir: Path, sym: str):
    b = pd.read_csv(bar_dir / "bars" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "close"])
    b = b.dropna(subset=["close"])
    b = b[b["close"] > 0]
    t = epoch(b["date"])
    c = b["close"].to_numpy(float)
    o = np.argsort(t, kind="stable")
    t, c = t[o], c[o]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], c[keep]


def main(out: Path) -> int:
    meta = json.loads((out / "meta.json").read_text())
    src, bar_dir = Path(meta["source_run"]), Path(meta["bar_dir"])
    step, seed, cost_frac = meta["step"], meta["seed"], meta["cost_frac_round_trip"]
    ev = pd.read_csv(src / "events.csv", usecols=["symbol", "t0", "t1", "side", "pos", "t1pos"])
    wf = pd.read_csv(src / "walk_forward_predictions.csv")
    assert wf["path"].nunique() == 1
    size = np.array([bet_size(float(p), step) for p in wf["p"]])
    plc = np.random.default_rng(seed + 7 + 99).permutation(size)
    sz = {"primary": {int(e): 1.0 for e in wf["event"]}, "meta": {int(e): float(s) for e, s in zip(wf["event"], size)}, "placebo": {int(e): float(s) for e, s in zip(wf["event"], plc)}}
    wf_events = set(int(e) for e in wf["event"])
    ev = ev.loc[sorted(wf_events)]
    bars = {}
    for sym in sorted(pd.read_csv(src / "events.csv", usecols=["symbol"])["symbol"].unique()):
        bars[sym] = load_bars(bar_dir, sym)
    problems = []

    def book(cost_side):
        """per variant: dict day -> [net_sum, n_active]"""
        res = {}
        for v in ("primary", "meta", "placebo"):
            days = {}
            for sym, g in ev.groupby("symbol"):
                t, c = bars[sym]
                n = len(t)
                s = np.zeros(n)
                k = np.zeros(n, dtype=np.int64)
                for e, t0, t1, side in zip(g.index, g["t0"], g["t1"], g["side"]):
                    i0, i1 = int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1))
                    if t[i0] != t0 or t[i1] != t1 or not i1 > i0:
                        problems.append(f"{sym}: event {e} timestamps do not sit on bars")
                        continue
                    s[i0:i1] += side * sz[v][int(e)]
                    k[i0:i1] += 1
                avg = np.where(k > 0, s / np.maximum(k, 1), 0.0)
                day = t // DAY
                prev = 0.0
                for i in range(n):
                    if cost_side and avg[i] != prev:
                        days.setdefault(int(day[i]), [0.0, 0])[0] -= cost_side * abs(avg[i] - prev)
                    prev = avg[i]
                for i in range(n - 1):
                    if k[i] > 0:
                        d1 = int(day[i + 1])
                        days.setdefault(d1, [0.0, 0])[0] += avg[i] * (c[i + 1] / c[i] - 1.0)
                act = set()
                for i in range(n - 1):
                    if k[i] > 0:
                        act.add(int(day[i]))
                        act.add(int(day[i + 1]))
                for d in act:
                    days.setdefault(d, [0.0, 0])[1] += 1
            res[v] = days
        return res

    def stats(daily, lo, hi, n_ref):
        eq, peak, mdd, rets = 1.0, 1.0, 0.0, []
        for d in range(lo, hi + 1):
            net, n = daily.get(d, [0.0, 0])
            if n == 0:
                r = 0.0
            elif n_ref is None:
                r = net / n
            else:
                r = net / max(n, n_ref)
            rets.append(r)
            eq *= 1.0 + r
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1.0)
        m = sum(rets) / len(rets)
        sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
        return {"gain_pct": 100.0 * (eq - 1.0), "max_drawdown_pct": 100.0 * mdd, "daily_sharpe_x_sqrt365": m / sd * math.sqrt(365) if sd > 0 else float("nan")}

    net = book(cost_frac / 2.0)
    gross = book(0.0)
    S = pd.read_csv(out / "summary.csv")
    all_days = sorted(net["primary"])
    first_day, last_day = min(all_days), max(all_days)
    # n_ref: median of the number of active instruments over active days (primary; n is the same for every variant)
    nref = int(np.median([v[1] for v in net["primary"].values() if v[1] > 0]))
    if nref != meta["n_ref_slots"]:
        problems.append(f"n_ref {nref} vs {meta['n_ref_slots']}")
    # day range: the backtest's own range is the first/last day with an active instrument; compare
    for nm, d in (("first_day", first_day), ("last_day", last_day)):
        if datetime.fromtimestamp(d * DAY, timezone.utc).strftime("%Y-%m-%d") != meta[nm]:
            problems.append(f"{nm} differs")
    spans = {"last_365_days": (last_day - 364, last_day), "whole_walk_forward_span": (first_day, last_day)}
    checked = 0
    for sname, (lo, hi) in spans.items():
        for conv, nr in (("slot", nref), ("per_active", None)):
            for cl, src_d in (("net", net), ("before_costs", gross)):
                for v in ("primary", "meta", "placebo"):
                    mine = stats(src_d[v], lo, hi, nr)
                    row = S[(S["span"] == sname) & (S["convention"] == conv) & (S["costs"] == cl) & (S["variant"] == v)]
                    if len(row) != 1:
                        problems.append(f"missing row {sname}/{conv}/{cl}/{v}")
                        continue
                    for key, val in mine.items():
                        theirs = float(row[key].iloc[0])
                        if abs(val - theirs) > 1e-9 * max(1.0, abs(val)) and not (np.isnan(val) and np.isnan(theirs)):
                            problems.append(f"{sname}/{conv}/{cl}/{v}/{key}: independent {val!r} vs backtest {theirs!r}")
                        checked += 1
    # benchmark
    evw = pd.read_csv(src / "events.csv", usecols=["symbol", "t0", "t1"])
    max_gap = 1 if meta["venue"] == "kraken" else 5
    for sname, (lo, hi) in spans.items():
        syms = sorted(evw[(evw["t1"] // DAY >= lo) & (evw["t0"] // DAY <= hi)]["symbol"].unique())
        last_close = {}
        for sym in syms:
            t, c = bars[sym]
            d = t // DAY
            for i in range(len(t)):
                last_close.setdefault(int(d[i]), {})[sym] = c[i]       # later bar of the same day overwrites: last close of the UTC day
        days = sorted(last_close)
        keep = [d for d in days if len(last_close[d]) / len(syms) >= .8]
        rets = {}
        for a, b in zip(keep[:-1], keep[1:]):
            if b - a <= max_gap:
                xs = [last_close[b][s] / last_close[a][s] - 1.0 for s in syms if s in last_close[b] and s in last_close[a]]
                if xs:
                    rets[b] = sum(xs) / len(xs)
        eq, peak, mdd, r_all = 1.0, 1.0, 0.0, []
        for d in range(lo, hi + 1):
            r = rets.get(d, 0.0)
            r_all.append(r)
            eq *= 1.0 + r
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1.0)
        row = S[(S["span"] == sname) & (S["variant"] == "equal_weight_buy_and_hold")]
        for key, val in (("gain_pct", 100 * (eq - 1)), ("max_drawdown_pct", 100 * mdd)):
            theirs = float(row[key].iloc[0])
            if abs(val - theirs) > 1e-9 * max(1.0, abs(val)):
                problems.append(f"benchmark {sname}/{key}: independent {val!r} vs backtest {theirs!r}")
            checked += 1
    msg = f"{meta['venue']}: {checked} numbers re-derived independently; {len(problems)} problems"
    print(msg)
    for p in problems[:30]:
        print("  PROBLEM:", p)
    (out / "checker_output.txt").write_text(msg + "\n" + "\n".join(problems) + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
