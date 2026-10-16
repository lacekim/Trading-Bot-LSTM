"""Independent re-derivation of book_otr_exit.py (side-free book bets exited by the Chapter 13 optimal trading rule: profit-taking pt, stop-loss sl in units of the event sigma, at most K bars) from the raw files (events.csv, walk_forward_predictions.csv, bar csv files).
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


def load_bars(bar_dir: Path, sym: str, break_s: int):
    b = pd.read_csv(bar_dir / "bars" / f"{sym}.csv", keep_default_na=False, na_values=[""], usecols=["date", "start", "close"])
    b = b.dropna(subset=["close"])
    b = b[b["close"] > 0]
    t, st = epoch(b["date"]), epoch(b["start"])
    c = b["close"].to_numpy(float)
    o = np.argsort(t, kind="stable")
    t, st, c = t[o], st[o], c[o]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, st, c = t[keep], st[keep], c[keep]
    seg = np.concatenate([[0], np.cumsum((st[1:] - t[:-1]) > break_s)]).astype(np.int64)
    return t, c, seg


def main(out: Path) -> int:
    meta = json.loads((out / "meta.json").read_text())
    src, bar_dir = Path(meta["source_run"]), Path(meta["bar_dir"])
    step, seed, cost_frac = meta["step"], meta["seed"], meta["cost_frac_round_trip"]
    sf = Path(meta["side_free_run"])
    sfm = json.loads((sf / "meta.json").read_text())
    ev0 = pd.read_csv(src / "events.csv", usecols=["symbol", "t0", "t1", "sigma"])
    wf = pd.read_csv(sf / "walk_forward_predictions.csv")
    idx_file = pd.read_csv(sf / "events_free_index.csv")
    problems = []
    break_s = int(json.loads((bar_dir / "manifest.json").read_text())["break_hours"]) * 3600
    bars = {sym: load_bars(bar_dir, sym, break_s) for sym in sorted(ev0["symbol"].unique())}
    pt, sl, K = float(meta["otr"]["pt"]), float(meta["otr"]["sl"]), int(meta["otr"]["K"])
    # events kept = those whose long price return (from the bars, located by timestamp) is not exactly 0
    keep, rows = [], []
    for k, (sym, t0, t1) in enumerate(zip(ev0["symbol"], ev0["t0"], ev0["t1"])):
        t, c, _sg = bars[sym]
        i0, i1 = int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1))
        if t[i0] != t0 or t[i1] != t1 or not i1 > i0:
            problems.append(f"{sym}: event {k} timestamps do not sit on bars")
            keep.append(False)
            continue
        r = c[i1] / c[i0] - 1.0
        keep.append(r != 0.0)
        rows.append(r)
    keep = np.array(keep)
    ev = ev0[keep].reset_index(drop=True)
    if len(ev) != len(idx_file) or not (idx_file["src_row"].to_numpy() == np.flatnonzero(keep)).all():
        problems.append("the kept-event set differs from events_free_index.csv")
    if len(ev) != meta["events"] or len(wf) != meta["walk_forward_rows"]:
        problems.append(f"event count {len(ev)} vs {meta['events']}")
    # up/down label and side-accuracy
    ret_long = np.array([bars[s][1][np.searchsorted(bars[s][0], t1)] / bars[s][1][np.searchsorted(bars[s][0], t0)] - 1.0 for s, t0, t1 in zip(ev["symbol"], ev["t0"], ev["t1"])])
    y = (ret_long > 0).astype(int)
    pidx = wf["event"].to_numpy(np.int64)
    pp = wf["p"].to_numpy(float)
    acc = float((((pp > 0.5).astype(int)) == y[pidx]).mean())
    if abs(acc - sfm["walk_forward"]["side_accuracy"]) > 1e-12:
        problems.append(f"side accuracy {acc} vs {meta['walk_forward']['side_accuracy']}")
    if abs(float(y[pidx].mean()) - sfm["walk_forward"]["up_share_test"]) > 1e-12:
        problems.append("up share of test events differs")
    # the model never saw a test label: every walk-forward event starts at or after the cutoff
    cut = int(pd.Timestamp(meta["cutoff"]).timestamp())
    if not (ev["t0"].to_numpy()[pidx] >= cut).all():
        problems.append("a walk-forward prediction is for an event that starts before the cutoff")
    side_pred = np.where(pp > 0.5, 1.0, -1.0)
    p_pred = np.maximum(pp, 1.0 - pp)
    size = np.array([bet_size(float(q), step) for q in p_pred])
    plc = np.random.default_rng(seed + 7 + 99).permutation(size)
    rs = np.random.default_rng(seed + 11).choice([-1.0, 1.0], size=len(pidx))
    ones = np.ones(len(pidx))

    def otr_exits(sides):
        """event row -> (entry bar, exit bar, truncated): first bar with side * (log c[k] - log c[entry]) / sigma >= pt or <= -sl within K bars of the same segment, else the last bar of the window."""
        res_ = {}
        for e, sd in zip(pidx, sides):
            sym = ev.at[int(e), "symbol"]
            t, c, sg_ = bars[sym]
            i0 = int(np.searchsorted(t, ev.at[int(e), "t0"]))
            sgm = float(ev.at[int(e), "sigma"])
            last = i0
            hit = None
            for k_ in range(1, K + 1):
                j = i0 + k_
                if j >= len(t) or sg_[j] != sg_[i0]:
                    break
                last = j
                q = sd * (math.log(c[j]) - math.log(c[i0])) / sgm
                if q >= pt or q <= -sl:
                    hit = j
                    break
            res_[int(e)] = (i0, hit if hit is not None else last, hit is None and last - i0 < K)
        return res_

    ex = {"model": otr_exits(side_pred), "rand": otr_exits(rs), "long": otr_exits(ones)}
    held = np.array([ex["model"][int(e)][1] - ex["model"][int(e)][0] for e in pidx])
    trunc_share = float(np.mean([ex["model"][int(e)][2] for e in pidx]))
    if abs(float(held.mean()) - meta["exit_stats"]["mean_bars_held_model"]) > 1e-9 or abs(float(np.median(held)) - meta["exit_stats"]["median_bars_held_model"]) > 1e-9 or abs(trunc_share - meta["exit_stats"]["truncated_share"]) > 1e-12:
        problems.append("exit statistics (bars held, truncated share) differ")
    saved = pd.read_csv(out / "otr_exits_model.csv")
    my_t1 = np.array([ex["model"][int(e)][1] for e in pidx])
    # saved t1pos are bar numbers of the runner's bar series, mine are positions in my own load of the same csv; they must be equal
    if not (saved["t1pos_otr"].to_numpy() == my_t1).all():
        problems.append("the OTR exit bars differ from otr_exits_model.csv")
    variants = {"book_bet_otr": (side_pred, size, "model"), "pred_side_full_size_otr": (side_pred, ones, "model"), "placebo_sizes_otr": (side_pred, plc, "model"), "random_side_otr": (rs, ones, "rand"), "always_long_otr": (ones, ones, "long")}

    def book(cost_side):
        res = {}
        for v, (sd, sz_, exk) in variants.items():
            sig = {int(e): float(a * b) for e, a, b in zip(pidx, sd, sz_)}
            days = {}
            for sym, g in ev.loc[pidx].groupby("symbol"):
                t, c, _sg = bars[sym]
                n = len(t)
                s_ = np.zeros(n)
                k = np.zeros(n, dtype=np.int64)
                for e in g.index:
                    i0, i1, _tr = ex[exk][int(e)]
                    s_[i0:i1] += sig[int(e)]
                    k[i0:i1] += 1
                avg = np.where(k > 0, s_ / np.maximum(k, 1), 0.0)
                day = t // DAY
                prev = 0.0
                for i in range(n):
                    if cost_side and avg[i] != prev:
                        days.setdefault(int(day[i]), [0.0, 0])[0] -= cost_side * abs(avg[i] - prev)
                    prev = avg[i]
                for i in range(n - 1):
                    if k[i] > 0:
                        days.setdefault(int(day[i + 1]), [0.0, 0])[0] += avg[i] * (c[i + 1] / c[i] - 1.0)
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
            r = 0.0 if n == 0 else (net / n if n_ref is None else net / max(n, n_ref))
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
    all_days = sorted(net["book_bet_otr"])
    first_day, last_day = min(all_days), max(all_days)
    nref = int(np.median([v[1] for v in net["book_bet_otr"].values() if v[1] > 0]))
    if nref != meta["n_ref_slots"]:
        problems.append(f"n_ref {nref} vs {meta['n_ref_slots']}")
    for nm, d in (("first_day", first_day), ("last_day", last_day)):
        if datetime.fromtimestamp(d * DAY, timezone.utc).strftime("%Y-%m-%d") != meta[nm]:
            problems.append(f"{nm} differs")
    spans = {"last_365_days": (last_day - 364, last_day), "whole_walk_forward_span": (first_day, last_day)}
    checked = 0
    for sname, (lo, hi) in spans.items():
        for conv, nr in (("slot", nref), ("per_active", None)):
            for cl, src_d in (("net", net), ("before_costs", gross)):
                for v in variants:
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
    evw = ev
    max_gap = 1 if meta["venue"] == "kraken" else 5
    for sname, (lo, hi) in spans.items():
        syms = sorted(evw[(evw["t1"] // DAY >= lo) & (evw["t0"] // DAY <= hi)]["symbol"].unique())
        last_close = {}
        for sym in syms:
            t, c, _sg = bars[sym]
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
