"""The book's SIDE-FREE path, end to end, on one venue's dollar bars, then a one-year percentage-gain backtest.  New file; nothing audited is edited.  No strategy from outside the book.

Book source (the extract's own text):
  Snippet 3.5 / 3.6 with side=None: the events carry NO side; the barriers are symmetric (ptSl[0] for both) and Snippet 3.7 case 1 labels by price action, bin = sign(return at the first
  touch / vertical barrier), i.e. in (-1, 1): "learning side and size".  Snippet 10.1 (multi-class, here two classes): the predicted class is the side, its probability gives the size,
  signal = side_predicted * (2 * Phi((p - 1/2) / sqrt(p (1 - p))) - 1); Snippet 10.3 discretises the size (step 0.1); bets of overlapping events are averaged (Snippet 10.2).
Construction here:
  events        the venue's CUSUM events and triple-barrier exits from the finished source run (events.csv; barriers 2 x daily vol, 4-day vertical).  The barriers are symmetric, so the exit
                time does not depend on the side: the long price return is ret_side * side; events whose long return is exactly 0 are dropped (counted).
  label         bin = 1 when the price return at the exit is > 0, else 0 (-1 in the book's notation).
  features      the 13 features of the earlier pipeline that do not contain a side (the 'side' column, the trailing-720-bar sign, is NOT given to the model).
  weights       Chapter 4 average uniqueness / return attribution (already in events.csv); sequential bootstrap; bagged entropy forest (Ch.6) - the audited meta_model / meta_eval code.
  validation    WALK-FORWARD: trained only on events whose labels ended before 2024-01-01, predicts every later event (all predictions out of sample).
  bet           side = + when p(up) > 1/2 else -; p_pred = max(p, 1 - p); size = Snippet 10.1 / 10.3 of p_pred; signal = side * size; overlapping bets averaged per bar (bet_averaging.py).
  variants      book_bet (side x confidence size), pred_side_full_size (same side, size 1), placebo_sizes (the book_bet sizes shuffled among the same events), random_side (seeded coin per
                event, size 1) and always_long (side +, size 1) as references.
  account       the one-year compounding of backtest_one_year.py (N_ref slots, no leverage, costs on every position change, inactive days in cash).
Usage: python book_side_free.py --source <meta run folder> --bars <dollar bar folder> --out <out folder> [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import backtest_one_year as BO  # noqa: E402
import meta_eval as ME  # noqa: E402
import meta_events as E  # noqa: E402
import meta_model as MM  # noqa: E402
import ports_common as PCm  # noqa: E402
import run_bet_averaging as RBA  # noqa: E402

DAY = 86400
FEATURES_FREE = [f for f in E.FEATURES if f != "side"]


def daily_rows(ev, closes, idx, size, plc, cost_side, cost_frac, windows):
    _, drow, _ = RBA.evaluate_scope(ev, closes, "wf", [(0, idx, size, plc)], cost_side, cost_frac, windows)
    return pd.DataFrame(drow)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--bars", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="20 trees (smoke test only)")
    a = ap.parse_args()
    t_start = time.time()
    src = Path(a.source).resolve()
    summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    venue = summ["meta"]["venue"]
    bd = Path(a.bars) if a.bars else Path(summ["meta"]["candle_dir"])
    p = summ["params"]
    step, seed, cost_frac, cutoff = float(p["step"]), int(p["seed"]), float(p["cost_frac"]), p["cutoff"]
    forest_kw = dict(p["forest"])
    if a.quick:
        forest_kw["n_estimators"] = 20
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    V = PCm.Venue(venue, src, bd, 40)
    ev0 = V.events.copy()
    ev0["src_row"] = np.arange(len(ev0))
    r_long = ev0["ret_side"].to_numpy(float) * ev0["side"].to_numpy(float)
    keep = r_long != 0.0
    ev = ev0[keep].reset_index(drop=True)
    ev["ret_side"] = r_long[keep]
    ev["side"] = 1.0
    ev["bin"] = (ev["ret_side"] > 0).astype(np.int8)
    allsyms = sorted(ev["symbol"].unique())
    bars = V.bars(allsyms)
    closes = {s: pd.Series(b.close, index=b.idx) for s, b in bars.items()}
    RBA.check_alignment(ev, closes)                                   # the long price return of every event equals the candle return (side set to +1 here)
    E.FEATURES = FEATURES_FREE                                        # in-process only: the model is not given a side
    log = lambda m: print(f"[{time.time() - t_start:7.0f}s] {m}", flush=True)  # noqa: E731
    log(f"{venue}: {len(ev):,} events ({int((~keep).sum())} dropped: long return exactly 0); up-share {ev['bin'].mean():.4f}; features {len(FEATURES_FREE)} (no side)")
    wpred, wmet = ME.run_walk_forward(ev, cutoff, forest_kw, seed + 500, log)
    y = ev["bin"].to_numpy()[wpred["event"].to_numpy()]
    pp = wpred["p"].to_numpy(float)
    wmet["side_accuracy"] = float((((pp > 0.5).astype(int)) == y).mean())
    wmet["share_p_eq_half"] = float((pp == 0.5).mean())
    wmet["up_share_test"] = float(y.mean())
    wmet["auc"] = float(roc_auc_score(y, pp))
    wpred.to_csv(out / "walk_forward_predictions.csv", index=False, float_format="%.12g")
    ev[["src_row", "symbol", "t0", "t1", "pos", "t1pos", "ret_side", "bin"]].to_csv(out / "events_free_index.csv", index=False, float_format="%.12g")
    log(f"walk-forward AUC {wmet['auc']:.4f}, side accuracy {wmet['side_accuracy']:.4f} (up-share of test events {wmet['up_share_test']:.4f})")

    idx = wpred["event"].to_numpy(np.int64)
    side_pred = np.where(pp > 0.5, 1.0, -1.0)
    p_pred = np.maximum(pp, 1.0 - pp)
    size = MM.bet_size(p_pred, step)
    plc = np.random.default_rng(seed + 7 + 99).permutation(size)
    rs = np.random.default_rng(seed + 11).choice([-1.0, 1.0], size=len(idx))
    ones = np.ones(len(idx))
    windows = {k: (int(v[0]), int(v[1])) for k, v in summ["meta"]["window_epoch"].items()}
    cost_side = cost_frac / 2.0
    variants_ev = {}
    ev_model = ev.copy()
    ev_model.loc[idx, "side"] = side_pred
    ev_rand = ev.copy()
    ev_rand.loc[idx, "side"] = rs
    ev_long = ev.copy()
    res = {}
    for label, cs in (("net", cost_side), ("before_costs", 0.0)):
        d1 = daily_rows(ev_model, closes, idx, size, plc, cs, cost_frac, windows)
        d2 = daily_rows(ev_rand, closes, idx, ones, ones, cs, cost_frac, windows)
        d3 = daily_rows(ev_long, closes, idx, ones, ones, cs, cost_frac, windows)
        parts = []
        for name, d, v in (("book_bet", d1, "meta"), ("pred_side_full_size", d1, "primary"), ("placebo_sizes", d1, "placebo"), ("random_side", d2, "primary"), ("always_long", d3, "primary")):
            x = d[d["variant"] == v].copy()
            x["variant"] = name
            parts.append(x)
        res[label] = pd.concat(parts, ignore_index=True)
        log(f"averaged bets evaluated ({label})")
    dn = res["net"]
    last_day, first_day = int(dn["day"].max()), int(dn["day"].min())
    n_ref = int(np.median(dn[dn["variant"] == "book_bet"]["n_active"].to_numpy()))
    spans = {"last_365_days": (last_day - 364, last_day), "whole_walk_forward_span": (first_day, last_day)}
    names = ("book_bet", "pred_side_full_size", "placebo_sizes", "random_side", "always_long")
    evd = pd.DataFrame({"s": ev["symbol"].to_numpy(), "d0": ev["t0"].to_numpy() // DAY, "d1": ev["t1"].to_numpy() // DAY})
    rows, curves, mrows = [], [], {}
    for sname, (lo, hi) in spans.items():
        days = np.arange(lo, hi + 1)
        syms = sorted(evd[(evd["d1"] >= lo) & (evd["d0"] <= hi)]["s"].unique())
        bm = BO.benchmark(bars, V.max_gap_days, lo, hi, syms)
        bc = pd.DataFrame({"day": days, "ret": bm})
        bc["equity"] = np.cumprod(1.0 + bc["ret"].to_numpy())
        bc["drawdown"] = bc["equity"] / np.maximum.accumulate(bc["equity"].to_numpy()) - 1.0
        for conv in ("slot", "per_active"):
            for cl, d in res.items():
                for v in names:
                    x = d[d["variant"] == v]
                    g, ns = x.set_index("day")["bp"], x.set_index("day")["n_active"]
                    c = BO.curve(g, ns, days, n_ref if conv == "slot" else None)
                    st = BO.stats(c)
                    nact = ns.reindex(days).fillna(0)
                    st.update({"span": sname, "convention": conv, "costs": cl, "variant": v, "start_day": pd.Timestamp(lo * DAY, unit="s").strftime("%Y-%m-%d"),
                               "end_day": pd.Timestamp(hi * DAY, unit="s").strftime("%Y-%m-%d"), "mean_active_instruments": float(nact[nact > 0].mean()), "active_days": int((nact > 0).sum())})
                    rows.append(st)
                    curves.append(c.assign(span=sname, convention=conv, costs=cl, variant=v))
                    if cl == "net" and conv == "slot":
                        mrows[(sname, v)] = BO.monthly(c)
        bs = BO.stats(bc)
        bs.update({"span": sname, "convention": "-", "costs": "none (buy and hold)", "variant": "equal_weight_buy_and_hold", "start_day": pd.Timestamp(lo * DAY, unit="s").strftime("%Y-%m-%d"),
                   "end_day": pd.Timestamp(hi * DAY, unit="s").strftime("%Y-%m-%d"), "instruments": len(syms)})
        rows.append(bs)
        curves.append(bc.assign(span=sname, convention="-", costs="none", variant="equal_weight_buy_and_hold"))
        mrows[(sname, "benchmark")] = BO.monthly(bc)
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False, float_format="%.10g")
    pd.concat(curves, ignore_index=True).to_csv(out / "equity_curves.csv", index=False, float_format="%.12g")
    pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in mrows.items()}).to_csv(out / "monthly_gain_pct.csv", float_format="%.10g")
    meta = {"venue": venue, "source_run": str(src), "bar_dir": str(bd), "seed": seed, "step": step, "cost_frac_round_trip": cost_frac, "cost_bp_per_side": cost_side * 1e4, "n_ref_slots": n_ref,
            "events": int(len(ev)), "events_dropped_zero_return": int((~keep).sum()), "features": FEATURES_FREE, "forest": forest_kw, "quick": bool(a.quick), "cutoff": cutoff,
            "walk_forward": wmet, "walk_forward_rows": int(len(wpred)), "first_day": pd.Timestamp(first_day * DAY, unit="s").strftime("%Y-%m-%d"),
            "last_day": pd.Timestamp(last_day * DAY, unit="s").strftime("%Y-%m-%d"), "seconds": round(time.time() - t_start),
            "script_md5": {n: BO.md5_file(HERE / n) for n in ("book_side_free.py", "backtest_one_year.py", "meta_model.py", "meta_eval.py", "bet_averaging.py", "run_bet_averaging.py", "meta_events.py")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    L = [f"# Book side-free path, {venue}: one-year backtest ({meta['first_day']} .. {meta['last_day']} out-of-sample)", "",
         f"Walk-forward AUC {wmet['auc']:.4f}; side accuracy {wmet['side_accuracy']:.4f} (share of test events that went up {wmet['up_share_test']:.4f}). N_ref = {n_ref} slots; "
         f"costs {meta['cost_bp_per_side']:.0f} bp per side. Trees {forest_kw['n_estimators']}{' (QUICK smoke run)' if a.quick else ''}.", ""]
    for sname in spans:
        x = S[(S["span"] == sname) & (S["convention"].isin(["slot", "-"]))]
        L += [f"## {sname}: {x['start_day'].iloc[0]} to {x['end_day'].iloc[0]}", "", "| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |", "|---|---|---:|---:|---:|"]
        for _, r in x.iterrows():
            L.append(f"| {r['variant']} | {r['costs']} | {r['gain_pct']:.2f} | {r['max_drawdown_pct']:.2f} | {r['daily_sharpe_x_sqrt365']:.2f} |")
        L.append("")
    (out / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
