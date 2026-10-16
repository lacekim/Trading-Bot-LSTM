"""The book's loss-limiting / profit-taking rule (Chapter 13, optimal trading rule on an Ornstein-Uhlenbeck process) applied as the EXIT of the book's side-free trades, then a one-year backtest.
New file; nothing audited is edited; no strategy from outside the book.

What is combined (all from the book):
  * entries: the CUSUM events of the venue's dollar bars (Ch.2, Snippet 2.4);
  * side + size: the side-free model of book_side_free_bookfeat.py (book-only features, walk-forward, every prediction out of sample); side = predicted class, size = Snippets 10.1 / 10.3;
  * exit: the optimal trading rule (profit-taking pt, stop-loss sl, in units of the event's bar volatility, maximum holding K bars) that the venue's Chapter 13 run chose from the
    pre-cutoff events only: Step 1 (Equations 13.5-13.7: phi, drift, sigma by regression on the side-signed paths of the events that END before 2024-01-01), Steps 2-5 by the book's loop
    (Snippet 13.2).  The numbers are read from the saved ports run (summary_numbers.json -> results.ch13): Kraken pt 7 / sl 7, Alpaca pt 9 / sl 3, K = 100.
    Exit at the first bar whose side-signed log return from the entry close, divided by the event's sigma, is >= pt (profit) or <= -sl (stop), else after K bars; the exit price is that bar's close.
    A path that reaches the end of its data segment (end of the series or a data hole) before K bars is closed at the last bar of the segment (counted).
  * overlapping bets are averaged (Snippet 10.2) and the account is the one of backtest_one_year.py (N_ref slots, no leverage, costs on every position change, last 365 UTC days).
Declared difference: the model was trained on the 2 x daily-vol triple-barrier labels, not on labels from the OTR barriers (the OTR is chosen after the labels exist and its pt / sl are not symmetric on Alpaca).
Usage: python book_otr_exit.py --source <meta run> --bars <dollar bars> --run <book_side_free_bookfeat output> --ports <book_ports run> --out <out folder>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import backtest_one_year as BO  # noqa: E402
import meta_model as MM  # noqa: E402
import ports_common as PCm  # noqa: E402
import run_bet_averaging as RBA  # noqa: E402

DAY = 86400


def otr_exits(bars, ev, pos_ev, side, pt, sl, K):
    """New t1pos for the events (rows of ev) listed in pos_ev with the given sides; returns (t1pos array, truncated flag array)."""
    out = np.zeros(len(pos_ev), np.int64)
    trunc = np.zeros(len(pos_ev), bool)
    sym = ev["symbol"].to_numpy()[pos_ev]
    pos = ev["pos"].to_numpy(np.int64)[pos_ev]
    sg = ev["sigma"].to_numpy(float)[pos_ev]
    for s in np.unique(sym):
        m = np.flatnonzero(sym == s)
        b = bars[s]
        lc = np.log(b.close)
        for i in m:
            p0 = pos[i]
            end = p0
            # last bar of the same segment
            hi = min(p0 + K, b.n - 1)
            seg_ok = b.seg[p0 + 1:hi + 1] == b.seg[p0] if hi > p0 else np.zeros(0, bool)
            nvalid = int(seg_ok.argmin()) if (len(seg_ok) and not seg_ok.all()) else len(seg_ok)
            if nvalid == 0:
                out[i], trunc[i] = p0 + 1 if p0 + 1 < b.n else p0, True      # no complete bar after the entry: never happens for kept events; guarded
                continue
            q = side[i] * (lc[p0 + 1:p0 + 1 + nvalid] - lc[p0]) / sg[i]
            hit = np.flatnonzero((q >= pt) | (q <= -sl))
            if len(hit):
                out[i] = p0 + 1 + int(hit[0])
            else:
                out[i] = p0 + nvalid
                trunc[i] = nvalid < K
    return out, trunc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--bars", required=True)
    ap.add_argument("--run", required=True, help="book_side_free_bookfeat output folder (walk_forward_predictions.csv, events_free_index.csv, meta.json)")
    ap.add_argument("--ports", required=True, help="book_ports run folder with summary_numbers.json (results.ch13)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_start = time.time()
    src, run, ports = Path(a.source).resolve(), Path(a.run).resolve(), Path(a.ports).resolve()
    summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    venue = summ["meta"]["venue"]
    p = summ["params"]
    step, seed, cost_frac = float(p["step"]), int(p["seed"]), float(p["cost_frac"])
    c13 = json.loads((ports / "summary_numbers.json").read_text(encoding="utf-8"))["results"]["ch13"]
    pt, sl, K = float(c13["mesh_choice"]["chosen_pt"]), float(c13["mesh_choice"]["chosen_sl"]), int(c13["step1"]["K"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    V = PCm.Venue(venue, src, Path(a.bars), 40)
    idxf = pd.read_csv(run / "events_free_index.csv")
    ev = V.events.iloc[idxf["src_row"].to_numpy()].reset_index(drop=True)          # the events of the side-free run, same row order
    ev["side"] = 1.0
    wf = pd.read_csv(run / "walk_forward_predictions.csv")
    bars = V.bars(sorted(ev["symbol"].unique()))
    closes = {s: pd.Series(b.close, index=b.idx) for s, b in bars.items()}
    idx = wf["event"].to_numpy(np.int64)
    pp = wf["p"].to_numpy(float)
    side_pred = np.where(pp > 0.5, 1.0, -1.0)
    size = MM.bet_size(np.maximum(pp, 1.0 - pp), step)
    plc = np.random.default_rng(seed + 7 + 99).permutation(size)
    rs = np.random.default_rng(seed + 11).choice([-1.0, 1.0], size=len(idx))
    ones = np.ones(len(idx))
    log = lambda m: print(f"[{time.time() - t_start:7.0f}s] {m}", flush=True)  # noqa: E731
    log(f"{venue}: OTR pt {pt:g} sl {sl:g} K {K}; {len(idx):,} walk-forward events")
    # exits per variant side (the exit depends on the side of the bet)
    t1_model, tr_model = otr_exits(bars, ev, idx, side_pred, pt, sl, K)
    t1_rand, _ = otr_exits(bars, ev, idx, rs, pt, sl, K)
    t1_long, _ = otr_exits(bars, ev, idx, ones, pt, sl, K)
    log(f"exits found; truncated at a segment end/data end: {int(tr_model.sum())} of {len(idx)} ({tr_model.mean():.4f})")
    stats_exit = {"mean_bars_held_model": float((t1_model - ev["pos"].to_numpy()[idx]).mean()), "median_bars_held_model": float(np.median(t1_model - ev["pos"].to_numpy()[idx])),
                  "truncated_share": float(tr_model.mean())}
    windows = {k: (int(v[0]), int(v[1])) for k, v in summ["meta"]["window_epoch"].items()}
    cost_side = cost_frac / 2.0

    def mk(side_arr, t1new):
        e = ev.copy()
        e.loc[idx, "side"] = side_arr
        e.loc[idx, "t1pos"] = t1new
        return e

    ev_model, ev_rand, ev_long = mk(side_pred, t1_model), mk(rs, t1_rand), mk(ones, t1_long)
    res = {}
    for label, cs in (("net", cost_side), ("before_costs", 0.0)):
        parts = []
        for name, e, sz, pl, v in (("book_bet_otr", ev_model, size, plc, "meta"), ("pred_side_full_size_otr", ev_model, size, plc, "primary"), ("placebo_sizes_otr", ev_model, size, plc, "placebo"),
                                   ("random_side_otr", ev_rand, ones, ones, "primary"), ("always_long_otr", ev_long, ones, ones, "primary")):
            key = (id(e), label)
            if key not in res:
                res[key] = pd.DataFrame(RBA.evaluate_scope(e, closes, "wf", [(0, idx, sz, pl)], cs, cost_frac, windows)[1])
            x = res[key]
            x = x[x["variant"] == v].copy()
            x["variant"] = name
            parts.append(x)
        res[label] = pd.concat(parts, ignore_index=True)
        log(f"averaged bets evaluated ({label})")
    names = ("book_bet_otr", "pred_side_full_size_otr", "placebo_sizes_otr", "random_side_otr", "always_long_otr")
    dn = res["net"]
    last_day, first_day = int(dn["day"].max()), int(dn["day"].min())
    n_ref = int(np.median(dn[dn["variant"] == "book_bet_otr"]["n_active"].to_numpy()))
    spans = {"last_365_days": (last_day - 364, last_day), "whole_walk_forward_span": (first_day, last_day)}
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
            for cl in ("net", "before_costs"):
                d = res[cl]
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
    pd.DataFrame({"event": idx, "side_pred": side_pred, "size": size, "t1pos_otr": t1_model}).to_csv(out / "otr_exits_model.csv", index=False, float_format="%.12g")
    meta = {"venue": venue, "source_run": str(src), "bar_dir": str(a.bars), "side_free_run": str(run), "ports_run": str(ports), "seed": seed, "step": step, "cost_frac_round_trip": cost_frac,
            "cost_bp_per_side": cost_side * 1e4, "n_ref_slots": n_ref, "otr": {"pt": pt, "sl": sl, "K": K, "phi": c13["step1"]["phi"], "half_life_bars": c13["step1"]["half_life_bars"],
            "simulated_sharpe_at_chosen": c13["mesh_choice"]["simulated_sharpe_at_chosen"], "realised_sharpe_at_chosen_test": c13["mesh_choice"]["realised_sharpe_at_chosen_test"]},
            "exit_stats": stats_exit, "events": int(len(ev)), "walk_forward_rows": int(len(wf)), "first_day": pd.Timestamp(first_day * DAY, unit="s").strftime("%Y-%m-%d"),
            "last_day": pd.Timestamp(last_day * DAY, unit="s").strftime("%Y-%m-%d"), "cutoff": p["cutoff"], "seconds": round(time.time() - t_start),
            "script_md5": {n: BO.md5_file(HERE / n) for n in ("book_otr_exit.py", "backtest_one_year.py", "meta_model.py", "bet_averaging.py", "run_bet_averaging.py")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    L = [f"# Book OTR exit + side-free book bet, {venue}: one-year backtest ({meta['first_day']} .. {meta['last_day']} out-of-sample)", "",
         f"OTR from Chapter 13 (pre-cutoff events only): profit-taking {pt:g} sigma, stop-loss {sl:g} sigma, at most {K} bars; phi {c13['step1']['phi']:.4f}. N_ref = {n_ref} slots; costs {meta['cost_bp_per_side']:.0f} bp per side.", ""]
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
