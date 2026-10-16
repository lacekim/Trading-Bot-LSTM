"""Builds RESULTS.md for the book_full runs of both venues.  Every number is read from a saved file: <run>/summary.csv, meta.json, hrp_weights.csv, the walk-forward predictions (bet-size distribution, written to
<run>/size_distribution.json by this script with the same Snippet 10.1 / 10.3 arithmetic) and the earlier OTR run's summary.csv (buy-and-hold benchmark).
Usage: python book_full_report.py <kraken run> <alpaca run> <kraken otr run> <alpaca otr run> <out RESULTS.md>"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def size_dist(run: Path):
    meta = json.loads((run / "meta.json").read_text())
    w = pd.read_csv(Path(meta["bookfeat_run"]) / "walk_forward_predictions.csv")
    p = np.maximum(w["p"].to_numpy(float), 1 - w["p"].to_numpy(float))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    z = (p - .5) / np.sqrt(p * (1 - p))
    s = np.array([math.erf(x / math.sqrt(2)) for x in z])                 # 2*Phi(z)-1
    s = np.clip(np.round(s / meta["step"]) * meta["step"], 0, 1)
    d = {"n_predictions": int(len(w)), "share_size_zero": float((s == 0).mean()), "mean_size": float(s.mean()), "max_size": float(s.max()), "p_median": float(np.median(p)), "p_95th": float(np.percentile(p, 95))}
    (run / "size_distribution.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    return d


def main() -> int:
    runs = {"kraken": Path(sys.argv[1]), "alpaca": Path(sys.argv[2])}
    otr = {"kraken": Path(sys.argv[3]), "alpaca": Path(sys.argv[4])}
    L = ["# The whole book as one strategy: one-year backtest on both venues", "",
         "What trades, in the book's own order: dollar bars and the CUSUM filter pick the decision points (Ch.2); labels are built with no side given and the model learns side and size (Snippets 3.5-3.7, side=None); "
         "sample weights from uniqueness and attribution, sequential bootstrap (Ch.4); features: FFD, SADF, entropy (plug-in, Kontoyiannis), Corwin-Schultz spread, daily volatility (Ch.5, 17-19, Snippet 3.1); "
         "bagged entropy forest (Ch.6) scored by combinatorial purged CV with 5 backtest paths and by walk-forward trained before 2024-01-01 only (Ch.7, 12); side = predicted class and size = Snippets 10.1/10.3 of its probability, "
         "overlapping bets averaged (10.2); every bet is closed by the Chapter 13 optimal trading rule fitted on pre-2024 data only (Snippets 13.1/13.2); the account is allocated across instruments by Hierarchical Risk Parity "
         "(Snippet 16.4) re-estimated every calendar quarter from the trailing 252 trading days of returns (nothing after the rebalance day is used).", "",
         "Account: an instrument's slice = its HRP weight (weights sum to 1, no leverage); the position inside the slice is the averaged signal in [-1, 1]; daily return = sum of weight x (gross - cost); costs on every position change "
         "(12 bp per side Kraken, 10 bp per side Alpaca); compounded daily; instruments without enough history sit in cash. 'equal' below is the same account with 1/N weights over the same instruments (isolates the HRP step).", ""]
    for v, run in runs.items():
        meta = json.loads((run / "meta.json").read_text())
        S = pd.read_csv(run / "summary.csv")
        sd = size_dist(run)
        bh = pd.read_csv(otr[v] / "summary.csv")
        bh = bh[(bh["variant"] == "equal_weight_buy_and_hold") & (bh["span"] == "last_365_days")].iloc[0]
        wf = S[(S["scope"] == "wf") & (S["span"] == "last_365_days") & (S["costs"] == "net")]
        first = wf["start_day"].iloc[0]
        last = wf["end_day"].iloc[0]
        L += [f"## {v}", "",
              f"Last 365 days: {first} to {last}. Events {meta['events']:,}; walk-forward AUC {meta['walk_forward']['auc']:.4f}, side accuracy {meta['walk_forward']['side_accuracy']:.4f} "
              f"(share of test events that went up {meta['walk_forward']['up_share_test']:.4f}); CPCV AUC by path {', '.join(f'{x:.4f}' for x in meta['cpcv']['auc_by_path'])}. "
              f"OTR: profit-taking {meta['otr']['pt']:g} sigma, stop-loss {meta['otr']['sl']:g} sigma, at most {meta['otr']['K']} bars. HRP rebalances: {meta['hrp_rebalances']}.", "",
              "| variant (net of costs) | HRP gain % | HRP max drawdown % | HRP Sharpe | equal-weight gain % | equal-weight max drawdown % |", "|---|---:|---:|---:|---:|---:|"]
        for var in ("book_bet", "pred_side_full_size", "placebo_sizes", "always_long", "random_side"):
            h = wf[(wf["variant"] == var) & (wf["scheme"] == "hrp")].iloc[0]
            e = wf[(wf["variant"] == var) & (wf["scheme"] == "equal")].iloc[0]
            L.append(f"| {var} | {h['gain_pct']:.2f} | {h['max_drawdown_pct']:.2f} | {h['daily_sharpe_x_sqrt365']:.2f} | {e['gain_pct']:.2f} | {e['max_drawdown_pct']:.2f} |")
        L += ["", f"Equal-weight buy and hold of the same instruments over the same year (earlier OTR run): {bh['gain_pct']:.2f} % gain, {bh['max_drawdown_pct']:.2f} % max drawdown.", ""]
        c = S[(S["scope"] != "wf") & (S["variant"] == "book_bet") & (S["costs"] == "net") & (S["span"] == "last_365_days")]
        L += [f"CPCV, five independent backtest paths, same year, book_bet, HRP, net: gains {', '.join(f'{x:.2f}' for x in c['gain_pct'])} % (min {c['gain_pct'].min():.2f}, max {c['gain_pct'].max():.2f}).", "",
              f"Bet sizes: of {sd['n_predictions']:,} walk-forward predictions, {100 * sd['share_size_zero']:.1f} % get size 0 from the book's bet-size function (median confidence {sd['p_median']:.4f}, 95th percentile {sd['p_95th']:.4f}, largest size {sd['max_size']:.1f}); mean size {sd['mean_size']:.4f}.", ""]
        wfull = S[(S["scope"] == "wf") & (S["span"] == "whole_walk_forward_span") & (S["costs"] == "net") & (S["scheme"] == "hrp")]
        b = wfull[wfull["variant"] == "book_bet"].iloc[0]
        L += [f"Whole walk-forward span {b['start_day']} to {b['end_day']}, book_bet, HRP, net: {b['gain_pct']:.2f} % gain, {b['max_drawdown_pct']:.2f} % max drawdown.", ""]
        w = pd.read_csv(run / "hrp_weights.csv", index_col=0).iloc[-1].dropna().sort_values(ascending=False)
        L += [f"Latest HRP weights (rebalance {pd.read_csv(run / 'hrp_weights.csv', index_col=0).index[-1]}): " + ", ".join(f"{k} {x:.3f}" for k, x in w.head(5).items()) + f"; {len(w)} instruments weighted.", ""]
    L += ["## Book-compliance ledger", "",
          "Choices the book leaves to the user (fixed earlier in this project, not from the book): dollar-bar size (one average clock-hour of traded value, trailing 30 days); Alpaca rows are hourly (the finest data downloaded); the CUSUM threshold; "
          "triple-barrier width 2 x daily vol and 4-day vertical barrier for the labels; SADF window 100 / lag 1, entropy window 100 / word 2, FFD threshold 0.01; forest of 100 entropy trees, min weight fraction leaf 0.05, at most 10,000 draws per tree; "
          "bet-size step 0.1; walk-forward cutoff 2024-01-01; flat round-trip costs; the HRP re-estimation window (252 trading days) and quarterly cadence; instruments need 80 % coverage in the window to get a weight.", "",
          "Book code used as written: Snippet 16.4's functions as ported (note: the book's code passes the square correlation-distance matrix to scipy's linkage, which treats its rows as observations; scipy warns about it; the checker does the same on purpose).", "",
          "Book components NOT applied: Snippet 10.4 dynamic position size / limit price; Chapter 15 strategy-risk sizing; Chapter 19 trade-level measures; the deflated Sharpe; exit calibration on labels built from the OTR barriers (labels use the 2 x vol barriers, the exit uses the OTR).", "",
          "References in the tables are not part of the strategy: always_long (every event long, same exits and allocation), random_side, placebo_sizes (the book_bet sizes shuffled), pred_side_full_size (same sides, size 1).", ""]
    Path(sys.argv[5]).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
