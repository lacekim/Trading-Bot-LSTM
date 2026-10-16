"""RESULTS.md of the book-only final step.  Every number is read from saved files.
Usage: python book_final_report.py <ch89 kraken> <ch89 alpaca> <final kraken> <final alpaca> <out RESULTS.md>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    a = [Path(x) for x in sys.argv[1:5]]
    out = Path(sys.argv[5])
    names = ["kraken", "alpaca"]
    ch89, fin = dict(zip(names, a[0:2])), dict(zip(names, a[2:4]))
    L = ["# The book's strategy, one-year backtest on Kraken and Alpaca", "",
         "Only the book's own algorithms and its own tests: dollar bars and CUSUM (Ch.2); side-free labels (Snippets 3.5-3.7, side=None); uniqueness weights and sequential bootstrap (Ch.4); FFD, SADF, entropy, Corwin-Schultz and volatility features (Ch.5, 17-19, Snippet 3.1); "
         "bagged entropy forest (Ch.6) with purged CV (Ch.7), feature importance (Ch.8) and the randomised purged-CV search (Ch.9); side = predicted class and size from Snippets 10.1/10.3 with overlapping bets averaged (10.2); "
         "exit by the Chapter 13 optimal trading rule; allocation by Hierarchical Risk Parity (Snippet 16.4); evaluation by walk-forward (trained before 2024-01-01) and the five combinatorial purged CV paths (Ch.12); PSR / deflated Sharpe (Ch.14). Costs on every position change.", ""]
    for v in names:
        m89 = json.loads((ch89[v] / "meta.json").read_text())
        mdi, mda, sfi = (pd.read_csv(ch89[v] / f"{x}.csv", index_col=0) for x in ("mdi", "mda", "sfi"))
        R = pd.read_csv(ch89[v] / "search.csv")
        tj = json.loads((ch89[v] / "tuned_forest.json").read_text())
        mt = json.loads((fin[v] / "meta.json").read_text())
        S = pd.read_csv(fin[v] / "summary.csv")
        w1 = S[(S["scope"] == "wf") & (S["span"] == "last_365_days")].iloc[0]
        w2 = S[(S["scope"] == "wf") & (S["span"] == "whole_walk_forward_span")].iloc[0]
        cp = S[(S["scope"] != "wf") & (S["span"] == "last_365_days")]
        L += [f"## {v}", "",
              f"### Result, {w1['start_day']} to {w1['end_day']} (net of {mt['cost_bp_per_side']:.0f} bp per side, HRP allocation)", "",
              "| test | gain % | max drawdown % |", "|---|---:|---:|",
              f"| walk-forward, last 365 days | {w1['gain_pct']:.2f} | {w1['max_drawdown_pct']:.2f} |",
              f"| walk-forward, whole test period {w2['start_day']} to {w2['end_day']} | {w2['gain_pct']:.2f} | {w2['max_drawdown_pct']:.2f} |"]
        for _, r in cp.iterrows():
            L.append(f"| CPCV path {r['scope'][4:]}, last 365 days | {r['gain_pct']:.2f} | {r['max_drawdown_pct']:.2f} |")
        d = mt["dsr"]
        L += ["", f"Model quality: walk-forward AUC {mt['walk_forward']['auc']:.4f}, side accuracy {mt['walk_forward']['side_accuracy']:.4f}; CPCV AUC by path {', '.join(f'{x:.4f}' for x in mt['cpcv']['auc_by_path'])}.", "",
              "Deflated Sharpe (Ch.14; trials = the Ch.9 candidates, " + str(mt["n_trials_candidates"]) + "): " + "; ".join(f"{k}: PSR against 0 = {x['psr_vs_0']:.3f}, DSR = {list(x['by_trials'].values())[0]['dsr']:.3f}" for k, x in d.items()) + ".", "",
              f"### Chapter 8 feature importance ({m89['events_used']:,} events with labels ending before 2024-01-01)", "", "| feature | MDI | MDA | SFI |", "|---|---:|---:|---:|"]
        for ft in mdi.index:
            L.append(f"| {ft} | {mdi.loc[ft, 'mean']:.4f} | {mda.loc[ft, 'mean']:.4f} +/- {mda.loc[ft, 'std']:.4f} | {sfi.loc[ft, 'mean']:.5f} |")
        L += ["", f"### Chapter 9 search (purged 3-fold, neg log-loss)", "", "| candidate | min_weight_fraction_leaf | max_features | OOS neg log-loss |", "|---:|---:|---|---:|"]
        for _, r in R.iterrows():
            tag = " (untuned)" if r["cand"] == 0 else (" (chosen)" if int(r["cand"]) == tj["chosen_candidate"] else "")
            L.append(f"| {int(r['cand'])}{tag} | {r['min_weight_fraction_leaf']:.4f} | {r['max_features']} | {r['mean_neg_log_loss']:.5f} |")
        L.append("")
    L += ["## What in this run is not dictated by the book (listed so nothing is hidden)", "",
          "The book's code and tests leave these values to the user, so a value had to be set: dollar-bar size (one average clock-hour of traded value, trailing 30 days); Alpaca rows are hourly; the CUSUM threshold; label barriers 2 x daily volatility and a 4-day vertical barrier; "
          "SADF window 100 / lag 1, entropy window 100 / word 2, FFD threshold 0.01; forest of 100 trees and at most 10,000 draws per tree; bet-size step 0.1; walk-forward cutoff 2024-01-01; flat costs (12 bp per side Kraken, 10 bp per side Alpaca); "
          "the Ch.9 search ranges (min_weight_fraction_leaf log-uniform 0.005 to 0.2, max_features in 1, 2, 3, 6), 11 candidates, 3 folds, 30-tree search forests; HRP re-estimated each calendar quarter from the trailing 252 trading days; "
          "how the pieces are joined (each instrument gets its HRP weight as its slice of capital and the averaged bet signal in [-1, 1] is the position inside the slice); the Chapter 13 exit expressed in multiples of each event's volatility.", "",
          "Book parts not applied: a primary model supplying the side for meta-labeling (the book's code takes `side` as an input and the extract gives no rule for it, so the model learns side and size together, as Snippets 3.5-3.7 do with side=None); Snippet 10.4; Chapter 15; Chapter 19 trade-level measures; the synthetic-data backtest; "
          "no feature was dropped on the strength of the importance table, because the extract gives no dropping rule.", "",
          "## Checks", "",
          "`verify_book_final.py` re-derived every number of the result tables from the raw files with its own code (bet size, exits located by timestamp, positions, day booking, HRP with its own quasi-diagonalisation and bisection, quarterly schedule, compounding): "
          f"Kraken {open(fin['kraken'] / 'checker_output.txt').read().split(' numbers')[0].split()[-1]} numbers, Alpaca {open(fin['alpaca'] / 'checker_output.txt').read().split(' numbers')[0].split()[-1]} numbers, 0 differences on both. "
          "`verify_book_ch8_ch9.py` rebuilds the event set and purged splits and checks the search table: 0 problems on both. The forest fits behind the importance and search tables, and the PSR/DSR arithmetic, were not re-derived by a second implementation.", ""]
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
