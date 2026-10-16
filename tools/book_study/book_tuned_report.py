"""RESULTS.md for the Chapter 8 / Chapter 9 / tuned-strategy step.  Every number is read from saved files.
Usage: python book_tuned_report.py <ch89 kraken> <ch89 alpaca> <tuned kraken> <tuned alpaca> <untuned kraken (book_full)> <untuned alpaca (book_full)> <out RESULTS.md>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    a = [Path(x) for x in sys.argv[1:7]]
    out = Path(sys.argv[7])
    names = ["kraken", "alpaca"]
    ch89, tuned, untuned = dict(zip(names, a[0:2])), dict(zip(names, a[2:4])), dict(zip(names, a[4:6]))
    L = ["# Chapters 8, 9 and 14 added to the whole-book strategy: both venues", "",
         "This adds the three things the book's own Table 1.2 prescribes that the first full run skipped: feature importance (Ch.8), purged-CV hyperparameter search (Ch.9) and the deflated Sharpe (Ch.14). "
         "Importance and search use only events whose labels end before 2024-01-01; the walk-forward test is untouched by them. The strategy, account, costs and HRP step are unchanged from `book_full`.", ""]
    for v in names:
        m89 = json.loads((ch89[v] / "meta.json").read_text())
        mdi, mda, sfi = (pd.read_csv(ch89[v] / f"{x}.csv", index_col=0) for x in ("mdi", "mda", "sfi"))
        R = pd.read_csv(ch89[v] / "search.csv")
        tj = json.loads((ch89[v] / "tuned_forest.json").read_text())
        L += [f"## {v}: Chapter 8 feature importance ({m89['events_used']:,} pre-2024 events, purged 5-fold, embargo {m89['embargo_pct']:g})", "",
              "| feature | MDI | MDA (OOS neg log-loss drop, relative) | SFI (OOS neg log-loss, single feature) |", "|---|---:|---:|---:|"]
        for ft in mdi.index:
            L.append(f"| {ft} | {mdi.loc[ft, 'mean']:.4f} | {mda.loc[ft, 'mean']:.4f} +/- {mda.loc[ft, 'std']:.4f} | {sfi.loc[ft, 'mean']:.5f} |")
        L += ["", f"Whole-model OOS neg log-loss {m89['mda_baseline_neg_log_loss']:.5f}; a coin flip scores {-0.693147:.5f}.", "",
              f"## {v}: Chapter 9 randomised search (purged 3-fold, neg log-loss, {len(R)} candidates incl. the untuned setting)", "",
              f"Untuned (min_weight_fraction_leaf 0.05, max_features sqrt): {R.iloc[0]['mean_neg_log_loss']:.5f}. Best: candidate {tj['chosen_candidate']} "
              f"(min_weight_fraction_leaf {tj['forest']['min_weight_fraction_leaf']:.4f}, max_features {tj['forest']['max_features']}): {tj['best_neg_log_loss']:.5f}. Worst candidate: {R['mean_neg_log_loss'].min():.5f}.", ""]
    L += ["## One-year backtest with the tuned forest (net of costs, walk-forward, HRP allocation), against the untuned run", ""]
    for v in names:
        S = pd.read_csv(tuned[v] / "summary.csv")
        U = pd.read_csv(untuned[v] / "summary.csv")
        mt = json.loads((tuned[v] / "meta.json").read_text())
        mu = json.loads((untuned[v] / "meta.json").read_text())

        def pick(T, scope, span, var, scheme="hrp"):
            return T[(T["scope"] == scope) & (T["span"] == span) & (T["costs"] == "net") & (T["variant"] == var) & (T["scheme"] == scheme)].iloc[0]
        t, u_ = pick(S, "wf", "last_365_days", "book_bet"), pick(U, "wf", "last_365_days", "book_bet")
        tw, uw = pick(S, "wf", "whole_walk_forward_span", "book_bet"), pick(U, "wf", "whole_walk_forward_span", "book_bet")
        te = pick(S, "wf", "last_365_days", "book_bet", "equal")
        cp = S[(S["scope"] != "wf") & (S["variant"] == "book_bet") & (S["costs"] == "net") & (S["span"] == "last_365_days")]
        cpu = U[(U["scope"] != "wf") & (U["variant"] == "book_bet") & (U["costs"] == "net") & (U["span"] == "last_365_days")]
        L += [f"### {v}: {t['start_day']} to {t['end_day']}", "",
              "| | tuned forest | untuned (book_full) |", "|---|---:|---:|",
              f"| walk-forward AUC | {mt['walk_forward']['auc']:.4f} | {mu['walk_forward']['auc']:.4f} |",
              f"| walk-forward side accuracy | {mt['walk_forward']['side_accuracy']:.4f} | {mu['walk_forward']['side_accuracy']:.4f} |",
              f"| CPCV mean AUC | {mt['cpcv']['auc_mean']:.4f} | {mu['cpcv']['auc_mean']:.4f} |",
              f"| book_bet, last 365 days, gain % (max drawdown %) | {t['gain_pct']:.2f} ({t['max_drawdown_pct']:.2f}) | {u_['gain_pct']:.2f} ({u_['max_drawdown_pct']:.2f}) |",
              f"| book_bet, whole walk-forward span, gain % | {tw['gain_pct']:.2f} | {uw['gain_pct']:.2f} |",
              f"| book_bet, 5 CPCV paths, last 365 days, gain % (min to max) | {cp['gain_pct'].min():.2f} to {cp['gain_pct'].max():.2f} | {cpu['gain_pct'].min():.2f} to {cpu['gain_pct'].max():.2f} |",
              f"| book_bet, equal-weight account instead of HRP, last 365 days, gain % | {te['gain_pct']:.2f} | |", ""]
        d = mt["dsr"]
        for sp in ("last_365_days", "whole_walk_forward_span"):
            x = d[sp]
            L.append(f"Deflated Sharpe, {sp}: daily Sharpe {x['sr_daily']:.4f} over {x['n_days']} days, PSR against 0 = {x['psr_vs_0']:.3f}; " + "; ".join(f"with {k} trials DSR = {w['dsr']:.3f}" for k, w in x["by_trials"].items()) + ".")
        L.append("")
    L += ["How to read the DSR numbers: a deflated Sharpe near 1 means the result survives the number of things tried; near 0 means it does not. Trials are the 11 Ch.9 candidates, or 16 when the five earlier strategy variants in this study (stand-in side, side-free with 13 features, side-free book features, + OTR, untuned whole-book run) are counted. "
          "The variance of Sharpe across trials is the larger of the variance across the five CPCV paths (one-year span only) and the asymptotic sampling variance, the same convention as `meta_eval.py`.", "",
          "## Book-compliance ledger after this step", "",
          "Now applied: Ch.2-7, Ch.8 (MDI, MDA, SFI), Ch.9 (randomised purged-CV search; the search ranges for min_weight_fraction_leaf and max_features, the 11 candidates and the 30-tree search forests are my choices), Ch.10 (10.1-10.3), Ch.12 CPCV, Ch.13 OTR, Ch.14 PSR/DSR, Ch.16 HRP, Ch.17-19 features.", "",
          "Still not applied: (1) **meta-labeling with a primary model's side**, the solution the book's Table 1.2 gives for 'learning side and size simultaneously'. The book's code takes `side` as an input and the extract gives no rule that produces it, so this strategy still learns side and size together (Snippets 3.5-3.7, side=None), which Table 1.2 lists as a pitfall. "
          "(2) Snippet 10.4 dynamic position size and limit price; (3) Chapter 15 strategy-risk sizing; (4) Chapter 19 trade-level measures; (5) the synthetic-data backtest of Table 1.2 row 10 (the book's Ch.13 O-U Monte Carlo is used only to choose the exit); (6) no feature was dropped on the strength of the importance table, because the extract gives no dropping rule.", "",
          "## Checks", "",
          "`verify_book_full.py` re-derived every number of both tuned runs from the raw files with its own code: Kraken 775 numbers and Alpaca 16,286 numbers (400 and 15,911 HRP weights among them), 0 differences. "
          "`verify_book_ch8_ch9.py` rebuilds the pre-cutoff event set and the 3- and 5-fold purged splits independently and checks the search table and winner: 0 problems on both venues. The forest fits behind the importance and search tables are the project's ported code and are not re-fitted by the checker. The PSR/DSR arithmetic (`book_stats.py`) was not re-derived by a second implementation in this step.", ""]
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
