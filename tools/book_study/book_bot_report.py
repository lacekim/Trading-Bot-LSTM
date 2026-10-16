"""RESULTS.md of the integrated book bot (book_bot.py).  Every number is read from saved files; nothing is typed in.
Usage (project root): python tools/book_study/book_bot_report.py <bot kraken run> <bot alpaca run> <old book_final kraken> <old book_final alpaca> <out RESULTS.md>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def f(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def main() -> int:
    a = [Path(x) for x in sys.argv[1:5]]
    out = Path(sys.argv[5])
    names = ["kraken", "alpaca"]
    new, old = dict(zip(names, a[0:2])), dict(zip(names, a[2:4]))
    L = ["# The book bot: one integrated run per venue, from dollar bars to the Chapter 14 and 15 statistics", "",
         "One command per venue runs every book component that can be computed from our data, in the book's order, with no side supplied by anything outside the book: "
         "dollar bars and CUSUM events (Ch.2); triple-barrier labels with side=None (Snippets 3.1-3.7, bin = sign of the return) and Snippet 3.8 dropLabels; uniqueness and attribution weights, sequential bootstrap and Snippet 4.11 time decay (Ch.4); "
         "twelve features: FFD (Ch.5), Snippet 3.1 volatility, SADF (Ch.17), plug-in and Kontoyiannis entropy (Ch.18), Corwin-Schultz spread and Becker-Parkinson volatility (19.1-19.2), Roll, Kyle / Amihud / Hasbrouck lambdas (Eq.82-92) and Chu-Stinchcombe-White CUSUM (17.3.2); "
         "the Chapter 13 optimal trading rule on a 20 x 20 mesh with 100,000 simulated paths; MDI / MDA / SFI / orthogonal features / weighted tau (Ch.8); the randomised purged-CV search (Ch.9); bagged entropy forest with sequential bootstrap (Ch.6); "
         "walk-forward and combinatorial purged CV (Ch.7, 12); bet size and averaging (Snippets 10.1-10.3); HRP allocation (Snippet 16.4); PSR / deflated Sharpe, bet timing, holding period, HHI, drawdown and time under water, annualised return (Ch.14); "
         "binHR, binFreq, binSR, probFailure and the t-value of precision (Ch.15); the Snippet 10.4 order ticket; the rebalance cost, spread cost and volume rule of section 2.4.1; the Eq.77-78 concentration of every HRP allocation. The last section lists every item of the book's Contents and what the bot does with it.", ""]
    S_, M_ = {}, {}
    for v in names:
        S_[v] = pd.read_csv(new[v] / "summary.csv")
        M_[v] = json.loads((new[v] / "meta.json").read_text())

    # ---------- headline ----------
    L += ["## Result (net of costs, HRP allocation, walk-forward trained before 2024-01-01)", "",
          "| venue | last 365 days: gain % | max drawdown % | whole test span: gain % | max drawdown % | Eq.49 annualised % (last 365) |", "|---|---:|---:|---:|---:|---:|"]
    for v in names:
        S = S_[v]
        w1 = S[(S["scope"] == "wf") & (S["span"] == "last_365_days")].iloc[0]
        w2 = S[(S["scope"] == "wf") & (S["span"] == "whole_walk_forward_span")].iloc[0]
        L.append(f"| {v} ({w1['start_day']} to {w1['end_day']}) | {f(w1['gain_pct'])} | {f(w1['max_drawdown_pct'])} | {f(w2['gain_pct'])} ({w2['start_day']} to {w2['end_day']}) | {f(w2['max_drawdown_pct'])} | {f(M_[v]['ch14']['annualised_return_pct_eq49'])} |")
    L += ["", "Combinatorial purged CV paths, last 365 days (gain % / max drawdown %):", "", "| venue | " + " | ".join(f"path {i}" for i in range(5)) + " |", "|---|" + "---:|" * 5]
    for v in names:
        S = S_[v]
        cp = S[(S["scope"] != "wf") & (S["span"] == "last_365_days")]
        L.append(f"| {v} | " + " | ".join(f"{f(r['gain_pct'])} / {f(r['max_drawdown_pct'])}" for _, r in cp.iterrows()) + " |")
    L.append("")

    # ---------- correction ----------
    L += ["## Correction to the earlier results", "",
          "The earlier delivered runs (`book_final`) were not side-free in two places. Their Chapter 13 exit rule was fitted on paths signed by a stand-in side (the sign of the trailing 720-bar return), and the event set dropped every event whose stand-in side was zero, "
          "whose 720-bar lookback crossed a data hole, or that lacked a 2,160-bar momentum or a 200-day market-state value. None of that is in the book. This bot removes all of it: the label has no side, the Chapter 13 fit uses the long price path, and an event is dropped only when one of its own twelve features, its volatility or its label is missing. "
          "The numbers below are the old run against the new run, both read from saved files.", "",
          "| venue | | old (book_final) | new (this bot) |", "|---|---|---:|---:|"]
    for v in names:
        mo = json.loads((old[v] / "meta.json").read_text())
        So = pd.read_csv(old[v] / "summary.csv")
        o1 = So[(So["scope"] == "wf") & (So["span"] == "last_365_days")].iloc[0]
        o2 = So[(So["scope"] == "wf") & (So["span"] == "whole_walk_forward_span")].iloc[0]
        S = S_[v]
        n1 = S[(S["scope"] == "wf") & (S["span"] == "last_365_days")].iloc[0]
        n2 = S[(S["scope"] == "wf") & (S["span"] == "whole_walk_forward_span")].iloc[0]
        mn = M_[v]
        L += [f"| {v} | events | {mo['events'] if not isinstance(mo['events'], dict) else mo['events']} | {mn['events']:,} |",
              f"| {v} | Ch.13 chosen pt / sl | {mo['otr']['pt']} / {mo['otr']['sl']} | {mn['otr']['pt']} / {mn['otr']['sl']} |",
              f"| {v} | walk-forward AUC | {f(mo['walk_forward']['auc'], 4)} | {f(mn['walk_forward']['auc'], 4)} |",
              f"| {v} | last 365 days gain % (max DD %) | {f(o1['gain_pct'])} ({f(o1['max_drawdown_pct'])}) | {f(n1['gain_pct'])} ({f(n1['max_drawdown_pct'])}) |",
              f"| {v} | whole span gain % (max DD %) | {f(o2['gain_pct'])} ({f(o2['max_drawdown_pct'])}) | {f(n2['gain_pct'])} ({f(n2['max_drawdown_pct'])}) |"]
    L.append("")

    # ---------- per venue ----------
    for v in names:
        m, S = M_[v], S_[v]
        ch13 = json.loads((new[v] / "ch13.json").read_text())
        mdi, mda, sfi = (pd.read_csv(new[v] / f"{x}.csv", index_col=0) for x in ("mdi", "mda", "sfi"))
        orth = pd.read_csv(new[v] / "mdi_orthogonal.csv", index_col=0)
        R = pd.read_csv(new[v] / "search.csv")
        tj = json.loads((new[v] / "tuned_forest.json").read_text())
        sym = pd.read_csv(new[v] / "symbols.csv")
        L += [f"## {v}", "", f"### Events (Ch.2-4)", "",
              f"{len(sym)} symbols, {m['events_cusum']:,} CUSUM events, {m['events_before_dropLabels']:,} kept with all twelve features and a label, {m['events']:,} after Snippet 3.8 (dropped label classes: {m['labels_dropped_3_8']}; shares before: "
              + ", ".join(f"{k}: {x:.4f}" for k, x in sorted(m["label_shares_before_dropLabels"].items())) + f"). Snippet 4.11 time decay at the snippet default c = {m['time_decay_4_11']['c']}: constant {m['time_decay_4_11']['const']}, slope {m['time_decay_4_11']['slope']} (no decay).",
              ""]
        if m.get("universe"):
            L += [f"Point-in-time universe: `{m['universe']}`.", ""]
        L += ["### Chapter 13 optimal trading rule (20 x 20 mesh, 100,000 paths)", "",
              f"Equations 13.5-13.7 fitted on {ch13['fit_events']:,} pre-cutoff events (long paths, K = {ch13['K']}): phi = {f(ch13['phi'], 4)}, half-life {f(ch13['half_life_bars'], 1)} bars, forecast {f(ch13['forecast_level_sigma'], 3)} sigma. "
              f"Step 5a chose profit-taking {ch13['chosen_pt']} / stop-loss {ch13['chosen_sl']} (simulated Sharpe {f(ch13['simulated_sharpe_at_chosen'], 4)} against a mesh median of {f(ch13['simulated_sharpe_median_over_mesh'], 4)}). "
              f"Definition 2 on {ch13['test_events']:,} post-cutoff events: Sharpe of the chosen rule {f(ch13['definition_2']['oos_sharpe_chosen'], 4)}, median over the mesh {f(ch13['definition_2']['oos_sharpe_mesh_median'], 4)}, best node in hindsight {f(ch13['definition_2']['oos_sharpe_mesh_best_in_hindsight'], 4)}; "
              f"overfit by the book's definition: **{ch13['definition_2']['overfit']}**." + (" The fitted phi is at or above 1, so the pooled paths are not mean-reverting: the book requires phi in (0, 1) for the O-U model and its half-life, so the model does not hold here and the mesh result is mechanical." if ch13["phi"] >= 1 else "") + f" The vectorised mesh was cross-checked against the verbatim Snippet 13.2 loop on {ch13['verbatim_loop_paths']:,} paths ({len(ch13['verbatim_loop_check'])} nodes): see `ch13.json`.", "",
              f"### Chapter 8 feature importance ({m['events_pre_cutoff']:,} events with labels ending before 2024-01-01)", "", "| feature | MDI | MDA | SFI |", "|---|---:|---:|---:|"]
        for ft in mdi.index:
            L.append(f"| {ft} | {mdi.loc[ft, 'mean']:.4f} | {mda.loc[ft, 'mean']:.4f} +/- {mda.loc[ft, 'std']:.4f} | {sfi.loc[ft, 'mean']:.5f} |")
        L += ["", "A blank (nan) MDI means no tree ever split on that feature: Snippet 8.2 replaces zero importances by NaN before averaging.", "", f"Orthogonal features (8.5): {tj['ortho_components']} principal components carry 95% of the variance; weighted Kendall tau between MDI and the inverse PCA rank (8.6) = {f(tj['weighted_tau_mdi_vs_inverse_pca_rank'], 4)}.", "",
              "### Chapter 9 search (purged 3-fold, neg log-loss)", "", "| candidate | min_weight_fraction_leaf | max_features | OOS neg log-loss |", "|---:|---:|---|---:|"]
        for _, r in R.iterrows():
            tag = " (untuned)" if r["cand"] == 0 else ""
            tag += " (chosen)" if int(r["cand"]) == tj["chosen_candidate"] else ""
            L.append(f"| {int(r['cand'])}{tag} | {r['min_weight_fraction_leaf']:.4f} | {r['max_features']} | {r['mean_neg_log_loss']:.5f} |")
        t = m["walk_forward"]["table_14_1"]
        L += ["", "### Model quality and classification scores (Table 14.1, walk-forward test)", "",
              f"Walk-forward AUC {f(m['walk_forward']['auc'], 4)}, {m['walk_forward']['n_test']:,} test events; CPCV AUC by path {', '.join(f(x, 4) for x in m['cpcv']['auc_by_path'])} (mean {f(m['cpcv']['auc_mean'], 4)}).", "",
              f"Accuracy {f(t['accuracy'], 4)}, precision {f(t['precision'], 4)}, recall {f(t['recall'], 4)}, F1 {f(t['f1'], 4)} (TP {t['tp']}, FP {t['fp']}, TN {t['tn']}, FN {t['fn']}).", "",
              "### PSR and deflated Sharpe (Ch.14; trials = the Ch.9 candidates)", "", "| span | days | daily SR | skew | kurtosis | PSR vs 0 | trials | SR* | DSR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, x in m["dsr"].items():
            L.append(f"| {k} | {x['n_days']} | {f(x['sr_daily'], 4)} | {f(x['skew'], 2)} | {f(x['kurt'], 2)} | {f(x['psr_vs_0'], 3)} | {x['trials']} | {f(x['sr_star'], 4)} | {f(x['dsr'], 3)} |")
        c = m["ch14"]
        L += ["", f"### Chapter 14 statistics, {c['start']} to {c['end']}", "",
              f"Net gain {f(c['gain_pct'])}% over {c['years_elapsed']:.3f} years; annualised return by Equation 49 {f(c['annualised_return_pct_eq49'])}%. Days with a non-zero target position {c['days_with_nonzero_position']} of {c['days']}; mean absolute position {f(c['mean_abs_position'], 4)}, maximum {f(c['max_abs_position'], 4)}. "
              f"Bets (14.1) {c['bets_14_1']}; average holding period (14.2) {f(c['avg_holding_period_days_14_2'], 2)} days; HHI of positive returns {f(c['hhi_positive_returns_14_3'], 4)}, of negative returns {f(c['hhi_negative_returns_14_3'], 4)}, of bets per month {f(c['hhi_bets_per_month_14_3'], 4)} (14.3); "
              f"{c['drawdowns_14_4']['count']} drawdowns, worst {f(c['drawdowns_14_4']['max_pct'])}%, 95th percentile {f(c['drawdowns_14_4']['p95_pct'])}%; time under water, max {f(c['time_under_water_years_14_4']['max'], 3)} years (14.4).", ""]
        c5 = m["ch15"]
        L += [f"### Chapter 15 (per-bet unit returns, last 365 days, {c5['n_bets']:,} bets)", "",
              f"Observed precision {f(c5['observed_precision'], 4)} (t-value under H0: p = 1/2, Equation 54: {f(c5['t_value_of_precision_eq54'], 2)}); mean positive bet return {f(c5['mean_positive_return'], 5)}, mean non-positive {f(c5['mean_nonpositive_return'], 5)}.", "",
              "| target Sharpe | implied precision (15.3) | implied frequency (15.4) | Sharpe at observed precision (Eq.56) | probability of failure (15.5) |", "|---:|---:|---:|---:|---:|"]
        for k, x in c5["targets"].items():
            L.append(f"| {k} | {f(x['implied_precision_15_3'], 4)} | {f(x['implied_frequency_15_4_at_observed_precision'], 1)} | {f(x['sharpe_of_observed_precision_eq56'], 4)} | {f(x['prob_failure_15_5'], 4)} |")
        o = m["ch10_orders"]
        L += ["", f"### Snippet 10.4 order tickets", "", f"{o['n_orders']:,} tickets in `orders_10_4.csv` (forecast price = the barrier of the predicted side, M* = 0.95, Q = 100): median target position {f(o['median_target_position'], 0)}, median limit offset {f(o['median_limit_offset_bp'], 1)} bp. Nothing is connected to any order path. The extract gives no forecast model, so the forecast price is my assumption (the barrier of the predicted side); that makes the ticket size the calibrated M* x Q for every order, so the tickets carry the side but no information about confidence. The confidence-dependent size is the Snippet 10.3 value in the `size_10_3` column, and that is what the account trades.", ""]
        re_ = m["rebalance_cost_effect"]
        vc = m["volume_capacity"]
        cc = m["concentration_eq77_78"]
        L += ["### Section 2.4.1: rebalance cost, spread, volume", "",
              f"Rebalance cost (item 1, Equation 19): on each quarterly HRP rebalance the whole old allocation is sold and the whole new one bought at the venue's cost per side, on the exposure carried into that day. "
              f"Last 365 days: {re_['last_365_days']['rebalances_charged']} rebalances charged, {f(re_['last_365_days']['total_charge_pct_of_capital'], 3)}% of capital in total; walk-forward gain {f(re_['last_365_days']['gain_pct_before'])}% before it, {f(re_['last_365_days']['gain_pct_after'])}% after (the table at the top is after). "
              f"Whole test span: {re_['whole_walk_forward_span']['rebalances_charged']} charged, {f(re_['whole_walk_forward_span']['total_charge_pct_of_capital'], 3)}% of capital; gain {f(re_['whole_walk_forward_span']['gain_pct_before'])}% before, {f(re_['whole_walk_forward_span']['gain_pct_after'])}% after.", "",
              "Bid-ask spread (item 2): the account already pays the venue's cost per side on every change of every instrument's averaged position, which is the book's per-unit crossing cost; nothing is added.", "",
              f"Volume (item 3): a trade changes an instrument's exposure by K x |de| for account size K; it fits when that is no more than the traded value of its own bar. Over the {vc['n_trades']:,} walk-forward trades the largest account K that fits every trade is {f(vc.get('capacity_min'), 0)}, "
              f"the 1st percentile {f(vc.get('capacity_p01'), 0)}, the 5th {f(vc.get('capacity_p05'), 0)}, the median {f(vc.get('capacity_median'), 0)} (currency units of the venue). Share of trades larger than their bar's volume at K = "
              + ", ".join(f"{int(float(k)):,}: {f(100 * x, 2)}%" for k, x in vc.get("share_not_fitting", {}).items()) + ". The account is not volume-capped: this is a capacity measurement.", "",
              f"### Equations 77-78: concentration of the HRP allocation", "",
              f"H over {cc['rebalances']} rebalances: first {f(cc['H_first'], 3)}, last {f(cc['H_last'], 3)}, mean {f(cc['H_mean'], 3)}, range {f(cc['H_min'], 3)} to {f(cc['H_max'], 3)} (file `portfolio_concentration_eq77_78.csv`). The extract prints the equations but not the definition of f; f = V'w (loadings on the covariance eigenvectors) is assumed.", ""]
        L += [f"Run: {m['seconds']} s, seed {m['seed']}, cost {m['cost_bp_per_side']:.0f} bp per side, `quick` = {m['quick']}, integrity of the original files unchanged = {m['integrity_unchanged']}.", ""]

    # ---------- ledger ----------
    L += ["## What is the book, what is a project choice, what is not applied", "",
          "**Project choices (the book leaves them open; fixed before any result, none searched):** dollar-bar size (one average clock-hour of traded value, trailing 30 days) and hourly Alpaca rows; CUSUM threshold (causal one-bar sigma x sqrt(bars per day)); barriers 2 x daily volatility and a 4-day vertical barrier with a path cap of 600 bars (Kraken) / 150 (Alpaca); "
          "SADF window 100 / lag 1, entropy window 100 / word 2, FFD threshold 0.01; 100-row windows for Roll, the three lambdas and CSW, 24 rows for the Becker-Parkinson mean; forest of 100 trees (the book uses 1,000) and at most 10,000 draws; bet-size step 0.1; cutoff 2024-01-01; flat costs 12 bp (Kraken) / 10 bp (Alpaca) per side; "
          "Ch.9 ranges (min_weight_fraction_leaf log-uniform 0.005-0.2; max_features in 1, 2, 3, 6, all) with the untuned setting as candidate 0; HRP re-estimated quarterly on the trailing 252 trading days; the account join (each instrument's HRP weight is its slice of capital, the averaged bet signal in [-1, 1] is the position inside it); "
          "the Alpaca universe (top 200 by dollar volume, point-in-time month-end membership).", "",
          "**Declared deviations from the book's data:** the three lambdas are computed on hourly reference rows with the tick rule on the row's close, not on trades inside each bar (we hold no trade tape for most of the history); the Chapter 13 fit uses long paths because there is no side before the model exists.", "",
          "The extract holds snippets, equations and tables with almost no prose, so where it is silent (thresholds, window lengths, costs, how an allocation is joined to a signal) the choice above is mine and is listed, not hidden.", ""]

    LEDGER = [
        ("Ch.1  Tables 1.1, 1.2", "Used as the map", "Rows 2-5, 7-10 are applied (feature importance, volume clock, FFD, triple barrier, uniqueness / sequential bootstrap, purged CV, CPCV, synthetic-data backtest and deflated Sharpe). Row 6 (learning side and size together; the fix is meta-labeling with a primary model) is NOT satisfied: the extract gives no primary rule, so this bot learns side and size together, which is that pitfall. Row 1 (meta-strategy) likewise needs a primary model."),
        ("Ch.2  Table 2.1, Fig 2.1", "Market data only", "No fundamental, analytics or alternative data in the project."),
        ("Ch.2  Eq 1-3 (time / tick / volume / dollar bars)", "Applied: dollar bars", "Bars are built by the project's builder (bar size is a project choice)."),
        ("Ch.2  Eq 4-15 (imbalance and run bars)", "Not applied", "Need the aggressor side of every trade; we hold no trade tape for most of the history."),
        ("Ch.2  Expression 1, Eq 16-18 (ETF-trick basket), Snippets 2.2-2.3 (roll gaps, rolled series)", "Not applied", "They build one price series from a roll-adjusted futures basket; we trade spot instruments one by one."),
        ("Ch.2  Expression 2, Eq 19-21 (rebalance cost, spread, volume)", "Applied", "Section 2.4.1 above: rebalance cost taken out of the account; spread paid per side; volume capacity measured."),
        ("Ch.2  Eq 22-25, Snippet 2.1 (PCA weights), Fig 2.2", "Not applied", "A risk-distribution weighting for the ETF-trick basket; the account is allocated by HRP (Ch.16)."),
        ("Ch.2  Eq 26-29, Snippet 2.4, Fig 2.3 (CUSUM filter)", "Applied", "Events are the CUSUM filter's output; the threshold is a project choice."),
        ("Ch.2  Eq 30-31 (fixed-horizon labels)", "Not used by design", "The method Chapter 3 replaces (Table 1.2 row 5)."),
        ("Ch.3  Snippets 3.1-3.4 (daily vol, triple barrier)", "Applied", "Barriers 2 x daily volatility, 4-day vertical barrier."),
        ("Ch.3  Snippets 3.5-3.7 (side and size labels)", "Applied with side = None", "bin = sign of the return at the first touch."),
        ("Ch.3  Snippet 3.8 (drop rare labels)", "Applied", "Removes the zero-return class when below 5%."),
        ("Ch.4  Snippets 4.1-4.2, Eq 32-33 (concurrency, uniqueness), 4.10 (return attribution)", "Applied", "Weights computed on all events, then 3.8, as the book orders it."),
        ("Ch.4  Snippets 4.3-4.5, Matrix 1, Fig 4.1 (sequential bootstrap)", "Applied", "Inside the forest; the uniqueness of the draws is logged per split (split_metrics.csv)."),
        ("Ch.4  Snippets 4.6-4.9, Fig 4.2, Function 1 (random-t1 Monte Carlo comparison of bootstraps)", "Not applied", "A demonstration on synthetic data."),
        ("Ch.4  Section 4.7, Snippet 4.11 (time decay)", "Applied at the snippet default c = 1", "That is no decay; c is the user's parameter and was not searched."),
        ("Ch.5  Snippets 5.1-5.4, Fig 5.1-5.5, Section 5.4", "Applied", "FFD with the smallest d that passes the stationarity test, per symbol."),
        ("Ch.5  Table 5.1 (futures ADF table)", "Reference data", "Not a procedure."),
        ("Ch.6  Eq 34-35, Section 6.3.1, Snippet 6.1, Fig 6.1-6.3", "Used as the rationale", "Bagging accuracy / variance reduction; no code to run."),
        ("Ch.6  Snippet 6.2", "Applied (clf1)", "Entropy trees, class_weight balanced, bagged with max_samples = average uniqueness, sequential bootstrap; 100 trees against the book's 1,000."),
        ("Ch.7  Snippets 7.1-7.4, Fig 7.1-7.3", "Applied", "Purging and embargo in the Ch.9 search, the CPCV and the walk-forward cutoff."),
        ("Ch.8  Snippets 8.2-8.5 (MDI, MDA, SFI, orthogonal features)", "Applied", "Tables above; weighted Kendall's tau of MDI against the inverse PCA rank."),
        ("Ch.8  Snippets 8.6-8.10, Fig 8.1-8.4 (synthetic test harness, plots)", "Not applied", "They validate the importance methods on synthetic data; the tables are saved instead of plots."),
        ("Ch.9  Snippets 9.2-9.4, Function 2, Log 1, Fig 9.1-9.2", "Applied", "Randomised purged-CV search with neg log-loss and sample weights."),
        ("Ch.9  Snippet 9.1 (grid search)", "Not used", "The randomised search (9.2) is the one run."),
        ("Ch.10  Snippets 10.1-10.3, Section 10.3", "Applied", "Size from the predicted probability, averaged over active bets, discretised in steps of 0.1."),
        ("Ch.10  Section 10.6, Snippet 10.4, Fig 10.3 (sigmoid)", "Applied", "Order tickets (forecast price is an assumption, see above)."),
        ("Ch.10  Eq 36 (sizing from bet concurrency with a Gaussian-mixture CDF), power-function alternative", "Not applied", "Alternatives to 10.3; the book gives no code and refers the mixture fit to another paper."),
        ("Ch.11  Eq 37-38, Fig 11.1-11.2 (CSCV, probability of backtest overfitting)", "Not applied", "Needs the profit series of every trial; the Ch.9 candidates were scored by log-loss, not by a saved P&L series."),
        ("Ch.12  Eq 39-44, Fig 12.1-12.2 (CPCV)", "Applied", "6 groups, 2 test groups, 15 splits, 5 paths, embargo 0.01."),
        ("Ch.13  Section 13.4, Steps 1-4, 5a, Definition 2, Snippet 13.2, Fig 13.x", "Applied", "20 x 20 mesh, 100,000 paths; the verbatim Snippet 13.2 loop is run as a cross-check."),
        ("Ch.13  Steps 5b-5c", "Not applied", "Step 5b needs a profit target supplied by the strategy and 5c a maximum stop-loss imposed by fund management; neither exists here."),
        ("Ch.13  Snippet 13.1, Table 13.1, Section 13.5.1 (synthetic grid over forecast and half-life)", "Not applied", "Illustrative experiment on assumed O-U parameters; here the parameters are fitted."),
        ("Ch.14  Snippets 14.1-14.4, Eq 50-51, Series 1-4", "Applied", "Last 365 days of the walk-forward account."),
        ("Ch.14  Eq 45-47 (returns on the capital base)", "Applied without dividends", "Exposure never exceeds capital; there is no dividend series in the data."),
        ("Ch.14  Eq 48-49", "Applied", "Cumulative return and annualised return."),
        ("Ch.14  Eq 52-53, Table 14.1, Fig 14.2-14.3", "Applied", "PSR, deflated Sharpe with the Ch.9 candidates as trials, classification scores."),
        ("Ch.15  Eq 54-57, Snippets 15.3-15.5", "Applied", "Implied precision and frequency, strategy risk, t-value of precision."),
        ("Ch.15  Snippets 15.1-15.2, Fig 15.1-15.3", "Not applied", "Monte Carlo and symbolic demonstrations of the Sharpe-precision relation."),
        ("Ch.16  Snippets 16.1-16.4, Section 16.4.1, Eq 58", "Applied", "HRP as printed (the square distance matrix goes to the linkage, which is Eq 58's distance of distances); quarterly."),
        ("Ch.16  Snippet 16.5, Table 16.1, appendices, CLA", "Not applied", "A Monte Carlo comparison of allocators on synthetic data."),
        ("Ch.17  Snippets 17.1-17.4, Eq 62-63 (SADF)", "Applied", "Daily log closes, window 100, minimum sample 20, one lag."),
        ("Ch.17  Section 17.3.2 (Chu-Stinchcombe-White)", "Applied", "sup of S over the last 100 hourly rows, as a feature (the critical-value test is not used)."),
        ("Ch.17  Section 17.3.1 (Brown-Durbin-Evans), Eq 59-61 (Chow-type DF), Eq 65-66 (sub / super-martingale tests)", "Not applied", "The extract prints the equations without the specification needed to run them (regressors, trimming, dummy definition, exponent)."),
        ("Ch.17  Eq 64, Table 17.1, Fig 17.1-17.3", "Not applied", "Computational cost and illustrations."),
        ("Ch.18  Eq 67-75, Snippets 18.1-18.4 (plug-in, Kontoyiannis entropy)", "Applied", "Window 100, word length 2."),
        ("Ch.18  Eq 76-78 (effective number, concentration)", "Applied for Eq 77-78", "On every HRP allocation; Eq 76 is the generalised mean they build on."),
        ("Ch.18  Eq 79-80 (PIN, VPIN)", "Not applied", "Need buy / sell volume of trades."),
        ("Ch.19  Eq 81-83, 90-92 (tick rule, Roll, Kyle / Amihud / Hasbrouck lambdas)", "Applied", "On hourly reference rows (declared deviation)."),
        ("Ch.19  Eq 84-89, Snippets 19.1-19.2 (Corwin-Schultz, Becker-Parkinson)", "Applied", "Spread and volatility features."),
        ("Ch.19  Fig 19.1-19.3, Sections 19.5.1-19.5.2", "Illustrations / PIN, VPIN", "See above."),
        ("Ch.20-22  Snippets 20.1-20.14, Eq 93-98, Snippets 21.1-21.7, Fig 22.x", "Not applied", "High-performance computing, brute-force dynamic portfolio and infrastructure; not strategy components."),
    ]
    L += ["## The book's Contents, item by item", "", "| item | status | what the bot does / why not |", "|---|---|---|"]
    for it, stt, why in LEDGER:
        L.append(f"| {it} | {stt} | {why} |")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
