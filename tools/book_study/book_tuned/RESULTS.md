# Chapters 8, 9 and 14 added to the whole-book strategy: both venues

This adds the three things the book's own Table 1.2 prescribes that the first full run skipped: feature importance (Ch.8), purged-CV hyperparameter search (Ch.9) and the deflated Sharpe (Ch.14). Importance and search use only events whose labels end before 2024-01-01; the walk-forward test is untouched by them. The strategy, account, costs and HRP step are unchanged from `book_full`.

## kraken: Chapter 8 feature importance (28,279 pre-2024 events, purged 5-fold, embargo 0.01)

| feature | MDI | MDA (OOS neg log-loss drop, relative) | SFI (OOS neg log-loss, single feature) |
|---|---:|---:|---:|
| ffd | 0.1638 | -0.0002 +/- 0.0004 | -0.70025 |
| log_trgt | 0.1876 | -0.0000 +/- 0.0006 | -0.69909 |
| sadf | 0.2277 | 0.0009 +/- 0.0006 | -0.69750 |
| ent_plugin | 0.1802 | -0.0009 +/- 0.0007 | -0.69911 |
| ent_konto | 0.1333 | 0.0004 +/- 0.0004 | -0.69898 |
| cs_spread | 0.1073 | 0.0004 +/- 0.0003 | -0.69782 |

Whole-model OOS neg log-loss -0.69633; a coin flip scores -0.69315.

## kraken: Chapter 9 randomised search (purged 3-fold, neg log-loss, 11 candidates incl. the untuned setting)

Untuned (min_weight_fraction_leaf 0.05, max_features sqrt): -0.69982. Best: candidate 7 (min_weight_fraction_leaf 0.1654, max_features 1): -0.69801. Worst candidate: -0.71037.

## alpaca: Chapter 8 feature importance (252,460 pre-2024 events, purged 5-fold, embargo 0.01)

| feature | MDI | MDA (OOS neg log-loss drop, relative) | SFI (OOS neg log-loss, single feature) |
|---|---:|---:|---:|
| ffd | 0.2354 | 0.0009 +/- 0.0001 | -0.69439 |
| log_trgt | 0.2057 | -0.0005 +/- 0.0003 | -0.69507 |
| sadf | 0.1683 | -0.0004 +/- 0.0002 | -0.69589 |
| ent_plugin | 0.1788 | 0.0001 +/- 0.0000 | -0.69497 |
| ent_konto | 0.1417 | -0.0001 +/- 0.0000 | -0.69519 |
| cs_spread | 0.0700 | 0.0000 +/- 0.0001 | -0.69464 |

Whole-model OOS neg log-loss -0.69448; a coin flip scores -0.69315.

## alpaca: Chapter 9 randomised search (purged 3-fold, neg log-loss, 11 candidates incl. the untuned setting)

Untuned (min_weight_fraction_leaf 0.05, max_features sqrt): -0.69532. Best: candidate 7 (min_weight_fraction_leaf 0.1654, max_features 1): -0.69421. Worst candidate: -0.69858.

## One-year backtest with the tuned forest (net of costs, walk-forward, HRP allocation), against the untuned run

### kraken: 2025-10-13 to 2026-10-12

| | tuned forest | untuned (book_full) |
|---|---:|---:|
| walk-forward AUC | 0.5085 | 0.5215 |
| walk-forward side accuracy | 0.4977 | 0.5078 |
| CPCV mean AUC | 0.5082 | 0.5187 |
| book_bet, last 365 days, gain % (max drawdown %) | 0.50 (-0.60) | -1.38 (-2.25) |
| book_bet, whole walk-forward span, gain % | -0.90 | -1.45 |
| book_bet, 5 CPCV paths, last 365 days, gain % (min to max) | -2.61 to -0.39 | -2.19 to -1.12 |
| book_bet, equal-weight account instead of HRP, last 365 days, gain % | 1.58 | |

Deflated Sharpe, last_365_days: daily Sharpe 0.0448 over 365 days, PSR against 0 = 0.806; with 11 trials DSR = 0.224; with 16 trials DSR = 0.174.
Deflated Sharpe, whole_walk_forward_span: daily Sharpe -0.0265 over 1016 days, PSR against 0 = 0.200; with 11 trials DSR = 0.007; with 16 trials DSR = 0.004.

### alpaca: 2025-10-10 to 2026-10-09

| | tuned forest | untuned (book_full) |
|---|---:|---:|
| walk-forward AUC | 0.5130 | 0.5126 |
| walk-forward side accuracy | 0.4745 | 0.4969 |
| CPCV mean AUC | 0.5122 | 0.5117 |
| book_bet, last 365 days, gain % (max drawdown %) | 0.01 (-0.01) | 0.01 (-0.07) |
| book_bet, whole walk-forward span, gain % | -0.02 | -0.08 |
| book_bet, 5 CPCV paths, last 365 days, gain % (min to max) | -0.00 to 0.02 | -0.04 to 0.02 |
| book_bet, equal-weight account instead of HRP, last 365 days, gain % | -0.04 | |

Deflated Sharpe, last_365_days: daily Sharpe 0.0180 over 365 days, PSR against 0 = 0.636; with 11 trials DSR = 0.101; with 16 trials DSR = 0.073.
Deflated Sharpe, whole_walk_forward_span: daily Sharpe -0.0236 over 1013 days, PSR against 0 = 0.226; with 11 trials DSR = 0.009; with 16 trials DSR = 0.005.

How to read the DSR numbers: a deflated Sharpe near 1 means the result survives the number of things tried; near 0 means it does not. Trials are the 11 Ch.9 candidates, or 16 when the five earlier strategy variants in this study (stand-in side, side-free with 13 features, side-free book features, + OTR, untuned whole-book run) are counted. The variance of Sharpe across trials is the larger of the variance across the five CPCV paths (one-year span only) and the asymptotic sampling variance, the same convention as `meta_eval.py`.

## Book-compliance ledger after this step

Now applied: Ch.2-7, Ch.8 (MDI, MDA, SFI), Ch.9 (randomised purged-CV search; the search ranges for min_weight_fraction_leaf and max_features, the 11 candidates and the 30-tree search forests are my choices), Ch.10 (10.1-10.3), Ch.12 CPCV, Ch.13 OTR, Ch.14 PSR/DSR, Ch.16 HRP, Ch.17-19 features.

Still not applied: (1) **meta-labeling with a primary model's side**, the solution the book's Table 1.2 gives for 'learning side and size simultaneously'. The book's code takes `side` as an input and the extract gives no rule that produces it, so this strategy still learns side and size together (Snippets 3.5-3.7, side=None), which Table 1.2 lists as a pitfall. (2) Snippet 10.4 dynamic position size and limit price; (3) Chapter 15 strategy-risk sizing; (4) Chapter 19 trade-level measures; (5) the synthetic-data backtest of Table 1.2 row 10 (the book's Ch.13 O-U Monte Carlo is used only to choose the exit); (6) no feature was dropped on the strength of the importance table, because the extract gives no dropping rule.

## Checks

`verify_book_full.py` re-derived every number of both tuned runs from the raw files with its own code: Kraken 775 numbers and Alpaca 16,286 numbers (400 and 15,911 HRP weights among them), 0 differences. `verify_book_ch8_ch9.py` rebuilds the pre-cutoff event set and the 3- and 5-fold purged splits independently and checks the search table and winner: 0 problems on both venues. The forest fits behind the importance and search tables are the project's ported code and are not re-fitted by the checker. The PSR/DSR arithmetic (`book_stats.py`) was not re-derived by a second implementation in this step.
