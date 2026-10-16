# The book's strategy, one-year backtest on Kraken and Alpaca

Only the book's own algorithms and its own tests: dollar bars and CUSUM (Ch.2); side-free labels (Snippets 3.5-3.7, side=None); uniqueness weights and sequential bootstrap (Ch.4); FFD, SADF, entropy, Corwin-Schultz and volatility features (Ch.5, 17-19, Snippet 3.1); bagged entropy forest (Ch.6) with purged CV (Ch.7), feature importance (Ch.8) and the randomised purged-CV search (Ch.9); side = predicted class and size from Snippets 10.1/10.3 with overlapping bets averaged (10.2); exit by the Chapter 13 optimal trading rule; allocation by Hierarchical Risk Parity (Snippet 16.4); evaluation by walk-forward (trained before 2024-01-01) and the five combinatorial purged CV paths (Ch.12); PSR / deflated Sharpe (Ch.14). Costs on every position change.

## kraken

### Result, 2025-10-13 to 2026-10-12 (net of 12 bp per side, HRP allocation)

| test | gain % | max drawdown % |
|---|---:|---:|
| walk-forward, last 365 days | 0.50 | -0.60 |
| walk-forward, whole test period 2024-01-01 to 2026-10-12 | -0.90 | -1.75 |
| CPCV path 0, last 365 days | -2.13 | -2.51 |
| CPCV path 1, last 365 days | -2.61 | -2.89 |
| CPCV path 2, last 365 days | -1.95 | -2.35 |
| CPCV path 3, last 365 days | -0.39 | -1.33 |
| CPCV path 4, last 365 days | -1.36 | -1.73 |

Model quality: walk-forward AUC 0.5085, side accuracy 0.4977; CPCV AUC by path 0.5295, 0.5067, 0.5056, 0.4990, 0.5002.

Deflated Sharpe (Ch.14; trials = the Ch.9 candidates, 11): last_365_days: PSR against 0 = 0.806, DSR = 0.224; whole_walk_forward_span: PSR against 0 = 0.200, DSR = 0.007.

### Chapter 8 feature importance (28,279 events with labels ending before 2024-01-01)

| feature | MDI | MDA | SFI |
|---|---:|---:|---:|
| ffd | 0.1638 | -0.0002 +/- 0.0004 | -0.70025 |
| log_trgt | 0.1876 | -0.0000 +/- 0.0006 | -0.69909 |
| sadf | 0.2277 | 0.0009 +/- 0.0006 | -0.69750 |
| ent_plugin | 0.1802 | -0.0009 +/- 0.0007 | -0.69911 |
| ent_konto | 0.1333 | 0.0004 +/- 0.0004 | -0.69898 |
| cs_spread | 0.1073 | 0.0004 +/- 0.0003 | -0.69782 |

### Chapter 9 search (purged 3-fold, neg log-loss)

| candidate | min_weight_fraction_leaf | max_features | OOS neg log-loss |
|---:|---:|---|---:|
| 0 (untuned) | 0.0500 | sqrt | -0.69982 |
| 1 | 0.0155 | 1 | -0.70199 |
| 2 | 0.0309 | 6 | -0.70377 |
| 3 | 0.0212 | 2 | -0.70177 |
| 4 | 0.0111 | 3 | -0.70685 |
| 5 | 0.0703 | 6 | -0.70153 |
| 6 | 0.0052 | 3 | -0.71037 |
| 7 (chosen) | 0.1654 | 1 | -0.69801 |
| 8 | 0.0868 | 1 | -0.69820 |
| 9 | 0.0096 | 2 | -0.70623 |
| 10 | 0.0364 | 6 | -0.70308 |

## alpaca

### Result, 2025-10-10 to 2026-10-09 (net of 10 bp per side, HRP allocation)

| test | gain % | max drawdown % |
|---|---:|---:|
| walk-forward, last 365 days | 0.01 | -0.01 |
| walk-forward, whole test period 2024-01-01 to 2026-10-09 | -0.02 | -0.03 |
| CPCV path 0, last 365 days | -0.00 | -0.00 |
| CPCV path 1, last 365 days | 0.01 | -0.02 |
| CPCV path 2, last 365 days | 0.02 | -0.04 |
| CPCV path 3, last 365 days | 0.00 | 0.00 |
| CPCV path 4, last 365 days | 0.01 | -0.02 |

Model quality: walk-forward AUC 0.5130, side accuracy 0.4745; CPCV AUC by path 0.5102, 0.5145, 0.5131, 0.5106, 0.5125.

Deflated Sharpe (Ch.14; trials = the Ch.9 candidates, 11): last_365_days: PSR against 0 = 0.636, DSR = 0.101; whole_walk_forward_span: PSR against 0 = 0.226, DSR = 0.009.

### Chapter 8 feature importance (252,460 events with labels ending before 2024-01-01)

| feature | MDI | MDA | SFI |
|---|---:|---:|---:|
| ffd | 0.2354 | 0.0009 +/- 0.0001 | -0.69439 |
| log_trgt | 0.2057 | -0.0005 +/- 0.0003 | -0.69507 |
| sadf | 0.1683 | -0.0004 +/- 0.0002 | -0.69589 |
| ent_plugin | 0.1788 | 0.0001 +/- 0.0000 | -0.69497 |
| ent_konto | 0.1417 | -0.0001 +/- 0.0000 | -0.69519 |
| cs_spread | 0.0700 | 0.0000 +/- 0.0001 | -0.69464 |

### Chapter 9 search (purged 3-fold, neg log-loss)

| candidate | min_weight_fraction_leaf | max_features | OOS neg log-loss |
|---:|---:|---|---:|
| 0 (untuned) | 0.0500 | sqrt | -0.69532 |
| 1 | 0.0155 | 1 | -0.69508 |
| 2 | 0.0309 | 6 | -0.69627 |
| 3 | 0.0212 | 2 | -0.69580 |
| 4 | 0.0111 | 3 | -0.69759 |
| 5 | 0.0703 | 6 | -0.69553 |
| 6 | 0.0052 | 3 | -0.69858 |
| 7 (chosen) | 0.1654 | 1 | -0.69421 |
| 8 | 0.0868 | 1 | -0.69469 |
| 9 | 0.0096 | 2 | -0.69695 |
| 10 | 0.0364 | 6 | -0.69625 |

## What in this run is not dictated by the book (listed so nothing is hidden)

The book's code and tests leave these values to the user, so a value had to be set: dollar-bar size (one average clock-hour of traded value, trailing 30 days); Alpaca rows are hourly; the CUSUM threshold; label barriers 2 x daily volatility and a 4-day vertical barrier; SADF window 100 / lag 1, entropy window 100 / word 2, FFD threshold 0.01; forest of 100 trees and at most 10,000 draws per tree; bet-size step 0.1; walk-forward cutoff 2024-01-01; flat costs (12 bp per side Kraken, 10 bp per side Alpaca); the Ch.9 search ranges (min_weight_fraction_leaf log-uniform 0.005 to 0.2, max_features in 1, 2, 3, 6), 11 candidates, 3 folds, 30-tree search forests; HRP re-estimated each calendar quarter from the trailing 252 trading days; how the pieces are joined (each instrument gets its HRP weight as its slice of capital and the averaged bet signal in [-1, 1] is the position inside the slice); the Chapter 13 exit expressed in multiples of each event's volatility.

Book parts not applied: a primary model supplying the side for meta-labeling (the book's code takes `side` as an input and the extract gives no rule for it, so the model learns side and size together, as Snippets 3.5-3.7 do with side=None); Snippet 10.4; Chapter 15; Chapter 19 trade-level measures; the synthetic-data backtest; no feature was dropped on the strength of the importance table, because the extract gives no dropping rule.

## Checks

`verify_book_final.py` re-derived every number of the result tables from the raw files with its own code (bet size, exits located by timestamp, positions, day booking, HRP with its own quasi-diagonalisation and bisection, quarterly schedule, compounding): Kraken 451 numbers, Alpaca 15962 numbers, 0 differences on both. `verify_book_ch8_ch9.py` rebuilds the event set and purged splits and checks the search table: 0 problems on both. The forest fits behind the importance and search tables, and the PSR/DSR arithmetic, were not re-derived by a second implementation.
