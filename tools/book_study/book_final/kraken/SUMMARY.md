# The book's strategy (tuned forest), kraken

Walk-forward AUC 0.5085, side accuracy 0.4977 (share of test events that went up 0.5072); CPCV AUC by path 0.5295, 0.5067, 0.5056, 0.4990, 0.5002. OTR pt 7 / sl 7 sigma, at most 100 bars. HRP re-estimated every quarter from the trailing 252 trading days. Costs 12 bp per side.

## walk-forward, last_365_days: 2025-10-13 to 2026-10-12 (HRP allocation, net of costs)

| variant | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---:|---:|---:|
| book_bet | 0.50 | -0.60 | 0.86 |

## walk-forward, whole_walk_forward_span: 2024-01-01 to 2026-10-12 (HRP allocation, net of costs)

| variant | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---:|---:|---:|
| book_bet | -0.90 | -1.75 | -0.51 |

## CPCV paths, last 365 days, book_bet, HRP, net of costs

| path | gain % | max drawdown % |
|---|---:|---:|
| cpcv0 | -2.13 | -2.51 |
| cpcv1 | -2.61 | -2.89 |
| cpcv2 | -1.95 | -2.35 |
| cpcv3 | -0.39 | -1.33 |
| cpcv4 | -1.36 | -1.73 |

Forest used: {'n_estimators': 100, 'bootstrap': 'sequential', 'max_draws': 10000, 'min_weight_fraction_leaf': 0.16544787290051172, 'max_features': 1}. DSR (Ch.14, HRP net daily series; trials = the 11 Ch.9 candidates): last_365_days: PSR vs 0 0.806, DSR 11 trials 0.224; whole_walk_forward_span: PSR vs 0 0.200, DSR 11 trials 0.007