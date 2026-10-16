# The book's strategy (tuned forest), alpaca

Walk-forward AUC 0.5130, side accuracy 0.4745 (share of test events that went up 0.5304); CPCV AUC by path 0.5102, 0.5145, 0.5131, 0.5106, 0.5125. OTR pt 9 / sl 3 sigma, at most 100 bars. HRP re-estimated every quarter from the trailing 252 trading days. Costs 10 bp per side.

## walk-forward, last_365_days: 2025-10-10 to 2026-10-09 (HRP allocation, net of costs)

| variant | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---:|---:|---:|
| book_bet | 0.01 | -0.01 | 0.34 |

## walk-forward, whole_walk_forward_span: 2024-01-01 to 2026-10-09 (HRP allocation, net of costs)

| variant | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---:|---:|---:|
| book_bet | -0.02 | -0.03 | -0.45 |

## CPCV paths, last 365 days, book_bet, HRP, net of costs

| path | gain % | max drawdown % |
|---|---:|---:|
| cpcv0 | -0.00 | -0.00 |
| cpcv1 | 0.01 | -0.02 |
| cpcv2 | 0.02 | -0.04 |
| cpcv3 | 0.00 | 0.00 |
| cpcv4 | 0.01 | -0.02 |

Forest used: {'n_estimators': 100, 'bootstrap': 'sequential', 'max_draws': 10000, 'min_weight_fraction_leaf': 0.16544787290051172, 'max_features': 1}. DSR (Ch.14, HRP net daily series; trials = the 11 Ch.9 candidates): last_365_days: PSR vs 0 0.636, DSR 11 trials 0.101; whole_walk_forward_span: PSR vs 0 0.226, DSR 11 trials 0.009