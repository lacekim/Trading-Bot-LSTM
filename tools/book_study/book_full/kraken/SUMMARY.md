# The book as one strategy, kraken

Walk-forward AUC 0.5215, side accuracy 0.5078 (share of test events that went up 0.5072); CPCV AUC by path 0.5313, 0.5169, 0.5170, 0.5133, 0.5150. OTR pt 7 / sl 7 sigma, at most 100 bars. HRP re-estimated every quarter from the trailing 252 trading days. Costs 12 bp per side.

## walk-forward, last_365_days: 2025-10-13 to 2026-10-12 (HRP allocation, net of costs)

| variant | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---:|---:|---:|
| book_bet | -1.38 | -2.25 | -0.98 |
| pred_side_full_size | -33.81 | -39.46 | -1.91 |
| placebo_sizes | -2.19 | -2.94 | -1.99 |
| always_long | 18.42 | -23.11 | 0.73 |
| random_side | -12.09 | -24.56 | -0.91 |

## walk-forward, whole_walk_forward_span: 2024-01-01 to 2026-10-12 (HRP allocation, net of costs)

| variant | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---:|---:|---:|
| book_bet | -1.45 | -2.25 | -0.50 |
| pred_side_full_size | -20.70 | -39.84 | -0.30 |
| placebo_sizes | -1.84 | -3.31 | -0.56 |
| always_long | 129.87 | -23.11 | 1.38 |
| random_side | -39.54 | -40.11 | -1.57 |

## CPCV paths, last 365 days, book_bet, HRP, net of costs

| path | gain % | max drawdown % |
|---|---:|---:|
| cpcv0 | -1.19 | -1.78 |
| cpcv1 | -2.19 | -2.49 |
| cpcv2 | -1.12 | -1.82 |
| cpcv3 | -1.13 | -1.55 |
| cpcv4 | -1.46 | -1.82 |