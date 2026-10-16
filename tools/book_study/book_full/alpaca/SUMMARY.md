# The book as one strategy, alpaca

Walk-forward AUC 0.5126, side accuracy 0.4969 (share of test events that went up 0.5304); CPCV AUC by path 0.5108, 0.5141, 0.5118, 0.5084, 0.5132. OTR pt 9 / sl 3 sigma, at most 100 bars. HRP re-estimated every quarter from the trailing 252 trading days. Costs 10 bp per side.

## walk-forward, last_365_days: 2025-10-10 to 2026-10-09 (HRP allocation, net of costs)

| variant | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---:|---:|---:|
| book_bet | 0.01 | -0.07 | 0.10 |
| pred_side_full_size | -0.54 | -1.60 | -0.29 |
| placebo_sizes | -0.15 | -0.15 | -4.00 |
| always_long | 2.81 | -3.33 | 0.79 |
| random_side | -4.18 | -4.28 | -3.72 |

## walk-forward, whole_walk_forward_span: 2024-01-01 to 2026-10-09 (HRP allocation, net of costs)

| variant | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---:|---:|---:|
| book_bet | -0.08 | -0.14 | -0.40 |
| pred_side_full_size | -4.28 | -5.97 | -0.70 |
| placebo_sizes | -0.26 | -0.26 | -2.49 |
| always_long | 14.67 | -6.87 | 1.00 |
| random_side | -14.77 | -14.80 | -4.45 |

## CPCV paths, last 365 days, book_bet, HRP, net of costs

| path | gain % | max drawdown % |
|---|---:|---:|
| cpcv0 | 0.02 | -0.04 |
| cpcv1 | 0.01 | -0.08 |
| cpcv2 | -0.04 | -0.10 |
| cpcv3 | -0.00 | -0.03 |
| cpcv4 | 0.02 | -0.06 |