# Book side-free path, alpaca: one-year backtest (2024-01-03 .. 2026-10-06 out-of-sample)

Walk-forward AUC 0.4956; side accuracy 0.4951 (share of test events that went up 0.5304). N_ref = 194 slots; costs 10 bp per side. Trees 100.

## last_365_days: 2025-10-07 to 2026-10-06

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet | net | -4.11 | -4.13 | -2.28 |
| pred_side_full_size | net | -24.60 | -24.83 | -2.39 |
| placebo_sizes | net | -3.48 | -3.51 | -2.85 |
| random_side | net | -13.69 | -13.79 | -5.63 |
| always_long | net | 5.96 | -15.26 | 0.40 |
| book_bet | before_costs | -2.58 | -2.90 | -1.43 |
| pred_side_full_size | before_costs | -13.83 | -14.54 | -1.24 |
| placebo_sizes | before_costs | -1.62 | -1.69 | -1.32 |
| random_side | before_costs | 2.94 | -1.82 | 1.18 |
| always_long | before_costs | 14.33 | -12.60 | 0.81 |
| equal_weight_buy_and_hold | none (buy and hold) | 23.98 | -10.02 | 1.16 |

## whole_walk_forward_span: 2024-01-03 to 2026-10-06

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet | net | -7.12 | -7.15 | -1.76 |
| pred_side_full_size | net | -44.21 | -44.39 | -2.01 |
| placebo_sizes | net | -7.43 | -7.47 | -2.51 |
| random_side | net | -38.42 | -38.49 | -7.13 |
| always_long | net | 36.31 | -22.22 | 0.73 |
| book_bet | before_costs | -3.02 | -4.00 | -0.73 |
| pred_side_full_size | before_costs | -19.48 | -23.95 | -0.72 |
| placebo_sizes | before_costs | -2.50 | -3.08 | -0.82 |
| random_side | before_costs | -0.72 | -4.74 | -0.10 |
| always_long | before_costs | 69.22 | -20.25 | 1.17 |
| equal_weight_buy_and_hold | none (buy and hold) | 143.51 | -21.92 | 1.71 |
