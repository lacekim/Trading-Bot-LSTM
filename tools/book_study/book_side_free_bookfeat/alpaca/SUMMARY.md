# Book side-free path, alpaca: one-year backtest (2024-01-03 .. 2026-10-06 out-of-sample)

Walk-forward AUC 0.5126; side accuracy 0.4969 (share of test events that went up 0.5304). N_ref = 194 slots; costs 10 bp per side. Trees 100.

## last_365_days: 2025-10-07 to 2026-10-06

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet | net | -0.34 | -0.83 | -0.47 |
| pred_side_full_size | net | -16.77 | -21.52 | -1.15 |
| placebo_sizes | net | -0.37 | -0.44 | -1.60 |
| random_side | net | -13.69 | -13.79 | -5.63 |
| always_long | net | 5.96 | -15.26 | 0.40 |
| book_bet | before_costs | -0.14 | -0.79 | -0.19 |
| pred_side_full_size | before_costs | -8.57 | -17.00 | -0.53 |
| placebo_sizes | before_costs | -0.02 | -0.26 | -0.09 |
| random_side | before_costs | 2.94 | -1.82 | 1.18 |
| always_long | before_costs | 14.33 | -12.60 | 0.81 |
| equal_weight_buy_and_hold | none (buy and hold) | 23.98 | -10.02 | 1.16 |

## whole_walk_forward_span: 2024-01-03 to 2026-10-06

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet | net | -1.06 | -1.15 | -0.72 |
| pred_side_full_size | net | -40.05 | -40.62 | -1.41 |
| placebo_sizes | net | -1.23 | -1.23 | -2.35 |
| random_side | net | -38.42 | -38.49 | -7.13 |
| always_long | net | 36.31 | -22.22 | 0.73 |
| book_bet | before_costs | -0.65 | -0.96 | -0.44 |
| pred_side_full_size | before_costs | -21.44 | -26.83 | -0.63 |
| placebo_sizes | before_costs | -0.29 | -0.33 | -0.56 |
| random_side | before_costs | -0.72 | -4.74 | -0.10 |
| always_long | before_costs | 69.22 | -20.25 | 1.17 |
| equal_weight_buy_and_hold | none (buy and hold) | 143.51 | -21.92 | 1.71 |
