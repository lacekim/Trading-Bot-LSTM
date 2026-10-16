# Book side-free path, kraken: one-year backtest (2024-01-01 .. 2026-10-12 out-of-sample)

Walk-forward AUC 0.5244; side accuracy 0.5137 (share of test events that went up 0.5072). N_ref = 25 slots; costs 12 bp per side. Trees 100.

## last_365_days: 2025-10-13 to 2026-10-12

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet | net | 2.43 | -3.30 | 0.60 |
| pred_side_full_size | net | 0.22 | -35.23 | 0.17 |
| placebo_sizes | net | 0.18 | -2.70 | 0.08 |
| random_side | net | -27.88 | -29.55 | -3.10 |
| always_long | net | -22.68 | -49.83 | -0.27 |
| book_bet | before_costs | 4.26 | -2.89 | 1.02 |
| pred_side_full_size | before_costs | 16.37 | -28.42 | 0.62 |
| placebo_sizes | before_costs | 2.51 | -2.11 | 0.88 |
| random_side | before_costs | -7.67 | -12.00 | -0.71 |
| always_long | before_costs | -21.52 | -49.35 | -0.24 |
| equal_weight_buy_and_hold | none (buy and hold) | -36.80 | -61.48 | -0.51 |

## whole_walk_forward_span: 2024-01-01 to 2026-10-12

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet | net | 3.56 | -4.64 | 0.31 |
| pred_side_full_size | net | 24.69 | -35.23 | 0.40 |
| placebo_sizes | net | 1.08 | -3.48 | 0.14 |
| random_side | net | -46.13 | -48.44 | -2.06 |
| always_long | net | -18.04 | -73.13 | 0.15 |
| book_bet | before_costs | 7.83 | -4.27 | 0.63 |
| pred_side_full_size | before_costs | 77.04 | -28.42 | 0.75 |
| placebo_sizes | before_costs | 6.90 | -2.11 | 0.81 |
| random_side | before_costs | -1.14 | -18.49 | 0.01 |
| always_long | before_costs | -15.06 | -72.66 | 0.17 |
| equal_weight_buy_and_hold | none (buy and hold) | -47.09 | -82.80 | -0.05 |
