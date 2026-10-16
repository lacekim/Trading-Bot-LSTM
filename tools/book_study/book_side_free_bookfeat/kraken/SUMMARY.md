# Book side-free path, kraken: one-year backtest (2024-01-01 .. 2026-10-12 out-of-sample)

Walk-forward AUC 0.5215; side accuracy 0.5078 (share of test events that went up 0.5072). N_ref = 25 slots; costs 12 bp per side. Trees 100.

## last_365_days: 2025-10-13 to 2026-10-12

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet | net | 0.31 | -2.38 | 0.16 |
| pred_side_full_size | net | -16.44 | -35.07 | -0.43 |
| placebo_sizes | net | -2.16 | -3.09 | -1.51 |
| random_side | net | -27.88 | -29.55 | -3.10 |
| always_long | net | -22.68 | -49.83 | -0.27 |
| book_bet | before_costs | 0.94 | -1.97 | 0.46 |
| pred_side_full_size | before_costs | -10.25 | -31.54 | -0.19 |
| placebo_sizes | before_costs | -0.86 | -2.11 | -0.60 |
| random_side | before_costs | -7.67 | -12.00 | -0.71 |
| always_long | before_costs | -21.52 | -49.35 | -0.24 |
| equal_weight_buy_and_hold | none (buy and hold) | -36.80 | -61.48 | -0.51 |

## whole_walk_forward_span: 2024-01-01 to 2026-10-12

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet | net | 3.21 | -3.66 | 0.46 |
| pred_side_full_size | net | 19.38 | -39.92 | 0.36 |
| placebo_sizes | net | -1.15 | -3.09 | -0.24 |
| random_side | net | -46.13 | -48.44 | -2.06 |
| always_long | net | -18.04 | -73.13 | 0.15 |
| book_bet | before_costs | 4.82 | -3.41 | 0.68 |
| pred_side_full_size | before_costs | 41.95 | -37.88 | 0.53 |
| placebo_sizes | before_costs | 2.09 | -2.11 | 0.45 |
| random_side | before_costs | -1.14 | -18.49 | 0.01 |
| always_long | before_costs | -15.06 | -72.66 | 0.17 |
| equal_weight_buy_and_hold | none (buy and hold) | -47.09 | -82.80 | -0.05 |
