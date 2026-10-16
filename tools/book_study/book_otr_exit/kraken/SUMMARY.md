# Book OTR exit + side-free book bet, kraken: one-year backtest (2024-01-01 .. 2026-10-12 out-of-sample)

OTR from Chapter 13 (pre-cutoff events only): profit-taking 7 sigma, stop-loss 7 sigma, at most 100 bars; phi 0.9972. N_ref = 25 slots; costs 12 bp per side.

## last_365_days: 2025-10-13 to 2026-10-12

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet_otr | net | 1.10 | -1.96 | 0.54 |
| pred_side_full_size_otr | net | -11.48 | -31.20 | -0.24 |
| placebo_sizes_otr | net | -1.54 | -2.61 | -1.05 |
| random_side_otr | net | -29.03 | -30.53 | -3.39 |
| always_long_otr | net | -20.75 | -50.50 | -0.21 |
| book_bet_otr | before_costs | 1.67 | -1.81 | 0.81 |
| pred_side_full_size_otr | before_costs | -5.79 | -29.42 | -0.04 |
| placebo_sizes_otr | before_costs | -0.32 | -1.80 | -0.21 |
| random_side_otr | before_costs | -11.05 | -14.48 | -1.11 |
| always_long_otr | before_costs | -20.49 | -50.43 | -0.20 |
| equal_weight_buy_and_hold | none (buy and hold) | -36.80 | -61.48 | -0.51 |

## whole_walk_forward_span: 2024-01-01 to 2026-10-12

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet_otr | net | 5.15 | -3.40 | 0.74 |
| pred_side_full_size_otr | net | 45.56 | -40.26 | 0.56 |
| placebo_sizes_otr | net | -0.20 | -2.61 | -0.04 |
| random_side_otr | net | -50.60 | -54.46 | -2.40 |
| always_long_otr | net | -19.41 | -74.33 | 0.14 |
| book_bet_otr | before_costs | 6.61 | -3.16 | 0.94 |
| pred_side_full_size_otr | before_costs | 68.51 | -38.58 | 0.71 |
| placebo_sizes_otr | before_costs | 2.79 | -1.80 | 0.61 |
| random_side_otr | before_costs | -14.37 | -28.98 | -0.48 |
| always_long_otr | before_costs | -18.73 | -74.24 | 0.15 |
| equal_weight_buy_and_hold | none (buy and hold) | -47.09 | -82.80 | -0.05 |
