# Book OTR exit + side-free book bet, alpaca: one-year backtest (2024-01-03 .. 2026-10-09 out-of-sample)

OTR from Chapter 13 (pre-cutoff events only): profit-taking 9 sigma, stop-loss 3 sigma, at most 100 bars; phi 0.9998. N_ref = 200 slots; costs 10 bp per side.

## last_365_days: 2025-10-10 to 2026-10-09

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet_otr | net | -0.29 | -0.97 | -0.36 |
| pred_side_full_size_otr | net | -13.16 | -19.43 | -0.79 |
| placebo_sizes_otr | net | -0.32 | -0.38 | -1.39 |
| random_side_otr | net | -8.86 | -9.16 | -1.78 |
| always_long_otr | net | 14.34 | -11.52 | 0.77 |
| book_bet_otr | before_costs | -0.20 | -0.95 | -0.24 |
| pred_side_full_size_otr | before_costs | -10.59 | -17.43 | -0.61 |
| placebo_sizes_otr | before_costs | -0.11 | -0.26 | -0.48 |
| random_side_otr | before_costs | -0.23 | -3.03 | -0.02 |
| always_long_otr | before_costs | 15.56 | -11.33 | 0.83 |
| equal_weight_buy_and_hold | none (buy and hold) | 22.95 | -10.02 | 1.12 |

## whole_walk_forward_span: 2024-01-03 to 2026-10-09

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| book_bet_otr | net | -0.94 | -1.42 | -0.57 |
| pred_side_full_size_otr | net | -31.68 | -34.54 | -0.94 |
| placebo_sizes_otr | net | -0.83 | -0.85 | -1.55 |
| random_side_otr | net | -26.83 | -27.27 | -2.41 |
| always_long_otr | net | 62.59 | -22.24 | 1.03 |
| book_bet_otr | before_costs | -0.74 | -1.33 | -0.45 |
| pred_side_full_size_otr | before_costs | -25.53 | -30.79 | -0.71 |
| placebo_sizes_otr | before_costs | -0.25 | -0.39 | -0.47 |
| random_side_otr | before_costs | -6.37 | -9.10 | -0.49 |
| always_long_otr | before_costs | 67.35 | -21.99 | 1.08 |
| equal_weight_buy_and_hold | none (buy and hold) | 142.69 | -21.92 | 1.70 |
