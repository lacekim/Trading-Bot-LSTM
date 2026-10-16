# One-year backtest, alpaca (2024-01-03 .. 2026-10-06 out-of-sample)

N_ref = 194 slots. Costs 10 bp per side. Candidates 351,040 events on 494 instruments. Equity starts at 100% and compounds daily; inactive days earn 0.

## last_365_days: 2025-10-07 to 2026-10-06 (365 days)

| variant | convention | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) | active days | mean active instruments |
|---|---|---|---:|---:|---:|---:|---:|
| primary | slot | net | -6.69 | -11.80 | -0.58 | 276 | 171.7 |
| meta | slot | net | -0.67 | -0.68 | -1.52 | 276 | 171.7 |
| placebo | slot | net | -0.49 | -0.56 | -1.47 | 276 | 171.7 |
| primary | slot | before_costs | 1.74 | -8.59 | 0.21 | 276 | 171.7 |
| meta | slot | before_costs | -0.26 | -0.42 | -0.60 | 276 | 171.7 |
| placebo | slot | before_costs | 0.15 | -0.23 | 0.46 | 276 | 171.7 |
| primary | per_active | net | -3.82 | -10.98 | -0.25 | 276 | 171.7 |
| meta | per_active | net | -0.38 | -0.58 | -0.64 | 276 | 171.7 |
| placebo | per_active | net | -0.61 | -0.78 | -1.53 | 276 | 171.7 |
| primary | per_active | before_costs | 5.73 | -7.85 | 0.51 | 276 | 171.7 |
| meta | per_active | before_costs | 0.04 | -0.39 | 0.08 | 276 | 171.7 |
| placebo | per_active | before_costs | 0.07 | -0.28 | 0.17 | 276 | 171.7 |
| equal_weight_buy_and_hold | - | none (buy and hold) | 23.98 | -10.02 | 1.16 |  |  |

## whole_walk_forward_span: 2024-01-03 to 2026-10-06 (1008 days)

| variant | convention | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) | active days | mean active instruments |
|---|---|---|---:|---:|---:|---:|---:|
| primary | slot | net | -13.73 | -21.38 | -0.45 | 763 | 169.8 |
| meta | slot | net | -0.37 | -0.68 | -0.33 | 763 | 169.8 |
| placebo | slot | net | -1.11 | -1.26 | -1.30 | 763 | 169.8 |
| primary | slot | before_costs | 9.93 | -15.32 | 0.38 | 763 | 169.8 |
| meta | slot | before_costs | 0.82 | -0.42 | 0.75 | 763 | 169.8 |
| placebo | slot | before_costs | 0.62 | -0.31 | 0.72 | 763 | 169.8 |
| primary | per_active | net | 0.15 | -16.36 | 0.06 | 763 | 169.8 |
| meta | per_active | net | 0.03 | -0.58 | 0.02 | 763 | 169.8 |
| placebo | per_active | net | -0.88 | -0.99 | -0.88 | 763 | 169.8 |
| primary | per_active | before_costs | 30.76 | -13.73 | 0.87 | 763 | 169.8 |
| meta | per_active | before_costs | 1.30 | -0.39 | 0.93 | 763 | 169.8 |
| placebo | per_active | before_costs | 0.97 | -0.34 | 0.97 | 763 | 169.8 |
| equal_weight_buy_and_hold | - | none (buy and hold) | 143.51 | -21.92 | 1.71 |  |  |
