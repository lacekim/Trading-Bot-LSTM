# One-year backtest, kraken (2024-01-01 .. 2026-10-12 out-of-sample)

N_ref = 25 slots. Costs 12 bp per side. Candidates 51,002 events on 27 instruments. Equity starts at 100% and compounds daily; inactive days earn 0.

## last_365_days: 2025-10-13 to 2026-10-12 (365 days)

| variant | convention | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) | active days | mean active instruments |
|---|---|---|---:|---:|---:|---:|---:|
| primary | slot | net | -1.08 | -31.80 | 0.16 | 365 | 22.3 |
| meta | slot | net | 0.88 | -0.97 | 0.51 | 365 | 22.3 |
| placebo | slot | net | -0.61 | -1.49 | -0.41 | 365 | 22.3 |
| primary | slot | before_costs | 4.66 | -29.27 | 0.31 | 365 | 22.3 |
| meta | slot | before_costs | 1.46 | -0.76 | 0.83 | 365 | 22.3 |
| placebo | slot | before_costs | 0.55 | -1.09 | 0.38 | 365 | 22.3 |
| primary | per_active | net | 10.06 | -32.33 | 0.44 | 365 | 22.3 |
| meta | per_active | net | 1.24 | -1.04 | 0.62 | 365 | 22.3 |
| placebo | per_active | net | -0.40 | -1.51 | -0.25 | 365 | 22.3 |
| primary | per_active | before_costs | 17.17 | -29.78 | 0.59 | 365 | 22.3 |
| meta | per_active | before_costs | 1.91 | -0.82 | 0.94 | 365 | 22.3 |
| placebo | per_active | before_costs | 0.89 | -1.11 | 0.56 | 365 | 22.3 |
| equal_weight_buy_and_hold | - | none (buy and hold) | -36.80 | -61.48 | -0.51 |  |  |

## whole_walk_forward_span: 2024-01-01 to 2026-10-12 (1016 days)

| variant | convention | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) | active days | mean active instruments |
|---|---|---|---:|---:|---:|---:|---:|
| primary | slot | net | -7.54 | -41.81 | 0.15 | 922 | 22.0 |
| meta | slot | net | 2.41 | -2.86 | 0.38 | 922 | 22.0 |
| placebo | slot | net | -2.25 | -3.11 | -0.52 | 922 | 22.0 |
| primary | slot | before_costs | 5.90 | -38.56 | 0.26 | 922 | 22.0 |
| meta | slot | before_costs | 4.23 | -2.68 | 0.65 | 922 | 22.0 |
| placebo | slot | before_costs | 0.56 | -2.04 | 0.14 | 922 | 22.0 |
| primary | per_active | net | 18.83 | -41.76 | 0.36 | 922 | 22.0 |
| meta | per_active | net | 4.34 | -2.91 | 0.60 | 922 | 22.0 |
| placebo | per_active | net | -1.99 | -3.11 | -0.41 | 922 | 22.0 |
| primary | per_active | before_costs | 39.52 | -38.51 | 0.49 | 922 | 22.0 |
| meta | per_active | before_costs | 6.49 | -2.73 | 0.88 | 922 | 22.0 |
| placebo | per_active | before_costs | 1.31 | -2.04 | 0.28 | 922 | 22.0 |
| equal_weight_buy_and_hold | - | none (buy and hold) | -47.09 | -82.80 | -0.05 |  |  |
