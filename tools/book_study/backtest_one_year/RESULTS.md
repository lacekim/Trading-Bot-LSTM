# One-year percentage-gain backtest of the book-built system (both venues)

What it is: the meta-labeling system built from the book (CUSUM events, triple barrier, side = sign of the trailing 720-bar return, meta model bet sizes, bar-by-bar averaging of overlapping bets), run on each venue's dollar bars. All predictions are walk-forward (out of sample).

Assumptions (mine, since 'it' was not specified): the account is split into N_ref equal slots (Kraken 25, Alpaca 194 = median number of instruments with an active candidate on active days); each day's return = sum of net results of the active instruments / max(n active, N_ref); inactive days = cash; no leverage; compounded daily; costs 12 bp per side on Kraken, 10 bp per side on Alpaca on every position change. Window = the last 365 UTC days of data. 'placebo' = the same meta sizes shuffled among the same events. The equal-weight buy-and-hold line is context only.

A first pass used the whole account split over only the instruments active that day. On Alpaca that put the entire account into 1-3 stocks on 24 stray Saturday bars, so I switched the headline to the slot convention; the old convention is kept in the tables ('per_active') for comparison with the earlier averaged-bet numbers.

## kraken: last_365_days: 2025-10-13 to 2026-10-12

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| primary | net | -1.08 | -31.80 | 0.16 |
| meta | net | 0.88 | -0.97 | 0.51 |
| placebo | net | -0.61 | -1.49 | -0.41 |
| primary | before_costs | 4.66 | -29.27 | 0.31 |
| meta | before_costs | 1.46 | -0.76 | 0.83 |
| placebo | before_costs | 0.55 | -1.09 | 0.38 |
| equal_weight_buy_and_hold | none (buy and hold) | -36.80 | -61.48 | -0.51 |

## kraken: whole_walk_forward_span: 2024-01-01 to 2026-10-12

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| primary | net | -7.54 | -41.81 | 0.15 |
| meta | net | 2.41 | -2.86 | 0.38 |
| placebo | net | -2.25 | -3.11 | -0.52 |
| primary | before_costs | 5.90 | -38.56 | 0.26 |
| meta | before_costs | 4.23 | -2.68 | 0.65 |
| placebo | before_costs | 0.56 | -2.04 | 0.14 |
| equal_weight_buy_and_hold | none (buy and hold) | -47.09 | -82.80 | -0.05 |

## alpaca: last_365_days: 2025-10-07 to 2026-10-06

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| primary | net | -6.69 | -11.80 | -0.58 |
| meta | net | -0.67 | -0.68 | -1.52 |
| placebo | net | -0.49 | -0.56 | -1.47 |
| primary | before_costs | 1.74 | -8.59 | 0.21 |
| meta | before_costs | -0.26 | -0.42 | -0.60 |
| placebo | before_costs | 0.15 | -0.23 | 0.46 |
| equal_weight_buy_and_hold | none (buy and hold) | 23.98 | -10.02 | 1.16 |

## alpaca: whole_walk_forward_span: 2024-01-03 to 2026-10-06

| variant | costs | gain % | max drawdown % | Sharpe (daily x sqrt365) |
|---|---|---:|---:|---:|
| primary | net | -13.73 | -21.38 | -0.45 |
| meta | net | -0.37 | -0.68 | -0.33 |
| placebo | net | -1.11 | -1.26 | -1.30 |
| primary | before_costs | 9.93 | -15.32 | 0.38 |
| meta | before_costs | 0.82 | -0.42 | 0.75 |
| placebo | before_costs | 0.62 | -0.31 | 0.72 |
| equal_weight_buy_and_hold | none (buy and hold) | 143.51 | -21.92 | 1.71 |

Per-active convention, last 365 days, net of costs (gain %):

- kraken: primary 10.06, meta 1.24, placebo -0.40
- alpaca: primary -3.82, meta -0.38, placebo -0.61

Checks: verify_backtest_one_year.py re-derives all 76 numbers per venue from the raw files with its own code (own bet-size function, own position loop, own compounding, own benchmark): 0 differences on both venues; planted errors in a copy of the summary are caught. Monthly gains are in monthly_gain_pct.csv, daily equity in equity_curves.csv.
Limits: one walk-forward path (a single trajectory, no confidence interval); Alpaca bars are close to hourly; Kraken has a data hole Apr-Jul 2025 (outside the last-year window); costs are the declared flat bp, no slippage or borrow model.