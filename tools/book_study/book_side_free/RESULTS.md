# The book's side-free path: one-year backtest on both venues

Followed from the book only (extract text, Snippets 3.5-3.7 with no side given, 10.1-10.3, 10.2 averaging; Ch.4 weights, Ch.6 forest, Ch.7 purged CV code). No strategy from outside the book. The model is NOT given a side: it learns up/down from the CUSUM events' triple-barrier outcomes (bin = sign of the return at the exit), side = predicted class, size = Snippet 10.1 of its probability (step 0.1), overlapping bets averaged. Walk-forward: trained only on events whose labels ended before 2024-01-01; every prediction scored is out of sample.

Account: N_ref equal slots (Kraken 25, Alpaca 194), no leverage, costs on every position change (12 bp per side Kraken, 10 bp per side Alpaca), inactive days in cash, compounded daily. Window = last 365 UTC days of data. 'book_bet' is the book's bet (side x confidence size).

## kraken

50,968 events; walk-forward train 28,279 / test 22,608; AUC 0.5244; side accuracy 0.5137 (share of test events that went up 0.5072).

Last 365 days: 2025-10-13 to 2026-10-12

| variant | costs | gain % | max drawdown % |
|---|---|---:|---:|
| book_bet | net | 2.43 | -3.30 |
| pred_side_full_size | net | 0.22 | -35.23 |
| placebo_sizes | net | 0.18 | -2.70 |
| random_side | net | -27.88 | -29.55 |
| always_long | net | -22.68 | -49.83 |
| book_bet | before_costs | 4.26 | -2.89 |
| pred_side_full_size | before_costs | 16.37 | -28.42 |
| placebo_sizes | before_costs | 2.51 | -2.11 |
| random_side | before_costs | -7.67 | -12.00 |
| always_long | before_costs | -21.52 | -49.35 |
| equal_weight_buy_and_hold | none (buy and hold) | -36.80 | -61.48 |

## alpaca

350,337 events; walk-forward train 252,460 / test 97,650; AUC 0.4956; side accuracy 0.4951 (share of test events that went up 0.5304).

Last 365 days: 2025-10-07 to 2026-10-06

| variant | costs | gain % | max drawdown % |
|---|---|---:|---:|
| book_bet | net | -4.11 | -4.13 |
| pred_side_full_size | net | -24.60 | -24.83 |
| placebo_sizes | net | -3.48 | -3.51 |
| random_side | net | -13.69 | -13.79 |
| always_long | net | 5.96 | -15.26 |
| book_bet | before_costs | -2.58 | -2.90 |
| pred_side_full_size | before_costs | -13.83 | -14.54 |
| placebo_sizes | before_costs | -1.62 | -1.69 |
| random_side | before_costs | 2.94 | -1.82 |
| always_long | before_costs | 14.33 | -12.60 |
| equal_weight_buy_and_hold | none (buy and hold) | 23.98 | -10.02 |

Variants: pred_side_full_size = same predicted side at size 1; placebo_sizes = the book_bet sizes shuffled among the same events; random_side = seeded coin per event at size 1; always_long = every event long at size 1 (references only); equal_weight_buy_and_hold = same instruments, context only.

Checks: verify_book_side_free.py re-derives 124 numbers per venue from the raw files with its own code (own bet-size function, own position loop, own compounding, own benchmark, label and side accuracy from the bars): 0 differences on both venues; planted errors in a copy (a flipped probability, an altered gain) are caught.
Limits: one walk-forward path (a single trajectory, no confidence interval); CPCV was not run for this variant; Alpaca bars are close to hourly; Kraken data hole Apr-Jul 2025 is outside the window; flat costs, no slippage or borrow model; events with a long return of exactly 0 are dropped (Kraken 34, Alpaca 703).