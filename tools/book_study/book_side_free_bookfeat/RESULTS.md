# Book side-free path, BOOK-ONLY features: one-year backtest on both venues

Same as the earlier side-free run except the model's inputs are only features that come from the book: ffd (Ch.5), log_trgt (Snippet 3.1 daily vol), sadf (Ch.17), ent_plugin and ent_konto (Ch.18), cs_spread (Ch.19). The earlier run also fed it momentum lookbacks, vol_ratio and a market state, which are not in the book. No side is given; side = predicted class, size = Snippet 10.1/10.3 of its probability; overlapping bets averaged; walk-forward trained only on events labelled before 2024-01-01 (every scored prediction is out of sample). Account, costs and window as in the earlier run (N_ref slots, no leverage, 12 bp/side Kraken, 10 bp/side Alpaca, last 365 UTC days).

## kraken

50,968 events; features ['ffd', 'log_trgt', 'sadf', 'ent_plugin', 'ent_konto', 'cs_spread']; walk-forward AUC 0.5215; side accuracy 0.5078 (share of test events that went up 0.5072).

Last 365 days: 2025-10-13 to 2026-10-12

| variant | costs | gain % | max drawdown % |
|---|---|---:|---:|
| book_bet | net | 0.31 | -2.38 |
| pred_side_full_size | net | -16.44 | -35.07 |
| placebo_sizes | net | -2.16 | -3.09 |
| random_side | net | -27.88 | -29.55 |
| always_long | net | -22.68 | -49.83 |
| book_bet | before_costs | 0.94 | -1.97 |
| pred_side_full_size | before_costs | -10.25 | -31.54 |
| placebo_sizes | before_costs | -0.86 | -2.11 |
| random_side | before_costs | -7.67 | -12.00 |
| always_long | before_costs | -21.52 | -49.35 |
| equal_weight_buy_and_hold | none (buy and hold) | -36.80 | -61.48 |

## alpaca

350,337 events; features ['ffd', 'log_trgt', 'sadf', 'ent_plugin', 'ent_konto', 'cs_spread']; walk-forward AUC 0.5126; side accuracy 0.4969 (share of test events that went up 0.5304).

Last 365 days: 2025-10-07 to 2026-10-06

| variant | costs | gain % | max drawdown % |
|---|---|---:|---:|
| book_bet | net | -0.34 | -0.83 |
| pred_side_full_size | net | -16.77 | -21.52 |
| placebo_sizes | net | -0.37 | -0.44 |
| random_side | net | -13.69 | -13.79 |
| always_long | net | 5.96 | -15.26 |
| book_bet | before_costs | -0.14 | -0.79 |
| pred_side_full_size | before_costs | -8.57 | -17.00 |
| placebo_sizes | before_costs | -0.02 | -0.26 |
| random_side | before_costs | 2.94 | -1.82 |
| always_long | before_costs | 14.33 | -12.60 |
| equal_weight_buy_and_hold | none (buy and hold) | 23.98 | -10.02 |

Checks: verify_book_side_free.py (independent code, own bet size / positions / compounding / benchmark) re-derives 124 numbers per venue: 0 differences on both. One walk-forward path, no CPCV for this variant, flat costs, no slippage; Kraken data hole Apr-Jul 2025 is outside the window.