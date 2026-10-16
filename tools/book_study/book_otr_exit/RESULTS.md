# The book's algo end to end: side-free bets + Chapter 13 optimal-trading-rule exit, one-year backtest on both venues

What trades: CUSUM events (Ch.2) -> a model trained only on book features learns the side (Snippets 3.5-3.7 with no side) and its probability gives the size (Snippets 10.1/10.3) -> the position is closed by the Chapter 13 optimal trading rule (profit-taking and stop-loss in sigma units, chosen from pre-2024 data only by the book's O-U procedure and mesh, Snippet 13.2) -> overlapping bets averaged (Snippet 10.2). Walk-forward: every scored prediction is out of sample.

Account: N_ref equal slots, no leverage, costs on every position change (12 bp/side Kraken, 10 bp/side Alpaca), inactive days in cash, compounded daily, last 365 UTC days of data.

## kraken

OTR chosen from pre-cutoff events: profit-taking 7 sigma, stop-loss 7 sigma, at most 100 bars (phi 0.9972, i.e. almost a random walk; simulated Sharpe at the chosen node 0.117, realised after the cutoff 0.034 per event). Median bars held 47.

Last 365 days: 2025-10-13 to 2026-10-12

| variant | costs | gain % | max drawdown % |
|---|---|---:|---:|
| book_bet_otr | net | 1.10 | -1.96 |
| pred_side_full_size_otr | net | -11.48 | -31.20 |
| placebo_sizes_otr | net | -1.54 | -2.61 |
| random_side_otr | net | -29.03 | -30.53 |
| always_long_otr | net | -20.75 | -50.50 |
| book_bet_otr | before_costs | 1.67 | -1.81 |
| pred_side_full_size_otr | before_costs | -5.79 | -29.42 |
| placebo_sizes_otr | before_costs | -0.32 | -1.80 |
| random_side_otr | before_costs | -11.05 | -14.48 |
| always_long_otr | before_costs | -20.49 | -50.43 |
| equal_weight_buy_and_hold | none (buy and hold) | -36.80 | -61.48 |

## alpaca

OTR chosen from pre-cutoff events: profit-taking 9 sigma, stop-loss 3 sigma, at most 100 bars (phi 0.9998, i.e. almost a random walk; simulated Sharpe at the chosen node 0.081, realised after the cutoff 0.009 per event). Median bars held 27.

Last 365 days: 2025-10-10 to 2026-10-09

| variant | costs | gain % | max drawdown % |
|---|---|---:|---:|
| book_bet_otr | net | -0.29 | -0.97 |
| pred_side_full_size_otr | net | -13.16 | -19.43 |
| placebo_sizes_otr | net | -0.32 | -0.38 |
| random_side_otr | net | -8.86 | -9.16 |
| always_long_otr | net | 14.34 | -11.52 |
| book_bet_otr | before_costs | -0.20 | -0.95 |
| pred_side_full_size_otr | before_costs | -10.59 | -17.43 |
| placebo_sizes_otr | before_costs | -0.11 | -0.26 |
| random_side_otr | before_costs | -0.23 | -3.03 |
| always_long_otr | before_costs | 15.56 | -11.33 |
| equal_weight_buy_and_hold | none (buy and hold) | 22.95 | -10.02 |

## Book-compliance ledger (everything in this algo that is not dictated by the book's text, in one place)

Choices the book leaves to the user (set by this project, not from the book): the dollar-bar size (one average clock-hour of traded value, trailing 30 days); Alpaca rows are hourly because that is the finest data downloaded; the CUSUM threshold (EWM bar sigma x sqrt(bars per day)); triple-barrier width 2 x daily vol and 4-day vertical barrier for the labels; SADF window 100 / lag 1, entropy window 100 / word 2, FFD threshold 0.01; forest of 100 entropy trees, min weight fraction leaf 0.05, at most 10,000 draws per tree; bet-size step 0.1; walk-forward cutoff 2024-01-01; flat round-trip costs; the slot account (N_ref) that turns per-instrument bets into one account.

Book components NOT applied to this backtest: Chapter 16 HRP allocation across instruments (ported and run in the all-snippets study, but the account here uses equal slots); Snippet 10.4 dynamic position size / limit price; Chapter 15 strategy-risk sizing; Chapter 19 trade-level measures; combinatorial purged CV and the deflated Sharpe for this exact variant (one walk-forward path only); calibrating the exit on labels built from the OTR barriers (the labels use the 2 x vol barriers, the exit uses the OTR).

Checks: verify_book_otr_exit.py re-derives 124 numbers per venue from the raw files with its own code (bet size, OTR exits bar by bar, positions, compounding, benchmark): 0 differences on both venues; planted errors in a copy (a moved exit bar, an altered gain) are caught.