# The whole book as one strategy: one-year backtest on both venues

What trades, in the book's own order: dollar bars and the CUSUM filter pick the decision points (Ch.2); labels are built with no side given and the model learns side and size (Snippets 3.5-3.7, side=None); sample weights from uniqueness and attribution, sequential bootstrap (Ch.4); features: FFD, SADF, entropy (plug-in, Kontoyiannis), Corwin-Schultz spread, daily volatility (Ch.5, 17-19, Snippet 3.1); bagged entropy forest (Ch.6) scored by combinatorial purged CV with 5 backtest paths and by walk-forward trained before 2024-01-01 only (Ch.7, 12); side = predicted class and size = Snippets 10.1/10.3 of its probability, overlapping bets averaged (10.2); every bet is closed by the Chapter 13 optimal trading rule fitted on pre-2024 data only (Snippets 13.1/13.2); the account is allocated across instruments by Hierarchical Risk Parity (Snippet 16.4) re-estimated every calendar quarter from the trailing 252 trading days of returns (nothing after the rebalance day is used).

Account: an instrument's slice = its HRP weight (weights sum to 1, no leverage); the position inside the slice is the averaged signal in [-1, 1]; daily return = sum of weight x (gross - cost); costs on every position change (12 bp per side Kraken, 10 bp per side Alpaca); compounded daily; instruments without enough history sit in cash. 'equal' below is the same account with 1/N weights over the same instruments (isolates the HRP step).

## kraken

Last 365 days: 2025-10-13 to 2026-10-12. Events 50,968; walk-forward AUC 0.5215, side accuracy 0.5078 (share of test events that went up 0.5072); CPCV AUC by path 0.5313, 0.5169, 0.5170, 0.5133, 0.5150. OTR: profit-taking 7 sigma, stop-loss 7 sigma, at most 100 bars. HRP rebalances: 51.

| variant (net of costs) | HRP gain % | HRP max drawdown % | HRP Sharpe | equal-weight gain % | equal-weight max drawdown % |
|---|---:|---:|---:|---:|---:|
| book_bet | -1.38 | -2.25 | -0.98 | 0.95 | -1.95 |
| pred_side_full_size | -33.81 | -39.46 | -1.91 | -13.19 | -31.69 |
| placebo_sizes | -2.19 | -2.94 | -1.99 | -1.73 | -2.69 |
| always_long | 18.42 | -23.11 | 0.73 | -16.50 | -47.83 |
| random_side | -12.09 | -24.56 | -0.91 | -27.73 | -29.28 |

Equal-weight buy and hold of the same instruments over the same year (earlier OTR run): -36.80 % gain, -61.48 % max drawdown.

CPCV, five independent backtest paths, same year, book_bet, HRP, net: gains -1.19, -2.19, -1.12, -1.13, -1.46 % (min -2.19, max -1.12).

Bet sizes: of 22,608 walk-forward predictions, 58.1 % get size 0 from the book's bet-size function (median confidence 0.5274, 95th percentile 0.5721, largest size 0.2); mean size 0.0431.

Whole walk-forward span 2024-01-01 to 2026-10-12, book_bet, HRP, net: -1.45 % gain, -2.25 % max drawdown.

Latest HRP weights (rebalance 2026-10-01): TRX 0.490, PAXG 0.102, ANKR 0.036, BNB 0.035, ATOM 0.032; 25 instruments weighted.

## alpaca

Last 365 days: 2025-10-10 to 2026-10-09. Events 350,337; walk-forward AUC 0.5126, side accuracy 0.4969 (share of test events that went up 0.5304); CPCV AUC by path 0.5108, 0.5141, 0.5118, 0.5084, 0.5132. OTR: profit-taking 9 sigma, stop-loss 3 sigma, at most 100 bars. HRP rebalances: 42.

| variant (net of costs) | HRP gain % | HRP max drawdown % | HRP Sharpe | equal-weight gain % | equal-weight max drawdown % |
|---|---:|---:|---:|---:|---:|
| book_bet | 0.01 | -0.07 | 0.10 | -0.12 | -0.41 |
| pred_side_full_size | -0.54 | -1.60 | -0.29 | -5.30 | -8.25 |
| placebo_sizes | -0.15 | -0.15 | -4.00 | -0.13 | -0.16 |
| always_long | 2.81 | -3.33 | 0.79 | 6.14 | -4.83 |
| random_side | -4.18 | -4.28 | -3.72 | -3.72 | -3.85 |

Equal-weight buy and hold of the same instruments over the same year (earlier OTR run): 22.95 % gain, -10.02 % max drawdown.

CPCV, five independent backtest paths, same year, book_bet, HRP, net: gains 0.02, 0.01, -0.04, -0.00, 0.02 % (min -0.04, max 0.02).

Bet sizes: of 97,650 walk-forward predictions, 86.5 % get size 0 from the book's bet-size function (median confidence 0.5132, 95th percentile 0.5443, largest size 0.1); mean size 0.0135.

Whole walk-forward span 2024-01-01 to 2026-10-09, book_bet, HRP, net: -0.08 % gain, -0.14 % max drawdown.

Latest HRP weights (rebalance 2026-10-01): IEF 0.107, O 0.025, SPY 0.023, TJX 0.014, BRK.B 0.013; 493 instruments weighted.

## Book-compliance ledger

Choices the book leaves to the user (fixed earlier in this project, not from the book): dollar-bar size (one average clock-hour of traded value, trailing 30 days); Alpaca rows are hourly (the finest data downloaded); the CUSUM threshold; triple-barrier width 2 x daily vol and 4-day vertical barrier for the labels; SADF window 100 / lag 1, entropy window 100 / word 2, FFD threshold 0.01; forest of 100 entropy trees, min weight fraction leaf 0.05, at most 10,000 draws per tree; bet-size step 0.1; walk-forward cutoff 2024-01-01; flat round-trip costs; the HRP re-estimation window (252 trading days) and quarterly cadence; instruments need 80 % coverage in the window to get a weight.

Book code used as written: Snippet 16.4's functions as ported (note: the book's code passes the square correlation-distance matrix to scipy's linkage, which treats its rows as observations; scipy warns about it; the checker does the same on purpose).

Book components NOT applied: Snippet 10.4 dynamic position size / limit price; Chapter 15 strategy-risk sizing; Chapter 19 trade-level measures; the deflated Sharpe; exit calibration on labels built from the OTR barriers (labels use the 2 x vol barriers, the exit uses the OTR).

References in the tables are not part of the strategy: always_long (every event long, same exits and allocation), random_side, placebo_sizes (the book_bet sizes shuffled), pred_side_full_size (same sides, size 1).
## Checks

verify_book_full.py re-derives from the raw files with its own code (bet size, exits located by timestamp, positions, day booking, HRP including its own quasi-diagonalisation and recursive bisection, quarterly schedule, compounding): Kraken 775 numbers (400 HRP weights among them), Alpaca 16,286 numbers (15,911 HRP weights among them), 0 differences on both. Planted errors in the checker's own calculation (bet size +1 %, costs +10 %, one HRP weight +5 %, exit thresholds 3 % off) are each caught on Kraken (112 / 180 / 700 / 336 problems); the exit error is also caught on Alpaca (360 problems). The CPCV prediction file for Alpaca is gzipped for size; gunzip it to cpcv_predictions.csv to re-run the checker.

Reading across the earlier tables: the slot account in book_otr_exit (equal slots) and the HRP account here are different accounts, so their gains differ for the same bets; compare variants within one table only.
