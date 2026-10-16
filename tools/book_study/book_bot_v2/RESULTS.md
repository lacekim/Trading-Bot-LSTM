# The book bot: one integrated run per venue, from dollar bars to the Chapter 14 and 15 statistics

One command per venue runs every book component that can be computed from our data, in the book's order, with no side supplied by anything outside the book: dollar bars and CUSUM events (Ch.2); triple-barrier labels with side=None (Snippets 3.1-3.7, bin = sign of the return) and Snippet 3.8 dropLabels; uniqueness and attribution weights, sequential bootstrap and Snippet 4.11 time decay (Ch.4); twelve features: FFD (Ch.5), Snippet 3.1 volatility, SADF (Ch.17), plug-in and Kontoyiannis entropy (Ch.18), Corwin-Schultz spread and Becker-Parkinson volatility (19.1-19.2), Roll, Kyle / Amihud / Hasbrouck lambdas (Eq.82-92) and Chu-Stinchcombe-White CUSUM (17.3.2); the Chapter 13 optimal trading rule on a 20 x 20 mesh with 100,000 simulated paths; MDI / MDA / SFI / orthogonal features / weighted tau (Ch.8); the randomised purged-CV search (Ch.9); bagged entropy forest with sequential bootstrap (Ch.6); walk-forward and combinatorial purged CV (Ch.7, 12); bet size and averaging (Snippets 10.1-10.3); HRP allocation (Snippet 16.4); PSR / deflated Sharpe, bet timing, holding period, HHI, drawdown and time under water, annualised return (Ch.14); binHR, binFreq, binSR, probFailure and the t-value of precision (Ch.15); the Snippet 10.4 order ticket; the rebalance cost, spread cost and volume rule of section 2.4.1; the Eq.77-78 concentration of every HRP allocation. The last section lists every item of the book's Contents and what the bot does with it.

## Result (net of costs, HRP allocation, walk-forward trained before 2024-01-01)

| venue | last 365 days: gain % | max drawdown % | whole test span: gain % | max drawdown % | Eq.49 annualised % (last 365) |
|---|---:|---:|---:|---:|---:|
| kraken (2025-10-13 to 2026-10-12) | -2.11 | -2.84 | -4.00 (2024-01-01 to 2026-10-12) | -4.76 | -2.11 |
| alpaca (2025-10-10 to 2026-10-09) | -0.74 | -0.77 | -1.90 (2024-01-01 to 2026-10-09) | -1.90 | -0.74 |

Combinatorial purged CV paths, last 365 days (gain % / max drawdown %):

| venue | path 0 | path 1 | path 2 | path 3 | path 4 |
|---|---:|---:|---:|---:|---:|
| kraken | -1.58 / -2.37 | -1.88 / -2.69 | -2.29 / -3.09 | -1.97 / -2.78 | -1.99 / -2.71 |
| alpaca | -0.72 / -0.74 | -0.70 / -0.72 | -0.82 / -0.86 | -0.74 / -0.78 | -0.74 / -0.78 |

## Correction to the earlier results

The earlier delivered runs (`book_final`) were not side-free in two places. Their Chapter 13 exit rule was fitted on paths signed by a stand-in side (the sign of the trailing 720-bar return), and the event set dropped every event whose stand-in side was zero, whose 720-bar lookback crossed a data hole, or that lacked a 2,160-bar momentum or a 200-day market-state value. None of that is in the book. This bot removes all of it: the label has no side, the Chapter 13 fit uses the long price path, and an event is dropped only when one of its own twelve features, its volatility or its label is missing. The numbers below are the old run against the new run, both read from saved files.

| venue | | old (book_final) | new (this bot) |
|---|---|---:|---:|
| kraken | events | 50968 | 63,979 |
| kraken | Ch.13 chosen pt / sl | 7.0 / 7.0 | 3.5 / 10.0 |
| kraken | walk-forward AUC | 0.5085 | 0.5063 |
| kraken | last 365 days gain % (max DD %) | 0.50 (-0.60) | -2.11 (-2.84) |
| kraken | whole span gain % (max DD %) | -0.90 (-1.75) | -4.00 (-4.76) |
| alpaca | events | 350337 | 352,005 |
| alpaca | Ch.13 chosen pt / sl | 9.0 / 3.0 | 10.0 / 9.5 |
| alpaca | walk-forward AUC | 0.5130 | 0.5007 |
| alpaca | last 365 days gain % (max DD %) | 0.01 (-0.01) | -0.74 (-0.77) |
| alpaca | whole span gain % (max DD %) | -0.02 (-0.03) | -1.90 (-1.90) |

## kraken

### Events (Ch.2-4)

27 symbols, 66,472 CUSUM events, 64,028 kept with all twelve features and a label, 63,979 after Snippet 3.8 (dropped label classes: [0]; shares before: -1: 0.4821, 0: 0.0008, 1: 0.5172). Snippet 4.11 time decay at the snippet default c = 1.0: constant 1.0, slope 0.0 (no decay).

### Chapter 13 optimal trading rule (20 x 20 mesh, 100,000 paths)

Equations 13.5-13.7 fitted on 37,035 pre-cutoff events (long paths, K = 100): phi = 0.9969, half-life 223.3 bars, forecast 1.785 sigma. Step 5a chose profit-taking 3.5 / stop-loss 10.0 (simulated Sharpe 0.0697 against a mesh median of 0.0254). Definition 2 on 26,738 post-cutoff events: Sharpe of the chosen rule -0.0326, median over the mesh -0.0179, best node in hindsight 0.0080; overfit by the book's definition: **True**. The vectorised mesh was cross-checked against the verbatim Snippet 13.2 loop on 3,000 paths (4 nodes): see `ch13.json`.

### Chapter 8 feature importance (37,049 events with labels ending before 2024-01-01)

| feature | MDI | MDA | SFI |
|---|---:|---:|---:|
| ffd | 0.0822 | -0.0003 +/- 0.0002 | -0.69853 |
| log_trgt | 0.0826 | 0.0005 +/- 0.0003 | -0.69520 |
| sadf | 0.1281 | 0.0006 +/- 0.0005 | -0.69460 |
| ent_plugin | 0.0773 | 0.0002 +/- 0.0004 | -0.69620 |
| ent_konto | 0.0442 | -0.0001 +/- 0.0001 | -0.69750 |
| cs_spread | 0.0460 | 0.0002 +/- 0.0001 | -0.69548 |
| bp_sigma | 0.0598 | 0.0002 +/- 0.0003 | -0.69629 |
| roll | 0.0661 | 0.0001 +/- 0.0002 | -0.69734 |
| kyle | 0.0385 | 0.0000 +/- 0.0001 | -0.69709 |
| amihud | nan | 0.0000 +/- 0.0000 | -0.69505 |
| hasbrouck | 0.0659 | 0.0001 +/- 0.0002 | -0.69564 |
| csw | 0.3094 | 0.0070 +/- 0.0011 | -0.68679 |

A blank (nan) MDI means no tree ever split on that feature: Snippet 8.2 replaces zero importances by NaN before averaging.

Orthogonal features (8.5): 10 principal components carry 95% of the variance; weighted Kendall tau between MDI and the inverse PCA rank (8.6) = -0.3615.

### Chapter 9 search (purged 3-fold, neg log-loss)

| candidate | min_weight_fraction_leaf | max_features | OOS neg log-loss |
|---:|---:|---|---:|
| 0 (untuned) (chosen) | 0.0500 | sqrt | -0.68479 |
| 1 | 0.0155 | 12 | -0.68563 |
| 2 | 0.0309 | 6 | -0.68532 |
| 3 | 0.0142 | 2 | -0.68639 |
| 4 | 0.0066 | 1 | -0.68811 |
| 5 | 0.0294 | 2 | -0.68609 |
| 6 | 0.0175 | 3 | -0.68657 |
| 7 | 0.0120 | 3 | -0.68496 |
| 8 | 0.0771 | 1 | -0.69069 |
| 9 | 0.0364 | 6 | -0.68508 |
| 10 | 0.0075 | 3 | -0.68682 |

### Model quality and classification scores (Table 14.1, walk-forward test)

Walk-forward AUC 0.5063, 26,849 test events; CPCV AUC by path 0.5144, 0.5142, 0.5137, 0.5172, 0.5122 (mean 0.5143).

Accuracy 0.5090, precision 0.5104, recall 0.5579, F1 0.5331 (TP 7525, FP 7219, TN 6142, FN 5963).

### PSR and deflated Sharpe (Ch.14; trials = the Ch.9 candidates)

| span | days | daily SR | skew | kurtosis | PSR vs 0 | trials | SR* | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| last_365_days | 365 | -0.0613 | 0.05 | 11.02 | 0.123 | 11 | 0.0855 | 0.003 |
| whole_walk_forward_span | 1016 | -0.0517 | 0.15 | 11.23 | 0.051 | 11 | 0.0513 | 0.001 |

### Chapter 14 statistics, 2025-10-13 to 2026-10-12

Net gain -2.11% over 0.999 years; annualised return by Equation 49 -2.11%. Days with a non-zero target position 364 of 365; mean absolute position 0.0394, maximum 0.1413. Bets (14.1) 52; average holding period (14.2) 4.48 days; HHI of positive returns 0.1486, of negative returns 0.0375, of bets per month 0.0264 (14.3); 1 drawdowns, worst 2.84%, 95th percentile 2.84%; time under water, max n/a years (14.4).

### Chapter 15 (per-bet unit returns, last 365 days, 9,816 bets)

Observed precision 0.6675 (t-value under H0: p = 1/2, Equation 54: 35.22); mean positive bet return 0.04055, mean non-positive -0.08426.

| target Sharpe | implied precision (15.3) | implied frequency (15.4) | Sharpe at observed precision (Eq.56) | probability of failure (15.5) |
|---:|---:|---:|---:|---:|
| 1.0 | 0.6798 | n/a | -1.6074 | 0.5222 |
| 2.0 | 0.6845 | n/a | -1.6074 | 0.5306 |

### Snippet 10.4 order tickets

9,816 tickets in `orders_10_4.csv` (forecast price = the barrier of the predicted side, M* = 0.95, Q = 100): median target position 95, median limit offset 318.5 bp. Nothing is connected to any order path. The extract gives no forecast model, so the forecast price is my assumption (the barrier of the predicted side); that makes the ticket size the calibrated M* x Q for every order, so the tickets carry the side but no information about confidence. The confidence-dependent size is the Snippet 10.3 value in the `size_10_3` column, and that is what the account trades.

### Section 2.4.1: rebalance cost, spread, volume

Rebalance cost (item 1, Equation 19): on each quarterly HRP rebalance the whole old allocation is sold and the whole new one bought at the venue's cost per side, on the exposure carried into that day. Last 365 days: 4 rebalances charged, 0.067% of capital in total; walk-forward gain -2.05% before it, -2.11% after (the table at the top is after). Whole test span: 11 charged, 0.150% of capital; gain -3.86% before, -4.00% after.

Bid-ask spread (item 2): the account already pays the venue's cost per side on every change of every instrument's averaged position, which is the book's per-unit crossing cost; nothing is added.

Volume (item 3): a trade changes an instrument's exposure by K x |de| for account size K; it fits when that is no more than the traded value of its own bar. Over the 27,884 walk-forward trades the largest account K that fits every trade is 72275, the 1st percentile 736133, the 5th 3563913, the median 248195956 (currency units of the venue). Share of trades larger than their bar's volume at K = 1,000: 0.00%, 10,000: 0.00%, 100,000: 0.01%, 1,000,000: 1.42%, 10,000,000: 10.74%. The account is not volume-capped: this is a capacity measurement.

### Equations 77-78: concentration of the HRP allocation

H over 16 rebalances: first 0.883, last 0.913, mean 0.883, range 0.802 to 0.923 (file `portfolio_concentration_eq77_78.csv`). The extract prints the equations but not the definition of f; f = V'w (loadings on the covariance eigenvectors) is assumed.

Run: 1340 s, seed 20261022, cost 12 bp per side, `quick` = False, integrity of the original files unchanged = True.

## alpaca

### Events (Ch.2-4)

495 symbols, 1,105,182 CUSUM events, 352,728 kept with all twelve features and a label, 352,005 after Snippet 3.8 (dropped label classes: [0]; shares before: -1: 0.4714, 0: 0.0020, 1: 0.5266). Snippet 4.11 time decay at the snippet default c = 1.0: constant 1.0, slope 0.0 (no decay).

Point-in-time universe: `runtime/book_meta_alpaca/20261012T023148Z/universe.csv`.

### Chapter 13 optimal trading rule (20 x 20 mesh, 100,000 paths)

Equations 13.5-13.7 fitted on 253,967 pre-cutoff events (long paths, K = 100): phi = 1.0001, half-life n/a bars, forecast -35.344 sigma. Step 5a chose profit-taking 10.0 / stop-loss 9.5 (simulated Sharpe 0.0255 against a mesh median of 0.0155). Definition 2 on 96,833 post-cutoff events: Sharpe of the chosen rule 0.0542, median over the mesh 0.0359, best node in hindsight 0.0562; overfit by the book's definition: **False**. The fitted phi is at or above 1, so the pooled paths are not mean-reverting: the book requires phi in (0, 1) for the O-U model and its half-life, so the model does not hold here and the mesh result is mechanical. The vectorised mesh was cross-checked against the verbatim Snippet 13.2 loop on 3,000 paths (4 nodes): see `ch13.json`.

### Chapter 8 feature importance (253,993 events with labels ending before 2024-01-01)

| feature | MDI | MDA | SFI |
|---|---:|---:|---:|
| ffd | 0.0752 | 0.0004 +/- 0.0001 | -0.69446 |
| log_trgt | 0.1156 | -0.0004 +/- 0.0004 | -0.69484 |
| sadf | 0.0534 | -0.0001 +/- 0.0001 | -0.69588 |
| ent_plugin | 0.0721 | 0.0001 +/- 0.0000 | -0.69496 |
| ent_konto | 0.0775 | 0.0001 +/- 0.0000 | -0.69504 |
| cs_spread | 0.0398 | 0.0005 +/- 0.0002 | -0.69458 |
| bp_sigma | 0.1067 | 0.0016 +/- 0.0005 | -0.69335 |
| roll | 0.0431 | 0.0001 +/- 0.0001 | -0.69456 |
| kyle | 0.0112 | -0.0000 +/- 0.0000 | -0.69455 |
| amihud | nan | 0.0000 +/- 0.0000 | -0.69462 |
| hasbrouck | nan | 0.0000 +/- 0.0000 | -0.69462 |
| csw | 0.4055 | 0.0287 +/- 0.0031 | -0.65001 |

A blank (nan) MDI means no tree ever split on that feature: Snippet 8.2 replaces zero importances by NaN before averaging.

Orthogonal features (8.5): 10 principal components carry 95% of the variance; weighted Kendall tau between MDI and the inverse PCA rank (8.6) = -0.4316.

### Chapter 9 search (purged 3-fold, neg log-loss)

| candidate | min_weight_fraction_leaf | max_features | OOS neg log-loss |
|---:|---:|---|---:|
| 0 (untuned) | 0.0500 | sqrt | -0.64963 |
| 1 (chosen) | 0.0155 | 12 | -0.64201 |
| 2 | 0.0309 | 6 | -0.64232 |
| 3 | 0.0142 | 2 | -0.65080 |
| 4 | 0.0066 | 1 | -0.66020 |
| 5 | 0.0294 | 2 | -0.65308 |
| 6 | 0.0175 | 3 | -0.64542 |
| 7 | 0.0120 | 3 | -0.64419 |
| 8 | 0.0771 | 1 | -0.67474 |
| 9 | 0.0364 | 6 | -0.64251 |
| 10 | 0.0075 | 3 | -0.64418 |

### Model quality and classification scores (Table 14.1, walk-forward test)

Walk-forward AUC 0.5007, 97,782 test events; CPCV AUC by path 0.5011, 0.5016, 0.5016, 0.5009, 0.5019 (mean 0.5014).

Accuracy 0.5018, precision 0.5310, recall 0.5234, F1 0.5272 (TP 27156, FP 23987, TN 21913, FN 24726).

### PSR and deflated Sharpe (Ch.14; trials = the Ch.9 candidates)

| span | days | daily SR | skew | kurtosis | PSR vs 0 | trials | SR* | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| last_365_days | 365 | -0.0920 | -0.32 | 5.99 | 0.038 | 11 | 0.0842 | 0.000 |
| whole_walk_forward_span | 1013 | -0.0706 | 0.11 | 42.15 | 0.015 | 11 | 0.0525 | 0.000 |

### Chapter 14 statistics, 2025-10-10 to 2026-10-09

Net gain -0.74% over 0.999 years; annualised return by Equation 49 -0.74%. Days with a non-zero target position 364 of 365; mean absolute position 0.0217, maximum 0.1036. Bets (14.1) 25; average holding period (14.2) 8.69 days; HHI of positive returns 0.2741, of negative returns 0.0286, of bets per month 0.0432 (14.3); 1 drawdowns, worst 0.77%, 95th percentile 0.77%; time under water, max n/a years (14.4).

### Chapter 15 (per-bet unit returns, last 365 days, 36,446 bets)

Observed precision 0.4909 (t-value under H0: p = 1/2, Equation 54: -3.46); mean positive bet return 0.07869, mean non-positive -0.07868.

| target Sharpe | implied precision (15.3) | implied frequency (15.4) | Sharpe at observed precision (Eq.56) | probability of failure (15.5) |
|---:|---:|---:|---:|---:|
| 1.0 | 0.5026 | n/a | -3.4455 | 0.5186 |
| 2.0 | 0.5052 | n/a | -3.4455 | 0.5228 |

### Snippet 10.4 order tickets

36,446 tickets in `orders_10_4.csv` (forecast price = the barrier of the predicted side, M* = 0.95, Q = 100): median target position 95, median limit offset 154.7 bp. Nothing is connected to any order path. The extract gives no forecast model, so the forecast price is my assumption (the barrier of the predicted side); that makes the ticket size the calibrated M* x Q for every order, so the tickets carry the side but no information about confidence. The confidence-dependent size is the Snippet 10.3 value in the `size_10_3` column, and that is what the account trades.

### Section 2.4.1: rebalance cost, spread, volume

Rebalance cost (item 1, Equation 19): on each quarterly HRP rebalance the whole old allocation is sold and the whole new one bought at the venue's cost per side, on the exposure carried into that day. Last 365 days: 4 rebalances charged, 0.042% of capital in total; walk-forward gain -0.69% before it, -0.74% after (the table at the top is after). Whole test span: 11 charged, 0.123% of capital; gain -1.78% before, -1.90% after.

Bid-ask spread (item 2): the account already pays the venue's cost per side on every change of every instrument's averaged position, which is the book's per-unit crossing cost; nothing is added.

Volume (item 3): a trade changes an instrument's exposure by K x |de| for account size K; it fits when that is no more than the traded value of its own bar. Over the 155,904 walk-forward trades the largest account K that fits every trade is 1126667437, the 1st percentile 59091430784, the 5th 183226010688, the median 3019455017100 (currency units of the venue). Share of trades larger than their bar's volume at K = 1,000: 0.00%, 10,000: 0.00%, 100,000: 0.00%, 1,000,000: 0.00%, 10,000,000: 0.00%. The account is not volume-capped: this is a capacity measurement.

### Equations 77-78: concentration of the HRP allocation

H over 35 rebalances: first 0.993, last 0.980, mean 0.988, range 0.975 to 0.996 (file `portfolio_concentration_eq77_78.csv`). The extract prints the equations but not the definition of f; f = V'w (loadings on the covariance eigenvectors) is assumed.

Run: 1930 s, seed 20261022, cost 10 bp per side, `quick` = False, integrity of the original files unchanged = True.

## What is the book, what is a project choice, what is not applied

**Project choices (the book leaves them open; fixed before any result, none searched):** dollar-bar size (one average clock-hour of traded value, trailing 30 days) and hourly Alpaca rows; CUSUM threshold (causal one-bar sigma x sqrt(bars per day)); barriers 2 x daily volatility and a 4-day vertical barrier with a path cap of 600 bars (Kraken) / 150 (Alpaca); SADF window 100 / lag 1, entropy window 100 / word 2, FFD threshold 0.01; 100-row windows for Roll, the three lambdas and CSW, 24 rows for the Becker-Parkinson mean; forest of 100 trees (the book uses 1,000) and at most 10,000 draws; bet-size step 0.1; cutoff 2024-01-01; flat costs 12 bp (Kraken) / 10 bp (Alpaca) per side; Ch.9 ranges (min_weight_fraction_leaf log-uniform 0.005-0.2; max_features in 1, 2, 3, 6, all) with the untuned setting as candidate 0; HRP re-estimated quarterly on the trailing 252 trading days; the account join (each instrument's HRP weight is its slice of capital, the averaged bet signal in [-1, 1] is the position inside it); the Alpaca universe (top 200 by dollar volume, point-in-time month-end membership).

**Declared deviations from the book's data:** the three lambdas are computed on hourly reference rows with the tick rule on the row's close, not on trades inside each bar (we hold no trade tape for most of the history); the Chapter 13 fit uses long paths because there is no side before the model exists.

The extract holds snippets, equations and tables with almost no prose, so where it is silent (thresholds, window lengths, costs, how an allocation is joined to a signal) the choice above is mine and is listed, not hidden.

## The book's Contents, item by item

| item | status | what the bot does / why not |
|---|---|---|
| Ch.1  Tables 1.1, 1.2 | Used as the map | Rows 2-5, 7-10 are applied (feature importance, volume clock, FFD, triple barrier, uniqueness / sequential bootstrap, purged CV, CPCV, synthetic-data backtest and deflated Sharpe). Row 6 (learning side and size together; the fix is meta-labeling with a primary model) is NOT satisfied: the extract gives no primary rule, so this bot learns side and size together, which is that pitfall. Row 1 (meta-strategy) likewise needs a primary model. |
| Ch.2  Table 2.1, Fig 2.1 | Market data only | No fundamental, analytics or alternative data in the project. |
| Ch.2  Eq 1-3 (time / tick / volume / dollar bars) | Applied: dollar bars | Bars are built by the project's builder (bar size is a project choice). |
| Ch.2  Eq 4-15 (imbalance and run bars) | Not applied | Need the aggressor side of every trade; we hold no trade tape for most of the history. |
| Ch.2  Expression 1, Eq 16-18 (ETF-trick basket), Snippets 2.2-2.3 (roll gaps, rolled series) | Not applied | They build one price series from a roll-adjusted futures basket; we trade spot instruments one by one. |
| Ch.2  Expression 2, Eq 19-21 (rebalance cost, spread, volume) | Applied | Section 2.4.1 above: rebalance cost taken out of the account; spread paid per side; volume capacity measured. |
| Ch.2  Eq 22-25, Snippet 2.1 (PCA weights), Fig 2.2 | Not applied | A risk-distribution weighting for the ETF-trick basket; the account is allocated by HRP (Ch.16). |
| Ch.2  Eq 26-29, Snippet 2.4, Fig 2.3 (CUSUM filter) | Applied | Events are the CUSUM filter's output; the threshold is a project choice. |
| Ch.2  Eq 30-31 (fixed-horizon labels) | Not used by design | The method Chapter 3 replaces (Table 1.2 row 5). |
| Ch.3  Snippets 3.1-3.4 (daily vol, triple barrier) | Applied | Barriers 2 x daily volatility, 4-day vertical barrier. |
| Ch.3  Snippets 3.5-3.7 (side and size labels) | Applied with side = None | bin = sign of the return at the first touch. |
| Ch.3  Snippet 3.8 (drop rare labels) | Applied | Removes the zero-return class when below 5%. |
| Ch.4  Snippets 4.1-4.2, Eq 32-33 (concurrency, uniqueness), 4.10 (return attribution) | Applied | Weights computed on all events, then 3.8, as the book orders it. |
| Ch.4  Snippets 4.3-4.5, Matrix 1, Fig 4.1 (sequential bootstrap) | Applied | Inside the forest; the uniqueness of the draws is logged per split (split_metrics.csv). |
| Ch.4  Snippets 4.6-4.9, Fig 4.2, Function 1 (random-t1 Monte Carlo comparison of bootstraps) | Not applied | A demonstration on synthetic data. |
| Ch.4  Section 4.7, Snippet 4.11 (time decay) | Applied at the snippet default c = 1 | That is no decay; c is the user's parameter and was not searched. |
| Ch.5  Snippets 5.1-5.4, Fig 5.1-5.5, Section 5.4 | Applied | FFD with the smallest d that passes the stationarity test, per symbol. |
| Ch.5  Table 5.1 (futures ADF table) | Reference data | Not a procedure. |
| Ch.6  Eq 34-35, Section 6.3.1, Snippet 6.1, Fig 6.1-6.3 | Used as the rationale | Bagging accuracy / variance reduction; no code to run. |
| Ch.6  Snippet 6.2 | Applied (clf1) | Entropy trees, class_weight balanced, bagged with max_samples = average uniqueness, sequential bootstrap; 100 trees against the book's 1,000. |
| Ch.7  Snippets 7.1-7.4, Fig 7.1-7.3 | Applied | Purging and embargo in the Ch.9 search, the CPCV and the walk-forward cutoff. |
| Ch.8  Snippets 8.2-8.5 (MDI, MDA, SFI, orthogonal features) | Applied | Tables above; weighted Kendall's tau of MDI against the inverse PCA rank. |
| Ch.8  Snippets 8.6-8.10, Fig 8.1-8.4 (synthetic test harness, plots) | Not applied | They validate the importance methods on synthetic data; the tables are saved instead of plots. |
| Ch.9  Snippets 9.2-9.4, Function 2, Log 1, Fig 9.1-9.2 | Applied | Randomised purged-CV search with neg log-loss and sample weights. |
| Ch.9  Snippet 9.1 (grid search) | Not used | The randomised search (9.2) is the one run. |
| Ch.10  Snippets 10.1-10.3, Section 10.3 | Applied | Size from the predicted probability, averaged over active bets, discretised in steps of 0.1. |
| Ch.10  Section 10.6, Snippet 10.4, Fig 10.3 (sigmoid) | Applied | Order tickets (forecast price is an assumption, see above). |
| Ch.10  Eq 36 (sizing from bet concurrency with a Gaussian-mixture CDF), power-function alternative | Not applied | Alternatives to 10.3; the book gives no code and refers the mixture fit to another paper. |
| Ch.11  Eq 37-38, Fig 11.1-11.2 (CSCV, probability of backtest overfitting) | Not applied | Needs the profit series of every trial; the Ch.9 candidates were scored by log-loss, not by a saved P&L series. |
| Ch.12  Eq 39-44, Fig 12.1-12.2 (CPCV) | Applied | 6 groups, 2 test groups, 15 splits, 5 paths, embargo 0.01. |
| Ch.13  Section 13.4, Steps 1-4, 5a, Definition 2, Snippet 13.2, Fig 13.x | Applied | 20 x 20 mesh, 100,000 paths; the verbatim Snippet 13.2 loop is run as a cross-check. |
| Ch.13  Steps 5b-5c | Not applied | Step 5b needs a profit target supplied by the strategy and 5c a maximum stop-loss imposed by fund management; neither exists here. |
| Ch.13  Snippet 13.1, Table 13.1, Section 13.5.1 (synthetic grid over forecast and half-life) | Not applied | Illustrative experiment on assumed O-U parameters; here the parameters are fitted. |
| Ch.14  Snippets 14.1-14.4, Eq 50-51, Series 1-4 | Applied | Last 365 days of the walk-forward account. |
| Ch.14  Eq 45-47 (returns on the capital base) | Applied without dividends | Exposure never exceeds capital; there is no dividend series in the data. |
| Ch.14  Eq 48-49 | Applied | Cumulative return and annualised return. |
| Ch.14  Eq 52-53, Table 14.1, Fig 14.2-14.3 | Applied | PSR, deflated Sharpe with the Ch.9 candidates as trials, classification scores. |
| Ch.15  Eq 54-57, Snippets 15.3-15.5 | Applied | Implied precision and frequency, strategy risk, t-value of precision. |
| Ch.15  Snippets 15.1-15.2, Fig 15.1-15.3 | Not applied | Monte Carlo and symbolic demonstrations of the Sharpe-precision relation. |
| Ch.16  Snippets 16.1-16.4, Section 16.4.1, Eq 58 | Applied | HRP as printed (the square distance matrix goes to the linkage, which is Eq 58's distance of distances); quarterly. |
| Ch.16  Snippet 16.5, Table 16.1, appendices, CLA | Not applied | A Monte Carlo comparison of allocators on synthetic data. |
| Ch.17  Snippets 17.1-17.4, Eq 62-63 (SADF) | Applied | Daily log closes, window 100, minimum sample 20, one lag. |
| Ch.17  Section 17.3.2 (Chu-Stinchcombe-White) | Applied | sup of S over the last 100 hourly rows, as a feature (the critical-value test is not used). |
| Ch.17  Section 17.3.1 (Brown-Durbin-Evans), Eq 59-61 (Chow-type DF), Eq 65-66 (sub / super-martingale tests) | Not applied | The extract prints the equations without the specification needed to run them (regressors, trimming, dummy definition, exponent). |
| Ch.17  Eq 64, Table 17.1, Fig 17.1-17.3 | Not applied | Computational cost and illustrations. |
| Ch.18  Eq 67-75, Snippets 18.1-18.4 (plug-in, Kontoyiannis entropy) | Applied | Window 100, word length 2. |
| Ch.18  Eq 76-78 (effective number, concentration) | Applied for Eq 77-78 | On every HRP allocation; Eq 76 is the generalised mean they build on. |
| Ch.18  Eq 79-80 (PIN, VPIN) | Not applied | Need buy / sell volume of trades. |
| Ch.19  Eq 81-83, 90-92 (tick rule, Roll, Kyle / Amihud / Hasbrouck lambdas) | Applied | On hourly reference rows (declared deviation). |
| Ch.19  Eq 84-89, Snippets 19.1-19.2 (Corwin-Schultz, Becker-Parkinson) | Applied | Spread and volatility features. |
| Ch.19  Fig 19.1-19.3, Sections 19.5.1-19.5.2 | Illustrations / PIN, VPIN | See above. |
| Ch.20-22  Snippets 20.1-20.14, Eq 93-98, Snippets 21.1-21.7, Fig 22.x | Not applied | High-performance computing, brute-force dynamic portfolio and infrastructure; not strategy components. |
