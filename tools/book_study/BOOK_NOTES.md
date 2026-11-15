# Book notes: what the book prescribes, what the code does, where it differs

Source: *Advances in Financial Machine Learning*, condensed edition (the PDF in the project uploads). Chapters 1-16 were read in full;
Chapters 17-22 (structural breaks, entropy, microstructure, and the closing chapters) were NOT read, and nothing here depends on them.
Everything below is paraphrased; no passages are reproduced.

| Book chapter | What it prescribes (paraphrase) | Where in the code | Followed exactly? |
|---|---|---|---|
| 2 (Snippet 2.4) | Sample events with a symmetric CUSUM filter instead of every bar | `book_stats.cusum_events` (literal port, tested against a literal copy) | Threshold is one causal daily sigma per bar instead of a fixed h (deviation 5) |
| 3 | Size barriers by a dynamic volatility estimate; vertical barrier; label by what is hit first | `book_study_core.build_events` (sigma = EWM std of hourly log returns, span 336, using only past data) | Barriers are chosen by Chapter 13's search, not fixed multiples |
| 4 | Overlapping labels are not independent: weight by average uniqueness | `book_stats.average_uniqueness`, used as weights in the O-U fit | Uniqueness is computed within each symbol only (deviation 7) |
| 7 | Purge training samples whose label span overlaps the test span; add an embargo after it | `book_stats.train_mask` | Embargo is 1% of the whole time span (deviation 6) |
| 12 | Combinatorial purged CV: N groups, k test groups, C(N,k) splits, C(N-1,k-1) backtest paths | `book_stats.cpcv_splits`, `cpcv_paths`; defaults N=10, k=2 -> 45 splits, 9 paths | Yes |
| 13 | Fit an O-U process to the in-sample price path; simulate many paths; evaluate a 20x20 mesh of profit-taking/stop-loss levels (0.5 to 10 sigma) with a maximum holding period; pick the best Sharpe; Definition 2 calls a rule overfit if its out-of-sample Sharpe is below the median out-of-sample Sharpe of the mesh | `book_stats.estimate_ou`, `simulate_ou_paths`, `mesh_exit_stats`, `best_node`; per split in `run_study` | Drift is estimated directly instead of taking a forecast (deviation 1); H = 8, 24, 100 all run (deviation 8) |
| 14 | Probabilistic Sharpe ratio (adjusts for length, skew, kurtosis); deflated Sharpe (compares with the expected maximum Sharpe of N unskilled trials) | `book_stats.psr`, `expected_max_sharpe`, `dsr` | DSR is conservative for mirrored long/short rows (deviation 11) |
| 15 | Relation between Sharpe, precision, and bet frequency; the break-even precision | `book_stats.bin_sr/bin_hr/bin_freq`, `break_even_precision`; shown as 'observed vs break-even precision' | Yes (precision is conditional on a barrier exit; vertical-exit share is printed beside it) |

## The test, in plain terms
For each situation (market state x direction) the study asks the book's question: if the best rule is chosen by simulation on the past and
then applied to data it never saw, does it make money after costs, on most backtest paths, after allowing for how many things were tried?
The placebo row (random sign per event) is the baseline for what the chosen rule earns by itself; a direction's edge is its row minus the placebo row, and it must survive the cost.

## Expected behaviour of a plain control
A control with no forecast behaves like a random walk (phi near 1, no drift). On a random walk every stop/profit rule has the same expected
gross result (zero), so the "optimal" rule is whichever node the simulation noise favours, and costs then make the net result negative.
On real candles the placebo can sit a few basis points above zero (short-horizon moves partly reverse, and a small-profit/wide-stop rule collects that whichever way it faces). The study can only report an edge if some state/direction shows real drift in the training data that persists in the held-out groups.

## Window reporting
Every table has window lines for June 2026 (the AI-correction month), the post-holdout sell-off (taken from the newest baseline run), and the
deepest fall of this study's own equal-weight index. Windows are whole UTC days; an event counts if it was entered inside the window.

## Deviations (also printed at the bottom of SUMMARY.md)
See `book_study_core.DEVIATIONS`; the same list is rendered into every SUMMARY.md.

## Added: momentum / trend rules and the trained models

**Momentum / trend rules** (`run_book_study.py --signals mom168 mom720 mom2160 ma200d --only-signals`). Run through the identical
pipeline as the controls: same events, same 45 purged and embargoed splits, same chosen-by-simulation stop/profit rule, same costs, same
placebo, same windows. The signal is the sign of the trailing log return over 7, 30 or 90 days, or the sign of price against its own
symbol's 200-day mean, read only on event bars. No parameter is fitted, but every (signal, state, horizon) row is a trial in the deflated Sharpe
(4 signals x 3 states x 2 horizons = 24). This is momentum conditional on a CUSUM event, not a rebalancing-date momentum portfolio, and the sign is
the plain sign, not the volatility-scaled sum of the published time-series-momentum rule. Cross-sectional (rank) momentum is a portfolio-level
construction and is not an event-level rule; the project's own `run_cross_sectional_momentum_experiment.py` and the Alpaca equity momentum tests
are separate records and are not re-run here.

**Trained models** (`run_model_scorecard.py`). The models were fitted once on a fixed 70/10/10/10 time split, so the book's combinatorial
cross-validation (which needs a model refitted per split) cannot be applied. What can be done honestly is to score the events inside each model's
own out-of-sample span (holdout, then post-holdout) and compare events the model flagged against events it did not flag, same symbols and period.
The stop/profit rule is the control study's, chosen without any model output. A rule-free read (hold H bars, and hold 1 bar, the models' target
horizon) sits beside it. Intervals come from resampling whole days. Nothing is promoted in either direction, so none of this would have traded.

**What the placebo means.** A cut-losers/ride-winners stop/profit rule earns money on paths that keep going after an event whatever the sign, and
loses on paths that revert. The placebo row measures that rule effect; a direction's or a model's edge is the difference from it.

## Added: rule-free decomposition of the momentum signals (`run_momentum_decomposition.py`)
Purpose: separate "the sign sorts future returns" from "long positions in surviving coins drift up". Every CUSUM event is held for H+1 bars (H = 24 and 100) with no stop,
no profit target and no simulated rule, so nothing in it was chosen by simulation. Per lookback (7, 20, 30, 40, 90 days) and market state it reports the signal's pooled gross
result, each leg, always-long on the same events, and winners-minus-losers (drift cancels), with 95% intervals from resampling blocks of whole UTC days (block = ceil((H+1)/24) days
because holds overlap), plus by calendar year and the June 2026, sell-off and deepest-fall windows (no intervals for windows). Not corrected for the number of lookbacks looked at; stated in the report.
This is a diagnostic: it selects nothing and is connected to nothing. Checker: `verify_momentum_decomposition.py` (recomputes every row, year and window line from events.csv, and events, signs,
forward returns and market state from the raw candles).

## Added: meta-labeling pipeline in the book's order (`run_meta_pipeline.py`, checker `verify_meta_pipeline.py`)
Built: CUSUM events (Ch.2) -> triple-barrier meta-labels with the book's daily-volatility target (Ch.3, Snippets 3.1, 3.2, 3.6, 3.7) -> concurrency, average uniqueness and
return-attribution weights (Ch.4, Snippets 4.1, 4.2, 4.10) -> sequential bootstrap (Snippet 4.5, exact incremental version, one per instrument) -> fractionally differentiated log
price at the smallest d on a 0.05 grid that passes ADF (Ch.5, Snippets 5.3, 5.4; FFD threshold 0.01; own ADF, no statsmodels) -> bagged entropy trees with balanced class weights
(Ch.6, Snippet 6.2 clf1) -> purged / embargoed CV (Ch.7) -> MDA and SFI importance (Ch.8, Snippets 8.3, 8.4, neg-log-loss) -> bet size from the probability (Ch.10, Snippets 10.1,
10.3) -> combinatorial purged CV (Ch.12) with PSR and deflated Sharpe (Ch.14). A walk-forward fit (train on labels ending before 2024-01-01) sits beside it as a time-ordered check.
Variants scored on the same events: primary (every event, size 1), meta (size from the model) and placebo (the meta sizes shuffled among the same events). June 2026, the sell-off and
the deepest-fall windows are reported for each, and so is everything by market state (above / below the equal-weight index's own 200-day mean), with the primary split long / short.

Version 2 adds the Chapter 17-19 features (`micro_features.py`, `test_micro.py`): SADF (Snippets 17.1-17.4) on a trailing 100-day window of daily log closes (min 20 days, constant, one
lag; the book's expanding window is not feasible on hourly history), plug-in entropy rate (word length 2) and the Kontoyiannis estimate (Snippets 18.1, 18.3, 18.4) of the sign-encoded
last 100 daily returns (the condensed edition has no encoding section; the Kontoyiannis estimate is read newest-first as the snippet's docstring suggests), and the Corwin-Schultz spread
(Snippet 19.1) on hourly High / Low with a 24-bar mean. Snippet 19.2 (high-low volatility) is not used: its printed first-term sign factor looks different from the paper's form and could not
be checked offline. About 74% of Corwin-Schultz values are exactly 0 because the snippet sets negative alphas to 0. All settings were declared before the run; one run; it counts as one more
look at this history (deflated-Sharpe prior count 82 = 81 + the first, 10-feature run).

Declared before any result: barriers +/-2 x daily vol, vertical 101 bars, FFD 0.01, side = sign of the trailing 720-bar return, 24 bp round trip, 14 features, 6 groups / 2 test groups
(15 splits, 5 paths), 100 trees, at most 10,000 draws per tree, bet step 0.1, seed 20261013. One run per version; no hyperparameter search. Deflated Sharpe is shown for N = 1, 10, 82 and 83.

NOT built, deliberately: volume or dollar bars (Ch.2; GMX candles carry no volume and TradingView volume stops 2025-11-30, so a volume clock cannot cover June 2026 or the
sell-off); Kyle / Amihud / Hasbrouck lambdas, VPIN, PIN and other volume or trade based microstructure features (Ch.19: same reason); HRP and the Chapter 16 allocation; the Chapter 20-22
material; the sample-weight time decay (Snippet 4.11); bar-by-bar averaging of active bets (Snippet 10.2). The primary side (30-day momentum) was chosen after seeing this history, so it is
in-sample by construction; the walk-forward block and the trial count in the deflated Sharpe are the guards, not a cure.

Checker: rebuilds every event from the raw candles (all symbols), including labels, weights, d, SADF, entropies, Corwin-Schultz, other features and state, and compares each column;
re-derives splits, purge / embargo counts, prediction membership, AUC, bet sizes, daily series, Sharpe / PSR, the deflated-Sharpe table, window lines, the by-state table and the walk-forward
numbers from events.csv. Not covered by the checker: the forest, the sequential bootstrap and the MDA / SFI values (`test_meta.py` covers those, including a literal-procedure comparison of
the sampler), and the placebo's exact permutation (checked for event counts, size multiset and consistency only).

## Added: three algorithms (long, short, momentum) and a forest gate (`run_three_algos.py`, `three_algo.py`, checker `verify_three_algos.py`)

What it tests: the intended architecture of three algorithms, each used only where it plays well, with something choosing between them. The book has no neural network, so the chooser is
the book's own machinery: bagged forests as secondary models (meta-labeling, Ch.3 / Ch.6), bet size from the probability (Ch.10), combinatorial purged CV (Ch.12), PSR and deflated Sharpe
(Ch.14). It reads a finished and separately verified meta-pipeline run (`events.csv`), so the events, barriers, weights and features are exactly those already checked; the only new
quantities are the 20-, 30- and 40-day trailing returns per event.

Algorithms: LONG = side +1 at every event; SHORT = side -1; MOM = long if the 20-, 30- and 40-day returns are all positive, short if all negative, otherwise no signal. Because the barriers
are symmetric on the simple return, a short exits at the same bar as the long (ret_long = side x ret_side), so no separate short-side labels exist.
Forests: F_dir (13 features, label = long return > 0, trained on every event; p(SHORT) = 1 - p(LONG)) and F_mom (14 features, label = the MOM side's return > 0, trained on MOM events).
Gate: the candidate with the largest confidence among LONG, SHORT and (if a signal exists) MOM; a bet is taken only above 1/2, sized by Snippets 10.1 / 10.3.
Variants: long_all, short_all, mom_all, mom_long_up (a fixed rule: MOM long while the index is above its 200-day mean), dir_model, mom_meta, gate, and two nulls (gate sides with shuffled sizes;
gate sizes with coin-flip sides). All settings were fixed before the first run. A quick smoke run (4 groups, 10 trees) was made to check the plumbing and its table was seen when checking the layout; no setting was changed afterwards.
Trials: prior 83 (after the 14-feature run) + 4 = 87; the gate is the 88th; deflated Sharpe shown for N = 1, 10, 87, 88.

Known weakness of the design (declared, not tuned away): the gate compares confidences of two different forests. If one of them is uninformative but confident-looking it can win the choice
and drag the gate down; the by-algorithm tables show where that happens.

## Added: market-regime short rule and matching long filter (`run_regime_study.py`, `regime_algo.py`, checker `verify_regime_study.py`)

What it tests: whether a market-level condition separates the two failures seen in the three-algorithm study: the fixed rule mom_long_up (MOM long while the equal-weight index is above its
200-day mean) lost about 100 bp per candidate in the sell-off, and short_all earns in June 2026, the deepest fall and the sell-off but loses 63 bp per candidate over the whole history.
Everything is a fixed rule, read from the verified meta run's `events.csv` plus the raw candles; nothing is fitted, so there is no cross-validation. No model, calibration or live file is touched.

MKT_DOWN = the equal-weight index (the one behind the 200-day state) has a negative log return over each of the last 480, 720 and 960 hours (the same three lookbacks as the stock signal; no new
parameter). Tight (240 / 360 / 480) and wide (960 / 1440 / 1920) neighbours are reported without selection. Rules: short_down, short_down_mom, long_up_notdown, regime_rule (the last two together),
regime_rule_tight, regime_rule_wide; baselines long_all, short_all, mom_all, mom_long_up are recomputed. Trials: prior 88 + 6 new rules = 94; deflated Sharpe shown for N = 1, 10, 88, 94.

Declared limits: the rules were motivated by the sell-off window, so they are in-sample by construction; the half / year splits show stability, not out-of-sample performance. The best-20-days share
is not meaningful for rules whose total profit is near zero. No funding or borrow cost is charged on shorts.

Checker: rebuilds each symbol's trailing returns and the equal-weight index from the raw candles, recomputes every rule, the halves, years, windows, MKT_DOWN shares, cells, long / short split and the
deflated-Sharpe tables, and matches each SUMMARY.md row cell by cell. `test_regime.py` (8 tests) covers the lookback arithmetic against a literal loop, the truth table, the rules by hand and a planted regime.

## Added: fast-drop condition on the equal-weight index (`run_fastdrop_study.py`, `fastdrop_algo.py`, checker `verify_fastdrop_study.py`, descriptive `diag_selloff.py`)

What it tests: the follow-up to the regime study, whose slow market-down state (20 / 30 / 40 days) never fired in the sell-off. Here a FAST drop of the equal-weight index is used as a filter on the
MOM-long rule and as a trigger for the MOM-short rule. z_L = index log return over L hours / (sigma x sqrt(L)), sigma = sample std of the index's 720 latest hourly log returns; FAST_L is z_L < -2
(2 = the barrier width). Base L = 72 hours; 24 and 168 hours are reported as neighbours, never selected. Rules: long_up_nofast, short_fast, short_fast_mom, fast_rule (the first and third together),
fast_rule_24, fast_rule_168; baselines long_all, short_all, mom_all, mom_long_up recomputed. Trials: prior 94 + 6 = 100; deflated Sharpe shown for N = 1, 10, 94, 100.

Flagged in advance: the detector is not from the book (the condensed edition has no crash detector; the nearest tool is SADF, Ch.17, which is used per symbol but was not applied to the index). The design
was written knowing that the sell-off existed, so the rules are in-sample by construction. Besides the rule tables the run reports what the long filter removes (mom_long_up trades FAST_72 blocks versus keeps,
by year, whole history and each window, with the number of distinct days) and a day-by-day table of the detector inside the sell-off window.

`diag_selloff.py <run folder>` is descriptive only: it reads `events_fastdrop.csv` (columns the checker verifies) and prints the index's trailing returns and the mom_long_up results per sell-off day. It is
not covered by the independent checker; its daily figures were cross-checked once with a separate stdlib script.

Checker: rebuilds the index and its rolling sigma from the raw candles with its own prefix-sum code, recomputes z, FAST, every rule, statistics, halves, years, windows, cells, the long / short split,
the removed-versus-kept table, the sell-off day table, FAST day counts, the deflated-Sharpe tables and every SUMMARY.md row. `test_fastdrop.py` (8 tests) covers sigma against a literal loop with gaps,
the z / flag edge cases (strict threshold, NaN), the rules by hand, a literal-loop evaluation including the filter and sell-off tables, and a planted fast drop.

## Added: market-level Chapter 17-18 features in the meta-labeling pipeline (`run_meta_market.py`, `market_features.py`, checker `verify_meta_market.py`)

What it tests: the book's own route to "when does the bet work" is a secondary model (meta-labeling, Ch.3 / Ch.6) fed with features, with MDA / SFI (Ch.8) saying which features matter. The 14-feature run already
had per-symbol SADF (Ch.17) and entropy (Ch.18); this run adds the same three features of the MARKET: SADF, plug-in entropy and Kontoyiannis entropy of the equal-weight index of the 200-day state, computed on the
daily log of the index (last hourly level of each UTC day) with exactly the declared per-symbol settings (100-day window, minimum 20, one lag; 100 daily returns, word length 2), read at the last completed UTC day.
No threshold and no new parameter; the forest and MDA decide whether they matter.

Everything else is the verified pipeline: the events, labels, weights and folds are those of the 14-feature run (read from its events.csv, md5 recorded), same seed (20261013), 6 groups / 2 test groups, 100 trees,
sequential bootstrap, walk-forward cutoff 2024-01-01, bet step 0.1. One run; it is look number 101 (prior count 100 = 83 after the 14-feature run + 4 + 1 + 6 + 6). June 2026 and the sell-off window are reported
separately, for the CPCV and the walk-forward, and the SUMMARY compares the two runs window by window.

Declared: applying SADF and entropy to a market index is this study's choice (the book applies them to a price series); a one-run comparison at the same seed shows a difference of a few thousandths in AUC that a
different seed could move.

Checker: `verify_meta_market.py` is `verify_meta_pipeline.py` (rebuilds every event from raw candles, re-derives splits, purge / embargo, AUC, bet sizes, daily series, Sharpe / PSR, deflated Sharpe, windows,
walk-forward) with the three market features rebuilt by its own SADF / entropy code from the raw candles' equal-weight index, the 14 original columns compared with the source run row by row, the trial count
(100 / 101) and the comparison section of SUMMARY.md checked against both runs' numbers. `test_market_features.py` (5 tests) covers the daily construction (last hour of each UTC day, empty days skipped),
causality (an event cannot see its own day), idempotent feature list, and a small CPCV with 17 columns.

## Added: the meta-labeling pipeline on the venues the bot trades (Kraken for crypto; `export_venue_candles.py`, `venue_loader.py`, `run_meta_venue.py`, checker `verify_meta_venue.py`)

Why: every earlier run read GMX (plus TradingView) hourly candles, which are research data only; the live bots trade crypto on Kraken and equities on Alpaca. This run is the 14-feature pipeline, unchanged
(same events, labels, weights, folds, seed 20261013, 6 groups / 2 test groups, 100 trees, cutoff 2024-01-01, bet step 0.1), on Kraken hourly candles. Cost is 24 bp round trip, from the project's own Kraken
qualifier (`ma200_asset_qualifier.py`: ENTRY_COST = EXIT_COST = 12 / 10000). One run; it is look number 102 (prior count 101) and the data are new.

Data: the hourly files `freqtrade-develop/user_data/data/kraken/<SYM>_USD-1h.feather` (27 coins), copied read-only to CSV by `export_venue_candles.py` into `runtime/venue_candles/kraken_1h` with an md5 manifest.
Limit, stated in the SUMMARY: 24 of the 27 coins end on 2026-01-04 / 09, ADA on 2026-05-02, BNB and LINK on 2026-08-13. The market-state feature needs at least 10 symbols with a return in the same hour, so
events stop at the start of January 2026; June 2026 and the sell-off window (2026-09-27 .. 2026-10-08) contain no events and are reported n/a, with the per-window count of symbols that have a bar inside.
Coins with under 0.5 years of bars (8 of 27) are skipped, as in the earlier runs. Not fabricated, not back-filled.

Alpaca equities (daily bars, close and volume only, SIP feed, 5,039 symbols, 2016-01-04 .. 2026-10-02) are exported by the same script (`alpaca_1d/alpaca_daily.csv.gz`); the ticker "NA" must be read with
keep_default_na=False. The equities pipeline is the next section.

Checker: `verify_meta_venue.py` is `verify_meta_pipeline.py` with its raw-candle reader replaced, plus: every candle CSV against the export manifest (md5), the per-window symbol coverage recomputed from the raw
bars, every SUMMARY.md table cell compared with summary_numbers.json, the venue and cost lines, and a stop with exit code 2 on a malformed file. 31 planted errors (events, summary cells, coverage, manifest,
candle byte, trial count, cost, blanks, truncations) were all caught.

## Added: the meta-labeling pipeline on Alpaca daily equities (`equity_events.py`, `meta_eval_daily.py`, `run_meta_daily.py`, tests `test_equity_events.py`, checker `verify_meta_daily.py`)

Same chapters, models and folds as the hourly runs; every hourly assumption restated for one bar = one trading day, taking each setting from the project's own `tools/run_alpaca_momentum_test.py` where that file declares one.
Seed 20261015; prior trial count 102 (this run is look 103); 6 groups / 2 test groups, 100 trees, embargo 1%, bet step 0.1, walk-forward cutoff 2024-01-01. Nothing was tuned.

Universe (project's screen): at each COMPLETED month-end, adjusted close > 5, volume > 0, >= 253 bars, >= 40 valid dollar-volume days in the last 60 rows, ranked by the trailing-60-day median dollar volume
(ties by symbol), top N. The project uses N = 1000; this run uses N = 200 so the rebuild in the checker stays quick (SUMMARY.md deviations 15 and 16 state the universe and the survivorship bias). Members exist from the first month-end with 200 eligible
names (2017-01-31). A signal on day d uses the universe of the last month-end strictly before d. 495 symbols were ever members.
Events: CUSUM on daily log returns, threshold = causal one-day sigma (EWM std, span 14). Side: sign of 0.5 x (C[t-21]/C[t-252]-1) + 0.5 x (C[t-21]/C[t-126]-1) at the signal bar, shorts scored with no borrow cost.
Entry at the NEXT bar's close; every feature is read at the signal bar. Barriers +/- 2 x the book's daily-volatility estimate (span 100), vertical barrier 5 bars after entry. A path is kept only if every
market trading day between the signal bar and the last path bar has a bar for the symbol. Cost 20 bp round trip (10 bp per side, project's momentum test). 13 features (no Corwin-Schultz: no high / low).
Annualisation sqrt(252); effective sample = days / 5.

Limits, stated in the SUMMARY: survivorship (Alpaca's current tradable list, so delisted names are missing); no high / low; N = 200 not 1000; events entered in the last 5 trading days of the file cannot be labelled,
so the sell-off window (2026-09-27 .. 2026-10-08) has bars for all 495 symbols but no labelled events and is reported n/a. This is an event study of a momentum side, not the project's long-only monthly-rebalanced strategy.

Result of the full run (cloud, 196,049 events entered 2017 .. 2026-09-25, barrier exits profit-take 0.230 / stop 0.213 / vertical 0.557, meta-label share 0.521):
CPCV AUC mean 0.525 (min 0.508, max 0.533); walk-forward AUC 0.525. CPCV mean bp per candidate event: primary -7.3 (Sharpe -0.0465 daily, 5 of 5 paths negative), meta 0.0 (0.0245 daily, 0.39 annualised;
share of events taken 0.07, mean size 0.007), placebo -0.1. Walk-forward: primary +4.1 bp, meta +0.1, placebo -0.0. Deflated Sharpe of the meta variant: 0.701 (N=1), 0.147 (10), 0.022 (102 and 103).
June 2026: 1,827 events per path, primary -36.6 bp, meta +0.6, placebo -0.3. Sell-off window: n/a (0 labelled events). Deepest 90-day fall of the index (2020-02-19 .. 2020-03-23, -39.7%): 2,883 events,
primary -253.0, meta -11.0. By state under CPCV: longs +10.3 bp (state 1) and +6.7 (state 0); shorts -38.7 and -61.7. MDA / SFI: every MDA mean is within +/-0.0006 of zero (largest: ffd +0.0006 with s.e. 0.0002); every SFI score is at or below the no-information level -0.6931.
Reading: the model's probabilities carry almost no information (AUC 0.525), it sizes nearly every event at zero, and the primary side loses after cost in cross-validation. Not a promotion criterion for any threshold.

Checker `verify_meta_daily.py` (46 checks, about 2.5 minutes, stdlib only): rebuilds the month-end universe and counts from the raw Alpaca file, every event and column, the per-symbol accounting, fracdiff, the market-index
state, the deepest-fall window and the window coverage, then everything the hourly checker re-derives (CPCV splits, purge / embargo, paths, daily series, Sharpe with sqrt 252, PSR / DSR, by-state tables, SUMMARY cells).
It also checks the raw file md5 against the run and the export manifest. 72 planted errors in the output files (events, universe, symbols, summary numbers, report cells, truncations) plus two edits to the raw file were tested.

Mac vs cloud reproduction (equities run): the Mac run (runtime/book_meta_alpaca/20261012T023148Z) passes `verify_meta_daily.py` 46/46 and its SUMMARY.md equals the cloud run's except the run-time line. The files are not byte-identical:
1,235 of about 2.5 million cells of events.csv differ by at most 1.3e-10 relative (sadf, vol_ratio, mkt_dist, ffd, sigma, trgt, returns); every integer and label column is identical; 125 numbers in summary_numbers.json differ by at most 7e-12 relative.
Cause not tested (likely floating-point summation order on arm64 vs x86). The Kraken run matched byte for byte. Compare runs numerically, not by md5; `md5_of_big_files.txt` holds the cloud hashes only.

## Added: Kraken candle refresh and the venue pipeline on the merged candles (`merge_kraken_refresh.py`, `test_merge_kraken_refresh.py`; run `runtime/book_meta_kraken_v2/20261013T022218Z`, look 104)

Why: the first Kraken export ended in early 2026, so neither the June 2026 window nor the sell-off window (2026-09-27 .. 2026-10-08) had events. Candles from 2026-01-01 were rebuilt from trades with `freqtrade download-data --dl-trades` (25 pairs, run by the user; see freqtrade-develop/README_ONNX_SETUP.md) into runtime/kraken_refresh/.
BTC and ICX were not refreshed and still end in early January 2026.

Merge rules (`merge_old_new`, 4 unit tests): the old file is kept unchanged up to its last bar; refresh bars are appended only after that hour; the last bar of every refresh file is dropped (partial hour); old-only symbols are copied, new-only symbols used alone. Each old CSV is checked against the md5 in runtime/venue_candles/manifest.json first.
Output: runtime/venue_candles_v2/kraken_1h/ with its own manifest.json (the pipeline checker reads that md5 list). The script refuses to overwrite an existing output folder.
Merge facts: junction gap 0 hours for every symbol except LRC (1 hour). On the overlap, trade-built closes differ from the old candle closes by at most 1.04% (XMR), 1.03% (ALGO), 0.92% (ATOM), 0.69% (NEAR), 0.62% (SOL), 0.60% (ADA), 0.53% (XRP); most are far smaller. The cause of the differences was not established, and the level step at a junction was not measured.
Merged series end at different hours of 2026-10-12. ANKR, BAL and LRC have fewer new bars than hours.

Run: `run_meta_venue.py --venue kraken --candle-dir runtime/venue_candles_v2/kraken_1h --prior-trials 103 --sell-off 2026-09-27 2026-10-08` (scripts unchanged; seed 20261013). 26 symbols (ICX skipped by the 0.5-year rule, 4,212 bars), 39,869 CUSUM events, 29,929 kept. Events kept per symbol: BAL 0, ANKR 1, LRC 27, the rest 379 and up.
The run's trial count: look 104, so DSR is shown for N = 1, 10, 103, 104. Deviation 16 in the SUMMARY still says "GMX / TradingView candles"; that text lives in meta_events.py (not edited) and is outdated for the venue runs: the earlier looks were on GMX / TradingView and Kraken candles.

Result: CPCV AUC mean 0.513 (0.496 .. 0.524); walk-forward AUC 0.529. CPCV per candidate event: primary +15.8 bp, meta +2.8, placebo +0.4; annualised Sharpe 0.40, 0.97, -0.04; no path negative for primary or meta; meta beats placebo on 5/5 paths (+2.35 bp). Walk-forward: primary -1.0 bp, meta +3.1, placebo +0.6. DSR of meta: 0.868 (N=1), 0.324 (10), 0.077 (103 and 104).
Old run (19 symbols, 23,484 events, no June / sell-off events): AUC 0.522, DSR 0.100 at its prior count. Unchanged by the merge: the deepest-fall window (1,153 events, primary +28.5 bp) and the walk-forward training set (15,539 events).
Windows (events per path, primary / meta CPCV / meta walk-forward / placebo CPCV / placebo walk-forward, bp): June 2026, 25 of 26 symbols have bars, 703 events, all state 0: +67.0 / +1.7 / +5.0 / +2.9 / +5.1. Sell-off, 25 of 26 symbols, 211 events, all state 1: -115.2 / -13.2 / -13.7 / -6.5 / -4.4. BTC is the missing symbol in both. Windows are a few correlated days and carry no intervals.
By state (CPCV): state 1 primary +86.4 bp (longs +124.3, shorts +1.3), meta +6.6; state 0 primary -36.0 (longs -19.9, shorts -45.3), meta 0.0. The primary's gain is concentrated in longs in up-states.
Features: largest MDA mean is mom24n 0.0059; SFI scores lie between -0.6935 and -0.6896 against the no-information level -0.6931.
Reading: close to no information in the probabilities; the meta variant's edge over the placebo is small and does not hold in either required window. Not a promotion criterion for any threshold.

Checker change (`verify_meta_venue.py`): the expected prior trial count is a constant (101) with a `--prior-trials N` override; the fracdiff check now expects a row for every symbol with at least one kept event and no other row, because the run writes no fracdiff row for a symbol without events (BAL). Before this change the checker failed 37/38 on the v2 run for that reason only. After it: v2 run 38/38 (--prior-trials 103) and the old Kraken run 38/38 (default).
Limits: fracdiff.csv lists 25 symbols although 26 were used (BAL); PAXG loses 1,124 of 2,006 events to gaps; the Kraken data is a mix of exchange candles and trade-built candles.

## Added: volume-based microstructure features (Ch.19, bar-level proxies) on the Kraken v2 candles (`volume_features.py`, `meta_events_vol.py`, `run_meta_volume.py`, tests `test_volume_features.py`, checker `verify_meta_volume.py`, `check_volume_junction.py`; run `runtime/book_meta_kraken_vol/20261013T042244Z`, look 105; control `runtime/book_meta_kraken_ctrl/20261013T043346Z`)

Four features fixed before any result and added to the 14 existing ones (18 in total); none removed afterwards. Definitions are in the docstring of volume_features.py. All windows count hourly bars.
kyle_lam: OLS slope (with intercept) of the bar return on tick-rule-signed dollar volume over the last 168 bars, times the mean hourly dollar volume of those bars. amihud_lam: the same for |return| on dollar volume. vpin: sum of |buy - sell| volume over 24 bars divided by volume, buy share from bulk volume classification on the bar return with sigma = std of the last 168 returns. dvol_z: z-score of log(1 + 24-bar mean dollar volume) against its last 720 values.
Where the book cannot be followed exactly: the book builds these from tick / trade data and volume buckets; here the tick rule uses bar closes, buckets are time bars, and the lambdas are rolling bar regressions. Whether the book's regressions include an intercept was not confirmed. Hasbrouck's lambda needs trade-level signed flow and is not built. PIN, volume-based structural-break features and HRP (Ch.16) are still not built.

Design: the pipeline scripts are unchanged. run_meta_volume.py runs run_meta_venue.main() after swapping in an event builder that calls meta_events.build_meta_events() unchanged and appends the four columns (meta_events_vol.py); events with a non-finite volume feature would be dropped and the uniqueness / attribution weights recomputed on the rest. Result here: 0 events dropped, 29,929 events, the same events as the v2 run. Seed 20261017, prior trials 104 (DSR shown for 1, 10, 104, 105).
Control: the unchanged 14-feature pipeline with the same seed (20261017) and prior trials, so that a difference is not just a seed change.
Tests (12): the features against a plain-loop reference, causality, missing and zero volume, window bounds, and the wrapper (14 base columns and weights equal the plain pipeline when nothing is dropped; weights recomputed when events are dropped). 7 planted mistakes in the feature code were all caught.
Checker: verify_meta_venue.py gained hooks (EXTRA_FEATURES, EXTRA_TOL, EXTRA_PREP, EXTRA_DROP_KEY, EXTRA_CHECKS; empty for the plain runs). verify_meta_volume.py recomputes the four features for every event from the raw candle file with plain loops, counts the dropped events independently, and checks volume_extra.json (declared features, windows 168 / 24 / 720, seed, prior trials, md5 of the three volume scripts). Volume run: 40/40. Control: 38/38. v2 run (--prior-trials 103) and the old Kraken run: 38/38 each after the hook edit. 12 planted errors in the output files (feature cells, a swapped pair, a removed column, missing / altered volume_extra.json, a wrong count, 14-feature list, wrong prior trials) were all caught.
Mac and cloud runs agree (same tables to the digits shown).

Volume definition at the junction (check_volume_junction.py, old export vs trade-built refresh on the bars present in both): summed volume ratio new / old 1.000 to 1.018 (largest ATOM 1.018, SOL 1.013, NEAR 1.011, BAL 1.010), median bar ratio 1.000 for every symbol, 98% to 100% of overlap bars equal within 1%. So the two sources count volume the same way on the overlap. The mean volume of the 168 bars after the junction against the 168 before ranges from 0.49 (DOT) to 4.30 (XMR); the cause was not examined, and beyond the overlap there is no second source to compare.

Result (volume run vs control, CPCV unless noted): AUC mean 0.515 (0.501 .. 0.532) vs 0.513 (0.492 .. 0.533); walk-forward AUC 0.528 vs 0.529. Meta per candidate event 3.1 bp vs 2.6, annualised Sharpe 1.13 vs 0.84, PSR 0.89 vs 0.82, no negative path in either; meta beats placebo on 4/5 paths in both (Sharpe difference 0.0460 vs 0.0313). DSR of meta: 0.904 / 0.395 / 0.108 / 0.107 vs 0.831 / 0.269 / 0.056 / 0.056. Walk-forward meta: 2.7 bp, Sharpe 1.06, PSR 0.77 vs 2.8, 1.16, 0.80.
For scale, the original v2 run (14 features, seed 20261013) had CPCV meta Sharpe 0.97 and AUC 0.513, so the same features moved by 0.13 in Sharpe between two seeds; the volume run is +0.29 over its same-seed control in CPCV and -0.10 in walk-forward. One seed pair does not separate this from noise.
Windows (primary / meta / placebo, bp; CPCV): June 2026 67.0 / 1.2 / 3.4 vs control 67.0 / 1.8 / 3.2; sell-off 2026-09-27 .. 2026-10-08 -115.2 / -10.7 / -5.5 vs -115.2 / -12.5 / -6.3. Walk-forward meta: June 3.5 vs 1.9, sell-off -11.6 vs -10.3. In the sell-off the meta variant lost more than the placebo in both runs.
By state (CPCV meta): state 1 7.3 vs 6.4 bp, state 0 0.0 vs -0.2.
Importance of the new features (MDA mean, s.e.; SFI): kyle_lam 0.0001 (0.0002), -0.6921; amihud_lam -0.0001 (0.0003), -0.6926; vpin 0.0004 (0.0001), -0.6912; dvol_z 0.0006 (0.0003), -0.6903. For comparison mom24n 0.0059 (0.0006), -0.6900. No-information level -0.6931.
Reading: the volume features did not raise out-of-sample AUC; the meta Sharpe changes in opposite directions in CPCV and walk-forward and are within what a seed change can do; the sell-off result is unchanged. Not a promotion criterion for any threshold.

## Added: paired multi-seed comparison (`run_seed_pairs.py`, `test_seed_pairs.py`, checker `verify_seed_pairs.py`; run `runtime/book_seed_pairs/volume_vs_14`)

Why: one seed pair cannot show whether a difference between two feature sets is stable or forest noise. The harness runs a control script and a treatment script on the same candles for several seeds fixed in advance (20261017, 20261101, 20261102, 20261103, 20261104; prior trials 104, Kraken v2 candles), runs each run's own checker, and tabulates treatment minus control per seed (mean, sd, mean / se, min, max, seeds above zero).
What it can and cannot say: seeds change only the bootstrap draws and the placebo shuffle; events, labels and features are identical, so the spread is model randomness, not the sampling uncertainty of the data. It shows whether a difference is stable across seeds; it does not make anything significant. The 11 metrics were listed in the script before the run; no one of them was named as primary, so read the table as a whole.
Checks: 6 unit tests; checker 13/13 on the real run (all 10 per-run checkers passed: 40/40 and 38/38 as before; headline metrics re-read from each run's SUMMARY.md and compared with the table; every summary cell recomputed); 19 planted errors in the tables, logs, run files and settings were all caught (tested on a quick 2-seed smoke run). Seed 20261017 reproduced the earlier Mac pair exactly (control CPCV meta Sharpe 0.84, volume run 1.13, DSR 0.056 and 0.107).

Result, treatment (18 features, with volume) minus control (14 features), 5 seeds; mean difference (sd of differences), seeds above zero:
CPCV AUC +0.0026 (0.0010), 5 of 5 (0.5128 -> 0.5154). CPCV meta annualised Sharpe +0.29 (0.045), 5 of 5 (0.94 -> 1.23). CPCV meta bp +0.45 (0.068), 5 of 5. Meta minus placebo daily Sharpe +0.013 (0.003), 5 of 5. DSR at N=105 +0.061, 5 of 5 (0.070 -> 0.131).
Walk-forward AUC -0.0008 (0.0022), 2 of 5 (0.5307 -> 0.5299). Walk-forward meta Sharpe +0.15 (0.175), 4 of 5, mean / se 1.97: not stable.
June 2026 (CPCV): meta bp -0.45 (0.24), 0 of 5 (1.84 -> 1.39); meta minus placebo -0.49 (0.43), 1 of 5. Sell-off 2026-09-27 .. 2026-10-08 (CPCV): meta bp +2.57 (0.94), 5 of 5 (-12.62 -> -10.05); meta minus placebo +2.64 (1.53), 5 of 5 (-8.06 -> -5.42). Meta stays negative and below the placebo in the sell-off in both arms.
Seed noise for scale: the control's own CPCV meta Sharpe has sd 0.068 over the 5 seeds (0.84 .. 1.01); the original v2 run (seed 20261013) gave 0.97.
Reading: the volume features give a small CPCV improvement that is stable across seeds (AUC still only 0.515, DSR 0.131), but it does not appear in the time-ordered walk-forward AUC, the June 2026 meta result is worse in all 5 seeds, and the sell-off loss shrinks without becoming a gain. Not a promotion criterion for any threshold. Amends the single-pair reading in the volume-feature section above.

## Added: Ch.4 time-decay weights, tested with the seed-pair harness (`time_decay.py`, `run_meta_decay.py`, `test_time_decay.py`, checker `verify_time_decay.py`; harness options in `run_seed_pairs.py`)

Question: does down-weighting older training events (the idea of Snippet 4.11) change the out-of-sample behaviour of the 14-feature Kraken forest? Declared before any result: c = 0.5 (looks 106) and c = 0.0 (look 107), both on the 14-feature pipeline, seeds 20261017, 20261101 .. 20261104, control = the 14-feature runs already in `runtime/book_seed_pairs/volume_vs_14/ctrl_<seed>` (reused with `--ctrl-from`, not re-run). Prior trial counts: control 104 (as run), c = 0.5 run 105, c = 0.0 run 106. The DSR row in the harness table is read at each run's own prior + 1 (control N = 105, c = 0.5 N = 106, c = 0.0 N = 107), so the DSR difference includes the extra deflation of the extra looks; the AUC, Sharpe, bp and window rows do not depend on N.

Definition (written from the idea, not copied; see the docstring of `time_decay.py`): events of a training set ordered by start time, cumulative average uniqueness U_j, factor = c + (1 - c) U_j / U_n for c >= 0 (newest event 1, oldest about c); for -1 < c < 0 the oldest |c| share of cumulative uniqueness gets 0. The factor multiplies the attribution weight of Snippet 4.10; nothing is rescaled.
Where the book is not followed exactly: (1) factors are computed per training set on the pooled events of all instruments, so "newest" means newest within that training set; (2) in CPCV a training set can contain events later than the test group, so the decay there is age inside the training set, not distance to the test period; only the walk-forward fit matches the book's picture of old past, newer past; (3) MDA / SFI keep the ordinary weights; (4) ties in start time are ordered by the events file order.
Wrapper: `run_meta_decay.py` swaps `meta_eval.fit_forest` for a function that multiplies the weights of the training events by the factors of that training set and records them (`decay_factors.csv`: call, event, factor; calls 0..n_splits-1 are the CPCV splits, the last call is the walk-forward fit; `decay_extra.json`: c, calls, seed, prior trials, md5 of the two scripts). run_meta_venue.py, meta_eval.py, meta_model.py are not edited.
Finding while testing: the tree learner is not bit-for-bit invariant to the overall scale of the weights (a rescaling by 0.37 changed the number of leaves of a test tree from 14 to 15 and near-tied splits are decided by rounding; powers of two give identical trees). An earlier draft of my docstring said "invariant"; it now says ratios only. Consequence: any c < 1 differs from the control both through the weight ratios and through such rounding ties.

Verification (on the cloud, quick settings, 12 symbols, seed 123): with c = 1 the CPCV and walk-forward predictions are byte-identical to the plain 14-feature run; with c = 0.5 they differ and events.csv is identical. `test_time_decay.py`: 13 tests (12 unit tests plus one pipeline-level test that runs when DECAY_TEST_CANDLES is set); 10 of 10 mutations of the factor function and 7 of 7 of the weight wrapper are caught (one mutation, `c > 0` for `c >= 0`, is equivalent at c = 0 and was left out of the count).
`verify_time_decay.py` (stdlib; 44 checks on a 14-feature run = the 38 of `verify_meta_venue.py` plus 6): c is taken from the command line (`--decay-c`), not from the run; every training set is rebuilt from events.csv and must equal the events of that call; every factor is recomputed with plain loops (tolerance 1e-9); monotone in time, newest 1, bounds. 44/44 on quick runs for c = 1, 0.5, 0.0, -0.5; 11 planted errors in the run's files were all caught and the clean run passed.
Harness additions (all off by default): `--ctrl-from TAG`, `--treat-prior-trials N`, `--treat-args "..."`, `--treat-check-args "..."`, `--same-features`, `--dsr-n N`. `verify_seed_pairs.py` has two more checks when they are used (control runs lie in the named folder; the decay constant is the same in the arguments, the checker arguments and every run's decay_extra.json). Old format still passes (smoke folder 13/13); a decay smoke pair passes 15/15; 15 planted errors in a harness folder were caught and the clean folder passed. `test_seed_pairs.py`: 8 tests.
Because `run_seed_pairs.py` changed, the recorded md5 in `volume_vs_14/seed_pairs.json` no longer matches: re-aggregate once (`--aggregate-only`, no model runs) to refresh it; the tables do not change.
Result (Mac, 5 seeds each, all run checkers and `verify_seed_pairs.py` 15/15 for both folders `runtime/book_seed_pairs/decay_c05_vs_14` and `decay_c00_vs_14`; the volume_vs_14 folder re-aggregated and passes 13/13). Treatment minus the shared 14-feature control; seeds change only forest randomness, so this is model noise, not data uncertainty; the DSR row compares control N = 105 with N = 106 (c = 0.5) and N = 107 (c = 0.0).
c = 0.5 (look 106): CPCV AUC -0.0012 (higher in 1 of 5 seeds); CPCV meta annualised Sharpe +0.085 (4/5; control sd 0.068); meta bp +0.08 (4/5); walk-forward AUC +0.0033 (5/5); walk-forward meta Sharpe +0.064 (5/5); June meta bp -0.55 (0/5); sell-off meta bp +0.45 (4/5); sell-off meta minus placebo +0.51 (4/5); DSR +0.015 (4/5).
c = 0.0 (look 107): CPCV AUC -0.0030 (0/5); CPCV meta Sharpe +0.30 (5/5); meta bp +0.43 (5/5); meta minus placebo daily Sharpe +0.0175 (5/5); walk-forward AUC +0.0033 (5/5); walk-forward meta Sharpe -0.138 (1/5); June meta bp -0.65 (1/5); sell-off meta bp +1.72 (5/5); sell-off meta minus placebo +1.30 (4/5); DSR +0.066 (5/5).
Share of events the meta variant takes (mean of 5 seeds, CPCV): control 0.416, volume features 0.409, c = 0.5 0.403, c = 0.0 0.369; share with p > 0.5: 0.710, 0.708, 0.696, 0.647.
Reading: decay does not improve CPCV ranking (AUC falls, clearly at c = 0.0) while the CPCV Sharpe and bp rise at c = 0.0, together with a more selective model (fewer events taken); the walk-forward AUC rises in all 10 runs by the same +0.0033 but the walk-forward Sharpe goes up at c = 0.5 and down at c = 0.0; June 2026 gets worse in both; the sell-off loss shrinks but stays negative (about -12.2 and -10.9 bp) and below the placebo. The CPCV caveat above applies (age inside the training set). Why Sharpe and AUC move in opposite directions was not examined. Not a promotion criterion for any threshold. Next prior trial count for a later run: 107.

## Added: Ch.10 bar-by-bar averaging of active bets as an evaluation of existing predictions (`bet_averaging.py`, `run_bet_averaging.py`, `run_bet_avg_pairs.py`, `test_bet_averaging.py`, checkers `verify_bet_averaging.py`, `verify_bet_avg_pairs.py`)

Why: meta_eval.py scores one unit of capital per candidate event and says Snippet 10.2 is not reproduced (DEVIATIONS, bet size entry). This tool re-scores the predictions of an existing run (no model is fitted, no new look: the trial count stays 107) with the averaging rule, so overlapping and opposite bets in the same instrument net out and the cost falls on position changes.
Rule (declared; written from my memory of the snippet's logic, not its text, so it may differ in detail): signal of an event = side x bet size; the position of an instrument at bar k = plain mean signal of that instrument's events active at k (pos <= k < t1pos), zeros included, 0 when none is active; not rounded again to the step (the book, as I recall, discretises the average again; not done here because the sizes are already rounded). Bar result = position x (close[k+1]/close[k] - 1), booked on the UTC day of bar k+1; cost = 12 bp per side x |change in position| (position before the first bar = 0), booked on the day of bar k; so an isolated bet pays 12 bp x size on entry and on exit, as in the per-event rule (one-bar case equals the per-event formula exactly, tested).
Combining instruments (my rule, not the book's): result of a UTC day = sum over instruments of (gross - cost) / number of instruments with at least one candidate event active that day (an instrument is active on a day if an active bar k has bar k or bar k+1 on that day); the denominator uses candidate events only, so it is the same for primary (all sizes 1), meta and placebo. Levels are therefore not comparable with the per-event numbers; compare variants inside this convention. Windows: mean bp per active instrument-day over the window's UTC days (whole days, same window definitions as the run).
Gate inside the tool: before evaluating, every event's entry / exit bar timestamps and side-signed return must match the candle files (so the bar indices are right), and the per-event convention recomputed with the run's bet-size function and the same placebo shuffle (rng default_rng(seed + 99), one permutation per path; walk-forward seed + 7 + 99) must reproduce paths.csv of the run for every path and variant. The tool stops otherwise. The meta and placebo sizes used are saved (averaged_sizes.csv).
Bars are not strictly hourly (e.g. ETH has 849 irregular gaps); positions are per bar index of each instrument's own series, as the events were built.
Verification (cloud, Kraken 14-feature run book_meta_kraken_ctrl, seed 20261017, 26 symbols): `test_bet_averaging.py` 16 tests (hand-worked cases for overlap, opposite bets, zero-size events, day booking across midnight, windows, a random brute-force comparison, plus a pipeline-level test when BA_TEST_RUN is set); 12 of 12 deliberate breakages of bet_averaging.py are caught. `verify_bet_averaging.py` (11 checks, about 1 minute): recomputes every averaged number by brute force (a loop over active events for each instrument and bar, no cumulative sums), the bet sizes from the probabilities, that placebo sizes are permutations of meta sizes, the per-event reproduction of the run's paths.csv, every daily row, window row, summary cell and table cell; 11/11 on the run, 17 planted errors in the output files all caught, clean copy passes. `verify_bet_avg_pairs.py` (7 checks) re-runs that checker on every per-run folder and recomputes the pair table; 7/7 on a smoke folder, 6 planted errors caught, clean passes.
Result (Mac, `run_bet_avg_pairs.py --tag volume_vs_14`, 5 seeds, treatment = 18 features with volume, control = 14 features; unit tests 16 OK incl. the pipeline test on the v2 run; `verify_bet_avg_pairs.py` 7/7 with all 10 per-run folders passing the brute-force checker). Seeds change only forest randomness.
Levels in the averaged convention (mean of 5 seeds, CPCV; bp per active instrument-day / annualised Sharpe): primary 10.88 / 0.844; control meta 0.83 / 0.794; volume meta 0.92 / 0.862; placebo 0.14 / 0.323 (control) and 0.15 / 0.346 (volume). Cost per active instrument-day: primary 1.88 bp against 24.48 bp under the per-event rule (the per-event rule charges every overlapping bet a full round trip); meta 0.27 against 1.14. Walk-forward Sharpe: primary 0.221, control meta 1.110, volume meta 0.944, placebo -0.324 (control) and -0.390 (volume).
Treatment minus control: CPCV meta bp +0.092 and annualised Sharpe +0.068 (both 5/5 seeds); meta minus placebo daily Sharpe +0.002 (4/5); meta minus primary daily Sharpe +0.004 (5/5; the control's own value is -0.003, the volume arm's +0.001); meta cost +0.008 bp (5/5); walk-forward meta Sharpe -0.166 (0/5 seeds higher; the per-event convention had given +0.154, 4/5); June 2026 meta bp -0.386 (0/5) and minus placebo -0.400 (0/5); sell-off meta bp +0.835 (5/5), still negative at -4.81 bp, and minus placebo +0.920 (5/5) but still negative at -1.46.
Paths on which meta beat the primary's Sharpe: 2-3 of 5 in every seed of both arms (CPCV); meta beat the placebo on 4-5 of 5 paths. The meta variant takes very small positions (mean size about 0.046), so its Sharpe matches the primary's at a small fraction of the primary's bp.
Reading: the CPCV gain from the volume features survives the averaged convention in the same direction but is smaller (+0.068 Sharpe against +0.29 per event), the walk-forward gain does not survive (sign flips), June 2026 gets worse in all 5 seeds in both conventions, and the sell-off loss shrinks but stays negative and below the placebo. The instrument-combination rule is mine, not the book's. Not a promotion criterion for any threshold. No new model look.

## Ch.9 hyperparameter search (run_hp_search.py, verify_hp_search.py, test_hp_search.py)

Declared before any result. Read-only diagnostic; no model, threshold, calibration file or execution code is touched.

Design:
- Source: the events.csv of an existing 14-feature run (`--run`). Search set = events whose labels end before the walk-forward cutoff (2024-01-01), i.e. exactly the walk-forward training set. Nothing at or after the cutoff is used to choose.
- Grid, fixed in the script: min_weight_fraction_leaf in {0.005, 0.01, 0.025, 0.05, 0.1, 0.2} x max_features in {sqrt, 0.5} = 12 configs (ids 0-11, grid order). 100 trees, sequential bootstrap and max_draws 10000 as in the pipeline. The pipeline setting (0.05, sqrt) is config 6.
- Score: purged 5-fold CV (meta_model.purged_kfold, 1% embargo), pooled out-of-fold sample-weighted log loss. Lowest wins, first in grid order on a tie. AUC is reported, not used to choose. The forest seed of a fold is the same for every config (search seed 20261201 + fold index).
- Post-cutoff: the default and the chosen config are each fitted once on the whole search set and scored on the events starting on or after the cutoff, for the five seeds, with the June 2026 AI-correction and the sell-off (2026-09-27 to 2026-10-08) windows reported separately. If the chosen config is the default there is no comparison.
- Gate: the default at the source run's own seed must reproduce the source walk_forward_predictions.csv within 1e-6 or the run stops (checked in the cloud copy: max difference 5.3e-13).
- Trial count: one new config reaches the post-cutoff data, so this is look 108 (prior 107). If the whole grid should count (12), say so; DSR inputs would change. The post-cutoff period was seen in every earlier run, so it is not a pristine holdout.
- Known risk of the criterion: log loss can rank a near-constant model first when the features carry little signal (the unit test shows this property on synthetic data). Out-of-fold AUC of the quick test was about 0.50. If the winner sits at the grid edge (min_weight_fraction_leaf 0.2) read that as "the search found nothing to tune", not as a better model. AUC is printed next to it for this reason.
- The checker (stdlib only) rebuilds the folds, recomputes every score, rank, post-cutoff metric (not the placebo, which is a seeded shuffle) and every cell of hp_summary.md. The forests are not refitted by the checker.

Tests: planted-error harness on a quick output: 59 of 59 planted errors caught (metrics, scores, ranks, oof, folds, settings, md5, predictions, summary cells, missing/truncated files); one declared blind spot (placebo column) confirmed undetected. test_hp_search.py: 12 tests (10 unit, 2 pipeline with HP_TEST_RUN, one of them with a 2-worker pool).

Commands (project root, conda env javier):
    python tools/book_study/run_hp_search.py --run "$(ls -d runtime/book_seed_pairs/volume_vs_14/ctrl_20261017/*/ | tail -1)" --jobs 2
    python tools/book_study/verify_hp_search.py "$(ls -d runtime/book_hp_search/*/ | tail -1)"

Results (runtime/book_hp_search/20261013T083620Z, source run volume_vs_14/ctrl_20261017/20261013T050219Z; checker 10/10; gate max |dp| 5.3e-13):
- Chosen config 9 (min_weight_fraction_leaf 0.1, max_features 0.5); default config 6. Out-of-fold weighted log loss 0.686952 (chosen) vs 0.688627 (default); fold sd 0.0083 and 0.0117. Out-of-fold AUC of all 12 configs 0.492 to 0.516; chosen 0.4952 (rank 10 of 12), default 0.4947. Share of p > 0.5: 0.817 chosen, 0.781 default, up to 0.946 (config 10).
- Reference (hp_constant_baseline.py, writes constant_baseline.json in the folder): out-of-fold constant prediction (weighted base rate of each fold's training events) has log loss 0.687335; only config 9 is below it (by about 0.0004, well inside the fold sd). In-sample constant at the weighted base rate 0.5568: 0.686683. Constant 0.5: 0.693147. So the criterion picked the config closest to a constant; no discrimination was found pre-cutoff.
- Post-cutoff (5 seeds, default -> chosen): AUC 0.5307 -> 0.5317 (4/5 up); meta bp 3.04 -> 3.35 (5/5); annualised Sharpe 1.17 -> 1.30 (4/5); meta minus placebo daily Sharpe 0.098 -> 0.104; share taken 0.438 -> 0.461. June 2026 meta bp 2.95 -> 1.75 (1/5 up; net of placebo 1.14 -> -0.49, 0/5). Sell-off meta bp -13.23 -> -11.42 (4/5 up; net of placebo -7.67 -> -4.47); the sell-off result stays negative.
- Conclusion: the grid found no configuration that improves ranking; no change to the pipeline's forest settings is supported. Look 108. Post-cutoff period is not a pristine holdout. Weighted base rate of label 1: 0.557 before the cutoff, 0.522 after (not investigated).

## Ch.16 allocation across symbols (hrp_alloc.py, run_hrp.py, verify_hrp.py, test_hrp.py)

Declared before any result. Read-only diagnostic: no model is fitted, no run folder, threshold, calibration file or execution code is touched.

Design:
- What is allocated: the daily result stream of the strategy per symbol (bet_averaging.symbol_daily: position = mean signal of the active events, zero when none, 12 bp per side on position changes), as a share of capital placed in that symbol. It is the strategy stream, not the asset's own return.
- Estimation: only the 730 UTC days before the 2024-01-01 cutoff, primary stream (side, size 1) of events whose labels end before the cutoff. A symbol is eligible only with candles on or before the first and on or after the last day of the window and at least 100 active days in it. Covariance = sample covariance (ddof 1) in bp of the eligible symbols, zeros on days without activity or without a bar. Weights are fitted once and held fixed.
- Schemes on the same eligible set: equal weight; inverse variance; HRP; long-only minimum variance (SLSQP, first-order optimality check, stops if it fails).
- HRP written from the idea, not the book snippets: d = sqrt((1 - rho) / 2); second distance = Euclidean distance between columns of d; single linkage (own implementation); leaf order of the dendrogram; recursive bisection, a list is cut into floor(n / 2) first items and the rest; part variance with inverse-variance weights; first part gets 1 - V1 / (V1 + V2).
- Deviations from the book, stated in advance: (1) minimum variance replaces the book's CLA (not built), so the book's HRP-vs-CLA Monte Carlo is not reproduced; (2) the left/right order of merged clusters changes the HRP weights, so it is fixed by rule: lower cluster id first; at each step the smallest linkage distance merges, ties by the smaller lower id then the smaller higher id; (3) a symbol whose candle data has ended (no bar that UTC day) is left out that day and the other weights are rescaled to sum to 1, for all schemes alike.
- Scoring: events starting on or after the cutoff of eligible symbols; daily portfolio result in bp of capital; primary stream and the meta stream of each seed (walk-forward bet sizes). Out-of-sample daily sd, mean, annualised Sharpe, in-sample vs out-of-sample sd, effective number of names, by calendar year, and the June 2026 AI-correction and the sell-off (2026-09-27 to 2026-10-08) windows separately. Windows entirely before the cutoff are not scored. There is no bootstrap: the primary comparison is a single deterministic number with no spread; the spread across seeds exists only for the meta stream and covers forest randomness only.
- Trial count: no model fitted; choosing among four schemes on post-cutoff data is three new comparisons after look 108. The post-cutoff period was seen in every earlier run, so it is not a pristine holdout. Ineligible symbols' post-cutoff events are dropped and their number is reported.

Tests: 25 tests in test_hrp.py (24 unit and synthetic end-to-end tests; the 25th, on a real run, needs HRP_TEST_RUN). Code mutation: 19 planted bugs caught of 19 effective (a 20th mutant is equivalent); three were missed by the first suite and caught after adding a three-asset hand case, a cutoff-straddling event and symbols with exactly 100 and 99 active days to the fixture. Checker: 79 planted errors in the output files caught of 79 (first pass 64 of 69; the five not caught were three harness slips and two mutations that hit a table the checker did not yet read; the checker was extended to read the eligibility table and the summary sentences and those were re-run, 15 of 15). The full 69 were run on the checker before that extension, the 15 on the final version. Minimum-variance weights are checked by the first-order optimality conditions (sufficient for a convex problem), not recomputed.

Commands (project root, conda env javier), one --run per seed of the volume_vs_14 control:
    python tools/book_study/test_hrp.py
    python tools/book_study/run_hrp.py --run runtime/book_seed_pairs/volume_vs_14/ctrl_20261017/20261013T050219Z runtime/book_seed_pairs/volume_vs_14/ctrl_20261101/20261013T050847Z runtime/book_seed_pairs/volume_vs_14/ctrl_20261102/20261013T051507Z runtime/book_seed_pairs/volume_vs_14/ctrl_20261103/20261013T052117Z runtime/book_seed_pairs/volume_vs_14/ctrl_20261104/20261013T052725Z
    python tools/book_study/verify_hrp.py "$(ls -d runtime/book_hrp/*/ | tail -1)"

Results (runtime/book_hrp/20261013T093329Z, five volume_vs_14 control seeds; checker 12/12; test_hrp.py 25 tests, 24 run and 1 skipped):
- Eligible 13 of 24 symbols: ADA, ALGO, ATOM, AVAX, BTC, DOT, ETH, LINK, SOL, TRX, XLM, XMR, XRP. Dropped: 4,625 of 14,346 post-cutoff events (candles do not cover the window: APT, ARB, BNB, DOGE, FIL, ICP, KSM, LRC, NEAR, OP; PAXG has 87 active days). BTC candles end on day 20458 (2025-12-30); its weight is rescaled away after that.
- Weights: HRP largest TRX 0.2175, BTC 0.1211, XRP 0.1201, XMR 0.1184, SOL 0.0210 smallest; effective number of names 8.80. Minimum variance: 6 of 13 names at zero, TRX 0.3897, BTC 0.2079, effective number 4.08. Inverse variance 11.15, equal weight 13.
- Primary stream, volatility (bp of capital per day, in-sample / out-of-sample): equal weight 266.95 / 217.88; inverse variance 237.27 / 199.90; HRP 217.43 / 189.42; minimum variance 195.09 / 184.13. Against equal weight: in-sample -11.1%, -18.5%, -26.9%; out-of-sample -8.3%, -13.1%, -15.5%. Out-of-sample sd is lower than in-sample for every scheme (ratio 0.82 to 0.94; minimum variance 0.94). The risk-based schemes keep part of their in-sample advantage; minimum variance, lowest out-of-sample sd on the primary stream, is below HRP by 5.3 bp (2.8%), so the book's claim of HRP below the optimiser out of sample is not reproduced here for this stream.
- Primary stream, return: mean bp/day 0.60, 0.84, 0.82, -0.65; annualised Sharpe 0.05, 0.08, 0.08, -0.07 (equal weight, inverse variance, HRP, minimum variance). By year (Sharpe): 2024 0.76, 0.82, 0.89, 0.66; 2025 -0.42, -0.56, -0.73, -1.37; 2026 -0.48, -0.35, -0.37, -0.30. No spread exists for these single deterministic numbers.
- Meta stream (5 seeds, mean): Sharpe 0.94, 0.91, 0.82, 0.49; daily sd 16.58, 15.85, 16.63, 20.69. Every alternative is below equal weight in Sharpe and mean bp on 5 of 5 seeds. Daily sd is below equal weight on 5 of 5 seeds for inverse variance, 1 of 5 for HRP and 1 of 5 for minimum variance. A possible reason is that the weights were estimated on the primary stream's covariance and the meta stream's positions differ; this was not tested.
- June 2026 window (mean bp/day, equal weight / inverse variance / HRP / minimum variance): primary 37.57 / 34.69 / 25.40 / 13.67; meta 1.31 / 1.08 / 0.79 / 0.55 (each alternative below equal weight on 0 of 5 seeds higher). Sell-off window: primary -64.27 / -63.25 / -63.93 / -53.81; meta -6.67 / -6.58 / -6.70 / -5.40 (minimum variance higher than equal weight on 5 of 5 seeds, +1.27; HRP 2 of 5). No scheme turns the sell-off positive.
- Conclusion: allocation by risk lowers the primary stream's out-of-sample volatility relative to equal weight (HRP -13.1%) but does not improve the Sharpe, and on the meta stream equal weight is best on Sharpe and mean on every seed. No allocation scheme is supported for use. Three new comparisons after look 108; the post-cutoff period is not a pristine holdout; 32% of post-cutoff events fall outside the eligible set.

## Three algorithms (long, short, momentum) and the forest gate on the venues the bot trades (Kraken hourly: run_three_algos_venue.py; Alpaca daily: three_algo_daily.py, run_three_algos_daily.py; tests test_three_algos_venue.py, test_three_algos_daily.py; checkers verify_three_algos_venue.py, verify_three_algos_daily.py)

Why: the earlier three-algorithm study (three_algo.py, run_three_algos.py) ran on GMX / TradingView candles only. The live dry run trades Kraken and Alpaca, so the same study is repeated on both. The original files are NOT edited (the old GMX run's md5 records depend on them).

Design:
- Kraken: run_three_algos.py is run unchanged against the Kraken candle folder recorded in a finished 14-feature Kraken run (only the candle reader is swapped, venue_loader.load_closes). Prior trials 115 (running count 111 after Ch.9 and Ch.16 + the 4 variants that carry a choice), the gate is the 116th; deflated Sharpe at N = 1, 10, 115, 116. Added: a diagnostic of the share of events whose 480 / 720 / 960-bar lookback spans more than 10% away from 20 / 30 / 40 days (Kraken's trade-built candles have gaps).
- Alpaca: three_algo_daily.py is three_algo.py executed with exactly three textual substitutions (each must occur once): the feature module (equity_events), the effective-sample divisor (holding period in days: h + 1 = 5), and the annualisation (sqrt 252). MOM = +1 if the 12-1 month return C[t-21]/C[t-252]-1 and the 6-1 month return C[t-21]/C[t-126]-1 (t = signal bar = bar before entry) are both > 0, -1 if both < 0, else none. Because the source run's side is the sign of the average of those two returns, MOM is the 'both agree' subset of the primary strategy and its side equals the source side (checked on every MOM event); it is not an independent signal as on the hourly venues. F_dir 12 features, F_mom 13. Prior trials 120 (116 after the Kraken study + 4), the gate is the 121st; N = 1, 10, 120, 121. The order Kraken-then-Alpaca is a convention declared before the run.
- Deviations stated in advance: survivorship in the Alpaca list (flatters long momentum), top-200 universe, no high / low, shorts without borrow cost, the gate compares confidences of two different forests, mom_all and mom_long_up use rules chosen on other history. The Alpaca sell-off window (2026-09-27 to 2026-10-08) has NO labelled events (last candle 2026-10-02 and a label needs 5 bars after entry): reported n/a, not filled in. Kraken's design text printed in SUMMARY.md is the unchanged GMX text (87 / 88 trials); the appended venue section states 115 / 116 and the checker requires that sentence.

Commands (project root, conda env javier):
    python tools/book_study/run_three_algos_venue.py --source "$(ls -d runtime/book_seed_pairs/volume_vs_14/ctrl_20261017/*/ | tail -1)"
    python tools/book_study/verify_three_algos_venue.py "$(ls -d runtime/book_three_algo_kraken/*/ | tail -1)"
    python tools/book_study/run_three_algos_daily.py --source runtime/book_meta_alpaca/20261012T023148Z
    python tools/book_study/verify_three_algos_daily.py "$(ls -d runtime/book_three_algo_alpaca/*/ | tail -1)"
    cd tools/book_study && python -m unittest test_three_algos_venue test_three_algos_daily

Verification (before delivery): Kraken full run on a development copy of the 14-feature source run (seed 20261017, 29,929 events, 26 symbols): checker 34/34 base + 5/5 Kraken additions, rerun reproduces the numbers exactly, 13 planted errors in the run folder all caught (12 by a failed check, 1 by the checker stopping). Alpaca full run on the real source run (196,049 events, 495 symbols, closes file md5 2a60b430...): checker 34/34 + 4/4, 27 planted errors all caught; code mutants of three_algo_daily.py / run_three_algos_daily.py / the checker's reader: all caught except one equivalent mutant (a speed-only symbol filter). Tests: 9 + 1 optional (venue), 19 + 1 optional (daily).

Results are in the paragraph below the commands once the Mac runs exist; the development-copy numbers are:

Results, Kraken (development copy of the 14-feature source run; the Mac run should reproduce them if its events.csv is the same file; all CPCV, bp per candidate event, annualised Sharpe in brackets): long_all 24.8 (0.61), short_all -72.8 (-2.40), mom_all 13.2 (0.33), mom_long_up 31.5 (1.77, 24% taken), dir_model 1.6 (-0.82), mom_meta 2.5 (0.79), gate 2.3 (-0.44, 5 of 5 paths negative). Gate against long_all, mom_meta and mom_long_up: higher on 0 of 5 paths each; against shuffled sizes 4 of 5 (+0.76 bp); against random sides 5 of 5. Deflated Sharpe at N = 1 / 10 / 115 / 116: mom_long_up 0.980 / 0.683 / 0.298 / 0.297; mom_meta 0.817 / 0.251 / 0.047 / 0.047; gate 0.310 / 0.019 / 0.001 / 0.001. Walk-forward (fit on labels before 2024-01-01): gate 3.5 bp (0.21), mom_long_up 22.3 (1.39), mom_meta 2.8 (1.38), long_all 4.2 (-0.29). AUC (CPCV mean): F_dir 0.514, F_mom 0.509. Windows (CPCV, bp per candidate): June 2026 long_all -184.4, short_all 136.4, gate 11.0; sell-off 2026-09-27..10-08 (211 events) long_all -119.9, short_all 71.9, mom_all -121.2, mom_long_up -117.8, gate -16.2; the other variants also lost (dir_model -16.5, mom_meta -15.2, gate_shuffled_size -12.7, gate_random_side -3.8). Lookback span diagnostic: 36 / 34 / 63 of 29,929 events (0.12% / 0.11% / 0.21%) have a 480 / 720 / 960-bar span more than 10% away from 20 / 30 / 40 days.
Results, Alpaca (real source run, full study run in the cloud workspace; Mac run pending): 196,049 events, MOM on 152,296 (111,767 long, 40,529 short). CPCV: long_all 9.1 (0.69, 0 of 5 paths negative), short_all -49.1 (-3.32), mom_all -6.5 (-0.67, 5 of 5 negative), mom_long_up 3.7 (0.46, 44% taken), dir_model -1.1 (-0.97), mom_meta 0.0 (0.36, 5% taken, mean size 0.005), gate -1.0 (-0.92, 5 of 5 negative). Gate against long_all, mom_meta, mom_long_up: higher on 0 of 5 paths; against mom_all 1 of 5; shuffled sizes 4 of 5 (-0.23 bp); random sides 5 of 5. Deflated Sharpe at N = 1 / 10 / 120 / 121: mom_long_up 0.739 / 0.175 / 0.025 / 0.025; mom_meta 0.689 / 0.140 / 0.018 / 0.018; gate 0.105 / 0.002 / 0.000 / 0.000. AUC (CPCV mean): F_dir 0.518 (0.506 to 0.529), F_mom 0.520. Walk-forward: long_all 17.6 bp (1.15), gate -0.7 (-1.11), mom_meta 0.1 (0.96, 1% taken), mom_long_up 4.3 (0.48). Windows (CPCV, bp per candidate): June 2026 (1,827 events) long_all -19.8, short_all -20.2, mom_all -41.9, gate +6.9; sell-off 2026-09-27..10-08: 0 events (n/a); deepest fall 2020-02-19..03-23 (2,883 events) long_all -351.9, short_all +311.9, gate -18.8 (walk-forward: 0 events, before the cutoff). Long_all's positive Sharpe is survivor-flattered.
Conclusion: on neither venue does the forest gate beat the plain baselines it is meant to improve on (long_all, mom_meta, mom_long_up), and its Sharpe is negative on every CPCV path on both. The only positive fixed rule (Kraken mom_long_up) was chosen on GMX history and its deflated Sharpe at the declared trial count is 0.30; on Alpaca it is 0.025. Nothing here supports promoting the gate or any of the three algorithms. The runs read only the source run and the closes file and write only to their own output folder; the run's integrity check (models and calibration files unchanged) is meaningful only on the Mac, where the Mac runs should show it true (it is trivially true in my workspace, which holds no live files).

## Volume clock: dollar bars (Table 1.2 item 3, Ch.2), Kraken and Alpaca (build_dollar_bars.py, bar_loader.py, bar_events.py, run_meta_bars.py, run_three_algos_bars.py; checkers verify_dollar_bars.py, verify_meta_bars.py, verify_three_algos_bars.py; tests test_dollar_bars.py, test_bar_events.py, test_verify_dollar_bars.py)

Why: the one pitfall of the book's Table 1.2 not yet covered was the volume clock. New files only; no audited original was edited (the earlier hourly runs' md5 records stay valid).

Data: Kraken = official 1-minute files to 2025-03-31, then 1-minute rows built from trades (runtime/kraken_fine_export); a row is one minute. Alpaca = SIP hourly bars (runtime/alpaca_hourly, 495 symbols); a row is one hour, so an Alpaca bar is a whole number of hours and the clock is only coarsely changed there (Alpaca minute bars were not fetched). The Kraken hour-data question was answered with runtime/kraken_hourly_check/compare.csv (3 pairs x 27 coins).
Definition (declared before any result): row value = (high+low+close)/3 x volume; a bar closes when the cumulative value reaches the trailing 30-day average traded value per clock hour (7 days of history needed; a break longer than 48 h Kraken / 120 h Alpaca ends a segment and no bar spans it); stamped with its END. An hourly reference series from the same rows feeds the market state and Corwin-Schultz.
Pipeline on bars (bar_events.py): CUSUM threshold = one-bar sigma x sqrt(bars per calendar day); Snippet 3.1 daily vol on irregular bars; vertical barrier = a 4-day TIME limit (bar cap Kraken 600, Alpaca 150); momentum lookbacks counted in bars (24/168/720/2160; the three-algo study 480/720/960) and missing across a data hole; Alpaca membership = point-in-time month-end universe. Shown equal to the hourly pipeline on regular hourly synthetic data (test_bar_events.py).
Fingerprint note: the builder's recorded md5 for Kraken BTC and ETH bar files was read before the file was fully saved; a clean rebuild in a scratch folder reproduced the files byte for byte, and the two manifest/meta entries were corrected (bars_md5_note in both).
Trial count (running): Kraken dollar-bar meta run = look 122 (prior 121); Alpaca meta = look 123 (prior 122); Kraken three-algo prior 127 (gate the 128th); Alpaca three-algo prior 132 (gate the 133rd). Running count 133. Kraken-before-Alpaca is a declared convention.

Verification: bars re-derived from source by verify_dollar_bars.py for all 27 Kraken and all 495 Alpaca symbols, every bar, hourly reference, threshold spot-derivations, md5s: all pass. verify_meta_bars.py 39/39 on both meta runs (SADF/entropy recomputed in full for about 300 events; everything else for every event); verify_three_algos_bars.py 34/34 base + 7/7 additions on both three-algo runs. Planted errors (a changed event value, barrier exit, summary number, lookback return, span value) were each caught.
Results: runtime/book_meta_kraken_bars/20261014T052224Z, runtime/book_meta_alpaca_bars/20261014T055028Z, runtime/book_three_algo_kraken_bars/20261014T060345Z, runtime/book_three_algo_alpaca_bars/20261014T062058Z. All numbers: python tools/book_study/collect_bar_results.py <four folders> (output saved as runtime/dollar_bars_transfer/results_dollar_bars.txt). Headline (CPCV, bp per candidate event): Kraken meta-labeling: primary 6.9, meta 0.5, placebo 0.1, AUC 0.507; Alpaca: primary -16.6, meta -0.4, placebo -0.5, AUC 0.507. Learned side (dir_model): Kraken -1.4, Alpaca -1.9, negative Sharpe on 5 of 5 paths on both. Gate: Kraken -0.6, Alpaca -1.9. June 2026 and the 2026-09-27..10-08 window are in each run's SUMMARY.md and in the results file.
Limits: Kraken data hole (official files and the trade files do not cover 2025-04-01 to about 2025-07-07 for most coins; no bar, event path or lookback crosses it; the Kraken API is not reachable from the cloud workspace, so it was not filled); BTC and ICX end 2026-01-05 / 2026-01-04; Alpaca bars are close to hourly; the DESIGN block and deviation list of the three-algo SUMMARY.md are printed unchanged from the hourly version (its '20/30/40 days of hourly bars' and 'time bars not volume bars, hourly candles' sentences do not hold on dollar bars; the appended venue section says so for the lookbacks); the 720-bar primary side and the 200-day state are in-sample choices from earlier history; one seed.
Commands (project root, conda env javier; the heavy runs were done in the cloud workspace):
    python tools/book_study/run_meta_bars.py --venue kraken --bars runtime/dollar_bars/kraken --cost-bp-per-side 12 --prior-trials 121 --sell-off 2026-09-27 2026-10-08 --out runtime/book_meta_kraken_bars
    python tools/book_study/run_meta_bars.py --venue alpaca --bars runtime/dollar_bars/alpaca --cost-bp-per-side 10 --prior-trials 122 --universe runtime/book_meta_alpaca/20261012T023148Z/universe.csv --sell-off 2026-09-27 2026-10-08 --out runtime/book_meta_alpaca_bars
    python tools/book_study/run_three_algos_bars.py --source runtime/book_meta_kraken_bars/20261014T052224Z --prior-trials 127
    python tools/book_study/run_three_algos_bars.py --source runtime/book_meta_alpaca_bars/20261014T055028Z --prior-trials 132
    python tools/book_study/verify_meta_bars.py <meta run folder> --prior-trials 121 --micro-sample 300      (Alpaca: 122)
    python tools/book_study/verify_three_algos_bars.py <three-algo folder> --prior-trials 127              (Alpaca: --prior-trials 132 --cost-bp-per-side 10)

## Verbatim ports of every numbered snippet (Ch.2-21), run on both venues' dollar bars -- 2026-10-14

Why: the earlier bar runs used implementations of several chapters written from memory of the book. The user's instruction is to follow the book snippet by snippet, in order. All 97 numbered snippets of the extract are now ported with the book's own function names (`book_mp.py` = Ch.20; `book_port_a.py` = 2.1-2.3, 3.8, 4.3-4.9, 5.1-5.2, 6; `book_port_b.py` = 7-9; `book_port_c.py` = 10, 13-15, 17, 18; `book_port_d.py` = 16, 21; `book_port_e.py` = 2.4, 3.1-3.7, 4.1, 4.2, 4.10, 4.11, 5.3, 5.4, 19). Each file header lists every change from the printed text (Python 3 / pandas 3 / numpy 2 / scikit-learn 1.x forced changes and typos in the print). Unit tests: `test_book_port_a/b/c/d/e.py` (84 tests, all pass).

Runner: `run_book_ports_bars.py --source <run_meta_bars folder>` (sections ch2 ... ch21, in the book's order), helpers `ports_common.py`, `ports_sections_a/b/c.py`. Output per venue: `SUMMARY.md`, `summary_numbers.json` (with the md5 of every script used), `tables/`, `run.log`. Run folders: `runtime/book_ports_kraken_bars/20261014T130136Z`, `runtime/book_ports_alpaca_bars/20261014T130136Z` (source runs: `book_meta_kraken_bars/20261014T052224Z`, `book_meta_alpaca_bars/20261014T055028Z`). Both: no section failed.

Independent checker: `verify_book_ports_bars.py <run folder>` (plain loops; imports none of the port code). It rebuilds from the raw event and bar files: the Ch.3 class shares; Ch.4 average uniqueness and the Monte-Carlo calibration; every Ch.7 fold (train/test counts, embargo, overlap count); the Ch.8 sample; the Ch.15 bet statistics, required precision, required frequency and window rows; the Ch.2 and Ch.16 daily-return panels; the whole Ch.21 exhaustive search; Snippet 4.6 against its exact expected value by enumeration; and compares the formatted numbers in SUMMARY.md. Result: Kraken 224 of 224, Alpaca 224 of 224. `test_verify_book_ports_bars.py` plants wrong numbers in a copy of a run and requires the checker to flag each one (all caught). Not re-derived by the checker (covered by the port tests and the cross-checks below): Ch.5 minimum-d tables, Ch.6/8/9 forest and grid accuracies, Ch.10 averaged evaluation, Ch.13 mesh, Ch.14 HHI/drawdown/holding period, Ch.16 HRP/CLA weights beyond the inverse-variance portfolio, Ch.18 Lempel-Ziv statistics.

Findings about the book's code (facts from the runs and tests):
- Snippet 3.1 (daily volatility) differs from the project's `bar_events.daily_vol_time` only in the tie rule: the book takes the last bar strictly before t-1 day, the project the last bar at or before it. With that one rule changed in the printed code, the largest relative difference to the project's target is 2.6e-06 or smaller on the three Kraken symbols checked (ARB, OP, BNB; exit times and meta-labels identical, 1.000). Without it the maximum is 0.22 (BNB). On Alpaca (FANG, AEP, LCID, the three symbols with the fewest events among those with at least 200) the same change gives 4.6e-12 or smaller, against up to 0.137 without it. Snippet 2.4 (CUSUM) gives identical events to `book_stats.cusum_events` on real bars (same constant threshold).
- Stored SADF, plug-in entropy and Kontoyiannis features recompute with the book's code to 7.5e-10 (Kraken) / 1.3e-10 (Alpaca) or better; Corwin-Schultz agrees to 2.6e-12 / 8.9e-14.
- Snippet 10.2-10.3: the book's averaging equals the project's at every change point (largest difference 0). The book rounds after averaging, the project before; mean absolute difference 0.0115 (Kraken), 0.0063 (Alpaca).
- Snippet 20.3 as printed records the LAST bar beyond the barrier, not the first (its `continue` only continues the inner loop); on the Kraken resample it differs from the first touch on 1,737 of 1,772 touched paths.
- Snippet 21.2 as printed repeats columns (a zero entry is multiplied by both signs): 80 columns, 38 distinct, for 3 assets.
- Snippet 15.5 as printed uses a standard deviation of p(1-p) in `norm.cdf`, not the standard error of the precision; the "probability of failure" therefore stays near 0.5 for any bet count. Kept as printed, caveat written in SUMMARY.
- Snippet 8.7's generator returns rows grouped by cluster, so contiguous cross-validation folds score below chance on it (recorded in a unit test).
- Snippet 4.6 on the book's own three labels: exact expected average uniqueness 0.6420 (standard bootstrap) and 0.7056 (sequential) by enumeration; the runs' 1,000-draw means are 0.6471 and 0.6990. Table 16.1 is reproduced to its printed two decimals (HRP, IVP, and the CLA stand-in). Snippet 8.6's printed call returns 0.8133.

Run outputs (not recommendations; costs 12 bp per side Kraken, 10 bp Alpaca; CPCV paths):
- Averaged-bet evaluation (Snippets 10.1-10.3), mean bp per active instrument-day, primary / meta / placebo: Kraken 5.58 / 0.16 / -0.15 (annualised Sharpe 0.53 / 0.22 / -0.11); Alpaca -3.13 / -0.12 / -0.28 (-0.61 / -0.65 / -1.96). Windows, same order: June 2026 Kraken 38.24 / 1.63 / 1.01, Alpaca 8.89 / -0.16 / 0.13; sell-off Kraken -47.75 / -4.11 / -1.46, Alpaca -10.07 / 0.18 / -0.30.
- Strategy-risk statistics (15.3-15.5) on CPCV path 0: Kraken primary precision 0.5031, meta 0.5133; Alpaca primary 0.4795, meta 0.4855 (Alpaca's observed precision is below 0.5 and its average loss exceeds its average win in both variants; the required frequency for Sharpe 1 and 2 is "none" there).
- Feature importance, hyper-parameter search, ensembles (purged-CV accuracy against the majority-class share): Kraken 0.512-0.520 against 0.521 (Ch.6 forests, Ch.8); Alpaca 0.528-0.543 against 0.501. In Ch.9 the grid and the random search have a hold-out accuracy equal to the label share and an F1 equal to that of always predicting 1 (Kraken 0.7113, Alpaca 0.7030), which is consistent with a model that predicts one class. Details in each SUMMARY.

Limits (do not drop): Kraken data hole Apr-Jul 2025 (days with fewer than 80% of the subset symbols are left out of the Ch.2/Ch.16 panels, and no return is formed across a gap); BTC and ICX end Jan 2026; Alpaca bars are close to hourly; Ch.19 Kyle/Amihud/Hasbrouck/VPIN/PIN need trade-level data and are not built; the real CLA code and Chapter 22 are not in the extract (Ch.16.5 uses the long-only minimum-variance end point, own code); 5.3 `getWeights_FFD` and 15.3/15.4 `binSR` are called by the printed code but their text is not in the extract (written from the book's definition / Equation 56); 5.4 uses an own ADF regression with MacKinnon constants from memory (statsmodels cannot be installed); 15.2 (SymPy) is replaced by a numeric identity check; 2.2/2.3 are futures-only (ported, unit-tested, not applicable); 8.10's plotting function is ported but the run writes tables. Run sizes are reduced against the printed examples and stated in each section (trees 1,000 -> 100/200, Monte Carlo iterations, 13.1 paths per node 100,000 -> 100, Ch.16.5 10,000 -> 1,000 iterations).

## STANDING RULES FROM THE USER (read before any work; written 2026-10-15 at the user's request so they are never "not recalled")

1. MIRROR THE BOOK. Copy the book's code and equations; the only changes allowed are the ones the language or library versions force, or a typo in the print, and each is listed in the port file's header. Where the book prints equations and no code (Section 2.3 imbalance and run bars, Equations 1-15; PIN and VPIN), there is nothing to copy: do not write code for it, list the formula only, and say there is no code.
2. FOLLOW ONLY THE BOOK. Do not add entries, rules, comparisons, stand-ins, tuning, or explanations that are not in the book. Anything written without the user's knowledge is Claude's job to take out, not to defend. Do not suggest "connecting your own algorithms" or any other fix that is not the book's; the book's Snippets 3.4 and 3.5 take `side` as an input and the extract says nothing about where it comes from, so side-free labelling (side=None, the book's own default) is what the book's code does.
3. BOTH VENUES, ALWAYS: Kraken and Alpaca. No GMX.
4. DO NOT REDO WHAT EXISTS. All 97 numbered snippets are already ported (see the section above); do not rewrite them or rebuild finished work. Do not flip between "I added X", "it was mislabelled" and "it does not work". Fix the one thing, state it once.
5. Numbers come only from saved outputs; where the book is silent, say so in one line; no clarifying questions where avoidable; short replies.

## Write-up of the study in the book's form (book_study_writeup.pdf) -- 2026-10-15

What it is: one entry per entry of the book's Contents (316 of 316; `writeup_coverage.json`), in the book's order and under the book's identifiers, with our Kraken and Alpaca numbers, code location or description. Built by `writeup_build.py` from `items_ch*.json` (written by `writeup_ch01_07.py`, `writeup_ch08_13.py`, `writeup_ch14_15.py`, `writeup_ch16.py`, `writeup_ch17_19.py`, `writeup_ch20_22.py`; helpers `writeup_core.py`, `writeup_panel.py`, `writeup_sims.py`, `writeup_hrpmc.py`); `writeup_verify.py` checks the headline numbers against the saved runs and finds every Contents id in the PDF text. Output under `runtime/book_writeup/`.

Corrections made in this round (so they are not repeated):
- Equation and snippet labels in the first draft were written from memory and did not match the printed pages; they were corrected against the page images. The Contents spelling is kept ("Snipper 8.4", "Equation 29a/29b").
- Six entries that are not in the book's Contents (Table 5.1 Alpaca, Snippet 7.4 our splits, Figure 10.1 ours, Equation 40 our paths, Figures 13.26 and 13.27) were removed from the PDF.
- The sentence about connecting the user's own long/short/momentum algorithms was removed from the PDF and from the text of Table 1.2 rows 1 and 6.
- An imbalance/run-bar construction for Section 2.3 was written and then removed: the book gives no code for it, and on hourly rows it was unstable (Alpaca tick and volume imbalance bars collapsed to one row each). Its two script files were moved to `_to_delete/` for the user to delete.

State of the results (saved outputs): walk-forward AUC about 0.50 on both venues; last-365-day gain Kraken -2.11%, Alpaca -0.74%; DSR near 0. The book prints no end-to-end result on data we hold, so there is no printed number to compare the whole run with. Individual book-printed numbers that ARE reproduced exactly: Table 16.1, Figure 12.1, Figure 16.5 order, Snippet 10.4, and the Chapter 14/15 formula checks (28 of 28).
Not done and not in the book's code: Section 2.3 (equations only), PIN and VPIN, Chapter 22 figures that report another group's experiment.

6. STRICT INSTRUCTIONS, NO JUDGMENT CALLS (added 2026-10-15 at the user's order). The only thing the user permitted to change is the DATA SOURCES (Kraken and Alpaca data in place of the book's data). Any other change to the book's code, equations or run sizes needs the user's yes first, in chat, per change. This covers forced syntax or library fixes, extra arguments or wrappers, substitutes for code the extract does not contain (5.3 getWeights_FFD, 5.4 ADF constants, 15.2, 15.4 binSR, 16.5 CLA), and reduced run sizes (trees, iterations). When the book's code cannot run as printed or is missing: STOP at that item, report it in one line, leave it unfilled, and wait. Do not patch it, substitute for it, shrink it, or decide it is "unavoidable". The existing ports contain such changes (listed in each file's header; summarised in the reply of 2026-10-15); they were made without permission and are open items for the user to rule on, one by one.

## NATENBERG, Option Volatility & Pricing (audiobook companion PDF, 162 pages) - run 2026-10-16
- Output: runtime/book_nat/20261016T095300Z/natenberg_study.pdf (98 pages), items.json (every entry and check). Code: tools/book_study/nat_core.py (the book's formulas as printed), nat_ctx.py, nat_a..nat_e.py (chapters), nat_build.py, nat_verify.py.
- Method: the book prints formulas, not code, so its formulas are what is run. Kraken = ETH, Alpaca = AAPL, both always. Only the data source changes: S = last daily close, sigma = std of the last 365 (Kraken) / 252 (Alpaca) daily log returns x sqrt(365 / 252). Where the book prints its inputs they are used as printed.
- Assumptions where the book gives no number: X = last close (or the book's strike as the same % of our close), 3 months, r = b = 0, calendars 2 and 4 months, theta /365, vega /100, Fig 21-9 time = days/365, tolerances for 'about'/'at the money' are ours and stated in each check.
- Result: 169 entries, 990 checks, 928 pass, 62 DIFFERS (kept as found, listed in the PDF). 11 entries not testable (Fig 1-3, 1-4, 21-14 to 21-19, 24-1 to 24-3: no inputs printed, or need option quotes). Independent verifier (nat_verify.py): 1,176 checks, 0 failed.
- Not yet resolved with the user: none of the DIFFERS were changed or tuned.

## NATENBERG REPLACEMENT RUN: the AFML bot with Natenberg's volatility (nat_vol.py, nat_book_events_v2.py, nat_book_bot.py; checker nat_verify_bot.py; comparison nat_vs_base.py) - 2026-10-16
- Idea (user's): the AFML bot stays whole; only what the second book specifies is replaced. Natenberg specifies one thing the bot also has: volatility (Appendix B: sample st. dev. of ln(p_i/p_{i-1}), mean 0, n-1). The book prints formulas, no code, so the formula is coded as printed (nat_vol.py; test_nat_vol.py reproduces Figure B-2's 39.93%).
- Replaced (three textual substitutions in a copy of book_events_v2.py; the originals are not edited): the per-bar sigma (CUSUM threshold, barrier unit) and the daily target trgt (Snippet 3.1) and so log_trgt. Becker-Parkinson (Ch.19) and everything else unchanged. Same data, seed 20261022, costs (Kraken 12 bp, Alpaca 10 bp per side), forest, folds.
- Book silent on the window n: used the replaced estimator's count (336 bars; 100 daily returns), carried over holes as before. Stated assumption.
- Results: runtime/book_bot_nat/kraken_20261016T135434Z, alpaca_20261016T144307Z (first Alpaca run was killed by a workspace restart: folder alpaca_20261016T142636Z_killed_partial, not used). Independent checker (own Appendix B code): Kraken 287,454 numbers 0 problems; Alpaca 1,529,354 numbers 0 problems; the baseline run fails it on sigma/trgt (so the swap is real) and a planted label error is caught.
- Comparison: runtime/book_bot_nat/COMPARISON.md (numbers read from saved meta.json / summary.csv).

## 2026-11-14: golden cross bar on both venues; the second book set (algorithmic_trading_book); standing instruction to write down and commit

Standing instruction from the user (2026-11-14): write down each thing as it is done and commit it as it is done. Secrets are never committed (.gitignore unchanged).

### 1. Golden cross bar (the user's rule, unchanged), Kraken and Alpaca - `_claude_scratch/gc_bar_20261114/`
- Kraken: `run()` from `tools/run_golden_cross_variants.py` unchanged (4h bars, 6 assets, 12 bp per side, 15% allocation, -8% stop). `gc_run.py` reproduces `logs/v5_golden_cross_variants.json` exactly (LIVE: 513 trades, PF 2.03, CAGR 59.14%, max DD -52.2%). `kraken_bar.py` runs the same function on `runtime/venue_candles_v2/kraken_1h`. BTC candles there end 2026-01-05; the other five run to 2026-10-12.
- Alpaca: `compute_trend_state`, `backtest_one`, `simulate_portfolio`, `robustness` unchanged from `tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py`, run on daily closes in `runtime/venue_candles/alpaca_1d/alpaca_daily.csv.gz`, robust-on-daily universe. 55 of 60 instruments are in the local file (missing QQQ, VTI, DIA, GLD, PPLT).
- Numbers (saved: `kraken_bar.json`, `alpaca_bar.json`, `alpaca_bar_checked.json`):

| venue | window | golden cross | AFML book bot v2 (`book_bot_v2/RESULTS.md`) |
|---|---|---|---|
| Kraken (LIVE variant) | 2024-01-01 to 2026-10-12 | +107.5%, max DD -21.5%, 145 trades, PF 1.82 | -4.00% |
| Kraken (LIVE variant) | 2025-10-13 to 2026-10-12 | +25.0%, max DD -14.2%, 34 trades, PF 1.70 | -2.11% |
| Alpaca (daily, open positions marked at last close) | entered from 2024-01-01, to 2026-10-02 | +15.15%, max DD -3.64% (closed trades only: -2.13%, 80 trades) | -1.90% |
| Alpaca (daily, open positions marked at last close) | entered from 2025-10-10, to 2026-10-02 | -2.54%, max DD -4.43% (closed trades only: -3.51%, 19 trades) | -0.74% |

- Check (`check.py`, separate code): Kraken 6 of 6 rows match when recomputed from the trade lists. Alpaca window trade counts match (80, 19). Whole-file Alpaca differs: the engine has 404 closed trades (+216.61%), a strict-cross count has 355 (+149.75%), because the engine opens a position on the first bar after the 200-bar warm-up when the 50MA is already above the 200MA. The engine was not changed.
- Caveats: marking open positions at the last close is an addition of this check (the engine counts closed trades only). The Alpaca universe was selected as robust on history that overlaps the windows. Trade lists and equity curves are CSV, which `.gitignore` excludes; they are on disk in the same folder.

### 2. The second book set - `~/Documents/GitHub/algorithmic_trading_book` (Successful Algorithmic Trading; Advanced Algorithmic Trading)
- Strategy code present and what each imports: SAT ch.14 moving average crossover 100/400 (numpy, pandas, the book's own backtester); SAT ch.15 intraday pairs mean reversion (statsmodels) and S&P forecast (sklearn.qda); AAT Kalman pairs, cointegration Bollinger, monthly rebalance, sentiment (qstrader); AAT HMM regime detection (qstrader, hmmlearn); AAT intraday ML (qstrader, sklearn); AAT ARIMA+GARCH (R).
- Installed where the code runs (2026-11-14): numpy 2.2.6 and pandas 2.3.3 only; no network. So only SAT ch.14 can be attempted.
- SAT ch.14 run as printed - `_claude_scratch/sat_ch14_mac_20261114/` (`book_code/` is an unchanged copy; `run_as_printed.py` is the book's `__main__` block with only `csv_dir` supplied; data: AAPL daily closes from the Alpaca file in the book's column layout, open/high/low/adj_close set to close because the file has close and volume only).
- RESULT: STOPPED. `data.py` line 130, `.sort()`: `AttributeError: 'DataFrame' object has no attribute 'sort'` (book written for pandas 0.16.2; `run_as_printed_AAPL_error.txt`). Change it would need: `.sort()` -> `.sort_index()`. NOT made; waiting for the user's yes. No result exists for this strategy yet.

### 3. SAT ch.14 moving average crossover: made to run, then tested against the golden cross bar (2026-11-14, after the user said "do what you need to, write it all down and commit EACH thing you do")
- Changes to the book's code, one commit each, diffs in `_claude_scratch/sat_ch14_mac_20261114/changes/`, log in `CHANGES.md`: (1) `data.py` line 130 `.sort()` -> `.sort_index()`; (2) `mac.py` line 61 `bars != []` -> `len(bars) > 0`. Nothing else in the book's code was changed.
- The book's own backtest as printed after those two changes (AAPL daily closes, 100 shares per order, the book's commission): Total Return 24.30%, Sharpe 11.75, Max Drawdown 8.58%, 9 fills (`run_step2.txt`). Finding: the book's `output_summary_stats` annualises with `periods=252*60*6.5` (minute bars) although the data is daily; recomputed with 252 from the saved `equity.csv` the Sharpe is 0.59 and the max drawdown -7.20%. Not changed.
- The fixed 100-unit order makes the book's own backtest meaningless on crypto (100 BTC against 100,000 of capital), so the book's rule (short MA > long MA: long; short < long: exit; 100/400) was put inside the user's golden cross engines with only the two window lengths changed:

| venue | window | golden cross 50/200 | book 100/400 | ahead |
|---|---|---|---|---|
| Kraken 4h, LIVE | 2024-01-01 to 2026-10-12 | +107.5% (max DD -21.5%, PF 1.82) | +44.4% (max DD -30.7%, PF 1.23) | golden cross |
| Kraken 4h, LIVE | last 365 days | +25.0% (max DD -14.2%) | +5.5% (max DD -30.9%) | golden cross |
| Kraken 4h, LIVE | whole history | +33503.5% | +7437.1% | golden cross |
| Alpaca daily, open positions marked | entered from 2024-01-01 | +15.15% (max DD -3.64%) | +17.49% (max DD -3.21%) | book, by 2.3 points |
| Alpaca daily, open positions marked | entered from 2025-10-10 | -2.54% | +0.87% | book, by 3.4 points |
| Alpaca daily, open positions marked | whole file | +264.71% (max DD -6.46%) | +247.48% (max DD -4.24%) | golden cross |

- VERDICT: NOT a clear win. Kraken: the golden cross is ahead in every window. Alpaca: the book's windows are slightly ahead in the two recent windows (few closed trades: 24 and 3; the lead is unrealised gains), behind on the whole file and on closed trades from 2024. The golden cross stays the bar.
- Still blocked (libraries not installed where the code runs; no network there): Kalman pairs, cointegration Bollinger, HMM regime detection, intraday ML, monthly rebalance, sentiment (qstrader / hmmlearn / scikit-learn / statsmodels), ARIMA+GARCH (R).

### 4. AAT regime detection (hidden Markov model) as a filter on the golden cross - `_claude_scratch/aat_regime_hmm_20261114/` (2026-11-14; user: "Do whats next, write everything down")
- What the book does: a moving average crossover plus a risk manager that fits a 2-state Gaussian HMM to the traded instrument's returns on data before the backtest, then blocks new long entries while the HMM says the regime is the undesirable (high-volatility) one; exits always allowed; a blocked entry waits for the next cross.
- hmmlearn is not installed where the code runs (no network), so the model is written out in `hmm_np.py` (Baum-Welch, bar-by-bar last Viterbi state). Stated differences from hmmlearn: deterministic start instead of seeded k-means; no covariance priors. The book's code assumes state 0 is the desirable regime; here it is the lower-variance state. Checks in `test_hmm_np.py` all pass (recovery on simulated data, EM never falls, two forward passes agree, brute-force Viterbi, no look-ahead, extreme-outlier regression).
- BUG OF MINE, FOUND AND FIXED (commit "step 2"): the first Kraken run fitted ETH in 2 iterations with a falling log-likelihood, caused by flooring emission densities on ETH's +700% / +175% bars of August 2015. Fixed by exact log-space scaling; the first numbers were discarded and never reported as a result.
- Each instrument's HMM is fitted on returns before 2024-01-01 only; both windows below are after that date. The engines are the user's; the only rule change is that a new entry needs the calm regime.

| venue | window | golden cross | golden cross + HMM filter |
|---|---|---|---|
| Kraken 4h, LIVE | from 2024-01-01 | +107.5%, max DD -21.5%, PF 1.82, 145 trades | +111.2%, max DD -20.3%, PF 1.98, 132 trades |
| Kraken 4h, LIVE | last 365 days | +25.0%, max DD -14.2% | +25.2%, max DD -14.2% |
| Kraken 4h, VALIDATED | from 2024-01-01 | +124.8%, max DD -17.7%, PF 2.48, 89 trades | +106.4%, max DD -17.9%, PF 2.39, 81 trades |
| Kraken 4h, VALIDATED | last 365 days | +28.2%, PF 2.39 | +24.7%, PF 2.01 |
| Alpaca daily, open positions marked | entered from 2024-01-01 | +15.15%, max DD -3.64%, PF 2.35 | +16.02%, max DD -3.12%, PF 2.66 |
| Alpaca daily, open positions marked | entered from 2025-10-10 | -2.54% | -1.94% |

- VERDICT: NOT a clear win. Alpaca: slightly better in both windows (under 1 point). Kraken: slightly better in the LIVE variant, 18 points worse in the VALIDATED variant, flat over the last 365 days. The golden cross stays the bar. ETH's wild state is fitted to its 2015 listing bars (23% per-bar volatility), so ETH is almost never blocked; not tuned.
- What this says: this regime detector reads price only (the instrument's own returns), like the AFML features and the earlier intermarket filter. Three price-only layers have now failed to add a clear edge to the golden cross.
- Sentiment inventory on disk (for the next step): `data/sentiment/social_sentiment_*.csv` are 69 files with ONE row each (a single hourly snapshot, 2026-06-17 23:00 UTC), so there is no sentiment history to backtest with. README section 3.2 records that GDELT news tone was tried as a gate, a feature and a volatility target and did not clear the walk-forward bar. The AAT sentiment strategy (enter when a sentiment score reaches an entry threshold, exit at an exit threshold) needs a dated sentiment series per instrument, which is not on disk.

### 5. CORRECTION to the sentiment inventory in section 4 (2026-11-14): Santiment IS in this project
Section 4 said there was no sentiment history on disk and asked the user for a source. That was wrong: the search output was cut off before the Santiment files and the record was not read through. The user pointed out that Santiment (santiment.net) was the source Claude itself had recommended. What the record holds:
- `market_data_collectors/collect_santiment_social.py`: hourly Santiment poll for BTC (social volume, weighted sentiment, social dominance), written during a 14-day Sanbase Pro trial. Its header records that Santiment's API is lagged about 30 days even on Pro, and that the measured relationship was a 0.12 correlation with the size of the next hour's move and no directional edge.
- `market_data_collectors/data/santiment_btc_social.csv`: 1,016 rows, 1,014 distinct hours, 2026-07-21 19:00 to 2026-09-04 00:00 UTC, collected 2026-08-21 to 2026-10-05. BTC only.
- `tools/run_santiment_social_experiment.py`: social-volume surge (z-score against a 24h baseline) as a predictor of the next hour's direction, walk-forward folds, binomial test, history pulled from the API for 2025-08-22 to 2026-07-22.
- `tools/run_social_exhaustion_experiment.py`: whether an extreme in social attention during an uptrend predicts a reversal at 4, 12 and 24 hours (an exit signal), history 2025-08-24 to 2026-07-25.
- Both experiment scripts and the collector are listed in `.gitignore` because they contain the API key; they stay uncommitted. Both experiments read GMX prices, which the user has since ruled out of this project.
- NOT FOUND: any saved result file for either experiment (searched notes, handoffs, logs/, runtime/ and commit messages). The only recorded outcome is the sentence in the collector's header.
- What this means for a sentiment test against the golden cross bar: the roughly 11 months of Santiment history the experiments used was fetched live and never saved; what is on disk is 45 days of BTC only, and the local Kraken BTC candles end 2026-01-05, so the two do not overlap. A test needs the Santiment history pulled again (API key, network) for the six Kraken assets, and Kraken BTC candles brought up to date. Neither can be done from the shell this code runs in (no network).

### 6. Santiment history pull script (2026-11-14) - `tools/pull_santiment_history.py`
- The user generated a new Santiment API token. The script reads it from `.env` as `SANTIMENT_API_KEY` (never hard-coded, never printed), pulls hourly social_volume_total, sentiment_weighted_total and social_dominance_total for BTC, ETH, XMR, PAXG, SOL and TRX from 2023-01-01 in 60-day pieces with the same `san.get` call the project's earlier scripts used, and writes `runtime/santiment/<ASSET>_social_1h.csv` plus `manifest.json` (which records every piece the API refused and the API's reason). `runtime/` is gitignored, so the data is not committed.
- Not run here: the shell this code runs in has no network and no `san` package (it stops with "The 'san' package is not installed"). The user runs it on the Mac. Slugs used (bitcoin, ethereum, monero, pax-gold, solana, tron) are untested until that run.
- Still needed after the pull for a BTC test: Kraken BTC candles after 2026-01-05 (`runtime/kraken_refresh` has no BTC file).

### 7. Santiment pull result and the sentiment test rule (2026-11-15) - `_claude_scratch/sentiment_santiment_20261115/CHANGES.md`
- Pull: 8,041 hourly rows per asset (BTC, ETH, XMR, PAXG, SOL, TRX), 2025-11-15 to 2026-10-16, no gaps; everything before 2025-11-15 refused (fits a 12-month history limit with a 30-day delay; the API's reason text was truncated by my script). PAXG, TRX and XMR have almost no social volume. No equity coverage, so Alpaca cannot be tested with Santiment.
- The test rule (book's Sentdex strategy mapped to percentiles, no look-ahead, user's engine, pass rule) is written in that CHANGES.md and committed before any result was computed.
- RESULT (2026-11-15, commit "Santiment step 2"): FAIL. Window 2025-12-16 to 2026-10-13, Kraken, user's engine. Golden cross LIVE +23.0% (max DD -14.4%), VALIDATED +18.5% (max DD -6.4%). Book's sentiment rule LIVE -21.4% (max DD -24.2%, 848 trades), VALIDATED -21.3% (845 trades); every asset lost. The state flips about 180 times per asset (mean hold 12.6 h); costs were about 27% of capital, and before costs the trades made only about +5.8%. PAXG's readings are mostly 0, so its state was ON 75% of the time. Checks pass (numbers recomputed, no look-ahead). Alpaca not tested: Santiment has no equity data.
- What it says: the book's rule uses sentiment as a fast in/out switch, and hourly crypto social sentiment is far too noisy for that. It does not test sentiment as a slow regime gate on the golden cross, which is a different rule and needs its own pre-set rule and commit before it is run.
- Test 2 rule (crowd-mood regime gate: block new golden cross entries when the 7-day BTC/ETH/SOL social-volume-weighted sentiment is at or above its trailing 60-day 90th percentile; exits unchanged) is written in `CHANGES.md` and committed before any result.
- Test 2 RESULT (2026-11-15, commit "Santiment step 5"): FAIL. LIVE +23.0% -> +22.8%; VALIDATED +18.5% -> +12.7%. Euphoria on 9.5% of bars. It blocked 3 of 23 fresh golden crosses: PAXG 2026-05-14 (-3.0%), TRX 2026-08-09 (-0.2%) and SOL 2026-08-13 (still open, +42.7%). Over 10 months there were only three euphoric crosses, one of them the biggest winner, so the data cannot separate "euphoria marks tops" from luck. The test needs years of sentiment history and the plan returns 12 months.
- Where this leaves sentiment: two rules tested, both fail on 10 months of Kraken data. Not tested, and each needs its own pre-set rule: euphoria as an EXIT or a size cut on an open position (instead of blocking entries), capitulation (low mood) as a re-entry signal, and news/LLM sentiment instead of social sentiment. Alpaca has no Santiment coverage.

### 8. Committed 2026-11-15: `tools/book_study/llm/fetch_alpaca_news.py` (written 2026-11-02, untracked until now)
- Step 1 of adding *LLMs in Finance* to the AFML bot: builds the book's `news_ticker.csv` input (Ch.7 `score_headlines`) from Alpaca's historical news API, which covers BOTH equities and crypto, for the bot's Kraken and Alpaca symbols. Keys are read from `.env`/environment, sent only in request headers, never printed or written; no key is in the file (checked before committing). Output goes to `runtime/llm_news/` (gitignored).
- Never run: `runtime/llm_news/` does not exist. It is the one sentiment source in the project that covers Alpaca equities as well as Kraken crypto, and Alpaca's news history is not limited to 12 months the way the Santiment plan is.
