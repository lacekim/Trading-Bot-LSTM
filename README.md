# Trading research and paper-trading project: Kraken (crypto) and Alpaca (equities)

This repository is a research and **paper-trading** project. It trades two venues and tests ideas on both:

- **Kraken**: crypto (BTC, ETH, SOL, XMR, TRX, PAXG and others), run through freqtrade.
- **Alpaca**: U.S. equities, ETFs, commodities/bond ETFs, and the crypto Alpaca can hold (BTC, ETH, SOL, PAXG), through Alpaca's paper-trading API.

**GMX is not used.** An earlier version of this project (called V5) was built on GMX market data. Its code, models and reports are still in the repository as history (file names such as `gmx_*`, `trading_bot_gmx_*`, and the V5 dashboards), but nothing in the current work reads GMX data. The earlier README described that system; it is replaced by this one and remains in the git history.

**No real money is at risk.** Everything that places orders places paper orders. There is no live-capital mode.

**Where the numbers come from.** Every figure below is copied from a saved output in this repository or its `runtime/` folder (the file is named next to each result). `runtime/` is not tracked by git, so those files exist on the project drive, not on GitHub. Results recorded only in session notes, and not in a saved output, are marked as such.

---

## 1. What is actually running (paper)

The strategy that runs is a **moving-average golden cross**: enter long when the 50-period average crosses above the 200-period average, exit on the death cross. Daily bars for equities, 4-hour bars for crypto. No ML model decides these trades. Backtest cost is 12 bp per side (24 bp round trip).

| Part | What it does | Source |
|---|---|---|
| Kraken, via freqtrade | `GoldenCrossStrategy` (port 8080, unfiltered) and `GoldenCrossRegimeFilteredStrategy` (port 8081, intermarket regime filter, forward-testing beside it) | `FULL_HANDOFF.md`, section C |
| Alpaca paper trader | the same golden cross on the equities/ETF/commodity/bond universe plus BTC/ETH/SOL/PAXG | `tools/run_alpaca_paper_trader.py` |
| Hard stops | -8% for the crypto streams, -12% for equities/bonds/commodities; each validated separately | `FULL_HANDOFF.md`, section E |
| Telegram | status and alerts through one bot ("Test_bot") | `FULL_HANDOFF.md`, section G |
| Pre-trade gate | `tools/trading_gate/`: a deterministic, fail-closed check that only **observes** the paper orders (shadow mode); it blocks nothing | `runtime/trading_gate_shadow/` |

State as of the handoff notes (2026-09-16 to 2026-09-20): the Alpaca book held 43 long positions (42 equities at about 2% of equity, one BTCUSD at about 15%); the Kraken whitelist had 5 pairs with BTC removed, because Alpaca is now the only executor of new BTC entries. The first live firing of the crypto -8% stop was XMR/USD, which closed on 2026-09-17. This is a dated snapshot, not a live status.

A real problem found along the way: the Alpaca paper account turned out to be a 4x-margin account, while every backtest assumed cash. The earlier positions had been sized under that false assumption; the old book was dumped to `runtime/alpaca_leveraged_positions_pre_reset_2026-09-13.csv` and the account was reset to the cash-constrained rule. A related backtest bug (entries sized on total capital without checking cash already in use) was found and fixed; at 45% allocation it had let the backtest deploy up to 270% of capital (`tools/verify_capital_constrained_sizing.py`).

---

## 2. How the tests are run, and why

The aim is to find out, with as little self-deception as possible, whether any idea beats the plain golden cross after costs. The rules used throughout:

- **Both venues, always.** Kraken and Alpaca data. No GMX.
- **Predeclare, then run.** Parameters, windows and pass/fail criteria are written down before the run; a run that fails is not tuned until it passes.
- **One change at a time** against a locked control.
- **Independent checks.** Each experiment has a separate verifier that recomputes the headline numbers by its own code (the `verify_*.py` files), and for the equities/ETF work a deterministic "Critic" step (`tools/critic_claims_check.py`) that rechecks verdicts and that the scripts were not edited after results (`runtime/critic_report.json`).
- **Costs are charged.** Kraken 12 bp and Alpaca 10 bp per side in the book study; 12 bp per side in the golden-cross backtests.
- **Failures are kept**, not deleted, and the order of tests is recorded in `runtime/test_ledger.json`.
- **Source-book tests mirror the book.** For the two books below, the book's own code or formulas are used, and only the data source changes (Kraken and Alpaca in place of the book's data). Where a book prints equations and no code, no code is written.

---

## 3. Results

### 3.1 Equities golden cross: exit-logic search and validation
Source: `EQUITIES_EXIT_LOGIC_VALIDATION_RECORD.md` (2026-09-27); results in `runtime/*_results.json`. Universe: 60 symbols (57 equities, 2 commodity futures, 1 Treasury ETF), 1962 to 2026, yfinance data, $100,000 start, 2% of capital per position, -12% stop, 24 bp round trip.

**Exit search: closed, nothing adopted.** Ten exit variants (ATR trailing stop, Chandelier, time-stagnation, partial profit-taking, at several thresholds) were tested against the locked control, with an out-of-sample holdout and block-bootstrap tests. The result was a unanimous reject; the two in-sample candidates (100-day stagnation, +100% partial profit) failed to reduce drawdown out of sample. The control stays unchanged.

**Then five checks on the control**, one at a time:

| Check | Result |
|---|---|
| Execution timing (fill at next open instead of same close) | CAGR 8.64% in both; no change |
| Gap-through-stop (fill at the gap-open) | 85 of 578 stop exits gapped, average 2.76 points worse than the stop; CAGR 8.64% to 8.57%, max drawdown -24.62% to -25.02% |
| Cost and liquidity | CAGR 8.09% to 8.64% across five scenarios, including 74 bp round trip and a 1%-of-volume cap; the volume cap costs more than tripling the spread |
| Independent equity-curve rebuild | all 1,785 trades and the daily equity matched to rounding |
| Universe and concentration | CAGR 8.46% to 9.16% across four universe definitions; **results are concentrated** (NVDA alone is more than double the next name; Technology is 30.5% of net P&L; dropping the top 3 winners takes CAGR to 7.78%) |

Control: CAGR 8.64%, Sharpe 0.96, max drawdown -24.62%, 1,785 trades. **Survivorship bias is unresolved**: the yfinance universe contains no delisted names, so these results are not a fully investable history.

### 3.2 Crypto golden cross and other crypto/equity ideas
Source: `FULL_HANDOFF.md`, sections A, C and E.

| Idea | Outcome |
|---|---|
| Intermarket regime filter on the golden cross | Does not improve CAGR and does not close the drawdown gap at any achievable allocation; kept as a lower-drawdown alternative, forward-testing live |
| Hard stop -8% (crypto) and -12% (equities) | Validated separately and adopted; the crypto stop fired live for the first time on 2026-09-17 |
| Fair Value Gap (FVG) | Crypto: weak and inconsistent. Equities, long-only at 2R to 3R: profit factor 1.17 to 1.40 in the backtest, not yet wired into any live strategy. Shorts: no edge anywhere tested |
| ARIMA / ETS / LightGBM direction forecasts | Real edge only at 30-second granularity (57% directional accuracy), where about 3 bp of signal meets about 26 bp of round-trip fees. At 1h/4h the edge mostly disappears; the one narrow survivor (lag-only LightGBM, 54%) selected trades no better than random when combined with the golden cross (p = 0.986). Shelved |
| Sentiment (GDELT news tone) as a confluence gate, a learned feature, and a volatility target | None cleared the walk-forward promotion bar; the apparent volatility lift was ordinary volatility clustering (commit message of `80f38af`) |

### 3.3 Test ledger (`runtime/test_ledger.json`)
Entries 1 to 5 are "recorded from session history; not machine-verified"; entry 6 comes from its script.

| # | Test | Outcome |
|---|---|---|
| 1 | Crypto LightGBM multibot (Kraken data) | Rejected by the Critic |
| 2 | Time-series momentum on ETFs (yfinance) | Fail |
| 3 | Cross-sectional equity momentum (Alpaca daily bars) | Pass (thin) |
| 4 | Time-series momentum paper baseline, ETFs | Fail |
| 5 | TSH control, ETFs (Alpaca data from 2017-07-03) | Fail. Saved numbers: 12-month trend sign CAGR 4.83%, historical-mean sign 9.40%, always-long control 11.11% |
| 6 | Kalman pairs on ETFs (Alpaca data) | Fail. CAGR 0.19%, Sharpe 0.20 against SPY's 15.21% and 0.88; 59 trades (`runtime/alpaca_pairs_results.json`) |

### 3.4 Book 1: *Advances in Financial Machine Learning* (López de Prado), on Kraken and Alpaca
Source: `tools/book_study/BOOK_NOTES.md`; write-up `tools/book_study/book_study_writeup.pdf` (one entry for each of the 316 entries in the book's Contents, in the book's order); runs in `runtime/book_*`.

**Why:** to give the idea "a model can filter or size trades" the book's full treatment (dollar bars, CUSUM events, triple-barrier labels, purged and combinatorial CV, deflated Sharpe, bet sizing, allocation), on the data the project actually trades. All 97 numbered snippets are ported and run on both venues.

**Headline result: no edge.** From the integrated bot:

| | Kraken | Alpaca |
|---|---|---|
| Walk-forward AUC | 0.5063 | 0.5007 |
| Gain, last 365 days | -2.11% | -0.74% |
| Deflated Sharpe | 0.0027 | 0.0003 |
| Whole walk-forward span (from 2024-01-01) | -4.00% | -1.90% |

(`runtime/book_bot_nat/COMPARISON.md`, baseline columns.) The sub-studies agree:

- **Meta-labeling, Kraken hourly:** cross-validated AUC 0.513; the meta model adds about 2.8 bp per candidate event; deflated Sharpe 0.077 once the 104 trials are counted.
- **Meta-labeling, Alpaca daily:** AUC 0.525, but the model sizes almost every event at zero (7% of events taken); the primary side loses 7.3 bp per event after costs, negative on all 5 paths; deflated Sharpe 0.022.
- **Three algorithms (long, short, momentum) chosen by a forest gate:** on neither venue does the gate beat the plain baselines it is meant to improve on, and its Sharpe is negative on every cross-validation path on both. The only positive fixed rule (Kraken "momentum long while the market is above its 200-day mean") has a deflated Sharpe of 0.30 on Kraken and 0.025 on Alpaca at the declared trial count. Nothing supports promoting the gate.
- **Volume features, time-decay weights, hyper-parameter search, bet averaging, HRP allocation:** each was run on the Kraken pipeline across 5 seeds. None produced a change that holds up out of sample (for example, the grid search "found no configuration that improves ranking", and no allocation scheme improved the Sharpe). The sell-off of 2026-09-27 to 2026-10-08 stayed negative in every variant.
- **Alpaca "long everything" looks positive** (Sharpe 0.69 under cross-validation) but is survivor-flattered, because the equity universe is built from names that still exist.

### 3.5 Book 2: *Option Volatility & Pricing* (Natenberg)
Source: `tools/book_study/BOOK_NOTES.md`; `runtime/book_nat/<run>/natenberg_study.pdf` (98 pages).

**Why:** the project's rule for a second book is that the AFML bot stays whole and only the parts the second book itself specifies are replaced inside it. Natenberg's book is about pricing options; the only part with a counterpart in the spot bot is **volatility** (Appendix B: standard deviation of log returns, mean zero, divided by n-1).

- **Formula study (Kraken = ETH, Alpaca = AAPL):** 169 entries, 990 checks, 928 pass, 62 differ from the printed numbers (kept as found and listed), 11 entries cannot be tested because the book prints no inputs. An independent verifier ran 1,176 checks with none failing.
- **Replacing the bot's volatility with Appendix B**, with everything else identical (data, seed, costs, forest, folds):

| | Kraken baseline | Kraken Natenberg | Alpaca baseline | Alpaca Natenberg |
|---|---|---|---|---|
| Walk-forward AUC | 0.5063 | 0.4986 | 0.5007 | 0.4988 |
| Last-365-day gain | -2.11% | -2.11% | -0.74% | -0.74% |
| Deflated Sharpe | 0.0027 | 0.0120 | 0.0003 | 0.0003 |
| Whole span gain | -4.00% | -0.41% | -1.90% | -1.90% |

Result: **no improvement.** The AUC fell on both venues; the Kraken whole-span change comes with a Sharpe near zero (-0.04) and a deflated Sharpe of 0.012, which is not evidence of an edge. The next item in the book (Black-Scholes, Chapter 18) has not been run; items are run one at a time.

### 3.6 Saved CNN/LSTM per-asset models, checked after their holdout
Source: `runtime/post_holdout_replay/`, `post_holdout_baseline/`, `post_holdout_exit_grid/` (2026-10-10; read-only diagnostics, 1h bars).

The earlier CNN/LSTM models were replayed over the candles that came after each model's holdout, and compared with plain moving-average rules that have no fitted parameters. In the replay of the 117 long models the median rank correlation between the model's probability and the next candle's signed return was 0.009, while it tracked the size of the move (0.223): the models see volatility, not direction. The plain 200-day-average long rule had a pooled profit factor of 0.74 over the same period. Neither is promotion evidence; the run itself says so.

### 3.7 Shadow trading gate
Source: `runtime/trading_gate_shadow/shadow_report.md` (2026-10-05). In shadow mode the gate evaluated 9 entries and 9 exits from the paper-order database: 0 of 9 entries approved (position size above limit and missing stop on all 9), all 9 exits approved, 10 events with adapter errors, and the orders matched the strategy's intent. The shadow loop runs until 2026-10-19T23:59:59Z; after that, from `tools/`, run `python -m trading_gate.shadow_report --out ../runtime/trading_gate_shadow`.

---

## 4. What this adds up to

1. The plain golden cross, with a hard stop, is the only strategy here that has survived every test applied to it, and its backtest is survivorship-biased and concentrated in a few winners.
2. Every attempt to add a model, a filter, a sizing scheme or a new exit on top of it has failed to show a durable improvement out of sample, on either venue.
3. The two books were tested as written, on Kraken and Alpaca data. The bot built from the first book has no edge (AUC about 0.50, negative gain over the last year); replacing its volatility with the second book's formula did not change that.
4. This is paper trading. Nothing here is evidence that the system would make money with real capital.

## 5. Known limits and open items

- Survivorship bias in the equities data is unresolved; fixing it needs point-in-time constituent and delisted-security data.
- Kraken candle coverage is uneven: some symbols' candles end before the study window ends (for example, BTC candles in the HRP study end on 2025-12-30), so those symbols drop out of parts of the study.
- The AFML ports contain changes the book's code did not allow without them (listed in each `book_port_*.py` header); they have not yet been ruled on by the project owner. A strict run of the book's unchanged code has not been started.
- Test-ledger entries 1 to 5 and the sections of the handoff notes are session records, not machine-verified outputs.
- Several file and package names still say `v4`/`v5` or `gmx` for compatibility with earlier work.

## 6. Repository map

- `trading_bot_v4/`: the bot package (signals, backtesting, ML trainers, execution). The `signals/ma200_*.py` modules are the golden-cross system.
- `tools/`: experiments (`run_*.py`), checkers (`verify_*.py`), tests (`test_*.py`), dashboards, the Alpaca paper trader, and `restart_bots.sh`.
- `tools/book_study/`: the two book studies (notes, ports, bot, runners, checkers, write-up).
- `tools/trading_gate/`: the shadow gate. `tools/post_holdout_replay/`: the post-holdout diagnostics.
- `model_training/`, `binance_margin_bot/`: earlier model-training work and an ensemble bot.
- `EQUITIES_EXIT_LOGIC_VALIDATION_RECORD.md`, `FULL_HANDOFF.md`: the validation record and the long handoff notes.
- `runtime/` (not tracked by git): saved run outputs, the test ledger and logs.

Environment: `environment.yml` (the project has used a conda environment named `javier`). Keys and tokens belong in `.env`, which is git-ignored; never commit them.
