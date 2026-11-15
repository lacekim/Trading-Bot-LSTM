# FinBERT sentiment (LLMs in Finance Listing 7.2 output) tested three ways against the golden cross (2026-11-15)
The user said "test all of them" to the three book-based options. Each is tested on its own, its rule committed before it runs.
Input for all three: `runtime/llm_news/<venue>/daily_sent.csv` (FinBERT probabilities per ticker per UTC day). net = sent_positive - sent_negative, in [-1, 1].

## Option 3: AAT Sentdex strategy thresholds. Rule fixed BEFORE any result was computed
SOURCE TABLE
| rule | source |
|---|---|
| per ticker: go long when sentiment >= 6; close when sentiment <= -1; act only when a new sentiment reading arrives | AAT sentdex_sentiment_backtest.py / sentdex_sentiment_strategy.py (sent_buy = 6, sent_sell = -1) |
| Sentdex scale is the ten integer levels -3..6. Mapped onto FinBERT net [-1, 1] as ten equal buckets of width 0.2: level 6 = net >= 0.8; levels -3, -2, -1 = net <= -0.4 | NOT in the book: the conversion the user agreed to by asking for this option to be tested. Fixed buckets on FinBERT's own range, not fitted to the data. Before writing this, only the share of days past each threshold was looked at (BTC 1.4% >= 0.8, 5.4% <= -0.4; ETH 1.3% / 7.6%; Alpaca all 6.7% / 10.5%); no returns were looked at, and the thresholds were not changed |
| no look-ahead: the decision at a bar uses the sentiment of the PREVIOUS UTC day only | NOT in the book (Sentdex sentiment arrives as dated events); needed because day D's headlines include ones after the bar closes |
| days with no headlines: the state holds | the book's strategy only acts on sentiment events |
| engines, costs, sizing, stop, assets, windows: the user's golden cross engines unchanged, only the signal replaced. Kraken: run() from tools/run_golden_cross_variants.py (4h bars, BTC ETH XMR PAXG SOL TRX, 12 bp per side, 15% allocation, -8% stop), LIVE and VALIDATED. Alpaca: compute_trend_state / backtest_one / simulate_portfolio from tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py on daily closes, the 55 instruments, 12 bp per side, 2% per trade, open positions marked at the last close | the user's code; same as the golden cross bar of 2026-11-14 |
Windows (same as the bar): Kraken 2024-01-01 to 2026-10-12 and 2025-10-13 to 2026-10-12; Alpaca entries from 2024-01-01 and from 2025-10-10, to 2026-10-02.
Pass rule: better return AND better return / max drawdown than the golden cross, in every window, on both venues (Kraken: both variants). Anything else is a fail.

### FinBERT option 3 step 1: fix the Sentdex-threshold rule before running it
Writes the rule for option 3 with its source table, before any result exists: the AAT Sentdex strategy (long at sentiment >= 6, close at <= -1, act only on new readings) with the Sentdex scale's ten levels mapped onto FinBERT net sentiment as fixed 0.2-wide buckets (entry net >= 0.8, exit net <= -0.4). That mapping is not in the book and is marked so; before writing it only the share of days past each threshold was looked at, no returns, and the thresholds were not changed. Decision uses the previous UTC day's sentiment only (no look-ahead). Run in the user's golden cross engines on both venues with only the signal replaced, same windows as the bar. Pass rule stated in advance. BOOK_NOTES.md section 12 started.

### FinBERT option 3 step 2: Sentdex thresholds on FinBERT sentiment vs the golden cross - FAILS the pre-set pass rule
Scripts: sentdex_state.py (the rule committed in 5d266fc), run_option3.py (the user's Kraken run() and Alpaca backtest_one with only the signal replaced; the golden cross run through the same code first and reproduces the bar exactly), check_option3.py with check_option3_output.txt (all numbers recomputed from saved files; no look-ahead; every entry follows an entry-level reading: ALL CHECKS PASSED).
Kraken LIVE: from 2024-01-01 golden cross +107.5% (max DD -21.5%, 145 trades) vs Sentdex rule +33.8% (max DD -15.3%, 95 trades); last 365 days +25.0% vs +5.1%. Kraken VALIDATED: +124.8% vs +2.5%; +28.2% vs -4.5%.
Alpaca, open positions marked at the last close: entered from 2024-01-01 +15.15% (max DD -3.64%) vs +10.88% (max DD -3.77%); entered from 2025-10-10 -2.54% vs +0.22%.
Fails in five of six windows. Noted: on closed Alpaca trades alone the Sentdex rule made +12.87% from 2024 vs -2.13% for the golden cross, whose lead there is 36 still-open positions.

## Option 2: LLMs in Finance Listing 7.1 per symbol. Rule fixed BEFORE any result was computed
SOURCE TABLE
| rule | source |
|---|---|
| per symbol: next-day return target, that day's headlines joined into one text, MiniLM (sentence-transformers/all-MiniLM-L6-v2) CLS embedding in batches of 16, previous day's return, inner join on date, first 70% of rows train / last 30% test, RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42), y_pred on the test rows | the book, Listing 7.1, p239-241, run from the user's mirror by tools/book_study/llm/run_listing_7_1.py |
| prices and headlines from the project (Kraken daily closes from the 1h candles, Alpaca daily closes; the last 10 years as the book's period="10y"; headlines from news_ticker.csv) | DATA SOURCE, the one allowed change (4 lines replaced in memory, listed in the script header) |
| how y_pred is traded: long while the latest prediction is > 0, flat otherwise; a prediction made from day D's row acts from day D+1 (Kraken: the 4h bars of D+1; Alpaca: the close of D+1) | NOT in the book: the book stops at y_pred and says backtesting should follow. Long-only to match the golden cross. Acting one day late removes the look-ahead from Alpaca headlines published after the close |
| engines, costs, sizing, stop: the user's golden cross engines unchanged, only the signal replaced (as option 3) | the user's code |
| window: from the LAST test-start date among the venue's symbols to the end of data, so every symbol is in its test period; the golden cross is run on the same window | NOT in the book; needed so no training-period prediction is ever traded |
| a symbol whose book run raises an error is recorded and left out on both sides | NOT in the book |
Pass rule: better return AND better return / max drawdown than the golden cross over that window, on both venues (Kraken: both variants).

### FinBERT option 2 step 1: Listing 7.1 runner and the rule for trading its predictions, before running
Adds tools/book_study/llm/run_listing_7_1.py: the book's Listing 7.1 (p239-241) run per symbol from the user's mirror, with only 4 data-source lines replaced in memory (no yfinance; each bot symbol instead of SPY; the project's daily closes, last 10 years; that symbol's headlines). Saves y_pred and y_test per test date; a symbol that errors is recorded and skipped. Tested in the cloud workspace with a stand-in text model and synthetic data: the target equals the next day's return (max difference 1e-16) and a symbol with no news is skipped with its error recorded.
The rule for trading y_pred is written in CHANGES.md with its source table before any result: long while the latest prediction is above 0, acting from the next day (removes look-ahead from after-close Alpaca headlines), in the user's engines, over the window where every symbol is in its test period, golden cross on the same window. That trading rule is not in the book (the book stops at y_pred) and is marked so. Pass rule stated in advance.

## Option 1: FinBERT sentiment as features inside the AFML book bot. Rule fixed BEFORE any result was computed
SOURCE TABLE
| rule | source |
|---|---|
| the AFML book bot, whole and unchanged: dollar bars, CUSUM events, side-free triple-barrier labels, weights, the twelve book features, Ch.13 exit rule, Ch.8 importance, Ch.9 search, bagged forest, CPCV and walk-forward (cutoff 2024-01-01), Ch.10 sizing, HRP, costs (Kraken 12 bp, Alpaca 10 bp per side), seed 20261022, same bars and Alpaca universe | tools/book_study/book_bot_v2.py and book_events_v2.py (AFML), as in the user's rule for a second book: "the AFML bot stays the whole bot" and only what the second book specifies is put inside it, the way Natenberg's volatility was |
| three features added to the twelve: sent_positive, sent_negative, sent_neutral from daily_sent.csv | LLMs in Finance Listing 7.2 output; the book's line after it: "'daily_sent' can be merged with returns, factors, and risk metrics" |
| the sentiment for an event on UTC day D is the latest day with headlines on or before D-1 (carried forward over days without headlines); events before the symbol's first headline have no sentiment and are dropped by the bot's own rule that every feature must exist | NOT in either book. D-1 is the bot's own rule for its daily features (prev = day_no - 1). Carrying forward is needed because the typical symbol has headlines on only 12% (Kraken) / 20% (Alpaca) of days; an exact day match would drop 80-88% of events. The age of the reading is saved per event (sent_age_days) |
| consequence, stated in advance: Kraken headlines start in 2022, so Kraken events before 2022 are dropped and the forest trains on 2022-2023 only (about 2 years instead of the baseline's longer history). Alpaca headlines start in 2015, close to the bars' 2016 start | follows from the row above |
| files: tools/book_study/sent_book_events_v2.py (copy of book_events_v2.py: the three features and the lookup, nothing else) and tools/book_study/sent_book_bot.py (copy of book_bot_v2.py: imports the copy and points it at the venue's daily_sent.csv, nothing else). The originals are not edited | same method as nat_book_events_v2.py / nat_book_bot.py |
Comparison: the bot's own reported numbers (meta.json ch14 for the last 365 days; equity_curves_walk_forward.csv for the whole test span from 2024-01-01), against (a) the golden cross bar and (b) book_bot_v2 without the sentiment features.
Pass rule: better return AND better return / max drawdown than the golden cross in both windows, on both venues.

### FinBERT option 1 step 1: AFML bot copy with three FinBERT features, and the rule, before running
Adds tools/book_study/sent_book_events_v2.py (copy of book_events_v2.py with sent_positive / sent_negative / sent_neutral added to the twelve book features, looked up as of the previous UTC day, carried forward over days without headlines, NaN before a symbol's first headline; the age of each reading is kept as sent_age_days) and tools/book_study/sent_book_bot.py (copy of book_bot_v2.py: imports the copy and points it at runtime/llm_news/<venue>/daily_sent.csv; nothing else changed). The originals are not edited; same method as the Natenberg copies.
Rule and source table in CHANGES.md, written before any result: the AFML bot stays whole (the user's rule for a second book), the three features are LLMs in Finance Listing 7.2's output merged as the book's closing line says; the D-1 as-of lookup is the bot's own daily-feature rule plus a carry-forward that is not in either book (marked). Stated in advance: Kraken training shrinks to 2022-2023 because Kraken headlines start in 2022. Comparison against the golden cross and against book_bot_v2. Pass rule stated in advance.
The lookup was tested alone (the bot needs scikit-learn, which is on the Mac): an event on day D sees headlines dated D-1 or earlier only; unknown symbols and days before the first headline get NaN.
