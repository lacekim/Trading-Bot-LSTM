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
