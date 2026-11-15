# Santiment sentiment against the golden cross bar (2026-11-15)

## What the Santiment pull returned (runtime/santiment/, pulled 2026-11-15 07:00 UTC by the user on the Mac)
- Every asset: 8,041 hourly rows, 2025-11-15 07:00 to 2026-10-16 07:00 UTC, all three metrics, no gaps. 51 pieces refused per asset, all before 2025-11-15.
- The received range is exactly one year back from the pull and stops 30 days before it, which fits a plan limit of 12 months of history and a 30-day delay. The API's own reason text was cut to 300 characters by my script and does not show the rule, so this is read from the dates, not from the message.
- Social volume, median per hour: BTC 106, ETH 24, SOL 17, XMR 3, TRX 2, PAXG 0. PAXG, TRX and XMR sentiment is mostly built from a handful of posts or none.
- Santiment covers crypto social data; it has no series for the 55 Alpaca equities. Alpaca cannot be tested with this source.

## Rule, fixed BEFORE any result was computed (book: Advanced Algorithmic Trading, Sentdex sentiment strategy)
The book: enter long when a ticker's sentiment reading is >= 6 (the top of Sentdex's -3..6 scale), close when it is <= -1 (the bottom three levels). Santiment's weighted sentiment has no fixed scale (mean about 0, standard deviation 0.09 to 0.34 by asset), so the book's levels are mapped to percentiles of each asset's own recent readings:
- 4h sentiment = mean of sentiment_weighted_total over the four hours of the 4h bar.
- No look-ahead: the decision at the close of bar t uses the 4h sentiment of bar t-1, ranked against bars t-180..t-1 (30 days, past only).
- Entry ("sentiment >= 6"): percentile >= 90%. Exit ("sentiment <= -1"): percentile <= 30%. In between, the previous state holds.
- Engine: the user's run() from tools/run_golden_cross_variants.py with only the signal column replaced (gc = the sentiment state above). Same 4h Kraken bars, six assets, 12 bp per side, 15% allocation, -8% stop. LIVE and VALIDATED both reported, as before.
- Window: from the first bar with 180 sentiment bars of history (about 2025-12-15) to 2026-10-12. The golden cross is run on exactly the same window and assets for the comparison. BTC candles end 2026-01-05.
- Pass rule: beats the golden cross on return and on return / max drawdown, in both variants. Anything else is a fail. Sample is about 10 months; a pass would still be a weak result.

### Santiment step 1: record the pull and fix the sentiment test rule before running it
Records what the Santiment pull returned (one year per asset, 2025-11-15 to 2026-10-16, older history refused; almost no social volume for PAXG, TRX, XMR; no equity coverage, so Alpaca cannot be tested with this source) and writes the test rule BEFORE any result exists: the book's Sentdex sentiment strategy (enter at the top of the scale, exit at the bottom three levels) mapped to each asset's own trailing 30-day percentiles (entry >= 90%, exit <= 30%), using only the previous completed 4h bar, inside the user's golden cross engine with only the signal column replaced, compared with the golden cross on the same window. Pass rule stated in advance. Also adds BOOK_NOTES.md section 7. The pulled data stays in runtime/ (gitignored).

### Santiment step 2: the book's sentiment rule against the golden cross on Kraken - FAILS the pre-set pass rule
Scripts: sentiment_state.py (the rule committed in d358369), run_sentiment_vs_gc.py (the user's run() with only the signal column replaced), check_sentiment.py. Window 2025-12-16 to 2026-10-13, six Kraken assets, 4h bars, 12 bp per side, 15% allocation, -8% stop.
Result: golden cross LIVE +23.0% (max DD -14.4%, PF 1.88, 27 trades), VALIDATED +18.5% (max DD -6.4%, 21 trades). Book's sentiment rule LIVE -21.4% (max DD -24.2%, PF 0.71, 848 trades), VALIDATED -21.3% (max DD -24.0%, 845 trades). Every asset lost money.
Why: the state flips about 180 times per asset in ten months (mean hold 12.6 hours). Diagnostic only, not a rule change: costs paid were about 27.3% of starting capital; before costs the same trades made about +5.8%, still far below the golden cross after costs.
Also found: PAXG's state was ON 75% of the time because most of its hourly readings are exactly 0 (almost no posts), which breaks the percentile mapping for that asset. Not tuned.
Checks (check_sentiment_output.txt): all four rows recomputed from the saved files match; cutting the sentiment data at 2026-03-01 and 2026-07-15 never changes an earlier state for ETH, SOL, XMR (no look-ahead).
Not tested: Alpaca (Santiment has no equity data).
