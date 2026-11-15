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

### Santiment step 3: write the sentiment result into BOOK_NOTES.md
Appends the result and what it says to section 7 of tools/book_study/BOOK_NOTES.md: the book's sentiment rule fails against the golden cross on Kraken (-21.4% vs +23.0%), mostly because it trades about 180 times per asset and costs take about 27% of capital; it is only +5.8% even before costs. Records that sentiment as a slow regime gate on the golden cross has not been tested and needs its own pre-set rule.

## Test 2: crowd-mood regime gate on the golden cross. Rule fixed BEFORE any result was computed (2026-11-15)
The user's idea: sentiment is what flips a regime, so it should switch the trend system off at turning points, not trade on its own. Test 1 used sentiment as a fast in/out signal and failed. This test leaves the golden cross alone and only blocks NEW entries when the crypto crowd is euphoric (the "crowd is tapped out" idea from the user's earlier social-exhaustion experiment).
- Market mood, hourly: the social-volume-weighted average of BTC, ETH and SOL weighted sentiment, sum(sentiment_i * volume_i) / sum(volume_i). XMR, TRX and PAXG are left out of the mood series because their median social volume is 0 to 3 posts per hour (measured in step 1, before any result).
- Slow version: mean over each 4h bar, then a 7-day rolling mean (42 bars), shifted one bar (decision at the close of bar t uses data through bar t-1).
- Euphoria: the 7-day mood is at or above the 90th percentile of its own trailing 60 days (360 bars, past only).
- Gate: while euphoric, the golden cross may not open a new position in any of the six assets. Exits and the -8% stop are unchanged. Before 60 days of 7-day mood exist (until about 2026-01-21), entries are allowed.
- Engine: the user's run() with one changed line, 'if want:' -> 'if want and bool(r.reg_ok):' (same as the HMM test). Window 2025-12-16 to 2026-10-13, same as test 1. LIVE and VALIDATED both reported.
- Pass rule: the gated golden cross beats the plain golden cross on return AND on return / max drawdown, in BOTH variants. The plain golden cross had 27 (LIVE) and 21 (VALIDATED) trades in this window, so only a few entries can be blocked; a pass would still be weak evidence, and the blocked trades will be listed one by one.
- Alpaca: not possible (Santiment has no equity data).

### Santiment step 4: fix the crowd-mood regime-gate rule before running it
Writes test 2's rule in CHANGES.md and a pointer in BOOK_NOTES.md section 7, before any result exists. The golden cross is left unchanged; the only change is that new entries are blocked while the crypto crowd is euphoric: 7-day mean of the BTC/ETH/SOL social-volume-weighted Santiment sentiment, previous bar only, at or above the 90th percentile of its trailing 60 days. Exits and the -8% stop are untouched. Same window and engine as test 1. Pass rule: better return and better return / max drawdown than the plain golden cross in both LIVE and VALIDATED. Stated in advance: only a few of the 27 / 21 trades in this window can be blocked, so even a pass is weak evidence. Alpaca cannot be tested with Santiment.
