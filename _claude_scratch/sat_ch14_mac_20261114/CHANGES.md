# Successful Algorithmic Trading ch.14 (moving average crossover): every change made to the book's code

`book_code_printed/` is the book's code as printed; `book_code/` is the copy that runs. Each change below is one commit,
with its diff in `changes/`. The user said on 2026-11-14: "do what you need to, write it all down and commit EACH thing you do".
The book's own source files are not committed (public repo, commercial book); the diffs and this log are.

### SAT ch.14 change 1: data.py line 130 .sort() -> .sort_index()
Why: pandas 2.3.3 has no DataFrame.sort (the book targets pandas 0.16.2); sort_index is the same operation (sort rows by the date index) under its current name. Diff: changes/01_data_py_sort_index.patch. Run after this change (run_step1.txt): next stop at mac.py line 61, 'bars != []' raises ValueError (truth value of an empty array is ambiguous) on numpy 2.2.6.

### SAT ch.14 change 2: mac.py line 61 'bars != []' -> 'len(bars) > 0'
Why: numpy 2.2.6 raises on comparing an array with an empty list; the line's purpose is to skip a symbol that has no bars yet, which len(bars) > 0 does. Nothing else in the strategy is touched (100/400 windows, long-only, 100 shares per order, the book's commission model). Diff: changes/02_mac_py_empty_check.patch. Run after this change (run_step2.txt): the book's backtest completes on AAPL daily closes from the Alpaca file - Total Return 24.30%, Sharpe Ratio 11.75, Max Drawdown 8.58%, Drawdown Duration 615, 9 signals / 9 orders / 9 fills. These are the book's own printed statistics, not yet checked independently.

### SAT ch.14 step 3: the book's 100/400 crossover against the golden cross bar on Kraken
Script: book_rule_in_bar_engine_kraken.py (the user's run() from tools/run_golden_cross_variants.py; the only difference between the two runs is the window lengths 50/200 -> 100/400; same 4h Kraken bars, 6 assets, 12 bp per side, 15% allocation, -8% stop). Saved: kraken_gc_in_bar_engine.json, kraken_book_in_bar_engine.json (each row carries an independent recomputation from the trade list and equity curve).
Result, LIVE variant (enter whenever short MA > long MA and flat, which is how the book's rule enters): from 2024-01-01 golden cross +107.5% (max DD -21.5%, PF 1.82) vs book 100/400 +44.4% (max DD -30.7%, PF 1.23); last 365 days +25.0% (max DD -14.2%) vs +5.5% (max DD -30.9%, PF 0.30); whole history +33503.5% vs +7437.1%. VALIDATED variant: +124.8% vs +31.5%; +28.2% vs +3.6%; +8481.4% vs +847.6%.
Verdict on Kraken: the book's crossover does NOT beat the golden cross in any window.
