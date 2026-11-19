# Confirming re-run of nf_book_bot.py on Kraken after the 2026-11-17/18 edits

Run: runtime/book_bot_nf/kraken_20261119T023401Z (user, Mac, 967 s), same command and settings as
runtime/book_bot_nf/kraken_20261116T070347Z (962 s): --venue kraken --bars runtime/dollar_bars/kraken --cost-bp-per-side 12, seed 20261022.

Result: trading output identical.
- 23 of the 25 output files are byte-identical (events, walk-forward predictions and equity curves, CPCV splits and metrics,
  MDI/MDA/SFI, search and tuned forest, HRP weights, concentration, Ch.13 mesh, Ch.14 daily return and target position,
  ch14_year.json, 10.4 orders, volume capacity, summary.csv, symbols).
- cpcv_predictions.csv.gz: contents equal (only the gzip header differs).
- ch15.json / meta.json: only the approved removals (the two 15.4 fields per target; time_decay_4_11 const/slope),
  run_utc, seconds and the 6 script fingerprints. dsr var_across_cpcv_paths is NaN in both (NaN != NaN in the check).
- Time under water: count 0 in both runs (at most one drawdown episode in the window), so the 365.2425 change had
  nothing to act on in this run.
- Printed output now shows the book's prints: "dropped label 0 0.0007652901855438246" (3.8) and "1.0 0.0" (4.11 const, slope).
- Walk-forward AUC 0.5099 in both.

## Alpaca (same check)
Run: runtime/book_bot_nf/alpaca_20261119T030922Z (user, Mac, 1,240 s) vs alpaca_20261116T072013Z (1,253 s); --cost-bp-per-side 10,
--universe runtime/book_meta_alpaca/20261012T023148Z/universe.csv, seed 20261022.
Result: trading output identical. 22 of 24 output files byte-identical; cpcv_predictions.csv.gz equal in content; ch15.json /
meta.json differ only by the approved removals (two 15.4 fields per target; time_decay_4_11 const/slope), run time and 6 script
fingerprints. Walk-forward AUC 0.4997 in both. Time under water count 0 (nothing for the year-length change to act on).
Printed: "dropped label 0 0.0020497380417772333" (3.8) and "1.0 0.0" (4.11).
