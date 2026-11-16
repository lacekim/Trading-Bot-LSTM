# AFML book bot without the `ffd` feature (2026-11-15). Rule fixed BEFORE any result was computed
Why: the user ruled "follow the book". The `ffd` feature is built with fixed-width fractional-difference weights (getWeights_FFD), which the AFML PDF never prints (Snippet 5.3 calls it, p51), plus the project's own minimum-d search. By the book it has no printed code, so it comes out.
SOURCE TABLE
| what | source |
|---|---|
| the AFML book bot, everything else unchanged (bars, events, labels, weights, the other eleven features, Ch.13, Ch.8, Ch.9, forest, CPCV, walk-forward cutoff 2024-01-01, Ch.10, HRP, costs, seed 20261022, Alpaca universe) | book_bot_v2.py / book_events_v2.py |
| `ffd` removed from FEATURES_BOOK in a copy (nf_book_events_v2.py); nf_book_bot.py imports the copy | the user's ruling; same copy method as the Natenberg and FinBERT runs |
| consequence stated in advance: an event is no longer dropped for lacking an ffd value, so the event count can rise slightly; the ffd column is still written to events.csv for the record but is not a feature | follows from the row above |
Expectation stated in advance: little change. In book_bot_v2's own Ch.8 results on Kraken, ffd has MDA -0.0003 +/- 0.0002 and the worst SFI of the twelve (-0.69853).
Comparison: against book_bot_v2 (twelve features) and the golden cross bar, read the same way as option 1 (compare_option1.py method). Pass rule against the golden cross as before: better return and better return / max drawdown in both windows, on both venues.
