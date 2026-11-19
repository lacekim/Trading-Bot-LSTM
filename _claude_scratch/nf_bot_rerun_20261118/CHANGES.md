
### Confirming re-run of the AFML bot (no ffd) on Kraken: trading output identical after all edits
User's Mac run kraken_20261119T023401Z vs kraken_20261116T070347Z: 23/25 files byte-identical, CPCV predictions equal, JSON differs only by the approved report removals. AUC 0.5099 both.

### Confirming re-run of the AFML bot (no ffd) on Alpaca: trading output identical; both venues confirmed
alpaca_20261119T030922Z vs alpaca_20261116T072013Z: 22/24 files byte-identical, CPCV predictions equal, JSON differs only by the approved removals. AUC 0.4997 both.

### Snippet 16.5 as printed, with runner (changes ruled by user 'ok')
snippet_16_5_as_printed.py: the book's Monte Carlo HRP vs IVP vs CLA, with the ruled changes (CLA import from the gitignored cla_2013 folder; HRP -> snippet_16_4_as_printed; en dash; range; print(); pd.concat). run_snippet_16_5_as_printed.py: not in the book; runs hrpMC() from runtime/snippet_16_5/, tees output, records versions. BOOK_NOTES section 20. CLA code stays out of git.
