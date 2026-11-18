
### AFML ports compared with BlackArbsCEO/Adv_Fin_ML_Exercises; every difference checked against the PDF
39 shared functions, 13 identical. 9 where the port follows the book and BlackArbs has its own edits; forced Py3/pandas-2 fixes listed; 10 unforced Claude changes listed for removal on the user's yes. No code changed. BOOK_NOTES section 15.

### AFML Snippet 3.6 ruled: the two fixes come from the book's own Snippet 3.3
inst['Close'] -> close and the three over-indented closing lines, both as printed in Snippet 3.3 (p29). User: 'isnt (A) by the book with the fixes?'. Port already matches; no code change. BOOK_NOTES section 16 started.

### AFML Snippet 16.4 copied from the PDF as printed (forced Python 3 changes only) and its runner
Ruling: 'Go by the book, remember?'. snippet_16_4_as_printed.py: 99 lines, diff vs PDF text shows only append->concat, /->// (x2), xrange->range (x2), savefig ligature, en dash, print() (x2). The October cols argument (not in the book) is no longer used for 16.4; its Table 16.1 reproduction is withdrawn as a book result. book_port_d.py not edited. Runner writes runtime/snippet_16_4/.

### AFML Snippet 16.4 run 1 stopped: pandas 3 refuses the printed integer weight Series
User's Mac run: pairs [(4,6),(1,7),(3,8),(3,9),(2,10)], ClusterWarning from the printed linkage call, then TypeError at w[cItems0]*=alpha (w=pd.Series(1,...) is int64; pandas 3.0.2). The book prints the integer 1 in 16.3 and 16.4. Stopped, waiting for the user's ruling.

### AFML Snippet 16.4: integer start weight 1 -> 1. (user ruling A)
pandas 3 refuses w[...]*=alpha on the printed integer Series; the user chose (a): pd.Series(1.,index=sortIx). Only change; header table updated.

### BOOK_NOTES: why the printed code keeps stopping; Snippet 17.2 alignment recorded
Notes the three sources of the stops (Python 2 era code, typesetting errors, unprinted code) at the user's request ('ok, note all this and continue on'). The standing yes for mechanical fixes was offered, not taken as granted. 17.2: 'continue on' taken as (a), reversible; port already matches; no code change.

### AFML Snippet 20.3 copied from the PDF as printed (range, print() only)
Ruling 'a'. 21 lines, diff vs PDF shows only xrange->range (x2) and print(). The printed continue keeps the last touch; the program prints only a timing. Claude's break version in book_mp.py is not used.

### AFML Snippet 20.4 copied from the PDF as printed, with 20.3's barrierTouch alongside
Ruling '(A) is fine then'. Diff vs PDF: only xrange->range and print(); 20.3's barrierTouch copied word for word (20.4 calls it without defining it). As printed the jobs cover 1,000 of 10,000 paths (rows count used to slice columns); kept, noted: the timing is not a speed-up measure vs 20.3. Spawn-mode test: one main1() 2.03 s, no error.

### AFML Snippet 21.4 ruled as printed (unseeded); seed argument excluded
User: 'i want by the book'. 21.4 runs unchanged on Python 3; its unseeded RandomState means each run of the Chapter 21 experiment differs. Claude's seed argument excluded from book results. Behaviour-change list complete; next: forced fixes in book order, then the parked 10.

### AFML Chapter 2 recorded: 2.1 function only, 2.2/2.3 not applicable to Kraken/Alpaca
User 'ok'. No printed call for 2.1; 2.2/2.3 are futures-roll code and the venues have no contracts or rolls. Nothing run or changed.

### AFML Snippet 3.1 ruled: typeset dash -> minus sign
User 'yes'. Only change in 3.1; port already matches. No code change.

### AFML Snippet 3.2 ruled: indent the printed lines into the function; iteritems -> items
User 'ok'. The printed page puts the second half of applyPtSlOnT1 at the left margin (SyntaxError as printed). Port already has both fixes; its NaT columns and dtype=float remain parked items. No code change.

### AFML Snippets 3.3 (as printed) and 3.4 (three lines, no wrapper) recorded
3.3 is clean on the page and needs no change. 3.4: user '(a)': the three printed lines are used as printed where t1 is needed; Claude's addVerticalBarrier wrapper is excluded from book results.

### AFML Snippet 3.5 ruled: ligature retyped as bfill
User: 'if thats is the way the book says, ok'. Only change; port already has it. The double definition of getBins (3.5, 3.7) is left for 3.7.

### AFML Snippet 3.7 ruled: plain quotes, bfill; getBins is 3.7's
User: 'if thats is the way the book say, ok'. Curly-quote docstring delimiters and the bfill ligature retyped. 3.7 replaces 3.5's getBins (and 3.6 replaces 3.3's getEvents); Claude's getBins_3_5 / getEvents_3_3 names excluded from book results.

### AFML Snippet 3.8 ruled (minPct spelling, print(), idxmin); Chapter 3 complete
User: 'if that is the way the book says, ok'. The verbose switch stays a parked item.

### AFML Snippet 4.1 ruled: quotes, fillna ligature, items(), 0 -> 0.
User: 'if thats the way the book says, ok'. Port already matches; no code change.

### AFML Snippet 4.2 ruled: items(), fillna ligature; loose lines used as printed
User: 'if thats the way the book says, ok'. Claude's getTW wrapper excluded from book results, as with 3.4.

### AFML Snippet 4.3 ruled: items() only
User: 'if that is the way the book says, ok'. Tested with the book's three labels on pandas 3. The t1_ rename (parked 4) is not needed.

### AFML Snippets 4.4-4.5 as printed; 4.6 copied from the PDF with print() only, plus runner
User: 'if that is the way the book says, ok'. snippet_4_6_as_printed.py holds 4.3-4.6; diff vs PDF shows only items() and four print() lines. Runner calls main() once (the book prints no call).

### AFML Snippets 4.7 (range only) and 4.8 (as printed) ruled
User 'ok'. Tested 200 draws on pandas 3: no warnings; averages match Figure 4.2.

### AFML Snippet 4.9 ruled: four fixes; the full-size run stands
User 'a'. Indentation of the closing lines, range, mpEngine -> book_mp (Ch.20 snippets), print(). The port's return-instead-of-print is Claude's and noted; the computation is identical, so the 1,000,000-iteration result (0.6074 / 0.6932) stands.

### AFML Snippet 4.10 ruled: items(); loose lines as printed
User 'ok'. Claude's getSampleW wrapper excluded from book results.

### AFML Snippet 4.11 ruled: print() only; Chapter 4 complete
User: 'if thats the way the book says, ok'. The port's changed return value stays parked.

### AFML Snippet 5.1 copied from the PDF; only the three missing imports added
User: 'if that is the way the book says, ok'. 22 lines identical to the PDF text; __main__ draws Figures 5.1/5.2.

### AFML Snippet 5.2 ruled (quotes, ligatures, ffill()); Chapter 5 complete
User 'ok'. pandas 3 removed fillna(method=); ffill() is the same step. 5.3/5.4 stay not run.

### AFML Snippet 6.1 copied from the PDF with its three forced changes
User: 'if that is what the book says, ok'. comb import moved in SciPy, range, print(). Workspace result 0.4812.

### AFML Snippet 6.2 ruled: estimator=, 'sqrt', ligature, imports; Chapter 6 complete
User: 'if that is what the book say, ok'. Both old scikit-learn arguments confirmed to fail on 1.9.1. Setup only; no run.

### AFML Snippet 7.1 ruled: quotes, items()
User: 'if that is what the book says ok'. Port already matches.

### AFML Snippet 7.2 ruled: concat; loose lines as printed
User: 'if that is what the book says ok'. Claude's embargo_usage wrapper excluded from book results.

### AFML Snippet 7.3 ruled: .iloc for the positional lookup; quotes
User: 'if that is what the book says ok'. As printed it raises KeyError on pandas 3 (tested); .iloc keeps the book's positional meaning.

### AFML Snippet 7.4 ruled: import points to 7.3's class; Chapter 7 complete
User: 'if that is what te book says ok'. The ﬁ ligature in identifiers is read as fi by Python (checked).

### AFML Snippet 8.2 ruled: closing lines aligned with the function body
User: 'if that is what the book say ok'. Notes that the port header's 'as printed' was inaccurate for 8.2.

### AFML Snippet 8.3 ruled: shuffle -> permutation (forced on pandas 3; same draws)
User: 'if that is what the book say ok'. Read-only ValueError confirmed; permutation equals shuffle on a copy for the same seed (checked). Parked item 7 resolved as forced.

### AFML Snippets 8.4 and 8.5 recorded: run as printed
Pages clean; tested on pandas 3 with no errors or warnings. No ruling needed (no change).

### AFML Snippet 8.7 ruled: date_range with today's names, joined line, range
User: 'if the book says so, ok'. Minimal date fix tested; the October port's normalize() is noted as Claude's extra. 8.9's printed-size result unaffected (dates are labels).

### AFML Snippet 8.8 ruled: max_samples as a row count; estimator=; mpEngine -> 20.7
User: 'if the book says so yes'. Confirmed on scikit-learn 1.9.1 that the printed max_samples=1. with weights summing to 1 fits each tree on one row.

### AFML Snippet 8.9: exact copy of 7.3-8.10 from the PDF with the ruled fixes, and runner
User: 'do what the book says'. Re-run from an exact copy instead of the October port. Small-size test passed (all three methods, plots, stats.csv). Includes the second split statement in 8.7 (same fix kind).

### AFML Snippet 9.1 ruled: iid dropped, None>0 guard, estimator=, max_samples count
User: 'do what the book says'. Both iid and None>0 confirmed to raise TypeError today; each fix keeps the book's meaning.

### AFML Snippet 9.2 recorded: runs as printed
Tested on scikit-learn 1.9.1, including inside bagging with sample weights. No change.

### AFML Snippet 9.3 ruled: 9.1's four fixes plus the closing-lines indentation
User 'ok'. 9.3's clfHyperFit replaces 9.1's. Port already matches.

### AFML Snippet 9.4 copied from the PDF with print() only
User: 'if that what the book says do it'. 18 lines; diff vs PDF shows only the two print lines. Draws Figure 9.1.

### AFML Snippet 10.1 ruled: as printed plus the norm import
User 'ok'. 10.1-10.3 tested together as printed on pandas 3: no errors or warnings.

### AFML Snippet 10.2 ruled: plain quotes only
User 'ok'. Runs as printed otherwise.

### AFML Snippet 10.3 recorded: runs as printed
No change.

### AFML Snippet 10.4 ruled: range, minus sign; reproduces the book's 97 / 112.3657 / 95; Chapter 10 complete
User 'ok'. main() prints nothing as printed, so the check is the workspace computation with the book's functions and settings.

### AFML Snippet 14.1 ruled: loose lines as printed, no change
User: 'if thats what the book says, ok'. Claude's getBets wrapper excluded. Hand-checked test.

### AFML Snippet 14.2 ruled: range only
User 'ok'. Hand-checked 2.0-day result in the workspace test.

### AFML Snippet 14.3 ruled: pd.Grouper(freq='ME'); usage lines as printed
User 'ok'. Both old names confirmed to fail on pandas 3. Claude's getHHIs wrapper excluded.

### AFML Snippet 14.4 ruled: year as 365.2425 days, 'first'; Chapter 14 complete
User 'ok'. numpy's own year length replaces the October port's 365.25. Hand-checked test.

### AFML Snippet 15.1 copied from the PDF with its three forced changes
User: 'if thats what the book sayes yes'. import, range, print().

### AFML Snippet 15.3 ruled: plain quotes, minus sign; checked against Equation 56
User 'ok'. The implied precision solves Equation 56 back to the target Sharpe ratio.

### AFML Snippet 15.5 copied from the PDF with print() and 15.3's binHR alongside
User: 'if that is what the book says, yes'. Diff vs PDF: only the print line. 15.4 stays not run (binSR not printed).

### AFML Snippets 16.1-16.3 recorded as covered by the 16.4 run; Chapter 16 complete
User: 'if thats what the book says, ok'. Same code and fixes as 16.4. 16.5 stays not run.

### AFML Snippet 17.1 ruled: None comparison keeps Python 2's behaviour
User 'ok'. Only change; port already matches.

### AFML Snippets 17.3-17.4 run as printed; Chapter 17 complete
Chain 17.1-17.4 tested: random walk -0.62 vs explosive end 14.3.

### AFML Snippet 18.1 ruled: range only
User 'ok'. Entropy test values match Equation 67's expectations.

### AFML Snippet 18.2 ruled: range only
User 'ok'. Standard LZ split reproduced.

### AFML Snippet 18.3 ruled: range only
User 'ok'.

### AFML Snippet 18.4 copied from the PDF with its ruled fixes, 18.1's import and 18.3's matchLength alongside
User 'ok'. Diff vs PDF shows only the quotes, range, //, print() lines.

### AFML Snippet 19.1 ruled: .rolling() for the removed functions, minus signs
User: 'if that is what the book says, ok'. Simulated-market test recovers the known spread to the right size. Alpaca file lacks High/Low (noted).

### AFML Snippet 19.2 ruled: minus signs; Chapter 19 complete
User 'ok'.

### AFML Snippet 20.1 copied from the PDF with print() only
User 'ok'.

### AFML Snippet 20.2 copied from the PDF with its three fixes
User 'ok'. Same 8 combinations as 20.1.

### AFML Snippet 20.5 recorded: runs as printed
No change.

### AFML Snippet 20.6 ruled: range only
User 'ok'. Balanced parts for a triangular job.

### AFML Snippet 20.7 ruled: quotes, argList -> pdObj, split line, range, concat
User 'ok'. Empty pd.Series() kept as printed (object dtype under pandas 2+, values unchanged).

### AFML Snippets 20.8 and 20.10 recorded: run as printed
No change.

### AFML Snippet 20.9 ruled: missing imports added
User 'ok'.

### AFML Snippet 20.11 ruled: copyreg and Python 3 method attribute names
User 'a'. Round-trip test of a bound method passes.

### AFML Snippet 20.12 ruled: quotes, copy and pandas imports
User 'ok'.

### AFML Snippet 20.13 ruled: range only
User 'ok'.

### AFML Snippet 20.14 recorded; Chapter 20 ruled code saved as snippet_20_as_printed.py; Chapter 20 complete
User 'ok'. File diffed against the PDF (only ruled fixes) and run with real worker processes.

### AFML Snippet 21.1 ruled: range only
User 'ok'.

### AFML Snippet 21.2 recorded: runs as printed (repeated columns kept)
No change.

### AFML Snippet 21.3 recorded: runs as printed
No change.

### AFML Snippet 21.5 ruled: parameter block at top level
User: 'if that is what the book says ok'. Claude's genParams wrapper excluded.

### AFML Snippets 21.6-21.7 ruled; Chapter 21 experiment copied from the PDF as snippet_21_as_printed.py
User 'ok'. Diff vs PDF: range and the two print lines only.

### Parked item 1: dropLabels verbose switch taken out (book_port_a.py, book_bot.py)
User: 'do what the book says'. dropLabels always prints as printed in 3.8; the book bot's one call no longer passes verbose=True (it already printed). No result changes. BOOK_NOTES section 17.

### Parked item 2: getTimeDecay prints const, slope and returns the weights only; 5 bots and 2 October scripts updated
User: 'Do what the book says'. No trade or result changes; const/slope move from the bots' logs/run.json to the printed output. 17 tests pass in the workspace. Old runs' recorded md5 of book_port files no longer match (old versions in git).

### Parked item 3: NaT pre-creation removed from applyPtSlOnT1 (not needed on pandas 3, tested)
User: 'Do what the book says'. The October claim that pandas 3 needs it was wrong. Chapter 3 tests pass. Bots unaffected.

### Parked item 4: testFunc made the book's version; superseded 8.9 runner removed; test updated
User: 'Do what the book says...'. book_port_b.testFunc = Snippet 8.9 with the ruled fixes only. run_snippet_8_9_as_printed.py removed (in git history). 10 tests pass.

### Parked item 5: plotFeatImportance title uses the book's 'tag='+tag
User: 'you said it, the book calls for a'. One line; the testFunc smoke test (plots) passes.

### Parked item 6: 17 added dtypes on empty Series/DataFrames removed (5 ports + book_mp); 84 tests pass
User: 'Fallow the BOOK!'. None in a bot code path. Also records the finding that the AFML bot still calls binSR, getBets/getHHIs wrappers and the 365.25-day computeDD_TuW.
