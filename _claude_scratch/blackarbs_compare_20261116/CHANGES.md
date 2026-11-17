
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
