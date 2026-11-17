
### AFML ports compared with BlackArbsCEO/Adv_Fin_ML_Exercises; every difference checked against the PDF
39 shared functions, 13 identical. 9 where the port follows the book and BlackArbs has its own edits; forced Py3/pandas-2 fixes listed; 10 unforced Claude changes listed for removal on the user's yes. No code changed. BOOK_NOTES section 15.

### AFML Snippet 3.6 ruled: the two fixes come from the book's own Snippet 3.3
inst['Close'] -> close and the three over-indented closing lines, both as printed in Snippet 3.3 (p29). User: 'isnt (A) by the book with the fixes?'. Port already matches; no code change. BOOK_NOTES section 16 started.

### AFML Snippet 16.4 copied from the PDF as printed (forced Python 3 changes only) and its runner
Ruling: 'Go by the book, remember?'. snippet_16_4_as_printed.py: 99 lines, diff vs PDF text shows only append->concat, /->// (x2), xrange->range (x2), savefig ligature, en dash, print() (x2). The October cols argument (not in the book) is no longer used for 16.4; its Table 16.1 reproduction is withdrawn as a book result. book_port_d.py not edited. Runner writes runtime/snippet_16_4/.
