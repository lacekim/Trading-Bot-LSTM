"""AFML Snippet 15.1, "Targeting a Sharpe ratio as a function of the number of bets" (PDF page 131), copied from the PDF.
RULING (user 2026-11-17, "if thats what the book sayes yes"): three changes only: the numpy import (used, not printed);
xrange -> range; print -> print(). Unseeded as printed. The October port's version returns the numbers (Claude's) and is not used.
Run on the Mac from the project root:  python tools/book_study/snippet_15_1_as_printed.py
"""
import numpy as np   # import not printed
out,p=[],.55
for i in range(1000000):
    rnd=np.random.binomial(n=1,p=p)
    x=(1 if rnd==1 else -1)
    out.append(x)
print(np.mean(out),np.std(out),np.mean(out)/np.std(out))
