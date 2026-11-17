"""AFML Snippet 6.1, "Accuracy of the bagging classifier" (PDF page 58), copied from the PDF.
RULING (user 2026-11-17, "if that is what the book says, ok"): three changes only:
  from scipy.misc import comb -> from scipy.special import comb (SciPy moved the same function; the old import fails);
  xrange -> range; print -> print().
Run on the Mac from the project root:  python tools/book_study/snippet_6_1_as_printed.py
"""
from scipy.special import comb
N,p,k=100,1./3,3.
p_=0
for i in range(0,int(N/k)+1):
    p_+=comb(N,i)*p**i*(1-p)**(N-i)
print(p,1-p_)
