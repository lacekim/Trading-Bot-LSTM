"""AFML Snippet 20.3, "Single-thread implementation of a one-touch double barrier" (PDF page 188), copied from the PDF.
RULING (user 2026-11-16, "a"): run as printed; the only changes are xrange -> range (x2) and print -> print().
As printed, `continue` only continues the inner loop, so t[j] holds the LAST bar beyond a barrier, not the earliest the comment
names; the program prints only the timing, which this does not change. book_mp.py's extra `break` version (Claude's, not in the
book) is not used. Synthetic data only. Run on the Mac from the project root:
    python tools/book_study/snippet_20_3_as_printed.py
"""
import numpy as np
#———————————————————————————————————————
def main0():
    # Path dependency: Sequential implementation
    r=np.random.normal(0,.01,size=(1000,10000))
    t=barrierTouch(r)
    return
#———————————————————————————————————————
def barrierTouch(r,width=.5):
    # find the index of the earliest barrier touch
    t,p={},np.log((1+r).cumprod(axis=0))
    for j in range(r.shape[1]): # go through columns
        for i in range(r.shape[0]): # go through rows
            if p[i,j]>=width or p[i,j]<=-width:
                t[j]=i
                continue
    return t
#———————————————————————————————————————
if __name__=='__main__':
    import timeit
    print(min(timeit.Timer('main0()',setup='from __main__ import main0').repeat(5,10)))
