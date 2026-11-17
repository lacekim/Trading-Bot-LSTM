"""AFML Snippet 20.4, "Multiprocessing implementation of a one-touch double barrier" (PDF page 188), copied from the PDF.
RULING (user 2026-11-16, "(A) is fine then"): run as printed. Changes: xrange -> range and print -> print(); and 20.4 calls
barrierTouch without defining it (it relies on Snippet 20.3's code), so 20.3's barrierTouch is copied in word for word, marked
below (on macOS the worker processes can only find it in this file).
As printed, `parts` is built from r.shape[0] (1,000 rows) but used to slice COLUMNS (the paths), so the 24 jobs cover columns
0-1,000 only: 1,000 of the 10,000 paths. The printed timing therefore covers a tenth of 20.3's work and is NOT a measure of the
speed-up over 20.3. Kept as printed (the user's ruling). Synthetic data only. Run on the Mac from the project root:
    python tools/book_study/snippet_20_4_as_printed.py
"""
import numpy as np
import multiprocessing as mp
#———————————————————————————————————————
# ---- from Snippet 20.3 (p188), word for word except xrange -> range ----
def barrierTouch(r,width=.5):
    # find the index of the earliest barrier touch
    t,p={},np.log((1+r).cumprod(axis=0))
    for j in range(r.shape[1]): # go through columns
        for i in range(r.shape[0]): # go through rows
            if p[i,j]>=width or p[i,j]<=-width:
                t[j]=i
                continue
    return t
# ---- end of the copy from Snippet 20.3 ----
#———————————————————————————————————————
def main1():
    # Path dependency: Multi-threaded implementation
    r,numThreads=np.random.normal(0,.01,size=(1000,10000)),24
    parts=np.linspace(0,r.shape[0],min(numThreads,r.shape[0])+1)
    parts,jobs=np.ceil(parts).astype(int),[]
    for i in range(1,len(parts)):
        jobs.append(r[:,parts[i-1]:parts[i]]) # parallel jobs
    pool,out=mp.Pool(processes=numThreads),[]
    outputs=pool.imap_unordered(barrierTouch,jobs)
    for out_ in outputs:out.append(out_) # asynchronous response
    pool.close();pool.join()
    return
#———————————————————————————————————————
if __name__=='__main__':
    import timeit
    print(min(timeit.Timer('main1()',setup='from __main__ import main1').repeat(5,10)))
