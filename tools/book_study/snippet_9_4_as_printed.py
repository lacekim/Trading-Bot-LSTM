"""AFML Snippet 9.4, "The logUniform_gen class" (PDF page 82), copied from the PDF; it draws Figure 9.1 (p82).
RULING (user 2026-11-17, "if that what the book says do it"): print -> print() (2 lines) is the only change.
The October port's test_logUniform (returns values, no plot) is Claude's and is not used.
Run on the Mac from the project root:  python tools/book_study/snippet_9_4_as_printed.py   (close the plot window to finish)
"""
import numpy as np,pandas as pd,matplotlib.pyplot as mpl
from scipy.stats import rv_continuous,kstest
#———————————————————————————————————————
class logUniform_gen(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e
    def _cdf(self,x):
        return np.log(x/self.a)/np.log(self.b/self.a)
def logUniform(a=1,b=np.exp(1)):return logUniform_gen(a=a,b=b,name='logUniform')
#———————————————————————————————————————
a,b,size=1E-3,1E3,10000
vals=logUniform(a=a,b=b).rvs(size=size)
print(kstest(rvs=np.log(vals),cdf='uniform',args=(np.log(a),np.log(b/a)),N=size))
print(pd.Series(vals).describe())
mpl.subplot(121)
pd.Series(np.log(vals)).hist()
mpl.subplot(122)
pd.Series(vals).hist()
mpl.show()
