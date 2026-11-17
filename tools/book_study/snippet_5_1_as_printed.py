"""AFML Snippet 5.1, "Weighting function" (PDF page 48), copied from the PDF; its __main__ block draws Figures 5.1 and 5.2.
RULING (user 2026-11-17, "if that is the way the book says, ok"): the only change is the three import lines below, which the
snippet uses but does not print (np, pd and mpl are the book's own names; 16.4 prints `import matplotlib.pyplot as mpl`).
Run on the Mac from the project root:  python tools/book_study/snippet_5_1_as_printed.py
Each mpl.show() opens a window; close it to get the next plot.
"""
import numpy as np                 # added: used but not printed in 5.1
import pandas as pd                # added: used but not printed in 5.1
import matplotlib.pyplot as mpl    # added: used but not printed in 5.1 (the book's name, Snippet 16.4)
def getWeights(d,size):
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w
#———————————————————————————————————————-
def plotWeights(dRange,nPlots,size):
    w=pd.DataFrame()
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_=getWeights(d,size=size)
        w_=pd.DataFrame(w_,index=range(w_.shape[0])[::-1],columns=[d])
        w=w.join(w_,how='outer')
    ax=w.plot()
    ax.legend(loc='upper left');mpl.show()
    return
#———————————————————————————————————————-
if __name__=='__main__':
    plotWeights(dRange=[0,1],nPlots=11,size=6)
    plotWeights(dRange=[1,2],nPlots=11,size=6)
