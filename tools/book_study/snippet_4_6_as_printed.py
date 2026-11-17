"""AFML Snippets 4.3, 4.4, 4.5 and 4.6 (PDF pages 38-40), copied from the PDF: 4.6's main() is the book's example of the
sequential bootstrap and uses 4.3-4.5. Rulings (user 2026-11-17): 4.3 .iteritems() -> .items(); 4.4 and 4.5 as printed;
4.6 print -> print() (4 lines). Nothing else changed. The book prints no call to main(); the runner calls it.
Unseeded as printed: every run draws differently.
"""
import pandas as pd,numpy as np
#———————————————————————————————————————
def getIndMatrix(barIx,t1):
    # Get indicator matrix
    indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
    for i,(t0,t1) in enumerate(t1.items()):indM.loc[t0:t1,i]=1.
    return indM
def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c=indM.sum(axis=1) # concurrency
    u=indM.div(c,axis=0) # uniqueness
    avgU=u[u>0].mean() # average uniqueness
    return avgU
def seqBootstrap(indM,sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:sLength=indM.shape[1]
    phi=[]
    while len(phi)<sLength:
        avgU=pd.Series()
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1]
        prob=avgU/avgU.sum() # draw prob
        phi+=[np.random.choice(indM.columns,p=prob)]
    return phi
def main():
    t1=pd.Series([2,3,5],index=[0,2,4]) # t0,t1 for each feature obs
    barIx=range(t1.max()+1) # index of bars
    indM=getIndMatrix(barIx,t1)
    phi=np.random.choice(indM.columns,size=indM.shape[1])
    print(phi)
    print('Standard uniqueness:',getAvgUniqueness(indM[phi]).mean())
    phi=seqBootstrap(indM)
    print(phi)
    print('Sequential uniqueness:',getAvgUniqueness(indM[phi]).mean())
    return
