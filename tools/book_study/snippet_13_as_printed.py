"""AFML Snippets 13.1 and 13.2 (p103), copied from the condensed-edition PDF.

SOURCE TABLE (every difference from the printed page)
| line(s) | printed | here | why |
|---|---|---|---|
| 13.2: mean,std=... / print ... / output1.append(...) | level with `for comb_` (after the node loop) | inside `for comb_` | the book's Step 4 (p102): "for each node we compute the Sharpe ratio"; the figures listed in Table 13.1 show every node. User ruling 2026-11-16: "what does the book say?" -> follow the text |
| 13.2: return output1 | at the left margin, outside `def batch` (SyntaxError) | inside `batch` | the function cannot return anything otherwise; same ruling |
| 13.2: print comb_[0],... | Python 2 print statement | print(comb_[0],...) | Python 3 (forced; on the open forced-changes list) |
Everything else is as printed, including 13.1's main() overwriting `output` on each of the 25 parameter pairs and returning only the last batch (each batch prints every node, which is the book's output).
"""
import numpy as np
from random import gauss
from itertools import product
#———————————————————————————————————————
def main():
    rPT=rSLm=np.linspace(0,10,21)
    count=0
    for prod_ in product([10,5,0,-5,-10],[5,10,25,50,100]):
        count+=1
        coeffs={'forecast':prod_[0],'hl':prod_[1],'sigma':1}
        output=batch(coeffs,nIter=1e5,maxHP=100,rPT=rPT,rSLm=rSLm)
    return output

def batch(coeffs,nIter=1e5,maxHP=100,rPT=np.linspace(.5,10,20),
    rSLm=np.linspace(.5,10,20),seed=0):
    phi,output1=2**(-1./coeffs['hl']),[]
    for comb_ in product(rPT,rSLm):
        output2=[]
        for iter_ in range(int(nIter)):
            p,hp,count=seed,0,0
            while True:
                p=(1-phi)*coeffs['forecast']+phi*p+coeffs['sigma']*gauss(0,1)
                cP=p-seed;hp+=1
                if cP>comb_[0] or cP<-comb_[1] or hp>maxHP:
                    output2.append(cP)
                    break
        mean,std=np.mean(output2),np.std(output2)
        print(comb_[0],comb_[1],mean,std,mean/std)
        output1.append((comb_[0],comb_[1],mean,std,mean/std))
    return output1
