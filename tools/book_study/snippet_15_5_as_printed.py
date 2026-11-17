"""AFML Snippet 15.5, "Calculating the strategy risk in practice" (PDF page 135), copied from the PDF.
RULING (user 2026-11-17, "if that is what the book says, yes"): print -> print(); and 15.5 calls binHR without defining it,
so Snippet 15.3's binHR (p132) is copied in word for word with its own ruled fixes (plain quotes, minus sign), marked below.
Ligatures in names (shufﬂe, ﬂoat) written as plain letters (Python reads them identically).
Kept as printed: ss.norm.cdf(thresP,p,p*(1-p)) passes p*(1-p) as the scale.
Unseeded as printed. Run on the Mac from the project root:  python tools/book_study/snippet_15_5_as_printed.py
"""
import numpy as np,scipy.stats as ss
#———————————————————————————————————————
# ---- from Snippet 15.3 (p132), word for word except ''' and the minus sign ----
def binHR(sl,pt,freq,tSR):
    '''
    Given a trading rule characterized by the parameters {sl,pt,freq},
    what's the min precision p required to achieve a Sharpe ratio tSR?
    1) Inputs
    sl: stop loss threshold
    pt: profit taking threshold
    freq: number of bets per year
    tSR: target annual Sharpe ratio
    2) Output
    p: the min precision rate p required to achieve tSR
    '''
    a=(freq+tSR**2)*(pt-sl)**2
    b=(2*freq*sl-tSR**2*(pt-sl))*(pt-sl)
    c=freq*sl**2
    p=(-b+(b**2-4*a*c)**.5)/(2.*a)
    return p
# ---- end of the copy from Snippet 15.3 ----
#———————————————————————————————————————
def mixGaussians(mu1,mu2,sigma1,sigma2,prob1,nObs):
    # Random draws from a mixture of gaussians
    ret1=np.random.normal(mu1,sigma1,size=int(nObs*prob1))
    ret2=np.random.normal(mu2,sigma2,size=int(nObs)-ret1.shape[0])
    ret=np.append(ret1,ret2,axis=0)
    np.random.shuffle(ret)
    return ret
#———————————————————————————————————————
def probFailure(ret,freq,tSR):
    # Derive probability that strategy may fail
    rPos,rNeg=ret[ret>0].mean(),ret[ret<=0].mean()
    p=ret[ret>0].shape[0]/float(ret.shape[0])
    thresP=binHR(rNeg,rPos,freq,tSR)
    risk=ss.norm.cdf(thresP,p,p*(1-p)) # approximation to bootstrap
    return risk
#———————————————————————————————————————
def main():
    #1) Parameters
    mu1,mu2,sigma1,sigma2,prob1,nObs=.05,-.1,.05,.1,.75,2600
    tSR,freq=2.,260
    #2) Generate sample from mixture
    ret=mixGaussians(mu1,mu2,sigma1,sigma2,prob1,nObs)
    #3) Compute prob failure
    probF=probFailure(ret,freq,tSR)
    print('Prob strategy will fail',probF)
    return
#———————————————————————————————————————
if __name__=='__main__':main()
