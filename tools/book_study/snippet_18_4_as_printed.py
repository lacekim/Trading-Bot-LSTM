"""AFML Snippet 18.4, "Implementation of algorithms discussed in Gao et al. [2008]" (PDF page 172), copied from the PDF.
RULING (user 2026-11-17, "ok"): ’’’ -> '''; xrange -> range (x2); len(msg)/2 -> len(msg)//2 (x2; Python 2 integer division,
a float is refused by range in Python 3); print -> print() (x2). 18.4 uses np and matchLength without printing them:
the import line is Snippet 18.1's (p168) and matchLength is Snippet 18.3 (p170, xrange -> range), both copied in, marked.
Run on the Mac from the project root:  python tools/book_study/snippet_18_4_as_printed.py
"""
import time,numpy as np   # Snippet 18.1's import line (p168)
#———————————————————————————————————————
# ---- from Snippet 18.3 (p170), word for word except xrange -> range ----
def matchLength(msg,i,n):
    # Maximum matched length+1, with overlap.
    # i>=n & len(msg)>=i+n
    subS=''
    for l in range(n):
        msg1=msg[i:i+l+1]
        for j in range(i-n,i):
            msg0=msg[j:j+l+1]
            if msg1==msg0:
                subS=msg1
                break # search for higher l.
    return len(subS)+1,subS # matched length + 1
# ---- end of the copy from Snippet 18.3 ----
#———————————————————————————————————————
def konto(msg,window=None):
    '''
    * Kontoyiannis’ LZ entropy estimate, 2013 version (centered window).
    * Inverse of the avg length of the shortest non-redundant substring.
    * If non-redundant substrings are short, the text is highly entropic.
    * window==None for expanding window, in which case len(msg)%2==0
    * If the end of msg is more relevant, try konto(msg[::-1])
    '''
    out={'num':0,'sum':0,'subS':[]}
    if not isinstance(msg,str):msg=''.join(map(str,msg))
    if window is None:
        points=range(1,len(msg)//2+1)
    else:
        window=min(window,len(msg)//2)
        points=range(window,len(msg)-window+1)
    for i in points:
        if window is None:
            l,msg_=matchLength(msg,i,i)
            out['sum']+=np.log2(i+1)/l # to avoid Doeblin condition
        else:
            l,msg_=matchLength(msg,i,window)
            out['sum']+=np.log2(window+1)/l # to avoid Doeblin condition
        out['subS'].append(msg_)
        out['num']+=1
    out['h']=out['sum']/out['num']
    out['r']=1-out['h']/np.log2(len(msg)) # redundancy, 0<=r<=1
    return out
#———————————————————————————————————————
if __name__=='__main__':
    msg='101010'
    print(konto(msg*2))
    print(konto(msg+msg[::-1]))
