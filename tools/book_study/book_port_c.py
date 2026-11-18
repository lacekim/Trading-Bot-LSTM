"""Chapters 10 (Snippet 10.4), 13-15, 17 and 18 of the book, ported to Python 3 / pandas 3 / numpy 2 with the book's names.

Changes from the printed code (each forced by the language / library versions or by a typo; nothing else is altered):
  10.1-10.3  `pd.Series()` as printed (October dtype=float removed 2026-11-17, parked item 6); `norm.cdf` is scipy.stats.norm.cdf; mpPandasObj is the port in book_mp.py.  The molecules are the sorted time points, as printed.
  10.4  xrange -> range; `ﬂoat` (ligature) -> float; `m**-2–1` (an en dash in the printed text) -> `m**-2-1`.
  13.2  `print comb_...` -> print(); `from random import gauss` kept (the printed generator).  In the printed layout the three lines `mean,std=...`, `print`, `output1.append` fall outside the
        `for comb_` loop, which would report only the last node; they are inside it here (the output list has one tuple per node, as the surrounding text describes).  `batch` returns output1.
        A node is exited when cP > pt, cP < -sl, or the holding period exceeds maxHP (as printed).
  13.1  main() as printed, with the loops' parameters passed in (nIter, grids) so that a run can be sized; the printed values are the defaults.
  14.1  as printed, wrapped in `getBets(tPos)`.
  14.2  xrange -> range.  The printed `hp=pd.DataFrame(columns=['dT','w'])` is kept.
  14.3  pd.TimeGrouper(freq='M') (removed) -> pd.Grouper(freq='ME').  The three printed lines are wrapped in `getHHIs(ret)`.
  14.4  np.timedelta64(1,'Y') is refused by numpy for a timedelta from a DatetimeIndex difference; the years are the day difference / 365.25.
  15.1  xrange -> range; the print is returned as (mean, std, mean/std).
  15.2  needs SymPy, which is not installed here (and cannot be installed: the package registry is blocked); `factor_variance_check` verifies the identity the snippet factors,
        v = m2 - m1**2 = p*(1-p)*(u-d)**2, numerically on a grid, which is the output the book's text relies on.
  15.3  `b**2–4*a*c` (en dash) -> `b**2-4*a*c`.  15.4: binSR is called by the printed code but its code is not in the extract; it is Equation 56 (given in the same chapter) and is defined below.
  15.5  `ﬂoat` -> float; `np.random.shufﬂe` -> np.random.shuffle; print -> return.
  17.1  `bsadf=None` is compared with a float (`allADF[-1]>bsadf`) which Python 3 refuses: the comparison is `bsadf is None or ...`.
  17.2  the printed indentation of the `if constant!='nc'` block is inconsistent; the block is taken to be inside the function (it is the end of getYX).
  17.3, 17.4  as printed.
  18.1  xrange -> range; `ﬂoat` -> float.
  18.2, 18.3  xrange -> range.
  18.4  xrange -> range; `len(msg)/2` -> `len(msg)//2` (Python 3 division); prints returned.
"""
from __future__ import annotations

from itertools import product
from random import gauss

import numpy as np
import pandas as pd
import scipy.stats as ss

from book_mp import mpPandasObj  # the book: from mpEngine import ... (Chapter 20); parked item 7, 2026-11-17


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 10, Snippets 10.1 - 10.3
# ---------------------------------------------------------------------------------------------------------------------
def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):
    # get signals from predictions
    if prob.shape[0] == 0:
        return pd.Series()
    # 1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal0 = (prob - 1. / numClasses) / (prob * (1. - prob)) ** .5  # t-value of OvR
    signal0 = pred * (2 * ss.norm.cdf(signal0) - 1)  # signal=side*size
    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']  # meta-labeling
    # 2) compute average signal among those concurrently open
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avgActiveSignals(df0, numThreads)
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal1


def avgActiveSignals(signals, numThreads):
    # compute the average signal among those active
    # 1) time points where signals change (either one starts or one ends)
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = list(tPnts)
    tPnts.sort()
    out = mpPandasObj(mpAvgActiveSignals, ('molecule', tPnts), numThreads, signals=signals)
    return out


def mpAvgActiveSignals(signals, molecule):
    '''
    At time loc, average signal among those still active.
    Signal is active if:
        a) issued before or at loc AND
        b) loc before signal's endtime, or endtime is still unknown (NaT).
    '''
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0  # no signals active at this time
    return out


def discreteSignal(signal0, stepSize):
    # discretize signal
    signal1 = (signal0 / stepSize).round() * stepSize  # discretize
    signal1[signal1 > 1] = 1  # cap
    signal1[signal1 < -1] = -1  # floor
    return signal1


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 10, Snippet 10.4
# ---------------------------------------------------------------------------------------------------------------------
def betSize(w, x):
    return x * (w + x ** 2) ** -.5


def getTPos(w, f, mP, maxPos):
    return int(betSize(w, f - mP) * maxPos)


def invPrice(f, w, m):
    return f - m * (w / (1 - m ** 2)) ** .5


def limitPrice(tPos, pos, f, w, maxPos):
    sgn = (1 if tPos >= pos else -1)
    lP = 0
    for j in range(abs(pos + sgn), abs(tPos + 1)):
        lP += invPrice(f, w, j / float(maxPos))
    lP /= tPos - pos
    return lP


def getW(x, m):
    # 0<alpha<1
    return x ** 2 * (m ** -2 - 1)


def main_10_4(pos=0, maxPos=100, mP=100, f=115, wParams=None):
    wParams = {'divergence': 10, 'm': .95} if wParams is None else wParams
    w = getW(wParams['divergence'], wParams['m'])  # calibrate w
    tPos = getTPos(w, f, mP, maxPos)  # get tPos
    lP = limitPrice(tPos, pos, f, w, maxPos)  # limit price for order
    return w, tPos, lP


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 13, Snippets 13.1 / 13.2
# ---------------------------------------------------------------------------------------------------------------------
def batch(coeffs, nIter=1e5, maxHP=100, rPT=np.linspace(.5, 10, 20), rSLm=np.linspace(.5, 10, 20), seed=0, verbose=False):
    phi, output1 = 2 ** (-1. / coeffs['hl']), []
    for comb_ in product(rPT, rSLm):
        output2 = []
        for iter_ in range(int(nIter)):
            p, hp, count = seed, 0, 0
            while True:
                p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0, 1)
                cP = p - seed
                hp += 1
                if cP > comb_[0] or cP < -comb_[1] or hp > maxHP:
                    output2.append(cP)
                    break
        mean, std = np.mean(output2), np.std(output2)
        if verbose:
            print(comb_[0], comb_[1], mean, std, mean / std)
        output1.append((comb_[0], comb_[1], mean, std, mean / std))
    return output1


def main_13_1(nIter=1e5, maxHP=100, rPT=None, forecasts=(10, 5, 0, -5, -10), hls=(5, 10, 25, 50, 100)):
    rPT = rSLm = (np.linspace(0, 10, 21) if rPT is None else rPT)
    out = {}
    for prod_ in product(forecasts, hls):
        coeffs = {'forecast': prod_[0], 'hl': prod_[1], 'sigma': 1}
        out[prod_] = batch(coeffs, nIter=nIter, maxHP=maxHP, rPT=rPT, rSLm=rSLm)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 14, Snippets 14.1 - 14.4
# ---------------------------------------------------------------------------------------------------------------------
def getBets(tPos):
    # A bet takes place between flat positions or position flips
    df0 = tPos[tPos == 0].index
    df1 = tPos.shift(1)
    df1 = df1[df1 != 0].index
    bets = df0.intersection(df1)  # flattening
    df0 = tPos.iloc[1:] * tPos.iloc[:-1].values
    bets = bets.union(df0[df0 < 0].index).sort_values()  # tPos flips
    if tPos.index[-1] not in bets:
        bets = bets.append(tPos.index[-1:])  # last bet
    return bets


def getHoldingPeriod(tPos):
    # Derive avg holding period (in days) using avg entry time pairing algo
    hp, tEntry = pd.DataFrame(columns=['dT', 'w']), 0.
    pDiff, tDiff = tPos.diff(), (tPos.index - tPos.index[0]) / np.timedelta64(1, 'D')
    for i in range(1, tPos.shape[0]):
        if pDiff.iloc[i] * tPos.iloc[i - 1] >= 0:  # increased or unchanged
            if tPos.iloc[i] != 0:
                tEntry = (tEntry * tPos.iloc[i - 1] + tDiff[i] * pDiff.iloc[i]) / tPos.iloc[i]
        else:  # decreased
            if tPos.iloc[i] * tPos.iloc[i - 1] < 0:  # flip
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(tPos.iloc[i - 1]))
                tEntry = tDiff[i]  # reset entry time
            else:
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(pDiff.iloc[i]))
    if hp['w'].sum() > 0:
        hp = (hp['dT'] * hp['w']).sum() / hp['w'].sum()
    else:
        hp = np.nan
    return hp


def getHHI(betRet):
    if betRet.shape[0] <= 2:
        return np.nan
    wght = betRet / betRet.sum()
    hhi = (wght ** 2).sum()
    hhi = (hhi - betRet.shape[0] ** -1) / (1. - betRet.shape[0] ** -1)
    return hhi


def getHHIs(ret):
    """The three printed lines of Snippet 14.3: concentration of positive returns, of negative returns, and of the number of bets per month."""
    rHHIPos = getHHI(ret[ret >= 0])  # concentration of positive returns per bet
    rHHINeg = getHHI(ret[ret < 0])  # concentration of negative returns per bet
    tHHI = getHHI(ret.groupby(pd.Grouper(freq='ME')).count())  # concentr. bets/month
    return rHHIPos, rHHINeg, tHHI


def computeDD_TuW(series, dollars=False):
    # compute series of drawdowns and the time under water associated with them
    df0 = series.to_frame('pnl')
    df0['hwm'] = series.expanding().max()
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm', 'min']
    df1.index = df0['hwm'].drop_duplicates(keep='first').index  # time of hwm
    df1 = df1[df1['hwm'] > df1['min']]  # hwm followed by a drawdown
    if dollars:
        dd = df1['hwm'] - df1['min']
    else:
        dd = 1 - df1['min'] / df1['hwm']
    tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, 'D') / 365.25).values  # in years
    tuw = pd.Series(tuw, index=df1.index[:-1])
    return dd, tuw


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 15, Snippets 15.1 - 15.5
# ---------------------------------------------------------------------------------------------------------------------
def sharpe_of_binomial_draws(n=1000000, p=.55):
    """Snippet 15.1: n draws of +1 (probability p) / -1; returns (mean, std, mean/std)."""
    out = []
    for i in range(n):
        rnd = np.random.binomial(n=1, p=p)
        x = (1 if rnd == 1 else -1)
        out.append(x)
    return np.mean(out), np.std(out), np.mean(out) / np.std(out)


def factor_variance_check(grid=None):
    """Snippet 15.2 (SymPy: factor(m2-m1**2)): the factored form is p*(1-p)*(u-d)**2; returns the largest absolute difference between the two forms on a grid."""
    grid = np.linspace(-2, 2, 9) if grid is None else grid
    worst = 0.
    for p in np.linspace(0, 1, 11):
        for u in grid:
            for d in grid:
                m2 = p * u ** 2 + (1 - p) * d ** 2
                m1 = p * u + (1 - p) * d
                worst = max(worst, abs((m2 - m1 ** 2) - p * (1 - p) * (u - d) ** 2))
    return worst


def binSR(sl, pt, freq, p):
    """Equation 56 with n = freq: Sharpe ratio of `freq` bets that earn pt with probability p and sl otherwise (sl < 0 < pt)."""
    return ((pt - sl) * p + sl) / ((pt - sl) * (p * (1 - p)) ** .5) * freq ** .5


def binHR(sl, pt, freq, tSR):
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
    a = (freq + tSR ** 2) * (pt - sl) ** 2
    b = (2 * freq * sl - tSR ** 2 * (pt - sl)) * (pt - sl)
    c = freq * sl ** 2
    p = (-b + (b ** 2 - 4 * a * c) ** .5) / (2. * a)
    return p


def binFreq(sl, pt, p, tSR):
    '''
    Given a trading rule characterized by the parameters {sl,pt,freq},
    what's the number of bets/year needed to achieve a Sharpe ratio
    tSR with precision rate p?
    Note: Equation with radicals, check for extraneous solution.
    '''
    freq = (tSR * (pt - sl)) ** 2 * p * (1 - p) / ((pt - sl) * p + sl) ** 2  # possible extraneous
    if not np.isclose(binSR(sl, pt, freq, p), tSR):
        return
    return freq


def mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs):
    # Random draws from a mixture of gaussians
    ret1 = np.random.normal(mu1, sigma1, size=int(nObs * prob1))
    ret2 = np.random.normal(mu2, sigma2, size=int(nObs) - ret1.shape[0])
    ret = np.append(ret1, ret2, axis=0)
    np.random.shuffle(ret)
    return ret


def probFailure(ret, freq, tSR):
    # Derive probability that strategy may fail
    rPos, rNeg = ret[ret > 0].mean(), ret[ret <= 0].mean()
    p = ret[ret > 0].shape[0] / float(ret.shape[0])
    thresP = binHR(rNeg, rPos, freq, tSR)
    risk = ss.norm.cdf(thresP, p, p * (1 - p))  # approximation to bootstrap
    return risk


def main_15_5(mu1=.05, mu2=-.1, sigma1=.05, sigma2=.1, prob1=.75, nObs=2600, tSR=2., freq=260):
    ret = mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs)
    return probFailure(ret, freq, tSR)


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 17, Snippets 17.1 - 17.4
# ---------------------------------------------------------------------------------------------------------------------
def get_bsadf(logP, minSL, constant, lags):
    y, x = getYX(logP, constant=constant, lags=lags)
    startPoints, bsadf, allADF = range(0, y.shape[0] + lags - minSL + 1), None, []
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        bMean_, bStd_ = getBetas(y_, x_)
        bMean_, bStd_ = bMean_[0, 0], bStd_[0, 0] ** .5
        allADF.append(bMean_ / bStd_)
        if bsadf is None or allADF[-1] > bsadf:
            bsadf = allADF[-1]
    out = {'Time': logP.index[-1], 'gsadf': bsadf}
    return out


def getYX(series, constant, lags):
    series_ = series.diff().dropna()
    x = lagDF(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] - 1:-1, 0]  # lagged level
    y = series_.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis=1)
        if constant == 'ctt':
            x = np.append(x, trend ** 2, axis=1)
    return y, x


def lagDF(df0, lags):
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy(deep=True)
        df_.columns = [str(i) + '_' + str(lag) for i in df_.columns]
        df1 = df1.join(df_, how='outer')
    return df1


def getBetas(y, x):
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx)
    bMean = np.dot(xxinv, xy)
    err = y - np.dot(x, bMean)
    bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
    return bMean, bVar


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 18, Snippets 18.1 - 18.4
# ---------------------------------------------------------------------------------------------------------------------
def plugIn(msg, w):
    # Compute plug-in (ML) entropy rate
    pmf = pmf1(msg, w)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / w
    return out, pmf


def pmf1(msg, w):
    # Compute the prob mass function for a one-dim discrete rv
    # len(msg)-w occurrences
    lib = {}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    for i in range(w, len(msg)):
        msg_ = msg[i - w:i]
        if msg_ not in lib:
            lib[msg_] = [i - w]
        else:
            lib[msg_] = lib[msg_] + [i - w]
    pmf = float(len(msg) - w)
    pmf = {i: len(lib[i]) / pmf for i in lib}
    return pmf


def lempelZiv_lib(msg):
    i, lib = 1, [msg[0]]
    while i < len(msg):
        for j in range(i, len(msg)):
            msg_ = msg[i:j + 1]
            if msg_ not in lib:
                lib.append(msg_)
                break
        i = j + 1
    return lib


def matchLength(msg, i, n):
    # Maximum matched length+1, with overlap.
    # i>=n & len(msg)>=i+n
    subS = ''
    for l in range(n):
        msg1 = msg[i:i + l + 1]
        for j in range(i - n, i):
            msg0 = msg[j:j + l + 1]
            if msg1 == msg0:
                subS = msg1
                break  # search for higher l.
    return len(subS) + 1, subS  # matched length + 1


def konto(msg, window=None):
    '''
    * Kontoyiannis' LZ entropy estimate, 2013 version (centered window).
    * Inverse of the avg length of the shortest non-redundant substring.
    * If non-redundant substrings are short, the text is highly entropic.
    * window==None for expanding window, in which case len(msg)%2==0
    * If the end of msg is more relevant, try konto(msg[::-1])
    '''
    out = {'num': 0, 'sum': 0, 'subS': []}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    if window is None:
        points = range(1, len(msg) // 2 + 1)
    else:
        window = min(window, len(msg) // 2)
        points = range(window, len(msg) - window + 1)
    for i in points:
        if window is None:
            l, msg_ = matchLength(msg, i, i)
            out['sum'] += np.log2(i + 1) / l  # to avoid Doeblin condition
        else:
            l, msg_ = matchLength(msg, i, window)
            out['sum'] += np.log2(window + 1) / l  # to avoid Doeblin condition
        out['subS'].append(msg_)
        out['num'] += 1
    out['h'] = out['sum'] / out['num']
    out['r'] = 1 - out['h'] / np.log2(len(msg))  # redundancy, 0<=r<=1
    return out
