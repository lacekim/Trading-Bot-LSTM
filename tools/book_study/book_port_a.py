"""Chapters 2-6 of the book, Snippets 2.1-2.3, 3.8, 4.3-4.9, 5.1-5.2, 6.1-6.2, ported to Python 3 / pandas 3 / numpy 2 / scikit-learn 1.x with the book's names.

Changes from the printed code (all forced by the language / library versions or by a typo; nothing else is altered):
  2.1  as printed.
  2.2  getRolledSeries reads an HDF file (pytables); the same steps on a DataFrame are `getRolledSeries_df`; rollGaps is as printed (keep='first' spelled as the book's ligature-free form).
  2.3  the printed lines are wrapped in `rolled_prices(raw)`.
  3.8  the printed signature says `minPtc` but the body uses `minPct` (a NameError as printed): minPct is used; `print 'dropped label'` -> print(); Series.argmin() returns a position in current
       pandas, the label is wanted here: idxmin().
  4.3  Series.iteritems() -> items().
  4.5  as printed (pd.Series() gets dtype float).
  4.6  `main` returns the two uniqueness numbers instead of printing them; print statements -> values.
  4.7  xrange -> range; pd.Series() gets dtype int.
  4.8  as printed.  4.9  mainMC uses book_mp.processJobs / processJobs_ (Snippet 20.8-20.10); numIters defaults to 1E6 as printed (the runner passes a smaller number); returns the describe() table.
  5.1  as printed.  5.2  fillna(method='ffill') -> ffill(); the rest as printed.
  6.1  scipy.misc.comb -> scipy.special.comb (same function, moved).
  6.2  `base_estimator=` -> `estimator=` (renamed in scikit-learn 1.2); max_features='auto' (removed in 1.3; for classifiers it meant sqrt) -> 'sqrt'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import comb
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import book_mp as MP


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 2
# ---------------------------------------------------------------------------------------------------------------------
def pcaWeights(cov, riskDist=None, riskTarget=1.):
    # Following the riskAlloc distribution, match riskTarget
    eVal, eVec = np.linalg.eigh(cov)  # must be Hermitian
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1.
    loads = riskTarget * (riskDist / eVal) ** .5
    wghts = np.dot(eVec, np.reshape(loads, (-1, 1)))
    # ctr=(loads/riskTarget)**2*eVal # verify riskDist
    return wghts


def rollGaps(series, dictio={'Instrument': 'FUT_CUR_GEN_TICKER', 'Open': 'PX_OPEN', 'Close': 'PX_LAST'}, matchEnd=True):
    # Compute gaps at each roll, between previous close and next open
    rollDates = series[dictio['Instrument']].drop_duplicates(keep='first').index
    gaps = series[dictio['Close']] * 0
    iloc = list(series.index)
    iloc = [iloc.index(i) - 1 for i in rollDates]  # index of days prior to roll
    gaps.loc[rollDates[1:]] = series[dictio['Open']].loc[rollDates[1:]] - series[dictio['Close']].iloc[iloc[1:]].values
    gaps = gaps.cumsum()
    if matchEnd:
        gaps -= gaps.iloc[-1]  # roll backward
    return gaps


def getRolledSeries_df(series):
    """Snippet 2.2 after the file has been read: the 'Time' index is set by the caller."""
    series = series.copy()
    gaps = rollGaps(series)
    for fld in ['Close', 'VWAP']:
        series[fld] -= gaps
    return series


def getRolledSeries(pathIn, key):
    series = pd.read_hdf(pathIn, key='bars/ES_10k')
    series['Time'] = pd.to_datetime(series['Time'], format='%Y%m%d%H%M%S%f')
    series = series.set_index('Time')
    return getRolledSeries_df(series)


def rolled_prices(raw):
    """Snippet 2.3 (the lines after `raw=pd.read_csv(...)`)."""
    gaps = rollGaps(raw, dictio={'Instrument': 'Symbol', 'Open': 'Open', 'Close': 'Close'})
    rolled = raw.copy(deep=True)
    for fld in ['Open', 'Close']:
        rolled[fld] -= gaps
    rolled['Returns'] = rolled['Close'].diff() / raw['Close'].shift(1)
    rolled['rPrices'] = (1 + rolled['Returns']).cumprod()
    return rolled


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 3
# ---------------------------------------------------------------------------------------------------------------------
def dropLabels(events, minPct=.05, verbose=False):
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        if verbose:
            print('dropped label', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 4
# ---------------------------------------------------------------------------------------------------------------------
def getIndMatrix(barIx, t1):
    # Get indicator matrix
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1_) in enumerate(t1.items()):
        indM.loc[t0:t1_, i] = 1.
    return indM


def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avgU = u[u > 0].mean()  # average uniqueness
    return avgU


def seqBootstrap(indM, sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series(dtype=float)
        for i in indM:
            indM_ = indM[phi + [i]]  # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()  # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi


def example_seq_bootstrap():
    """Snippet 4.6 (`main`): returns (standard sample, standard uniqueness, sequential sample, sequential uniqueness)."""
    t1 = pd.Series([2, 3, 5], index=[0, 2, 4])  # t0,t1 for each feature obs
    barIx = range(t1.max() + 1)  # index of bars
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    std = getAvgUniqueness(indM[phi]).mean()
    phi2 = seqBootstrap(indM)
    seq = getAvgUniqueness(indM[phi2]).mean()
    return list(phi), std, phi2, seq


def getRndT1(numObs, numBars, maxH):
    # random t1 Series
    t1 = pd.Series(dtype=int)
    for i in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    return t1.sort_index()


def auxMC(numObs, numBars, maxH):
    # Parallelized auxiliary function
    t1 = getRndT1(numObs, numBars, maxH)
    barIx = range(t1.max() + 1)
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return {'stdU': stdU, 'seqU': seqU}


def mainMC(numObs=10, numBars=100, maxH=5, numIters=1E6, numThreads=24):
    # Monte Carlo experiments
    jobs = []
    for i in range(int(numIters)):
        job = {'func': auxMC, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
        jobs.append(job)
    if numThreads == 1:
        out = MP.processJobs_(jobs)
    else:
        out = MP.processJobs(jobs, numThreads=numThreads)
    return pd.DataFrame(out).describe()


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 5
# ---------------------------------------------------------------------------------------------------------------------
def getWeights(d, size):
    # thres>0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def fracDiff(series, d, thres=.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].ffill().dropna(), pd.Series(dtype=float)
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue  # exclude NAs
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 6
# ---------------------------------------------------------------------------------------------------------------------
def bagging_accuracy(N=100, p=1. / 3, k=3.):
    """Snippet 6.1: returns (p, probability that more than N/k of the N classifiers are right), i.e. the printed `print p,1-p_`."""
    p_ = 0
    for i in range(0, int(N / k) + 1):
        p_ += comb(N, i) * p ** i * (1 - p) ** (N - i)
    return p, 1 - p_


def three_rf_setups(avgU):
    """Snippet 6.2: the three classifiers (returned, not fitted)."""
    clf0 = RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample', criterion='entropy')
    clf1 = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', class_weight='balanced')
    clf1 = BaggingClassifier(estimator=clf1, n_estimators=1000, max_samples=avgU)
    clf2 = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
    clf2 = BaggingClassifier(estimator=clf2, n_estimators=1000, max_samples=avgU, max_features=1.)
    return clf0, clf1, clf2
