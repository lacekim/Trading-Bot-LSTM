"""Chapters 2-5 and 19 of the book that work on a time-indexed price Series, Snippets 2.4, 3.1-3.7, 4.1, 4.2, 4.10, 4.11, 5.3, 5.4 and 19.1-19.2, ported to Python 3 / pandas 3 / numpy 2 with the book's names.
(book_port_a.py has 2.1-2.3, 3.8, 4.3-4.9, 5.1-5.2, 6.x.)

Changes from the printed code (each forced by the language / library versions or by a typo; nothing else is altered):
  2.4   as printed.
  3.1   `close.index[df0–1]` (an en dash) -> `df0-1`.
  3.2   Series.iteritems() -> items(); `pd.Series(index=...)` as printed (October dtype=float removed 2026-11-17, parked item 6).  (The October NaT pre-creation of 'sl'/'pt' was removed 2026-11-17: not needed on pandas 3, tested.)
  3.3   as printed (mpPandasObj is the port in book_mp.py).
  3.4   the three printed lines are wrapped in `addVerticalBarrier(close, tEvents, numDays)`.
  3.5   `bﬁll` ligature -> 'bfill'.   The printed getBins of 3.5 is `getBins_3_5`; the one of 3.7 (which includes it) is `getBins`.  Likewise getEvents_3_3 / getEvents (3.6).
  3.6   the printed body uses `inst['Close']` (an undefined name from the author's session) as the price series: `close`, the function's own argument, is used.
  4.1   Series.iteritems() -> items(); `.ﬁllna` ligature; the counter starts as a float Series (pandas 3 refuses to add 1. to an integer Series element by element).
  4.2   pd.Series(index=molecule) -> dtype float; the lines under the function are wrapped in `getTW(events, close, numThreads)`.
  4.10  as 4.2: wrapped in `getSampleW(events, numCoEvents, close, numThreads)`.
  4.11  print -> the (const, slope) pair is returned with the weights.
  5.3   `getWeights_FFD` is called by the printed code but its text is not in the extract: it is written here from the book's definition (the recursion of Snippet 5.1, stopped at the first |w| < thres);
        fillna(method='ffill') -> ffill(); pd.Series() -> dtype float.
  5.4   needs statsmodels' adfuller and a CSV of E-mini futures; statsmodels is not installed (and cannot be installed here).  `adfuller_c1` is the same regression written directly (Delta y_t on y_{t-1},
        Delta y_{t-1}, a constant; the t-statistic of the lagged level; maxlag=1, autolag=None) with the 5% critical value from MacKinnon's response surface (the coefficients are quoted from memory of MacKinnon [2010]
        and checked only against the asymptotic value -2.8621 and against the project's own ADF); the loop of plotMinFFD is `minFFD_table(close_df)` and takes the price frame as argument.
  19.1  pd.stats.moments.rolling_sum / rolling_mean / rolling_max / rolling_min (removed from pandas) -> Series.rolling(...).sum() / mean() / max() / min(); `3–2*2**.5` (en dash) -> `3-2*2**.5`.
  19.2  `3–2*2**.5` -> `3-2*2**.5`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from book_mp import mpPandasObj  # the book: from mpEngine import ... (Chapter 20); parked item 7, 2026-11-17
import book_port_a as PA


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 2
# ---------------------------------------------------------------------------------------------------------------------
def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 3
# ---------------------------------------------------------------------------------------------------------------------
def getDailyVol(close, span0=100):
    # daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


def getEvents_3_3(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    side_ = pd.Series(1., index=trgt.index)
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads, close=close, events=events, ptSl=[ptSl, ptSl])
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    events = events.drop('side', axis=1)
    return events


def addVerticalBarrier(close, tEvents, numDays):
    """Snippet 3.4: returns t1."""
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])  # NaNs at end
    return t1


def getBins_3_5(events, close):
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    out['bin'] = np.sign(out['ret'])
    return out


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads, close=close, events=events, ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    if side is None:
        events = events.drop('side', axis=1)
    return events


def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 4
# ---------------------------------------------------------------------------------------------------------------------
def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count.
    '''
    # 1) find events that span the period [molecule[0],molecule[-1]]
    t1 = t1.fillna(closeIdx[-1])  # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()]  # events that start at or before t1[molecule].max()
    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0., index=closeIdx[iloc[0]:iloc[1] + 1])
    for tIn, tOut in t1.items():
        count.loc[tIn:tOut] += 1.
    return count.loc[molecule[0]:t1[molecule].max()]


def mpSampleTW(t1, numCoEvents, molecule):
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn:tOut]).mean()
    return wght


def getTW(events, close, numThreads):
    """The lines printed under Snippet 4.2: returns (numCoEvents, tW)."""
    numCoEvents = mpPandasObj(mpNumCoEvents, ('molecule', events.index), numThreads, closeIdx=close.index, t1=events['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    tW = mpPandasObj(mpSampleTW, ('molecule', events.index), numThreads, t1=events['t1'], numCoEvents=numCoEvents)
    return numCoEvents, tW


def mpSampleW(t1, numCoEvents, close, molecule):
    # Derive sample weight by return attribution
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


def getSampleW(events, numCoEvents, close, numThreads):
    """The lines printed under Snippet 4.10: returns the weights scaled to sum to the number of events."""
    w = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1=events['t1'], numCoEvents=numCoEvents, close=close)
    w *= w.shape[0] / w.sum()
    return w


def getTimeDecay(tW, clfLastW=1.):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1. / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    print(const, slope)
    return clfW


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 5
# ---------------------------------------------------------------------------------------------------------------------
def getWeights_FFD(d, thres):
    """Not in the extract (see the module docstring): the recursion of Snippet 5.1, stopped at the first weight below thres in absolute value."""
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def fracDiff_FFD(series, d, thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    # 1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    # 2) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].ffill().dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def adf_crit5_mackinnon(nobs):
    """MacKinnon (2010) response surface, regression 'c', one variable, 5%: c0 + c1/N + c2/N^2 + c3/N^3 (coefficients quoted from memory; see the module docstring)."""
    c = (-2.86154, -2.8903, -4.234, -40.040)
    return c[0] + c[1] / nobs + c[2] / nobs ** 2 + c[3] / nobs ** 3


def adfuller_c1(y):
    """ADF with one lagged difference and a constant, as `adfuller(y, maxlag=1, regression='c', autolag=None)`: returns (statistic, 5% critical value, nobs used)."""
    y = np.asarray(y, float)
    dy = np.diff(y)
    Y = dy[1:]
    X = np.column_stack([y[1:-1], dy[:-1], np.ones(len(Y))])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ b
    s2 = e @ e / (len(Y) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    return b[0] / se, adf_crit5_mackinnon(len(Y)), len(Y)


def minFFD_table(df0, thres=.01, ds=None):
    """The body of Snippet 5.4's plotMinFFD for a price frame with a 'Close' column: for each d, the ADF statistic, the 5% value and the correlation with the original log series."""
    ds = np.linspace(0, 1, 11) if ds is None else ds
    out = pd.DataFrame(columns=['adfStat', 'nObs', '95% conf', 'corr'])
    for d in ds:
        df1 = np.log(df0[['Close']]).resample('1D').last()  # downcast to daily obs
        df2 = fracDiff_FFD(df1, d, thres=thres)
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
        stat, crit, nobs = adfuller_c1(df2['Close'].values)
        out.loc[d] = [stat, nobs, crit, corr]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 19
# ---------------------------------------------------------------------------------------------------------------------
def getBeta(series, sl):
    hl = series[['High', 'Low']].values
    hl = np.log(hl[:, 0] / hl[:, 1]) ** 2
    hl = pd.Series(hl, index=series.index)
    beta = hl.rolling(window=2).sum()
    beta = beta.rolling(window=sl).mean()
    return beta.dropna()


def getGamma(series):
    h2 = series['High'].rolling(window=2).max()
    l2 = series['Low'].rolling(window=2).min()
    gamma = np.log(h2.values / l2.values) ** 2
    gamma = pd.Series(gamma, index=h2.index)
    return gamma.dropna()


def getAlpha(beta, gamma):
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0  # set negative alphas to 0 (see p.727 of paper)
    return alpha.dropna()


def corwinSchultz(series, sl=1):
    # Note: S<0 iif alpha<0
    beta = getBeta(series, sl)
    gamma = getGamma(series)
    alpha = getAlpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    startTime = pd.Series(series.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, startTime], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread


def getSigma(beta, gamma):
    k2 = (8 / np.pi) ** .5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -.5 - 1) * beta ** .5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** .5
    sigma[sigma < 0] = 0
    return sigma
