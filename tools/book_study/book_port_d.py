"""Chapter 16 (Snippets 16.1-16.5, including the Appendix 16.A.3 full implementation) and Chapter 21 (Snippets 21.1-21.7), ported to Python 3 / pandas 3 / numpy 2 with the book's names.

Changes from the printed code (each forced by the language / library versions or by a typo; nothing else is altered):
  16.1  the three printed lines are wrapped in `treeCluster(x)`.
  16.2  Series.append (removed from pandas) -> pd.concat.
  16.3  `len(i)/2` (a float in Python 3, unusable as a slice bound) -> `len(i)//2`; xrange -> range; `w=pd.Series(1,...)` is an integer Series and pandas 3 refuses `w[items]*=alpha` with
        a float, so the Series starts as 1. (float, the same values).
  16.4  an optional `cols` argument is added to generateData: with Python 3's random.randint the seed-12345 draw of `cols` is [3, 0, 2, 2, 1] (0-based), not the one behind the book's
        figures and Table 16.1 ((6,3),(7,1),(8,5),(9,2),(10,2) in the book's labels = cols [2, 0, 4, 1, 1], read from the Figure 16.5 caption); passing cols=[2,0,4,1,1] reproduces Table 16.1.
        As printed with print -> return; `size0–1` (en dash) -> `size0-1`; `mpl.saveﬁg` -> savefig; xrange -> range; `random.randint` is Python 3's (its draws for a given seed can differ from Python 2's).
  16.5  the printed text imports a module `CLA` (Bailey and Lopez de Prado [2013]) whose code is NOT in the book extract.  `getCLA` here takes `cla.w[-1]`, the LAST turning point of the critical line
        algorithm, which with bounds 0 <= w <= 1, full investment and an arbitrary mean vector is the long-only minimum-variance portfolio; `minVarLongOnly` solves exactly that problem by an
        active-set method (its own code, not the book's), and is tested against a general solver.  Series.append -> pd.concat; xrange -> range; print -> a returned table; the experiment returns
        the stats frame instead of writing 'stats.csv' (the runner writes it); `size0–1` -> `size0-1`; `len(...)` as printed.
  21.1  xrange -> range.   21.2  `ﬂoat` -> float; as printed, a zero entry of a partition is multiplied by both signs, so identical columns repeat (e.g. n=2, k=2: 12 columns, 8 distinct); kept.   21.3  as printed.
  21.4  as printed (an unseeded RandomState); an optional `seed` argument is added so that a run is repeatable.
  21.5  the printed fragment `main` code is wrapped in `genParams`.   21.6, 21.7  as printed; the prints are returned.
"""
from __future__ import annotations

import random
from itertools import combinations_with_replacement, product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as mpl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.cluster.hierarchy as sch  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 16
# ---------------------------------------------------------------------------------------------------------------------
def treeCluster(x):
    """Snippet 16.1."""
    cov, corr = x.cov(), x.corr()
    dist = ((1 - corr) / 2.) ** .5  # distance matrix
    link = sch.linkage(dist, 'single')  # linkage matrix
    return cov, corr, dist, link


def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def getClusterVar(cov, cItems):
    # Compute variance per cluster
    cov_ = cov.loc[cItems, cItems]  # matrix slice
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1., index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.) ** .5  # distance matrix
    return dist


def plotCorrMatrix(path, corr, labels=None):
    # Heatmap of the correlation matrix
    if labels is None:
        labels = []
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.xticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.savefig(path)
    mpl.clf()
    mpl.close()  # reset pylab
    return


def generateData(nObs, size0, size1, sigma1, cols=None):
    # Time series of correlated variables
    # 1) generating some uncorrelated data
    np.random.seed(seed=12345)
    random.seed(12345)
    x = np.random.normal(0, 1, size=(nObs, size0))  # each row is a variable
    # 2) creating correlation between the variables
    if cols is None:
        cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))
    return x, cols


def main_16_4(nObs=10000, size0=5, size1=5, sigma1=.25, plot_dir=None, cols=None):
    """The printed main(): returns (pairs, hrp weights) and writes the two heat-maps when plot_dir is given."""
    x, cols = generateData(nObs, size0, size1, sigma1, cols=cols)
    pairs = [(j + 1, size0 + i) for i, j in enumerate(cols, 1)]
    cov, corr = x.cov(), x.corr()
    if plot_dir:
        plotCorrMatrix(plot_dir + 'HRP3_corr0.png', corr, labels=corr.columns)
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    df0 = corr.loc[sortIx, sortIx]  # reorder
    if plot_dir:
        plotCorrMatrix(plot_dir + 'HRP3_corr1.png', df0, labels=df0.columns)
    hrp = getRecBipart(cov, sortIx)
    return pairs, hrp


# --- Snippet 16.5 ----------------------------------------------------------------------------------------------------
def generateData_MC(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
    # Time series of correlated variables
    # 1) generate random uncorrelated data
    x = np.random.normal(mu0, sigma0, size=(nObs, size0))
    # 2) create correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    # 3) add common random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]])
    # 4) add specific random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[point, cols[-1]] = np.array([-.5, 2])
    return x, cols


def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()


def minVarLongOnly(cov, tol=1e-12):
    """Minimum-variance portfolio with 0 <= w, sum w = 1 (the upper bound 1 cannot bind), by an active-set method: solve on the free set, drop the most negative weight, and release a
    fixed asset whose gradient (V w)_i is below the multiplier.  This is the problem whose solution is the last turning point of the critical line algorithm for these bounds."""
    V = np.asarray(cov, float)
    n = V.shape[0]
    free = np.ones(n, bool)
    for _ in range(10 * n + 10):
        idx = np.where(free)[0]
        Vf = V[np.ix_(idx, idx)]
        x = np.linalg.solve(Vf, np.ones(len(idx)))
        lam = 1. / x.sum()
        wf = x * lam
        if (wf < -tol).any():
            free[idx[np.argmin(wf)]] = False
            continue
        w = np.zeros(n)
        w[idx] = wf
        g = V @ w
        out = np.where(~free)[0]
        if len(out) and (g[out] < lam - 1e-10).any():
            free[out[np.argmin(g[out] - lam)]] = True
            continue
        return np.clip(w, 0, None) / np.clip(w, 0, None).sum()
    raise RuntimeError('minVarLongOnly did not converge')


def getCLA(cov, **kargs):
    # Compute CLA's minimum variance portfolio (see the module docstring: the CLA class is not in the extract)
    return minVarLongOnly(cov)


def hrpMC(numIters=1e4, nObs=520, size0=5, size1=5, mu0=0, sigma0=1e-2, sigma1F=.25, sLength=260, rebal=22, methods=None):
    # Monte Carlo experiment on HRP
    methods = [getIVP, getHRP, getCLA] if methods is None else methods
    stats, numIter = {i.__name__: pd.Series() for i in methods}, 0
    pointers = range(sLength, nObs, rebal)
    while numIter < numIters:
        # 1) Prepare data for one experiment
        x, cols = generateData_MC(nObs, sLength, size0, size1, mu0, sigma0, sigma1F)
        r = {i.__name__: pd.Series() for i in methods}
        # 2) Compute portfolios in-sample
        for pointer in pointers:
            x_ = x[pointer - sLength:pointer]
            cov_, corr_ = np.cov(x_, rowvar=0), np.corrcoef(x_, rowvar=0)
            # 3) Compute performance out-of-sample
            x_ = x[pointer:pointer + rebal]
            for func in methods:
                w_ = func(cov=cov_, corr=corr_)  # callback
                r_ = pd.Series(np.dot(x_, w_))
                r[func.__name__] = pd.concat([r[func.__name__], r_])
        # 4) Evaluate and store results
        for func in methods:
            r_ = r[func.__name__].reset_index(drop=True)
            p_ = (1 + r_).cumprod()
            stats[func.__name__].loc[numIter] = p_.iloc[-1] - 1
        numIter += 1
    # 5) Report results
    stats = pd.DataFrame.from_dict(stats, orient='columns')
    df0, df1 = stats.std(), stats.var()
    report = pd.concat([df0, df1, df1 / df1['getHRP'] - 1], axis=1)
    report.columns = ['std', 'var', 'var_vs_HRP']
    return stats, report


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 21
# ---------------------------------------------------------------------------------------------------------------------
def pigeonHole(k, n):
    # Pigeonhole problem (organize k objects in n slots)
    for j in combinations_with_replacement(range(n), k):
        r = [0] * n
        for i in j:
            r[i] += 1
        yield r


def getAllWeights(k, n):
    # 1) Generate partitions
    parts, w = pigeonHole(k, n), None
    # 2) Go through partitions
    for part_ in parts:
        w_ = np.array(part_) / float(k)  # abs(weight) vector
        for prod_ in product([-1, 1], repeat=n):  # add sign
            w_signed_ = (w_ * prod_).reshape(-1, 1)
            if w is None:
                w = w_signed_.copy()
            else:
                w = np.append(w, w_signed_, axis=1)
    return w


def evalTCosts(w, params):
    # Compute t-costs of a particular trajectory
    tcost = np.zeros(w.shape[1])
    w_ = np.zeros(shape=w.shape[0])
    for i in range(tcost.shape[0]):
        c_ = params[i]['c']
        tcost[i] = (c_ * abs(w[:, i] - w_) ** .5).sum()
        w_ = w[:, i].copy()
    return tcost


def evalSR(params, w, tcost):
    # Evaluate SR over multiple horizons
    mean, cov = 0, 0
    for h in range(w.shape[1]):
        params_ = params[h]
        mean += np.dot(w[:, h].T, params_['mean'])[0] - tcost[h]
        cov += np.dot(w[:, h].T, np.dot(params_['cov'], w[:, h]))
    sr = mean / cov ** .5
    return sr


def dynOptPort(params, k=None):
    # Dynamic optimal portfolio
    # 1) Generate partitions
    if k is None:
        k = params[0]['mean'].shape[0]
    n = params[0]['mean'].shape[0]
    w_all, sr = getAllWeights(k, n), None
    # 2) Generate trajectories as cartesian products
    for prod_ in product(w_all.T, repeat=len(params)):
        w_ = np.array(prod_).T  # concatenate product into a trajectory
        tcost_ = evalTCosts(w_, params)
        sr_ = evalSR(params, w_, tcost_)  # evaluate trajectory
        if sr is None or sr < sr_:  # store trajectory if better
            sr, w = sr_, w_.copy()
    return w


def rndMatWithRank(nSamples, nCols, rank, sigma=0, homNoise=True, seed=None):
    # Produce random matrix X with given rank
    rng = np.random.RandomState(seed)
    U, _, _ = np.linalg.svd(rng.randn(nCols, nCols))
    x = np.dot(rng.randn(nSamples, rank), U[:, :rank].T)
    if homNoise:
        x += sigma * rng.randn(nSamples, nCols)  # Adding homoscedastic noise
    else:
        sigmas = sigma * (rng.rand(nCols) + .5)  # Adding heteroscedastic noise
        x += rng.randn(nSamples, nCols) * sigmas
    return x


def genMean(size):
    # Generate a random vector of means
    rMean = np.random.normal(size=(size, 1))
    return rMean


def genParams(size=3, horizon=2, nSamples=1000, seed=None):
    """The printed `1) Parameters` block."""
    params = []
    for h in range(horizon):
        x = rndMatWithRank(nSamples, size, size, 0., seed=None if seed is None else seed + h)
        mean_, cov_ = genMean(size), np.cov(x, rowvar=False)
        c_ = np.random.uniform(size=cov_.shape[0]) * np.diag(cov_) ** .5
        params.append({'mean': mean_, 'cov': cov_, 'c': c_})
    return params


def statOptPortf(cov, a):
    # Static optimal porftolio
    # Solution to the "unconstrained" portfolio optimization problem
    cov_inv = np.linalg.inv(cov)
    w = np.dot(cov_inv, a)
    w /= np.dot(np.dot(a.T, cov_inv), a)  # np.dot(w.T,a)==1
    w /= abs(w).sum()  # re-scale for full investment
    return w


def staticSolution(params):
    """Snippet 21.6, block `2) Static optimal portfolios`: returns (w_stat, static SR)."""
    w_stat = None
    for params_ in params:
        w_ = statOptPortf(cov=params_['cov'], a=params_['mean'])
        if w_stat is None:
            w_stat = w_.copy()
        else:
            w_stat = np.append(w_stat, w_, axis=1)
    tcost_stat = evalTCosts(w_stat, params)
    sr_stat = evalSR(params, w_stat, tcost_stat)
    return w_stat, sr_stat


def dynamicSolution(params):
    """Snippet 21.7: returns (w_dyn, dynamic SR)."""
    w_dyn = dynOptPort(params)
    tcost_dyn = evalTCosts(w_dyn, params)
    sr_dyn = evalSR(params, w_dyn, tcost_dyn)
    return w_dyn, sr_dyn
