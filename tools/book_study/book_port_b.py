"""Chapters 7-9 of the book, Snippets 7.1-7.4, 8.2-8.10, 9.1-9.4, ported to Python 3 / pandas 3 / numpy 2 / scikit-learn 1.x / scipy with the book's names.

Changes from the printed code (forced by language / library versions or typos; nothing else is altered):
  7.1  Series.iteritems() -> items().
  7.2  Series.append (removed from pandas) -> pd.concat; the three trailing lines of the snippet (usage) are `embargo_usage(mbrg, t1, dt0, dt1)`.
  7.3  PurgedKFold as printed, except `self.t1[test_indices]` (integer positions on a datetime-indexed Series, which old pandas treated as positions and pandas 3 treats as labels) is `self.t1.iloc[test_indices]` (sklearn's _BaseKFold is private but still present; the constructor keyword `shuffle=False` is as printed).
  7.4  as printed (the import `from clfSequential import PurgedKFold` is replaced by this module's class).
  8.2  as printed.
  8.3  pd.Series() / pd.DataFrame(columns=...) get dtype float; np.random.shuffle(X1_[j].values) shuffles a read-only view under pandas' copy-on-write, so the column is replaced by a permutation of its
       values (the same permutation of a single column).
  8.4  as printed.  8.5  as printed.  8.6  runs the printed example and returns the statistic (scipy.stats.weightedtau).
  8.7  pd.DatetimeIndex(periods=, freq=, end=) and pd.datetime.today() no longer exist: pd.bdate_range(end=today, periods=n_samples); xrange -> range; the printed `pd.DataFrame(trnsX,index=df0),` + newline
       is a syntax error as printed (missing line continuation): the same tuple is built on one line.
  8.8  base_estimator= -> estimator= (scikit-learn 1.2); max_samples is passed as an integer number of draws (scikit-learn >= 1.7 measures a float max_samples against the sum of the sample weights, and the book's weights sum to 1); mpPandasObj is book_mp.mpPandasObj; `PurgedKFold` is this module's.
  8.9  izip -> zip; print -> print(); the final `out['method',...]` is written with a list of columns (the printed tuple key is invalid).
  8.10 matplotlib's Agg backend (no window); otherwise as printed.
  9.1 / 9.3  the bagging max_samples is an integer number of draws for the same reason; GridSearchCV / RandomizedSearchCV have no `iid` argument any more (removed in scikit-learn 0.24; False was the only sensible value): dropped; base_estimator= -> estimator=.
  9.2  as printed.  9.4  as printed (the test at the bottom of the snippet is `test_logUniform`, returning the KS result and the describe() table instead of plotting).
"""
from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as mpl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kstest, rv_continuous, weightedtau  # noqa: E402
from sklearn.ensemble import BaggingClassifier  # noqa: E402
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # noqa: E402
from sklearn.model_selection._split import _BaseKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

import book_mp as MP  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 7
# ---------------------------------------------------------------------------------------------------------------------
def getTrainTimes(t1, testTimes):
    '''
    Given testTimes, find the times of the training observations.
    -t1.index: Time when the observation started.
    -t1.value: Time when the observation ended.
    -testTimes: Times of testing observations.
    '''
    trn = t1.copy(deep=True)
    for i, j in testTimes.items():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index  # train starts within test
        df1 = trn[(i <= trn) & (trn <= j)].index  # train ends within test
        df2 = trn[(trn.index <= i) & (j <= trn)].index  # train envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def getEmbargoTimes(times, pctEmbargo):
    # Get embargo time for each bar
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = pd.concat([mbrg, pd.Series(times[-1], index=times[-step:])])
    return mbrg


def embargo_usage(mbrg, t1, dt0, dt1):
    """The three lines printed under Snippet 7.2: returns (trainTimes, testTimes)."""
    testTimes = pd.Series(mbrg[dt1], index=[dt0])  # include embargo before purge
    trainTimes = getTrainTimes(t1, testTimes)
    testTimes = t1.loc[dt0:dt1].index
    return trainTimes, testTimes


class PurgedKFold(_BaseKFold):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    '''

    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            yield train_indices, test_indices


def cvScore(clf, X, y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=None):
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 8
# ---------------------------------------------------------------------------------------------------------------------
def featImpMDI(fit, featNames):
    # feat importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0] ** -.5}, axis=1)
    imp /= imp['mean'].sum()
    return imp


def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    scr0, scr1 = pd.Series(dtype=float), pd.DataFrame(columns=X.columns, dtype=float)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            X1_[j] = np.random.permutation(X1_[j].values)  # permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)
    imp = (-scr1).add(scr0, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -scr1
    else:
        imp = imp / (1. - scr1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -.5}, axis=1)
    return imp, scr0.mean()


def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    imp = pd.DataFrame(columns=['mean', 'std'], dtype=float)
    for featName in featNames:
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cvGen=cvGen)
        imp.loc[featName, 'mean'] = df0.mean()
        imp.loc[featName, 'std'] = df0.std() * df0.shape[0] ** -.5
    return imp


def get_eVec(dot, varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[idx], eVec[:, idx]
    # 2) only positive eVals
    eVal = pd.Series(eVal, index=['PC_' + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:, eVal.index]
    # 3) reduce dimension, form PCs
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[:dim + 1], eVec.iloc[:, :dim + 1]
    return eVal, eVec


def orthoFeats(dfX, varThres=.95):
    # Given a dataframe dfX of features, compute orthofeatures dfP
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)  # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    return dfP


def weighted_tau_example():
    """Snippet 8.6, as printed: the weighted Kendall's tau between a feature-importance vector and the inverse PCA rank."""
    featImp = np.array([.55, .33, .07, .05])  # feature importance
    pcRank = np.array([1, 2, 4, 3])  # PCA rank
    return weightedtau(featImp, pcRank ** -1.)[0]


def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    # generate a random dataset for a classification problem
    from sklearn.datasets import make_classification
    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, random_state=0, shuffle=False)
    df0 = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_samples)
    trnsX, cont = pd.DataFrame(trnsX, index=df0), pd.Series(cont, index=df0).to_frame('bin')
    df0 = ['I_' + str(i) for i in range(n_informative)] + ['R_' + str(i) for i in range(n_redundant)]
    df0 += ['N_' + str(i) for i in range(n_features - len(df0))]
    trnsX.columns = df0
    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont


def featImportance(trnsX, cont, n_estimators=1000, cv=10, max_samples=1., numThreads=24, pctEmbargo=0, scoring='accuracy', method='SFI', minWLeaf=0., **kargs):
    # feature importance from a random forest
    n_jobs = (-1 if numThreads > 1 else 1)  # run 1 thread with ht_helper in dirac1
    # 1) prepare classifier,cv. max_features=1, to prevent masking
    clf = DecisionTreeClassifier(criterion='entropy', max_features=1, class_weight='balanced', min_weight_fraction_leaf=minWLeaf)
    # scikit-learn >= 1.7 reads a float max_samples as a fraction of the SUM OF THE SAMPLE WEIGHTS when weights are passed; the book's weights sum to 1, which would draw one observation per tree.
    # The number of draws is therefore passed as an integer: max_samples (the book's fraction of the observations) x the number of observations.
    ms = max_samples if not isinstance(max_samples, float) else max(1, int(round(max_samples * trnsX.shape[0])))
    clf = BaggingClassifier(estimator=clf, n_estimators=n_estimators, max_features=1., max_samples=ms, oob_score=True, n_jobs=n_jobs)
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    oob = fit.oob_score_
    if method == 'MDI':
        imp = featImpMDI(fit, featNames=trnsX.columns)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring).mean()
    elif method == 'MDA':
        imp, oos = featImpMDA(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring)
    elif method == 'SFI':
        cvGen = PurgedKFold(n_splits=cv, t1=cont['t1'], pctEmbargo=pctEmbargo)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cvGen=cvGen).mean()
        clf.n_jobs = 1  # paralellize auxFeatImpSFI rather than clf
        imp = MP.mpPandasObj(auxFeatImpSFI, ('featNames', trnsX.columns), numThreads, clf=clf, trnsX=trnsX, cont=cont, scoring=scoring, cvGen=cvGen)
    return imp, oob, oos


def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    mpl.figure(figsize=(10, imp.shape[0] / 5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    if method == 'MDI':
        mpl.xlim([0, imp.sum(axis=1).max()])
        mpl.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index):
        ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha='center', va='center', color='black')
    mpl.title('tag=' + str(tag) + ' | simNum=' + str(simNum) + ' | oob=' + str(round(oob, 4)) + ' | oos=' + str(round(oos, 4)))
    mpl.savefig(pathOut + 'featImportance_' + str(simNum) + '.png', dpi=100)
    mpl.clf()
    mpl.close()
    return


def testFunc(n_features=40, n_informative=10, n_redundant=10, n_estimators=1000, n_samples=10000, cv=10, pathOut='./testFunc/', numThreads=24, plot=True):
    # test the performance of the feat importance functions on artificial data
    # Nr noise features = n_features-n_informative-n_redundant
    import os
    os.makedirs(pathOut, exist_ok=True)
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    dict0 = {'minWLeaf': [0.], 'scoring': ['accuracy'], 'method': ['MDI', 'MDA', 'SFI'], 'max_samples': [1.]}
    jobs, out = (dict(zip(dict0, i)) for i in itertools.product(*dict0.values())), []
    kargs = {'pathOut': pathOut, 'n_estimators': n_estimators, 'tag': 'testFunc', 'cv': cv, 'numThreads': numThreads}
    for job in jobs:
        job['simNum'] = job['method'] + '_' + job['scoring'] + '_' + '%.2f' % job['minWLeaf'] + '_' + str(job['max_samples'])
        kargs.update(job)
        imp, oob, oos = featImportance(trnsX=trnsX, cont=cont, **kargs)
        if plot:
            plotFeatImportance(imp=imp, oob=oob, oos=oos, **kargs)
        df0 = imp[['mean']] / imp['mean'].abs().sum()
        df0['type'] = [i[0] for i in df0.index]
        df0 = df0.groupby('type')['mean'].sum().to_dict()
        df0.update({'oob': oob, 'oos': oos})
        df0.update(job)
        out.append(df0)
    out = pd.DataFrame(out).sort_values(['method', 'scoring', 'minWLeaf', 'max_samples'])
    out = out[['method', 'scoring', 'minWLeaf', 'max_samples', 'I', 'R', 'N', 'oob', 'oos']]
    out.to_csv(kargs['pathOut'] + 'stats.csv')
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 9
# ---------------------------------------------------------------------------------------------------------------------
class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0, None, 1.], rndSearchIter=0, n_jobs=-1, pctEmbargo=0, **fit_params):
    """Snippet 9.3 (with rndSearchIter=0 it is Snippet 9.1)."""
    if set(lbl.values) == {0, 1}:
        scoring = 'f1'  # f1 for meta-labeling
    else:
        scoring = 'neg_log_loss'  # symmetric towards all cases
    # 1) hyperparameter search, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    if rndSearchIter == 0:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs, n_iter=rndSearchIter)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_  # pipeline
    # 2) fit validated model on the entirety of the data
    if bagging[1] is not None and bagging[1] > 0:
        # integer number of draws, for the reason given in featImportance (scikit-learn >= 1.7 measures a float max_samples against the sum of the sample weights)
        gs = BaggingClassifier(estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]), max_samples=max(1, int(round(float(bagging[1]) * feat.shape[0]))), max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(feat, lbl, sample_weight=fit_params[gs.estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs


class logUniform_gen(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def logUniform(a=1, b=np.exp(1)):
    return logUniform_gen(a=a, b=b, name='logUniform')


def test_logUniform(a=1E-3, b=1E3, size=10000, seed=None):
    """The test at the bottom of Snippet 9.4: KS test of log(vals) against the uniform on [log a, log b], and the describe() table."""
    vals = logUniform(a=a, b=b).rvs(size=size, random_state=seed)
    ks = kstest(np.log(vals), 'uniform', args=(np.log(a), np.log(b / a)))
    return ks, pd.Series(vals).describe()
