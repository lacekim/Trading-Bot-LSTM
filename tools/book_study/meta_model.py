"""Chapters 4, 6, 7, 8 and 10 for the meta-labeling model:
  * sequential bootstrap (Snippets 4.3-4.5) applied per instrument, with max_samples = average uniqueness (Snippet 6.2, clf1),
  * a bagged forest of entropy decision trees with balanced class weights, return-attribution sample weights and a large
    min_weight_fraction_leaf (Ch.6.4),
  * purged k-fold with embargo (Snippets 7.1-7.4) for pooled multi-instrument events,
  * MDA and SFI feature importance (Snippets 8.3, 8.4) with sample-weighted negative log-loss,
  * bet size from a probability (Snippets 10.1, 10.3) in the meta-labeling case.
Needs numpy, pandas, scikit-learn.  No project imports.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

import book_stats as B

_N = NormalDist()


# ---------------------------------------------------------------------------------------------------------------------
# sequential bootstrap
# ---------------------------------------------------------------------------------------------------------------------
def seq_bootstrap_one_symbol(starts: np.ndarray, ends: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Exact sequential bootstrap over one instrument's events (Snippet 4.5): draw m indices; at each draw candidate i has probability
    proportional to its average uniqueness over bars starts[i]..ends[i] (inclusive) if it were added to the draws so far
    (concurrency counts every earlier draw, duplicates included).  starts must be sorted ascending."""
    n = len(starts)
    if n == 0 or m <= 0:
        return np.empty(0, dtype=np.int64)
    off = int(starts.min())
    a = (starts - off).astype(np.int64)
    b = (ends - off).astype(np.int64)
    c = np.zeros(int(b.max()) + 2, dtype=np.float64)          # draws covering each bar
    inv = np.ones_like(c)                                      # 1 / (c + 1)
    span_max = int((b - a).max()) + 1
    avg = np.ones(n)
    out = np.empty(m, dtype=np.int64)
    for d in range(m):
        cum = np.cumsum(avg)
        j = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
        if j >= n:
            j = n - 1
        out[d] = j
        c[a[j]:b[j] + 1] += 1.0
        inv[a[j]:b[j] + 1] = 1.0 / (c[a[j]:b[j] + 1] + 1.0)
        # candidates whose span touches [a_j, b_j]: start in [a_j - span_max, b_j]
        lo = int(np.searchsorted(a, a[j] - span_max, side="left"))
        hi = int(np.searchsorted(a, b[j], side="right"))
        if hi > lo:
            base = int(a[lo])
            top = int(b[lo:hi].max()) + 1
            pref = np.concatenate([[0.0], np.cumsum(inv[base:top])])
            s_i = a[lo:hi] - base
            e_i = b[lo:hi] - base + 1
            avg[lo:hi] = (pref[e_i] - pref[s_i]) / (e_i - s_i)
    return out


def seq_bootstrap(sym: np.ndarray, starts: np.ndarray, ends: np.ndarray, m_total: int, rng: np.random.Generator) -> np.ndarray:
    """Per-instrument sequential bootstraps whose draw counts add up to m_total in proportion to each instrument's number of events.
    Returns indices into the input arrays. (Events of different instruments never overlap in uniqueness, so this is the book's
    procedure applied to each instrument's own indicator matrix.)"""
    n = len(sym)
    out = []
    order = np.argsort(sym, kind="stable")
    ssym = sym[order]
    bounds = np.flatnonzero(np.diff(ssym)) + 1
    for grp in np.split(order, bounds):
        if not len(grp):
            continue
        g = grp[np.argsort(starts[grp], kind="stable")]
        m_s = max(1, int(round(m_total * len(g) / n)))
        local = seq_bootstrap_one_symbol(starts[g], ends[g], m_s, rng)
        out.append(g[local])
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def sample_avg_uniqueness(sym: np.ndarray, starts: np.ndarray, ends: np.ndarray, idx: np.ndarray) -> float:
    """Mean over drawn rows (duplicates included) of their average uniqueness within the sample (Snippet 4.4 on the drawn columns)."""
    vals = []
    for s in np.unique(sym[idx]):
        k = idx[sym[idx] == s]
        a, e = starts[k], ends[k]
        off = int(a.min())
        diff = np.zeros(int(e.max()) - off + 2)
        np.add.at(diff, a - off, 1.0)
        np.add.at(diff, e - off + 1, -1.0)
        conc = np.cumsum(diff)
        inv = np.where(conc > 0, 1.0 / np.where(conc > 0, conc, 1.0), 0.0)
        pref = np.concatenate([[0.0], np.cumsum(inv)])
        vals.append((pref[e - off + 1] - pref[a - off]) / (e - a + 1))
    return float(np.concatenate(vals).mean())


# ---------------------------------------------------------------------------------------------------------------------
# the forest
# ---------------------------------------------------------------------------------------------------------------------
class BaggedForest:
    """Bagging of entropy decision trees (Snippet 6.2, clf1). Each tree is fit on max_samples draws made by the sequential bootstrap
    ('sequential') or by ordinary sampling with replacement ('standard'), with the events' sample weights and balanced class weights.
    max_samples = average uniqueness of the training events, capped at max_draws."""

    def __init__(self, n_estimators=100, bootstrap="sequential", max_draws=20000, min_weight_fraction_leaf=0.05, max_features="sqrt", seed=0):
        self.n_estimators, self.bootstrap, self.max_draws = n_estimators, bootstrap, max_draws
        self.mwfl, self.max_features, self.seed = min_weight_fraction_leaf, max_features, seed
        self.trees, self.draws_info = [], {}

    def fit(self, X, y, w, sym, starts, ends, avg_u):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        m = int(min(max(1, round(float(avg_u) * n)), self.max_draws))
        self.draws_info = {"n_train": int(n), "avg_uniqueness": float(avg_u), "draws_per_tree": m, "bootstrap": self.bootstrap}
        self.trees = []
        samp_u = []
        for t in range(self.n_estimators):
            if self.bootstrap == "sequential":
                idx = seq_bootstrap(sym, starts, ends, m, rng)
            else:
                idx = rng.integers(0, n, size=m)
            if len(samp_u) < 5:
                samp_u.append(sample_avg_uniqueness(sym, starts, ends, idx))
            tree = DecisionTreeClassifier(criterion="entropy", max_features=self.max_features, class_weight="balanced",
                                          min_weight_fraction_leaf=self.mwfl, random_state=int(rng.integers(0, 2 ** 31 - 1)))
            tree.fit(X[idx], y[idx], sample_weight=w[idx])
            self.trees.append(tree)
        self.draws_info["sample_avg_uniqueness_first5"] = float(np.mean(samp_u))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba1(self, X):
        """Probability of class 1 averaged over trees (trees lacking a class contribute 0 or 1 accordingly)."""
        acc = np.zeros(len(X))
        for tree in self.trees:
            p = tree.predict_proba(X)
            cls = list(tree.classes_)
            acc += p[:, cls.index(1)] if 1 in cls else 0.0
        return acc / len(self.trees)


# ---------------------------------------------------------------------------------------------------------------------
# purged k-fold (pooled instruments)
# ---------------------------------------------------------------------------------------------------------------------
def purged_kfold(t0: np.ndarray, t1: np.ndarray, n_splits: int, embargo_pct: float):
    """Contiguous folds of equal event counts in time order (Snippet 7.3's array_split on sorted events); the training set drops events whose
    label interval touches the test span and events starting within the embargo after it (Snippets 7.1, 7.2). Yields (train_idx, test_idx)."""
    order = np.argsort(t0, kind="stable")
    parts = np.array_split(order, n_splits)
    t_min, t_max = float(t0.min()), float(t0.max()) + 1.0
    embargo = embargo_pct * (t_max - t_min)
    for te in parts:
        lo, hi = float(t0[te].min()), float(t1[te].max())
        keep = np.ones(len(t0), dtype=bool)
        keep[te] = False
        keep &= ~((t0 <= hi) & (t1 >= lo))
        keep &= ~((t0 > hi) & (t0 <= hi + embargo))
        yield np.flatnonzero(keep), np.sort(te)


def weighted_log_loss(y, p1, w):
    p = np.clip(p1, 1e-6, 1 - 1e-6)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def mda_importance(X, y, w, sym, starts, ends, avg_u_fn, t0, t1, feature_names, n_splits=5, embargo_pct=0.01, make_forest=None, seed=0):
    """Snippet 8.3 with neg-log-loss: the drop in out-of-sample score when one feature's test values are permuted, divided by -score1.
    Returns (DataFrame mean/std per feature, baseline score)."""
    rng = np.random.default_rng(seed)
    scr0, scr1 = [], []
    for tr, te in purged_kfold(t0, t1, n_splits, embargo_pct):
        f = make_forest().fit(X[tr], y[tr], w[tr], sym[tr], starts[tr], ends[tr], avg_u_fn(tr))
        s0 = -weighted_log_loss(y[te], f.predict_proba1(X[te]), w[te])
        row = []
        for j in range(X.shape[1]):
            Xp = X[te].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            row.append(-weighted_log_loss(y[te], f.predict_proba1(Xp), w[te]))
        scr0.append(s0)
        scr1.append(row)
    scr0, scr1 = np.array(scr0), np.array(scr1)
    imp = (-scr1 + scr0[:, None]) / -scr1                 # (score0 - score1) / -score1, per fold
    out = pd.DataFrame({"mean": imp.mean(axis=0), "std": imp.std(axis=0, ddof=1) * imp.shape[0] ** -0.5}, index=feature_names)
    return out, float(scr0.mean())


def sfi_importance(X, y, w, sym, starts, ends, avg_u_fn, t0, t1, feature_names, n_splits=5, embargo_pct=0.01, make_forest=None):
    """Snippet 8.4: out-of-sample neg-log-loss of a forest using one feature at a time."""
    rows = []
    folds = list(purged_kfold(t0, t1, n_splits, embargo_pct))
    for j, name in enumerate(feature_names):
        sc = []
        for tr, te in folds:
            f = make_forest().fit(X[tr][:, [j]], y[tr], w[tr], sym[tr], starts[tr], ends[tr], avg_u_fn(tr))
            sc.append(-weighted_log_loss(y[te], f.predict_proba1(X[te][:, [j]]), w[te]))
        sc = np.array(sc)
        rows.append({"feature": name, "mean": float(sc.mean()), "std": float(sc.std(ddof=1) * len(sc) ** -0.5)})
    return pd.DataFrame(rows).set_index("feature")


# ---------------------------------------------------------------------------------------------------------------------
# bet size
# ---------------------------------------------------------------------------------------------------------------------
def bet_size(p1, step: float = 0.0):
    """Snippets 10.1 (meta-labeling, two classes) and 10.3: size = 2*Phi((p-1/2)/sqrt(p(1-p))) - 1 when p > 1/2 (the model says 'take the bet'),
    else 0; optionally rounded to multiples of `step` and capped at 1."""
    p = np.clip(np.asarray(p1, dtype=np.float64), 1e-9, 1 - 1e-9)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    cdf = np.array([_N.cdf(v) for v in z]) if z.size < 5 else _vec_cdf(z)
    size = np.where(p > 0.5, 2.0 * cdf - 1.0, 0.0)
    if step > 0:
        size = np.round(size / step) * step
    return np.clip(size, 0.0, 1.0)


def _vec_cdf(z):
    from math import erf, sqrt
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
