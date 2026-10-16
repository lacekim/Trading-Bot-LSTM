"""Minimal leaf-wise gradient-boosted decision tree (binary logistic loss),
pure numpy, no external ML deps -- a from-scratch GBDT since the real
lightgbm package can't be installed in this sandbox (no network, no
compatible wheel). Same core algorithm family: gradient/hessian-based
histogram splits, leaf-wise (best-first) tree growth, shrinkage.
"""
from __future__ import annotations
import numpy as np


class _Node:
    __slots__ = ("is_leaf", "feature", "threshold", "left", "right", "value", "depth")
    def __init__(self):
        self.is_leaf = True
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = 0.0
        self.depth = 0


def _best_split(X_col, grad, hess, lam, min_samples_leaf):
    """Find best threshold on one feature column (pre-sorted approach)."""
    order = np.argsort(X_col)  # default quicksort; stability not needed (tie
    # positions are never valid split candidates -- verified equivalent to
    # mergesort across 200 randomized high-tie-count trials before this change
    xs = X_col[order]
    g = grad[order]
    h = hess[order]
    n = len(xs)
    if n < 2 * min_samples_leaf:
        return None
    g_cum = np.cumsum(g)
    h_cum = np.cumsum(h)
    G_total, H_total = g_cum[-1], h_cum[-1]
    # candidate split points: where x value changes
    diff = np.diff(xs) != 0
    idxs = np.nonzero(diff)[0]  # split after this index
    if len(idxs) == 0:
        return None
    valid = (idxs + 1 >= min_samples_leaf) & (n - (idxs + 1) >= min_samples_leaf)
    idxs = idxs[valid]
    if len(idxs) == 0:
        return None
    GL = g_cum[idxs]
    HL = h_cum[idxs]
    GR = G_total - GL
    HR = H_total - HL
    gain = (GL * GL) / (HL + lam) + (GR * GR) / (HR + lam) - (G_total * G_total) / (H_total + lam)
    best_i = np.argmax(gain)
    best_gain = gain[best_i]
    split_idx = idxs[best_i]
    threshold = (xs[split_idx] + xs[split_idx + 1]) / 2.0
    return best_gain, threshold


def _build_tree(X, grad, hess, max_leaves, lam, min_samples_leaf, min_gain, feature_subsample_rng, feature_frac):
    n_samples, n_features = X.shape
    root = _Node()
    root_idx = np.arange(n_samples)
    G = grad.sum(); H = hess.sum()
    root.value = -G / (H + lam)

    # leaf-wise growth: priority queue of (-, gain, node, row_idx)
    leaves = [(root, root_idx)]
    n_leaves = 1
    while n_leaves < max_leaves:
        best = None  # (gain, leaf_pos, feature, threshold, left_idx, right_idx)
        for pos, (node, idx) in enumerate(leaves):
            if len(idx) < 2 * min_samples_leaf:
                continue
            g_node = grad[idx]
            h_node = hess[idx]
            n_feat_use = max(1, int(round(n_features * feature_frac)))
            feat_candidates = feature_subsample_rng.choice(n_features, size=n_feat_use, replace=False)
            for f in feat_candidates:
                res = _best_split(X[idx, f], g_node, h_node, lam, min_samples_leaf)
                if res is None:
                    continue
                gain, threshold = res
                if gain <= min_gain:
                    continue
                if best is None or gain > best[0]:
                    mask = X[idx, f] <= threshold
                    best = (gain, pos, f, threshold, idx[mask], idx[~mask])
        if best is None:
            break
        gain, pos, f, threshold, left_idx, right_idx = best
        node, _ = leaves.pop(pos)
        node.is_leaf = False
        node.feature = f
        node.threshold = threshold
        left = _Node(); right = _Node()
        left.depth = node.depth + 1; right.depth = node.depth + 1
        left.value = -grad[left_idx].sum() / (hess[left_idx].sum() + lam)
        right.value = -grad[right_idx].sum() / (hess[right_idx].sum() + lam)
        node.left, node.right = left, right
        leaves.append((left, left_idx))
        leaves.append((right, right_idx))
        n_leaves += 1
    return root


def _predict_tree(root, X):
    """Vectorized traversal: push whole row-index groups down the tree at
    once (numpy boolean masking per node) instead of looping row-by-row in
    Python. Mathematically identical to the row-at-a-time version -- same
    leaf a row ends up in, just batched -- needed because real datasets
    here run ~10-100x larger than the row-loop version can handle in
    reasonable time."""
    n = X.shape[0]
    out = np.empty(n)
    stack = [(root, np.arange(n))]
    while stack:
        node, idx = stack.pop()
        if len(idx) == 0:
            continue
        if node.is_leaf:
            out[idx] = node.value
        else:
            vals = X[idx, node.feature]
            mask = vals <= node.threshold
            stack.append((node.left, idx[mask]))
            stack.append((node.right, idx[~mask]))
    return out


class GBDTClassifier:
    """Binary classifier, logistic loss, leaf-wise GBDT (LightGBM-style)."""
    def __init__(self, n_estimators=100, learning_rate=0.05, max_leaves=15,
                 lam=1.0, min_samples_leaf=20, min_gain=1e-7, feature_frac=0.8,
                 random_state=0, early_stopping_rounds=15):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.lam = lam
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.feature_frac = feature_frac
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.trees_ = []
        self.base_score_ = 0.0
        self.best_n_trees_ = 0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base_score_ = np.log(p / (1 - p))
        raw = np.full(n, self.base_score_)
        rng = np.random.default_rng(self.random_state)

        raw_val = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)
            raw_val = np.full(len(y_val), self.base_score_)

        best_val_loss = np.inf
        best_n = 0
        no_improve = 0
        self.trees_ = []
        for t in range(self.n_estimators):
            pred = self._sigmoid(raw)
            grad = pred - y          # d(logloss)/d(raw)
            hess = pred * (1 - pred)
            hess = np.maximum(hess, 1e-6)
            tree = _build_tree(X, grad, hess, self.max_leaves, self.lam,
                                self.min_samples_leaf, self.min_gain, rng, self.feature_frac)
            update = _predict_tree(tree, X)
            raw += self.learning_rate * update
            self.trees_.append(tree)

            if X_val is not None:
                raw_val += self.learning_rate * _predict_tree(tree, X_val)
                pv = np.clip(self._sigmoid(raw_val), 1e-9, 1 - 1e-9)
                val_loss = -np.mean(y_val * np.log(pv) + (1 - y_val) * np.log(1 - pv))
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_n = t + 1
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.early_stopping_rounds:
                        break
        self.best_n_trees_ = best_n if X_val is not None else len(self.trees_)
        if self.best_n_trees_ == 0:
            self.best_n_trees_ = len(self.trees_)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        raw = np.full(X.shape[0], self.base_score_)
        for tree in self.trees_[: self.best_n_trees_]:
            raw += self.learning_rate * _predict_tree(tree, X)
        return self._sigmoid(raw)
