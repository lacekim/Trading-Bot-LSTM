"""Allocation schemes for Ch.16 (written from the idea of Hierarchical Risk Parity, not from the snippet text).

Inputs are a covariance matrix (numpy, symmetric, positive diagonal) whose rows/columns are in a fixed order (the caller sorts the symbols alphabetically).

  * equal_weight, inverse_variance (weights proportional to 1 / variance);
  * hrp_weights: correlation -> distance d = sqrt((1 - rho) / 2); a second distance D[i, j] = Euclidean distance between the columns i and j of d; single-linkage clustering on D;
    leaves put in the order of the dendrogram; recursive bisection of that order: a list of items is cut into a first part of floor(n / 2) items and a second part of the rest;
    each part's variance is that of the part held with inverse-variance weights; the first part receives 1 - V1 / (V1 + V2) of the capital the list holds, the second the rest;
  * min_variance_long_only: minimise w' S w subject to w >= 0, sum w = 1 (SLSQP), with a first-order optimality check that stops on failure.

Single linkage here is implemented explicitly so that the left / right order of the two children of every merge is fixed by a stated rule instead of by a library:
clusters are numbered 0..n-1 for the leaves and n, n+1, ... for the merges in the order made; at each step the two clusters with the smallest linkage distance are merged (ties:
the pair with the smaller lower id, then the smaller higher id); the merged cluster lists the child with the LOWER id first.  The leaf order is the left-to-right order of the leaves.
"""
from __future__ import annotations

import numpy as np


def corr_from_cov(cov: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.diag(cov))
    c = cov / np.outer(s, s)
    np.fill_diagonal(c, 1.0)
    return np.clip(c, -1.0, 1.0)


def distance_of_distances(corr: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    n = len(d)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(((d[:, i] - d[:, j]) ** 2).sum())
    return D


def single_linkage(D: np.ndarray):
    """Returns a list of merges (left_id, right_id, height, size) with left_id < right_id; new cluster ids are n, n+1, ..."""
    n = len(D)
    active = {i: {j: float(D[i, j]) for j in range(n) if j != i} for i in range(n)}
    size = {i: 1 for i in range(n)}
    merges = []
    nxt = n
    while len(active) > 1:
        best = None
        for a in sorted(active):
            for b in sorted(active[a]):
                if b > a:
                    key = (active[a][b], a, b)
                    if best is None or key < best:
                        best = key
        h, a, b = best
        new = {}
        for c in active:
            if c in (a, b):
                continue
            new[c] = min(active[a][c], active[b][c])
        for c in list(active):
            if c in (a, b):
                continue
            del active[c][a]
            del active[c][b]
            active[c][nxt] = new[c]
        del active[a]
        del active[b]
        active[nxt] = dict(new)
        size[nxt] = size[a] + size[b]
        merges.append((a, b, h, size[nxt]))
        nxt += 1
    return merges


def leaf_order(merges, n: int):
    """Left-to-right leaf order of the dendrogram (children of a merge: lower id first, as stored)."""
    if n == 1:
        return [0]
    children = {n + i: (m[0], m[1]) for i, m in enumerate(merges)}
    out, stack = [], [n + len(merges) - 1]
    while stack:
        c = stack.pop()
        if c < n:
            out.append(c)
        else:
            a, b = children[c]
            stack.append(b)
            stack.append(a)
    return out


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def inverse_variance(cov: np.ndarray) -> np.ndarray:
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def cluster_variance(cov: np.ndarray, items) -> float:
    items = list(items)
    sub = cov[np.ix_(items, items)]
    w = 1.0 / np.diag(sub)
    w = w / w.sum()
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray):
    """Returns (weights, order, merges)."""
    n = len(cov)
    if n == 1:
        return np.array([1.0]), [0], []
    merges = single_linkage(distance_of_distances(corr_from_cov(cov)))
    order = leaf_order(merges, n)
    w = np.ones(n)
    lists = [order]
    while lists:
        cur = lists.pop()
        if len(cur) <= 1:
            continue
        h = len(cur) // 2
        first, second = cur[:h], cur[h:]
        v1, v2 = cluster_variance(cov, first), cluster_variance(cov, second)
        alpha = 1.0 - v1 / (v1 + v2)
        w[first] *= alpha
        w[second] *= 1.0 - alpha
        lists += [first, second]
    return w / w.sum(), order, merges


def kkt_gap(cov: np.ndarray, w: np.ndarray, tol_zero: float = 1e-9) -> float:
    """Largest violation of the first-order conditions of min w'Sw, w >= 0, sum w = 1 (gradient 2 S w): held names share one gradient level lam, unheld names are not below it.  Relative to the mean gradient."""
    g = cov @ w
    held = w > tol_zero
    lam = float((w[held] * g[held]).sum() / w[held].sum())
    v_held = np.abs(g[held] - lam).max() if held.any() else 0.0
    v_zero = np.clip(lam - g[~held], 0.0, None).max() if (~held).any() else 0.0
    return float(max(v_held, v_zero) / abs(g.mean()))


def min_variance_long_only(cov: np.ndarray, kkt_tol: float = 1e-4) -> np.ndarray:
    from scipy.optimize import minimize
    n = len(cov)
    scale = float(np.diag(cov).mean())
    S = cov / scale
    x0 = np.full(n, 1.0 / n)
    res = minimize(lambda w: float(w @ S @ w), x0, jac=lambda w: 2.0 * (S @ w), bounds=[(0.0, 1.0)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
                   method="SLSQP", options={"ftol": 1e-15, "maxiter": 2000})
    w = np.clip(res.x, 0.0, None)
    w[w < 1e-9] = 0.0
    w = w / w.sum()
    gap = kkt_gap(cov, w)
    if gap > kkt_tol:
        raise SystemExit(f"minimum-variance solution fails the first-order conditions (gap {gap:.3g} > {kkt_tol}); optimiser status: {res.message}")
    return w
