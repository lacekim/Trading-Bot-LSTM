"""Two-state Gaussian hidden Markov model in plain numpy.

Why this file exists: Advanced Algorithmic Trading's regime-detection chapter (regime_hmm_train.py) fits
hmmlearn.hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000) to one column of returns and its
risk manager uses hmm_model.predict(returns)[-1] as the current regime. hmmlearn is not installed where this
code runs and there is no network, so the same model is written out here.

Same as the book's call: 2 states, one Gaussian per state on the returns, EM (Baum-Welch) up to 1000 iterations
with hmmlearn's default stop (log-likelihood gain < 0.01), uniform start and transition probabilities at the start.
Different from hmmlearn (stated, not hidden): starting means/variances come from a deterministic split of the
returns by absolute size (hmmlearn uses k-means with a random seed), and hmmlearn's small Bayesian priors on the
covariance are not applied (plain maximum likelihood, variance floored at 1e-12).
predict-last: last_state_online(x)[t] equals the last entry of the Viterbi path of x[:t+1], which is what
hmm_model.predict(returns)[-1] returns when called bar by bar. It uses no data after bar t.
"""
import math
import numpy as np

TINY = 1e-300  # only used for log(0) of a start/transition probability


def _log_emis(x, mu, var):
    L = np.empty((len(x), 2))
    for j in range(2):
        L[:, j] = -0.5 * (x - mu[j]) ** 2 / var[j] - 0.5 * math.log(2 * math.pi * var[j])
    return L


def _emis(x, mu, var):
    """Emission densities with each row divided by its larger entry, plus the log of that divisor.
    Exact (no floor): an extreme return can no longer underflow to zero in both states."""
    L = _log_emis(x, mu, var)
    rowmax = L.max(axis=1)
    return np.exp(L - rowmax[:, None]), rowmax


def _forward_backward(B, pi, A):
    T = len(B)
    alpha = np.empty((T, 2)); beta = np.empty((T, 2)); c = np.empty(T)
    a0 = pi[0] * B[0, 0]; a1 = pi[1] * B[0, 1]; s = a0 + a1
    alpha[0, 0] = a0 / s; alpha[0, 1] = a1 / s; c[0] = s
    A00, A01, A10, A11 = A[0][0], A[0][1], A[1][0], A[1][1]
    p0, p1 = alpha[0, 0], alpha[0, 1]
    Bl = B.tolist()
    for t in range(1, T):
        b0, b1 = Bl[t]
        n0 = (p0 * A00 + p1 * A10) * b0; n1 = (p0 * A01 + p1 * A11) * b1; s = n0 + n1
        p0 = n0 / s; p1 = n1 / s
        alpha[t, 0] = p0; alpha[t, 1] = p1; c[t] = s
    q0 = q1 = 1.0; beta[T - 1, 0] = beta[T - 1, 1] = 1.0
    cl = c.tolist()
    for t in range(T - 2, -1, -1):
        b0, b1 = Bl[t + 1]; s = cl[t + 1]
        m0 = (A00 * b0 * q0 + A01 * b1 * q1) / s; m1 = (A10 * b0 * q0 + A11 * b1 * q1) / s
        q0, q1 = m0, m1
        beta[t, 0] = q0; beta[t, 1] = q1
    return alpha, beta, c


def fit(x, n_iter=1000, tol=1e-2):
    """Returns dict(pi, A, mu, var, loglik, iters, ll_path). x: 1-D array of returns."""
    x = np.asarray(x, float); T = len(x)
    dev = np.abs(x - x.mean()); med = np.median(dev)
    lo, hi = x[dev <= med], x[dev > med]
    mu = [float(lo.mean()), float(hi.mean())]
    var = [max(float(lo.var()), 1e-12), max(float(hi.var()), 1e-12)]
    pi = [0.5, 0.5]; A = [[0.5, 0.5], [0.5, 0.5]]
    ll_path = []; prev = -np.inf
    for it in range(1, n_iter + 1):
        B, rowmax = _emis(x, mu, var)
        alpha, beta, c = _forward_backward(B, pi, A)
        ll = float(np.log(c).sum() + rowmax.sum()); ll_path.append(ll)
        if it > 1 and abs(ll - prev) < tol:
            break
        prev = ll
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        newA = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                newA[i][j] = float((alpha[:-1, i] * A[i][j] * B[1:, j] * beta[1:, j] / c[1:]).sum())
            s = newA[i][0] + newA[i][1]
            newA[i] = [newA[i][0] / s, newA[i][1] / s]
        A = newA
        pi = [float(gamma[0, 0]), float(gamma[0, 1])]
        for j in range(2):
            w = gamma[:, j]; sw = w.sum()
            mu[j] = float((w * x).sum() / sw)
            var[j] = max(float((w * (x - mu[j]) ** 2).sum() / sw), 1e-12)
    return dict(pi=pi, A=A, mu=mu, var=var, loglik=ll_path[-1], iters=len(ll_path), ll_path=ll_path)


def last_state_online(x, m):
    """state[t] = last entry of the Viterbi path of x[:t+1] (what hmmlearn's predict(x[:t+1])[-1] gives)."""
    x = np.asarray(x, float)
    logB = _log_emis(x, m["mu"], m["var"]).tolist()
    lA = [[math.log(max(m["A"][i][j], TINY)) for j in range(2)] for i in range(2)]
    d0 = math.log(max(m["pi"][0], TINY)) + logB[0][0]; d1 = math.log(max(m["pi"][1], TINY)) + logB[0][1]
    out = np.empty(len(x), dtype=np.int8); out[0] = 0 if d0 >= d1 else 1
    for t in range(1, len(x)):
        n0 = max(d0 + lA[0][0], d1 + lA[1][0]) + logB[t][0]
        n1 = max(d0 + lA[0][1], d1 + lA[1][1]) + logB[t][1]
        d0, d1 = n0, n1
        out[t] = 0 if d0 >= d1 else 1
    return out


def calm_state(m):
    """Index of the lower-variance state: the book's 'desirable' regime (its text calls it the low-volatility regime;
    its code assumes that is state 0, which hmmlearn does not guarantee)."""
    return 0 if m["var"][0] <= m["var"][1] else 1
