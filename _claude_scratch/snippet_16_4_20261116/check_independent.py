# Independent check of Snippet 16.4's printed weights (run in Claude's workspace, python -I; scipy 1.18.1, pandas 3.0.5, numpy 2.5.3).
# No code from the snippet is imported: the data are rebuilt from the same seeds, the cluster order is scipy's leaves_list,
# and the bisection is written as a recursion on numpy arrays.
# Result 2026-11-16: cols [3, 0, 2, 2, 1]; leaf order 8,3,9,2,10,1,7,5,4,6 = printed order; max |independent - printed| = 3.59e-07
# (the printout has 6 decimals); printed weights sum to 0.999999.
import numpy as np, random, warnings, scipy.cluster.hierarchy as sch
warnings.simplefilter("ignore")
np.random.seed(12345); random.seed(12345)
x = np.random.normal(0, 1, size=(10000, 5))
cols = [random.randint(0, 4) for _ in range(5)]
x = np.hstack([x, x[:, cols] + np.random.normal(0, .25, size=(10000, 5))])
cov = np.cov(x, rowvar=False); sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
np.fill_diagonal(corr, 1.0); dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
order = list(sch.leaves_list(sch.linkage(dist, 'single')))
def cvar(ix):
    c = cov[np.ix_(ix, ix)]; w = 1 / np.diag(c); w /= w.sum(); return w @ c @ w
w = np.ones(10)
def split(ix):
    if len(ix) < 2: return
    a, b = ix[:len(ix)//2], ix[len(ix)//2:]
    v0, v1 = cvar(a), cvar(b); al = 1 - v0 / (v0 + v1)
    w[a] *= al; w[b] *= 1 - al
    split(a); split(b)
split(order)
printed = {8:0.071408,3:0.075961,9:0.125317,2:0.067596,10:0.063236,1:0.109945,7:0.103968,5:0.193357,4:0.097549,6:0.091662}
print("cols", cols, "order", [int(i)+1 for i in order], "sum", round(sum(printed.values()), 6))
print("max |independent - printed| =", f"{max(abs(w[k-1] - v) for k, v in printed.items()):.2e}")
