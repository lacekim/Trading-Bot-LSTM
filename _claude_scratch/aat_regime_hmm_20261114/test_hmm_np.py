"""Checks of hmm_np.py. Run: python3 test_hmm_np.py"""
import math, itertools, numpy as np
import hmm_np as H
rng = np.random.default_rng(20261114)
# 1. synthetic two-regime returns with known parameters
T = 6000; A = np.array([[0.98, 0.02], [0.05, 0.95]]); sd = [0.006, 0.025]; mu = [0.0006, -0.001]
s = np.empty(T, int); s[0] = 0
for t in range(1, T): s[t] = rng.choice(2, p=A[s[t - 1]])
x = rng.normal(np.take(mu, s), np.take(sd, s))
m = H.fit(x)
calm = H.calm_state(m); wild = 1 - calm
ll = np.array(m["ll_path"])
print("iters", m["iters"], "sd calm %.5f (true 0.006) sd wild %.5f (true 0.025)" % (math.sqrt(m["var"][calm]), math.sqrt(m["var"][wild])))
assert (np.diff(ll) >= -1e-6).all(), "EM log-likelihood must never fall"
assert abs(math.sqrt(m["var"][calm]) / 0.006 - 1) < 0.15 and abs(math.sqrt(m["var"][wild]) / 0.025 - 1) < 0.15
assert abs(m["A"][calm][calm] - 0.98) < 0.03 and abs(m["A"][wild][wild] - 0.95) < 0.05
st = H.last_state_online(x, m)
acc = ((st == wild).astype(int) == s).mean()
print("online regime accuracy vs truth: %.3f" % acc); assert acc > 0.85
# 2. log-likelihood by a separate log-space forward pass equals the scaled forward pass
def ll_logspace(x, m):
    lb = lambda v, j: -0.5 * math.log(2 * math.pi * m["var"][j]) - 0.5 * (v - m["mu"][j]) ** 2 / m["var"][j]
    a = [math.log(m["pi"][j]) + lb(x[0], j) for j in range(2)]
    for v in x[1:]:
        a = [math.log(sum(math.exp(a[i] - max(a)) * m["A"][i][j] for i in range(2))) + max(a) + lb(v, j) for j in range(2)]
    return max(a) + math.log(sum(math.exp(q - max(a)) for q in a))
B, rowmax = H._emis(x, m["mu"], m["var"]); _, _, c = H._forward_backward(B, m["pi"], m["A"])
d = abs(ll_logspace(x, m) - float(np.log(c).sum() + rowmax.sum())); print("loglik difference between the two forward passes: %.2e" % d); assert d < 1e-6
# 3. Viterbi-last-state online equals brute-force Viterbi on short prefixes (all 2^n paths)
xs = x[:12]
def brute_last(xp):
    best = None
    for path in itertools.product((0, 1), repeat=len(xp)):
        lp = math.log(m["pi"][path[0]]) + H._log_emis(np.array([xp[0]]), m["mu"], m["var"])[0, path[0]]
        for t in range(1, len(xp)):
            lp += math.log(m["A"][path[t - 1]][path[t]]) + H._log_emis(np.array([xp[t]]), m["mu"], m["var"])[0, path[t]]
        if best is None or lp > best[0]: best = (lp, path[-1])
    return best[1]
on = H.last_state_online(xs, m)
assert all(on[k - 1] == brute_last(xs[:k]) for k in range(1, 13)); print("online last state == brute-force Viterbi last state on 12 prefixes")
# 4. no look-ahead: the state at bar t does not change when later bars are removed
full = H.last_state_online(x, m)
for k in (50, 777, 3001, 5999): assert (H.last_state_online(x[:k + 1], m) == full[:k + 1]).all()
print("no look-ahead: states unchanged when later data is removed")
# 5. regression: extreme outliers (found on ETH: +700% and +175% 4h bars in its first weeks in August 2015)
xo = x.copy(); xo[100] = 7.0; xo[900] = 1.75; xo[901] = -0.92
mo = H.fit(xo); llo = np.array(mo["ll_path"])
assert (np.diff(llo) >= -1e-6).all() and mo["iters"] > 2, "EM must keep climbing with extreme outliers present"
print("outlier regression: %d iterations, log-likelihood never falls" % mo["iters"])
print("ALL CHECKS PASSED")
