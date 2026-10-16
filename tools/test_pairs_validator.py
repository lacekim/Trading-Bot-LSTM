"""Synthetic smoke + bug-injection tests for tools/verify_pairs_independent.py (offline; no real data)."""
import copy, sys, time
import numpy as np, pandas as pd
import run_alpaca_tsmom_paper as tp
import run_pairs_test as rp
import verify_pairs_independent as vp

def make(seed=0, T=2000, coint=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=T)
    L = {s: np.cumsum(rng.normal(0.0002, 0.01, T)) + np.log(50) for s in tp.ALL_SYMS}
    if coint:
        for (y, x) in rp.PAIRS:
            ou = np.zeros(T)
            for t in range(1, T):
                ou[t] = 0.93 * ou[t - 1] + rng.normal(0, 0.01)
            L[y] = 0.8 * L[x] + 0.5 + ou
    return pd.DataFrame({s: np.exp(L[s]) for s in tp.ALL_SYMS}, index=idx)

NR = 30
def baseline(close):
    res = rp.run_study(close, "synthetic", write=False, n_random=NR, ledger=False)
    return res

def quiet(fn, *a, **k):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()

def validate(close, saved, ex=rp):
    out, txt = quiet(vp.run, close, "synthetic", ex, saved, False, NR)
    return out

if __name__ == "__main__":
    close = make(0)
    t0 = time.time()
    saved, _ = quiet(baseline, close)
    print("baseline executor done", round(time.time() - t0, 1), "s; overall", saved["overall_pass"], "trades", saved["n_trades"])
    v = validate(close, saved)
    print("CLEAN RUN ->", "VERIFIED" if v["all_passed"] else "FAILED", {k: x["passed"] for k, x in v["checks"].items()})
    if not v["all_passed"]:
        for k, x in v["checks"].items():
            if not x["passed"]:
                print("  ", k, x["detail"])
        sys.exit(1)
    print("validator time", round(time.time() - t0, 1), "s")


# ---------------- bug injection ----------------
import types
SRC = open(rp.__file__).read()

def mutated(old, new, count=1, which=None):
    assert SRC.count(old) >= 1, f"pattern not found: {old!r}"
    if which is None:
        s = SRC.replace(old, new, count)
    else:   # replace only the `which`-th occurrence (0-based)
        parts = SRC.split(old)
        s = old.join(parts[:which + 1]) + new + old.join(parts[which + 1:])
    m = types.ModuleType("run_pairs_test_mut")
    m.__file__ = rp.__file__
    exec(compile(s, "mut", "exec"), m.__dict__)
    return m

MUTS = [
 ("same-bar fill (exec = signal day)", "pos, side, sig, ex, bt = s, s, t, t + 1, float(beta[t])", "pos, side, sig, ex, bt = s, s, t, t, float(beta[t])", None),
 ("exit threshold wrong sign", "(pos == 1 and z[t] >= -EXIT_Z)", "(pos == 1 and z[t] >= EXIT_Z)", None),
 ("time stop off by one", "(t - ex >= MAX_HOLD)", "(t - ex > MAX_HOLD)", None),
 ("beta cap removed", "abs(beta[t]) > BETA_CAP", "abs(beta[t]) > 1e9", None),
 ("borrow on current close", "abs(sy) * P[t - 1, iy] if sy < 0", "abs(sy) * P[t, iy] if sy < 0", None),
 ("exit costs dropped", "E -= cst\n                dc[k] -= cst\n        S =", "pass\n        S =", None),
 ("slot capital from prior equity", "S = E / N_SLOTS                      #", "S = prev / N_SLOTS                      #", None),
 ("Kalman centred on whole-sample mean (look-ahead)", "mx, my = xs.mean(), ys.mean()", "mx, my = lx[i0:].mean(), ly[i0:].mean()", None),
 ("signal uses next-day z (look-ahead)", "if z[t] > ENTRY_Z:\n                s = -1\n            elif z[t] < -ENTRY_Z:", "if z[t + 1] > ENTRY_Z:\n                s = -1\n            elif z[t + 1] < -ENTRY_Z:", None),
 ("Sharpe annualised with 365", "np.sqrt(252)) if s > 0", "np.sqrt(365)) if s > 0", None),
 ("verdict (vii) comparator flipped", 'm["sharpe"] > m_f["sharpe"]', 'm["sharpe"] < m_f["sharpe"]', None),
 ("entry threshold drift 2.0 -> 2.5", "BURN, ENTRY_Z, EXIT_Z, MAX_HOLD, BETA_CAP = 252, 2.0, 0.5, 60, 5.0", "BURN, ENTRY_Z, EXIT_Z, MAX_HOLD, BETA_CAP = 252, 2.5, 0.5, 60, 5.0", None),
 ("bootstrap direction flipped", "if cen[idx].mean() >= obs:", "if cen[idx].mean() <= obs:", None),
 ("June window start moved", "JUNE_START, JUNE_END = tp.JUNE_START, tp.JUNE_END", 'JUNE_START, JUNE_END = pd.Timestamp("2026-06-01"), tp.JUNE_END', None),
 ("random seed drift 7 -> 8", "N_RANDOM, RANDOM_SEED = 500, 7", "N_RANDOM, RANDOM_SEED = 500, 8", None),
 ("halves split at one third", "mid = len(rs) // 2", "mid = len(rs) // 3", None),
 ("frozen hedge fitted on full sample (look-ahead)", "lo, hi = i0, i0 + BURN", "lo, hi = i0, T", None),
 ("top-pair share ignores sign", "pos_sum = float(sum(v for v in tot_c if v > 0))", "pos_sum = float(sum(abs(v) for v in tot_c))", None),
 ("SPY correlation uses lagged SPY", "corr = float(rs.corr(spy))", "corr = float(rs.corr(spy.shift(1)))", None),
 ("entry cost only on y leg", "cst = cost_bp / 1e4 * (abs(sy) * P[t, iy] + abs(sx) * P[t, ix])\n            E -= cst\n            dc[k] -= cst\n            open_", "cst = cost_bp / 1e4 * (abs(sy) * P[t, iy])\n            E -= cst\n            dc[k] -= cst\n            open_", None),
]

def bigbeta(seed=3, T=2000):
    c = make(seed, T)
    rng = np.random.default_rng(99)
    ou = np.zeros(T)
    for t in range(1, T):
        ou[t] = 0.93 * ou[t - 1] + rng.normal(0, 0.01)
    c["GLD"] = np.exp(6.0 * np.log(c["SLV"]) - 20.0 + ou)       # beta ~ 6 > cap
    return c

def mixed(seed=5, T=2000):
    c = make(seed, T, coint=False)
    rng = np.random.default_rng(11)
    for (y, x) in rp.PAIRS[:4]:
        ou = np.zeros(T)
        for t in range(1, T):
            ou[t] = 0.93 * ou[t - 1] + rng.normal(0, 0.01)
        c[y] = np.exp(0.8 * np.log(c[x]) + 0.5 + ou)
    return c

def run_mutations():
    base = make(0)
    big = bigbeta()
    mix = mixed()
    ok_all = True
    for name, old, new, which in MUTS:
        m = mutated(old, new, which=which)
        data = big if "beta cap" in name else (mix if "top-pair" in name else base)
        try:
            saved, _ = quiet(m.run_study, data, "synthetic", False, NR, False)
            v = validate(data, saved, ex=m)
            caught = [k for k, x in v["checks"].items() if not x["passed"]]
        except Exception as exc:  # a crash is also a (loud) detection
            caught = [f"CRASH {type(exc).__name__}: {exc}"]
        status = "CAUGHT" if caught else "MISSED"
        if not caught:
            ok_all = False
        print(f"  {status:7s} {name:52s} by: {', '.join(caught) if caught else '-'}")
    print("\nALL INJECTED BUGS CAUGHT" if ok_all else "\nSOME BUGS WERE MISSED")
    return ok_all

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "mut":
    run_mutations()
