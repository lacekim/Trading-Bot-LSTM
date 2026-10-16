"""
INDEPENDENT VERIFICATION of the Kalman-pairs test (tools/run_pairs_test.py, test #6).  VALIDATOR stage.
OFFLINE. Reads only the cached ETF prices (runtime/alpaca_tsmom_etfs/etfs.parquet) and, if present, the saved results
runtime/alpaca_pairs_results.json. No network, no API keys, touches no live-bot file.
Writes runtime/pairs_verification_results.json.

PURPOSE: catch bugs that a test written by the same author with the same code cannot. Everything is re-implemented with
different code paths: 2x2 matrix Kalman with Joseph-form covariance update and a least-squares initialisation, a forward
scan trade finder (instead of a state machine), an array-based position ledger (instead of a dict of open trades), loop
based monthly compounding, bootstrap and metrics. Only the price loader and the executor UNDER TEST are imported.

------------------------- PREDECLARED CHECKS AND TOLERANCES (fixed before any result) -------------------------
 1. SPEC DRIFT      : constants re-declared here equal the constants the executor actually uses (exact).
 2. DATA INTEGRITY  : dates unique+increasing, no non-positive prices, no gap > 5 business days inside any series,
                      all 16 pair symbols present.
 3. KALMAN / FROZEN : independent z and hedge ratio vs executor for all 8 pairs (and delta 1e-5, 1e-7), max abs diff
                      <= 1e-8; frozen-hedge z <= 1e-10.
 4. TRADES          : trade lists (signal day, execution day, exit execution day, side) identical; hedge ratios <= 1e-8.
 5. DAILY RETURNS   : independent ledger vs executor simulate() for headline, frozen control, 20bp/3% borrow, lag 1:
                      max abs daily diff <= 1e-9, per-pair contributions <= 1e-9.
 6. METRICS         : independent CAGR / vol / Sharpe vs the executor metric function on the SAME series <= 1e-9;
                      maxDD <= 0.005 (the executor omits the starting equity 1.0 from the peak).
 7. SAVED RESULTS   : independent numbers vs the saved JSON (headline metrics, SPY correlation, trades, frozen control,
                      random-pair percentiles and rank, halves, calendar years, June 2026 window, per-pair table,
                      bootstrap p, every sensitivity row, top-pair share): abs diff <= 1e-8 (<= 0.005 maxDD).
 8. VERDICT         : the 8 predeclared conditions and the overall verdict recomputed from the INDEPENDENT numbers equal
                      the saved verdict.
 9. LOOK-AHEAD AUDIT: at 8 random days (seed 2024) overwrite ALL prices after that day with random values. z, hedge ratio,
                      every trade decided by that day and every daily return up to that day must stay identical
                      (<= 1e-12). Run on BOTH the independent code and the executor.
VERDICT: "PAIRS TEST VERIFIED" only if every check passes. A pass proves the code does what the written spec says;
it does NOT prove the spec is a good strategy.
-----------------------------------------------------------------------------------------------------------------

Run (project root, `javier` env):  python tools/verify_pairs_independent.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import date

import numpy as np
import pandas as pd

import run_alpaca_tsmom_paper as tp          # price loader
import run_pairs_test as rp                  # executor under test

OUT_PATH = tp.PROJECT_ROOT / "runtime" / "pairs_verification_results.json"

# ---- spec, re-declared here on purpose (check 1 compares against the executor's constants) ----
SYMS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "EWZ", "FXI",
        "IEF", "TLT", "TIP", "LQD", "HYG",
        "GLD", "SLV", "USO", "UNG", "DBA", "DBB",
        "UUP", "FXE", "FXY", "FXB", "FXA", "FXC"]
PAIRS = [("GLD", "SLV"), ("IEF", "TLT"), ("LQD", "HYG"), ("SPY", "QQQ"),
         ("EFA", "EEM"), ("FXA", "FXC"), ("FXE", "UUP"), ("DBA", "DBB")]
DELTA = 1e-6
BURN, ENTRY_Z, EXIT_Z, MAX_HOLD, BETA_CAP = 252, 2.0, 0.5, 60, 5.0
N_SLOTS = 8
COST_BP, BORROW = 5.0, 0.01
N_RANDOM, RANDOM_SEED = 500, 7
BOOT_N, BOOT_BLOCK, BOOT_SEED = 2000, 6, 123
BAR = dict(sharpe=0.50, p=0.01, corr=0.30, pct=95.0)
JUNE = (pd.Timestamp("2026-05-15"), pd.Timestamp("2026-07-01"))
N_POISON = 8
TOL = dict(kal=1e-8, frozen=1e-10, beta=1e-8, ret=1e-9, contrib=1e-9, met=1e-9, dd=0.005, saved=1e-8, poison=1e-12)


# ======================= independent implementation =======================

def ikalman(ly: np.ndarray, lx: np.ndarray, i0: int, delta: float):
    """Matrix Kalman, Joseph-form update, lstsq initialisation. Returns (z, beta); NaN before i0+BURN."""
    T = len(ly)
    z = np.full(T, np.nan)
    beta = np.full(T, np.nan)
    xw, yw = lx[i0:i0 + BURN], ly[i0:i0 + BURN]
    mx, my = xw.mean(), yw.mean()
    X0 = np.column_stack([xw - mx, np.ones(BURN)])
    coef, *_ = np.linalg.lstsq(X0, yw - my, rcond=None)
    theta = np.array([coef[0], 0.0])
    resid = (yw - my) - X0 @ np.array([coef[0], coef[1]])
    s2 = float(resid @ resid) / (BURN - 2)
    C = s2 * np.linalg.inv(X0.T @ X0)
    W = delta / (1.0 - delta) * np.eye(2)
    I2 = np.eye(2)
    for t in range(i0 + BURN, T):
        F = np.array([lx[t] - mx, 1.0])
        R = C + W
        e = (ly[t] - my) - F @ theta
        Q = F @ R @ F + s2
        z[t] = e / np.sqrt(Q)
        K = R @ F / Q
        theta = theta + K * e
        A = I2 - np.outer(K, F)
        C = A @ R @ A.T + s2 * np.outer(K, K)
        beta[t] = theta[0]
    return z, beta


def ifrozen(ly: np.ndarray, lx: np.ndarray, i0: int):
    xw, yw = lx[i0:i0 + BURN], ly[i0:i0 + BURN]
    A = np.column_stack([xw, np.ones(BURN)])
    (b, a), *_ = np.linalg.lstsq(A, yw, rcond=None)
    res = yw - (b * xw + a)
    sd = np.sqrt(((res - res.mean()) ** 2).sum() / (BURN - 1))
    T = len(ly)
    z = np.full(T, np.nan)
    z[i0:] = (ly[i0:] - b * lx[i0:] - a) / sd
    be = np.full(T, np.nan)
    be[i0:] = b
    return z, be


def itrades(z: np.ndarray, beta: np.ndarray, i0: int):
    """Forward-scan trade finder. Returns list of (signal_day, exec_day, exit_exec_day or -1, side, beta)."""
    T = len(z)
    out = []
    t = i0 + BURN
    last_sig = T - 2                      # last day on which a signal can still be executed (next close exists)
    while t <= last_sig:
        side = 0
        if z[t] > ENTRY_Z:
            side = -1
        elif z[t] < -ENTRY_Z:
            side = 1
        if side == 0 or np.isnan(beta[t]) or abs(beta[t]) > BETA_CAP:
            t += 1
            continue
        ex = t + 1
        u = t + 1
        exit_exec = -1
        while u <= last_sig:
            if (side == 1 and z[u] >= -EXIT_Z) or (side == -1 and z[u] <= EXIT_Z) or (u - ex >= MAX_HOLD):
                exit_exec = u + 1
                break
            u += 1
        out.append((t, ex, exit_exec, side, float(beta[t])))
        if exit_exec == -1:
            break
        t = u + 1                          # earliest next signal is the day after the exit signal
    return out


def iledger(P: np.ndarray, cols: list, trades: list, cost_bp: float, borrow: float, start: int, lag: int = 0):
    """Array-based ledger. Returns (daily returns from `start`, per-pair contributions from `start`)."""
    T, K = P.shape[0], len(cols)
    iy = np.array([c[0] for c in cols])
    ix = np.array([c[1] for c in cols])
    sy, sx = np.zeros(K), np.zeros(K)
    ent = [[] for _ in range(T)]
    ext = [[] for _ in range(T)]
    for k, trs in enumerate(trades):
        for (sg, ex, xe, side, b) in trs:
            if ex + lag < T:
                ent[ex + lag].append((k, side, b))
                if xe != -1 and xe + lag < T:
                    ext[xe + lag].append(k)
    E = 1.0
    ret = np.zeros(T)
    con = np.zeros((T, K))
    c_bp = cost_bp / 1e4
    for t in range(1, T):
        prev = E
        dy = np.where(sy != 0, P[t, iy] - P[t - 1, iy], 0.0)      # flat slots contribute exactly 0 even if the price is NaN
        dx = np.where(sx != 0, P[t, ix] - P[t - 1, ix], 0.0)
        short = np.where(sy < 0, -sy * P[t - 1, iy], 0.0) + np.where(sx < 0, -sx * P[t - 1, ix], 0.0)
        c = sy * dy + sx * dx - borrow / 252.0 * short
        E += float(c.sum())
        for k in ext[t]:
            cst = c_bp * (abs(sy[k]) * P[t, iy[k]] + abs(sx[k]) * P[t, ix[k]])
            E -= cst
            c[k] -= cst
            sy[k] = sx[k] = 0.0
        S = E / N_SLOTS
        for (k, side, b) in ent[t]:
            ny = side * S / (1.0 + abs(b))
            nx = -side * b * S / (1.0 + abs(b))
            sy[k], sx[k] = ny / P[t, iy[k]], nx / P[t, ix[k]]
            cst = c_bp * (abs(sy[k]) * P[t, iy[k]] + abs(sx[k]) * P[t, ix[k]])
            E -= cst
            c[k] -= cst
        ret[t] = E / prev - 1.0
        con[t] = c
    return ret[start:], con[start:]


def imetrics(r: np.ndarray) -> dict:
    r = np.asarray(r, float)
    n = len(r)
    eq = np.cumprod(1.0 + r)
    cagr = float(eq[-1] ** (252.0 / n) - 1.0)
    sd = float(np.sqrt(((r - r.mean()) ** 2).sum() / (n - 1)))
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1.0).min())
    return dict(cagr=cagr, vol=sd * np.sqrt(252.0), sharpe=float(r.mean() / sd * np.sqrt(252.0)) if sd > 0 else float("nan"), max_dd=dd)


def imonthly(r: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    out, cur, key = [], 1.0, None
    for v, d in zip(r, dates):
        k = (d.year, d.month)
        if key is not None and k != key:
            out.append(cur - 1.0)
            cur = 1.0
        key = k
        cur *= 1.0 + v
    out.append(cur - 1.0)
    return np.array(out)


def iboot(d: np.ndarray) -> float:
    rng = np.random.default_rng(BOOT_SEED)
    n = len(d)
    obs = float(np.sum(d) / n)
    cen = [v - obs for v in d]
    nb = -(-n // BOOT_BLOCK)
    hits = 0
    for _ in range(BOOT_N):
        starts = rng.integers(0, n, nb)
        picks = []
        for s0 in starts:
            for j in range(BOOT_BLOCK):
                picks.append((int(s0) + j) % n)
        picks = picks[:n]
        if sum(cen[i] for i in picks) / n >= obs:
            hits += 1
    return (hits + 1) / (BOOT_N + 1)


def isharpe(r: np.ndarray) -> float:
    return imetrics(r)["sharpe"]


def window_prod(r: np.ndarray, dates: pd.DatetimeIndex, lo, hi) -> float:
    m = [(d >= lo) and (d <= hi) for d in dates]
    x = r[np.array(m, bool)]
    return float(np.prod(1.0 + x) - 1.0) if len(x) else float("nan")


# ======================= verification =======================

def run(close: pd.DataFrame, provenance: str, ex=rp, saved: dict | None = None, write: bool = True,
        n_random: int | None = None) -> dict:
    results = {}

    def record(name, ok, detail):
        results[name] = dict(passed=bool(ok), detail=detail)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(f"VALIDATOR  provenance: {provenance}")
    print("\n=== 1. SPEC DRIFT ===")
    pairs_ex = [tuple(p) for p in ex.PAIRS]
    pairs_ok = pairs_ex == PAIRS
    drift = []
    for nm, mine, theirs in [("DELTA", DELTA, ex.DELTA), ("BURN", BURN, ex.BURN), ("ENTRY_Z", ENTRY_Z, ex.ENTRY_Z),
                             ("EXIT_Z", EXIT_Z, ex.EXIT_Z), ("MAX_HOLD", MAX_HOLD, ex.MAX_HOLD), ("BETA_CAP", BETA_CAP, ex.BETA_CAP),
                             ("N_SLOTS", N_SLOTS, ex.N_SLOTS), ("COST_BP", COST_BP, ex.COST_BP), ("BORROW", BORROW, ex.BORROW),
                             ("N_RANDOM", N_RANDOM, ex.N_RANDOM), ("RANDOM_SEED", RANDOM_SEED, ex.RANDOM_SEED),
                             ("BOOT_N", BOOT_N, ex.BOOT_N), ("BOOT_BLOCK", BOOT_BLOCK, ex.BOOT_BLOCK), ("BOOT_SEED", BOOT_SEED, ex.BOOT_SEED)]:
        if mine != theirs:
            drift.append(f"{nm}: validator {mine} vs executor {theirs}")
    if tuple(sorted(BAR.items())) != tuple(sorted(ex.BAR.items())):
        drift.append(f"BAR: validator {BAR} vs executor {ex.BAR}")
    if [s for s in tp.ALL_SYMS] != SYMS:
        drift.append("symbol universe/order differs")
    if not pairs_ok:
        drift.append(f"PAIRS: validator {PAIRS} vs executor {pairs_ex}")
    record("spec_drift", not drift, "all constants, pairs and universe identical" if not drift else "; ".join(drift))

    syms = [s for s in SYMS if s in close.columns]
    c = close[syms].sort_index().ffill()
    idx = c.index
    P = c.values.astype(float)
    T = len(idx)
    col = {s: i for i, s in enumerate(syms)}

    print("\n=== 2. DATA INTEGRITY ===")
    problems = []
    if not idx.is_unique or not idx.is_monotonic_increasing:
        problems.append("dates not unique/increasing")
    if np.nanmin(P) <= 0:
        problems.append("non-positive price")
    need = sorted({s for p in PAIRS for s in p})
    miss = [s for s in need if s not in col]
    if miss:
        problems.append(f"missing symbols {miss}")
    bd = np.busday_count(idx.values.astype("datetime64[D]")[:-1], idx.values.astype("datetime64[D]")[1:])
    if (bd > 5).any():
        problems.append(f"gap > 5 business days at {idx[int(np.argmax(bd))].date()}")
    record("data_integrity", not problems, f"{T} dates {idx[0].date()}..{idx[-1].date()}, {len(syms)} symbols" if not problems else "; ".join(problems))
    if miss:
        return dict(checks=results, all_passed=False)

    i0s = {s: int(np.argmax(~np.isnan(P[:, col[s]]))) for s in syms}
    start = BURN + 1 + max(i0s.values())
    L = np.log(P)

    # ---- per pair independent objects ----
    def indep_pair(y, x, delta=DELTA, frozen=False):
        i0 = max(i0s[y], i0s[x])
        z, b = (ifrozen if frozen else (lambda a, bb, c0: ikalman(a, bb, c0, delta)))(L[:, col[y]], L[:, col[x]], i0)
        return z, b, itrades(z, b, i0), i0

    ind = {"kal": [indep_pair(y, x) for (y, x) in PAIRS],
           "frz": [indep_pair(y, x, frozen=True) for (y, x) in PAIRS],
           "d1e-5": [indep_pair(y, x, 1e-5) for (y, x) in PAIRS],
           "d1e-7": [indep_pair(y, x, 1e-7) for (y, x) in PAIRS]}
    exe = {}
    for key, delta, fr in [("kal", DELTA, False), ("frz", None, True), ("d1e-5", 1e-5, False), ("d1e-7", 1e-7, False)]:
        lst = []
        for (y, x) in PAIRS:
            i0 = max(i0s[y], i0s[x])
            if fr:
                z, b = ex.frozen_z(L[:, col[y]], L[:, col[x]], i0)
            else:
                z, b = ex.kalman_z(L[:, col[y]], L[:, col[x]], i0, delta)
            lst.append((z, b, ex.generate_trades(z, b, i0), i0))
        exe[key] = lst

    print("\n=== 3. KALMAN / FROZEN z AND HEDGE RATIO ===")
    bad, worst = [], 0.0
    for key in ind:
        for k, (y, x) in enumerate(PAIRS):
            zi, bi = ind[key][k][:2]
            ze, be = exe[key][k][:2]
            m = ~np.isnan(zi)
            if (m != ~np.isnan(ze)).any():
                bad.append(f"{key} {y}/{x}: NaN pattern differs")
                continue
            mm = m & ~np.isnan(ze)
            dz = float(np.max(np.abs(zi[mm] - ze[mm]))) if mm.any() else 0.0
            db = float(np.max(np.abs(bi[mm] - be[mm]))) if mm.any() else 0.0
            tol = TOL["frozen"] if key == "frz" else TOL["kal"]
            worst = max(worst, dz)
            if dz > tol or db > TOL["beta"]:
                bad.append(f"{key} {y}/{x}: dz {dz:.2e}, dbeta {db:.2e}")
    record("kalman_frozen", not bad, f"worst |dz| {worst:.2e} over 4 variants x 8 pairs" if not bad else "; ".join(bad))

    print("\n=== 4. TRADE LISTS ===")
    bad = []
    n_tr = {}
    for key in ind:
        n_tr[key] = 0
        for k, (y, x) in enumerate(PAIRS):
            a, b = ind[key][k][2], exe[key][k][2]
            n_tr[key] += len(a)
            if [t[:4] for t in a] != [t[:4] for t in b]:
                bad.append(f"{key} {y}/{x}: independent {len(a)} trades vs executor {len(b)}")
            elif any(abs(u[4] - v[4]) > TOL["beta"] for u, v in zip(a, b)):
                bad.append(f"{key} {y}/{x}: hedge ratio at entry differs")
    record("trades", not bad, f"identical; trades kal/frozen/d1e-5/d1e-7 = {n_tr['kal']}/{n_tr['frz']}/{n_tr['d1e-5']}/{n_tr['d1e-7']}" if not bad else "; ".join(bad))

    print("\n=== 5. DAILY RETURNS AND PER-PAIR CONTRIBUTIONS ===")
    colsP = [(col[y], col[x]) for (y, x) in PAIRS]
    scen = [("headline", "kal", COST_BP, BORROW, 0), ("frozen", "frz", COST_BP, BORROW, 0),
            ("20bp/3%", "kal", 20.0, 0.03, 0), ("lag1", "kal", COST_BP, BORROW, 1),
            ("1.5bp", "kal", 1.5, BORROW, 0), ("10bp", "kal", 10.0, BORROW, 0),
            ("borrow0", "kal", COST_BP, 0.0, 0), ("borrow3", "kal", COST_BP, 0.03, 0),
            ("delta1e-5", "d1e-5", COST_BP, BORROW, 0), ("delta1e-7", "d1e-7", COST_BP, BORROW, 0)]
    ind_r, ind_c, exe_r = {}, {}, {}
    bad = []
    for name, key, cbp, bor, lag in scen:
        ri, ci = iledger(P, colsP, [t[2] for t in ind[key]], cbp, bor, start, lag)
        re_, ce_, _, _ = ex.simulate(P, colsP, [t[2] for t in exe[key]], cbp, bor, start, lag)
        ind_r[name], ind_c[name], exe_r[name] = ri, ci, re_
        dr = float(np.max(np.abs(ri - re_)))
        dc = float(np.max(np.abs(ci - ce_)))
        print(f"    {name:10s} max|dr|={dr:.2e} max|dcontrib|={dc:.2e}")
        if dr > TOL["ret"] or dc > TOL["contrib"]:
            bad.append(f"{name}: dr {dr:.2e}, dcontrib {dc:.2e}")
    record("daily_returns", not bad, "all 10 scenarios within tolerance" if not bad else "; ".join(bad))

    print("\n=== 6. METRICS (independent vs executor metric function, same series) ===")
    bad = []
    for name in ("headline", "frozen", "20bp/3%"):
        s = pd.Series(exe_r[name], index=idx[start:])
        ma, mi = tp.base.metrics(s), imetrics(exe_r[name])
        for key in ("cagr", "vol", "sharpe"):
            if abs(ma[key] - mi[key]) > TOL["met"]:
                bad.append(f"{name}.{key} {ma[key]:.8f} vs {mi[key]:.8f}")
        if abs(ma["max_dd"] - mi["max_dd"]) > TOL["dd"]:
            bad.append(f"{name}.max_dd")
    record("metrics", not bad, "agree" if not bad else "; ".join(bad))

    # ---- independent study numbers ----
    r = ind_r["headline"]
    dates = idx[start:]
    mh = imetrics(r)
    spy_full = np.full(T, np.nan)
    spy_full[1:] = P[1:, col["SPY"]] / P[:-1, col["SPY"]] - 1.0
    spy = spy_full[start:]
    corr = float(np.corrcoef(r, spy)[0, 1])
    mm = imonthly(r, dates)
    p_boot = iboot(mm)
    mid = len(r) // 2
    half_tot = {"first half": float(np.prod(1 + r[:mid]) - 1), "second half": float(np.prod(1 + r[mid:]) - 1)}
    frozen_sh = imetrics(ind_r["frozen"])["sharpe"]
    cagr20 = imetrics(ind_r["20bp/3%"])["cagr"]
    contrib_pct = ind_c["headline"].sum(0) * 100
    pos_sum = float(sum(v for v in contrib_pct if v > 0))
    top_share = float(contrib_pct.max() / pos_sum) if pos_sum > 0 else float("nan")

    nrand = n_random if n_random is not None else (saved["random_n"] if saved and "random_n" in saved else N_RANDOM)
    all_pairs = list(itertools.combinations(SYMS, 2))
    rng = np.random.default_rng(RANDOM_SEED)
    pc = {}

    def pair_obj(pi):
        if pi not in pc:
            y, x = all_pairs[pi]
            i0 = max(i0s[y], i0s[x])
            z, b = ikalman(L[:, col[y]], L[:, col[x]], i0, DELTA)
            pc[pi] = ((col[y], col[x]), itrades(z, b, i0))
        return pc[pi]

    rsh = []
    for _ in range(nrand):
        pick = rng.choice(len(all_pairs), size=N_SLOTS, replace=False)
        objs = [pair_obj(int(q)) for q in pick]
        rr = iledger(P, [o[0] for o in objs], [o[1] for o in objs], COST_BP, BORROW, start)[0]
        rsh.append(isharpe(rr))
    rsh = np.array(rsh)
    pct = {q: float(np.nanpercentile(rsh, q)) for q in (5, 25, 50, 75, 95)}
    rank = float((rsh <= mh["sharpe"]).mean() * 100)
    ind_verdict = {
        "i_sharpe_ge_0.50": bool(mh["sharpe"] >= BAR["sharpe"]),
        "ii_bootstrap_p_lt_0.01": bool(p_boot < BAR["p"]),
        "iii_cagr_pos_at_20bp_3pct_borrow": bool(cagr20 > 0),
        "iv_abs_corr_spy_le_0.30": bool(abs(corr) <= BAR["corr"]),
        "v_sharpe_ge_p95_random_pairs": bool(mh["sharpe"] >= pct[95]),
        "vi_positive_both_halves": bool(all(v > 0 for v in half_tot.values())),
        "vii_beats_frozen_hedge": bool(mh["sharpe"] > frozen_sh),
        "viii_top_pair_share_le_50pct": bool(top_share <= 0.5) if top_share == top_share else False,
    }
    ind_overall = all(ind_verdict.values())
    june = window_prod(r, dates, *JUNE)
    june_spy = window_prod(np.nan_to_num(spy), dates, *JUNE)
    print(f"\n  independent headline: Sharpe {mh['sharpe']:.4f}, CAGR {mh['cagr']*100:.3f}%, corr SPY {corr:+.3f}, boot p {p_boot:.4f}, "
          f"random-pair p95 {pct[95]:.3f} (n={nrand}), frozen Sharpe {frozen_sh:.3f}, June 2026 {june*100:.2f}%")

    print("\n=== 7. SAVED RESULTS (your terminal run as written to JSON) ===")
    if saved is None and ex.OUT_PATH.exists():
        saved = json.loads(ex.OUT_PATH.read_text())
    if saved is None:
        record("saved_results", False, f"{ex.OUT_PATH} missing - run tools/run_pairs_test.py first")
    else:
        mism = []

        def cmp(label, a, b, tol=TOL["saved"]):
            na = a is None or (isinstance(a, float) and a != a)
            nb_ = b is None or (isinstance(b, float) and b != b)
            if na or nb_:
                if not (na and nb_):
                    mism.append(f"{label}: saved {a} vs independent {b}")
                return
            if abs(float(a) - float(b)) > tol:
                mism.append(f"{label}: saved {float(a):.8f} vs independent {float(b):.8f}")

        for k in ("cagr", "vol", "sharpe"):
            cmp(f"overall.{k}", saved["overall"][k], mh[k])
        cmp("overall.max_dd", saved["overall"]["max_dd"], mh["max_dd"], TOL["dd"])
        cmp("corr_spy", saved["corr_spy"], corr)
        cmp("n_trades", saved["n_trades"], n_tr["kal"], 0)
        cmp("frozen.sharpe", saved["frozen"]["sharpe"], frozen_sh)
        cmp("bootstrap_p", saved["bootstrap_p"], p_boot, 1e-12)
        cmp("monthly_mean", saved["monthly_mean"], float(mm.mean()))
        cmp("n_months", saved["n_months"], len(mm), 0)
        if saved.get("random_n") == nrand:
            for q in (5, 25, 50, 75, 95):
                cmp(f"random p{q}", saved["random_percentiles"][str(q)] if str(q) in saved["random_percentiles"] else saved["random_percentiles"][q], pct[q])
            cmp("chosen_percentile_rank", saved["chosen_percentile_rank"], rank, 1e-9)
        else:
            mism.append(f"random_n differs (saved {saved.get('random_n')} vs validator {nrand}) - percentiles not compared")
        for kname, v in half_tot.items():
            cmp(f"half total {kname}", saved["halves"][kname]["total"], v)
        yrs_ind = {}
        for d_, v in zip(dates, r):
            yrs_ind[d_.year] = yrs_ind.get(d_.year, 1.0) * (1.0 + v)
        yrs_spy = {}
        for d_, v in zip(dates, np.nan_to_num(spy)):
            yrs_spy[d_.year] = yrs_spy.get(d_.year, 1.0) * (1.0 + v)
        for y_, v in yrs_ind.items():
            cmp(f"year {y_} pairs", saved["calendar_years"][str(y_)]["pairs"], (v - 1) * 100, 1e-6)
            cmp(f"year {y_} spy", saved["calendar_years"][str(y_)]["spy"], (yrs_spy[y_] - 1) * 100, 1e-6)
        cmp("june pairs", saved["june_2026"]["pairs"], june)
        cmp("june spy", saved["june_2026"]["spy"], june_spy)
        for k, (y, x) in enumerate(PAIRS):
            pp = saved["per_pair"][f"{y}/{x}"]
            cmp(f"{y}/{x} contribution_pct", pp["contribution_pct"], contrib_pct[k], 1e-6)
            cmp(f"{y}/{x} trades", pp["trades"], len(ind["kal"][k][2]), 0)
        cmp("top_pair_share", saved["top_pair_share"], top_share)
        cmp("sum_pair_contrib_pct", saved["sum_pair_contrib_pct"], float(contrib_pct.sum()), 1e-6)
        sens_map = {"cost1.5_borrow0.01": "1.5bp", "cost10_borrow0.01": "10bp", "cost20_borrow0.01": None,
                    "cost5_borrow0": "borrow0", "cost5_borrow0.03": "borrow3", "cost20_borrow0.03": "20bp/3%",
                    "delta1e-05": "delta1e-5", "delta1e-07": "delta1e-7", "lag1": "lag1"}
        for sk, nm in sens_map.items():
            if sk not in saved["sensitivity"]:
                mism.append(f"sensitivity row {sk} missing in saved results")
                continue
            if nm is None:
                ri, _ = iledger(P, colsP, [t[2] for t in ind["kal"]], 20.0, BORROW, start)
            else:
                ri = ind_r[nm]
            mi = imetrics(ri)
            cmp(f"sens {sk}.sharpe", saved["sensitivity"][sk]["sharpe"], mi["sharpe"])
            cmp(f"sens {sk}.cagr", saved["sensitivity"][sk]["cagr"], mi["cagr"])
        record("saved_results", not mism, f"{len(sens_map)} sensitivity rows + headline, controls, random pairs, halves, years, June, per-pair all match" if not mism else "; ".join(mism[:12]))

    print("\n=== 8. VERDICT RECOMPUTED FROM INDEPENDENT NUMBERS ===")
    if saved is None:
        record("verdict", False, "no saved results to compare")
    else:
        diffs = [k for k in ind_verdict if saved["verdict"].get(k) != ind_verdict[k]]
        missing = [k for k in ind_verdict if k not in saved["verdict"]]
        ok = not diffs and not missing and bool(saved["overall_pass"]) == ind_overall
        for k, v in ind_verdict.items():
            print(f"    {k:40s} independent {v}   saved {saved['verdict'].get(k)}")
        record("verdict", ok, f"overall independent {'PASS' if ind_overall else 'FAIL'} vs saved {'PASS' if saved['overall_pass'] else 'FAIL'}" if ok
               else f"conditions differing {diffs}, missing {missing}, overall independent {ind_overall} vs saved {saved['overall_pass']}")

    print("\n=== 9. LOOK-AHEAD AUDIT (poison every price after day j) ===")
    prng = np.random.default_rng(2024)
    js = sorted(int(v) for v in prng.choice(np.arange(start + 20, T - 5), size=N_POISON, replace=False))
    leaks = []
    for j in js:
        Q = P.copy()
        Q[j + 1:] = prng.uniform(20.0, 200.0, size=Q[j + 1:].shape)
        LQ = np.log(Q)
        bad_here = []
        itr_q, etr_q = [], []
        for k, (y, x) in enumerate(PAIRS):
            i0 = max(i0s[y], i0s[x])
            zi, bi = ikalman(LQ[:, col[y]], LQ[:, col[x]], i0, DELTA)
            ze, be = ex.kalman_z(LQ[:, col[y]], LQ[:, col[x]], i0, DELTA)
            zi0, bi0 = ind["kal"][k][:2]
            ze0, be0 = exe["kal"][k][:2]
            ti, te = itrades(zi, bi, i0), ex.generate_trades(ze, be, i0)
            itr_q.append(ti)
            etr_q.append(te)
            for nm, a, b0 in (("indep z", zi, zi0), ("indep beta", bi, bi0), ("exec z", ze, ze0), ("exec beta", be, be0)):
                sl = slice(0, j + 1)
                dd = np.nanmax(np.abs(a[sl] - b0[sl])) if (~np.isnan(a[sl])).any() else 0.0
                if dd > TOL["poison"]:
                    bad_here.append(f"{PAIRS[k]} {nm} d={dd:.1e}")

            def decided(trs0, trs1):
                def cut(trs):
                    out = []
                    for (sg, ex_, xe, sd, bb) in trs:
                        if sg <= j:
                            out.append((sg, ex_, xe if (xe != -1 and xe - 1 <= j) else None, sd))
                    return out
                return cut(trs0) == cut(trs1)
            if not decided(ind["kal"][k][2], ti):
                bad_here.append(f"{PAIRS[k]} indep trades decided <= j changed")
            if not decided(exe["kal"][k][2], te):
                bad_here.append(f"{PAIRS[k]} exec trades decided <= j changed")
        ri = iledger(Q, colsP, itr_q, COST_BP, BORROW, 1)[0]
        re_ = ex.simulate(Q, colsP, etr_q, COST_BP, BORROW, 1)[0]
        ri0 = iledger(P, colsP, [t[2] for t in ind["kal"]], COST_BP, BORROW, 1)[0]
        re0 = ex.simulate(P, colsP, [t[2] for t in exe["kal"]], COST_BP, BORROW, 1)[0]
        di = float(np.max(np.abs(ri[:j] - ri0[:j])))      # ret index 0 == day 1, so [:j] covers days 1..j
        de = float(np.max(np.abs(re_[:j] - re0[:j])))
        if di > TOL["poison"]:
            bad_here.append(f"indep returns d={di:.1e}")
        if de > TOL["poison"]:
            bad_here.append(f"exec returns d={de:.1e}")
        print(f"    day {idx[j].date()} (index {j}): returns up to j changed by indep {di:.1e} / executor {de:.1e}" + (f"  <- {bad_here}" if bad_here else ""))
        if bad_here:
            leaks.append(f"{idx[j].date()}: {bad_here[:3]}")
    record("lookahead_audit", not leaks, f"{len(js)} days, nothing decided on or before the day changed" if not leaks else "; ".join(leaks))

    all_ok = all(v["passed"] for v in results.values())
    print(f"\n  OVERALL: {'PAIRS TEST VERIFIED' if all_ok else 'VERIFICATION FAILED - see failing checks above'}")
    print("  Meaning: the executor does what its written spec says. It does not say the spec is a good strategy,")
    print("  and it does not check the cached price data against an outside source.")
    res = dict(provenance=provenance, checks=results, all_passed=all_ok, tolerances=TOL,
               independent=dict(sharpe=mh["sharpe"], cagr=mh["cagr"], corr_spy=corr, bootstrap_p=p_boot, frozen_sharpe=frozen_sh,
                                cagr_20bp_3pct=cagr20, random_percentiles={str(k): v for k, v in pct.items()}, random_n=nrand,
                                june_2026=june, top_pair_share=top_share, verdict=ind_verdict, overall_pass=ind_overall))
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(res, indent=2, default=str))
        print(f"\nwrote {OUT_PATH}")
    return res


def main():
    p = tp.CACHE_DIR / "etfs.parquet"
    if not p.exists():
        raise SystemExit("ETF cache not found. Run tools/run_alpaca_tsmom_paper.py once first.")
    close, feed = tp.load_close()
    prov = f"etfs.parquet md5={hashlib.md5(p.read_bytes()).hexdigest()} feed={feed} rows={len(close)} {close.index[0].date()}..{close.index[-1].date()}"
    run(close, prov)


if __name__ == "__main__":
    main()
