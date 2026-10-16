"""
KALMAN PAIRS TRADE on ETFs (test #6 in the test ledger).  EXECUTOR stage.
OFFLINE: reads only the cached ETF prices (runtime/alpaca_tsmom_etfs/etfs.parquet). No network, no API keys.
Writes runtime/alpaca_pairs_results.json and appends to runtime/test_ledger.json. Touches no live-bot file.

------------------------- PREDECLARED SPECIFICATION (fixed before any result) -------------------------
PAIRS (fixed now, y/x): GLD/SLV, IEF/TLT, LQD/HYG, SPY/QQQ, EFA/EEM, FXA/FXC, FXE/UUP, DBA/DBB.
DATA: cached Alpaca daily ETF closes (split+dividend adjusted), forward-filled; log prices.
KALMAN (per pair; v2 parameterization, see REVISION NOTE): burn-in = first 252 trading days of the pair.
  Centre log prices by their burn-in means (mx, my).  OLS of centred y on centred x over the burn-in gives
  theta_0 = (beta_0, alpha_0=0), s2 = residual variance (divisor n-2), C_0 = diag(s2/Sxx, s2/n), vt = s2.
  From the first post-burn-in day: y_t = beta_t * x_t + alpha_t, state theta=(beta,alpha) a random walk, state noise
  W = delta/(1-delta) * I with delta = 1e-6, R_t = C_{t-1} + W.  Prediction error e_t = y_t - F_t theta_{t-1|t-1},
  Q_t = F_t R_t F_t' + vt, signal z_t = e_t / sqrt(Q_t). Hedge ratio = posterior beta_t. Day t uses data up to day t only.
  REVISION NOTE: the first draft used theta_0 = 0, C_0 = 0, vt = 1e-3, delta = 1e-4 on raw log prices. A synthetic-data
  smoke test (no real result had been produced) showed that setting makes the filter absorb the spread, so |z| almost
  never exceeds 2 and the real test would "fail" for a trivial reason. It was replaced by the burn-in OLS initialisation
  and delta = 1e-6, chosen so that z has std about 1 on a stationary synthetic spread. Disclosed to the user before any run.
BURN-IN: no trading in the first 252 trading days of a pair.
RULES (signal at close of day t, executed at the close of day t+1):
  flat: z_t >  2.0 -> SHORT spread (short y, long beta*x);  z_t < -2.0 -> LONG spread (long y, short beta*x);
        skipped if |beta_t| > 5.
  long spread exits when z_t >= -0.5;  short spread exits when z_t <= +0.5;  forced exit when
  (t - entry execution day) >= 60.  No re-entry on the same signal day. Hedge ratio fixed at entry.
SIZING: 8 slots, one per pair. Slot capital S = E/8 (E = equity at the execution close, after that day's P&L and
  exit costs, before any of that day's entry costs; computed once per day). y-leg notional = side*S/(1+|beta|), x-leg notional = -side*beta*S/(1+|beta|) (gross = S).
  Shares fixed after entry (dollar exposure drifts). Idle slot = cash earning 0.
COSTS: 5 bp of traded notional (both legs, entry and exit); borrow 1.0%/yr on the prior close's short notional,
  charged daily; no interest on cash; no rebate on shorts.
EVALUATION: daily returns from the first possible execution day (index 253) to the end, including idle days.
CONTROLS: (C1) FROZEN HEDGE: OLS of y on x over the 252-day burn-in only; z = residual/burn-in residual std;
  same rules and costs.  (C3) RANDOM PAIRS: 500 random portfolios of 8 distinct pairs drawn from the 300
  unordered pairs of the 25 ETFs (seed 7), same rules and costs.
INFORMATION ONLY (no verdict): delta 1e-5 and 1e-7; costs 1.5/10/20 bp; borrow 0% and 3%; execution lagged one extra
  day; per-pair statistics.
BAR (stricter because this is test #6 on the same data; recorded in runtime/test_ledger.json):
VERDICT "PASS" only if ALL hold (headline: 5 bp, 1% borrow):
  (i)    net Sharpe (rf=0) >= 0.50;
  (ii)   moving-block bootstrap (6-month blocks, circular, 2000 resamples, seed 123) one-sided p < 0.01
         for mean monthly net return > 0;
  (iii)  CAGR > 0 at 20 bp and 3% borrow;
  (iv)   |daily correlation with SPY| <= 0.30;
  (v)    Sharpe >= 95th percentile of the 500 random-pair portfolios;
  (vi)   net return > 0 in BOTH halves of the evaluation window;
  (vii)  Sharpe > frozen-hedge control Sharpe;
  (viii) no single pair contributes more than half of total positive pair P&L.
June 2026 (2026-05-15 -> 2026-07-01) is reported on its own. No parameter changes after results.
Sample ~9 years, 8 hand-picked pairs: weak evidence either way.
-------------------------------------------------------------------------------------------------------

Run (project root, `javier` env):  python tools/run_pairs_test.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_alpaca_tsmom_paper as tp          # cached-data loader + metrics only

PROJECT_ROOT = tp.PROJECT_ROOT
OUT_PATH = PROJECT_ROOT / "runtime" / "alpaca_pairs_results.json"
LEDGER_PATH = PROJECT_ROOT / "runtime" / "test_ledger.json"

PAIRS = [("GLD", "SLV"), ("IEF", "TLT"), ("LQD", "HYG"), ("SPY", "QQQ"),
         ("EFA", "EEM"), ("FXA", "FXC"), ("FXE", "UUP"), ("DBA", "DBB")]
DELTA = 1e-6
BURN, ENTRY_Z, EXIT_Z, MAX_HOLD, BETA_CAP = 252, 2.0, 0.5, 60, 5.0
N_SLOTS = 8
COST_BP, BORROW = 5.0, 0.01
N_RANDOM, RANDOM_SEED = 500, 7
BOOT_N, BOOT_BLOCK, BOOT_SEED = 2000, 6, 123
JUNE_START, JUNE_END = tp.JUNE_START, tp.JUNE_END
BAR = dict(sharpe=0.50, p=0.01, corr=0.30, pct=95.0)

PRIOR_TESTS = [
    dict(id=1, name="crypto_lightgbm_multibot", outcome="REJECTED (Critic)"),
    dict(id=2, name="tsmom_etf_yfinance", outcome="FAIL"),
    dict(id=3, name="equity_momentum_alpaca", outcome="PASS (thin)"),
    dict(id=4, name="tsmom_paper_baseline_etf", outcome="FAIL"),
    dict(id=5, name="tsh_control_etf", outcome="FAIL"),
]


# ======================= Kalman + rules =======================

def kalman_z(ly: np.ndarray, lx: np.ndarray, i0: int, delta: float = DELTA):
    """Kalman filter (v2): OLS-initialised on the burn-in window [i0, i0+BURN), scalar algebra.
    Returns (z, beta_posterior), NaN before i0+BURN."""
    T = len(ly)
    z = np.full(T, np.nan)
    beta = np.full(T, np.nan)
    xs, ys = lx[i0:i0 + BURN], ly[i0:i0 + BURN]
    mx, my = xs.mean(), ys.mean()
    xc, yc = xs - mx, ys - my
    sxx = float((xc ** 2).sum())
    b = float((xc * yc).sum() / sxx)
    a = 0.0
    s2 = float(((yc - b * xc) ** 2).sum() / (BURN - 2))
    vt = s2
    w = delta / (1.0 - delta)
    c00, c01, c10, c11 = s2 / sxx, 0.0, 0.0, s2 / BURN
    for t in range(i0 + BURN, T):
        x, y = lx[t] - mx, ly[t] - my
        r00, r01, r10, r11 = c00 + w, c01, c10, c11 + w
        rf0 = r00 * x + r01
        rf1 = r10 * x + r11
        q = x * rf0 + rf1 + vt
        e = y - (b * x + a)
        z[t] = e / np.sqrt(q)
        a0, a1 = rf0 / q, rf1 / q
        b += a0 * e
        a += a1 * e
        fr0 = x * r00 + r10          # F R (row)
        fr1 = x * r01 + r11
        c00, c01 = r00 - a0 * fr0, r01 - a0 * fr1
        c10, c11 = r10 - a1 * fr0, r11 - a1 * fr1
        beta[t] = b
    return z, beta


def frozen_z(ly: np.ndarray, lx: np.ndarray, i0: int):
    """Control C1: OLS on burn-in window only; z = residual / burn-in residual std; constant beta."""
    T = len(ly)
    lo, hi = i0, i0 + BURN
    xs, ys = lx[lo:hi], ly[lo:hi]
    xm, ym = xs.mean(), ys.mean()
    b = float(((xs - xm) * (ys - ym)).sum() / ((xs - xm) ** 2).sum())
    a = float(ym - b * xm)
    sd = float((ys - (b * xs + a)).std(ddof=1))
    z = np.full(T, np.nan)
    z[i0:] = (ly[i0:] - (b * lx[i0:] + a)) / sd
    beta = np.full(T, np.nan)
    beta[i0:] = b
    return z, beta


def generate_trades(z: np.ndarray, beta: np.ndarray, i0: int):
    """State machine. A trade = (sig_t, exec_t, exit_exec_t or -1, side, beta)."""
    T = len(z)
    trades = []
    pos = 0
    sig = ex = 0
    side = 0
    bt = 0.0
    for t in range(i0 + BURN, T - 1):
        if pos == 0:
            if z[t] > ENTRY_Z:
                s = -1
            elif z[t] < -ENTRY_Z:
                s = 1
            else:
                continue
            if np.isnan(beta[t]) or abs(beta[t]) > BETA_CAP:
                continue
            pos, side, sig, ex, bt = s, s, t, t + 1, float(beta[t])
        else:
            hit = (pos == 1 and z[t] >= -EXIT_Z) or (pos == -1 and z[t] <= EXIT_Z) or (t - ex >= MAX_HOLD)
            if hit:
                trades.append((sig, ex, t + 1, side, bt))
                pos = 0
    if pos != 0:
        trades.append((sig, ex, -1, side, bt))
    return trades


# ======================= portfolio simulation =======================

def simulate(P: np.ndarray, pair_cols: list, pair_trades: list, cost_bp: float, borrow: float,
             start_idx: int, lag: int = 0):
    """pair_cols[k] = (iy, ix) column indices in P; pair_trades[k] = trades of pair k."""
    T = P.shape[0]
    K = len(pair_cols)
    entries: dict[int, list] = {}
    exits: dict[int, list] = {}
    for k, trs in enumerate(pair_trades):
        for (sg, ex, xe, side, b) in trs:
            e_t = ex + lag
            if e_t >= T:
                continue
            entries.setdefault(e_t, []).append((k, side, b, id_ := (k, sg)))
            if xe != -1 and xe + lag < T:
                exits.setdefault(xe + lag, []).append((k, id_))
    E = 1.0
    open_: dict = {}
    ret = np.zeros(T)
    contrib = np.zeros((T, K))
    pos_days = np.zeros(K)
    n_tr = np.zeros(K, dtype=int)
    for t in range(1, T):
        prev = E
        dc = np.zeros(K)
        for key, (k, sy, sx, iy, ix) in open_.items():
            pnl = sy * (P[t, iy] - P[t - 1, iy]) + sx * (P[t, ix] - P[t - 1, ix])
            short = (abs(sy) * P[t - 1, iy] if sy < 0 else 0.0) + (abs(sx) * P[t - 1, ix] if sx < 0 else 0.0)
            c = pnl - borrow / 252.0 * short
            dc[k] += c
            E += c
            if t >= start_idx:
                pos_days[k] += 1
        for (k, key) in exits.get(t, []):
            if key in open_:
                _, sy, sx, iy, ix = open_.pop(key)
                cst = cost_bp / 1e4 * (abs(sy) * P[t, iy] + abs(sx) * P[t, ix])
                E -= cst
                dc[k] -= cst
        S = E / N_SLOTS                      # slot capital fixed once per day: after P&L and exit costs, before any entry cost
        for (k, side, b, key) in entries.get(t, []):
            iy, ix = pair_cols[k]
            ny = side * S / (1.0 + abs(b))
            nx = -side * b * S / (1.0 + abs(b))
            sy, sx = ny / P[t, iy], nx / P[t, ix]
            cst = cost_bp / 1e4 * (abs(sy) * P[t, iy] + abs(sx) * P[t, ix])
            E -= cst
            dc[k] -= cst
            open_[key] = (k, sy, sx, iy, ix)
            n_tr[k] += 1
        if t >= start_idx:
            ret[t] = E / prev - 1.0
            contrib[t] = dc
    return ret[start_idx:], contrib[start_idx:], pos_days, n_tr


# ======================= statistics =======================

def monthly(r: pd.Series) -> np.ndarray:
    return ((1 + r).groupby([r.index.year, r.index.month]).prod() - 1).values


def block_boot_p(d: np.ndarray) -> float:
    rng = np.random.default_rng(BOOT_SEED)
    n = len(d)
    obs = d.mean()
    cen = d - obs
    nb = int(np.ceil(n / BOOT_BLOCK))
    cnt = 0
    for _ in range(BOOT_N):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(BOOT_BLOCK)[None, :]).ravel()[:n] % n
        if cen[idx].mean() >= obs:
            cnt += 1
    return (cnt + 1) / (BOOT_N + 1)


def sharpe_of(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(252)) if s > 0 else float("nan")


def update_ledger(outcome: str) -> list:
    led = json.loads(LEDGER_PATH.read_text()) if LEDGER_PATH.exists() else {"tests": [dict(t, source="recorded by Claude from session history; not machine-verified") for t in PRIOR_TESTS]}
    led["tests"] = [t for t in led["tests"] if t["id"] != 6]
    led["tests"].append(dict(id=6, name="pairs_kalman_etf", outcome=outcome, source="run_pairs_test.py"))
    LEDGER_PATH.write_text(json.dumps(led, indent=2))
    return led["tests"]


# ======================= main study =======================

def run_study(close: pd.DataFrame, feed: str = "n/a", write: bool = True, n_random: int = N_RANDOM,
              ledger: bool = True) -> dict:
    t_start = time.time()
    syms = [s for s in tp.ALL_SYMS if s in close.columns]
    c = close[syms].sort_index().ffill()
    P = c.values.astype(float)
    L = np.log(P)
    idx = c.index
    T = len(idx)
    col = {s: i for i, s in enumerate(syms)}
    i0s = {s: int(np.argmax(~np.isnan(P[:, col[s]]))) for s in syms}
    start_idx = BURN + 1 + max(i0s.values())
    print(f"\ndata: {len(syms)} ETFs, {idx[0].date()} -> {idx[-1].date()}, feed={feed}; evaluation from {idx[start_idx].date()} (index {start_idx})")
    print("REMINDER: 8 hand-picked pairs, ~9y sample, test #6 on this data; weak evidence either way.")

    def build(pairs, delta=DELTA, frozen=False):
        cols, trs, zs = [], [], []
        for (y, x) in pairs:
            i0 = max(i0s[y], i0s[x])
            z, b = frozen_z(L[:, col[y]], L[:, col[x]], i0) if frozen else kalman_z(L[:, col[y]], L[:, col[x]], i0, delta)
            cols.append((col[y], col[x]))
            trs.append(generate_trades(z, b, i0))
            zs.append((z, b))
        return cols, trs, zs

    cols, trs, zs = build(PAIRS)
    r, contrib, pos_days, n_tr = simulate(P, cols, trs, COST_BP, BORROW, start_idx)
    rs = pd.Series(r, index=idx[start_idx:])
    m = tp.base.metrics(rs)
    print(f"\n=== HEADLINE (5 bp, 1%/yr borrow, 8 pairs) ===\n  Kalman pairs     {tp.base.fmt(m)}")
    spy = pd.Series(np.nan_to_num(P[1:, col['SPY']] / P[:-1, col['SPY']] - 1.0), index=idx[1:]).reindex(rs.index)
    spy_m = tp.base.metrics(spy)
    print(f"  SPY buy&hold     {tp.base.fmt(spy_m)}")
    corr = float(rs.corr(spy))
    yrs = len(rs) / 252.0
    print(f"  daily correlation with SPY {corr:+.2f}; approx Sharpe SE {tp.sharpe_se(m['sharpe'], yrs):.2f} over {yrs:.1f}y; "
          f"{int(n_tr.sum())} trades")

    # controls
    cols_f, trs_f, _ = build(PAIRS, frozen=True)
    r_f = simulate(P, cols_f, trs_f, COST_BP, BORROW, start_idx)[0]
    m_f = tp.base.metrics(pd.Series(r_f, index=rs.index))
    print(f"  C1 frozen hedge  {tp.base.fmt(m_f)}")

    # random pairs
    print(f"\n  building {n_random} random 8-pair portfolios ...")
    rng = np.random.default_rng(RANDOM_SEED)
    all_pairs = [(syms[i], syms[j]) for i in range(len(syms)) for j in range(i + 1, len(syms))]
    cache: dict[int, tuple] = {}

    def pair_data(pi):
        if pi not in cache:
            y, x = all_pairs[pi]
            i0 = max(i0s[y], i0s[x])
            z, b = kalman_z(L[:, col[y]], L[:, col[x]], i0)
            cache[pi] = ((col[y], col[x]), generate_trades(z, b, i0))
        return cache[pi]

    rand_sh = []
    for _ in range(n_random):
        pick = rng.choice(len(all_pairs), size=N_SLOTS, replace=False)
        pd_ = [pair_data(int(pi)) for pi in pick]
        rr = simulate(P, [a for a, _ in pd_], [b for _, b in pd_], COST_BP, BORROW, start_idx)[0]
        rand_sh.append(sharpe_of(rr))
    rand_sh = np.array(rand_sh)
    pct = {q: float(np.nanpercentile(rand_sh, q)) for q in (5, 25, 50, 75, 95)}
    rank = float((rand_sh <= m['sharpe']).mean() * 100)
    print(f"  C3 random-pair Sharpe percentiles: " + ", ".join(f"p{q} {v:.2f}" for q, v in pct.items())
          + f"; chosen Sharpe {m['sharpe']:.2f} = percentile {rank:.1f}")

    # halves, years, june
    mid = len(rs) // 2
    halves = {"first half": rs.iloc[:mid], "second half": rs.iloc[mid:]}
    print("\n=== HALVES (information; condition vi uses total return) ===")
    half_tot = {}
    for k, v in halves.items():
        half_tot[k] = float((1 + v).prod() - 1)
        print(f"  {k:12s} {tp.base.fmt(tp.base.metrics(v))}  total {half_tot[k]*100:6.2f}%")
    yr = ((1 + rs).groupby(rs.index.year).prod() - 1) * 100
    yr_spy = ((1 + spy).groupby(spy.index.year).prod() - 1) * 100
    print("\n=== CALENDAR YEARS (total return %) ===\n  year   pairs     SPY")
    for y in yr.index:
        print(f"  {y}  {yr[y]:7.1f} {yr_spy[y]:7.1f}")
    june = tp.window_ret(rs, JUNE_START, JUNE_END)
    june_spy = tp.window_ret(spy, JUNE_START, JUNE_END)
    print(f"\n=== JUNE 2026 WINDOW ({JUNE_START.date()} -> {JUNE_END.date()}), on its own ===\n"
          f"  Kalman pairs {june*100:7.2f}%   SPY {june_spy*100:7.2f}%")

    # per pair
    tot_c = contrib.sum(0) * 100
    print("\n=== PER PAIR (net contribution, % of starting equity; trades; share of days in position) ===")
    per_pair = {}
    for k, (y, x) in enumerate(PAIRS):
        trl = trs[k]
        holds = [(xe - ex) for (_, ex, xe, _, _) in trl if xe != -1]
        per_pair[f"{y}/{x}"] = dict(contribution_pct=float(tot_c[k]), trades=int(n_tr[k]),
                                    pct_days_in_position=float(pos_days[k] / len(rs) * 100),
                                    avg_hold_days=float(np.mean(holds)) if holds else None,
                                    median_beta=float(np.nanmedian(zs[k][1][start_idx:])))
        pp = per_pair[f"{y}/{x}"]
        print(f"  {y}/{x:5s} {pp['contribution_pct']:7.2f}  trades {pp['trades']:4d}  in-pos {pp['pct_days_in_position']:5.1f}%  "
              f"avg hold {pp['avg_hold_days'] if pp['avg_hold_days'] is None else round(pp['avg_hold_days'],1)}  median beta {pp['median_beta']:.2f}")
    pos_sum = float(sum(v for v in tot_c if v > 0))
    top_share = float(max(tot_c) / pos_sum) if pos_sum > 0 else float("nan")
    print(f"  sum of pair contributions {tot_c.sum():.2f} (portfolio total return {((1+rs).prod()-1)*100:.2f}%); "
          f"top pair share of positive P&L {top_share*100:.1f}%")

    # bootstrap
    mm = monthly(rs)
    p_boot = block_boot_p(mm)
    print(f"\n  monthly net return mean {mm.mean()*100:.3f}% over {len(mm)} months; block-bootstrap one-sided p = {p_boot:.4f}")

    # sensitivities (information only)
    print("\n=== SENSITIVITY (information only) ===")
    sens = {}
    for cbp, bor in [(1.5, BORROW), (10.0, BORROW), (20.0, BORROW), (COST_BP, 0.0), (COST_BP, 0.03), (20.0, 0.03)]:
        rr = simulate(P, cols, trs, cbp, bor, start_idx)[0]
        sens[f"cost{cbp:g}_borrow{bor:g}"] = tp.base.metrics(pd.Series(rr, index=rs.index))
        print(f"  cost {cbp:5.1f} bp borrow {bor*100:3.1f}%  {tp.base.fmt(sens[f'cost{cbp:g}_borrow{bor:g}'])}")
    for dl in (1e-5, 1e-7):
        cd, td, _ = build(PAIRS, delta=dl)
        rr = simulate(P, cd, td, COST_BP, BORROW, start_idx)[0]
        sens[f"delta{dl:g}"] = tp.base.metrics(pd.Series(rr, index=rs.index))
        print(f"  delta {dl:g}          {tp.base.fmt(sens[f'delta{dl:g}'])}")
    rr = simulate(P, cols, trs, COST_BP, BORROW, start_idx, lag=1)[0]
    sens["lag1"] = tp.base.metrics(pd.Series(rr, index=rs.index))
    print(f"  execution +1 day   {tp.base.fmt(sens['lag1'])}")
    cagr20 = sens["cost20_borrow0.03"]["cagr"]

    v = {
        "i_sharpe_ge_0.50": bool(m["sharpe"] >= BAR["sharpe"]),
        "ii_bootstrap_p_lt_0.01": bool(p_boot < BAR["p"]),
        "iii_cagr_pos_at_20bp_3pct_borrow": bool(cagr20 > 0),
        "iv_abs_corr_spy_le_0.30": bool(abs(corr) <= BAR["corr"]),
        "v_sharpe_ge_p95_random_pairs": bool(m["sharpe"] >= pct[95]),
        "vi_positive_both_halves": bool(all(x > 0 for x in half_tot.values())),
        "vii_beats_frozen_hedge": bool(m["sharpe"] > m_f["sharpe"]),
        "viii_top_pair_share_le_50pct": bool(top_share <= 0.5) if top_share == top_share else False,
    }
    print("\n=== VERDICT (predeclared; headline 5 bp, 1% borrow; stricter bar for test #6) ===")
    print(f"  (i)    net Sharpe {m['sharpe']:.2f} (>= {BAR['sharpe']})  -> {v['i_sharpe_ge_0.50']}")
    print(f"  (ii)   bootstrap p {p_boot:.4f} (< {BAR['p']})  -> {v['ii_bootstrap_p_lt_0.01']}")
    print(f"  (iii)  CAGR at 20 bp / 3% borrow {cagr20*100:.2f}%  -> {v['iii_cagr_pos_at_20bp_3pct_borrow']}")
    print(f"  (iv)   |corr with SPY| {abs(corr):.2f} (<= {BAR['corr']})  -> {v['iv_abs_corr_spy_le_0.30']}")
    print(f"  (v)    Sharpe {m['sharpe']:.2f} vs random-pair p95 {pct[95]:.2f}  -> {v['v_sharpe_ge_p95_random_pairs']}")
    print(f"  (vi)   half totals {half_tot['first half']*100:.2f}% / {half_tot['second half']*100:.2f}%  -> {v['vi_positive_both_halves']}")
    print(f"  (vii)  Sharpe {m['sharpe']:.2f} vs frozen hedge {m_f['sharpe']:.2f}  -> {v['vii_beats_frozen_hedge']}")
    print(f"  (viii) top pair share {top_share*100:.1f}% (<= 50%)  -> {v['viii_top_pair_share_le_50pct']}")
    overall = all(v.values())
    outcome = "PASS" if overall else "FAIL"
    print(f"\n  OVERALL VERDICT: {outcome}  -- 8 hand-picked ETF pairs, ~{yrs:.0f}y sample, Sharpe SE ~{tp.sharpe_se(m['sharpe'], yrs):.2f}; "
          f"test #6 on this data. A PASS would be a lead, not a result.")
    tests = update_ledger(outcome) if (ledger and write) else PRIOR_TESTS + [dict(id=6, name="pairs_kalman_etf", outcome=outcome)]
    print(f"  test ledger: {len(tests)} predeclared tests recorded; Bonferroni-style reference alpha = 0.05/{len(tests)} = {0.05/len(tests):.4f}")

    res = dict(feed=feed, eval_start=str(idx[start_idx].date()), start_idx=start_idx, n_days=len(rs),
               overall=m, spy=spy_m, corr_spy=corr, sharpe_se=tp.sharpe_se(m["sharpe"], yrs),
               n_trades=int(n_tr.sum()), frozen=m_f, random_percentiles=pct, random_n=int(n_random),
               chosen_percentile_rank=rank, halves={k: dict(total=half_tot[k], **tp.base.metrics(vv)) for k, vv in halves.items()},
               calendar_years={str(y): dict(pairs=float(yr[y]), spy=float(yr_spy[y])) for y in yr.index},
               june_2026=dict(pairs=june, spy=june_spy), per_pair=per_pair, sum_pair_contrib_pct=float(tot_c.sum()),
               portfolio_total_return_pct=float(((1 + rs).prod() - 1) * 100), top_pair_share=top_share,
               monthly_mean=float(mm.mean()), n_months=len(mm), bootstrap_p=p_boot, sensitivity=sens,
               verdict=v, overall_pass=overall, test_ledger=tests, bar=BAR)
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(res, indent=2, default=str))
        print(f"\nwrote {OUT_PATH}")
    print(f"(elapsed {time.time()-t_start:.0f}s)")
    return res


def main():
    p = tp.CACHE_DIR / "etfs.parquet"
    if not p.exists():
        raise SystemExit("ETF cache not found. Run tools/run_alpaca_tsmom_paper.py once first.")
    close, feed = tp.load_close()
    run_study(close, feed)


if __name__ == "__main__":
    main()
