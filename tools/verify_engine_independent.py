"""
INDEPENDENT VERIFICATION of the ETF TSMOM/TSH engine (tools/run_alpaca_tsmom_paper.py, run_tsh_control_test.py).
OFFLINE. Reads only the cached ETF prices (runtime/alpaca_tsmom_etfs/etfs.parquet) and, if present, the saved
results JSONs. No network, no API keys, touches no live-bot file. Writes runtime/engine_verification_results.json.

PURPOSE: catch engine bugs that tests written by the same author with the same code cannot. Everything below is
re-implemented with different code paths (manual EWMA recursion, prefix-sum signal, share-count accounting
instead of weight drift, own metrics). Only the price data loader and the engine UNDER TEST are imported.

------------------------- PREDECLARED CHECKS AND TOLERANCES (fixed before any result) -------------------------
 1. SPEC DRIFT     : constants hard-coded here equal the constants the engine actually uses (exact).
 2. DATA INTEGRITY : dates unique+increasing, no non-positive prices, no gap > 5 business days inside any series.
 3. SIGNALS/WEIGHTS: at every rebalance, active sets equal and signal signs equal for TSM and TSH (100%);
                     target weights (TSM, TSH, always-long CONTROL, EW) agree, max abs diff <= 1e-9.
 4. DAILY RETURNS  : for TSM, TSH, CONTROL, EW, SPY, 60/40 the independent share-count simulation vs the engine:
                     max abs daily diff <= 1e-4, |CAGR diff| <= 0.05 pts, |Sharpe diff| <= 0.01,
                     |maxDD diff| <= 0.005. (Wider than 1e-9 because the engines differ in post-cost sizing
                     convention: positions sized on pre-cost vs post-cost equity, an effect of order cost x r.)
 5. METRICS        : independent metrics vs engine metrics on the SAME engine return series: CAGR, vol, Sharpe
                     <= 1e-9; maxDD <= 0.005 (the engine omits the starting equity 1.0 from the peak).
 6. SAVED RESULTS  : independent TSM / TSH headline Sharpe within 0.01 and CAGR within 0.05 pts of the saved JSONs
                     (skipped, and said so, if a file is missing).
 7. LOOK-AHEAD AUDIT: at 8 random rebalance dates (seed 2024), overwrite ALL prices after the signal date with random
                     values. Target weights at that date must stay identical (<= 1e-12) and no daily return on or
                     before that date may change (<= 1e-12). Run on BOTH the independent code and the engine.
VERDICT: "ENGINE VERIFIED" only if every check passes. Any failure is reported with its numbers.
A pass proves the code does what the written spec says; it does NOT prove the spec is a good strategy.
-----------------------------------------------------------------------------------------------------------------

Run (project root, `javier` env):  python tools/verify_engine_independent.py
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

import run_alpaca_tsmom_paper as tp          # engine under test + cached-data loader
import run_tsh_control_test as th            # engine under test (TSH targets, simulate_ls)

OUT_PATH = tp.PROJECT_ROOT / "runtime" / "engine_verification_results.json"

# ---- spec, re-declared here on purpose (check 1 compares against the engine's constants) ----
SYMS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "EWZ", "FXI",
        "IEF", "TLT", "TIP", "LQD", "HYG",
        "GLD", "SLV", "USO", "UNG", "DBA", "DBB",
        "UUP", "FXE", "FXY", "FXB", "FXA", "FXC"]
LOOKBACK, SIG_MIN, HALFLIFE, VOL_MIN = 252, 240, 252, 126
TARGET_VOL, GROSS_CAP, MIN_ACTIVE, COST_BP = 0.40, 6.0, 8, 5.0
HIST_MIN = 252
N_POISON = 8
TOL = dict(w=1e-9, ret=1e-4, cagr=0.0005, sharpe=0.01, dd=0.005, met=1e-9, saved_sh=0.01, saved_cagr=0.0005, poison=1e-12)


# ======================= independent implementation =======================

def ewma_ann_vol(r: np.ndarray) -> np.ndarray:
    """sqrt(252 * EWMA(r^2)), halflife 252, zero-mean, valid once >= 126 observations. Manual recursion."""
    alpha = 1.0 - np.exp(np.log(0.5) / HALFLIFE)
    decay = 1.0 - alpha
    num = den = 0.0
    cnt = 0
    out = np.full(len(r), np.nan)
    for t in range(len(r)):
        num *= decay
        den *= decay
        if not np.isnan(r[t]):
            num += r[t] ** 2
            den += 1.0
            cnt += 1
        if cnt >= VOL_MIN:
            out[t] = np.sqrt(252.0 * num / den)
    return out


def month_ends(idx: pd.DatetimeIndex) -> list[int]:
    ym = idx.year.values * 100 + idx.month.values
    return [t for t in range(len(idx)) if t == len(idx) - 1 or ym[t] != ym[t + 1]]


def prep(P: np.ndarray):
    """returns, EWMA vol, TSM signal S, expanding-mean history, from a (T,M) forward-filled price array."""
    T, M = P.shape
    R = np.full((T, M), np.nan)
    R[1:] = P[1:] / P[:-1] - 1.0
    SIG = np.column_stack([ewma_ann_vol(R[:, m]) for m in range(M)])
    Z = np.full((T, M), np.nan)
    Z[1:] = R[1:] / SIG[:-1]
    valid = ~np.isnan(Z)
    Zc = np.where(valid, Z, 0.0)
    cs = np.vstack([np.zeros((1, M)), np.cumsum(Zc, axis=0)])
    cc = np.vstack([np.zeros((1, M)), np.cumsum(valid, axis=0)])
    S = np.full((T, M), np.nan)
    for t in range(T):
        lo = max(0, t - LOOKBACK + 1)
        n_valid = cc[t + 1] - cc[lo]
        s = cs[t + 1] - cs[lo]
        S[t] = np.where(n_valid >= SIG_MIN, s, np.nan)
    rv = ~np.isnan(R)
    csr = np.cumsum(np.where(rv, R, 0.0), axis=0)
    ccr = np.cumsum(rv, axis=0)
    HM = np.where(ccr >= HIST_MIN, csr / np.maximum(ccr, 1), np.nan)
    return R, SIG, S, HM


def make_targets(P: np.ndarray, idx: pd.DatetimeIndex, kind: str, prepped=None):
    R, SIG, S, HM = prepped if prepped is not None else prep(P)
    T, M = P.shape
    me = month_ends(idx)
    out, active_log, sign_log = {}, {}, {}
    for j in me:
        e = j + 1
        if e >= T:
            continue
        act = ~np.isnan(S[j]) & ~np.isnan(SIG[j]) & (SIG[j] > 0)
        na = int(act.sum())
        if na < MIN_ACTIVE:
            continue
        w = np.zeros(M)
        if kind in ("tsm", "tsh", "control"):
            sgn = {"tsm": np.sign(S[j]), "tsh": np.sign(HM[j]), "control": np.ones(M)}[kind]
            w[act] = np.nan_to_num(sgn[act]) * (TARGET_VOL / SIG[j][act]) / na
            g = np.abs(w).sum()
            if g > GROSS_CAP:
                w *= GROSS_CAP / g
            active_log[e] = act.copy()
            sign_log[e] = np.sign(S[j]) if kind == "tsm" else np.sign(HM[j])
        elif kind == "ew":
            w[act] = 1.0 / na
        elif kind == "spy":
            w[SYMS.index("SPY")] = 1.0
        elif kind == "6040":
            w[SYMS.index("SPY")], w[SYMS.index("IEF")] = 0.6, 0.4
        out[e] = w
    return out, active_log, sign_log


def indep_sim(P: np.ndarray, targets: dict, cost_bp: float, first: int) -> np.ndarray:
    """Share-count accounting: dollars of P&L from share holdings; cost on traded notional."""
    T, M = P.shape
    Pz = np.nan_to_num(P)
    shares = np.zeros(M)
    E = 1.0
    out = np.full(T, np.nan)
    for t in range(1, T):
        if t < first:
            continue
        prev = E
        ok = ~np.isnan(P[t]) & ~np.isnan(P[t - 1])
        E += float(np.sum(shares[ok] * (P[t][ok] - P[t - 1][ok])))
        if t in targets:
            wt = targets[t]
            new = np.zeros(M)
            nz = wt != 0
            new[nz] = wt[nz] * E / P[t][nz]
            E -= cost_bp / 1e4 * float(np.sum(np.abs(new - shares) * Pz[t]))
            shares = new
        out[t] = E / prev - 1.0
    return out


def indep_metrics(r: np.ndarray) -> dict:
    r = r[~np.isnan(r)]
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return dict(cagr=float(eq[-1] ** (252.0 / len(r)) - 1.0), vol=float(r.std(ddof=1) * np.sqrt(252)),
                sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(252)), max_dd=float((eq / peak - 1.0).min()))


# ======================= verification =======================

def run(close: pd.DataFrame, provenance: str = "n/a", write: bool = True) -> dict:
    results: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: str):
        results[name] = dict(passed=bool(ok), detail=detail)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(f"data provenance: {provenance}")
    print("\n=== 1. SPEC DRIFT (hard-coded here vs constants the engine uses) ===")
    drift = []
    for nm, mine, eng in [("LOOKBACK", LOOKBACK, tp.LOOKBACK), ("SIG_MIN_TERMS", SIG_MIN, tp.SIG_MIN_TERMS),
                          ("VOL_HALFLIFE", HALFLIFE, tp.VOL_HALFLIFE), ("VOL_MIN", VOL_MIN, tp.VOL_MIN),
                          ("TARGET_VOL", TARGET_VOL, tp.TARGET_VOL), ("GROSS_CAP", GROSS_CAP, tp.GROSS_CAP),
                          ("MIN_ACTIVE", MIN_ACTIVE, tp.MIN_ACTIVE), ("COST", COST_BP, tp.COST_HEADLINE),
                          ("SYMBOL LIST", SYMS, tp.ALL_SYMS)]:
        if mine != eng:
            drift.append(f"{nm}: here {mine} vs engine {eng}")
    record("spec_drift", not drift, "none" if not drift else "; ".join(drift))

    ts = tp.TS(close)
    if ts.syms != SYMS:
        raise SystemExit(f"engine symbol order differs from the declared list (missing: {ts.missing})")
    c = close[SYMS].sort_index()

    print("\n=== 2. DATA INTEGRITY ===")
    idx = c.index
    problems = []
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        problems.append("dates not strictly increasing/unique")
    if (c.values[~np.isnan(c.values)] <= 0).any():
        problems.append("non-positive price")
    bd = pd.Series(idx).diff().dt.days
    big_gap = int((bd > 7).sum())
    if big_gap:
        problems.append(f"{big_gap} calendar gaps > 7 days between consecutive dates")
    for s in SYMS:
        v = c[s].dropna()
        if len(v) and v.index.to_series().diff().dt.days.max() > 7:
            problems.append(f"{s}: internal gap > 7 days")
    record("data_integrity", not problems, "clean" if not problems else "; ".join(problems))

    P = c.ffill().values.astype(float)
    pre = prep(P)

    print("\n=== 3. SIGNALS and WEIGHTS: independent vs engine ===")
    t_eng = {"tsm": ts.targets("tsmom"), "control": ts.targets("control"), "ew": ts.targets("ew"),
             "spy": ts.targets("spy"), "6040": ts.targets("6040")}
    hist = ts.r.expanding(min_periods=tp.LOOKBACK).mean().values
    t_eng["tsh"] = th.sign_targets(ts, np.sign(hist))
    t_ind = {}
    act_ind = {}
    sgn_ind = {}
    for k in ("tsm", "tsh", "control", "ew", "spy", "6040"):
        t_ind[k], act_ind[k], sgn_ind[k] = make_targets(P, idx, k, pre)
    same_dates = all(set(t_eng[k]) == set(t_ind[k]) for k in t_ind)
    record("rebalance_dates_equal", same_dates, f"{len(t_ind['tsm'])} rebalances (engine {len(t_eng['tsm'])})")
    wdiff = {}
    for k in t_ind:
        common = sorted(set(t_eng[k]) & set(t_ind[k]))
        wdiff[k] = max((float(np.abs(t_eng[k][e] - t_ind[k][e]).max()) for e in common), default=float("nan"))
    record("target_weights", all(v <= TOL["w"] for v in wdiff.values()),
           "max abs diff " + ", ".join(f"{k} {v:.2e}" for k, v in wdiff.items()))
    # sign/active agreement for TSM and TSH
    mism = 0
    tot = 0
    for e, act in act_ind["tsm"].items():
        j = e - 1
        eng_act = ts.active_mask(j)
        mism += int((eng_act != act).sum())
        tot += len(act)
        s_eng = np.sign(ts.Sv[j])
        mism += int(((s_eng != sgn_ind["tsm"][e]) & act & ~(np.isnan(s_eng) & np.isnan(sgn_ind["tsm"][e]))).sum())
    for e, act in act_ind["tsh"].items():
        j = e - 1
        h_eng = np.sign(hist[j])
        both_nan = np.isnan(h_eng) & np.isnan(sgn_ind["tsh"][e])
        mism += int(((h_eng != sgn_ind["tsh"][e]) & act & ~both_nan).sum())
    record("active_sets_and_signs", mism == 0, f"{mism} disagreements over {tot} asset-rebalance slots (TSM) plus TSH signs")

    print("\n=== 4. DAILY RETURNS: independent share-count simulation vs engine ===")
    e0 = min(t_eng["tsm"])
    eng_ret = {
        "tsm": ts.simulate(t_eng["tsm"], COST_BP)[0],
        "tsh": ts.simulate(t_eng["tsh"], COST_BP)[0],
        "control": ts.simulate(t_eng["control"], COST_BP, start_e=e0)[0],
        "ew": ts.simulate(t_eng["ew"], COST_BP, start_e=e0)[0],
        "spy": ts.simulate({e0: t_eng["spy"][e0]}, COST_BP, start_e=e0)[0],
        "6040": ts.simulate(t_eng["6040"], COST_BP, start_e=e0)[0],
    }
    ind_ret = {}
    for k in eng_ret:
        tg = {e0: t_ind["spy"][e0]} if k == "spy" else t_ind[k]
        ind_ret[k] = indep_sim(P, tg, COST_BP, e0)
    worst = {}
    fails = []
    for k in eng_ret:
        a = eng_ret[k].values
        b = ind_ret[k][e0:]
        d = float(np.nanmax(np.abs(a - b)))
        ma, mb = tp.base.metrics(eng_ret[k]), indep_metrics(b)
        dc, ds, dd = abs(ma["cagr"] - mb["cagr"]), abs(ma["sharpe"] - mb["sharpe"]), abs(ma["max_dd"] - mb["max_dd"])
        worst[k] = (d, dc, ds, dd)
        if d > TOL["ret"] or dc > TOL["cagr"] or ds > TOL["sharpe"] or dd > TOL["dd"]:
            fails.append(k)
        print(f"    {k:8s} max|dr|={d:.2e}  dCAGR={dc*100:.4f}pts  dSharpe={ds:.4f}  dMaxDD={dd:.4f}   "
              f"(indep sharpe {mb['sharpe']:.3f} cagr {mb['cagr']*100:.2f}% | engine sharpe {ma['sharpe']:.3f} cagr {ma['cagr']*100:.2f}%)")
    record("daily_returns", not fails, "all six series within tolerance" if not fails else f"out of tolerance: {fails}")

    print("\n=== 5. METRICS: independent vs engine metric function on the SAME series ===")
    bad = []
    for k, r in eng_ret.items():
        ma, mi = tp.base.metrics(r), indep_metrics(r.values)
        for key in ("cagr", "vol", "sharpe"):
            if abs(ma[key] - mi[key]) > TOL["met"]:
                bad.append(f"{k}.{key}")
        if abs(ma["max_dd"] - mi["max_dd"]) > TOL["dd"]:
            bad.append(f"{k}.max_dd")
    record("metrics", not bad, "agree" if not bad else f"disagree: {bad}")

    print("\n=== 6. SAVED RESULTS (your earlier terminal runs, as written to JSON) ===")
    saved_ok, saved_msgs = True, []
    for label, path, key, ik in [("TSMOM paper run", tp.OUT_PATH, "TSMOM paper baseline (5bp)", "tsm"),
                                 ("TSH control run", th.OUT_PATH, "TSH (hist. mean sign)", "tsh")]:
        if not path.exists():
            saved_msgs.append(f"{label}: file missing, skipped")
            continue
        try:
            sv = json.loads(path.read_text())["overall"][key]
            mi = indep_metrics(ind_ret[ik][e0:])
            okk = abs(sv["sharpe"] - mi["sharpe"]) <= TOL["saved_sh"] and abs(sv["cagr"] - mi["cagr"]) <= TOL["saved_cagr"]
            saved_ok &= okk
            saved_msgs.append(f"{label}: saved sharpe {sv['sharpe']:.3f}/cagr {sv['cagr']*100:.2f}% vs independent "
                              f"{mi['sharpe']:.3f}/{mi['cagr']*100:.2f}% -> {'ok' if okk else 'MISMATCH'}")
        except Exception as exc:  # noqa: BLE001
            saved_ok = False
            saved_msgs.append(f"{label}: could not read ({exc})")
    record("saved_results", saved_ok, " | ".join(saved_msgs))

    print("\n=== 7. LOOK-AHEAD AUDIT (poison every price after the signal date) ===")
    rng = np.random.default_rng(2024)
    cand = sorted(t_ind["tsm"])
    cand = cand[6:-1]
    picks = sorted(rng.choice(len(cand), size=min(N_POISON, len(cand)), replace=False))
    leaks = []
    for pi in picks:
        e = cand[pi]
        j = e - 1
        Q = P.copy()
        Q[j + 1:] = rng.uniform(20.0, 200.0, size=Q[j + 1:].shape)
        # independent
        tq, _, _ = make_targets(Q, idx, "tsm")
        dw = float(np.abs(tq[e] - t_ind["tsm"][e]).max())
        rq = indep_sim(Q, tq, COST_BP, e0)
        dr = float(np.nanmax(np.abs(rq[:j + 1] - ind_ret["tsm"][:j + 1])))
        # engine under test
        cq = pd.DataFrame(Q, index=idx, columns=SYMS)
        tsq = tp.TS(cq)
        tqe = tsq.targets("tsmom")
        dwe = float(np.abs(tqe[e] - t_eng["tsm"][e]).max())
        rqe = tsq.simulate(tqe, COST_BP)[0]
        rcl = eng_ret["tsm"]
        common = rcl.index[rcl.index <= idx[j]]
        dre = float(np.nanmax(np.abs(rqe.reindex(common).values - rcl.reindex(common).values)))
        bad_here = [n for n, v in (("indep weights", dw), ("indep returns", dr), ("engine weights", dwe), ("engine returns", dre)) if v > TOL["poison"]]
        print(f"    signal date {idx[j].date()}: dW indep {dw:.1e} engine {dwe:.1e} | dRet(<=signal date) indep {dr:.1e} engine {dre:.1e}")
        if bad_here:
            leaks.append(f"{idx[j].date()}: {bad_here}")
    record("lookahead_audit", not leaks, f"{len(picks)} dates, no change" if not leaks else "; ".join(leaks))

    all_ok = all(v["passed"] for v in results.values())
    print(f"\n  OVERALL: {'ENGINE VERIFIED' if all_ok else 'VERIFICATION FAILED - see failing checks above'}")
    print("  Meaning: the engine does what its written spec says. It does not say the spec is a good strategy,")
    print("  and it does not check the cached price data against an outside source.")
    res = dict(provenance=provenance, checks=results, all_passed=all_ok, tolerances=TOL)
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
