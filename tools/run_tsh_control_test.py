"""
TSH control test (after Huang, Li, Wang, Zhou, "Time Series Momentum: Is it There?", JFE 2020).
OFFLINE research script. Uses ONLY the ETF price cache already on disk from
tools/run_alpaca_tsmom_paper.py (runtime/alpaca_tsmom_etfs/etfs.parquet): no network, no API keys,
no file written outside runtime/alpaca_tsh_control_results.json. Touches no live-bot file.

QUESTION: does the 12-month trend sign (TSM) add anything over the sign of each ETF's own historical
mean return (TSH, a non-predictive control), when both use identical sizing, universe, dates and costs?

------------------------- PREDECLARED SPECIFICATION (fixed before any result) -------------------------
DATA/ENGINE: exactly the cached 25-ETF panel and engine of run_alpaca_tsmom_paper.py (imported, not copied):
  EWMA vol (halflife 252d), sizing w_i = sign_i * (40%/sigma_i)/N_active, gross cap 6, month-end signal,
  next-day-close execution, 5 bp per unit turnover, no financing (headline), same first trade date, same
  active set at every rebalance (active = valid 252d TSM signal).
TSM : sign of the 252-day sum of vol-scaled returns (the previous test's baseline; reproduction is checked
      against the saved results JSON).
TSH : sign of the expanding mean of daily returns from each ETF's first bar to the signal date.
TESTS (headline 5 bp): net Sharpe of each; TSM-minus-TSH daily-return difference (Sharpe, annualised mean);
  moving-block bootstrap of the MONTHLY difference (block 6 months, circular, 2000 resamples, seed 123;
  null imposed by centring; one-sided p for mean > 0); per-ETF gross P&L, TSM vs TSH (count of ETFs where
  TSM is higher); long-leg vs short-leg gross P&L for each; signal agreement rate; halves, calendar years;
  cost sensitivity 1.5/5/10/20 bp and flat 3%/yr financing on exposure above 1x (information only).
COMPARATORS: always-long control with the same sizing, equal-weight long-only, SPY, 60/40.
VERDICT "the 12-month signal adds value over the historical-mean control" requires ALL of:
  (a) TSM net Sharpe > TSH net Sharpe;
  (b) block-bootstrap one-sided p < 0.10 for mean monthly (TSM - TSH) > 0;
  (c) TSM gross P&L > TSH gross P&L in more than half of the ETFs;
  (d) mean monthly (TSM - TSH) > 0 in BOTH halves of the sample.
Otherwise the verdict is FAIL (the signal is not shown to beat its control).
June 2026 window (2026-05-15 -> 2026-07-01) is reported on its own for every series.
~9 years of data and a 25-ETF proxy universe: weak evidence either way. No parameter changes after results.
-------------------------------------------------------------------------------------------------------

Run (project root, `javier` env):  python tools/run_tsh_control_test.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_alpaca_tsmom_paper as tp

base = tp.base
OUT_PATH = tp.PROJECT_ROOT / "runtime" / "alpaca_tsh_control_results.json"
BOOT_N, BOOT_BLOCK, BOOT_SEED = 2000, 6, 123
COST = tp.COST_HEADLINE


def load_cached() -> tuple[pd.DataFrame, str]:
    p = tp.CACHE_DIR / "etfs.parquet"
    if not p.exists():
        raise SystemExit("ETF cache not found. Run tools/run_alpaca_tsmom_paper.py once first (it downloads and caches the bars).")
    return tp.load_close()


def sign_targets(ts: "tp.TS", sign_rows: np.ndarray) -> dict[int, np.ndarray]:
    """Same sizing/active-set/cap as TS.targets('tsmom'), with the direction supplied by sign_rows[j]."""
    out: dict[int, np.ndarray] = {}
    m, n = len(ts.syms), len(ts.dates)
    for j in ts.me:
        e = j + 1
        if e >= n:
            continue
        act = ts.active_mask(j)
        na = int(act.sum())
        if na < tp.MIN_ACTIVE:
            continue
        w = np.zeros(m)
        sgn = sign_rows[j]
        w[act] = np.nan_to_num(sgn[act]) * (tp.TARGET_VOL / ts.sg[j][act]) / na
        g = np.abs(w).sum()
        if g > tp.GROSS_CAP:
            w *= tp.GROSS_CAP / g
        out[e] = w
    return out


def simulate_ls(ts: "tp.TS", targets: dict, cost_bp: float, fin_rate: float = 0.0, start_e: int | None = None):
    """Same as TS.simulate, additionally splitting daily contributions into long and short legs."""
    n, m = ts.R.shape
    w = np.zeros(m)
    net = np.full(n, np.nan)
    contrib = np.zeros((n, m))
    c_long = np.zeros((n, m))
    c_short = np.zeros((n, m))
    turns = []
    first = start_e if start_e is not None else min(targets)
    for t in range(1, n):
        c = w * ts.R[t]
        pr = c.sum()
        gross = np.abs(w).sum()
        rg = pr - fin_rate / 252.0 * max(gross - 1.0, 0.0)
        cl = np.where(w > 0, c, 0.0)
        cs = np.where(w < 0, c, 0.0)
        w = w * (1.0 + ts.R[t]) / (1.0 + rg)
        cost = 0.0
        if t in targets and t >= first:
            wt = targets[t]
            turn = float(np.abs(wt - w).sum())
            cost = cost_bp / 1e4 * turn
            w = wt.copy()
            turns.append(turn)
        if t >= first:
            net[t] = (1.0 + rg) * (1.0 - cost) - 1.0
            contrib[t], c_long[t], c_short[t] = c, cl, cs
    s = pd.Series(net, index=ts.dates)
    return s.iloc[first:], contrib[first:], c_long[first:], c_short[first:], turns


def monthly(r: pd.Series) -> pd.Series:
    r = r.dropna()
    return (1 + r).groupby([r.index.year, r.index.month]).prod() - 1


def block_boot_p(d: np.ndarray, n_boot: int = BOOT_N, block: int = BOOT_BLOCK, seed: int = BOOT_SEED) -> float:
    rng = np.random.default_rng(seed)
    n = len(d)
    obs = d.mean()
    cen = d - obs
    nb = int(np.ceil(n / block))
    cnt = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        if cen[idx].mean() >= obs:
            cnt += 1
    return (cnt + 1) / (n_boot + 1)


def run_study(close: pd.DataFrame, feed: str = "n/a", write: bool = True) -> dict:
    ts = tp.TS(close)
    print(f"\ndata: {len(ts.syms)}/{len(tp.ALL_SYMS)} ETFs, {ts.dates[0].date()} -> {ts.dates[-1].date()}, feed={feed} (cached, no download)")
    if ts.missing:
        print(f"MISSING symbols: {ts.missing}")
    t_tsm = ts.targets("tsmom")
    if not t_tsm:
        raise SystemExit("no valid rebalance dates")
    e0 = min(t_tsm)
    hist_mean = ts.r.expanding(min_periods=tp.LOOKBACK).mean().values
    t_tsh = sign_targets(ts, np.sign(hist_mean))
    assert set(t_tsh) == set(t_tsm), "TSM/TSH rebalance sets differ"
    print(f"first trade date {ts.dates[e0].date()}; REMINDER: ETF proxy universe, ~9y sample, weak evidence either way.")

    # signal agreement and long fraction (at rebalances, active assets)
    agree, tot, long_tsm, long_tsh = 0, 0, 0, 0
    for j in ts.me:
        if j + 1 not in t_tsm:
            continue
        act = ts.active_mask(j)
        s1, s2 = np.sign(ts.Sv[j][act]), np.sign(hist_mean[j][act])
        agree += int((s1 == s2).sum()); tot += int(act.sum())
        long_tsm += int((s1 > 0).sum()); long_tsh += int((s2 > 0).sum())
    print(f"signal agreement TSM vs TSH: {agree/tot*100:.1f}% of asset-months; long share TSM {long_tsm/tot*100:.1f}%, TSH {long_tsh/tot*100:.1f}%")

    r_tsm, c_tsm, cl_tsm, cs_tsm, tn_tsm = simulate_ls(ts, t_tsm, COST)
    r_tsh, c_tsh, cl_tsh, cs_tsh, tn_tsh = simulate_ls(ts, t_tsh, COST)

    # integrity: our simulate_ls must equal the previous engine; TSM must reproduce the saved previous result
    chk = ts.simulate(t_tsm, COST)[0]
    assert np.allclose(chk.values, r_tsm.values, atol=1e-12), "simulate_ls differs from the base engine"
    m_tsm = base.metrics(r_tsm)
    if tp.OUT_PATH.exists():
        try:
            prev = json.loads(tp.OUT_PATH.read_text())["overall"]["TSMOM paper baseline (5bp)"]
            ok = abs(prev["cagr"] - m_tsm["cagr"]) < 1e-4 and abs(prev["sharpe"] - m_tsm["sharpe"]) < 1e-4
            print(f"reproduction check vs saved TSMOM run: {'OK' if ok else 'MISMATCH (investigate before trusting)'} "
                  f"(saved sharpe {prev['sharpe']:.3f} / cagr {prev['cagr']*100:.2f}%, now {m_tsm['sharpe']:.3f} / {m_tsm['cagr']*100:.2f}%)")
        except Exception as exc:  # noqa: BLE001
            print(f"reproduction check skipped ({exc})")
    else:
        print("reproduction check skipped (no saved previous results file)")

    series = {
        "TSM (12m trend sign)": r_tsm,
        "TSH (hist. mean sign)": r_tsh,
        "CONTROL always long": ts.simulate(ts.targets("control"), COST, start_e=e0)[0],
        "EW long-only": ts.simulate(ts.targets("ew"), COST, start_e=e0)[0],
        "SPY buy&hold": ts.simulate({e0: ts.targets("spy")[e0]}, COST, start_e=e0)[0],
        "60/40 (SPY/IEF)": ts.simulate(ts.targets("6040"), COST, start_e=e0)[0],
    }
    names = list(series)
    print(f"\n=== OVERALL ({COST:g} bp per unit turnover, no financing) ===")
    allm = {}
    for nme in names:
        allm[nme] = base.metrics(series[nme])
        print(f"  {nme:26s} {base.fmt(allm[nme])}")
    yrs = len(r_tsm) / 252.0
    diff = r_tsm - r_tsh
    print(f"  turnover/rebalance (x equity): TSM {np.mean(tn_tsm):.2f}, TSH {np.mean(tn_tsh):.2f}")
    dm = base.metrics(diff)
    print(f"  TSM - TSH daily return difference: annualised mean {diff.mean()*252*100:.2f}%, vol {diff.std()*np.sqrt(252)*100:.2f}%, "
          f"Sharpe {dm['sharpe']:.2f} (SE ~{tp.sharpe_se(dm['sharpe'], yrs):.2f})")

    mm_tsm, mm_tsh = monthly(r_tsm), monthly(r_tsh)
    d_m = (mm_tsm - mm_tsh).values
    p_boot = block_boot_p(d_m)
    print(f"  monthly TSM-TSH: mean {d_m.mean()*100:.3f}%/month over {len(d_m)} months, "
          f"block-bootstrap one-sided p = {p_boot:.3f}")

    mid = r_tsm.index[len(r_tsm) // 2]
    print(f"\n=== FIRST vs SECOND HALF (split {mid.date()}) ===")
    halves = {}
    for lab, mask in (("first half", r_tsm.index < mid), ("second half", r_tsm.index >= mid)):
        halves[lab] = {}
        print(f"  --- {lab} ---")
        for nme in names[:2]:
            mt = base.metrics(series[nme][mask])
            halves[lab][nme] = mt
            print(f"    {nme:26s} {base.fmt(mt)}")
        mdiff = (diff[mask]).mean() * 21 * 100
        halves[lab]["mean_monthly_diff_pct"] = float(mdiff)
        print(f"    mean TSM-TSH difference: {mdiff:.3f}% per month (daily mean x 21)")

    print("\n=== CALENDAR YEARS (total return %) ===")
    yr = pd.DataFrame({n: (1 + series[n].dropna()).groupby(series[n].dropna().index.year).prod() - 1 for n in names}) * 100
    print("  year  " + "".join(f"{n[:18]:>20s}" for n in names))
    for y, row in yr.iterrows():
        print(f"  {y}  " + "".join(f"{v:20.1f}" for v in row.values))

    print(f"\n=== JUNE 2026 WINDOW ({tp.JUNE_START.date()} -> {tp.JUNE_END.date()}), on its own ===")
    june = {}
    for nme in names:
        june[nme] = tp.window_ret(series[nme], tp.JUNE_START, tp.JUNE_END)
        print(f"  {nme:26s} {june[nme]*100:7.2f}%")

    print("\n=== LONG vs SHORT LEG (gross of costs; sum of daily contributions, % of starting equity) ===")
    legs = {"TSM": (cl_tsm.sum() * 100, cs_tsm.sum() * 100), "TSH": (cl_tsh.sum() * 100, cs_tsh.sum() * 100)}
    for k, (lo, sh_) in legs.items():
        print(f"  {k}: long leg {lo:7.1f}   short leg {sh_:7.1f}   total {lo+sh_:7.1f}")

    per_tsm = pd.Series(c_tsm.sum(0) * 100, index=ts.syms)
    per_tsh = pd.Series(c_tsh.sum(0) * 100, index=ts.syms)
    wins = int((per_tsm > per_tsh).sum())
    print(f"\n=== PER-ETF gross P&L: TSM beats TSH in {wins} of {len(ts.syms)} ETFs ===")
    by_cls = {}
    for c, syms in tp.CLASSES.items():
        ss = [s for s in syms if s in ts.syms]
        by_cls[c] = (float(per_tsm[ss].sum()), float(per_tsh[ss].sum()), int((per_tsm[ss] > per_tsh[ss]).sum()), len(ss))
        a, b, w_, n_ = by_cls[c]
        print(f"  {c:10s} TSM {a:7.1f}  TSH {b:7.1f}  TSM higher in {w_}/{n_}")
    gd = (per_tsm - per_tsh).sort_values()
    print("  largest TSM shortfalls vs TSH: " + ", ".join(f"{s} {v:+.1f}" for s, v in gd.head(4).items()))
    print("  largest TSM gains vs TSH:      " + ", ".join(f"{s} {v:+.1f}" for s, v in gd.tail(4)[::-1].items()))

    print("\n=== SENSITIVITY (information only) ===")
    sens = {}
    for cbp in tp.COSTS_SENS:
        a = base.metrics(ts.simulate(t_tsm, cbp)[0]); b = base.metrics(ts.simulate(t_tsh, cbp)[0])
        sens[f"cost_{cbp:g}"] = (a, b)
        print(f"  @{cbp:5.1f} bp  TSM sharpe {a['sharpe']:5.2f} cagr {a['cagr']*100:6.2f}% | TSH sharpe {b['sharpe']:5.2f} cagr {b['cagr']*100:6.2f}%")
    a = base.metrics(ts.simulate(t_tsm, COST, fin_rate=tp.FIN_SENS)[0]); b = base.metrics(ts.simulate(t_tsh, COST, fin_rate=tp.FIN_SENS)[0])
    sens["financing_3pct"] = (a, b)
    print(f"  5bp + 3%/yr fin on >1x  TSM sharpe {a['sharpe']:5.2f} cagr {a['cagr']*100:6.2f}% | TSH sharpe {b['sharpe']:5.2f} cagr {b['cagr']*100:6.2f}%")

    v = {
        "a_tsm_sharpe_gt_tsh": bool(allm["TSM (12m trend sign)"]["sharpe"] > allm["TSH (hist. mean sign)"]["sharpe"]),
        "b_bootstrap_p_lt_0.10": bool(p_boot < 0.10),
        "c_tsm_beats_tsh_in_majority_of_etfs": bool(wins > len(ts.syms) / 2),
        "d_positive_diff_in_both_halves": bool(halves["first half"]["mean_monthly_diff_pct"] > 0 and halves["second half"]["mean_monthly_diff_pct"] > 0),
    }
    print("\n=== VERDICT (predeclared; headline 5 bp, no financing) ===")
    print(f"  (a) TSM Sharpe {allm['TSM (12m trend sign)']['sharpe']:.2f} vs TSH {allm['TSH (hist. mean sign)']['sharpe']:.2f}  -> {v['a_tsm_sharpe_gt_tsh']}")
    print(f"  (b) block-bootstrap p {p_boot:.3f} (<0.10 required)  -> {v['b_bootstrap_p_lt_0.10']}")
    print(f"  (c) TSM higher in {wins}/{len(ts.syms)} ETFs (> half required)  -> {v['c_tsm_beats_tsh_in_majority_of_etfs']}")
    print(f"  (d) mean monthly diff first half {halves['first half']['mean_monthly_diff_pct']:+.3f}%, second half "
          f"{halves['second half']['mean_monthly_diff_pct']:+.3f}%  -> {v['d_positive_diff_in_both_halves']}")
    overall = all(v.values())
    print(f"\n  OVERALL VERDICT: {'PASS (signal beats its control in this sample)' if overall else 'FAIL (signal not shown to beat its control)'}"
          f"  -- ETF proxy universe, ~{yrs:.0f}y sample; weak evidence either way.")

    res = dict(feed=feed, first_trade=str(ts.dates[e0].date()), overall=allm, june_2026=june, diff_sharpe=dm["sharpe"],
               monthly_diff_mean=float(d_m.mean()), bootstrap_p=p_boot, halves=halves, legs=legs,
               tsm_beats_tsh_etfs=wins, n_etfs=len(ts.syms), verdict=v, overall_pass=overall,
               agreement_pct=agree / tot * 100)
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(res, indent=2, default=str))
        print(f"\nwrote {OUT_PATH}")
    return res


def main():
    print("=== cached ETF bars ===")
    close, feed = load_cached()
    run_study(close, feed)


if __name__ == "__main__":
    main()
