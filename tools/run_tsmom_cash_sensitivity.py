#!/usr/bin/env python3
"""TSMOM cash-yield / financing-cost sensitivity. INFORMATIONAL ONLY.

This is a SECOND LOOK at the same price data already used by
tools/run_tsmom_test.py (verdict: FAIL on the predeclared bar). It does not
replace, overturn or re-score that verdict, whatever it shows. Its only job
is to size two simplifications the first run disclosed:

  (1) cash earned 0% (understates TSMOM, which sat in cash a lot in the
      2022-2025 high-rate years), and
  (2) borrowing above 1x leverage was free (overstates TSMOM).

PREDECLARED (before any result was seen):
  - Prices: the exact runtime/tsmom_prices.csv the first run saved (no new
    price download for the 8 ETFs). Fingerprint is printed and compared.
  - Rate series: yfinance ^IRX (13-week T-bill discount rate, in % per year),
    day t-1's value applied to day t. Treated as an annualised simple rate,
    daily accrual = rate/100/252. (Discount rate vs. bond-equivalent yield is
    a small difference; not corrected.)
  - Variant A: 0% cash, free leverage  -> must reproduce the first run
    exactly (this is the integrity check; the script says so if it doesn't).
  - Variant B: idle cash earns ^IRX; leverage still free.
  - Variant C: idle cash earns ^IRX; borrowed amount (gross positions above
    equity) pays ^IRX + 0.50% per year. Variant C is the realistic one.
  - Everything else is byte-identical to the first run: same universe, 252d
    signal, inverse-vol weights, 10% vol target, 1.5x cap, monthly
    rebalance, next-day application, 5bp/side costs, same comparators
    (applied the same cash/financing rules; they hold ~no cash/leverage so
    barely move), same eval window and sub-period split.
  - Same Calmar bar is printed for B and C for reference ONLY, labelled as a
    post-hoc sensitivity. If B or C "passes", the first run's FAIL stands:
    this is a re-analysis of the same data, not independent evidence.
  - June 2026 window (2026-05-15 -> 2026-07-01) reported separately for each
    variant, never folded into any aggregate.

No performance claim is made or implied by any number this prints.

USAGE (javier env; needs network only for ^IRX):
    cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
    python tools/run_tsmom_cash_sensitivity.py
Writes runtime/tsmom_irx.csv and runtime/tsmom_cash_sensitivity_results.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tsmom_test as base  # noqa: E402

PROJECT_ROOT = base.PROJECT_ROOT
IRX_PATH = PROJECT_ROOT / "runtime" / "tsmom_irx.csv"
OUT_PATH = PROJECT_ROOT / "runtime" / "tsmom_cash_sensitivity_results.json"
BORROW_SPREAD = 0.005  # +0.50%/yr over ^IRX, predeclared for variant C
FIRST_RUN_FINGERPRINT = "746c994e87a30d09960a70cd75f03b9c"


def download_irx() -> pd.Series:
    import yfinance as yf
    last = None
    for _ in range(3):
        try:
            df = yf.download("^IRX", start=base.DOWNLOAD_START, interval="1d", auto_adjust=False,
                             progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = df["Close"].dropna()
            if len(s) < 1000:
                raise ValueError(f"only {len(s)} rows")
            s.index = pd.to_datetime(s.index).tz_localize(None)
            IRX_PATH.parent.mkdir(parents=True, exist_ok=True)
            s.to_csv(IRX_PATH, header=["irx_pct"])
            print(f"  ^IRX: {len(s)} rows, {s.index.min().date()} -> {s.index.max().date()}, "
                  f"latest={s.iloc[-1]:.3f}%")
            return s
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise SystemExit(f"^IRX download failed: {last}")


def simulate_fin(prices: pd.DataFrame, weight_fn, rebal_dates: list, start_date,
                 rate_annual_pct: pd.Series | None, borrow_spread: float):
    """Same logic as base.simulate, plus explicit cash/borrow accrual.
    Cash (equity - gross positions) compounds at rate/252 when positive; when
    negative (levered) it compounds at (rate + borrow_spread)/252. With
    rate_annual_pct=None and borrow_spread=0 this reproduces base.simulate.
    Returns (daily_return_series, gross_exposure_over_equity_series)."""
    rets = prices.pct_change()
    idx = prices.index
    pos_of = {d: i for i, d in enumerate(idx)}
    if rate_annual_pct is None:
        r_cash = pd.Series(0.0, index=idx)
    else:
        # day j accrues at the rate observed on day j-1 (already known)
        r_cash = (rate_annual_pct.reindex(idx).ffill().shift(1) / 100.0 / 252.0).fillna(0.0)
    spread_d = borrow_spread / 252.0
    rebal = [d for d in rebal_dates if d >= start_date]
    daily = pd.Series(0.0, index=idx)
    gross = pd.Series(np.nan, index=idx)
    equity = 1.0
    cur_pos = pd.Series(0.0, index=prices.columns)
    first_i = pos_of[rebal[0]]
    for k, d in enumerate(rebal):
        i = pos_of[d]
        target_w = weight_fn(d).reindex(prices.columns).fillna(0.0)
        old_w = cur_pos / equity if equity != 0 else cur_pos * 0.0
        turnover = float((target_w - old_w).abs().sum())
        cost = turnover * base.COST_PER_SIDE
        equity *= (1.0 - cost)
        daily.iloc[i] = (1.0 + daily.iloc[i]) * (1.0 - cost) - 1.0 if i > first_i else -cost
        pos = target_w * equity
        j_end = pos_of[rebal[k + 1]] if k + 1 < len(rebal) else len(idx) - 1
        cash = equity - pos.sum()
        gross.iloc[i] = pos.sum() / equity if equity else np.nan
        for j in range(i + 1, j_end + 1):
            rate_d = float(r_cash.iloc[j])
            cash = cash * (1.0 + rate_d) if cash >= 0 else cash * (1.0 + rate_d + spread_d)
            pos = pos * (1.0 + rets.iloc[j].fillna(0.0))
            new_equity = cash + pos.sum()
            daily.iloc[j] = new_equity / equity - 1.0 if equity != 0 else 0.0
            equity = new_equity
            gross.iloc[j] = pos.sum() / equity if equity else np.nan
        cur_pos = pos
    return daily.iloc[first_i:], gross.iloc[first_i:]


def run(prices: pd.DataFrame, irx: pd.Series, write: bool = True) -> dict:
    tsmom_fn, ew_fn, sf_fn, eligible = base.build_weight_fns(prices)
    rebal_dates = base.month_end_dates(prices.index)
    n_elig = eligible.sum(axis=1)
    start_date = [d for d in rebal_dates if n_elig.loc[d] >= base.MIN_ELIGIBLE][0]
    print(f"\nevaluation start: {start_date.date()}   data end: {prices.index.max().date()}")

    variants = {
        "A (0% cash, free leverage)": (None, 0.0),
        "B (cash earns T-bill, free leverage)": (irx, 0.0),
        "C (cash earns T-bill, borrow = T-bill + 0.50%)": (irx, BORROW_SPREAD),
    }
    fns = {"TSMOM": tsmom_fn, "EqualWeight": ew_fn, "60/40": sf_fn}
    results = {}
    out = dict(eval_start=str(start_date.date()), data_end=str(prices.index.max().date()), variants={})
    sims = {}
    for vname, (rate, spread) in variants.items():
        sims[vname] = {}
        for sname, fn in fns.items():
            sims[vname][sname] = simulate_fin(prices, fn, rebal_dates, start_date, rate, spread)

    # integrity check: variant A must equal the first run's simulator exactly
    va = list(variants)[0]
    max_diff = 0.0
    for sname, fn in fns.items():
        ref = base.simulate(prices, fn, rebal_dates, start_date)
        d = float((sims[va][sname][0] - ref).abs().max())
        max_diff = max(max_diff, d)
    print(f"integrity check, variant A vs first run's simulator: max abs daily diff = {max_diff:.2e} "
          f"-> {'REPRODUCES first run' if max_diff < 1e-9 else 'DOES NOT REPRODUCE -- do not trust this run'}")
    out["integrity_max_abs_diff"] = max_diff

    tsm_gross = sims[va]["TSMOM"][1]
    print(f"TSMOM average gross exposure/equity: {tsm_gross.mean():.2f}x; "
          f"fraction of days with idle cash (gross<1): {(tsm_gross < 0.999).mean()*100:.0f}%; "
          f"levered (gross>1.001): {(tsm_gross > 1.001).mean()*100:.0f}%")
    post = tsm_gross[tsm_gross.index >= base.SPLIT_DATE]
    print(f"  post-2012 only: avg gross {post.mean():.2f}x, idle-cash days {(post < 0.999).mean()*100:.0f}%, "
          f"levered days {(post > 1.001).mean()*100:.0f}%")
    out["tsmom_gross"] = dict(avg=float(tsm_gross.mean()), idle_cash_frac=float((tsm_gross < 0.999).mean()),
                              levered_frac=float((tsm_gross > 1.001).mean()))

    for vname in variants:
        print(f"\n################ VARIANT {vname} ################")
        vres = dict(overall={}, pre2012={}, post2012={}, june_2026={}, calendar_years={})
        for label, sel in [("OVERALL", lambda r: r),
                           ("PRE-2012", lambda r: r[r.index < base.SPLIT_DATE]),
                           ("POST-2012", lambda r: r[r.index >= base.SPLIT_DATE])]:
            print(f"--- {label} ---")
            key = {"OVERALL": "overall", "PRE-2012": "pre2012", "POST-2012": "post2012"}[label]
            for sname in fns:
                m = base.metrics(sel(sims[vname][sname][0]))
                vres[key][sname] = m
                print(f"  {sname:12s} {base.fmt(m)}")
        print("--- JUNE 2026 WINDOW (2026-05-15 -> 2026-07-01), on its own ---")
        for sname in fns:
            w = base.window_return(sims[vname][sname][0], base.JUNE_START, base.JUNE_END)
            vres["june_2026"][sname] = w
            print(f"  {sname:12s} " + ("no data" if w["return_pct"] is None
                                       else f"days={w['days']} return={w['return_pct']:.2f}%"))
        yrs = sorted(set(sims[vname]["TSMOM"][0].index.year))
        for y in yrs:
            vres["calendar_years"][int(y)] = {
                s: float(((1 + sims[vname][s][0][sims[vname][s][0].index.year == y]).prod() - 1) * 100) for s in fns}
        out["variants"][vname] = vres

    print("\n=== CALENDAR-YEAR TSMOM RETURN % ACROSS VARIANTS (plus 60/40 under C) ===")
    names = list(variants)
    print(f"  {'year':6s}{'TSMOM A':>10s}{'TSMOM B':>10s}{'TSMOM C':>10s}{'60/40 C':>10s}")
    for y in sorted(out["variants"][names[0]]["calendar_years"]):
        row = [out["variants"][n]["calendar_years"][y]["TSMOM"] for n in names]
        sf = out["variants"][names[2]]["calendar_years"][y]["60/40"]
        print(f"  {y:<6d}{row[0]:10.1f}{row[1]:10.1f}{row[2]:10.1f}{sf:10.1f}")

    print("\n=== REFERENCE ONLY: the first run's Calmar bar re-applied (post-hoc sensitivity, NOT a re-verdict) ===")
    for vname in names:
        parts = {}
        for label in ["pre2012", "post2012"]:
            p = out["variants"][vname][label]
            t, ew, sf = p["TSMOM"]["calmar"], p["EqualWeight"]["calmar"], p["60/40"]["calmar"]
            parts[label] = bool(np.isfinite([t, ew, sf]).all() and t > ew and t > sf)
            print(f"  {vname[:44]:44s} {label}: TSMOM {t:.2f} vs EW {ew:.2f}, 60/40 {sf:.2f} "
                  f"-> {'beats both' if parts[label] else 'does NOT beat both'}")
        out["variants"][vname]["bar_reference_only"] = parts
    print("\n  The first run's verdict (FAIL) stands regardless of the above: same data, second look, "
          "informational only.")

    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {OUT_PATH}")
    return out


def main():
    if not base.PRICES_PATH.exists():
        raise SystemExit(f"{base.PRICES_PATH} missing -- run tools/run_tsmom_test.py first")
    prices = pd.read_csv(base.PRICES_PATH, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    fp = hashlib.md5(prices.round(6).to_csv().encode()).hexdigest()
    print(f"loaded {base.PRICES_PATH.name}: {prices.shape}, fingerprint {fp} "
          f"({'matches' if fp == FIRST_RUN_FINGERPRINT else 'DIFFERS FROM'} first run's {FIRST_RUN_FINGERPRINT})")
    print("=== downloading ^IRX ===")
    irx = download_irx()
    run(prices, irx)


if __name__ == "__main__":
    main()
